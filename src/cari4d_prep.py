#!/usr/bin/env python3
"""Prepare one episode as a CARI4D (official v2d_cari4d) wild-video input.

The official pipeline wants, per sequence:
  <seq>.0.color.mp4           RGB video (every frame is processed)
  <seq>_masks_k0.h5           masks for EVERY video frame, keys <seq>/<frame>-k0.{person_mask,obj_rend_mask}.png
  object mesh (metric scale)  trimesh-loadable, scale preserved by the pipeline

Our masks (work/masks/epNNNNNN.npz) cover frames [first_scored-30, len-1] only, so the video is
TRIMMED to that range: it is re-encoded from the already-extracted JPG frames (work/frames) so
that frame i of the trimmed video is exactly original frame F0+i and matches mask row i. The
offset F0 is written to inputs/<seq>/meta.json for the export step.

Object mesh: our per-object SAM 3D Objects mesh (work/meshes_v2/<object>.glb) scaled so that its
largest extent equals the 3-episode hand-depth consensus size used by sub_v6 (the README's
"mesh autoscaling" is not shipped in the repo, so this is our substitute).
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import h5py
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUTS = ROOT / "work/v2d_official/inputs"


def prep(ep: int, mesh_dir: str, overwrite: bool = False, max_faces: int = 30000) -> Path:
    seq = f"ep{ep:06d}"
    out = INPUTS / seq
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and not overwrite:
        print(f"{seq}: already prepared"); return out
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}[ep]
    z = np.load(ROOT / f"work/masks/{seq}.npz")
    fids, packed = z["frame_ids"], z["packed"]; H, W = (int(v) for v in z["shape"])
    F0, N = int(fids[0]), len(fids)
    assert np.array_equal(fids, np.arange(F0, F0 + N)), "masks must be contiguous"
    frames_dir = ROOT / f"work/frames/{seq}"
    assert all((frames_dir / f"{f:06d}.jpg").exists() for f in fids), "missing jpg frames"

    # 1) video from the extracted frames (exact frame alignment with the masks)
    video = out / f"{seq}.0.color.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-start_number", str(F0),
           "-i", str(frames_dir / "%06d.jpg"), "-frames:v", str(N), "-c:v", "libx264", "-preset", "fast",
           "-crf", "10", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    subprocess.run(cmd, check=True)
    import av
    with av.open(str(video)) as c:
        st = c.streams.video[0]; n_declared = int(st.frames); hw = (st.codec_context.height, st.codec_context.width)
        n_dec = sum(1 for _ in c.decode(st))
    assert n_declared == N and n_dec == N and hw == (H, W), (n_declared, n_dec, hw, N, H, W)

    # 2) masks H5 in the packer's schema (channel 0 = object, channel 1 = person in our npz)
    h5 = out / f"{seq}_masks_k0.h5"
    with h5py.File(h5, "w") as f:
        f.attrs["schema"] = "v2d.cari4d.wild_masks.v1"; f.attrs["sequence"] = seq
        f.attrs["frame_count"] = N; f.attrs["width"] = W; f.attrs["height"] = H
        g = f.require_group(seq)
        n_person_empty = n_obj_empty = 0
        for i in range(N):
            m = np.unpackbits(packed[i], axis=-1)[:, :, :W]
            person = (m[1] > 0).astype(np.uint8) * 255; obj = (m[0] > 0).astype(np.uint8) * 255
            n_person_empty += int(person.sum() == 0); n_obj_empty += int(obj.sum() == 0)
            g.create_dataset(f"{i:06d}-k0.person_mask.png", data=person, compression="lzf", shuffle=True)
            g.create_dataset(f"{i:06d}-k0.obj_rend_mask.png", data=obj, compression="lzf", shuffle=True)

    # 3) object mesh at metric scale (consensus size, as sub_v6)
    mesh_src = ROOT / mesh_dir / f"{meta['object']}.glb"
    mesh = trimesh.load(str(mesh_src), force="mesh", process=False)
    K = json.load((ROOT / "work/camera/intrinsics.json").open())[meta["camera"]]
    from src.export_episodes import _object_metric_size
    S = _object_metric_size(ep, K, mesh)
    scale = S / float(np.max(mesh.extents))
    mesh.apply_scale(scale)
    n_src = len(mesh.faces)
    if n_src > max_faces:
        # SAM 3D Objects meshes are 100k-400k faces; FoundationPose rendering and the step-07
        # contact/penetration terms (point-to-triangle over every face, every frame, every step)
        # do not need that. Quadric decimation, vertex colours carried over by nearest vertex.
        import fast_simplification
        from scipy.spatial import cKDTree
        try:
            vis = mesh.visual if isinstance(mesh.visual, trimesh.visual.ColorVisuals) else mesh.visual.to_color()
            colors = np.asarray(vis.vertex_colors)
        except Exception:
            colors = None
        v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64), target_count=max_faces)
        dec = trimesh.Trimesh(v, f, process=False)
        if colors is not None:
            _, idx = cKDTree(np.asarray(mesh.vertices)).query(v)
            dec.visual = trimesh.visual.ColorVisuals(dec, vertex_colors=colors[idx])
        mesh = dec
    mesh_out = out / "object_metric.glb"
    mesh.export(str(mesh_out))
    chk = trimesh.load(str(mesh_out), force="mesh", process=False)
    json.dump(dict(ep=ep, seq=seq, object=meta["object"], camera=meta["camera"], K=K, frame_offset=F0, n_frames=N,
                   height=H, width=W, mesh_source=str(mesh_src), metric_size_m=S, mesh_scale=scale,
                   mesh_extents_m=[float(v) for v in chk.extents], n_vertices=int(len(chk.vertices)), n_faces=int(len(chk.faces)), n_faces_source=int(n_src),
                   watertight=bool(chk.is_watertight),
                   person_mask_empty_frames=n_person_empty, object_mask_empty_frames=n_obj_empty),
              meta_path.open("w"), indent=1)
    print(f"{seq}: frames {F0}..{F0+N-1} ({N}), object {meta['object']} size {S:.3f} m, mesh {len(chk.vertices)} v, "
          f"empty masks person/object {n_person_empty}/{n_obj_empty}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--mesh-dir", default="work/meshes_v2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-faces", type=int, default=30000)
    a = ap.parse_args()
    for ep in a.eps:
        prep(ep, a.mesh_dir, a.overwrite, a.max_faces)
