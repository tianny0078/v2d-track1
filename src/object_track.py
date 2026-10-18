#!/usr/bin/env python3
"""Per-frame object ROTATION tracking by silhouette (detection-by-tracking, rotation part).

Seed: the episode's own SAM 3D Objects (mesh, R) pair at its best frame (object_episode_pose.py).
Track: forward from the seed frame to the end, backward to the start. At each frame, candidates =
previous rotation composed with K small random rotations (+ identity); each candidate is rendered
(pytorch3d rasterizer, 1/4 resolution) centred on the mask bbox and scaled to the mask extent --
so the score is the silhouette SHAPE only, independent of the depth estimate -- and scored by IoU
against the SAM2 mask. Frames whose mask is missing or shrunk by occlusion keep the previous
rotation. Finally the rotation sequence is smoothed per scored stretch.

Output: work/objtrack/epNNNNNN_rot.npz (frame_ids, R [N,3,3], iou, iou_seed_const, iou_identity).
"""
import os
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, trimesh
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
from v2dlb.mhr_submission import stretches
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
from pytorch3d.structures import Meshes


def load_budget_mesh(path):
    from v2dlb.mesh_budget import budget_mesh
    m = trimesh.load(path, force="mesh", process=False)
    if len(m.faces) > 4096 or len(m.vertices) > 4096:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            m.export(f.name)
        try:
            bv, bf = budget_mesh(f.name, 4096, 4096)
        finally:
            os.unlink(f.name)                                                # was leaking into /tmp
        bf = np.asarray(bf, np.int64); bf = bf[(bf[:, 0] != bf[:, 1]) | (bf[:, 1] != bf[:, 2])]
        m = trimesh.Trimesh(np.asarray(bv, np.float64), bf, process=False)
    v = np.asarray(m.vertices, np.float32); v = v - v.mean(0); v = v / max(np.ptp(v, axis=0).max(), 1e-6)   # unit size, centred
    return v, np.asarray(m.faces, np.int64)


class Sil:
    """batch silhouette renderer at a reduced resolution; pytorch3d screen-space camera convention
    (x left, y up in NDC) handled by flipping x and y of the camera-frame points."""
    def __init__(self, K, H, W, down, dev):
        self.dev = dev; self.h, self.w = H // down, W // down
        self.fx, self.fy, self.cx, self.cy = K["fx"] / down, K["fy"] / down, K["cx"] / down, K["cy"] / down
        self.ras = MeshRasterizer(raster_settings=RasterizationSettings(image_size=(self.h, self.w), blur_radius=0.0, faces_per_pixel=1, bin_size=0))

    def render(self, pts, faces):
        """pts [B,V,3] camera-frame (x right, y down, z forward) -> [B,h,w] bool"""
        B = pts.shape[0]
        p = pts.clone(); p[..., 0] = -p[..., 0]; p[..., 1] = -p[..., 1]
        cam = PerspectiveCameras(focal_length=torch.tensor([[self.fx, self.fy]] * B, device=self.dev), principal_point=torch.tensor([[self.cx, self.cy]] * B, device=self.dev),
                                 image_size=torch.tensor([[self.h, self.w]] * B, device=self.dev), in_ndc=False, device=self.dev)
        mesh = Meshes(verts=list(p), faces=[faces] * B)
        frag = self.ras(mesh, cameras=cam)
        return frag.pix_to_face[..., 0] >= 0


def small_rotations(K, deg, rng):
    rv = rng.normal(size=(K, 3)); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    ang = np.abs(rng.normal(scale=np.radians(deg), size=(K, 1)))
    R = Rot.from_rotvec(rv * ang).as_matrix()
    return np.concatenate([np.eye(3)[None], R], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--deg", type=float, default=6.0)
    ap.add_argument("--down", type=int, default=4)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--redetect-below", type=float, default=0.0, help="global re-detection when the tracked IoU drops below this")
    ap.add_argument("--tag", default="", help="output suffix: ep_rot<tag>.npz / state file")
    ap.add_argument("--seed-search", type=int, default=0, help="random rotations to try at the seed frame (0 = trust the layout seed)")
    a = ap.parse_args(); dev = torch.device(a.device); rng = np.random.RandomState(0)
    K = json.load(open(ROOT / "work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
    sc = json.load(open(ROOT / "work/scored_frames_summary.json"))
    (ROOT / "state/objtrack").mkdir(parents=True, exist_ok=True)
    for ep in a.eps:
        t0 = time.time(); k = K[meta[ep]["camera"]]
        pose = json.load(open(ROOT / f"work/objtrack/ep{ep:06d}_pose.json")); R0 = np.array(pose["R_full"]); f0 = pose["view"]["frame"]
        v, faces = load_budget_mesh(ROOT / f"work/objtrack/ep{ep:06d}_mesh.glb")
        V = torch.tensor(v, device=dev); Fc = torch.tensor(faces, device=dev)
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); H, W = [int(x) for x in z["shape"]]; mpos = {f: i for i, f in enumerate(z["frame_ids"])}
        fr = np.concatenate([np.arange(s, e + 1) for s, e in sc[str(ep)]["spans"]])
        sil = Sil(k, H, W, a.down, dev)
        masks = {}; areas = np.zeros(len(fr))
        for t, f in enumerate(fr):
            if f in mpos:
                m = np.unpackbits(z["packed"][mpos[f]], axis=-1)[0, :, :W].astype(bool)
                md = m.reshape(H // a.down, a.down, W // a.down, a.down).any(axis=(1, 3))
                if m.sum() > 30: masks[t] = (md, m.sum(), np.where(m)); areas[t] = m.sum()
        ref_area = np.median(areas[areas > 0]) if (areas > 0).any() else 1.0
        zz = 3.0

        def score(Rs, t):
            """IoU of each candidate rotation at frame t (bbox-centred, extent-scaled)."""
            md, area, (ys, xs) = masks[t]
            ext = max(np.ptp(xs), np.ptp(ys)); cxm, cym = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            Rt = torch.tensor(Rs, dtype=torch.float32, device=dev)
            P = torch.einsum("bij,vj->bvi", Rt, V)                          # rotated unit mesh
            u = k["fx"] * P[..., 0] / zz; vv = k["fy"] * P[..., 1] / zz        # projected at depth zz, size 1
            e0 = torch.maximum(u.max(1).values - u.min(1).values, vv.max(1).values - vv.min(1).values).clamp(min=1e-6)
            S = (ext / e0)[:, None, None]                                      # metric size matching the mask extent
            P = P * S
            ub = (u.max(1).values + u.min(1).values) / 2 * S[:, 0, 0]; vb = (vv.max(1).values + vv.min(1).values) / 2 * S[:, 0, 0]
            X = (cxm - k["cx"]) / k["fx"] * zz - ub / k["fx"] * zz; Y = (cym - k["cy"]) / k["fy"] * zz - vb / k["fy"] * zz
            P = P + torch.stack([X, Y, torch.full_like(X, zz)], -1)[:, None, :]
            r = sil.render(P, Fc)
            m = torch.tensor(md, device=dev)[None]
            inter = (r & m).flatten(1).sum(1).float(); uni = (r | m).flatten(1).sum(1).float().clamp(min=1)
            return (inter / uni).cpu().numpy()

        Rs = np.tile(R0[None], (len(fr), 1, 1)); iou = np.zeros(len(fr)); upd = np.zeros(len(fr), bool); redet = []
        i0 = int(np.argmin(np.abs(fr - f0)))
        seed_iou = float(score(R0[None], i0)[0]) if i0 in masks else float("nan")
        if a.seed_search > 0 and i0 in masks:
            # GLOBAL search at the seed frame: the layout rotation is sometimes poor (white desk
            # lying on the floor: IoU 0.11), and a local tracker cannot leave a bad basin. Try
            # uniformly random rotations in batches, then refine the best with small steps.
            best_R, best_s = R0, seed_iou
            for _ in range(a.seed_search // 64):
                cands = Rot.random(64, random_state=rng).as_matrix(); sc_ = score(cands, i0); j = int(np.argmax(sc_))
                if sc_[j] > best_s: best_R, best_s = cands[j], float(sc_[j])
            for deg in (15.0, 6.0, 3.0):
                cands = np.einsum("kij,jl->kil", small_rotations(96, deg, rng), best_R); sc_ = score(cands, i0); j = int(np.argmax(sc_))
                if sc_[j] > best_s: best_R, best_s = cands[j], float(sc_[j])
            if best_s > seed_iou + 0.05:
                print(f"  ep{ep}: seed search improved IoU {seed_iou:.3f} -> {best_s:.3f}", flush=True); R0 = best_R
        for order in (range(i0, len(fr)), range(i0 - 1, -1, -1)):
            order = list(order)
            for n, t in enumerate(order):
                prev = Rs[order[n - 1]] if n > 0 else R0
                if t not in masks or areas[t] < 0.4 * ref_area:
                    Rs[t] = prev; continue
                cands = np.einsum("kij,jl->kil", small_rotations(a.K, a.deg, rng), prev)
                s = score(cands, t); j = int(np.argmax(s)); Rs[t] = cands[j]; iou[t] = s[j]; upd[t] = True
                if a.redetect_below > 0 and iou[t] < a.redetect_below and areas[t] >= 0.7 * ref_area:
                    # LOST: the local search cannot leave its basin (white desk carried across the
                    # room, ep11: seed 0.85 but tracked mean 0.37). Re-detect by a global search at
                    # this frame, accept only if clearly better.
                    bR, bs = Rs[t], iou[t]
                    for _ in range(4):
                        cg = Rot.random(64, random_state=rng).as_matrix(); sg = score(cg, t); jg = int(np.argmax(sg))
                        if sg[jg] > bs: bR, bs = cg[jg], float(sg[jg])
                    for deg in (15.0, 6.0):
                        cg = np.einsum("kij,jl->kil", small_rotations(64, deg, rng), bR); sg = score(cg, t); jg = int(np.argmax(sg))
                        if sg[jg] > bs: bR, bs = cg[jg], float(sg[jg])
                    if bs > iou[t] + 0.15:
                        Rs[t] = bR; iou[t] = bs; redet.append((int(fr[t]), round(bs, 2)))
        # smoothing per stretch: average rotations in a Gaussian window (rotvec relative to the centre frame)
        Rsm = Rs.copy()
        if a.smooth > 0:
            for sl in stretches(fr):
                if sl.stop - sl.start < 3: continue
                idx = np.arange(sl.start, sl.stop); w0 = int(3 * a.smooth)
                for t in idx:
                    lo, hi = max(sl.start, t - w0), min(sl.stop - 1, t + w0); nb = np.arange(lo, hi + 1)
                    wts = np.exp(-0.5 * ((nb - t) / a.smooth) ** 2)
                    rel = Rot.from_matrix(np.einsum("nij,kj->nik", Rs[nb], Rs[t])).as_rotvec()   # R_n R_t^T
                    Rsm[t] = Rot.from_rotvec((wts[:, None] * rel).sum(0) / wts.sum()).as_matrix() @ Rs[t]
        # evaluation: tracked (smoothed) vs seed-constant vs identity, on frames with a usable mask
        ev = [t for t in range(len(fr)) if t in masks]
        sel = ev[:: max(1, len(ev) // 60)]
        def mean_iou(fn): return float(np.mean([score(fn(t)[None], t)[0] for t in sel]))
        res = dict(ep=ep, frames=int(len(fr)), updated=int(upd.sum()), redetections=len(redet), seconds=round(time.time() - t0, 1),
                   iou_track=round(mean_iou(lambda t: Rsm[t]), 3), iou_seed_const=round(mean_iou(lambda t: R0), 3), seed_frame_iou=round(seed_iou, 3), iou_identity=round(mean_iou(lambda t: np.eye(3)), 3))
        np.savez_compressed(ROOT / f"work/objtrack/ep{ep:06d}_rot{a.tag}.npz", frame_ids=fr, R=Rsm.astype(np.float32), R_raw=Rs.astype(np.float32), iou=iou, updated=upd)
        (ROOT / "state/objtrack" / f"ep{ep:06d}{a.tag}.done").write_text(json.dumps(res))
        print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
