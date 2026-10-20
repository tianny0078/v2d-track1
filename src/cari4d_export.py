#!/usr/bin/env python3
"""Convert official CARI4D outputs (work/v2d_official/runs/<seq>/) into our submission NPZs.

Source bundle, in order of preference: inference/refined.pth (step 07) > inference/coconet.pth
(step 06) > inference/init_bundle.pth > mhr_init/<seq>.pkl + foundationpose/<seq>.pkl.

Human: the bundle's 7 MHR blocks -> our 136-d `pose` (+ scales[68], shape[45]) by running the
SAME MHRHead the pipeline decodes with (`return_model_params=True` gives the 204-d model params
the TorchScript model consumed, with the global translation at 0) and putting the translation
back as pose[0:3] = 10 * diag(1,-1,-1) * mhr_trans (one pose unit = 0.1 m, y/z flipped). Shape and
scales are per-episode medians (the submission takes one identity per episode). The conversion is
checked by a round trip: our src/mhr_local.MHR forward of the exported pose must reproduce the
pipeline's own decoded vertices.

Object: refined `pr.pose_abs` is object->camera for export/<seq>/object_mesh/output_aligned.glb;
that mesh is budgeted to 4096 v/f and centred on its vertex mean c, so t' = t + R c, scale 1.

Frames: the run covers trimmed video frames i -> original frame F0+i (inputs/<seq>/meta.json);
rows outside the run take the nearest run frame. Units metres, camera frame y down z forward.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "work/v2d_official"
SOURCE_ROOT = OFFICIAL / "reconstruction/modules/v2d_cari4d/lib/cari4d"
SAM3D_LIB = OFFICIAL / "reconstruction/modules/v2d_sam3d_body/lib"
for p in (OFFICIAL / "shims", SOURCE_ROOT, SAM3D_LIB, ROOT, ROOT / "work/kit/v2d_submission_kit"):
    sys.path.insert(0, str(p))
os.environ.setdefault("MHR_ASSETS_ROOT", str(OFFICIAL / "weights/sam3d_body"))
os.environ.setdefault("MOMENTUM_ENABLED", "0")
FLIP = np.array([1.0, -1.0, -1.0], np.float32)


def load_source(run: Path, seq: str):
    for name in ("inference/refined.pth", "inference/coconet.pth", "inference/init_bundle.pth"):
        p = run / name
        if p.exists():
            b = torch.load(p, map_location="cpu", weights_only=False)
            return name, {k: np.asarray(v, np.float32) for k, v in b["pr"].items() if k.startswith("mhr_")}, np.asarray(b["pr"]["pose_abs"], np.float32), [str(f) for f in b["frames"]]
    from lib_mhr import load_mhr_result
    import joblib
    init = load_mhr_result(run / f"mhr_init/{seq}.pkl"); fp = joblib.load(run / f"foundationpose/{seq}.pkl")
    params = {k: np.asarray(init[k], np.float32) for k in ("mhr_global_rot6d", "mhr_trans", "mhr_body_pose_cont", "mhr_hand", "mhr_shape", "mhr_scale", "mhr_face")}
    return "mhr_init+foundationpose", params, np.asarray(fp["obj_pose_world"], np.float32), [str(f) for f in init["frames"]]


def to_pose136(params: dict, device="cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> pose[T,136], scales[T,68], shape[T,45], layer_vertices[T,V,3] (their decode, for the round trip)."""
    import roma
    from lib_mhr import MHRLayer
    from lib_mhr.rotations import rot6d_to_rotmat
    from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ["MHR_ASSETS_ROOT"], device=device)
    head = layer.backend._ensure_head(torch.device(device))
    T = len(params["mhr_trans"])
    poses, scales_all, verts_all = [], [], []
    with torch.no_grad():
        for s in range(0, T, 64):
            p = {k: torch.as_tensor(v[s:s + 64], device=device) for k, v in params.items()}
            n = len(p["mhr_trans"])
            euler = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(p["mhr_global_rot6d"]))
            body133 = compact_cont_to_model_params_body(p["mhr_body_pose_cont"])
            face = p.get("mhr_face", torch.zeros(n, 72, device=device))
            verts, model_params = head.mhr_forward(global_trans=torch.zeros(n, 3, device=device), global_rot=euler, body_pose_params=body133, hand_pose_params=p["mhr_hand"],
                                                   scale_params=p["mhr_scale"], shape_params=p["mhr_shape"], expr_params=face, return_model_params=True)
            pose = model_params[:, :136].clone()
            pose[:, :3] = 10.0 * torch.as_tensor(FLIP, device=device) * p["mhr_trans"]
            poses.append(pose.cpu().numpy()); scales_all.append(model_params[:, 136:].cpu().numpy())
            verts_all.append((verts * torch.as_tensor(FLIP, device=device) + p["mhr_trans"][:, None]).cpu().numpy())
    return np.concatenate(poses), np.concatenate(scales_all), params["mhr_shape"].copy(), np.concatenate(verts_all)


def _smooth(pose: np.ndarray, R: np.ndarray, t: np.ndarray, ep: int, F0: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """OUR post-processing (not part of the official pipeline): wrap-safe Gaussian smoothing of the
    pose (sigma frames) and of the object translation (max(sigma, 6)) within each scored stretch.
    The official pipeline gets its temporal coherence from CoCoNet, which we cannot run."""
    from scipy.ndimage import gaussian_filter1d
    from src.export_episodes import _smooth_wrap_safe
    from v2dlb.mhr_submission import stretches
    sc = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]
    fr = np.concatenate([np.arange(a, b + 1) for a, b in sc["spans"]]); pose = pose.copy(); t = t.copy()
    for sl in stretches(fr):
        rows = fr[sl] - F0
        if len(rows) >= 3:
            pose[rows] = _smooth_wrap_safe(pose[rows], sigma)
            t[rows] = gaussian_filter1d(t[rows], max(sigma, 6.0), axis=0, mode="nearest")
    return pose, t


def export(ep: int, out_dir: Path, device: str, smooth: float = 0.0) -> dict:
    seq = f"ep{ep:06d}"
    run = OFFICIAL / "runs" / seq
    meta = json.load((OFFICIAL / "inputs" / seq / "meta.json").open())
    F0 = int(meta["frame_offset"])
    source, params, pose_abs, frames = load_source(run, seq)
    assert [int(f) for f in frames] == list(range(len(frames)))
    N = len(frames)
    pose, scales_t, shape_t, their_verts = to_pose136(params, device)
    scales = np.median(scales_t, axis=0).astype(np.float32); shape = np.median(shape_t, axis=0).astype(np.float32)

    # round trip with OUR MHR wrapper: per-frame identity (exact) and the episode median identity
    from src.mhr_local import MHR
    mhr = MHR(device)
    idx = np.linspace(0, N - 1, min(N, 16)).astype(int)
    v_ours, _ = mhr.forward(np.concatenate([pose[idx], scales_t[idx]], 1), shape_t[idx[0]])  # shape per frame differs; use frame idx[0]'s shape for a strict check below
    v0, _ = mhr.forward(np.concatenate([pose[idx[:1]], scales_t[idx[:1]]], 1), shape_t[idx[0]])
    rt_exact = float(np.abs(v0[0] - their_verts[idx[0]]).max())
    v_med, _ = mhr.forward(np.concatenate([pose[idx], np.broadcast_to(scales, (len(idx), 68))], 1), shape)
    rt_median = float(np.linalg.norm(v_med - their_verts[idx], axis=-1).mean())
    # projection sanity: root joint vs person-mask centroid on the first frame
    K = meta["K"]; _, j = mhr.forward(np.concatenate([pose[:1], scales[None]], 1), shape)
    root = j[0, 1]; u, v = K["fx"] * root[0] / root[2] + K["cx"], K["fy"] * root[1] / root[2] + K["cy"]
    z = np.load(ROOT / f"work/masks/{seq}.npz"); W = int(z["shape"][1]); pm = np.unpackbits(z["packed"][0], axis=-1)[1, :, :W]; ys, xs = np.where(pm)

    # object
    mesh = trimesh.load(str(run / "export" / seq / "object_mesh/output_aligned.glb"), force="mesh", process=False)
    from v2dlb.mesh_budget import budget_mesh
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        mesh.export(f.name)
    try:
        bv, bf = budget_mesh(f.name, 4096, 4096)
    finally:
        os.unlink(f.name)
    bf = np.asarray(bf, np.int64); bf = bf[(bf[:, 0] != bf[:, 1]) | (bf[:, 1] != bf[:, 2])]
    bv = np.asarray(bv, np.float64); c = bv.mean(0); bv = bv - c
    R = pose_abs[:, :3, :3].astype(np.float32); t = (pose_abs[:, :3, 3] + np.einsum("tij,j->ti", pose_abs[:, :3, :3], c)).astype(np.float32)

    if smooth > 0:
        pose, t = _smooth(pose, R, t, ep, F0, smooth)
    eps_len = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in (ROOT / "data/hf/track_1/meta/episodes.jsonl").open()}
    sc = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]
    T = max(eps_len[ep], sc["last"] + 1)
    pick = np.clip(np.arange(T) - F0, 0, N - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"episode_{ep:06d}.npz", pose=pose[pick].astype(np.float32), scales=scales, shape=shape,
             object_rotation=R[pick], object_translation=t[pick], object_scale=np.float32(1.0))
    trimesh.Trimesh(bv, bf, process=False).export(out_dir / f"episode_{ep:06d}_object.obj")
    rec = dict(ep=ep, source=source, smooth_sigma=smooth, T=int(T), run_frames=int(N), frame_offset=F0, roundtrip_exact_max_m=round(rt_exact, 6), roundtrip_median_identity_mean_m=round(rt_median, 4),
               root_px=[round(float(u)), round(float(v))], mask_centroid_px=[round(float(xs.mean())), round(float(ys.mean()))] if len(xs) else None,
               root_depth_m=round(float(root[2]), 3), obj_t_first=[round(float(x), 3) for x in t[0]], obj_extent_m=[round(float(x), 3) for x in (bv.max(0) - bv.min(0))],
               shape_spread=round(float(np.linalg.norm(shape_t - shape, axis=1).mean()), 3), scales_spread=round(float(np.abs(scales_t - scales).mean()), 4))
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--out", default="work/sub_cari4d")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smooth", type=float, default=0.0, help="our wrap-safe Gaussian smoothing (sigma frames) within scored stretches; 0 = official output as is")
    a = ap.parse_args()
    recs = []
    for ep in a.eps:
        try:
            recs.append(export(ep, ROOT / a.out, a.device, a.smooth))
        except Exception as exc:
            print(json.dumps(dict(ep=ep, error=f"{type(exc).__name__}: {exc}")), flush=True)
    (ROOT / a.out / "export_report.json").write_text(json.dumps(recs, indent=1))
