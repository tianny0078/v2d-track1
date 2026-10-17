#!/usr/bin/env python3
"""Body proxies for export dirs: ACC-H proxy (22 ACC joints, cm/frame^2, within scored stretches,
exactly the metric's recipe; GT's own value 0.145) and reprojection error vs VitPose (px)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, "work/kit/v2d_submission_kit")
from src.mhr_local import MHR
from src.fit_mhr import ACC_JOINTS, COCO_TO_MHR
from v2dlb.mhr_submission import stretches
dirs = [d for d in sys.argv[1:] if not d.isdigit()]; eps = [int(e) for e in sys.argv[1:] if e.isdigit()] or list(range(30))
mhr = MHR("cuda:3"); sc = json.load(open("work/scored_frames_summary.json"))
K = json.load(open("work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open("data/hf/track_1/meta/episodes_metadata.jsonl")}
cs = sorted(COCO_TO_MHR); mi = [COCO_TO_MHR[c] for c in cs]
tot = {d: [] for d in dirs}
for ep in eps:
    fr = np.concatenate([np.arange(a, b + 1) for a, b in sc[str(ep)]["spans"]]); k = K[meta[ep]["camera"]]
    kz = np.load(f"work/kp2d/ep{ep:06d}.npz"); kp = np.nan_to_num(kz["kp"]); kpos = {f: i for i, f in enumerate(kz["frame_ids"])}
    line = f"ep{ep:2d}"
    for d in dirs:
        z = np.load(f"{d}/episode_{ep:06d}.npz"); p = np.concatenate([z["pose"][fr], np.broadcast_to(z["scales"], (len(fr), 68))], 1); _, j = mhr.forward(p, z["shape"])
        acc = np.mean([np.linalg.norm((j[s][:, list(ACC_JOINTS)] * 100)[:-2] - 2 * (j[s][:, list(ACC_JOINTS)] * 100)[1:-1] + (j[s][:, list(ACC_JOINTS)] * 100)[2:], axis=-1).mean() for s in stretches(fr) if s.stop - s.start >= 3])
        sel = [t for t, f in enumerate(fr) if f in kpos]; P = j[sel][:, mi]; kk = kp[[kpos[fr[t]] for t in sel]][:, cs]
        u = k["fx"] * P[..., 0] / P[..., 2] + k["cx"]; v = k["fy"] * P[..., 1] / P[..., 2] + k["cy"]
        px = float((np.hypot(u - kk[..., 0], v - kk[..., 1]) * kk[..., 2]).sum() / max(kk[..., 2].sum(), 1))
        tot[d].append((acc, px)); line += f" | {Path(d).name}: ACC-H {acc:.3f} px {px:5.1f}"
    print("ROW", line, flush=True)
for d in dirs:
    a = np.array(tot[d]); print(f"RESULT {Path(d).name}: ACC-H mean {a[:,0].mean():.3f} max {a[:,0].max():.3f} | px mean {a[:,1].mean():.1f} max {a[:,1].max():.1f}")
