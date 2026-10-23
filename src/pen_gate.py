#!/usr/bin/env python3
"""Fast PEN gate on an export dir: the host's own penetration_depth of OUR hand vertices inside
OUR submitted object (budgeted mesh, per-frame rotation/translation/scale), scored frames only --
exactly what the host measures, without score_local's CD/ACC machinery."""
import sys, json, numpy as np, trimesh
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
from src.mhr_local import MHR
from v2dlb.mhr_submission import penetration_depth
D = Path(sys.argv[1]); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:3"
mhr = MHR(dev); sc = json.load(open(ROOT / "work/scored_frames_summary.json")); hv = mhr.hand_vertices
worst = 0.0; frames_pen = 0; total = 0; per_ep = {}
for ep in range(30):
    z = np.load(D / f"episode_{ep:06d}.npz"); m = trimesh.load(D / f"episode_{ep:06d}_object.obj", force="mesh", process=False)
    mv = np.asarray(m.vertices, np.float64); mf = np.asarray(m.faces, np.int64); mf = mf[(mf[:, 0] != mf[:, 1]) | (mf[:, 1] != mf[:, 2])]
    fr = np.concatenate([np.arange(a, b + 1) for a, b in sc[str(ep)]["spans"]])
    p = np.concatenate([z["pose"][fr], np.broadcast_to(z["scales"], (len(fr), 68))], 1); verts, _ = mhr.forward(p, z["shape"])
    s = float(z["object_scale"]); mx = 0.0; n = 0
    for t, f in enumerate(fr):
        local = ((verts[t, hv].astype(np.float64) - z["object_translation"][f]) @ z["object_rotation"][f]) / s
        d = penetration_depth(local, mv, mf).max() * s * 100
        if d > 0: n += 1; mx = max(mx, d)
    per_ep[ep] = (round(mx, 3), n); worst = max(worst, mx); frames_pen += n; total += len(fr)
    print(f"ep{ep:2d}: max depth {mx:.3f} cm, frames with penetration {n}/{len(fr)}", flush=True)
print(f"RESULT PEN gate {D.name}: worst depth {worst:.3f} cm, frames with penetration {frames_pen}/{total} -> {'PASS' if frames_pen == 0 else 'FAIL'}", flush=True)
