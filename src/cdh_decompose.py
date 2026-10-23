#!/usr/bin/env python3
"""Decompose BEHAVE CD-H into where the error comes from (GT needed, BEHAVE only).
  A  official form: one Sim(3) from the first scored frame, applied to all frames
  B  per-frame Sim(3): articulation/shape error only
  C  A + per-frame centroid-translation fix: removes the whole-body trajectory error
  D  A + per-frame fix of the centroid error ALONG the camera ray only (depth error)
  E  A + per-frame fix of the centroid error PERPENDICULAR to the ray only (lateral)
Usage: cdh_decompose.py work/behave_pred/<name> [...]"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.score_local import load_episode, fit_similarity_transform, apply_similarity_transform, symmetric_chamfer_distance
from src.mhr_local import MHR
GT = ROOT / "work/behave_gt"; mhr = MHR("cuda:0")
for pd_ in sys.argv[1:]:
    pd_ = Path(pd_); rows = []
    for seq in sorted(p.name for p in GT.iterdir() if (pd_ / p.name).exists()):
        fr = np.array(json.load((GT / seq / "scored_frames.json").open())["frames"])
        P, R = load_episode(pd_ / seq, 0, fr, mhr), load_episode(GT / seq, 0, fr, mhr)
        al, body = mhr.alignment_vertices, mhr.body_proxy_vertices
        t = fit_similarity_transform(P["verts"][0, al], R["verts"][0, al])
        ph = apply_similarity_transform(P["verts"][:, body], t); rh = R["verts"][:, body]
        cd = lambda a: float(symmetric_chamfer_distance(a, rh).mean() * 100)
        A = cd(ph)
        B = cd(np.stack([apply_similarity_transform(P["verts"][i:i+1, body], fit_similarity_transform(P["verts"][i, al], R["verts"][i, al]))[0] for i in range(len(fr))]))
        d = rh.mean(1) - ph.mean(1)                                   # per-frame centroid error [T,3]
        ray = rh.mean(1) / np.linalg.norm(rh.mean(1), axis=1, keepdims=True)
        dz = (d * ray).sum(1, keepdims=True) * ray; dl = d - dz
        C, D, E = cd(ph + d[:, None]), cd(ph + dz[:, None]), cd(ph + dl[:, None])
        drift = np.linalg.norm(d, axis=1) * 100
        rows.append(dict(seq=seq.split("_", 2)[2], A=A, B=B, C=C, D=D, E=E, err_first=drift[:10].mean(), err_last=drift[-50:].mean(), depth_err=float(np.abs((d*ray).sum(1)).mean()*100)))
    print("===", pd_.name)
    for r in rows: print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("MEAN", {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k != "seq"}, flush=True)
