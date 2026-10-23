#!/usr/bin/env python3
"""Where does BEHAVE CD-O come from? (GT needed, BEHAVE only.) All in the host's frame: our object
posed, then the frame-0 HUMAN Sim(3) applied, vs the GT object.
  A   official form
  P   + per-frame position oracle: shift our object so its surface centroid = GT surface centroid
  PS  + size oracle: also rescale about the centroid so RMS radius = GT's
  PSR + rotation oracle: also best rotation per frame (Kabsch on nearest-neighbour pairs, 5 ICP iters)
  A-P = position error, P-PS = size, PS-PSR = orientation, PSR = shape (mesh) floor
Usage: cdo_decompose.py work/behave_pred/<name> [...]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.score_local import load_episode, fit_similarity_transform, apply_similarity_transform, symmetric_chamfer_distance, sample_mesh_surface
from src.mhr_local import MHR
GT = ROOT / "work/behave_gt"; mhr = MHR("cuda:0")
def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0); H = (a - ca).T @ (b - cb); U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); return Vt.T @ D @ U.T
for pd_ in sys.argv[1:]:
    pd_ = Path(pd_); rows = []
    for i, seq in enumerate(sorted(p.name for p in GT.iterdir() if (pd_ / p.name).exists())):
        fr = np.array(json.load((GT / seq / "scored_frames.json").open())["frames"])[::3]
        P, R = load_episode(pd_ / seq, 0, fr, mhr), load_episode(GT / seq, 0, fr, mhr)
        al = mhr.alignment_vertices; t = fit_similarity_transform(P["verts"][0, al], R["verts"][0, al])
        n = len(mhr.body_proxy_vertices)
        po, ro = (sample_mesh_surface(e["mesh_v"], e["mesh_f"], n, i) for e in (P, R))
        surf = lambda e, loc: e["scale"] * np.einsum("tij,pj->tpi", e["rotation"], loc) + e["translation"][:, None]
        ps, rs = apply_similarity_transform(surf(P, po), t), surf(R, ro)
        cd = lambda a: float(symmetric_chamfer_distance(a, rs).mean() * 100)
        A = cd(ps)
        cp, cr = ps.mean(1, keepdims=True), rs.mean(1, keepdims=True)
        pP = ps - cp + cr; Pv = cd(pP)
        rad = lambda x, c: np.sqrt(((x - c) ** 2).sum(-1).mean(1, keepdims=True))[..., None]
        pS = (pP - cr) * (rad(rs, cr) / rad(pP, cr)) + cr; PS = cd(pS)
        pR = pS.copy()
        for f in range(len(fr)):
            x = pR[f]
            for _ in range(5):
                _, j = cKDTree(rs[f]).query(x); Rk = kabsch(x, rs[f][j]); x = (x - x.mean(0)) @ Rk.T + rs[f].mean(0)
            pR[f] = x
        PSR = cd(pR)
        rows.append(dict(seq=seq.split("_", 2)[2], A=A, P=Pv, PS=PS, PSR=PSR, pos_err=float(np.linalg.norm((cp - cr)[:, 0], axis=1).mean() * 100),
                         size_ratio=float(np.median(rad(pP, cr) / rad(rs, cr)))))
    print("===", pd_.name)
    for r in rows: print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("MEAN", {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k != "seq"}, flush=True)
