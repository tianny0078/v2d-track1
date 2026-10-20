#!/usr/bin/env python3
"""PEN-clean variant of an export dir: apply OUR hand->centroid push-out (the two-pass envelope
scheme of src/export_episodes.py::_place_object, PUSH_MODE "centroid") to the final object
translations only. Nothing else is touched (pose, rotation, scale, mesh). Used on the official
CARI4D exports, whose step 07 constrains object-inside-body but not hand-inside-object.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))


def push_out(ep: int, src: Path, dst: Path, mhr) -> dict:
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    from v2dlb.mhr_submission import penetration_depth, stretches
    from src.mhr_local import L_WRIST, R_WRIST
    sc = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]
    fr = np.concatenate([np.arange(a, b + 1) for a, b in sc["spans"]])
    z = dict(np.load(src / f"episode_{ep:06d}.npz"))
    m = trimesh.load(src / f"episode_{ep:06d}_object.obj", force="mesh", process=False)
    mv = np.asarray(m.vertices, np.float64); mf = np.asarray(m.faces, np.int64); mf = mf[(mf[:, 0] != mf[:, 1]) | (mf[:, 1] != mf[:, 2])]
    T = len(z["pose"]); obj_s = float(z["object_scale"])
    p = np.concatenate([z["pose"][fr], np.broadcast_to(z["scales"], (len(fr), 68))], 1)
    verts, j = mhr.forward(p, z["shape"])
    hand_pts = verts[:, mhr.hand_vertices].astype(np.float64)
    hands = j[:, [L_WRIST, R_WRIST]]
    Rf = z["object_rotation"][fr].astype(np.float64); t3 = z["object_translation"][fr].astype(np.float64)
    cen = np.einsum("tij,j->ti", Rf, mv.mean(0) * obj_s) + t3  # world centroid
    near = np.linalg.norm(hands - cen[:, None], axis=-1).argmin(1); hand_c = hands[np.arange(len(fr)), near]

    def pen_max(t, tt):
        local = ((hand_pts[t] - tt) @ Rf[t]) / obj_s
        return penetration_depth(local, mv, mf).max()

    def need(t, dirv, base):
        if pen_max(t, base) <= 0.0:
            return 0.0
        lo, hi = 0.0, None
        for step in (0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55):
            if pen_max(t, base + step * dirv) <= 0.0:
                hi = step; break
            lo = step
        if hi is None:
            return 0.55
        while hi - lo > 0.005:
            mid = 0.5 * (lo + hi)
            if pen_max(t, base + mid * dirv) <= 0.0: hi = mid
            else: lo = mid
        return hi

    dirs = np.zeros((len(fr), 3)); push = np.zeros(len(fr))
    for t in range(len(fr)):
        d = cen[t] - hand_c[t]; n = np.linalg.norm(d); dirs[t] = d / n if n > 1e-6 else np.array([0, 0, 1.0])
        push[t] = need(t, dirs[t], t3[t])
    n_pen0 = int((push > 0).sum()); max0 = float(push.max())
    push_s = np.zeros(len(fr))
    for sl in stretches(fr):
        if sl.stop - sl.start >= 3:
            push_s[sl] = gaussian_filter1d(maximum_filter1d(push[sl], size=25, mode="nearest"), 6.0, mode="nearest")
            dsm = gaussian_filter1d(dirs[sl], 6.0, axis=0, mode="nearest"); nrm = np.linalg.norm(dsm, axis=1, keepdims=True)
            dirs[sl] = np.where(nrm > 1e-6, dsm / np.maximum(nrm, 1e-6), dirs[sl])
        else:
            push_s[sl] = push[sl]
    t3 = t3 + push_s[:, None] * dirs
    for _ in range(3):
        nd = np.array([need(t, dirs[t], t3[t]) for t in range(len(fr))])
        if nd.max() <= 0.0:
            break
        env = np.zeros_like(nd)
        for sl in stretches(fr):
            env[sl] = gaussian_filter1d(maximum_filter1d(nd[sl], size=37, mode="nearest"), 6.0, mode="nearest") + 0.02 if sl.stop - sl.start >= 3 else nd[sl]
        env = np.where(nd > 0, np.maximum(env, 0.0), env); push_s = push_s + env; t3 = t3 + env[:, None] * dirs
    resid = np.zeros(len(fr))
    for t in range(len(fr)):
        resid[t] = need(t, dirs[t], t3[t]); t3[t] = t3[t] + resid[t] * dirs[t]
    full = z["object_translation"].astype(np.float64).copy()
    idx = np.clip(np.searchsorted(fr, np.arange(T)), 0, len(fr) - 1); left = np.clip(idx - 1, 0, len(fr) - 1)
    pick = np.where(np.abs(fr[idx] - np.arange(T)) <= np.abs(fr[left] - np.arange(T)), idx, left)
    full[:] = t3[pick]
    z["object_translation"] = full.astype(np.float32)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"episode_{ep:06d}.npz", **z)
    m.export(dst / f"episode_{ep:06d}_object.obj")
    rec = dict(ep=ep, frames_pen_before=n_pen0, max_pen_push_before_m=round(max0, 3), smoothed_push_max_m=round(float(push_s.max()), 3), resid_max_m=round(float(resid.max()), 3), resid_frames=int((resid > 0).sum()))
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--eps", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--device", default="cuda:3")
    a = ap.parse_args()
    from src.mhr_local import MHR
    mhr = MHR(a.device)
    recs = [push_out(ep, ROOT / a.src, ROOT / a.dst, mhr) for ep in a.eps]
    (ROOT / a.dst / "pushout_report.json").write_text(json.dumps(recs, indent=1))
