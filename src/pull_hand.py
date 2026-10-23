#!/usr/bin/env python3
"""Pull the object toward the nearest hand: per scored frame, centroid c -> h + alpha (c - h), where h is
the nearer wrist joint; the shift is smoothed (sigma) per stretch. Rotation/scale kept.
Usage: pull_hand.py SRC DST --alpha 0.7 [--sigma 6] [--root .]"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.ndimage import gaussian_filter1d
ROOT0 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT0)); sys.path.insert(0, str(ROOT0 / "work/kit/v2d_submission_kit"))
from v2dlb.mhr_submission import stretches
from src.mhr_local import MHR, L_WRIST, R_WRIST
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--sigma", type=float, default=6); ap.add_argument("--root", default=str(ROOT0)); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args(); R0 = Path(a.root); src, dst = R0 / a.src, R0 / a.dst; dst.mkdir(parents=True, exist_ok=True)
mhr = MHR(a.device); sc_all = json.load(open(R0 / "work/scored_frames_summary.json"))
for k in sorted(sc_all, key=int):
    ep = int(k); f = src / f"episode_{ep:06d}.npz"
    if not f.exists(): continue
    z = dict(np.load(f)); fr = np.concatenate([np.arange(s, e + 1) for s, e in sc_all[k]["spans"]])
    p = np.concatenate([z["pose"][fr], np.broadcast_to(z["scales"], (len(fr), 68))], 1); _, j = mhr.forward(p, z["shape"])
    hands = j[:, [L_WRIST, R_WRIST]]                                                   # [T,2,3] m
    mv = np.asarray(trimesh.load(str(f).replace(".npz", "_object.obj"), process=False).vertices).mean(0)
    c = np.einsum("tij,j->ti", z["object_rotation"][fr], float(z["object_scale"]) * mv) + z["object_translation"][fr]
    near = np.linalg.norm(hands - c[:, None], axis=2).argmin(1); h = hands[np.arange(len(fr)), near]
    shift = (a.alpha - 1.0) * (c - h)
    for sl in stretches(fr):
        if sl.stop - sl.start >= 3: shift[sl] = gaussian_filter1d(shift[sl], a.sigma, axis=0, mode="nearest")
    z["object_translation"] = z["object_translation"].copy(); z["object_translation"][fr] += shift.astype(z["object_translation"].dtype)
    np.savez(dst / f.name, **z); shutil.copy(str(f).replace(".npz", "_object.obj"), dst / f.name.replace(".npz", "_object.obj"))
    print(f"ep{ep}: centroid-hand {np.linalg.norm(c-h,axis=1).mean()*100:.0f} cm -> shift {np.linalg.norm(shift,axis=1).mean()*100:.0f} cm", flush=True)
