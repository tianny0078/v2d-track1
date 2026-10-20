#!/usr/bin/env python3
"""Extra Gaussian smoothing of object_translation inside each scored stretch (ACC-O only reads translation).
Usage: smooth_obj_t.py SRC DST --sigma S [--root work/behave_root]   (dirs relative to root)"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT0 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT0 / "work/kit/v2d_submission_kit"))
from v2dlb.mhr_submission import stretches

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--sigma", type=float, required=True)
ap.add_argument("--root", default=str(ROOT0)); a = ap.parse_args(); R0 = Path(a.root)
src, dst = R0 / a.src, R0 / a.dst; dst.mkdir(parents=True, exist_ok=True)
sc_all = json.load(open(R0 / "work/scored_frames_summary.json"))
for k in sorted(sc_all, key=int):
    ep = int(k); f = src / f"episode_{ep:06d}.npz"
    if not f.exists(): continue
    fr = np.concatenate([np.arange(s, e + 1) for s, e in sc_all[k]["spans"]])
    z = dict(np.load(f)); t = z["object_translation"].astype(np.float64)
    for s in stretches(fr):
        if s.stop - s.start >= 3: t[fr[s]] = gaussian_filter1d(t[fr[s]], a.sigma, axis=0, mode="nearest")
    z["object_translation"] = t.astype(np.float32); np.savez(dst / f.name, **z)
    shutil.copy(src / f"episode_{ep:06d}_object.obj", dst / f"episode_{ep:06d}_object.obj")
print("done", dst)
