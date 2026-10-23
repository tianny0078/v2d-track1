#!/usr/bin/env python3
"""Ground-contact depth correction (GT-free). Monocular depth error is mostly a scale along the camera
ray; the feet stay on one ground plane. Per scored frame: g = n.v_lowest (height of the lowest body
vertex along the MoGe ground normal, camera-centred, so the floor sits at g≈-1.2 m), k = g_ref / g,
k smoothed (sigma) per stretch, and the body is TRANSLATED by (k-1)*centroid (depth moved along the
centroid ray; shape and size untouched). g_ref = median g over the episode. Objects are moved with the
same per-frame shift so object-to-body stays as exported.
Usage: ground_ray.py SRC DST [--sigma 10] [--root work/behave_root] [--max-dk 0.15]"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT0 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT0)); sys.path.insert(0, str(ROOT0 / "work/kit/v2d_submission_kit"))
from v2dlb.mhr_submission import stretches
from src.mhr_local import MHR
POSE_UNIT_M, POSE_SIGN = 0.1, np.array([1.0, -1.0, -1.0])
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--sigma", type=float, default=10)
ap.add_argument("--root", default=str(ROOT0)); ap.add_argument("--max-dk", type=float, default=0.15); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--move-object", type=int, default=1)
ap.add_argument("--mode", choices=["ray", "vertical"], default="ray", help="ray: depth along the centroid ray; vertical: shift along the ground normal by (g_ref - g)")
a = ap.parse_args(); R0 = Path(a.root); src, dst = R0 / a.src, R0 / a.dst; dst.mkdir(parents=True, exist_ok=True)
mhr = MHR(a.device)
sc_all = json.load(open(R0 / "work/scored_frames_summary.json")); ground = json.load(open(R0 / "work/camera/ground.json"))
meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(R0 / "data/hf/track_1/meta/episodes_metadata.jsonl")}
for k_ in sorted(sc_all, key=int):
    ep = int(k_); f = src / f"episode_{ep:06d}.npz"
    if not f.exists(): continue
    cam = meta[ep]["camera"]; gk = cam if cam in ground else next((g for g in ground if g.endswith(cam) or cam.endswith(g)), None)
    n = np.array(ground[gk]["normal"], float); n /= np.linalg.norm(n)
    z = dict(np.load(f)); fr = np.concatenate([np.arange(s, e + 1) for s, e in sc_all[k_]["spans"]])
    p = np.concatenate([z["pose"][fr], np.broadcast_to(z["scales"], (len(fr), 68))], 1); v, _ = mhr.forward(p, z["shape"])
    g = (v @ n).min(1)                      # lowest point along "up" (n points up, floor is below camera: g<0)
    c = v.mean(1)
    gref = np.median(g); kk = gref / g
    for sl in stretches(fr):
        if sl.stop - sl.start >= 3: kk[sl] = gaussian_filter1d(kk[sl], a.sigma, mode="nearest")
    kk = np.clip(kk, 1 - a.max_dk, 1 + a.max_dk)
    shift = (kk - 1)[:, None] * c           # metres, camera frame
    if a.mode == "vertical":
        dg = gref - g
        for sl in stretches(fr):
            if sl.stop - sl.start >= 3: dg[sl] = gaussian_filter1d(dg[sl], a.sigma, mode="nearest")
        shift = dg[:, None] * n[None]
    z["pose"] = z["pose"].copy(); z["pose"][fr, :3] += (shift / (POSE_SIGN * POSE_UNIT_M)).astype(z["pose"].dtype)
    if a.move_object:
        z["object_translation"] = z["object_translation"].copy(); z["object_translation"][fr] += shift.astype(z["object_translation"].dtype)
    # roundtrip check: the vertices must move by exactly `shift`
    p2 = np.concatenate([z["pose"][fr[:1]], z["scales"][None]], 1); v2, _ = mhr.forward(p2, z["shape"])
    err = float(np.abs((v2[0] - v[0]) - shift[0]).max())
    np.savez(dst / f.name, **z); shutil.copy(src / f"episode_{ep:06d}_object.obj", dst / f"episode_{ep:06d}_object.obj")
    print(f"ep{ep}: g_ref {gref:.2f} m, k p5/p95 {np.percentile(kk,5):.3f}/{np.percentile(kk,95):.3f}, |shift| mean {np.linalg.norm(shift,axis=1).mean()*100:.1f} cm, roundtrip {err:.1e} m", flush=True)
