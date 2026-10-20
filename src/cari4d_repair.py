#!/usr/bin/env python3
"""Repair lost-object spans in an official-CARI4D export (GT-free).

FoundationPose sometimes loses the object (off-image hoop, occluded box) and CoCoNet then puts it
metres away or behind the camera (challenge ep2: z -1.96 m; BEHAVE boxmedium/boxsmall p90 distance
to the nearest wrist 3.6-4.2 m, plasticcontainer the whole clip at ~10 m). Those frames are easy to
flag without ground truth:
  invalid = z < 0.5 m  OR  > MAX_PELVIS m from the pelvis  OR  > 0.5 m from the 31-frame running median.
Invalid frames get translation interpolated and rotation slerped from the nearest valid frames,
then translation is re-smoothed (sigma 3) inside each scored stretch. If more than half of an
episode's scored frames are invalid, the whole object (mesh, R, t, scale) is taken from a fallback
export instead (our own MoGe-placed object).
Usage: cari4d_repair.py SRC DST --fallback DIR [--eps ...] [--root work/behave_root]"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
ROOT0 = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--fallback", required=True)
    ap.add_argument("--eps", type=int, nargs="*"); ap.add_argument("--root", default=str(ROOT0)); ap.add_argument("--max-pelvis", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-invalid", type=float, default=0.5, help="above this fraction of invalid scored frames, take the whole object from --fallback")
    a = ap.parse_args(); R0 = Path(a.root)
    sys.path.insert(0, str(R0)); sys.path.insert(0, str(R0 / "src")); sys.path.insert(0, str(ROOT0 / "work/kit/v2d_submission_kit"))
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.spatial.transform import Rotation as Rot, Slerp
    from v2dlb.mhr_submission import stretches
    from mhr_local import MHR
    mhr = MHR(a.device)
    src, dst, fb = (R0 / a.src), (R0 / a.dst), (R0 / a.fallback); dst.mkdir(parents=True, exist_ok=True)
    sc_all = json.load(open(R0 / "work/scored_frames_summary.json"))
    eps = a.eps if a.eps else sorted(int(k) for k in sc_all)
    for ep in eps:
        sc = sc_all[str(ep)]; fr = np.concatenate([np.arange(s, e + 1) for s, e in sc["spans"]])
        z = dict(np.load(src / f"episode_{ep:06d}.npz")); t = z["object_translation"].astype(np.float64); Rm = z["object_rotation"].astype(np.float64)
        p = np.concatenate([z["pose"][fr], np.broadcast_to(z["scales"], (len(fr), 68))], 1); _, j = mhr.forward(p, z["shape"])
        tf = t[fr]; med = median_filter(tf, size=(31, 1), mode="nearest")
        bad_f = (tf[:, 2] < 0.5) | (np.linalg.norm(tf - j[:, 1], axis=1) > a.max_pelvis) | (np.linalg.norm(tf - med, axis=1) > 0.5)
        frac = float(bad_f.mean()); obj_src = "official"
        if frac > a.max_invalid:
            for suf in (".npz", "_object.obj"):
                pass
            fz = dict(np.load(fb / f"episode_{ep:06d}.npz"))
            for k in ("object_rotation", "object_translation", "object_scale"): z[k] = fz[k]
            shutil.copy(fb / f"episode_{ep:06d}_object.obj", dst / f"episode_{ep:06d}_object.obj"); obj_src = "fallback"
        else:
            shutil.copy(src / f"episode_{ep:06d}_object.obj", dst / f"episode_{ep:06d}_object.obj")
            if bad_f.any():
                good = fr[~bad_f]; bad = fr[bad_f]
                for c in range(3): t[bad, c] = np.interp(bad, good, t[good, c])
                Rm[bad] = Slerp(good, Rot.from_matrix(Rm[good]))(np.clip(bad, good[0], good[-1])).as_matrix()
                for s in stretches(fr):
                    if s.stop - s.start >= 3: t[fr[s]] = gaussian_filter1d(t[fr[s]], 3.0, axis=0, mode="nearest")
            z["object_translation"] = t.astype(np.float32); z["object_rotation"] = Rm.astype(np.float32)
        np.savez(dst / f"episode_{ep:06d}.npz", **z)
        print(f"RESULT ep{ep}: invalid {int(bad_f.sum())}/{len(fr)} ({frac:.0%}) -> {obj_src}", flush=True)

if __name__ == "__main__":
    main()
