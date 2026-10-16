#!/usr/bin/env python3
"""Local ACC diagnostic, calibrated by the leaderboard.

The format-floor submission predicted a perfectly static body, so its ACC-H of 0.14528 IS the
reference's own mean joint acceleration: error = |a_pred - a_gt| = |a_gt| when a_pred = 0.
Same for ACC-O: 0.07429 is the reference object's own translation acceleration.

That turns ACC into something we can steer locally. Our own |a_pred| is computable; if it is far
above 0.145 we are injecting jitter and will score worse than doing nothing. Predicting zero
motion is a hard FLOOR of 0.145 that heavy smoothing can always reach, so there is no reason ever
to score above it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
from src.mhr_local import MHR, BODY_JOINT_INDICES  # noqa: E402
from v2dlb.mhr_submission import stretches  # noqa: E402

GT_ACC_H = 0.14528      # measured: the static submission's ACC-H
GT_ACC_O = 0.07429


def episode_accel(pose, scales, shape, frames, mhr) -> float:
    params = np.concatenate([pose, np.broadcast_to(scales, (len(pose), 68))], 1)
    _, j = mhr.forward(params, shape)
    j = j[:, list(BODY_JOINT_INDICES)] * 100.0        # cm, as the metric uses
    out = []
    for sl in stretches(frames):
        seg = j[sl]
        if len(seg) < 3:
            continue
        a = seg[:-2] - 2 * seg[1:-1] + seg[2:]
        out.append(np.linalg.norm(a, axis=-1).mean())
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=Path, default=ROOT / "work/body")
    ap.add_argument("--eps", type=int, nargs="*")
    a = ap.parse_args()
    mhr = MHR()
    rows = []
    for ep in (a.eps if a.eps else range(30)):
        z = np.load(a.body / f"ep{ep:06d}.npz")
        v = episode_accel(z["pose"], z["scales"], z["shape"], z["frame_ids"], mhr)
        rows.append(v)
        print(f"ep{ep:02d} |a_pred| = {v:7.4f} cm/frame^2   ({v / GT_ACC_H:5.1f}x 真值自身加速度)",
              flush=True)
    m = float(np.nanmean(rows))
    print(f"\nMEAN |a_pred| = {m:.4f}   真值 |a_gt| = {GT_ACC_H:.4f}   比值 {m / GT_ACC_H:.1f}x")
    print("静止预测的 ACC-H 地板 = %.4f；我们上次提交 = 1.35547" % GT_ACC_H)


if __name__ == "__main__":
    main()
