#!/usr/bin/env python3
"""Score a prediction directory against the BEHAVE ground truth in work/behave_gt.

Layout (both sides = our export format):  <dir>/<seq>/episode_000000.npz + episode_000000_object.obj
Metrics follow metric_code/track_1 exactly in structure: one Sim(3) fitted on the FIRST scored frame's
alignment vertices and applied to everything, symmetric Chamfer on sparse body points (CD-H) and on
surface samples of the posed object (CD-O), 22-joint / object-translation acceleration error per
contiguous stretch (ACC-H / ACC-O), and penetration of our hand points into our own object (PEN).
Point-set caveat: the host's role_alignment/role_body/hand index sets are not in the kit, so the
FPS proxies from src/mhr_local.py are used (score_local.py does the same). Numbers are therefore
ablation signals with the right units, not leaderboard predictions; ACC and PEN are exact in form.

  python src/score_behave.py --pred work/behave_pred/<name> [--seqs ...] [--frames 200-210] [--out f.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.score_local import load_episode, penetration, compare  # noqa: E402
from src.mhr_local import MHR  # noqa: E402

GT = ROOT / "work/behave_gt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--frames", help="a-b (half-open) to score a sub-range instead of the whole clip")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    seqs = a.seqs or sorted(p.name for p in GT.iterdir() if (p / "episode_000000.npz").exists() and (a.pred / p.name).exists())
    mhr = MHR(a.device)
    rows = []
    for i, seq in enumerate(seqs):
        frames = np.array(json.load((GT / seq / "scored_frames.json").open())["frames"])
        if a.frames:
            lo, hi = map(int, a.frames.split("-")); frames = frames[(frames >= lo) & (frames < hi)]
        pred = load_episode(a.pred / seq, 0, frames, mhr)
        ref = load_episode(GT / seq, 0, frames, mhr)
        r = dict(seq=seq, n_scored=int(len(frames)))
        r.update(penetration(pred, mhr))
        c = compare(pred, ref, mhr, frames, seed=i)
        r.update({k.replace("_proxy", ""): v for k, v in c.items()})
        r["pen_cm"] = r["pen_cm"] * r["align_scale"]           # host multiplies by the frame-0 alignment scale
        rows.append(r); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    keys = ["cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm", "pen_cm", "align_scale"]
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    print("MEAN", json.dumps({k: round(v, 4) for k, v in mean.items()}))
    if a.out:
        a.out.write_text(json.dumps(dict(per_seq=rows, mean=mean), indent=1))


if __name__ == "__main__":
    main()
