#!/usr/bin/env python3
"""Pick, per episode, the rotation track with the best mean silhouette IoU among the variants
(_rot, _rot_ss, _rot_rd) and copy it to ep_rot_best.npz. Reads state/objtrack/*.done."""
import json, shutil, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = []
for ep in range(30):
    best = None
    for tag in ("", "_ss", "_rd"):
        f = ROOT / f"state/objtrack/ep{ep:06d}{tag}.done"
        if f.exists():
            r = json.loads(f.read_text()); r["tag"] = tag
            if best is None or r["iou_track"] > best["iou_track"]: best = r
    shutil.copy(ROOT / f"work/objtrack/ep{ep:06d}_rot{best['tag']}.npz", ROOT / f"work/objtrack/ep{ep:06d}_rot_best.npz")
    rows.append((ep, best["tag"] or "base", best["iou_track"]))
print("RESULT best-of tracks: mean IoU %.3f; picks %s" % (np.mean([r[2] for r in rows]), [(e, t, i) for e, t, i in rows if t != "base"]))
