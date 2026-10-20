#!/usr/bin/env python3
"""Turn an exported submission dir (episode_00000k.npz, pose for every frame) into a body dir
(work/body_*/ep00000k.npz: scored-frame rows only, like work/body_flow_px) so export_episodes.py
can place our objects relative to that body.  Usage: body_from_export.py work/sub_cari4d_cc_fix work/body_cari4d"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
src, dst = ROOT / sys.argv[1], ROOT / sys.argv[2]; dst.mkdir(parents=True, exist_ok=True)
sc = json.load(open(ROOT / "work/scored_frames_summary.json"))
for k in sorted(sc, key=int):
    ep = int(k); z = np.load(src / f"episode_{ep:06d}.npz")
    fr = np.concatenate([np.arange(s, e + 1) for s, e in sc[k]["spans"]])
    np.savez(dst / f"ep{ep:06d}.npz", pose=z["pose"][fr], scales=z["scales"], shape=z["shape"],
             frame_ids=fr, covered=np.ones(len(fr), bool), stats=np.array({"src": str(src)}))
print("wrote", len(sc), "->", dst)
