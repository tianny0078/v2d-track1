#!/usr/bin/env python3
"""Decode each episode to JPEG once. Disk is 92% full, so this is sized before it is run.

Only frames from (first_scored - LEAD) onward are kept: earlier frames cannot affect any scored
row except through tracking warm-up, and LEAD=30 (1 s) is more warm-up than any module needs.
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "data/hf/track_1/videos/chunk-000/observation.images.exo_camera"
OUT = ROOT / "work/frames"
STATE = ROOT / "state/frames"
LEAD = 30

scored = json.load((ROOT / "work/scored_frames_summary.json").open())
eps = sys.argv[1:] or [f"{i:06d}" for i in range(30)]

for e in eps:
    ep = int(e)
    done = STATE / f"ep{ep:06d}.done"
    if done.exists():
        continue
    s = scored[str(ep)]
    start = max(0, s["first"] - LEAD)
    d = OUT / f"ep{ep:06d}"
    d.mkdir(parents=True, exist_ok=True)
    # -start_number makes the filename the true video frame index, so nothing downstream
    # has to carry an offset that it could get wrong.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(VID / f"episode_{ep:06d}.mp4"),
                    "-vf", f"select='gte(n\\,{start})'", "-vsync", "0", "-q:v", "2",
                    "-start_number", str(start), str(d / "%06d.jpg")], check=True)
    n = len(list(d.glob("*.jpg")))
    done.write_text(json.dumps(dict(start=start, n=n, last=s["last"])))
    print(f"ep{ep:06d}: {n} frames from {start}", flush=True)
