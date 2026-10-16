#!/usr/bin/env python3
"""One representative frame per episode (mid-point of the scored contact span), as a web thumbnail."""
import json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "data/hf/track_1/videos/chunk-000/observation.images.exo_camera"
OUT = ROOT / "web/assets"; OUT.mkdir(parents=True, exist_ok=True)
scored = json.load((ROOT / "work/scored_frames_summary.json").open())
for ep in range(30):
    s = scored.get(str(ep), {})
    mid = (s.get("first", 100) + s.get("last", 200)) // 2
    dst = OUT / f"ep{ep:06d}.jpg"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(mid / 30.0),
                    "-i", str(VID / f"episode_{ep:06d}.mp4"), "-frames:v", "1",
                    "-vf", "scale=384:-1", "-q:v", "6", str(dst)], check=True)
print("wrote", len(list(OUT.glob('ep*.jpg'))), "thumbnails")
