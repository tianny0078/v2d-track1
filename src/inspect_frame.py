#!/usr/bin/env python3
"""Render a frame with a labelled pixel grid, so a box can be read off it by eye.

Used when automatic detection picks the wrong object. Any seed chosen this way is a MANUAL step
and must be listed as such in the technical report -- see notes/00-plan-v2.md section 7.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def render(ep: int, frame: int, crop: list[int] | None, step: int, out: Path) -> Path:
    img = Image.open(ROOT / f"work/frames/ep{ep:06d}/{frame:06d}.jpg").convert("RGB")
    ox, oy = 0, 0
    if crop:
        ox, oy = crop[0], crop[1]
        img = img.crop(tuple(crop))
    scale = max(1, 1400 // max(img.size))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for x in range(0, img.width, step * scale):
        d.line([(x, 0), (x, img.height)], fill=(255, 0, 0), width=1)
        d.text((x + 2, 2), str(ox + x // scale), fill=(255, 255, 0))
    for y in range(0, img.height, step * scale):
        d.line([(0, y), (img.width, y)], fill=(255, 0, 0), width=1)
        d.text((2, y + 2), str(oy + y // scale), fill=(0, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--crop", type=int, nargs=4, help="x0 y0 x1 y1 in full-frame pixels")
    ap.add_argument("--step", type=int, default=100, help="grid spacing in source pixels")
    a = ap.parse_args()
    p = ROOT / f"work/inspect/ep{a.ep:06d}_f{a.frame:06d}.jpg"
    print(render(a.ep, a.frame, a.crop, a.step, p))
