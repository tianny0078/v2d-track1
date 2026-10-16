#!/usr/bin/env python3
"""Mask overlay strips: the check that segmentation is actually on the right thing.

Writes web/assets/seg_epNNNNNN.jpg -- a row of frames spanning the scored span with the object
mask in amber and the person mask in cyan. Low-contrast outlines are useless at thumbnail size,
so masks are drawn as filled alpha plus a hard edge.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/assets"
COLORS = np.array([[255, 176, 0], [0, 200, 255]], np.float32)  # object, person


def strip(ep: int, n: int = 5, width: int = 320) -> Path:
    z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz")
    H, W = z["shape"]
    frame_ids = z["frame_ids"]
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]
    picks = np.linspace(scored["first"], scored["last"], n).astype(int)
    tiles = []
    for f in picks:
        i = int(np.searchsorted(frame_ids, f))
        img = np.asarray(Image.open(ROOT / f"work/frames/ep{ep:06d}/{frame_ids[i]:06d}.jpg")
                         .convert("RGB"), np.float32)
        m = np.unpackbits(z["packed"][i], axis=-1)[:, :, :W].astype(bool)
        for c in (0, 1):
            img[m[c]] = 0.55 * img[m[c]] + 0.45 * COLORS[c]
            edge = m[c] ^ np.pad(m[c], 1, mode="edge")[2:, 1:-1]
            img[edge] = COLORS[c]
        t = Image.fromarray(img.astype(np.uint8)).resize((width, int(width * H / W)))
        tiles.append(np.asarray(t))
    out = Image.fromarray(np.concatenate(tiles, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"seg_ep{ep:06d}.jpg"
    out.save(p, quality=82)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eps", type=int, nargs="+"); a = ap.parse_args()
    for ep in a.eps:
        print(strip(ep))
