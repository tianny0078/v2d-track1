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


def strip(ep: int, n: int = 5, width: int = 320, zoom: bool = False) -> Path:
    """zoom=True crops each tile around the OBJECT mask. Small objects (a paint roller is ~1k px
    in a 1.8M px frame) are invisible at strip scale, so a full-frame strip cannot verify them."""
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
        box = None
        if zoom and m[0].any():
            ys, xs = np.where(m[0])
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            half = max(80, max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.8 / 2)
            box = (int(max(0, cx - half)), int(max(0, cy - half)),
                   int(min(W, cx + half)), int(min(H, cy + half)))
        for c in (0, 1):
            img[m[c]] = 0.55 * img[m[c]] + 0.45 * COLORS[c]
            edge = m[c] ^ np.pad(m[c], 1, mode="edge")[2:, 1:-1]
            img[edge] = COLORS[c]
        tile = Image.fromarray(img.astype(np.uint8))
        if box:
            tile = tile.crop(box)
        t = tile.resize((width, int(width * tile.height / tile.width)))
        tiles.append(np.asarray(t))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    out = Image.fromarray(np.concatenate(tiles, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"segzoom_ep{ep:06d}.jpg" if zoom else f"seg_ep{ep:06d}.jpg")
    out.save(p, quality=82)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eps", type=int, nargs="+")
    ap.add_argument("--zoom", action="store_true")
    a = ap.parse_args()
    for ep in a.eps:
        print(strip(ep, zoom=a.zoom))
