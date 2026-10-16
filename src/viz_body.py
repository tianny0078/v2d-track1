#!/usr/bin/env python3
"""Overlay the fitted MHR body on the frames.

A low reprojection error does not mean the 3D pose is right -- a limb flipped in depth projects
to almost the same place. So this draws the full mesh (depth-shaded), not just the joints, and
adds a side view that makes a depth flip obvious.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/assets"


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.mhr_local import MHR
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    z = np.load(ROOT / f"work/body/ep{a.ep:06d}.npz")
    fids = z["frame_ids"]
    K = json.load((ROOT / "work/camera/intrinsics.json").open())
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    k = K[meta[a.ep]["camera"]]

    mhr = MHR(a.device)
    picks = np.linspace(0, len(fids) - 1, a.n).astype(int)
    params = np.concatenate([z["pose"][picks], np.broadcast_to(z["scales"], (len(picks), 68))], 1)
    verts, _ = mhr.forward(params, z["shape"])
    faces = mhr.faces

    tiles = []
    for n, i in enumerate(picks):
        img = Image.open(ROOT / f"work/frames/ep{a.ep:06d}/{fids[i]:06d}.jpg").convert("RGB")
        v = verts[n]
        u = k["fx"] * v[:, 0] / v[:, 2] + k["cx"]
        w = k["fy"] * v[:, 1] / v[:, 2] + k["cy"]
        d = ImageDraw.Draw(img, "RGBA")
        depth = v[:, 2]
        lo, hi = depth.min(), depth.max()
        step = max(1, len(faces) // 6000)
        order = np.argsort(-depth[faces[::step, 0]])
        for fi in faces[::step][order]:
            # Flat fill with a narrow shade range. A wide depth ramp made the nearest body part
            # read as a white blob and sent me chasing a "deformed head" that measurement showed
            # was within 3% of the rest shape.
            t = float((depth[fi].mean() - lo) / max(1e-6, hi - lo))
            col = (int(60 + 40 * (1 - t)), int(170 + 30 * (1 - t)), int(235), 150)
            d.polygon([(u[q], w[q]) for q in fi], fill=col)
        img.thumbnail((430, 430))
        # side view: rotate the same mesh 90 deg about y, so a depth flip is visible
        c = v - v.mean(0)
        side = np.stack([c[:, 2], c[:, 1]], 1)
        s = 170 / max(1e-6, np.abs(side).max())
        sv = Image.new("RGB", (img.height, img.height), (16, 16, 16))
        ds = ImageDraw.Draw(sv)
        for q in range(0, len(side), 7):
            x, y = side[q] * s + img.height / 2
            ds.point((x, y), fill=(120, 210, 255))
        tiles.append(np.concatenate([np.asarray(img),
                                     np.asarray(sv.resize((img.height, img.height)))], 1))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"body_ep{a.ep:06d}.jpg"
    Image.fromarray(np.concatenate(tiles, 0)).save(p, quality=84)
    print(p)


if __name__ == "__main__":
    main()
