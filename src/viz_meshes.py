#!/usr/bin/env python3
"""Render each object mesh AT THE SUBMISSION BUDGET, from three viewpoints.

Decimating first is the point: the host scores the 4096-vertex / 4096-face mesh the packer
produces, not the 400k-face mesh the generator emits, so that is what has to be inspected.
Aspect ratios alone do not tell you whether a shape is right.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MESHES, OUT = ROOT / "work/meshes", ROOT / "web/assets"
BUDGET_V = BUDGET_F = 4096
SIZE = 260


def budget(tm: trimesh.Trimesh) -> trimesh.Trimesh:
    """Same path the official packer takes: fast_simplification to the face budget."""
    import fast_simplification
    if len(tm.faces) <= BUDGET_F and len(tm.vertices) <= BUDGET_V:
        return tm
    v, f = fast_simplification.simplify(np.asarray(tm.vertices, np.float32),
                                        np.asarray(tm.faces, np.int32),
                                        target_count=BUDGET_F)
    return trimesh.Trimesh(v, f, process=False)


def shade(tm: trimesh.Trimesh, yaw: float, pitch: float) -> np.ndarray:
    """Painter's-algorithm software render: no GL, no display, works headless."""
    v = np.asarray(tm.vertices, np.float64)
    v = v - v.mean(0)
    s = np.abs(v).max() or 1.0
    v /= s
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    p = v @ R.T
    f = np.asarray(tm.faces)
    tri = p[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln == 0, 1, ln)
    light = np.array([0.4, 0.7, 0.6]); light /= np.linalg.norm(light)
    inten = np.clip(np.abs(n @ light), 0, 1) * 0.75 + 0.2
    order = np.argsort(tri[:, :, 2].mean(1))            # far to near
    img = np.full((SIZE, SIZE, 3), 18, np.uint8)
    xy = ((tri[..., :2] * 0.42 + 0.5) * SIZE).astype(np.int32)
    from PIL import ImageDraw
    pim = Image.fromarray(img); d = ImageDraw.Draw(pim)
    for i in order:
        c = int(inten[i] * 215)
        d.polygon([tuple(xy[i, k]) for k in range(3)], fill=(c, int(c * 0.93), int(c * 0.8)))
    return np.asarray(pim.transpose(Image.FLIP_TOP_BOTTOM))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    from PIL import ImageDraw
    for p in sorted(MESHES.glob("*.glb")):
        tm = budget(trimesh.load(str(p), force="mesh", process=False))
        name = p.stem
        views = [shade(tm, y, 0.35) for y in (0.0, 2.1, 4.2)]
        inp = Image.open(MESHES / f"{name}_input.png").convert("RGB")
        inp.thumbnail((SIZE, SIZE))
        pad = Image.new("RGB", (SIZE, SIZE), (18, 18, 18))
        pad.paste(inp, ((SIZE - inp.width) // 2, (SIZE - inp.height) // 2))
        strip = np.concatenate([np.asarray(pad)] + views, 1)
        lab = Image.new("RGB", (strip.shape[1], 22), (0, 0, 0))
        ImageDraw.Draw(lab).text((5, 5), f"{name}   {len(tm.vertices)}v / {len(tm.faces)}f  "
                                         f"(输入 | 三视角)", fill=(255, 220, 0))
        rows.append(np.concatenate([np.asarray(lab), strip], 0))
        stats[name] = dict(v=int(len(tm.vertices)), f=int(len(tm.faces)),
                           watertight=bool(tm.is_watertight), euler=int(tm.euler_number))
        Image.fromarray(strip).save(OUT / f"mesh_{name}.jpg", quality=84)
    Image.fromarray(np.concatenate(rows, 0)).save(ROOT / "work/mesh_review.jpg", quality=82)
    (ROOT / "work/mesh_budget_stats.json").write_text(json.dumps(stats, indent=1))
    for k, v in stats.items():
        print(f"{k:<19} {v['v']:>5}v {v['f']:>5}f watertight={v['watertight']} euler={v['euler']}")


if __name__ == "__main__":
    main()
