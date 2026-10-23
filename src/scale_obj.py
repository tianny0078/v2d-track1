#!/usr/bin/env python3
"""Multiply object_scale by a factor about the object's vertex centroid (exported meshes are NOT centred:
world centroid = R (s mean(v)) + t, so t is corrected to keep that point fixed).
Usage: scale_obj.py SRC DST --factor 0.8 [--root .]"""
import argparse, shutil
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--factor", type=float, required=True); ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args(); src, dst = Path(a.root) / a.src, Path(a.root) / a.dst; dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("episode_*.npz")):
    import trimesh
    z = dict(np.load(f)); mv = np.asarray(trimesh.load(str(f).replace(".npz", "_object.obj"), process=False).vertices).mean(0)
    s0 = float(z["object_scale"]); c_loc = s0 * mv
    z["object_translation"] = (z["object_translation"] + (1 - a.factor) * np.einsum("tij,j->ti", z["object_rotation"], c_loc)).astype(z["object_translation"].dtype)
    z["object_scale"] = np.asarray(s0 * a.factor, dtype=z["object_scale"].dtype); np.savez(dst / f.name, **z)
    shutil.copy(str(f).replace(".npz", "_object.obj"), dst / f.name.replace(".npz", "_object.obj"))
print("scaled", len(list(dst.glob('*.npz'))), "->", dst)
