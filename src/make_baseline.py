#!/usr/bin/env python3
"""Baseline episode NPZs + object meshes: the format floor.

Purpose is NOT score, it is to prove the output contract end to end: the packer accepts it,
every scored row is covered, and the Kaggle uploader has something real to reject or accept.
Human = MHR rest pose. Object = unit box at the origin.
"""
import json
import numpy as np
import trimesh
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/baseline_episodes"
OUT.mkdir(parents=True, exist_ok=True)
scored = json.load((ROOT / "work/scored_frames_summary.json").open())
eps = {json.loads(l)["episode_index"]: json.loads(l)["length"]
       for l in (ROOT / "data/hf/track_1/meta/episodes.jsonl").open()}

for ep, length in sorted(eps.items()):
    T = max(length, scored[str(ep)]["last"] + 1)
    np.savez(
        OUT / f"episode_{ep:06d}.npz",
        pose=np.zeros((T, 136), np.float32),
        scales=np.zeros(68, np.float32),
        shape=np.zeros(45, np.float32),
        object_rotation=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
        object_translation=np.zeros((T, 3), np.float32),
        object_scale=np.float32(1.0),
    )
    # A box is a legitimate placeholder: it has volume, so PEN is defined, and area, so CD-O samples.
    trimesh.creation.box(extents=(0.3, 0.3, 0.3)).export(OUT / f"episode_{ep:06d}_object.obj")

print("wrote", len(list(OUT.glob("*.npz"))), "npz +", len(list(OUT.glob("*_object.obj"))), "meshes")
