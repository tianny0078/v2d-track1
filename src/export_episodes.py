#!/usr/bin/env python3
"""Assemble per-episode submission NPZs from whatever stages have finished.

The packer indexes `pose` by VIDEO FRAME (row t is frame t), but the body fit only solves the
scored frames, so unscored rows are filled from the nearest solved frame. Those rows are never
read by the scorer; they exist only so the array is dense.

Object pose is optional: with --object none the object keeps the format-floor placeholder, which
is what isolates a human-only change on the leaderboard.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object", choices=["none", "mesh"], default="mesh",
                    help="'none' keeps the unit-box placeholder; 'mesh' uses the reconstructed mesh")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    eps_len = {json.loads(l)["episode_index"]: json.loads(l)["length"]
               for l in (ROOT / "data/hf/track_1/meta/episodes.jsonl").open()}
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())

    report = []
    for ep in range(30):
        T = max(eps_len[ep], scored[str(ep)]["last"] + 1)
        b = np.load(ROOT / f"work/body/ep{ep:06d}.npz")
        fids, pose_s = b["frame_ids"], b["pose"]
        # nearest-solved-frame fill for the rows the scorer never reads
        idx = np.clip(np.searchsorted(fids, np.arange(T)), 0, len(fids) - 1)
        left = np.clip(idx - 1, 0, len(fids) - 1)
        pick = np.where(np.abs(fids[idx] - np.arange(T)) <= np.abs(fids[left] - np.arange(T)),
                        idx, left)
        pose = pose_s[pick].astype(np.float32)

        obj_mesh = ROOT / f"work/meshes/{meta[ep]['object']}.glb"
        if a.object == "mesh" and obj_mesh.exists():
            mesh = trimesh.load(str(obj_mesh), force="mesh", process=False)
        else:
            mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        mesh.export(a.out / f"episode_{ep:06d}_object.obj")

        np.savez(a.out / f"episode_{ep:06d}.npz",
                 pose=pose, scales=b["scales"].astype(np.float32), shape=b["shape"].astype(np.float32),
                 object_rotation=np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
                 object_translation=np.zeros((T, 3), np.float32), object_scale=np.float32(1.0))
        report.append(dict(ep=ep, T=int(T), solved=int(len(fids)),
                           filled=int(T - len(fids)), object=meta[ep]["object"]))
    print(json.dumps(dict(episodes=len(report), total_rows=sum(r["T"] for r in report),
                          solved=sum(r["solved"] for r in report)), indent=1))


if __name__ == "__main__":
    main()
