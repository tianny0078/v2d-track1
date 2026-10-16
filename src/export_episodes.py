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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/kit/v2d_submission_kit"))
from v2dlb.mhr_submission import stretches  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fit_mhr import FOOT_JOINTS, POSE_UNIT_M, POSE_SIGN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object", choices=["none", "mesh"], default="mesh",
                    help="'none' keeps the unit-box placeholder; 'mesh' uses the reconstructed mesh")
    ap.add_argument("--body", type=Path, default=ROOT / "work/body")
    ap.add_argument("--reground", action="store_true",
                    help="After smoothing, slide each frame along the floor normal so the lowest "
                         "foot returns to the floor. Smoothing the pose averages the feet off the "
                         "ground it was fitted onto (0.44 cm -> 4.70 cm at sigma 6); the "
                         "correction is itself smoothed, so it restores contact (-> 2.09 cm) "
                         "without putting the jitter back (|a_pred| 0.3665 -> 0.3718).")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                    help="Gaussian sigma in frames, applied to pose WITHIN each scored stretch. "
                         "Depth is the one direction 2D keypoints do not constrain, so the limbs "
                         "swing in depth; that shows up as |a_pred| 11x the reference's own joint "
                         "acceleration and, we think, as the CD-H regression too. Post-hoc "
                         "filtering beats in-loop penalties here, which saturate: w_jacc=5 left "
                         "|a_pred| at 2.8x while driving reprojection error to 66 px.")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    eps_len = {json.loads(l)["episode_index"]: json.loads(l)["length"]
               for l in (ROOT / "data/hf/track_1/meta/episodes.jsonl").open()}
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())
    ground = json.load((ROOT / "work/camera/ground.json").open())

    report = []
    for ep in range(30):
        T = max(eps_len[ep], scored[str(ep)]["last"] + 1)
        b = np.load(a.body / f"ep{ep:06d}.npz")
        fids, pose_s = b["frame_ids"], b["pose"].copy()
        if a.smooth > 0:
            from scipy.ndimage import gaussian_filter1d
            # per stretch: the metric never differences across a gap, and smoothing across one
            # would drag the two sides of a 360-frame gap together
            for sl in stretches(fids):
                if sl.stop - sl.start >= 3:
                    pose_s[sl] = gaussian_filter1d(pose_s[sl], a.smooth, axis=0, mode="nearest")
            if a.reground:
                from src.mhr_local import MHR
                global _MHR
                try:
                    mhr = _MHR
                except NameError:
                    mhr = _MHR = MHR()
                g = ground[meta[ep]["camera"]]
                gn = np.array(g["normal"])
                pr = np.concatenate([pose_s, np.broadcast_to(b["scales"], (len(pose_s), 68))], 1)
                _, j = mhr.forward(pr, b["shape"])
                # SCALE-FREE: restore the lowest foot to this stretch's own MEAN height, not to
                # the floor's measured distance. Pinning to the absolute distance mixes the body's
                # metric scale with MoGe's and is what sent CD-H from 82.68 to 96.92.
                h = (j[:, list(FOOT_JOINTS)] @ gn).min(1)
                for sl in stretches(fids):
                    if sl.stop - sl.start >= 3:
                        dh = gaussian_filter1d(h[sl].mean() - h[sl], a.smooth, mode="nearest")
                        pose_s[sl, :3] += (dh[:, None] * gn[None, :]) / (np.array(POSE_SIGN) * POSE_UNIT_M)
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
