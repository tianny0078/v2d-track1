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


def _euler_to_mat(e):
    """Host's euler-xyz convention (metric_code/track_1/*.py::_euler_xyz_to_mat), transcribed."""
    cx, cy, cz = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
    sx, sy, sz = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
    return np.stack([np.stack([cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz], -1),
                     np.stack([cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz], -1),
                     np.stack([-sy, sx * cy, cx * cy], -1)], -2)


def _mat_to_euler(R):
    y = np.arcsin(np.clip(-R[..., 2, 0], -1, 1))
    return np.stack([np.arctan2(R[..., 2, 1], R[..., 2, 2]), y,
                     np.arctan2(R[..., 1, 0], R[..., 0, 0])], -1)


def _smooth_wrap_safe(seg: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a pose stretch WITHOUT blending the root rotation across the +-pi wrap.

    pose[3:6] is the root rotation as euler xyz. A subject who turns around (ep26/ep18: sit on
    the stool and rotate) sweeps yaw through +-pi; the raw sequence then jumps by 2*pi on two
    channels (ep26: 20 such frames). A plain Gaussian interpolates across that discontinuity and
    manufactures a half-turn -- measured: ep26's joint acceleration went 1.65 -> 9.58 cm/frame^2
    AFTER smoothing, and the leaderboard ACC-H came back 1.103 against a predicted ~0.12.
    Smoothing the first two columns of the rotation matrix (a 6D representation with no wrap),
    re-orthonormalising and converting back fixes it: ep26 -> 0.116 at sigma 3.
    Translation and the 130 body-pose params are small-angle and get the plain Gaussian.
    """
    from scipy.ndimage import gaussian_filter1d
    out = gaussian_filter1d(seg, sigma, axis=0, mode="nearest")
    R = _euler_to_mat(seg[:, 3:6])
    c = gaussian_filter1d(R[..., :2], sigma, axis=0, mode="nearest")
    a = c[..., 0] / np.linalg.norm(c[..., 0], axis=-1, keepdims=True)
    b = c[..., 1] - (a * c[..., 1]).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    out[:, 3:6] = _mat_to_euler(np.stack([a, b, np.cross(a, b)], -1))
    return out


def _sam3d_as_body(path: Path) -> dict:
    """SAM 3D Body per-frame output -> the same dict layout the fitter writes."""
    from src.fit_mhr import POSE_UNIT_M, POSE_SIGN
    z = np.load(path)
    ok = z["ok"]
    if not ok.any():
        raise SystemExit(f"{path}: SAM 3D Body produced no frames")
    mp, ct = z["model_params"].copy(), z["cam_t"]
    # frames SAM3D missed: borrow the nearest good frame so the sequence is dense
    good = np.where(ok)[0]
    near = good[np.abs(good[None, :] - np.arange(len(ok))[:, None]).argmin(1)]
    mp, ct = mp[near], ct[near]
    pose = mp[:, :136].astype(np.float32)
    pose[:, :3] = (ct * (10.0 / (POSE_UNIT_M * 10.0)) * np.array(POSE_SIGN, np.float32)).astype(np.float32)
    scales = np.median(mp[good, 136:], axis=0).astype(np.float32)
    shape = np.median(z["shape"][good], axis=0).astype(np.float32)
    return dict(frame_ids=z["frame_ids"], pose=pose, scales=scales, shape=shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object", choices=["none", "mesh"], default="mesh",
                    help="'none' keeps the unit-box placeholder; 'mesh' uses the reconstructed mesh")
    ap.add_argument("--body", type=Path, default=ROOT / "work/body")
    ap.add_argument("--body-source", choices=["fit", "sam3d"], default="fit",
                    help="'sam3d' reads work/sam3d/epNNNNNN.npz (SAM 3D Body per-frame output) and "
                         "builds pose from model_params[:136] with pose[0:3] taken from pred_cam_t "
                         "(verified in our camera frame by reprojection: 20 px from the VitPose "
                         "centroid; every flipped hypothesis was 485 px off or behind the camera). "
                         "shape and scales are the per-episode MEDIAN over good frames, because "
                         "the submission requires one identity per episode.")
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
        if a.body_source == "sam3d":
            b = _sam3d_as_body(ROOT / f"work/sam3d/ep{ep:06d}.npz")
        else:
            b = np.load(a.body / f"ep{ep:06d}.npz")
        fids, pose_s = b["frame_ids"], b["pose"].copy()
        if a.smooth > 0:
            from scipy.ndimage import gaussian_filter1d
            # per stretch: the metric never differences across a gap, and smoothing across one
            # would drag the two sides of a 360-frame gap together
            for sl in stretches(fids):
                if sl.stop - sl.start >= 3:
                    pose_s[sl] = _smooth_wrap_safe(pose_s[sl], a.smooth)
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
