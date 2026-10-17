#!/usr/bin/env python3
"""Fit MHR parameters to 2D keypoints by gradient descent, directly in MHR parameter space.

Why not convert from another body model: the official converter (tools/track1/MESH_TO_MHR_PARAMS.md)
takes MHR-TOPOLOGY vertices, so any non-MHR body model needs a surface correspondence step whose
error lands on top of the reconstruction error. The released MHR TorchScript turns out to be fully
differentiable (gradients reach 201/204 pose dims), so we optimise the submission parameters
themselves and that conversion error is identically zero.

Frames: MHR's native output is centimetres with y and z flipped relative to the submission frame.
We work in the SUBMISSION frame (metres, diag(1,-1,-1)) where y is down and z is forward, which is
an ordinary pinhole camera frame, so projection is u = fx*x/z + cx.
`pose[0:3]` stays in MHR native units because that is what the submission wants.

Scale: the scorer fits one Sim(3) from the frame-0 body and applies it to everything, so absolute
scale is free. What is NOT free is relative depth, which the focal length controls -- hence the
per-camera intrinsics from MoGe.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FLIP = torch.tensor([1.0, -1.0, -1.0])

#: COCO-17 index -> MHR joint index. Verified geometrically on the rest pose (nose highest,
#: ankles lowest, anatomical left has x>0). Eyes and ears are left out: MHR has no matching
#: joints and a guessed correspondence would pull the head pose the wrong way.
#: The 22 joints ACC-H scores, in the metric's order. Smoothing is applied to THESE positions,
#: not to the 136 pose parameters: penalising the parameters averages over 136 dims, so a few
#: jittering dims get diluted and the penalty saturates (measured: w_smooth 2 -> 2000 only moved
#: |a_pred| from 2.29 to 0.71 while px_err blew up to 12). The metric measures joint positions,
#: so that is what the regulariser has to see.
ACC_JOINTS = (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)

#: Ankles and toe balls -- the joints that touch the floor when standing or walking.
FOOT_JOINTS = (4, 20, 8, 24)
#: Pelvis and neck: the segment between them is the body's up axis, used to keep the torso
#: upright relative to the measured floor normal.
PELVIS_J, NECK_J = 1, 110

COCO_TO_MHR = {5: 75, 6: 39, 7: 76, 8: 40, 9: 78, 10: 42,
               11: 2, 12: 18, 13: 3, 14: 19, 15: 4, 16: 20}
#: COCO's nose has no MHR JOINT counterpart -- joint 113 (c_head) is inside the skull, so fitting
#: it to the nose drags the whole head forward (measured: head centre landed 0.9 px from the nose,
#: where a real head centre sits 8-12 px behind it). Dropping the nose instead leaves the head
#: unconstrained and it drifts 49 px. So we target the MESH's own nose: vertex 2767, the -z
#: extreme of the head's skinned vertices, 12 cm in front of and 5.4 cm above the head joint and
#: 13.6 cm below the crown -- anatomically a nose. -z is the facing direction (the resting wrists
#: are at -z too).
NOSE_VERTEX = 2767
NOSE_COCO = 0
TORSO_COCO = [5, 6, 11, 12]
BODY_HEIGHT_M = 1.727          # MHR rest-pose total height, measured
#: Measured, not assumed: pose[0:3] += 100 moves the pelvis by exactly 10 m, with pose[0] -> +x,
#: pose[1] -> -y and pose[2] -> -z in the submission frame. So one pose unit is 0.1 m and the y/z
#: signs carry the diag(1,-1,-1) flip. Guessing "MHR units are centimetres" puts the body 10x too
#: far away, which is exactly the 52 m mean depth the first run produced.
POSE_UNIT_M = 0.1
POSE_SIGN = (1.0, -1.0, -1.0)
#: Pelvis position in the submission frame at zero pose, and the nose-to-ankle span, both measured.
PELVIS_REST = (0.0, -0.924, 0.0)
NOSE_ANKLE_SPAN_M = 1.463


def gm_loss(residual: torch.Tensor, sigma: float) -> torch.Tensor:
    """Geman-McClure: bounded, so a few badly-detected joints cannot dominate the fit."""
    r2 = residual.pow(2).sum(-1)
    return r2 / (r2 + sigma ** 2)


class Fitter:
    def __init__(self, device: str):
        self.dev = torch.device(device)
        self.m = torch.jit.load(str(ROOT / "work/models/mhr_model.pt"), map_location=self.dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.flip = FLIP.to(self.dev)
        self.coco = sorted(COCO_TO_MHR) + [NOSE_COCO]      # nose last, matching forward()
        self.mhr_idx = [COCO_TO_MHR[c] for c in sorted(COCO_TO_MHR)]

    def forward(self, pose, scales, shape):
        """pose [T,136] -> the fitted 2D targets' 3D points [T,K+1,3] in the submission frame.

        The nose comes from a mesh vertex, the rest from joints; the model computes vertices
        anyway, so carrying one of them costs nothing.
        """
        T = pose.shape[0]
        params = torch.cat([pose, scales.expand(T, 68)], 1)
        v, j = self.m(shape.expand(T, 45), params, torch.zeros(T, 72, device=self.dev))
        pts = torch.cat([j[:, self.mhr_idx, :3], v[:, NOSE_VERTEX:NOSE_VERTEX + 1]], 1)
        jm = j[..., :3] / 100.0 * self.flip
        return pts / 100.0 * self.flip, jm[:, list(ACC_JOINTS)], jm

    def project(self, pts, fx, fy, cx, cy):
        z = pts[..., 2].clamp(min=0.3)
        return torch.stack([fx * pts[..., 0] / z + cx, fy * pts[..., 1] / z + cy], -1)

    def fit(self, kp: np.ndarray, K: dict, iters=(250, 400), lr=(0.05, 0.02),
            w_smooth=2.0, w_reg=1e-3, w_jacc=0.0, stretch_slices=None,
            ground=None, w_ground=0.0, w_upright=0.0, sil=None, w_sil=0.0, sil_batch=16,
            lr_identity=1e-3, identity_clamp=2.0, verbose=True) -> dict:
        """lr_identity / identity_clamp: shape and scales get their own, much smaller, Adam step.
        Measured on the silhouette term alone: at lr 0.02 shape marched to |shape| 13 and the
        loss being optimised got WORSE; at pose lr 5e-3 / identity lr 1e-3 it improved
        (dt_loss 0.097 -> 0.066). Under a keypoint-only loss shape had ~zero gradient so this
        never showed; the silhouette is the first term that actually moves it."""
        """w_jacc penalises the scored joints' own second difference, in cm -- the quantity
        ACC-H measures. Target: the reference's own |a_gt| = 0.1453 cm/frame^2, which a fully
        static prediction already achieves, so there is never a reason to land above it."""
        T = len(kp)
        k2d = torch.tensor(kp[:, self.coco, :2], dtype=torch.float32, device=self.dev)
        conf = torch.tensor(kp[:, self.coco, 2], dtype=torch.float32, device=self.dev).clamp(0, 1)
        conf = torch.nan_to_num(conf) * (~torch.isnan(k2d).any(-1)).float()
        k2d = torch.nan_to_num(k2d)
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

        # --- initialisation: depth from apparent height, centre from the keypoint centroid
        vis = conf > 0.2
        big, small = torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max
        ys_hi = torch.where(vis, k2d[..., 1], torch.full_like(k2d[..., 1], small)).max(1).values
        ys_lo = torch.where(vis, k2d[..., 1], torch.full_like(k2d[..., 1], big)).min(1).values
        n_vis = vis.sum(1).clamp(min=1)
        xs_mean = (torch.where(vis, k2d[..., 0], torch.zeros_like(k2d[..., 0])).sum(1) / n_vis)
        ys_mean = (torch.where(vis, k2d[..., 1], torch.zeros_like(k2d[..., 1])).sum(1) / n_vis)
        h_px = (ys_hi - ys_lo).clamp(min=20)
        # keypoints span nose..ankle, about 0.93 of full stature
        z0 = (fy * NOSE_ANKLE_SPAN_M / h_px).clamp(1.0, 15.0)
        # pelvis image position: hip midpoint when both hips are seen, else the keypoint centroid
        hip = [self.coco.index(c) for c in (11, 12)]
        hip_ok = (conf[:, hip] > 0.2).all(1)
        px = torch.where(hip_ok, k2d[:, hip, 0].mean(1), torch.nan_to_num(xs_mean, nan=cx))
        py = torch.where(hip_ok, k2d[:, hip, 1].mean(1), torch.nan_to_num(ys_mean, nan=cy))
        X = (px - cx) / fx * z0
        Y = (py - cy) / fy * z0

        # invert the measured relation: joint = PELVIS_REST + sign * POSE_UNIT_M * pose[0:3]
        pose = torch.zeros(T, 136, device=self.dev)
        pose[:, 0] = (X - PELVIS_REST[0]) / (POSE_SIGN[0] * POSE_UNIT_M)
        pose[:, 1] = (Y - PELVIS_REST[1]) / (POSE_SIGN[1] * POSE_UNIT_M)
        pose[:, 2] = (z0 - PELVIS_REST[2]) / (POSE_SIGN[2] * POSE_UNIT_M)
        pose.requires_grad_(True)
        scales = torch.zeros(68, device=self.dev, requires_grad=True)
        shape = torch.zeros(45, device=self.dev, requires_grad=True)

        hist = []
        for stage, (n, lr_s) in enumerate(zip(iters, lr)):
            if stage == 0:   # global placement only: root translation and root rotation
                params = [pose]
                mask = torch.zeros(136, device=self.dev); mask[:6] = 1.0
                sel = [c for c in range(len(self.coco)) if self.coco[c] in TORSO_COCO]
            else:
                params = [pose, scales, shape]
                mask = torch.ones(136, device=self.dev)
                sel = list(range(len(self.coco)))
            if stage == 0:
                opt = torch.optim.Adam(params, lr=lr_s)
            else:
                opt = torch.optim.Adam([{"params": [pose], "lr": lr_s},
                                        {"params": [scales, shape], "lr": lr_identity}])
            for it in range(n):
                opt.zero_grad(set_to_none=True)
                j3, acc_j, jm = self.forward(pose, scales, shape)
                p2 = self.project(j3, fx, fy, cx, cy)
                res = (p2[:, sel] - k2d[:, sel]) / 100.0        # ~image-scale normalisation
                data = (gm_loss(res, 0.6) * conf[:, sel]).sum() / conf[:, sel].sum().clamp(min=1)
                loss = data
                if stage > 0:
                    vel = pose[1:] - pose[:-1]
                    acc = pose[2:] - 2 * pose[1:-1] + pose[:-2]
                    loss = loss + w_smooth * (acc.pow(2).mean() + 0.1 * vel.pow(2).mean())
                    if w_jacc > 0:
                        for sl in (stretch_slices or [slice(0, T)]):
                            if sl.stop - sl.start < 3:
                                continue
                            seg = acc_j[sl] * 100.0          # cm, exactly the metric's units
                            ja = seg[:-2] - 2 * seg[1:-1] + seg[2:]
                            loss = loss + w_jacc * ja.norm(dim=-1).mean()
                    loss = loss + w_reg * (shape.pow(2).sum() + scales.pow(2).sum()
                                           + pose[:, 6:].pow(2).mean())
                if ground is not None and (w_ground > 0 or w_upright > 0):
                    gn = torch.tensor(ground["normal"], dtype=torch.float32, device=self.dev)
                    # SCALE-FREE form. Pinning the feet to the floor's measured DISTANCE mixes two
                    # independent metric scales -- the body's (set by the assumed 1.727 m stature)
                    # and MoGe's -- and injects a per-frame inconsistency that the scorer's single
                    # frame-0 Sim(3) cannot absorb. Measured: that version moved CD-H from 82.68
                    # to 96.92. What IS scale-free, and what the floor legitimately tells us, is
                    # that the lowest foot stays at the SAME height all episode: the subject walks
                    # on a plane, they do not sink into it or float above it. So penalise the
                    # VARIANCE of the lowest foot's height and leave its value free.
                    h = jm[:, list(FOOT_JOINTS)] @ gn           # offset dropped on purpose
                    low = h.min(1).values
                    for sl in (stretch_slices or [slice(0, T)]):
                        if sl.stop - sl.start < 3:
                            continue
                        seg = low[sl]
                        loss = loss + w_ground * (seg - seg.mean()).pow(2).mean()
                    if w_upright > 0:
                        # pelvis -> neck points the same way as the floor normal when upright:
                        # verified on the rest pose, up . gn = +0.9999. The first version wrote
                        # (1 + up.gn)^2, which is minimised at up.gn = -1 -- it was REWARDING an
                        # upside-down body, and it was active in the submission that scored
                        # CD-H 96.92.
                        up = jm[:, NECK_J] - jm[:, PELVIS_J]
                        up = up / up.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        loss = loss + w_upright * (1.0 - (up * gn).sum(-1)).pow(2).mean()
                if sil is not None and w_sil > 0 and stage > 0:
                    # separate small forward for a random subset: keeping the full-episode vertex
                    # graph alive would cost ~180 MB per 800 frames plus autograd, and the subset
                    # still reaches every frame over a few hundred iterations
                    k = min(sil_batch, T)
                    idx = torch.randperm(T, device=self.dev)[:k]
                    pp = torch.cat([pose[idx], scales.expand(k, 68)], 1)
                    vv, _ = self.m(shape.expand(k, 45), pp, torch.zeros(k, 72, device=self.dev))
                    vv = vv / 100.0 * self.flip
                    loss = loss + w_sil * (sil.loss(vv, idx, K) if hasattr(sil, "dt")
                                           else sil.loss(vv, idx))
                loss.backward()
                if stage == 0:
                    pose.grad.mul_(mask)
                opt.step()
                if stage > 0 and identity_clamp:
                    with torch.no_grad():
                        shape.clamp_(-identity_clamp, identity_clamp)
                        scales.clamp_(-identity_clamp, identity_clamp)
                if verbose and it % max(1, n // 3) == 0:
                    hist.append((stage, it, float(data), float(loss)))
        with torch.no_grad():
            j3, _, jm_f = self.forward(pose, scales, shape)
            p2 = self.project(j3, fx, fy, cx, cy)
            err = ((p2 - k2d).norm(dim=-1) * conf).sum() / conf.sum().clamp(min=1)
        return dict(pose=pose.detach().cpu().numpy(), scales=scales.detach().cpu().numpy(),
                    shape=shape.detach().cpu().numpy(), px_err=float(err), hist=hist,
                    mean_depth_m=float(j3[..., 2].mean()))


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segment import pick_gpu
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path, default=ROOT / "work/body")
    ap.add_argument("--w-ground", type=float, default=1.0,
                    help="1.0 is the measured elbow: without it the feet sit 17.2 cm BELOW the "
                         "floor and reprojection is 1.20 px; at 1.0 they are on it (0.55 cm) for "
                         "2.99 px; at 50 reprojection collapses to 11 px for no further gain.")
    ap.add_argument("--w-upright", type=float, default=0.1)
    a = ap.parse_args()
    dev = a.device or pick_gpu(16000)
    K_all = json.load((ROOT / "work/camera/intrinsics.json").open())
    G_all = json.load((ROOT / "work/camera/ground.json").open())
    sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
    from v2dlb.mhr_submission import stretches
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    out_dir, state = a.out, ROOT / "state/body"
    out_dir.mkdir(parents=True, exist_ok=True); state.mkdir(parents=True, exist_ok=True)

    f = Fitter(dev)
    for ep in (a.eps if a.eps else range(30)):
        z = np.load(ROOT / f"work/kp2d/ep{ep:06d}.npz")
        t0 = time.time()
        cam = meta[ep]["camera"]
        r = f.fit(z["kp"], K_all[cam], stretch_slices=list(stretches(z["frame_ids"])),
                  ground=G_all[cam], w_ground=a.w_ground, w_upright=a.w_upright)
        np.savez_compressed(out_dir / f"ep{ep:06d}.npz", pose=r["pose"], scales=r["scales"],
                            shape=r["shape"], frame_ids=z["frame_ids"])
        rec = dict(ep=ep, frames=len(z["kp"]), px_err=round(r["px_err"], 2),
                   mean_depth_m=round(r["mean_depth_m"], 2), w_ground=a.w_ground,
                   seconds=round(time.time() - t0, 1))
        (state / f"ep{ep:06d}.done").write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
