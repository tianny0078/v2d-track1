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
        return pts / 100.0 * self.flip

    def project(self, pts, fx, fy, cx, cy):
        z = pts[..., 2].clamp(min=0.3)
        return torch.stack([fx * pts[..., 0] / z + cx, fy * pts[..., 1] / z + cy], -1)

    def fit(self, kp: np.ndarray, K: dict, iters=(250, 400), lr=(0.05, 0.02),
            w_smooth=2.0, w_reg=1e-3, verbose=True) -> dict:
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
            opt = torch.optim.Adam(params, lr=lr_s)
            for it in range(n):
                opt.zero_grad(set_to_none=True)
                j3 = self.forward(pose, scales, shape)
                p2 = self.project(j3, fx, fy, cx, cy)
                res = (p2[:, sel] - k2d[:, sel]) / 100.0        # ~image-scale normalisation
                data = (gm_loss(res, 0.6) * conf[:, sel]).sum() / conf[:, sel].sum().clamp(min=1)
                loss = data
                if stage > 0:
                    vel = pose[1:] - pose[:-1]
                    acc = pose[2:] - 2 * pose[1:-1] + pose[:-2]
                    loss = loss + w_smooth * (acc.pow(2).mean() + 0.1 * vel.pow(2).mean())
                    loss = loss + w_reg * (shape.pow(2).sum() + scales.pow(2).sum()
                                           + pose[:, 6:].pow(2).mean())
                loss.backward()
                if stage == 0:
                    pose.grad.mul_(mask)
                opt.step()
                if verbose and it % max(1, n // 3) == 0:
                    hist.append((stage, it, float(data), float(loss)))
        with torch.no_grad():
            j3 = self.forward(pose, scales, shape)
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
    a = ap.parse_args()
    dev = a.device or pick_gpu(16000)
    K_all = json.load((ROOT / "work/camera/intrinsics.json").open())
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    out_dir, state = ROOT / "work/body", ROOT / "state/body"
    out_dir.mkdir(parents=True, exist_ok=True); state.mkdir(parents=True, exist_ok=True)

    f = Fitter(dev)
    for ep in (a.eps if a.eps else range(30)):
        z = np.load(ROOT / f"work/kp2d/ep{ep:06d}.npz")
        t0 = time.time()
        r = f.fit(z["kp"], K_all[meta[ep]["camera"]])
        np.savez_compressed(out_dir / f"ep{ep:06d}.npz", pose=r["pose"], scales=r["scales"],
                            shape=r["shape"], frame_ids=z["frame_ids"])
        rec = dict(ep=ep, frames=len(z["kp"]), px_err=round(r["px_err"], 2),
                   mean_depth_m=round(r["mean_depth_m"], 2), seconds=round(time.time() - t0, 1))
        (state / f"ep{ep:06d}.done").write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
