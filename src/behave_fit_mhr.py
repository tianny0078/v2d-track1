#!/usr/bin/env python3
"""Fit MHR parameters (pose per frame, scales+shape per sequence) to BEHAVE SMPL-H GT meshes.

Loss: vertex correspondence from nvidia/CARI4D `mhr2smplh_mapping.npz` -- every SMPL-H vertex maps to a
point on the MHR surface (triangle id + barycentrics; the mapping's `mhr_faces` equal our model's
faces, checked). The MHR TorchScript is differentiable, so we optimise the submission parameters
directly (src/fit_mhr.py pattern) in the submission frame (metres, diag(1,-1,-1) of native).

Stages: (1) rigid multi-start (24 octahedral root rotations) on keyframes, root pose only;
(2) keyframes: full pose + shared identity; (3) all frames initialised from the nearest keyframe,
then pose + identity refined jointly, chunked over frames.  Output: work/behave_gt/<seq>/episode_000000.npz
in our export format (object arrays copied from object_pose.npz) + fit_report.json with the residuals.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.export_episodes import _mat_to_euler  # noqa: E402  host euler-xyz convention
from src.fit_mhr import POSE_UNIT_M, POSE_SIGN  # noqa: E402

FLIP = torch.tensor([1.0, -1.0, -1.0])
MAPPING = ROOT / "work/behave_gt/_assets/mhr2smplh_mapping.npz"


def octahedral() -> np.ndarray:
    """The 24 rotation matrices of the cube group, as root-rotation starts."""
    from itertools import permutations, product
    out = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            R = np.zeros((3, 3)); R[np.arange(3), perm] = signs
            if np.linalg.det(R) > 0:
                out.append(R)
    return np.stack(out)


class Fit:
    def __init__(self, device):
        self.dev = torch.device(device)
        self.m = torch.jit.load(str(ROOT / "work/models/mhr_model.pt"), map_location=self.dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.flip = FLIP.to(self.dev)
        z = np.load(MAPPING)
        faces = torch.as_tensor(z["mhr_faces"], device=self.dev)
        self.corner = faces[torch.as_tensor(z["triangle_ids"], device=self.dev)]        # [6890,3]
        self.bary = torch.as_tensor(z["baryc_coords"], dtype=torch.float32, device=self.dev)  # [6890,3]
        self.faces_np = z["mhr_faces"]

    def verts(self, pose, scales, shape):
        T = pose.shape[0]
        params = torch.cat([pose, scales.expand(T, 68)], 1)
        v, _ = self.m(shape.expand(T, 45), params, torch.zeros(T, 72, device=self.dev))
        return v / 100.0 * self.flip

    def corr(self, v):
        return (v[:, self.corner] * self.bary[None, :, :, None]).sum(2)                  # [T,6890,3]

    def residual(self, pose, scales, shape, target):
        return (self.corr(self.verts(pose, scales, shape)) - target).norm(dim=-1)         # [T,6890] m

    def optimise(self, pose, scales, shape, target, iters, lr, lr_id, cols=None, w_id=1e-3, w_pose=1e-5, log=None):
        """Adam on the given (possibly chunked) tensors. cols: slice of pose columns to optimise."""
        pose = pose.clone().requires_grad_(True)
        groups = [{"params": [pose], "lr": lr}]
        if lr_id > 0:
            scales = scales.clone().requires_grad_(True); shape = shape.clone().requires_grad_(True)
            groups.append({"params": [scales, shape], "lr": lr_id})
        opt = torch.optim.Adam(groups)
        mask = torch.zeros(136, device=self.dev); mask[cols if cols is not None else slice(None)] = 1
        for it in range(iters):
            opt.zero_grad()
            r = self.residual(pose, scales, shape, target)
            loss = (r ** 2).mean() * 1e4 + w_id * ((scales ** 2).sum() + (shape ** 2).sum()) + w_pose * (pose[:, 6:] ** 2).sum()
            loss.backward()
            pose.grad *= mask
            opt.step()
            if log and (it % 100 == 0 or it == iters - 1):
                log(f"    it {it:4d} mean cm {r.mean().item() * 100:.3f}")
        return pose.detach(), scales.detach(), shape.detach(), r.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--key-every", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=90)
    ap.add_argument("--iters", type=int, nargs=3, default=(120, 400, 300), help="rigid, keyframe, all-frame iterations")
    a = ap.parse_args()
    t0 = time.time()
    gt = ROOT / f"work/behave_gt/{a.seq}"
    g = np.load(gt / "gt_smplh.npz")
    target_all = torch.as_tensor(g["verts"], device=a.device)                           # [T,6890,3]
    T = len(target_all)
    f = Fit(a.device)
    log = lambda s: print(s, flush=True)

    # ---- stage 1: rigid multi-start on keyframes ------------------------------------------
    keys = np.arange(0, T, a.key_every)
    R24 = octahedral()
    eul = torch.as_tensor(_mat_to_euler(R24), dtype=torch.float32, device=a.device)      # [24,3]
    K, S = len(keys), len(R24)
    pose = torch.zeros(K * S, 136, device=a.device)
    pose[:, 3:6] = eul.repeat(K, 1)
    scales, shape = torch.zeros(68, device=a.device), torch.zeros(45, device=a.device)
    tgt = target_all[keys].repeat_interleave(S, 0)
    with torch.no_grad():                                      # translation: match centroids
        c = f.corr(f.verts(pose, scales, shape)).mean(1)
        delta = (tgt.mean(1) - c) / POSE_UNIT_M * torch.tensor(POSE_SIGN, device=a.device)
        pose[:, :3] = delta
    log(f"[{a.seq}] stage 1: {K} keyframes x {S} starts")
    pose, _, _, r = f.optimise(pose, scales, shape, tgt, a.iters[0], 0.05, 0.0, cols=slice(0, 6), log=log)
    best = r.mean(1).view(K, S).argmin(1)
    kpose = pose.view(K, S, 136)[torch.arange(K), best]
    log(f"  rigid residual per keyframe (cm): {np.round(r.mean(1).view(K, S).min(1).values.cpu().numpy() * 100, 2).tolist()}")

    # ---- stage 2: keyframes, full pose + identity ------------------------------------------
    log(f"[{a.seq}] stage 2: keyframes full fit")
    kpose, scales, shape, r = f.optimise(kpose, scales, shape, target_all[keys], a.iters[1], 0.02, 5e-3, log=log)
    log(f"  keyframe residual {r.mean().item() * 100:.3f} cm, |shape| {shape.norm():.2f} |scales| {scales.norm():.2f}")

    # ---- stage 3: all frames from the nearest keyframe, chunked joint refinement -----------
    near = np.abs(np.arange(T)[:, None] - keys[None]).argmin(1)
    pose = kpose[torch.as_tensor(near, device=a.device)].clone()
    log(f"[{a.seq}] stage 3: {T} frames, chunk {a.chunk}")
    chunks = [slice(s, min(s + a.chunk, T)) for s in range(0, T, a.chunk)]
    pose_c = [pose[c].clone().requires_grad_(True) for c in chunks]
    scales = scales.clone().requires_grad_(True); shape = shape.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": pose_c, "lr": 0.02}, {"params": [scales, shape], "lr": 2e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters[2], eta_min=1e-3)
    for ep in range(a.iters[2]):
        tot = 0.0
        for p, c in zip(pose_c, chunks):
            opt.zero_grad()
            r = f.residual(p, scales, shape, target_all[c])
            loss = (r ** 2).mean() * 1e4 + 1e-3 * ((scales ** 2).sum() + (shape ** 2).sum()) + 1e-5 * (p[:, 6:] ** 2).sum()
            loss.backward(); opt.step(); tot += r.mean().item() * len(r)
        sched.step()
        if ep % 50 == 0 or ep == a.iters[2] - 1:
            log(f"    epoch {ep:4d} mean cm {tot / T * 100:.3f}")
    pose = torch.cat([p.detach() for p in pose_c])
    scales, shape = scales.detach(), shape.detach()

    # ---- residuals: correspondence distance and vertex-to-surface (both ways) ---------------
    with torch.no_grad():
        v = torch.cat([f.verts(pose[c], scales, shape) for c in chunks])
        r = (f.corr(v) - target_all).norm(dim=-1)                                          # [T,6890]
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.loss.point_mesh_distance import point_face_distance
    faces = torch.as_tensor(f.faces_np, device=a.device)
    sfaces = torch.as_tensor(np.load(ROOT / "work/behave_gt/_assets/smplh_male_v12.npz")["f"].astype(np.int64), device=a.device)
    sub = np.arange(0, T, 10)
    p2s, s2p = [], []
    for i in sub:
        mesh = Meshes([v[i]], [faces]); pc = Pointclouds([target_all[i]])
        d = point_face_distance(pc.points_packed(), pc.cloud_to_packed_first_idx(), mesh.verts_packed()[mesh.faces_packed()],
                                mesh.mesh_to_faces_packed_first_idx(), pc.num_points_per_cloud().max().item())
        p2s.append(d.sqrt().mean().item())
        mesh = Meshes([target_all[i]], [sfaces]); pc = Pointclouds([v[i]])
        d = point_face_distance(pc.points_packed(), pc.cloud_to_packed_first_idx(), mesh.verts_packed()[mesh.faces_packed()],
                                mesh.mesh_to_faces_packed_first_idx(), pc.num_points_per_cloud().max().item())
        s2p.append(d.sqrt().mean().item())
    per_frame = r.mean(1).cpu().numpy() * 100
    report = dict(seq=a.seq, frames=int(T), corr_cm_mean=float(per_frame.mean()), corr_cm_max_frame=float(per_frame.max()),
                  corr_cm_p95_vertex=float(np.percentile(r.cpu().numpy(), 95) * 100),
                  smplh_vert_to_mhr_surface_cm=float(np.mean(p2s) * 100), mhr_vert_to_smplh_surface_cm=float(np.mean(s2p) * 100),
                  shape_norm=float(shape.norm()), scales_norm=float(scales.norm()), seconds=round(time.time() - t0, 1))
    obj = np.load(gt / "object_pose.npz")
    np.savez(gt / "episode_000000.npz", pose=pose.cpu().numpy().astype(np.float32), scales=scales.cpu().numpy().astype(np.float32),
             shape=shape.cpu().numpy().astype(np.float32), object_rotation=obj["rotation"], object_translation=obj["translation"],
             object_scale=np.float32(obj["scale"]), fit_residual_cm=per_frame.astype(np.float32))
    (gt / "fit_report.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report))


if __name__ == "__main__":
    main()
