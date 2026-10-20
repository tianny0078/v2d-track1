#!/usr/bin/env python3
"""Minimal SMPL-H (52 joints, 10 betas, full axis-angle hands) forward in torch.

Transcribed from BEHAVE's libsmpl/smplpytorch SMPL_Layer(hands=True): axis-angle for all 52
joints, pose blendshapes on the 51 non-root joints (posedirs 6890x3x459, 'lrotmin'), NO hand
mean added, trans added after skinning. The model npz is the plain SMPL-H release layout
(v_template, shapedirs, posedirs, J_regressor, weights, kintree_table, f).
"""
from __future__ import annotations
import numpy as np
import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MODELS = {g: ROOT / f"work/models/smplh/camd/body_models/smplh/SMPLH_{g.upper()}.npz" for g in ("male", "female", "neutral")}


def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """[...,3] axis-angle -> [...,3,3]."""
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    K = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=r.dtype, device=r.device) + th.sin() * K + (1 - th.cos()) * (K @ K)


class SMPLH:
    def __init__(self, gender: str = "male", device: str = "cpu", path=None, dtype=torch.float32):
        z = np.load(path or MODELS[gender], allow_pickle=True)
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), dtype=dtype, device=device)
        self.dev, self.dtype = torch.device(device), dtype
        self.v_template = t(z["v_template"])                       # [6890,3]
        self.shapedirs = t(z["shapedirs"][:, :, :10])               # [6890,3,10]
        self.posedirs = t(z["posedirs"]).reshape(-1, 459)           # [6890*3,459]
        J = z["J_regressor"]
        self.J_regressor = t(J.toarray() if hasattr(J, "toarray") else J)  # [52,6890]
        self.weights = t(z["weights"])                              # [6890,52]
        self.parents = np.asarray(z["kintree_table"][0], np.int64); self.parents[0] = -1
        self.faces = np.asarray(z["f"], np.int64)

    def __call__(self, poses, betas, trans):
        """poses [T,156] axis-angle, betas [T,10], trans [T,3] -> verts [T,6890,3], joints [T,52,3] (m)."""
        poses = torch.as_tensor(poses, dtype=self.dtype, device=self.dev)
        betas = torch.as_tensor(betas, dtype=self.dtype, device=self.dev)
        trans = torch.as_tensor(trans, dtype=self.dtype, device=self.dev)
        T = poses.shape[0]
        v_shaped = self.v_template + torch.einsum("vcb,tb->tvc", self.shapedirs, betas)
        J = torch.einsum("jv,tvc->tjc", self.J_regressor, v_shaped)
        R = rodrigues(poses.reshape(T, 52, 3))                      # [T,52,3,3]
        pose_feat = (R[:, 1:] - torch.eye(3, dtype=self.dtype, device=self.dev)).reshape(T, 459)
        v_posed = v_shaped + (pose_feat @ self.posedirs.T).reshape(T, 6890, 3)
        G = [None] * 52
        for j in range(52):
            p = self.parents[j]
            loc = torch.eye(4, dtype=self.dtype, device=self.dev).repeat(T, 1, 1)
            loc[:, :3, :3] = R[:, j]
            loc[:, :3, 3] = J[:, j] - (J[:, p] if p >= 0 else 0)
            G[j] = loc if p < 0 else G[p] @ loc
        G = torch.stack(G, 1)                                       # [T,52,4,4]
        joints = G[:, :, :3, 3]
        Jh = torch.cat([J, torch.zeros(T, 52, 1, dtype=self.dtype, device=self.dev)], -1)
        G_rel = G.clone(); G_rel[:, :, :3, 3] -= (G @ Jh[..., None])[:, :, :3, 0]
        A = torch.einsum("vj,tjab->tvab", self.weights, G_rel)
        vh = torch.cat([v_posed, torch.ones(T, 6890, 1, dtype=self.dtype, device=self.dev)], -1)
        verts = (A @ vh[..., None])[:, :, :3, 0]
        return verts + trans[:, None], joints + trans[:, None]


if __name__ == "__main__":
    # cross-check against smplx (needs dummy hand-PCA keys for its loader)
    import tempfile, os, smplx
    z = np.load(MODELS["male"], allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d.update(hands_componentsl=np.zeros((45, 45)), hands_componentsr=np.zeros((45, 45)),
             hands_meanl=np.zeros(45), hands_meanr=np.zeros(45))
    tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/smplh")
    np.savez(f"{tmp}/smplh/SMPLH_MALE.npz", **d)
    ref = smplx.SMPLH(f"{tmp}/smplh", gender="male", use_pca=False, flat_hand_mean=True, num_betas=10, ext="npz")
    rng = np.random.default_rng(0)
    T = 4
    poses = rng.normal(0, 0.4, (T, 156)).astype(np.float32); betas = rng.normal(0, 1, (T, 10)).astype(np.float32)
    trans = rng.normal(0, 1, (T, 3)).astype(np.float32)
    o = ref(betas=torch.tensor(betas), global_orient=torch.tensor(poses[:, :3]), body_pose=torch.tensor(poses[:, 3:66]),
            left_hand_pose=torch.tensor(poses[:, 66:111]), right_hand_pose=torch.tensor(poses[:, 111:156]),
            transl=torch.tensor(trans))
    v, j = SMPLH("male")(poses, betas, trans)
    print("max |ours - smplx| verts (mm):", float((v - o.vertices).abs().max() * 1000),
          "joints[:52] (mm):", float((j - o.joints[:, :52]).abs().max() * 1000))
