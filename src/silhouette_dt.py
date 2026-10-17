#!/usr/bin/env python3
"""Rasteriser-free silhouette term: 2D Chamfer between projected vertices and the person mask.

The soft-IoU term (src/silhouette.py) failed in a way that does not look like tuning: starting
from a good fit and optimising ONLY that term drove IoU from 0.553 to 0.415, and finite
differences of its loss did not converge as the step shrank (cos(analytic, FD) jumped between
-0.84 and +0.73 across step sizes). The non-smoothness comes from the rasteriser. This term has no
rasteriser at all -- gradients pass only through the MHR forward and a pinhole projection, which
is exactly as smooth as the keypoint term that works.

Two pieces, both on the precomputed distance transform (DT) of the target mask:
  inside : vertices that project OUTSIDE the mask pay DT(uv)        -> pushes the body into the
                                                                        silhouette
  cover  : mask boundary points pay their distance to the nearest    -> pulls the body out to the
           projected vertex                                             silhouette edge (fixes
                                                                        "body too thin")
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
N_BOUNDARY = 512
N_VERTS = 3000


class SilhouetteDT:
    def __init__(self, ep: int, frame_ids: np.ndarray, device, seed: int = 0):
        from scipy.ndimage import distance_transform_edt, binary_erosion
        self.dev = torch.device(device)
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz")
        self.H, self.W = (int(x) for x in z["shape"])
        idx = np.searchsorted(z["frame_ids"], frame_ids)
        masks = np.unpackbits(z["packed"][idx], axis=-1)[:, 1, :, :self.W].astype(bool)

        T = len(masks)
        # DT in pixels, positive outside the mask, 0 inside. Stored at quarter resolution:
        # 800 x 1152 x 1536 floats would be 5.6 GB; 800 x 288 x 384 is 350 MB.
        self.ds = 4
        h, w = self.H // self.ds, self.W // self.ds
        dt = np.zeros((T, h, w), np.float32)
        bnd = np.zeros((T, N_BOUNDARY, 2), np.float32)
        self.valid = np.zeros(T, bool)
        rng = np.random.default_rng(seed)
        for t, m in enumerate(masks):
            if m.sum() < 50:
                continue
            self.valid[t] = True
            small = m[::self.ds, ::self.ds]
            dt[t] = distance_transform_edt(~small) * self.ds
            edge = m & ~binary_erosion(m, iterations=2)
            ys, xs = np.where(edge)
            pick = rng.choice(len(xs), N_BOUNDARY, replace=len(xs) < N_BOUNDARY)
            bnd[t] = np.stack([xs[pick], ys[pick]], 1)
        self.dt = torch.tensor(dt, device=self.dev)
        self.boundary = torch.tensor(bnd, device=self.dev)
        self.valid_t = torch.tensor(self.valid, device=self.dev)
        self.vsub = torch.tensor(rng.choice(18439, N_VERTS, replace=False), device=self.dev)

    def loss(self, verts: torch.Tensor, sel: torch.Tensor, K: dict,
             w_inside: float = 1.0, w_cover: float = 1.0) -> torch.Tensor:
        """verts [B,18439,3] in the submission frame (metres); sel [B] frame indices."""
        v = verts[:, self.vsub]
        zc = v[..., 2].clamp(min=0.3)
        u = K["fx"] * v[..., 0] / zc + K["cx"]
        w = K["fy"] * v[..., 1] / zc + K["cy"]
        B = len(sel)
        # --- inside: bilinear lookup of the DT at every projected vertex
        gx = (u / (self.W - 1)) * 2 - 1
        gy = (w / (self.H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], -1)[:, None]                      # [B,1,N,2]
        d = F.grid_sample(self.dt[sel][:, None], grid, align_corners=True,
                          padding_mode="border")[:, 0, 0]                # [B,N] px
        inside = d.mean(1)
        # --- cover: boundary point -> nearest projected vertex
        uv = torch.stack([u, w], -1)                                     # [B,N,2]
        bp = self.boundary[sel]                                          # [B,M,2]
        dist = torch.cdist(bp, uv)                                       # [B,M,N]
        cover = dist.min(2).values.mean(1)
        valid = self.valid_t[sel]
        per = (w_inside * inside + w_cover * cover) / 100.0               # ~image-scale units
        return per[valid].mean() if valid.any() else per.sum() * 0

    @torch.no_grad()
    def iou(self, verts: torch.Tensor, sel: torch.Tensor, K: dict) -> float:
        """Hard IoU of the projected-vertex convex-free coverage vs the mask, for reporting only."""
        from scipy.ndimage import binary_dilation
        v = verts
        zc = v[..., 2].clamp(min=0.3)
        u = (K["fx"] * v[..., 0] / zc + K["cx"]).round().long().clamp(0, self.W - 1)
        w = (K["fy"] * v[..., 1] / zc + K["cy"]).round().long().clamp(0, self.H - 1)
        ious = []
        for b in range(len(sel)):
            if not self.valid[int(sel[b])]:
                continue
            img = np.zeros((self.H, self.W), bool)
            img[w[b].cpu().numpy(), u[b].cpu().numpy()] = True
            img = binary_dilation(img, iterations=6)
            tgt = self.dt[sel[b]].cpu().numpy() == 0
            tgt_full = np.kron(tgt, np.ones((self.ds, self.ds), bool))[:self.H, :self.W]
            ious.append((img & tgt_full).sum() / max(1, (img | tgt_full).sum()))
        return float(np.mean(ious)) if ious else float("nan")
