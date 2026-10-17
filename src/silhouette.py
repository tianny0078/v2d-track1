#!/usr/bin/env python3
"""Differentiable silhouette term for the MHR fit.

2D keypoints pin the body in the image but barely constrain limb DEPTH -- the fit reprojects at
0.85 px and still loses to a static rest pose on CD-H. A silhouette does constrain depth: swinging
an arm towards or away from the camera changes the outline even when the joint projects to the
same pixel. We already have person masks for all 30 episodes.

Two things make this affordable:
  - the crop window comes from the TARGET mask's bounding box, so it is fixed per frame and the
    resized target can be precomputed once;
  - the loss is evaluated on a random subset of frames each iteration, which bounds memory and
    time while still touching every frame over a few hundred iterations.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
S = 160
PAD = 1.35
#: pytorch3d's screen space has +x left and +y up; our submission frame has +x right, +y down.
P3D_FLIP = (-1.0, -1.0, 1.0)


class SilhouetteTerm:
    def __init__(self, ep: int, frame_ids: np.ndarray, K: dict, faces: np.ndarray, device):
        from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings, MeshRasterizer,
                                        MeshRenderer, SoftSilhouetteShader, BlendParams)
        self.dev = torch.device(device)
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz")
        H, W = z["shape"]
        idx = np.searchsorted(z["frame_ids"], frame_ids)
        packed = z["packed"][idx]
        masks = np.unpackbits(packed, axis=-1)[:, 1, :, :W].astype(bool)

        boxes, tgt = [], np.zeros((len(masks), S, S), np.float32)
        for t, m in enumerate(masks):
            ys, xs = np.where(m)
            if len(xs) < 50:
                boxes.append((0.0, 0.0, 1.0)); continue
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            half = max(xs.max() - xs.min(), ys.max() - ys.min()) * PAD / 2
            boxes.append((cx - half, cy - half, S / (2 * half)))
            x0, y0 = int(round(cx - half)), int(round(cy - half))
            sub = m[max(0, y0):y0 + int(2 * half), max(0, x0):x0 + int(2 * half)]
            if sub.size:
                from PIL import Image
                tgt[t] = np.asarray(Image.fromarray(sub.astype(np.uint8) * 255)
                                    .resize((S, S), Image.BILINEAR), np.float32) / 255.0
        b = np.array(boxes, np.float32)
        self.valid = torch.tensor(b[:, 2] > 0, device=self.dev)
        self.target = torch.tensor(tgt, device=self.dev)
        self.focal = torch.tensor(np.stack([K["fx"] * b[:, 2], K["fy"] * b[:, 2]], 1),
                                  dtype=torch.float32, device=self.dev)
        self.pp = torch.tensor(np.stack([(K["cx"] - b[:, 0]) * b[:, 2],
                                         (K["cy"] - b[:, 1]) * b[:, 2]], 1),
                               dtype=torch.float32, device=self.dev)
        self.faces = torch.tensor(faces, dtype=torch.int64, device=self.dev)
        self.flip = torch.tensor(P3D_FLIP, device=self.dev)
        self._Cam = PerspectiveCameras
        self.raster = RasterizationSettings(image_size=S, blur_radius=1e-4,
                                            faces_per_pixel=12, bin_size=0)
        self._Renderer, self._Raster = MeshRenderer, MeshRasterizer
        self._Shader, self._Blend = SoftSilhouetteShader, BlendParams

    def render(self, verts: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        from pytorch3d.structures import Meshes
        n = len(sel)
        cam = self._Cam(focal_length=self.focal[sel], principal_point=self.pp[sel],
                        image_size=((S, S),) * n, in_ndc=False, device=self.dev)
        ren = self._Renderer(self._Raster(cameras=cam, raster_settings=self.raster),
                             self._Shader(blend_params=self._Blend(sigma=1e-5, gamma=1e-5)))
        mesh = Meshes(verts=verts * self.flip, faces=self.faces[None].expand(n, -1, -1))
        return ren(mesh)[..., 3]

    def loss(self, verts: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        """Soft IoU against the person mask. Lower is better."""
        pred = self.render(verts, sel)
        tgt = self.target[sel]
        inter = (pred * tgt).sum((1, 2))
        union = (pred + tgt - pred * tgt).sum((1, 2)).clamp(min=1.0)
        return (1.0 - inter / union)[self.valid[sel]].mean()
