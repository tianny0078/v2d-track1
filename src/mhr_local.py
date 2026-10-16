#!/usr/bin/env python3
"""MHR forward + vertex roles, from the official TorchScript model alone.

The host scores with a NumPy re-implementation (metric_code/track_1/*.py::_MHRBody) driven by an
asset .npz that the participant copy omits. That asset carries four vertex index sets --
role_alignment, role_body, role_left_hand, role_right_hand -- so we cannot reproduce them exactly.

What we CAN do exactly: the forward pass, and the hand sets. Hand vertices are derived from the
model's own skinning weights: every vertex whose dominant joint is the wrist or a descendant of
it. A superset of the host's 512 hand points is SAFE for the penetration metric -- no penetration
on a superset implies none on the subset -- which is the property PEN optimisation needs.

Units: the host returns metres in the diag(1,-1,-1) frame. We match that exactly, so anything
written against the host's metric functions accepts our output unchanged.
"""
from __future__ import annotations
import functools
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "work/models/mhr_model.pt"
FLIP = torch.tensor([1.0, -1.0, -1.0])

#: From metric_code/track_1/*.py -- the 22 joints ACC-H scores, in its order.
BODY_JOINT_INDICES = (1, 2, 18, 35, 3, 19, 36, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)
L_WRIST, R_WRIST = 78, 42


class MHR:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.m = torch.jit.load(str(MODEL), map_location=self.device).eval()
        ct = self.m.character_torch
        self.parents = ct.skeleton.joint_parents.cpu().numpy().astype(np.int64)
        self.faces = ct.mesh.faces.cpu().numpy().astype(np.int64)
        self.rest = ct.mesh.rest_vertices.cpu().numpy()
        lbs = ct.linear_blend_skinning
        self._skin = (lbs.vert_indices_flattened.cpu().numpy(),
                      lbs.skin_indices_flattened.cpu().numpy().astype(np.int64),
                      lbs.skin_weights_flattened.cpu().numpy())
        self.flip = FLIP.to(self.device)

    # ---- forward -------------------------------------------------------------
    def forward(self, model_params: np.ndarray, shape: np.ndarray, batch: int = 512):
        """model_params [T,204], shape [45] -> vertices [T,18439,3] m, joints [T,127,3] m."""
        mp = torch.as_tensor(np.atleast_2d(model_params), dtype=torch.float32, device=self.device)
        sh = torch.as_tensor(np.asarray(shape).reshape(45), dtype=torch.float32, device=self.device)
        vs, js = [], []
        with torch.no_grad():
            for s in range(0, len(mp), batch):
                chunk = mp[s:s + batch]
                v, j = self.m(sh.expand(len(chunk), 45), chunk,
                              torch.zeros(len(chunk), 72, device=self.device))
                vs.append((v / 100.0 * self.flip).cpu())
                js.append((j[..., :3] / 100.0 * self.flip).cpu())
        return torch.cat(vs).numpy(), torch.cat(js).numpy()

    # ---- roles ---------------------------------------------------------------
    def descendants(self, root: int) -> np.ndarray:
        keep = {root}
        for j, p in enumerate(self.parents):
            # parents are topologically ordered in MHR, so one forward sweep suffices
            if p in keep:
                keep.add(j)
        return np.array(sorted(keep), dtype=np.int64)

    @functools.cached_property
    def dominant_joint(self) -> np.ndarray:
        """[18439] the joint carrying each vertex's largest skinning weight."""
        verts, joints, weights = self._skin
        best_w = np.zeros(len(self.rest), np.float32)
        best_j = np.zeros(len(self.rest), np.int64)
        order = np.argsort(weights)  # ascending, so the largest writes last
        for v, j, w in zip(verts[order], joints[order], weights[order]):
            best_w[v], best_j[v] = w, j
        return best_j

    @functools.cached_property
    def hand_vertices(self) -> np.ndarray:
        """Superset of the host's 512 hand points: vertices skinned to either wrist's subtree."""
        hands = np.concatenate([self.descendants(L_WRIST), self.descendants(R_WRIST)])
        return np.where(np.isin(self.dominant_joint, hands))[0]

    @functools.cached_property
    def body_proxy_vertices(self) -> np.ndarray:
        """64 non-hand vertices, farthest-point sampled on the rest mesh.

        A PROXY for the host's role_body, not a reproduction: CD-H computed with it is an
        ablation signal, never a predicted leaderboard score.
        """
        return self._fps(np.setdiff1d(np.arange(len(self.rest)), self.hand_vertices), 64)

    @functools.cached_property
    def alignment_vertices(self) -> np.ndarray:
        """256 whole-body points for the frame-0 Sim(3). More points than the host uses, which
        makes the fitted transform strictly more stable, not less."""
        return self._fps(np.arange(len(self.rest)), 256)

    def _fps(self, pool: np.ndarray, n: int) -> np.ndarray:
        pts = self.rest[pool]
        picked = [0]
        d = np.linalg.norm(pts - pts[0], axis=1)
        for _ in range(n - 1):
            i = int(d.argmax()); picked.append(i)
            d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
        return np.sort(pool[np.array(picked)])


if __name__ == "__main__":
    mhr = MHR()
    v, j = mhr.forward(np.zeros((4, 204), np.float32), np.zeros(45, np.float32))
    print("verts", v.shape, "joints", j.shape)
    print("height m", round(float(v[0, :, 1].max() - v[0, :, 1].min()), 4))
    print("faces", mhr.faces.shape)
    print("hand vertices", len(mhr.hand_vertices), "(host uses 512)")
    print("left subtree joints", len(mhr.descendants(L_WRIST)), "right", len(mhr.descendants(R_WRIST)))
    hv = v[0, mhr.hand_vertices]
    print("hand bbox extent m", np.round(hv.max(0) - hv.min(0), 3))
    print("body proxy", len(mhr.body_proxy_vertices), "alignment", len(mhr.alignment_vertices))
