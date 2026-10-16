#!/usr/bin/env python3
"""Estimate the floor plane per physical camera, in camera coordinates.

Why this and not more keypoints: 2D keypoints pin the body in the image but say almost nothing
about depth, which is why the keypoint-only fit reprojects at 0.81 px and still loses to a static
rest pose on CD-H (82.7 vs 70.6). The floor is the strongest depth cue available here -- the
subject stands and walks on it for the whole episode -- and once the plane is known, putting the
feet on it determines depth almost completely.

The plane is fitted by RANSAC on MoGe's point map over the lower part of the frame, excluding the
person and the target object. Cameras are static, so one plane per camera serves all its episodes,
and pooling frames makes the estimate far steadier than any single image.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FRAMES, MASKS = ROOT / "work/frames", ROOT / "work/masks"
OUT = ROOT / "work/camera"
N_PER_CAM = 8


def fit_plane_ransac(pts: np.ndarray, iters=600, thresh=0.03, rng=None):
    """Return (normal, offset, inlier_fraction) for n·p + d = 0, normal pointing up (-y)."""
    rng = rng or np.random.default_rng(0)
    best = (None, None, 0.0)
    n_pts = len(pts)
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            continue
        n = n / ln
        d = -n @ p0
        inl = np.abs(pts @ n + d) < thresh
        frac = inl.mean()
        if frac > best[2]:
            best = (n, d, frac)
    n, d, frac = best
    if n is None:
        return None, None, 0.0
    # refit on inliers, then orient the normal upward (y is down, so up is -y)
    inl = np.abs(pts @ n + d) < thresh
    q = pts[inl] - pts[inl].mean(0)
    n = np.linalg.svd(q, full_matrices=False)[2][-1]
    d = -n @ pts[inl].mean(0)
    if n[1] > 0:
        n, d = -n, -d
    return n, float(d), float(inl.mean())


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segment import pick_gpu
    from moge.model.v2 import MoGeModel
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    a = ap.parse_args()
    dev = a.device or pick_gpu(8000)
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()

    meta = [json.loads(l) for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()]
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())
    by_cam: dict[str, list[int]] = {}
    for r in meta:
        by_cam.setdefault(r["camera"], []).append(r["episode_index"])

    result = {}
    for cam, eps in sorted(by_cam.items()):
        pts_all, normals = [], []
        per_ep = max(1, N_PER_CAM // len(eps))
        for ep in eps:
            s = scored[str(ep)]
            z = np.load(MASKS / f"ep{ep:06d}.npz")
            H, W = z["shape"]; fids = z["frame_ids"]; packed = z["packed"]
            for f in np.linspace(s["first"], s["last"], per_ep).astype(int):
                i = int(np.searchsorted(fids, f))
                if i >= len(fids) or fids[i] != f:
                    continue
                m = np.unpackbits(packed[i], axis=-1)[:, :, :W].astype(bool)
                img = torch.tensor(np.asarray(Image.open(FRAMES / f"ep{ep:06d}/{f:06d}.jpg")
                                              .convert("RGB")) / 255.0,
                                   dtype=torch.float32, device=dev).permute(2, 0, 1)
                with torch.no_grad():
                    o = model.infer(img)
                pm = o["points"].cpu().numpy()            # [H,W,3] camera-frame points
                valid = o["mask"].cpu().numpy()
                floor = valid & ~m[0] & ~m[1]
                floor[: int(H * 0.55)] = False            # the floor is in the lower frame
                pts = pm[floor]
                if len(pts) > 4000:
                    pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
                if len(pts) >= 500:
                    n, d, frac = fit_plane_ransac(pts)
                    if n is not None and frac > 0.5:
                        normals.append(n)
                        pts_all.append(pts[np.abs(pts @ n + d) < 0.03][:1500])
        if not normals:
            print(f"{cam}: no plane", flush=True); continue
        # median normal, then one final global refit on the pooled inliers
        nm = np.median(np.stack(normals), 0); nm /= np.linalg.norm(nm)
        P = np.concatenate(pts_all)
        n, d, frac = fit_plane_ransac(P, iters=400, thresh=0.04)
        tilt = float(np.degrees(np.arccos(abs(n @ np.array([0.0, -1.0, 0.0])))))
        result[cam] = dict(normal=[round(float(x), 5) for x in n], offset=round(d, 5),
                           inlier_frac=round(frac, 3), n_frames=len(normals),
                           tilt_from_vertical_deg=round(tilt, 2),
                           normal_spread_deg=round(float(np.degrees(np.arccos(
                               np.clip(np.stack(normals) @ nm, -1, 1)).std())), 2))
        print(json.dumps({cam: result[cam]}), flush=True)
    (OUT / "ground.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
