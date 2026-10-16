#!/usr/bin/env python3
"""Camera intrinsics per PHYSICAL camera, from MoGe.

The 30 episodes come from only 4 physical cameras, so intrinsics are estimated 4 times, not 30,
and each estimate is the median over several frames of several episodes -- a single-image FOV
estimate is noisy, and a static camera gives us free repeats.

Why we need this at all: both the MHR body fit and the object silhouette fit project metric 3D
into the image, and the focal length sets how depth trades off against size. Everything else in
the pipeline is scale-free (the scorer's frame-0 Sim(3) absorbs global scale), but the focal is
not -- it controls RELATIVE depth, between body parts and between the human and the object.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "work/frames"
OUT = ROOT / "work/camera"
N_PER_CAM = 12


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

    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for cam, eps in sorted(by_cam.items()):
        picks = []
        per_ep = max(1, N_PER_CAM // len(eps))
        for ep in eps:
            s = scored[str(ep)]
            for f in np.linspace(s["first"], s["last"], per_ep).astype(int):
                picks.append((ep, int(f)))
        fovs, foc = [], []
        for ep, f in picks:
            p = FRAMES / f"ep{ep:06d}/{f:06d}.jpg"
            if not p.exists():
                continue
            img = torch.tensor(np.asarray(Image.open(p).convert("RGB")) / 255.0,
                               dtype=torch.float32, device=dev).permute(2, 0, 1)
            with torch.no_grad():
                out = model.infer(img)
            H, W = img.shape[1:]
            inr = out["intrinsics"].cpu().numpy()      # normalised: fx is in units of image width
            fx, fy = float(inr[0, 0]) * W, float(inr[1, 1]) * H
            fovs.append(float(np.degrees(2 * np.arctan(0.5 * W / fx))))
            foc.append((fx, fy, float(inr[0, 2]) * W, float(inr[1, 2]) * H))
        arr = np.array(foc)
        med = np.median(arr, 0)
        result[cam] = dict(episodes=sorted(eps), n_samples=len(foc),
                           fx=round(float(med[0]), 2), fy=round(float(med[1]), 2),
                           cx=round(float(med[2]), 2), cy=round(float(med[3]), 2),
                           fov_x_deg=round(float(np.median(fovs)), 2),
                           fx_iqr=round(float(np.percentile(arr[:, 0], 75) - np.percentile(arr[:, 0], 25)), 2))
        print(json.dumps({cam: result[cam]}), flush=True)
    (OUT / "intrinsics.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
