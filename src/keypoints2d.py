#!/usr/bin/env python3
"""COCO-17 2D keypoints for the acting person, every scored frame.

These are the evidence the MHR fit is driven by. The person box comes from our own person mask,
not from a detector, because the mask already resolves which of the people in the room is the
one acting -- the detector does not (it picked a seated bystander in six episodes).

Output per episode: kp [N,17,3] as (x, y, score) in full-frame pixels, plus the frame ids.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MASKS, FRAMES = ROOT / "work/masks", ROOT / "work/frames"
OUT, STATE = ROOT / "work/kp2d", ROOT / "state/kp2d"
BATCH = 32


def person_boxes(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return None
    return np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], float)


def run(ep: int, model, proc, device: str) -> dict:
    from PIL import Image
    z = np.load(MASKS / f"ep{ep:06d}.npz")
    H, W = z["shape"]; fids = z["frame_ids"]; packed = z["packed"]
    spans = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]["spans"]
    want = set(np.concatenate([np.arange(a, b + 1) for a, b in spans]).tolist())
    idxs = [i for i, f in enumerate(fids) if int(f) in want]

    kps = np.full((len(idxs), 17, 3), np.nan, np.float32)
    missing = 0
    for s in range(0, len(idxs), BATCH):
        chunk = idxs[s:s + BATCH]
        imgs, boxes, keep = [], [], []
        for k, i in enumerate(chunk):
            m = np.unpackbits(packed[i], axis=-1)[1, :, :W].astype(bool)
            b = person_boxes(m)
            if b is None:
                missing += 1
                continue
            imgs.append(Image.open(FRAMES / f"ep{ep:06d}/{fids[i]:06d}.jpg").convert("RGB"))
            boxes.append(b[None]); keep.append(s + k)
        if not imgs:
            continue
        inp = proc(imgs, boxes=boxes, return_tensors="pt").to(device)
        inp["pixel_values"] = inp["pixel_values"].half()   # model is fp16; the processor emits fp32
        with torch.no_grad():
            out = model(**inp, dataset_index=torch.zeros(len(imgs), dtype=torch.int64, device=device))
        out.heatmaps = out.heatmaps.float()   # post-processing runs scipy, which has no fp16 path
        res = proc.post_process_pose_estimation(out, boxes=boxes)
        for k, r in zip(keep, res):
            p = r[0]
            kps[k, :, :2] = p["keypoints"].cpu().numpy()
            kps[k, :, 2] = p["scores"].cpu().numpy()

    OUT.mkdir(parents=True, exist_ok=True); STATE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"ep{ep:06d}.npz", kp=kps, frame_ids=fids[idxs])
    ok = ~np.isnan(kps[:, 0, 0])
    conf = float(np.nanmean(kps[ok][:, :, 2])) if ok.any() else 0.0
    rec = dict(ep=ep, frames=len(idxs), no_person=missing, mean_conf=round(conf, 3),
               low_conf_frames=int((np.nanmean(kps[:, :, 2], 1) < 0.3).sum()))
    (STATE / f"ep{ep:06d}.done").write_text(json.dumps(rec))
    return rec


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segment import pick_gpu
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    ap.add_argument("--device")
    a = ap.parse_args()
    dev = a.device or pick_gpu(8000)
    p = ROOT / "work/models/vitpose"
    proc = AutoProcessor.from_pretrained(p)
    model = VitPoseForPoseEstimation.from_pretrained(p, dtype=torch.float16).to(dev).eval()
    for ep in (a.eps if a.eps else range(30)):
        print(json.dumps(run(ep, model, proc, dev)), flush=True)


if __name__ == "__main__":
    main()
