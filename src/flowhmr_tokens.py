#!/usr/bin/env python3
"""SAM 3D Body pose tokens for FlowHMR, from OUR frames, masks and intrinsics.

FlowHMR conditions on SAM 3D Body's decoder pose tokens (body + left hand + right hand), which
its Sam3DTokenExtractor captures with forward hooks. Its extract_video_tokens() reads a video
file and runs its own detector; we call extract_frame() per frame instead, with the person
bbox from our SAM2 mask and our MoGe intrinsics -- the same inputs we already validated for the
MHR export. Tokens need inference_type="full" (hands included), so this is a separate pass from
the body-only MHR batch.

The model is a motion generator at 30 fps and assumes CONTIGUOUS frames, but Track 1 scores
only the hand-object contact span, which can have gaps. We extract the whole contiguous range
first_scored-LEAD .. last_scored (frames exist on disk) and pick scored frames from the output.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/sam3dbody_code"))
sys.path.insert(0, str(ROOT / "work/flowhmr"))
OUT = ROOT / "work/flowhmr_tokens"


def main() -> None:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    a = ap.parse_args()
    model, cfg = load_sam_3d_body(str(ROOT / "work/models/sam3dbody/model.ckpt"), device="cuda",
                                  mhr_path=str(ROOT / "work/models/mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    ext = Sam3DTokenExtractor(est)
    K_all = json.load((ROOT / "work/camera/intrinsics.json").open())
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in (a.eps if a.eps else range(30)):
        if (OUT / f"ep{ep:06d}.npz").exists():
            continue
        K = K_all[meta[ep]["camera"]]
        cam = torch.tensor([[[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]]], dtype=torch.float32)  # must be a tensor: the extractor calls .to()
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); H, W = z["shape"]; fids = z["frame_ids"]; packed = z["packed"]
        toks, ok = [], np.zeros(len(fids), bool); t0 = time.time(); D = None
        for t, f in enumerate(fids):
            m = np.unpackbits(packed[t], axis=-1)[1, :, :W].astype(np.uint8)
            ys, xs = np.where(m)
            if len(xs) < 50:
                toks.append(None); continue
            bbox = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
            img = np.asarray(Image.open(ROOT / f"work/frames/ep{ep:06d}/{f:06d}.jpg").convert("RGB"))
            try:
                tok, _ = ext.extract_frame(img, bboxes=bbox, cam_int=cam, inference_type="full")
            except Exception as exc:
                print(f"ep{ep:06d} f{f}: {type(exc).__name__}: {str(exc)[:90]}", flush=True); toks.append(None); continue
            if tok is None or tok.numel() == 0:
                toks.append(None); continue
            toks.append(tok.detach().float().cpu().numpy()); ok[t] = True; D = toks[-1].shape[-1]
        if D is None:
            print(f"ep{ep:06d}: no tokens", flush=True); continue
        arr = np.zeros((len(fids), D), np.float32)
        good = np.where(ok)[0]
        for t in range(len(fids)):            # fill misses from the nearest good frame
            arr[t] = toks[t] if ok[t] else toks[good[np.abs(good - t).argmin()]]
        np.savez_compressed(OUT / f"ep{ep:06d}.npz", tokens=arr, frame_ids=fids, ok=ok)
        print(json.dumps(dict(ep=ep, frames=int(len(fids)), ok=int(ok.sum()), dim=int(D),
                              sec_per_frame=round((time.time() - t0) / max(1, len(fids)), 2))), flush=True)


if __name__ == "__main__":
    main()
