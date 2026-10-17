#!/usr/bin/env python3
"""SAM 3D Body on every scored frame: the learned 3D prior the keypoint fit was missing.

Inputs are OUR person mask (bbox + mask prompt) and OUR MoGe intrinsics, exactly as the
organisers' estimate_mhr_params.py does it, so the built-in detector (detectron2) is never
needed. Weights come from the ungated Blendcap/sam-3d-body mirror; the MHR model is the
Apache-2.0 release the kit requires (hash verified), not the one bundled with the mirror.

Verified on ep0 f200: feeding the returned mhr_model_params(204)+shape(45) through our own MHR
TorchScript reproduces SAM3D's pred_vertices to 0.00 cm -- no hidden convention between the two.
Root translation is NOT in model_params (those are 0); it lives in pred_cam_t, which is saved raw
here and resolved into pose[0:3] at export time (frame of pred_cam_t is checked separately).

Device: the model hardcodes `cuda` internally, so pick the card with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/sam3dbody_code"))
OUT, STATE = ROOT / "work/sam3d", ROOT / "state/body_sam3d"


def main() -> None:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    a = ap.parse_args()
    model, cfg = load_sam_3d_body(str(ROOT / "work/models/sam3dbody/model.ckpt"), device="cuda",
                                  mhr_path=str(ROOT / "work/models/mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    K_all = json.load((ROOT / "work/camera/intrinsics.json").open())
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    OUT.mkdir(parents=True, exist_ok=True); STATE.mkdir(parents=True, exist_ok=True)

    for ep in (a.eps if a.eps else range(30)):
        if (STATE / f"ep{ep:06d}.done").exists():
            continue
        K = K_all[meta[ep]["camera"]]
        cam = torch.tensor([[[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]]], dtype=torch.float32)
        kz = np.load(ROOT / f"work/kp2d/ep{ep:06d}.npz"); fids = kz["frame_ids"]
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); H, W = z["shape"]
        midx = np.searchsorted(z["frame_ids"], fids); packed = z["packed"]
        T = len(fids)
        mp = np.zeros((T, 204), np.float32); sh = np.zeros((T, 45), np.float32)
        ct = np.zeros((T, 3), np.float32); k3 = np.zeros((T, 70, 3), np.float32)
        ok = np.zeros(T, bool); t0 = time.time()
        for t, (f, mi) in enumerate(zip(fids, midx)):
            m = np.unpackbits(packed[mi], axis=-1)[1, :, :W].astype(np.uint8)
            ys, xs = np.where(m)
            if len(xs) < 50:
                continue
            bbox = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
            img = np.asarray(Image.open(ROOT / f"work/frames/ep{ep:06d}/{f:06d}.jpg").convert("RGB"))
            try:
                out = est.process_one_image(img, bboxes=bbox, masks=m[None], cam_int=cam,
                                            use_mask=True, inference_type="body")
            except Exception as exc:
                print(f"ep{ep:06d} f{f}: {type(exc).__name__}: {str(exc)[:80]}", flush=True); continue
            if not out:
                continue
            o = out[0]; g = lambda k: np.asarray(o[k].detach().cpu() if hasattr(o[k], "detach") else o[k])
            mp[t] = g("mhr_model_params").reshape(-1)[:204]; sh[t] = g("shape_params").reshape(-1)[:45]
            ct[t] = g("pred_cam_t").reshape(-1)[:3]
            kk = g("pred_keypoints_3d").reshape(-1, 3); k3[t, :min(70, len(kk))] = kk[:70]
            ok[t] = True
        np.savez_compressed(OUT / f"ep{ep:06d}.npz", model_params=mp, shape=sh, cam_t=ct,
                            keypoints3d=k3, ok=ok, frame_ids=fids)
        rec = dict(ep=ep, frames=int(T), ok=int(ok.sum()), sec_per_frame=round((time.time() - t0) / max(1, T), 2),
                   shape_norm_med=round(float(np.median(np.linalg.norm(sh[ok], axis=1))), 2) if ok.any() else None,
                   depth_med=round(float(np.median(ct[ok, 2])), 2) if ok.any() else None)
        (STATE / f"ep{ep:06d}.done").write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
