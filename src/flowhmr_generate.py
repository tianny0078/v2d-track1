#!/usr/bin/env python3
"""Run FlowHMR generation on our SAM3D tokens and cache the raw output (SMPL-H rot6d/shapes/trans).

Deliberately bypasses V2MInferenceEngine.build_lite(): it constructs SMPLSkeleton/SMPLMesh
unconditionally, which needs the gated SMPL-H neutral model we do not have yet. Generation itself
(build_pipeline + pipeline.generate) needs only the checkpoint. FK / export / conversion to MHR
happen later, once the body model is in place.

Camera: identity rotation for every frame -- exactly what run_generation() does for the web and
lite paths. Our cameras are static, so that is not an approximation. World frame then equals
the OpenCV camera frame, which is our submission frame.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/flowhmr"))
CKPT = ROOT / "work/models/flowhmr/flowhmr_latest/flowhmr_latest.ckpt"
CFG = ROOT / "work/models/flowhmr/flowhmr_latest/config.yml"
TOK, OUT = ROOT / "work/flowhmr_tokens", ROOT / "work/flowhmr_out"


def main() -> None:
    from flowhmr.pipeline.v2m_inference import build_pipeline, load_yaml, build_feature_dict
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_yaml(str(CFG))
    # the pipeline resolves "assets/body_models/..." relative to the CWD; give it the absolute
    # path (build_lite forwards the same kwarg) and also chdir so any other relative asset works
    smplh = str(ROOT / "work/flowhmr/assets/body_models/smplh/neutral/model.npz")
    import os; os.chdir(str(ROOT / "work/flowhmr"))
    pipe = build_pipeline(cfg, device=a.device, ckpt=str(CKPT), smpl_model_path=smplh)
    print(f"pipeline params: {sum(p.numel() for p in pipe.parameters()):,}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in (a.eps if a.eps else range(30)):
        f = TOK / f"ep{ep:06d}.npz"
        if not f.exists():
            print(f"ep{ep:06d}: no tokens", flush=True); continue
        z = np.load(f); tokens = torch.from_numpy(z["tokens"]); T = tokens.shape[0]
        camera_RT = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
        feat = build_feature_dict(tokens, camera_RT, torch.device(a.device))
        t0 = time.time()
        with torch.no_grad():
            out = pipe.generate(feature=feat, seeds=[a.seed], length=T, cfg_scale=1.0)
        keep = {k: (v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        np.savez_compressed(OUT / f"ep{ep:06d}.npz", frame_ids=z["frame_ids"],
                            **{k: v for k, v in keep.items() if isinstance(v, np.ndarray)})
        shapes = {k: tuple(v.shape) for k, v in keep.items() if isinstance(v, np.ndarray)}
        print(json.dumps(dict(ep=ep, T=int(T), seconds=round(time.time() - t0, 1), out=shapes)), flush=True)


if __name__ == "__main__":
    main()
