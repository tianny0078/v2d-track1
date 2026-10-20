#!/usr/bin/env python3
"""Harness sanity check: SAM 3D Body (body mode) on a few BEHAVE frames -> our export format.

Not a pipeline run: the person bbox and mask come from the projected GT mesh (so the only thing being
tested is the frame convention + the GT + the scorer). Output goes to work/behave_pred/sam3d_sanity/<seq>/
with the object arrays copied from the GT, so CD-O there only measures the Sim(3) leakage.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2, torch, shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/sam3dbody_code"))
from src.fit_mhr import POSE_UNIT_M, POSE_SIGN  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Date01_Sub01_chairwood_hand")
    ap.add_argument("--frames", default="200-210")
    a = ap.parse_args()
    lo, hi = map(int, a.frames.split("-"))
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    model, cfg = load_sam_3d_body(str(ROOT / "work/models/sam3dbody/model.ckpt"), device="cuda",
                                  mhr_path=str(ROOT / "work/models/mhr_model.pt"))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    meta = json.load((ROOT / f"work/behave/{a.seq}/meta.json").open())
    g = np.load(ROOT / f"work/behave_gt/{a.seq}/gt_smplh.npz")
    faces = np.load(ROOT / "work/behave_gt/_assets/smplh_male_v12.npz")["f"].astype(np.int64)
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
    cam = torch.tensor(K[None], dtype=torch.float32)
    W, H = meta["width"], meta["height"]
    T = len(g["verts"]); mp = np.zeros((T, 204), np.float32); sh = np.zeros((T, 45), np.float32); ct = np.zeros((T, 3), np.float32)
    ok = np.zeros(T, bool)
    for t in range(lo, hi):
        v = g["verts"][t]; uv = (K @ v.T).T; uv = uv[:, :2] / uv[:, 2:]
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [uv[f].astype(np.int32) for f in faces], 1)
        ys, xs = np.where(mask)
        bbox = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
        img = cv2.cvtColor(cv2.imread(str(ROOT / f"work/behave/{a.seq}/frames/{t:06d}.jpg")), cv2.COLOR_BGR2RGB)
        out = est.process_one_image(img, bboxes=bbox, masks=mask[None], cam_int=cam, use_mask=True, inference_type="body")
        if not out:
            print("no output", t); continue
        o = out[0]; gg = lambda k: np.asarray(o[k].detach().cpu() if hasattr(o[k], "detach") else o[k])
        mp[t] = gg("mhr_model_params").reshape(-1)[:204]; sh[t] = gg("shape_params").reshape(-1)[:45]
        ct[t] = gg("pred_cam_t").reshape(-1)[:3]; ok[t] = True
        print(t, "cam_t", np.round(ct[t], 3), "gt pelvis", np.round(g["joints"][t, 0], 3), flush=True)
    good = np.where(ok)[0]
    near = good[np.abs(good[None, :] - np.arange(T)[:, None]).argmin(1)]
    pose = mp[near, :136].copy()
    pose[:, :3] = ct[near] / POSE_UNIT_M * np.array(POSE_SIGN, np.float32)
    scales = np.median(mp[good, 136:], 0).astype(np.float32); shape = np.median(sh[good], 0).astype(np.float32)
    gt = np.load(ROOT / f"work/behave_gt/{a.seq}/object_pose.npz")
    out = ROOT / f"work/behave_pred/sam3d_sanity/{a.seq}"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "episode_000000.npz", pose=pose.astype(np.float32), scales=scales, shape=shape,
             object_rotation=gt["rotation"], object_translation=gt["translation"], object_scale=np.float32(gt["scale"]))
    shutil.copy(ROOT / f"work/behave_gt/{a.seq}/episode_000000_object.obj", out / "episode_000000_object.obj")
    print("wrote", out, "frames ok", int(ok.sum()))


if __name__ == "__main__":
    main()
