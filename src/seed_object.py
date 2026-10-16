#!/usr/bin/env python3
"""Re-propagate the object mask from a manually chosen seed box.

The automatic path (src/refix_object.py) fails on small objects that a near-duplicate distractor
outscores -- ep19's paint roller lost to a yellow floor marker at 0.457. For those, the seed box
is read off an inspect_frame.py grid by eye. MANUAL STEP: every episode seeded this way is
recorded in notes/manual_seeds.md and must appear in the technical report.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MASKS, STATE = ROOT / "work/masks", ROOT / "state/seg"


def propagate(ep: int, seed_frame: int, box: np.ndarray, device: str) -> dict:
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    z = np.load(MASKS / f"ep{ep:06d}.npz")
    H, W = z["shape"]; frame_ids = z["frame_ids"]
    m = np.unpackbits(z["packed"], axis=-1)[:, :, :, :W].astype(bool)
    i = int(np.searchsorted(frame_ids, seed_frame))
    if i >= len(frame_ids) or frame_ids[i] != seed_frame:
        raise SystemExit(f"ep{ep}: frame {seed_frame} not extracted (have {frame_ids[0]}..{frame_ids[-1]})")

    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        st = pred.init_state(video_path=str(ROOT / f"work/frames/ep{ep:06d}"),
                             offload_video_to_cpu=True, offload_state_to_cpu=True)
        pred.add_new_points_or_box(st, frame_idx=i, obj_id=0, box=box.astype(np.float32))
        new = np.zeros((len(frame_ids), H, W), bool)
        for rev in (False, True):
            for idx, ids, lg in pred.propagate_in_video(st, reverse=rev):
                new[idx] = (lg[0][0] > 0).cpu().numpy()
    del st, pred; gc.collect(); torch.cuda.empty_cache()

    area = new.reshape(len(frame_ids), -1).sum(-1)
    m[:, 0] = new
    np.savez_compressed(MASKS / f"ep{ep:06d}.npz", packed=np.packbits(m, axis=-1),
                        frame_ids=frame_ids, shape=np.array([H, W]))
    rec = json.loads((STATE / f"ep{ep:06d}.done").read_text())
    rec["obj_px_median"], rec["obj_lost_frames"] = int(np.median(area)), int((area == 0).sum())
    rec["manual_seed"] = dict(frame=seed_frame, box=[int(v) for v in box])
    (STATE / f"ep{ep:06d}.done").write_text(json.dumps(rec))
    return dict(ep=ep, seed_frame=seed_frame, obj_px_median=rec["obj_px_median"],
                obj_lost_frames=rec["obj_lost_frames"])


if __name__ == "__main__":
    from src.segment import pick_gpu
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--box", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--device")
    a = ap.parse_args()
    print(json.dumps(propagate(a.ep, a.frame, np.array(a.box, float), a.device or pick_gpu())))
