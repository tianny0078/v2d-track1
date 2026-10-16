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


def propagate(ep: int, seeds: list[tuple[int, np.ndarray]], device: str) -> dict:
    """seeds: [(video_frame, box_xyxy), ...]. SAM 2 accepts prompts at several frames for the same
    object id, which is what long occlusions need -- a single seed propagated across a stretch
    where the body hides the object loses it and never recovers (ep7: 260 of 597 frames lost from
    one seed)."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    z = np.load(MASKS / f"ep{ep:06d}.npz")
    H, W = z["shape"]; frame_ids = z["frame_ids"]
    m = np.unpackbits(z["packed"], axis=-1)[:, :, :, :W].astype(bool)
    idxs = []
    for f, _ in seeds:
        i = int(np.searchsorted(frame_ids, f))
        if i >= len(frame_ids) or frame_ids[i] != f:
            raise SystemExit(f"ep{ep}: frame {f} not extracted (have {frame_ids[0]}..{frame_ids[-1]})")
        idxs.append(i)

    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        st = pred.init_state(video_path=str(ROOT / f"work/frames/ep{ep:06d}"),
                             offload_video_to_cpu=True, offload_state_to_cpu=True)
        for i, (_, box) in zip(idxs, seeds):
            pred.add_new_points_or_box(st, frame_idx=i, obj_id=0, box=box.astype(np.float32))
        new = np.zeros((len(frame_ids), H, W), bool)
        # Forward from the EARLIEST seed and backward from the LATEST, so the two sweeps overlap
        # across the whole episode. Starting each sweep at the near end instead leaves the span
        # BETWEEN the seeds untouched -- which is exactly the stretch the extra seed was added for.
        for rev in (False, True):
            for idx, ids, lg in pred.propagate_in_video(st, reverse=rev,
                                                        start_frame_idx=max(idxs) if rev else min(idxs)):
                new[idx] = (lg[0][0] > 0).cpu().numpy()
    del st, pred; gc.collect(); torch.cuda.empty_cache()

    area = new.reshape(len(frame_ids), -1).sum(-1)
    m[:, 0] = new
    np.savez_compressed(MASKS / f"ep{ep:06d}.npz", packed=np.packbits(m, axis=-1),
                        frame_ids=frame_ids, shape=np.array([H, W]))
    rec = json.loads((STATE / f"ep{ep:06d}.done").read_text())
    rec["obj_px_median"], rec["obj_lost_frames"] = int(np.median(area)), int((area == 0).sum())
    rec["manual_seed"] = [dict(frame=int(f), box=[int(v) for v in b]) for f, b in seeds]
    (STATE / f"ep{ep:06d}.done").write_text(json.dumps(rec))
    return dict(ep=ep, seeds=[int(f) for f, _ in seeds], obj_px_median=rec["obj_px_median"],
                obj_lost_frames=rec["obj_lost_frames"])


if __name__ == "__main__":
    from src.segment import pick_gpu
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--seed", action="append", required=True, metavar="FRAME,X0,Y0,X1,Y1",
                    help="repeatable: one seed per occlusion-free view of the object")
    ap.add_argument("--device")
    a = ap.parse_args()
    seeds = []
    for spec in a.seed:
        v = [int(x) for x in spec.split(",")]
        if len(v) != 5:
            raise SystemExit(f"--seed needs FRAME,X0,Y0,X1,Y1, got {spec!r}")
        seeds.append((v[0], np.array(v[1:], float)))
    print(json.dumps(propagate(a.ep, seeds, a.device or pick_gpu())))
