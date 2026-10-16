#!/usr/bin/env python3
"""Re-seed the OBJECT mask only, by detecting inside a crop around the (reliable) person.

Why: person tracking succeeded on all 30 episodes with zero lost frames, but the object failed
wherever it is small in a 1536x1152 wide shot -- paint_roller medians of 0-832 px, foam block at
354 px, white_desk ep9 at 3.7k against 22k for the same object elsewhere. Detecting inside a crop
around the person raises the object's effective resolution by roughly the crop ratio, and during
the scored span the object is by definition within reach of the hands, so the crop always
contains it.

Two further changes over the first pass:
  - try SEVERAL seed frames and keep the best-scoring detection, instead of trusting frame `first`
    where the object is often still occluded by the approaching hand;
  - try a SHORT noun phrase alongside the dataset's long descriptive prompt. OWLv2 degrades on
    long prompts: "a foam block with a pixalated design of various shades of green and various
    shades of brown." scored 0.13, while "a foam block" is what the detector was trained on.

The person channel of the existing mask file is reused untouched.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRAMES, MASKS, STATE = ROOT / "work/frames", ROOT / "work/masks", ROOT / "state/seg"
N_SEEDS, CROP_PAD = 12, 1.6


def short_prompt(obj: str) -> str:
    return "a " + obj.replace("_", " ")


def person_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    return None if len(xs) == 0 else np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def crop_box(bbox: np.ndarray, W: int, H: int) -> tuple[int, int, int, int]:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * CROP_PAD / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    return x0, y0, x1, y1


def best_seed(ep: int, det, meta: dict, spans: list, masks_person, frame_ids, W: int, H: int):
    """Scan candidate frames x {short, long} prompt x person-crop; return the best detection."""
    cands = np.unique(np.concatenate([np.linspace(a, b, max(2, N_SEEDS // len(spans))).astype(int)
                                      for a, b in spans]))
    prompts = [short_prompt(meta["object"]), meta["object_prompt"].rstrip(".")]
    best = (0.0, None, None, None)
    for f in cands:
        i = int(np.searchsorted(frame_ids, f))
        if i >= len(frame_ids) or frame_ids[i] != f:
            continue
        bb = person_bbox(masks_person[i])
        if bb is None:
            continue
        x0, y0, x1, y1 = crop_box(bb, W, H)
        img = Image.open(FRAMES / f"ep{ep:06d}/{f:06d}.jpg").convert("RGB").crop((x0, y0, x1, y1))
        for score, box in det.boxes(img, prompts):
            if box is None or score <= best[0]:
                continue
            best = (float(score), box + np.array([x0, y0, x0, y0]), int(i), prompts)
    return best


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segment import Detector, pick_gpu
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", required=True)
    ap.add_argument("--device")
    a = ap.parse_args()
    device = a.device or pick_gpu()
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())
    det = Detector(device)

    for ep in a.eps:
        z = np.load(MASKS / f"ep{ep:06d}.npz")
        H, W = z["shape"]; frame_ids = z["frame_ids"]
        m = np.unpackbits(z["packed"], axis=-1)[:, :, :, :W].astype(bool)
        score, box, seed_i, prompts = best_seed(ep, det, meta[ep], scored[str(ep)]["spans"],
                                                m[:, 1], frame_ids, W, H)
        if box is None:
            print(f"ep{ep:06d}: still no detection", flush=True); continue

        pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
        with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            st = pred.init_state(video_path=str(FRAMES / f"ep{ep:06d}"),
                                 offload_video_to_cpu=True, offload_state_to_cpu=True)
            pred.add_new_points_or_box(st, frame_idx=seed_i, obj_id=0, box=box)
            new = np.zeros((len(frame_ids), H, W), bool)
            for rev in (False, True):
                for idx, ids, lg in pred.propagate_in_video(st, reverse=rev):
                    new[idx] = (lg[0][0] > 0).cpu().numpy()
        del st, pred; gc.collect(); torch.cuda.empty_cache()

        old_px = int(np.median(m[:, 0].reshape(len(frame_ids), -1).sum(-1)))
        new_px = int(np.median(new.reshape(len(frame_ids), -1).sum(-1)))
        lost = int((new.reshape(len(frame_ids), -1).sum(-1) == 0).sum())
        m[:, 0] = new
        np.savez_compressed(MASKS / f"ep{ep:06d}.npz", packed=np.packbits(m, axis=-1),
                            frame_ids=frame_ids, shape=np.array([H, W]))
        rec = json.loads((STATE / f"ep{ep:06d}.done").read_text())
        rec.update(refix=dict(det_score=round(score, 3), seed_frame=int(frame_ids[seed_i]),
                              obj_px_before=old_px, obj_px_after=new_px, obj_lost_after=lost))
        rec["obj_px_median"], rec["obj_lost_frames"] = new_px, lost
        (STATE / f"ep{ep:06d}.done").write_text(json.dumps(rec))
        print(json.dumps(dict(ep=ep, **rec["refix"])), flush=True)


if __name__ == "__main__":
    main()
