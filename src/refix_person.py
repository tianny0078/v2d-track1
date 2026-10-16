#!/usr/bin/env python3
"""Re-seed the PERSON mask by picking the person who is actually touching the object.

The first pass prompted OWLv2 with "a person" and kept the top-scoring box. In a room with
seated bystanders that is close to arbitrary -- person scores were 0.14-0.25 across the board --
and six episodes ended up tracking a bystander at a desk while the acting person went unmasked.

The acting person is the one in contact with the target object, and the object masks are now
reliable for all 30 episodes, so the object itself disambiguates: among all person detections,
keep the box whose distance to the object mask's centroid is smallest (zero when the centroid
falls inside the box). No ground truth and no hand detector needed.

Only channel 1 is rewritten; the object channel is left untouched.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MASKS, FRAMES, STATE = ROOT / "work/masks", ROOT / "work/frames", ROOT / "state/seg"


def all_person_boxes(det, image, threshold: float = 0.04):
    """Every person detection, not just the best-scoring one."""
    inp = det.proc(text=[["a person"]], images=image, return_tensors="pt",
                   padding=True, truncation=True).to(det.device)
    with torch.no_grad():
        out = det.model(**{k: (v.half() if v.is_floating_point() else v) for k, v in inp.items()})
    res = det.proc.post_process_grounded_object_detection(
        out, target_sizes=torch.tensor([[image.height, image.width]]), threshold=threshold)[0]
    return [(float(s), b.float().cpu().numpy()) for s, b in zip(res["scores"], res["boxes"])]


def point_box_distance(p: np.ndarray, box: np.ndarray) -> float:
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return float(np.hypot(dx, dy))


def run(ep: int, det, device: str) -> dict:
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from PIL import Image
    z = np.load(MASKS / f"ep{ep:06d}.npz")
    H, W = z["shape"]; fids = z["frame_ids"]
    m = np.unpackbits(z["packed"], axis=-1)[:, :, :, :W].astype(bool)
    obj_area = m[:, 0].reshape(len(fids), -1).sum(-1)
    spans = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]["spans"]
    scored = np.concatenate([np.arange(a, b + 1) for a, b in spans])
    cand = np.array([i for i, f in enumerate(fids) if f in set(scored.tolist())])
    seed_i = int(cand[np.argmax(obj_area[cand])])          # object most visible => interaction clear

    ys, xs = np.where(m[seed_i, 0])
    centroid = np.array([xs.mean(), ys.mean()])
    img = Image.open(FRAMES / f"ep{ep:06d}/{fids[seed_i]:06d}.jpg").convert("RGB")
    boxes = all_person_boxes(det, img)
    if not boxes:
        raise RuntimeError(f"ep{ep}: no person detected at frame {fids[seed_i]}")
    dists = [point_box_distance(centroid, b) for _, b in boxes]
    k = int(np.argmin(dists))
    score, box = boxes[k]

    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        st = pred.init_state(video_path=str(FRAMES / f"ep{ep:06d}"),
                             offload_video_to_cpu=True, offload_state_to_cpu=True)
        pred.add_new_points_or_box(st, frame_idx=seed_i, obj_id=0, box=box)
        new = np.zeros((len(fids), H, W), bool)
        for rev in (False, True):
            for idx, ids, lg in pred.propagate_in_video(st, reverse=rev, start_frame_idx=seed_i):
                new[idx] = (lg[0][0] > 0).cpu().numpy()
    del st, pred; gc.collect(); torch.cuda.empty_cache()

    before = int(np.median(m[:, 1].reshape(len(fids), -1).sum(-1)))
    m[:, 1] = new
    area = new.reshape(len(fids), -1).sum(-1)
    np.savez_compressed(MASKS / f"ep{ep:06d}.npz", packed=np.packbits(m, axis=-1),
                        frame_ids=fids, shape=np.array([H, W]))
    rec = json.loads((STATE / f"ep{ep:06d}.done").read_text())
    rec["person_px_median"], rec["person_lost_frames"] = int(np.median(area)), int((area == 0).sum())
    rec["person_refix"] = dict(seed_frame=int(fids[seed_i]), n_candidates=len(boxes),
                               det_score=round(score, 3), centroid_dist_px=round(dists[k], 1),
                               px_before=before, px_after=rec["person_px_median"])
    (STATE / f"ep{ep:06d}.done").write_text(json.dumps(rec))
    return dict(ep=ep, **rec["person_refix"], lost=rec["person_lost_frames"])


if __name__ == "__main__":
    from src.segment import Detector, pick_gpu
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    ap.add_argument("--device")
    a = ap.parse_args()
    dev = a.device or pick_gpu()
    det = Detector(dev)
    for ep in (a.eps if a.eps else range(30)):
        try:
            print(json.dumps(run(ep, det, dev)), flush=True)
        except Exception as exc:
            (STATE / f"ep{ep:06d}.personfail").write_text(f"{type(exc).__name__}: {exc}")
            print(f"FAIL ep{ep:06d}: {exc}", flush=True)
