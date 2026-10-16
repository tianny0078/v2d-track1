#!/usr/bin/env python3
"""Per-episode person and target-object masks: OWLv2 seeds a box, SAM 2.1 propagates it.

Seeded at the FIRST SCORED FRAME, because that is where the hand-object contact span starts and
so where the object is both present and being interacted with. Propagation runs both directions
from there, so the seed quality matters far more than any single later frame.

Masks are stored bit-packed: 30 episodes x ~400 frames x 2 instances of 1536x1152 bits is ~2.5 GB
as bytes and ~320 MB packed, and this box is 92% full.
"""
from __future__ import annotations
import argparse, gc, json, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "work/frames"
OUT = ROOT / "work/masks"
STATE = ROOT / "state/seg"
MODELS = ROOT / "work/models"
H, W = 1152, 1536


def pick_gpu(min_free_mb: int = 20000) -> str:
    """The box is shared. Take the emptiest card, and refuse to start if none is free enough."""
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    free = sorted(((int(l.split(",")[1]), int(l.split(",")[0])) for l in out), reverse=True)
    if not free or free[0][0] < min_free_mb:
        raise SystemExit(f"no GPU with {min_free_mb} MB free; current: {free}")
    return f"cuda:{free[0][1]}"


class Detector:
    def __init__(self, device: str):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        p = MODELS / "owlv2-large-patch14-ensemble"
        self.proc = Owlv2Processor.from_pretrained(p)
        self.model = Owlv2ForObjectDetection.from_pretrained(p, dtype=torch.float16).to(device).eval()
        self.device = device

    def boxes(self, image, prompts: list[str]) -> list[tuple[float, np.ndarray]]:
        # prompts differ in length, so the text branch must pad -- without this the processor
        # raises "Unable to create tensor" on any episode whose two prompts tokenize unevenly.
        inp = self.proc(text=[prompts], images=image, return_tensors="pt",
                        padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**{k: (v.half() if v.is_floating_point() else v) for k, v in inp.items()})
        res = self.proc.post_process_grounded_object_detection(
            out, target_sizes=torch.tensor([[image.height, image.width]]), threshold=0.05)[0]
        best = []
        for i in range(len(prompts)):
            sel = res["labels"] == i
            if not sel.any():
                best.append((0.0, None)); continue
            k = res["scores"][sel].argmax()
            best.append((float(res["scores"][sel][k]), res["boxes"][sel][k].float().cpu().numpy()))
        return best


def run_episode(ep: int, det: Detector, device: str, meta: dict, scored: dict) -> dict:
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from PIL import Image
    d = FRAMES / f"ep{ep:06d}"
    files = sorted(d.glob("*.jpg"))
    frame_ids = [int(f.stem) for f in files]
    seed_frame = scored["first"]
    seed_idx = frame_ids.index(seed_frame)

    prompts = [meta["object_prompt"].rstrip("."), "a person"]
    img = Image.open(files[seed_idx]).convert("RGB")
    dets = det.boxes(img, prompts)
    if any(b is None for _, b in dets):
        raise RuntimeError(f"ep{ep}: OWLv2 found no box for {[p for (s,b),p in zip(dets,prompts) if b is None]}")

    pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        state = pred.init_state(video_path=str(d), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for oid, (_, box) in enumerate(dets):
            pred.add_new_points_or_box(state, frame_idx=seed_idx, obj_id=oid, box=box)
        masks = np.zeros((len(files), 2, H, W), dtype=bool)
        seen = np.zeros(len(files), bool)
        for reverse in (False, True):
            for idx, ids, logits in pred.propagate_in_video(state, reverse=reverse):
                seen[idx] = True
                for oid, lg in zip(ids, logits):
                    masks[idx, oid] = (lg[0] > 0).cpu().numpy()
    del state, pred; gc.collect(); torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"ep{ep:06d}.npz", packed=np.packbits(masks, axis=-1),
                        frame_ids=np.array(frame_ids), shape=np.array([H, W]))
    area = masks.reshape(len(files), 2, -1).sum(-1)
    return dict(ep=ep, frames=len(files), seed_frame=seed_frame,
                det_scores=[round(s, 3) for s, _ in dets], propagated=int(seen.sum()),
                obj_px_median=int(np.median(area[:, 0])), person_px_median=int(np.median(area[:, 1])),
                obj_lost_frames=int((area[:, 0] == 0).sum()), person_lost_frames=int((area[:, 1] == 0).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="*")
    ap.add_argument("--device")
    a = ap.parse_args()
    device = a.device or pick_gpu()
    print("device", device, flush=True)
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())
    det = Detector(device)
    STATE.mkdir(parents=True, exist_ok=True)
    for ep in (a.eps if a.eps else range(30)):
        if (STATE / f"ep{ep:06d}.done").exists():
            continue
        try:
            r = run_episode(ep, det, device, meta[ep], scored[str(ep)])
        except Exception as exc:  # one bad episode must not abort the other 29
            import traceback
            (STATE / f"ep{ep:06d}.fail").write_text(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            print(f"FAIL ep{ep:06d}: {type(exc).__name__}: {exc}", flush=True)
            continue
        (STATE / f"ep{ep:06d}.fail").unlink(missing_ok=True)
        (STATE / f"ep{ep:06d}.done").write_text(json.dumps(r))
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
