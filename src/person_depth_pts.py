#!/usr/bin/env python3
"""MoGe-2 metric point cloud of the PERSON per scored frame (for a depth-ICP body refinement, the
organisers' CARI4D stage 3/4 recipe). Saves up to N points per frame inside the person mask,
in the camera frame (x right, y down, z forward, metres; MoGe's own scale -- SAM3D's depth is
1.17x MoGe's on this data, so the fitter solves one scale per episode).
Output work/moge_pts/epNNNNNN.npz: frame_ids, pts [T,N,3] (NaN-padded), n [T]."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.export_episodes import _sam3d_as_body

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eps", type=int, nargs="+", required=True); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args(); dev = torch.device(a.device)
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    K = json.load(open(ROOT / "work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
    out_dir = ROOT / "work/moge_pts"; out_dir.mkdir(exist_ok=True); (ROOT / "state/moge_pts").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for ep in a.eps:
        t0 = time.time(); fids = _sam3d_as_body(ROOT / f"work/sam3d/ep{ep:06d}.npz")["frame_ids"]
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); H, W = [int(x) for x in z["shape"]]; mpos = {f: i for i, f in enumerate(z["frame_ids"])}
        k = K[meta[ep]["camera"]]; fov = k["fov_x_deg"]
        pts = np.full((len(fids), a.n, 3), np.nan, np.float32); n = np.zeros(len(fids), np.int64)
        for t, f in enumerate(fids):
            img = np.asarray(Image.open(ROOT / f"work/frames/ep{ep:06d}/{f:06d}.jpg").convert("RGB"))
            o = model.infer(torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1), fov_x=fov)
            P = o["points"].float().cpu().numpy(); valid = np.isfinite(P).all(-1) & (P[..., 2] > 0)
            if "mask" in o: valid &= o["mask"].cpu().numpy().astype(bool)
            m = np.unpackbits(z["packed"][mpos[f]], axis=-1)[1, :, :W].astype(bool) if f in mpos else np.zeros((H, W), bool)
            # erode the mask by 2 px so boundary pixels (depth bleeding into the background) are dropped
            from scipy.ndimage import binary_erosion
            m = binary_erosion(m, iterations=2) & valid
            ys, xs = np.where(m)
            if len(xs) == 0: continue
            sel = rng.choice(len(xs), min(a.n, len(xs)), replace=False)
            # MoGe's point map uses the image's own intrinsics (fov locked): x right, y down, z forward
            pts[t, :len(sel)] = P[ys[sel], xs[sel]]; n[t] = len(sel)
        np.savez_compressed(out_dir / f"ep{ep:06d}.npz", frame_ids=fids, pts=pts, n=n)
        (ROOT / "state/moge_pts" / f"ep{ep:06d}.done").touch()
        print(f"RESULT ep{ep}: {len(fids)} frames, {time.time()-t0:.0f}s, median pts/frame {int(np.median(n))}, median person z {np.nanmedian(pts[..., 2]):.2f} m", flush=True)

if __name__ == "__main__":
    main()
