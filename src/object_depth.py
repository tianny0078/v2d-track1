#!/usr/bin/env python3
"""Per-frame MoGe-2 depth statistics inside the object mask and the person mask.

Why: the object's global depth is the CD-O killer (object centroid 1-2 m from the hand on the worst
episodes). Mask-size depth needs the object's metric size and a clean silhouette; both fail on thin
or occluded objects. MoGe gives a dense depth map whose GLOBAL scale is uncertain, but the RATIO
object-depth / person-depth within one frame is scale-free, and the person's depth we already know
from SAM 3D Body. So: z_obj = z_person_sam3d * d_obj / d_person.

Output work/objdepth/epNNNNNN.npz: frame_ids, d_obj (median, p20, p80), d_person (same), n_obj,
n_person px, obj_px (mask centroid). Depth at the camera's fixed fov_x (MoGe intrinsics-locked).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.export_episodes import _sam3d_as_body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device)
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    K = json.load(open(ROOT / "work/camera/intrinsics.json"))
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
    out_dir = ROOT / "work/objdepth"; out_dir.mkdir(exist_ok=True)
    (ROOT / "state/objdepth").mkdir(parents=True, exist_ok=True)
    for ep in a.eps:
        t0 = time.time()
        fids = _sam3d_as_body(ROOT / f"work/sam3d/ep{ep:06d}.npz")["frame_ids"][:: a.stride]
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); H, W = z["shape"]; mpos = {f: i for i, f in enumerate(z["frame_ids"])}
        fov = K[meta[ep]["camera"]]["fov_x_deg"]
        rows = []
        for f in fids:
            img = np.asarray(Image.open(ROOT / f"work/frames/ep{ep:06d}/{f:06d}.jpg").convert("RGB"))
            t = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
            o = model.infer(t, fov_x=fov)
            d = o["depth"].float().cpu().numpy(); valid = np.isfinite(d) & (d > 0)
            if "mask" in o:
                valid &= o["mask"].cpu().numpy().astype(bool)
            m = np.unpackbits(z["packed"][mpos[f]], axis=-1)[:, :, :W].astype(bool) if f in mpos else np.zeros((2, H, W), bool)
            def stats(mk):
                v = d[mk & valid]
                return (np.nan, np.nan, np.nan, 0) if v.size < 20 else (float(np.median(v)), float(np.percentile(v, 20)), float(np.percentile(v, 80)), int(v.size))
            so, sp = stats(m[0]), stats(m[1])
            ys, xs = np.where(m[0]); c = (float(xs.mean()), float(ys.mean())) if xs.size else (np.nan, np.nan)
            rows.append((f, *so, *sp, *c))
        r = np.array(rows, dtype=np.float64)
        np.savez_compressed(out_dir / f"ep{ep:06d}.npz", frame_ids=r[:, 0].astype(np.int64), d_obj=r[:, 1:4], n_obj=r[:, 4].astype(np.int64),
                            d_person=r[:, 5:8], n_person=r[:, 8].astype(np.int64), obj_px=r[:, 9:11])
        (ROOT / f"state/objdepth/ep{ep:06d}.done").touch()
        ok = np.isfinite(r[:, 1]) & np.isfinite(r[:, 5])
        print(f"RESULT ep {ep}: {len(fids)} frames {time.time()-t0:.0f}s; ratio obj/person median {np.median(r[ok,1]/r[ok,5]):.3f} "
              f"(p10 {np.percentile(r[ok,1]/r[ok,5],10):.3f} p90 {np.percentile(r[ok,1]/r[ok,5],90):.3f}); frames w/o object {int((~np.isfinite(r[:,1])).sum())}", flush=True)

if __name__ == "__main__":
    main()
