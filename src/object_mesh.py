#!/usr/bin/env python3
"""One reconstructed mesh per OBJECT (10), not per episode (30).

Each object appears in exactly three episodes, and the same physical object has the same
geometry in all three, so a single mesh is reconstructed from the single best view found across
all three and reused. That removes the shape variance between an object's three episodes and
gives CD-O one scale to get right instead of three.

View selection matters more than anything else here: the model sees ONE image. We score every
scored frame of all three episodes and keep the view that is large, unoccluded by the hands, and
clear of the image border.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S3O = Path("/mnt/nas3/simfoundry-deps/sam-3d-objects")
sys.path.insert(0, str(ROOT / "work/shims"))
sys.path.insert(0, str(S3O))
MASKS, FRAMES = ROOT / "work/masks", ROOT / "work/frames"
OUT, STATE = ROOT / "work/meshes", ROOT / "state/objmesh"
PAD = 1.25


def view_score(obj: np.ndarray, person: np.ndarray, H: int, W: int) -> tuple[float, dict]:
    """Bigger, less hand-occluded, further from the border is better.

    Occlusion is read off the object mask's own convexity: a hand across the object splits it or
    bites a hole, so (area / convex-hull-area) drops. That needs no hand detection and no GT.
    """
    ys, xs = np.where(obj)
    if len(xs) < 200:
        return -1.0, {}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    area = len(xs)
    bbox_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    border = min(x0, y0, W - 1 - x1, H - 1 - y1)
    if border < 4:                      # clipped by the frame edge: the shape is incomplete
        return -1.0, {}
    # person pixels sitting inside the object's bbox are the occlusion proxy
    occl = float(person[y0:y1 + 1, x0:x1 + 1].sum()) / bbox_area
    fill = area / bbox_area
    score = np.sqrt(area) * (1.0 - min(occl, 0.9)) * (0.5 + 0.5 * fill)
    return float(score), dict(area=int(area), occl=round(occl, 3), fill=round(fill, 3),
                              bbox=[int(x0), int(y0), int(x1), int(y1)])


def best_view(episodes: list[int]) -> dict:
    best = dict(score=-1.0)
    for ep in episodes:
        z = np.load(MASKS / f"ep{ep:06d}.npz")
        H, W = z["shape"]; fids = z["frame_ids"]; packed = z["packed"]
        spans = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]["spans"]
        scored = np.concatenate([np.arange(a, b + 1) for a, b in spans])
        # every 3rd scored frame: neighbouring frames are near-duplicates
        for f in scored[::3]:
            i = int(np.searchsorted(fids, f))
            if i >= len(fids) or fids[i] != f:
                continue
            m = np.unpackbits(packed[i], axis=-1)[:, :, :W].astype(bool)
            s, info = view_score(m[0], m[1], H, W)
            if s > best["score"]:
                best = dict(score=round(s, 1), ep=int(ep), frame=int(f), **info)
    return best


def crop(ep: int, frame: int, bbox: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    z = np.load(MASKS / f"ep{ep:06d}.npz")
    H, W = z["shape"]; fids = z["frame_ids"]
    i = int(np.searchsorted(fids, frame))
    m = np.unpackbits(z["packed"][i], axis=-1)[0, :, :W].astype(bool)
    img = np.asarray(Image.open(FRAMES / f"ep{ep:06d}/{frame:06d}.jpg").convert("RGB"))
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * PAD / 2
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(W, cx + half)), int(min(H, cy + half))
    return img[b:d, a:c], m[b:d, a:c]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*")
    ap.add_argument("--device", default=None)
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()

    meta = [json.loads(l) for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()]
    by_obj: dict[str, list[int]] = {}
    for r in meta:
        by_obj.setdefault(r["object"], []).append(r["episode_index"])
    names = a.objects or sorted(by_obj)

    OUT.mkdir(parents=True, exist_ok=True); STATE.mkdir(parents=True, exist_ok=True)
    views = {}
    for name in names:
        v = best_view(by_obj[name])
        views[name] = v
        print(f"{name:<20} ep{v['ep']:02d} frame {v['frame']:>4}  area={v['area']:>6} "
              f"occl={v['occl']:.2f} fill={v['fill']:.2f} score={v['score']}", flush=True)
    (ROOT / "work/best_views.json").write_text(json.dumps(views, indent=1))
    if a.select_only:
        return

    import s3o_compat  # noqa: F401  -- must precede sam3d_objects
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from PIL import Image
    import trimesh

    if a.device:
        torch.cuda.set_device(int(a.device.split(":")[-1]))
    cfg = OmegaConf.load(S3O / "checkpoints/hf/pipeline.yaml")
    cfg.rendering_engine = "pytorch3d"   # nvdiffrast is not built here
    cfg.compile_model = False
    cfg.workspace_dir = str(S3O / "checkpoints/hf")
    pipe = instantiate(cfg)

    for name in names:
        v = views[name]
        rgb, mask = crop(v["ep"], v["frame"], v["bbox"])
        rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], -1)
        out = pipe.run(rgba, None, 42, stage1_only=False, with_mesh_postprocess=False,
                       with_texture_baking=False, with_layout_postprocess=False,
                       use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
        mesh = out["mesh"]
        if isinstance(mesh, (list, tuple)):      # the pipeline returns one entry per batch item
            mesh = mesh[0]
        def _np(x):
            return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
        for attr in ("vertices", "verts"):
            if hasattr(mesh, attr):
                verts = _np(getattr(mesh, attr)); break
        else:
            raise SystemExit(f"unexpected mesh object {type(mesh)}: {dir(mesh)[:40]}")
        faces = _np(getattr(mesh, "faces", None) if hasattr(mesh, "faces") else mesh.faces_list()[0])
        tm = trimesh.Trimesh(verts, faces, process=False)
        tm.export(OUT / f"{name}.glb")
        Image.fromarray(rgba, "RGBA").save(OUT / f"{name}_input.png")
        rec = dict(object=name, **v, vertices=int(len(tm.vertices)), faces=int(len(tm.faces)),
                   extent=[round(float(x), 4) for x in tm.extents], watertight=bool(tm.is_watertight))
        (STATE / f"{name}.done").write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
