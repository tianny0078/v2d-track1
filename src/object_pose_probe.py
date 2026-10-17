#!/usr/bin/env python3
"""Re-run SAM 3D Objects on one best view and dump what the pipeline returns besides the mesh:
the layout pose (rotation / translation / scale in the input camera frame) is what we need to
orient the object in the video (our exports have used identity rotation everywhere)."""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import object_mesh as om
import s3o_compat  # noqa
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
import trimesh
name = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:3"
torch.cuda.set_device(int(dev.split(":")[-1]))
views = json.load(open(ROOT / "work/best_views.json")); v = views[name]
cfg = OmegaConf.load(om.S3O / "checkpoints/hf/pipeline.yaml"); cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False; cfg.workspace_dir = str(om.S3O / "checkpoints/hf")
pipe = instantiate(cfg)
rgb, mask = om.crop(v["ep"], v["frame"], v["bbox"]); rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], -1)
out = pipe.run(rgba, None, 42, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
def show(x, depth=0):
    if hasattr(x, "shape"): return f"tensor{tuple(x.shape)}"
    if isinstance(x, dict): return {k: show(val, depth + 1) for k, val in x.items()} if depth < 2 else "dict"
    if isinstance(x, (list, tuple)): return [show(i, depth + 1) for i in x[:4]]
    return type(x).__name__
print("KEYS", json.dumps(show(out), default=str)[:3000])
for k in out:
    if any(s in k.lower() for s in ("pose", "rot", "quat", "trans", "scale", "layout", "cam")):
        val = out[k]; val = val[0] if isinstance(val, (list, tuple)) else val
        try: print("VAL", k, np.asarray(val.detach().cpu() if hasattr(val, "detach") else val).round(4).tolist())
        except Exception as e: print("VAL", k, type(val), e)
mesh = out["mesh"]; mesh = mesh[0] if isinstance(mesh, (list, tuple)) else mesh
verts = np.asarray((mesh.vertices if hasattr(mesh, "vertices") else mesh.verts).detach().cpu() if hasattr(getattr(mesh, "vertices", None), "detach") else getattr(mesh, "vertices", None))
old = trimesh.load(ROOT / f"work/meshes/{name}.glb", force="mesh")
print("MESH new verts", verts.shape, "old", old.vertices.shape, "max|dv|", float(np.abs(verts - old.vertices).max()) if verts.shape == old.vertices.shape else "shape differs")
