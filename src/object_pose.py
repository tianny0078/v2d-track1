#!/usr/bin/env python3
"""SAM 3D Objects for the 10 objects, keeping the LAYOUT POSE this time (rotation quaternion,
translation, scale in the crop camera frame). The first run (object_mesh.py) kept only the mesh and
every export has used identity rotation -- the 62 cm foam roller lies across the image while our
mesh stood upright. The pipeline is not bit-reproducible (vertex counts differ run to run), so the
mesh is re-saved next to its pose: work/meshes_v2/<name>.glb + <name>_pose.json."""
import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import object_mesh as om
import s3o_compat  # noqa
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
import trimesh
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"; names = sys.argv[2:]
torch.cuda.set_device(int(dev.split(":")[-1]))
views = json.load(open(ROOT / "work/best_views.json")); names = names or list(views)
OUT = ROOT / "work/meshes_v2"; OUT.mkdir(exist_ok=True); (ROOT / "state/objpose_v2").mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(om.S3O / "checkpoints/hf/pipeline.yaml"); cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False; cfg.workspace_dir = str(om.S3O / "checkpoints/hf")
pipe = instantiate(cfg)
def npy(x): x = x[0] if isinstance(x, (list, tuple)) else x; return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
for name in names:
    t0 = time.time(); v = views[name]
    rgb, mask = om.crop(v["ep"], v["frame"], v["bbox"]); rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], -1)
    out = pipe.run(rgba, None, 42, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
    mesh = out["mesh"]; mesh = mesh[0] if isinstance(mesh, (list, tuple)) else mesh
    verts = npy(mesh.vertices if hasattr(mesh, "vertices") else mesh.verts); faces = npy(mesh.faces)
    tm = trimesh.Trimesh(verts, faces, process=False); tm.export(OUT / f"{name}.glb")
    pose = dict(object=name, view=v, rotation_quat=npy(out["rotation"]).reshape(-1).tolist(), rot6d=npy(out["6drotation_normalized"]).reshape(-1).tolist(),
                translation=npy(out["translation"]).reshape(-1).tolist(), scale=npy(out["scale"]).reshape(-1).tolist(),
                translation_scale=float(npy(out["translation_scale"]).reshape(-1)[0]), extent=[float(x) for x in tm.extents], vertices=int(len(verts)), faces=int(len(faces)),
                crop_hw=[int(rgba.shape[0]), int(rgba.shape[1])])
    np.save(OUT / f"{name}_pointmap.npy", npy(out["pointmap"]))
    (OUT / f"{name}_pose.json").write_text(json.dumps(pose, indent=1)); (ROOT / "state/objpose_v2" / f"{name}.done").touch()
    print(f"RESULT {name}: {len(verts)} verts, quat {np.round(pose['rotation_quat'],3).tolist()}, t {np.round(pose['translation'],3).tolist()}, scale {pose['scale'][0]:.3f}, {time.time()-t0:.0f}s", flush=True)
