#!/usr/bin/env python3
"""Per-EPISODE object detection: SAM 3D Objects on the episode's own best frame, keeping the
mesh AND its layout pose as a consistent pair. Why per episode: the pipeline's canonical frame is
not stable across runs (pink roll: long axis z in one run, y in another), so a rotation from one
run cannot be applied to another run's mesh -- but each run's (mesh, rotation) pair matches its
image (silhouette IoU 0.72-0.79). One mesh per episode is allowed by the format. The rotation at
the best frame then seeds the per-frame silhouette tracker (object_track.py).
Output: work/objtrack/epNNNNNN_mesh.glb, epNNNNNN_pose.json (R_full in our camera, frame)."""
import sys, json, time, numpy as np, trimesh
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import object_mesh as om
import s3o_compat  # noqa
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation as Rot
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"; eps = [int(x) for x in sys.argv[2:]] or list(range(30))
torch.cuda.set_device(int(dev.split(":")[-1]))
K = json.load(open(ROOT / "work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
OUT = ROOT / "work/objtrack"; OUT.mkdir(exist_ok=True); (ROOT / "state/objdet").mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(om.S3O / "checkpoints/hf/pipeline.yaml"); cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False; cfg.workspace_dir = str(om.S3O / "checkpoints/hf")
pipe = instantiate(cfg)
def npy(x): x = x[0] if isinstance(x, (list, tuple)) else x; return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
for ep in eps:
    t0 = time.time(); v = om.best_view([ep]); k = K[meta[ep]["camera"]]
    rgb, mask = om.crop(ep, v["frame"], v["bbox"]); rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], -1)
    out = pipe.run(rgba, None, 42, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
    mesh = out["mesh"]; mesh = mesh[0] if isinstance(mesh, (list, tuple)) else mesh
    verts = npy(mesh.vertices if hasattr(mesh, "vertices") else mesh.verts); faces = npy(mesh.faces)
    trimesh.Trimesh(verts, faces, process=False).export(OUT / f"ep{ep:06d}_mesh.glb")
    q = npy(out["rotation"]).reshape(-1); R = Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix().T
    b = v["bbox"]; ray = np.array([((b[0] + b[2]) / 2 - k["cx"]) / k["fx"], ((b[1] + b[3]) / 2 - k["cy"]) / k["fy"], 1.0]); ray /= np.linalg.norm(ray)
    ax = np.cross([0, 0, 1.0], ray); sn = np.linalg.norm(ax); R_ray = Rot.from_rotvec(ax / sn * np.arctan2(sn, ray[2])).as_matrix() if sn > 1e-8 else np.eye(3)
    R_full = R_ray @ R
    pose = dict(ep=ep, object=meta[ep]["object"], view=v, rotation_quat_wxyz=q.tolist(), R_full=R_full.tolist(), translation=npy(out["translation"]).reshape(-1).tolist(),
                scale=npy(out["scale"]).reshape(-1).tolist(), vertices=int(len(verts)), faces=int(len(faces)), extent=[float(x) for x in np.ptp(verts, axis=0)])
    (OUT / f"ep{ep:06d}_pose.json").write_text(json.dumps(pose, indent=1)); (ROOT / "state/objdet" / f"ep{ep:06d}.done").touch()
    print(f"RESULT ep{ep} {meta[ep]['object']}: frame {v['frame']} score {v['score']}, {len(verts)} verts, rotvec {np.round(Rot.from_matrix(R_full).as_rotvec(),2).tolist()}, {time.time()-t0:.0f}s", flush=True)
