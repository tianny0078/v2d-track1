#!/usr/bin/env python3
"""Feasibility of keyframe layout detection: run SAM 3D Objects on crops of OTHER frames of an
episode (stage1_only to see if the layout head is in stage 1 and how fast), apply the returned
rotation (wxyz^T, R_ray) to the episode's v2 mesh and score silhouette IoU against the mask."""
import sys, json, time, numpy as np, trimesh
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import object_mesh as om
import s3o_compat  # noqa
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation as Rot
from object_rot_convention import silhouette
ep = int(sys.argv[1]); frames = [int(x) for x in sys.argv[2].split(",")]; dev = sys.argv[3] if len(sys.argv) > 3 else "cuda:3"
torch.cuda.set_device(int(dev.split(":")[-1]))
K = json.load(open(ROOT / "work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
k = K[meta[ep]["camera"]]; name = meta[ep]["object"]
mesh = trimesh.load(ROOT / f"work/meshes_v2/{name}.glb", force="mesh"); V = np.asarray(mesh.vertices); Fc = np.asarray(mesh.faces)
if len(Fc) > 30000: Fc = Fc[np.random.RandomState(0).choice(len(Fc), 30000, replace=False)]
cfg = OmegaConf.load(om.S3O / "checkpoints/hf/pipeline.yaml"); cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False; cfg.workspace_dir = str(om.S3O / "checkpoints/hf")
pipe = instantiate(cfg)
z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); W = int(z["shape"][1]); fids = z["frame_ids"]
def npy(x): x = x[0] if isinstance(x, (list, tuple)) else x; return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
for f in frames:
    i = int(np.searchsorted(fids, f)); mask = np.unpackbits(z["packed"][i], axis=-1)[0, :, :W].astype(bool); ys, xs = np.where(mask); H = mask.shape[0]
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    rgb, mk = om.crop(ep, f, bbox); rgba = np.concatenate([rgb, (mk.astype(np.uint8) * 255)[..., None]], -1)
    for s1 in (False,):
        t0 = time.time()
        try:
            out = pipe.run(rgba, None, 42, stage1_only=s1, with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
        except Exception as e:
            print(f"RESULT f{f} stage1_only={s1}: ERROR {type(e).__name__}: {str(e)[:120]}"); continue
        dt = time.time() - t0
        if "rotation" not in out:
            print(f"RESULT f{f} stage1_only={s1}: no rotation in output, keys {list(out)[:8]} ({dt:.0f}s)"); continue
        q = npy(out["rotation"]).reshape(-1); R = Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix().T
        ray = np.array([((bbox[0] + bbox[2]) / 2 - k["cx"]) / k["fx"], ((bbox[1] + bbox[3]) / 2 - k["cy"]) / k["fy"], 1.0]); ray /= np.linalg.norm(ray)
        ax = np.cross([0, 0, 1.0], ray); sn = np.linalg.norm(ax); R_ray = Rot.from_rotvec(ax / sn * np.arctan2(sn, ray[2])).as_matrix() if sn > 1e-8 else np.eye(3)
        Rf = R_ray @ R; vr = V @ Rf.T; uv = (xs.mean(), ys.mean()); ext = max(np.ptp(xs), np.ptp(ys))
        best = 0.0
        for zz in (2.0, 3.0):
            vv = vr - vr.mean(0); P0 = vv / max(np.ptp(vv, axis=0).max(), 1e-6); e0 = max(np.ptp(k["fx"] * P0[:, 0] / zz), np.ptp(k["fy"] * P0[:, 1] / zz)); S = ext / max(e0, 1e-6)
            sil = silhouette(vr, Fc, k, uv, zz, S, H, W); best = max(best, (sil & mask).sum() / max((sil | mask).sum(), 1))
        base = 0.0
        for zz in (2.0, 3.0):
            vv = V - V.mean(0); P0 = vv / max(np.ptp(vv, axis=0).max(), 1e-6); e0 = max(np.ptp(k["fx"] * P0[:, 0] / zz), np.ptp(k["fy"] * P0[:, 1] / zz)); S = ext / max(e0, 1e-6)
            sil = silhouette(V, Fc, k, uv, zz, S, H, W); base = max(base, (sil & mask).sum() / max((sil | mask).sum(), 1))
        tz = npy(out["translation"]).reshape(-1)[2] if "translation" in out else float("nan")
        own = float("nan")
        if not s1 and "mesh" in out:
            mo = out["mesh"]; mo = mo[0] if isinstance(mo, (list, tuple)) else mo
            Vo = npy(mo.vertices if hasattr(mo, "vertices") else mo.verts); Fo = npy(mo.faces)
            if len(Fo) > 30000: Fo = Fo[np.random.RandomState(0).choice(len(Fo), 30000, replace=False)]
            vo = Vo @ Rf.T; own = 0.0
            for zz in (2.0, 3.0):
                vv = vo - vo.mean(0); P0 = vv / max(np.ptp(vv, axis=0).max(), 1e-6); e0 = max(np.ptp(k["fx"] * P0[:, 0] / zz), np.ptp(k["fy"] * P0[:, 1] / zz)); S = ext / max(e0, 1e-6)
                sil = silhouette(vo, Fo, k, uv, zz, S, H, W); own = max(own, (sil & mask).sum() / max((sil | mask).sum(), 1))
            # canonical-frame agreement between this run's mesh and the v2 mesh: extents and PCA axes
            ev2 = np.ptp(V, axis=0); eo = np.ptp(Vo, axis=0)
            print(f"   canonical extents v2 {np.round(ev2,3).tolist()} vs own {np.round(eo,3).tolist()}")
        print(f"RESULT f{f} stage1_only={s1}: {dt:.1f}s, rotvec {np.round(Rot.from_matrix(Rf).as_rotvec(), 2).tolist()}, IoU layout(v2 mesh) {best:.3f} own-mesh {own:.3f} vs identity {base:.3f}, t_z {tz:.2f}", flush=True)
