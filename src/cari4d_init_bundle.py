#!/usr/bin/env python3
"""Build a CARI4D refinement bundle from the INITIALIZATION (no CoCoNet).

The official step 06 (tools/run_mhr_wild_inference.py) needs the gated nvidia/cari4d_commercial
checkpoint, which this machine has no Hugging Face token for. Step 07
(learning.training.mhr_opt_refineout, smplh_parity mode) only consumes from the step-06 bundle:
  pr.pose_abs [T,4,4]      object pose in the aligned-mesh frame (= FoundationPose obj_pose_world,
                           the wild export's storage_to_training transform is the identity)
  pr.mhr_* (7 blocks)      MHR parameters (here: the SAM 3D Body initialization from step 03)
  pr.contact_logits [T,2]  CoCoNet's per-hand contact logits; step 07 keeps a hand only when
                           logit > 0 AND its initial hand-surface-to-object distance < 0.05 m.
                           Without the network we set logit = +1 iff that geometric test passes,
                           so the activation is the geometric gate alone.
  observations.postopt_*   256x256 full-resolution crops of the masks + ROI intrinsics, built with
                           the same lib_mhr.postopt_crop helpers step 06 uses.
Everything else in the official bundle is provenance that step 07 copies through untouched.

Run with the CARI4D PYTHONPATH (src/cari4d_run.py sets it) from the cari4d source root.
"""
from __future__ import annotations
import argparse, copy, json, os
from pathlib import Path

import joblib
import numpy as np
import torch

from lib_mhr import MHRLayer, MHR_PARAM_DIMS, assert_mhr_schema, load_mhr_result
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from lib_mhr.postopt_crop import build_postopt_crop, stack_postopt_crops
from learning.training.mhr_opt_refineout import DEFAULT_HAND_SURFACE_SPEC, _point_triangle_surface_distances
from lib_mhr.object_texture import concatenate_object_mesh_parts, load_object_mesh_parts
from prep.mhr_export_utils import camera_calibration, frame_names, load_edex, read_mask, resolve_object_mesh_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-seq", required=True)
    ap.add_argument("--mhr-init", required=True)
    ap.add_argument("--foundationpose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact-distance-m", type=float, default=0.05)
    a = ap.parse_args()
    export_seq = Path(a.export_seq)
    names = frame_names(export_seq, 0)
    init = load_mhr_result(a.mhr_init)
    assert_mhr_schema(init, require_geometry=True)
    assert [str(v) for v in init["frames"]] == names
    fp = joblib.load(a.foundationpose)
    assert [str(v) for v in fp["frames"]] == names
    poses = np.asarray(fp["obj_pose_world"], dtype=np.float32)
    K, w2c = camera_calibration(load_edex(export_seq), 0)
    assert np.allclose(w2c, np.eye(4), atol=1e-6)
    T = len(names)
    params = {k: np.asarray(init[k], dtype=np.float32) for k in MHR_PARAM_DIMS}

    # geometric contact gate (same hand-surface spec, same triangle distance as step 07)
    device = torch.device("cuda")
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ.get("MHR_ASSETS_ROOT"), device="cuda")
    faces = layer.mesh_faces(device=device)
    faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    spec = load_mhr_hand_surface_spec(str(DEFAULT_HAND_SURFACE_SPEC), faces=faces_np)
    hand_idx = torch.as_tensor(np.asarray(spec.vertex_indices, dtype=np.int64), device=device)
    mesh = concatenate_object_mesh_parts(load_object_mesh_parts(resolve_object_mesh_path(export_seq)))
    ov = torch.as_tensor(np.asarray(mesh.vertices, np.float32), device=device)
    of = torch.as_tensor(np.asarray(mesh.faces, np.int64), device=device)
    dists = np.zeros((T, 2), np.float32)
    with torch.no_grad():
        for s in range(0, T, 32):
            e = min(T, s + 32)
            p = {k: torch.as_tensor(v[s:e], device=device) for k, v in params.items()}
            verts = layer.mhr_forward(p).vertices
            hands = verts[:, hand_idx, :]  # [t,2,S,3]
            R = torch.as_tensor(poses[s:e, :3, :3], device=device); t = torch.as_tensor(poses[s:e, :3, 3], device=device)
            hands_obj = (hands - t[:, None, None, :]) @ R[:, None, :, :]
            d = _point_triangle_surface_distances(hands_obj, ov, of).min(dim=-1).values
            dists[s:e] = d.cpu().numpy()
    contact_logits = np.where(dists < a.contact_distance_m, 1.0, -1.0).astype(np.float32)

    crops = [build_postopt_crop(read_mask(export_seq, "human", 0, n), read_mask(export_seq, "object", 0, n), K) for n in names]
    observations = stack_postopt_crops(crops, K)
    pr = {"pose_abs": poses.copy(), "pose_abs_1st_delta": poses.copy(), **{k: v.copy() for k, v in params.items()}, "contact_logits": contact_logits}
    bundle = {
        "schema": "cari4d.mhr_wild_inference.v1-init-only", "frames": names, "frame_meta": [{"frame": n, "src_frame": int(n), "kid": 0} for n in names], "kid": 0,
        "checkpoint": {"path": None, "step": 0}, "faces": faces_np.astype(np.int32),
        "pr": pr, "pr_initial": copy.deepcopy(pr), "in": {"pose_abs": poses.copy(), **{k: v.copy() for k, v in params.items()}}, "gt": {},
        "observations": dict(observations), "raw": {},
        "metadata": {"ground_truth_used": False, "coconet": "skipped: gated checkpoint unavailable", "contact_logits_source": f"geometric: hand-surface distance < {a.contact_distance_m} m at initialization",
                     "mhr_init_source": str(a.mhr_init), "foundationpose_source": str(a.foundationpose), "object_mesh": str(resolve_object_mesh_path(export_seq)),
                     "object_pose_frame": "centered_axis_aligned", "contact_frames_left_right": [int((contact_logits[:, 0] > 0).sum()), int((contact_logits[:, 1] > 0).sum())],
                     "hand_object_distance_min_m": [float(dists[:, 0].min()), float(dists[:, 1].min())]},
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out, pickle_protocol=4)
    print(json.dumps({"out": str(out), "frames": T, **bundle["metadata"]}, indent=1))


if __name__ == "__main__":
    main()
