#!/usr/bin/env python3
"""Local scorer for a Track 1 episode directory.

PEN is EXACT up to one scalar. The host measures our hand points against OUR OWN mesh; the only
reference quantity entering it is the scale of the frame-0 Sim(3) alignment, which is ~1 when our
body scale is right. So PEN is the one metric we can drive to its optimum with no leaderboard
feedback at all -- see notes/00-plan-v2.md section 2.3.

CD-H / CD-O / ACC-H / ACC-O need the hidden reference. Against a `--ref` directory they are
ABLATION SIGNALS (A vs B), never predicted leaderboard scores: our role_body is a proxy for the
host's, so the absolute numbers differ.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
sys.path.insert(0, str(ROOT))

from v2dlb.mhr_submission import penetration_depth, sample_mesh_surface, stretches  # noqa: E402
from v2dlb.mhr_metrics import (acceleration_error_per_frame, apply_similarity_transform,  # noqa: E402
                               fit_similarity_transform, symmetric_chamfer_distance)
from src.mhr_local import MHR, BODY_JOINT_INDICES  # noqa: E402


def load_episode(d: Path, ep: int, frames: np.ndarray, mhr: MHR):
    z = np.load(d / f"episode_{ep:06d}.npz")
    import trimesh
    mesh = trimesh.load(str(next(d.glob(f"episode_{ep:06d}_object.*"))), force="mesh", process=False)
    pose, scales, shape = z["pose"][frames], z["scales"], z["shape"]
    params = np.concatenate([pose, np.broadcast_to(scales, (len(pose), 68))], 1)
    verts, joints = mhr.forward(params, shape)
    return dict(
        verts=verts, joints=joints[:, list(BODY_JOINT_INDICES)],
        rotation=z["object_rotation"][frames].astype(np.float64),
        translation=z["object_translation"][frames].astype(np.float64),
        scale=float(z["object_scale"]),
        mesh_v=np.asarray(mesh.vertices, np.float64), mesh_f=np.asarray(mesh.faces, np.int64),
    )


def penetration(e, mhr: MHR, align: float = 1.0) -> dict:
    """Exact PEN, modulo the frame-0 alignment scale (pass --align once it is known)."""
    hands = e["verts"][:, mhr.hand_vertices].astype(np.float64)
    local = np.einsum("tpi,tij->tpj", hands - e["translation"][:, None], e["rotation"]) / e["scale"]
    depth = penetration_depth(local.reshape(-1, 3), e["mesh_v"], e["mesh_f"]).reshape(hands.shape[:2])
    # host convention: inside is negative, outside contributes zero, denominator is all points
    per_frame = np.maximum(depth, 0.0) * e["scale"] * align * 100.0
    return dict(pen_cm=float(per_frame.mean()), worst_frame_cm=float(per_frame.mean(1).max()),
                frames_with_contact=int((per_frame.mean(1) > 0).sum()),
                max_depth_cm=float(per_frame.max()))


def compare(pred, ref, mhr: MHR, frames: np.ndarray, seed: int) -> dict:
    t = fit_similarity_transform(pred["verts"][0, mhr.alignment_vertices],
                                 ref["verts"][0, mhr.alignment_vertices])
    body = mhr.body_proxy_vertices
    out = {"align_scale": float(t[0])}
    ph = apply_similarity_transform(pred["verts"][:, body], t)
    out["cd_h_cm_proxy"] = float(symmetric_chamfer_distance(ph, ref["verts"][:, body]).mean() * 100)
    po, ro = (sample_mesh_surface(e["mesh_v"], e["mesh_f"], len(body), seed) for e in (pred, ref))
    surf = lambda e, loc: e["scale"] * np.einsum("tij,pj->tpi", e["rotation"], loc) + e["translation"][:, None]
    out["cd_o_cm_proxy"] = float(symmetric_chamfer_distance(
        apply_similarity_transform(surf(pred, po), t), surf(ref, ro)).mean() * 100)
    acc_h, acc_o = [], []
    for sl in stretches(frames):
        if sl.stop - sl.start < 3:
            continue
        acc_h.append(acceleration_error_per_frame(
            apply_similarity_transform(pred["joints"][sl], t), ref["joints"][sl]).mean() * 100)
        acc_o.append(acceleration_error_per_frame(
            apply_similarity_transform(pred["translation"][sl], t), ref["translation"][sl]).mean() * 100)
    out["acc_h_cm_proxy"] = float(np.mean(acc_h)) if acc_h else float("nan")
    out["acc_o_cm_proxy"] = float(np.mean(acc_o)) if acc_o else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--ref", type=Path, help="reference episode dir; enables the CD/ACC proxies")
    ap.add_argument("--align", type=float, default=1.0)
    ap.add_argument("--eps", type=int, nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    scored = json.load((ROOT / "work/scored_frames_summary.json").open())
    sample = _scored_frames()
    mhr = MHR()
    rows = []
    for ep in (a.eps if a.eps else range(30)):
        frames = sample[ep]
        pred = load_episode(a.episodes, ep, frames, mhr)
        r = dict(ep=ep, n_scored=len(frames), stretches=scored[str(ep)]["stretches"])
        r.update(penetration(pred, mhr, a.align))
        if a.ref:
            r.update(compare(pred, load_episode(a.ref, ep, frames, mhr), mhr, frames, seed=ep))
        rows.append(r)
        print(json.dumps(r), flush=True)
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ("ep", "n_scored", "stretches")}
    print("MEAN", json.dumps({k: round(v, 4) for k, v in agg.items()}))
    if a.out:
        a.out.write_text(json.dumps(dict(per_episode=rows, mean=agg), indent=1))


def _scored_frames() -> dict[int, np.ndarray]:
    import pandas as pd
    s = pd.read_parquet(ROOT / "work/kit/v2d_submission_kit/data/track_1_sample_submission.parquet",
                        columns=["row_id"])
    r = s.row_id.str.extract(r"t1_(\d{6})_(\d{6})_4_")
    r = r.dropna().astype(int)
    return {ep: np.sort(g[1].unique()) for ep, g in r.groupby(0)}


if __name__ == "__main__":
    main()
