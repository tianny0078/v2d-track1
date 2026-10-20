#!/usr/bin/env python3
"""Run the official v2d_cari4d wild-video stages on one of our episodes (no Docker).

Mirrors reconstruction/modules/v2d_cari4d/lib/run_inference.py stage by stage (same scripts,
same flags) but with host paths, our weights layout (work/v2d_official/weights) and two
substitutions that are recorded in notes/cari4d-repro.md:
  * the camera K written by step 01 (MoGe-2's own estimate) is replaced by our per-camera
    intrinsics (work/camera/intrinsics.json) unless --moge-k is given;
  * step 06 (CoCoNet) needs the gated nvidia/cari4d_commercial checkpoint; when it is absent the
    step is skipped and step 07 is run from an initialization bundle (src/cari4d_init_bundle.py).

State: state/cari4d/<stage>/epNNNNNN.done ; logs: logs/cari4d/epNNNNNN_<stage>.log
"""
from __future__ import annotations
import argparse, json, os, pickle, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "work/v2d_official"
MODULES = OFFICIAL / "reconstruction/modules"
SOURCE_ROOT = MODULES / "v2d_cari4d/lib/cari4d"
SAM3D_LIB = MODULES / "v2d_sam3d_body/lib"
FP_DIR = MODULES / "v2d_foundation_pose/lib/FoundationPose"
WEIGHTS = OFFICIAL / "weights"
INPUTS = OFFICIAL / "inputs"
RUNS = OFFICIAL / "runs"
STATE = ROOT / "state/cari4d"
LOGS = ROOT / "logs/cari4d"
CARI4D_CKPT = WEIGHTS / "cari4d/2026-08-25-09-35-57/step200000.pth"
CARI4D_CFG = WEIGHTS / "cari4d/2026-08-25-09-35-57/resolved_config.yaml"
STAGES = ["01_depth", "02_export", "03_human_initialization", "04_depth_alignment", "05_object_pose", "06_coconet", "07_refinement", "08_render"]


def env_for(gpu: str) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join([str(OFFICIAL / "shims"), str(SOURCE_ROOT), str(SAM3D_LIB), str(FP_DIR)]),
        "HF_HOME": str(WEIGHTS / "hf_home"), "TORCH_HOME": str(WEIGHTS / "sam3d_body/torch_home"),
        "SAM3D_BODY_ROOT": str(SAM3D_LIB), "MHR_ASSETS_ROOT": str(WEIGHTS / "sam3d_body"),
        "FOUNDATIONPOSE_WEIGHTS_DIR": str(WEIGHTS / "foundationpose/nvlabs_pytorch"),
        "OPENCV_IO_ENABLE_OPENEXR": "1", "PYOPENGL_PLATFORM": "egl", "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONUNBUFFERED": "1",
        "CUDA_HOME": "/usr/local/cuda-12.8", "TORCH_CUDA_ARCH_LIST": "12.0", "CUDA_VISIBLE_DEVICES": gpu,
        "MOMENTUM_ENABLED": "0", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env["PATH"]
    return env


def run_stage(name: str, cmd: list[str], ep: int, env: dict, force: bool) -> float:
    seq = f"ep{ep:06d}"
    marker = STATE / name / f"{seq}.done"
    if marker.exists() and not force:
        print(f"{seq} {name}: done (skip)", flush=True); return 0.0
    marker.parent.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{seq}_{name}.log"
    print(f"{seq} {name}: start -> {log}", flush=True)
    t0 = time.time()
    with log.open("w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, cwd=SOURCE_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{seq} {name} failed (rc {r.returncode}, {dt:.0f}s): see {log}")
    marker.write_text(json.dumps(dict(ep=ep, stage=name, seconds=round(dt, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))))
    print(f"{seq} {name}: ok {dt:.0f}s", flush=True)
    return dt


def patch_intrinsics(pkl: Path, K: dict) -> None:
    """Replace MoGe's K (first-frame estimate) by our calibrated per-camera K; keep MoGe's record."""
    p = pickle.load(pkl.open("rb"))
    if p.get("intrinsics_substituted"):
        return
    p["moge_estimate"] = {k: p[k] for k in ("fx", "fy", "cx", "cy")}
    p.update(fx=float(K["fx"]), fy=float(K["fy"]), cx=float(K["cx"]), cy=float(K["cy"]), intrinsics_substituted="work/camera/intrinsics.json")
    tmp = pkl.with_suffix(".tmp")
    pickle.dump(p, tmp.open("wb")); os.replace(tmp, pkl)
    print(f"  K: moge fx {p['moge_estimate']['fx']:.1f} cx {p['moge_estimate']['cx']:.1f} cy {p['moge_estimate']['cy']:.1f} -> ours fx {K['fx']} cx {K['cx']} cy {K['cy']}", flush=True)


def run_episode(ep: int, gpu: str, stages: list[str], force: bool, moge_k: bool, render: bool, steps: int) -> None:
    seq = f"ep{ep:06d}"
    inp = INPUTS / seq
    meta = json.load((inp / "meta.json").open())
    video = inp / f"{seq}.0.color.mp4"; mask_h5 = inp / f"{seq}_masks_k0.h5"; mesh = inp / "object_metric.glb"
    out = RUNS / seq
    env = env_for(gpu)
    py = sys.executable
    raw_depth = out / "depth/moge2/raw.npy"; depth_intr = out / "depth/moge2/intrinsics.pkl"; depth_report = out / "depth/moge2/depth_report.json"
    export_root = out / "export"; export_seq = export_root / seq
    mhr_init = out / f"mhr_init/{seq}.pkl"; sam3d_cache = out / f"mhr_init/{seq}.sam3d.h5"
    aligned_depth = out / "depth/moge2/aligned.h5"
    fp_init = out / f"foundationpose/{seq}.pkl"
    coconet_bundle = out / "inference/coconet.pth"; input_cache = out / "inference/materialized_inputs.h5"
    init_bundle = out / "inference/init_bundle.pth"
    refined = out / "inference/refined.pth"; refine_ckpt = out / "inference/refinement_checkpoint.pth"
    for p in (raw_depth, mhr_init, aligned_depth, fp_init, coconet_bundle, refined):
        p.parent.mkdir(parents=True, exist_ok=True)
    timings = {}
    sam3d_ckpt = WEIGHTS / "sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt"; mhr_model = WEIGHTS / "sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    aligned_mesh = export_seq / "object_mesh/output_aligned.glb"

    if "01_depth" in stages:
        cmd = [py, str(SOURCE_ROOT / "prep/mhr_wild_depth.py"), "--video", str(video), "--output-depth", str(raw_depth), "--output-intrinsics", str(depth_intr),
               "--output-report", str(depth_report), "--device", "cuda", "--batch-size", "8", "--local-files-only", "--expected-frames", str(meta["n_frames"]), "--overwrite"]
        timings["01"] = run_stage("01_depth", cmd, ep, env, force)
        if not moge_k:
            patch_intrinsics(depth_intr, meta["K"])
    if "02_export" in stages:
        cmd = [py, str(SOURCE_ROOT / "prep/prepare_mhr_wild_export.py"), "--video", str(video), "--mask-h5", str(mask_h5), "--object-mesh", str(mesh),
               "--intrinsics-file", str(depth_intr), "--output-root", str(export_root), "--redo"]
        timings["02"] = run_stage("02_export", cmd, ep, env, force)
    if "03_human_initialization" in stages:
        cmd = [py, str(SOURCE_ROOT / "prep/run_sam3d_mhr_export.py"), str(export_seq), "--out-file", str(mhr_init), "--camera-id", "0", "--depth-root", str(export_seq),
               "--sam3d-ckpt", str(sam3d_ckpt), "--mhr-path", str(mhr_model), "--chunk-size", "16", "--sam3d-cache", str(sam3d_cache), "--alignment-workers", "1",
               "--refit-batch-size", "512", "--human-prompt-mode", "sam2_bbox_mask", "--no-align-to-gt-depth", "--redo"]
        timings["03"] = run_stage("03_human_initialization", cmd, ep, env, force)
    if "04_depth_alignment" in stages:
        cmd = [py, str(SOURCE_ROOT / "prep/align_depth_to_mhr_wild.py"), str(export_seq), "--raw-depth", str(raw_depth), "--depth-report", str(depth_report),
               "--mhr-init", str(mhr_init), "--output", str(aligned_depth), "--render-batch-size", "4", "--encoding-workers", "8", "--write-batch-size", "64", "--redo"]
        timings["04"] = run_stage("04_depth_alignment", cmd, ep, env, force)
    if "05_object_pose" in stages:
        cmd = [py, str(SOURCE_ROOT / "prep/run_foundationpose_mhr_export.py"), str(export_seq), "--depth-root", str(aligned_depth), "--out-file", str(fp_init), "--camera-id", "0",
               "--iteration", "5", "--max-attempts", "5", "--register-first-then-track", "--no-first-usable-frame-gt-rotation-oracle", "--redo"]
        timings["05"] = run_stage("05_object_pose", cmd, ep, env, force)
    have_ckpt = CARI4D_CKPT.is_file() and CARI4D_CFG.is_file()
    if "06_coconet" in stages:
        if have_ckpt:
            cmd = [py, str(SOURCE_ROOT / "tools/run_mhr_wild_inference.py"), str(export_seq), "--depth-h5", str(aligned_depth), "--mhr-init", str(mhr_init), "--foundationpose-file", str(fp_init),
                   "--config", str(CARI4D_CFG), "--checkpoint", str(CARI4D_CKPT), "--output", str(coconet_bundle), "--stride", "96", "--render-batch-size", "32", "--crop-workers", "8",
                   "--crop-buffer-count", "2", "--input-cache", str(input_cache), "--device", "cuda", "--offline-supervision-contract", "--overwrite"]
            timings["06"] = run_stage("06_coconet", cmd, ep, env, force)
        else:
            cmd = [py, str(ROOT / "src/cari4d_init_bundle.py"), "--export-seq", str(export_seq), "--mhr-init", str(mhr_init), "--foundationpose", str(fp_init), "--out", str(init_bundle)]
            timings["06"] = run_stage("06_init_bundle", cmd, ep, env, force)
    bundle = coconet_bundle if have_ckpt else init_bundle
    if "07_refinement" in stages:
        cmd = [py, "-m", "learning.training.mhr_opt_refineout", "--bundle", str(bundle), "--object-mesh", str(aligned_mesh), "--out", str(refined), "--mode", "smplh_parity",
               "--num-steps", str(steps), "--batch-size", "0", "--device", "cuda", "--postopt-checkpoint", str(refine_ckpt), "--mhr-assets-root", str(WEIGHTS / "sam3d_body"),
               "--w-temporal", "100.0", "--w-human-pose-prior", "200.0", "--contact-activation-distance-m", "0.05", "--report-every", "10", "--diagnostics-every", "500"]
        if refine_ckpt.exists() and force:
            refine_ckpt.unlink()
        timings["07"] = run_stage("07_refinement", cmd, ep, env, force)
    if "08_render" in stages and render:
        cmd = [py, str(SOURCE_ROOT / "tools/render_mhr_wild_inference.py"), str(export_seq), "--source-video", str(video), "--object-mesh", str(aligned_mesh), "--before-bundle", str(bundle),
               "--after-bundle", str(refined), "--output", str(out / f"inference/{seq}_reconstruction.mp4"), "--comparison-output", str(out / f"inference/{seq}_before_after.mp4"), "--device", "cuda", "--batch-size", "4", "--overwrite"]
        timings["08"] = run_stage("08_render", cmd, ep, env, force)
    (out / "timings.json").write_text(json.dumps({**(json.loads((out / "timings.json").read_text()) if (out / "timings.json").exists() else {}), **{k: round(v, 1) for k, v in timings.items() if v}}, indent=1))
    print(f"{seq}: timings {timings}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", required=True)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", action="store_true", help="ignore done markers")
    ap.add_argument("--moge-k", action="store_true", help="keep MoGe-2's own intrinsics (official default) instead of ours")
    ap.add_argument("--render", action="store_true", help="also run step 08 (videos)")
    ap.add_argument("--steps", type=int, default=300, help="refinement steps (official 300)")
    a = ap.parse_args()
    stages = [s for s in STAGES if s in a.stages.split(",") or s[:2] in a.stages.split(",")]
    for ep in a.eps:
        try:
            run_episode(ep, a.gpu, stages, a.force, a.moge_k, a.render, a.steps)
        except Exception as exc:  # keep going with the other episodes
            print(f"ep{ep:06d}: FAILED {exc}", flush=True)


if __name__ == "__main__":
    main()
