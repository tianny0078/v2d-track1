#!/usr/bin/env python3
"""Disk pruning for the CARI4D runs (root disk is at 96%).

Per episode, as stages complete:
  stage 04 done                     -> delete depth/moge2/raw.npy (1.3 GB; only 04 reads it; 05/06/07 read aligned.h5)
  stage 06 (init bundle) done       -> delete depth/moge2/aligned.h5, mhr_init/*.sam3d.h5 (07 needs neither)
  stage 07 done AND export written  -> delete inference/init_bundle.pth, refinement_checkpoint.pth,
                                       materialized_inputs.h5, export/<seq>/{images,human_masks,object_masks},
                                       render mp4/jpg. Keep refined.pth, mhr_init/<seq>.pkl, foundationpose/<seq>.pkl,
                                       export/<seq>/{edex,wild_export.json,object_mesh}, timings.json.
--loop: every 2 min export episodes whose stage 07 is done (src/cari4d_export.py, GPU --gpu) and prune.
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work/v2d_official/runs"
STATE = ROOT / "state/cari4d"
CC = False
SUB = ROOT / "work/sub_cari4d"  # overridden by --sub


def _rm(p: Path) -> int:
    if not p.exists():
        return 0
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    shutil.rmtree(p) if p.is_dir() else p.unlink()
    return size


def _refined_is_current(run: Path) -> bool:
    """With CoCoNet: refined.pth must postdate coconet.pth (07 rerun on the official bundle)."""
    if not CC:
        return True
    cc, rf = run / "inference/coconet.pth", run / "inference/refined.pth"
    return cc.exists() and rf.exists() and rf.stat().st_mtime > cc.stat().st_mtime


def prune(ep: int) -> int:
    seq = f"ep{ep:06d}"; run = RUNS / seq; freed = 0
    if not run.exists():
        return 0
    if (STATE / "04_depth_alignment" / f"{seq}.done").exists() and (not CC or _refined_is_current(run)):  # cc mode: stale first-pass markers must not prune step-06 inputs
        freed += _rm(run / "depth/moge2/raw.npy")
    if ((STATE / "06_init_bundle" / f"{seq}.done").exists() or (STATE / "06_coconet" / f"{seq}.done").exists()) and (not CC or _refined_is_current(run)):
        freed += _rm(run / "depth/moge2/aligned.h5") + _rm(run / "mhr_init" / f"{seq}.sam3d.h5")
    exported = (SUB / f"episode_{ep:06d}.npz").exists() and (SUB / f"episode_{ep:06d}_object.obj").exists()
    if (STATE / "07_refinement" / f"{seq}.done").exists() and (run / "inference/refined.pth").exists() and exported and _refined_is_current(run):
        for rel in ("inference/init_bundle.pth", "inference/refinement_checkpoint.pth", "inference/materialized_inputs.h5",
                    f"export/{seq}/images", f"export/{seq}/human_masks", f"export/{seq}/object_masks", "foundationpose/debug"):
            freed += _rm(run / rel)
        for f in list((run / "inference").glob("*.mp4")) + list((run / "inference").glob("*.jpg")):
            freed += _rm(f)
    return freed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--keep-render", type=int, nargs="*", default=[14], help="episodes whose render videos are kept")
    ap.add_argument("--sub", default="work/sub_cari4d", help="export dir")
    ap.add_argument("--cc", action="store_true", help="CoCoNet mode: only export/prune episodes whose refined.pth postdates coconet.pth")
    a = ap.parse_args()
    global SUB, CC
    SUB = ROOT / a.sub; CC = a.cc
    while True:
        todo = [ep for ep in range(30) if (STATE / "07_refinement" / f"ep{ep:06d}.done").exists() and not (SUB / f"episode_{ep:06d}.npz").exists() and _refined_is_current(RUNS / f"ep{ep:06d}")]
        if todo:
            subprocess.run([sys.executable, str(ROOT / "src/cari4d_export.py"), "--eps", *map(str, todo), "--device", "cuda", "--out", str(SUB.relative_to(ROOT))], env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": a.gpu})
        total = 0
        for ep in range(30):
            if ep in a.keep_render:
                run = RUNS / f"ep{ep:06d}"
                for f in list((run / "inference").glob("*.mp4")) + list((run / "inference").glob("*.jpg")):
                    f.rename(f.with_suffix(f.suffix + ".keep")) if not f.name.endswith(".keep") else None
            total += prune(ep)
            if ep in a.keep_render:
                for f in (RUNS / f"ep{ep:06d}" / "inference").glob("*.keep"):
                    f.rename(f.with_name(f.name[:-5]))
        free = shutil.disk_usage("/").free / 2**30
        print(f"{time.strftime('%H:%M:%S')} pruned {total/2**30:.2f} GB; exported {todo}; root free {free:.0f} GB", flush=True)
        if not a.loop:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
