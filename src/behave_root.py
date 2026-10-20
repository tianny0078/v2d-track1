#!/usr/bin/env python3
"""Build an isolated run root for the BEHAVE validation clips so the UNCHANGED pipeline code runs on
them: work/behave_root/{src,data,work,state,logs}. src/ is a copy of the project's src (ROOT is
derived from __file__, so a copy -- not a symlink -- makes every script resolve to behave_root);
models/kit/flowhmr/... are symlinked. Clip k (sorted sequence names) becomes episode k.

Frames are resized 2048x1536 -> 1536x1152 (exactly x0.75, same 4:3 aspect as the challenge) and the
intrinsics scaled to match; all frames of a clip form one scored stretch.
Re-run with --sync-src after changing code in src/ to refresh the copy."""
import argparse, json, math, shutil, subprocess
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / "work/behave_root"
S = 0.75
W, H = 1536, 1152

def sync_src():
    (BR / "src").mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", f"{ROOT}/src/", f"{BR}/src/"], check=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sync-src", action="store_true"); a = ap.parse_args()
    sync_src()
    if a.sync_src:
        print("src synced"); return
    for d in ("data/hf/track_1/meta", "work/camera", "work/frames", "state", "logs"):
        (BR / d).mkdir(parents=True, exist_ok=True)
    for name in ("models", "kit", "flowhmr", "sam3dbody_code", "shims"):
        link = BR / "work" / name
        if not link.exists() and (ROOT / "work" / name).exists():
            link.symlink_to(ROOT / "work" / name)
    seqs = sorted(p.name for p in (ROOT / "work/behave").iterdir() if (p / "meta.json").exists())
    meta_l, eps_l, intr, scored, ground, mapping = [], [], {}, {}, {}, {}
    for k, seq in enumerate(seqs):
        m = json.load(open(ROOT / f"work/behave/{seq}/meta.json"))
        src_frames = sorted((ROOT / f"work/behave/{seq}/frames").glob("*.jpg"))
        out = BR / f"work/frames/ep{k:06d}"; out.mkdir(parents=True, exist_ok=True)
        for f in src_frames:
            dst = out / f.name
            if not dst.exists():
                Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR).save(dst, quality=95)
        n = len(src_frames); cam = f"behave_{seq}"
        fx, fy = m["fx"] * S, m["fy"] * S
        cx, cy = (m["cx"] + 0.5) * S - 0.5, (m["cy"] + 0.5) * S - 0.5
        intr[cam] = dict(episodes=[k], fx=fx, fy=fy, cx=cx, cy=cy, fov_x_deg=round(math.degrees(2 * math.atan(W / 2 / fx)), 2), source="BEHAVE calibration x0.75")
        meta_l.append(dict(episode_index=k, sequence_id=seq, camera=cam, object=m["object"], object_prompt=m["object_prompt"], video_key="behave"))
        eps_l.append(dict(episode_index=k, length=n, tasks=[f"BEHAVE {seq}"]))
        scored[str(k)] = dict(n=n, first=0, last=n - 1, stretches=1, spans=[[0, n - 1]])
        ground[cam] = dict(normal=[0.0, -1.0, 0.0], offset=1.0, note="placeholder; only used by --reground")
        mapping[k] = seq
        print(f"ep{k} <- {seq}: {n} frames, fx {fx:.1f}", flush=True)
    (BR / "data/hf/track_1/meta/episodes_metadata.jsonl").write_text("".join(json.dumps(x) + "\n" for x in meta_l))
    (BR / "data/hf/track_1/meta/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in eps_l))
    json.dump(intr, open(BR / "work/camera/intrinsics.json", "w"), indent=1)
    json.dump(ground, open(BR / "work/camera/ground.json", "w"), indent=1)
    json.dump(scored, open(BR / "work/scored_frames_summary.json", "w"), indent=1)
    json.dump(mapping, open(BR / "episode_map.json", "w"), indent=1)
    print("behave_root ready:", BR)

if __name__ == "__main__":
    main()
