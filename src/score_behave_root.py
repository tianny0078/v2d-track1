#!/usr/bin/env python3
"""Score an export dir from work/behave_root (episode_00000k.* for BEHAVE clip k) with src/score_behave.py,
which expects <pred>/<seq>/episode_000000.*. Builds the symlink layout from behave_root/episode_map.json.
Usage: score_behave_root.py <export dir under behave_root, e.g. work/sub_flow> [more dirs...]"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; BR = ROOT / "work/behave_root"
emap = json.load(open(BR / "episode_map.json"))
for d in sys.argv[1:]:
    src = (BR / d).resolve(); lay = ROOT / "work/behave_pred" / src.name
    for k, seq in emap.items():
        o = lay / seq; o.mkdir(parents=True, exist_ok=True)
        for suf in (".npz", "_object.obj"):
            s = src / f"episode_{int(k):06d}{suf}"; t = o / f"episode_000000{suf}"
            if t.is_symlink() or t.exists(): t.unlink()
            if s.exists(): t.symlink_to(s)
    print(f"=== {src.name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src/score_behave.py"), "--pred", str(lay)], cwd=ROOT)
