#!/usr/bin/env python3
"""Object-centroid distance to the nearest wrist / pelvis and translation |a|, per episode, for
one or more export dirs. A cheap placement sanity signal: the scored span is the contact span."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mhr_local import MHR
from src.export_episodes import _sam3d_as_body
from src.compare_ref import load_obj
sys.path.insert(0, "work/kit/v2d_submission_kit")
from v2dlb.mhr_submission import stretches
dirs = [d for d in sys.argv[1:] if not d.isdigit()]; eps = [int(e) for e in sys.argv[1:] if e.isdigit()] or list(range(30))
mhr = MHR("cuda:2")
for ep in eps:
    fids = _sam3d_as_body(Path(f"work/sam3d/ep{ep:06d}.npz"))["frame_ids"]; line = f"ep{ep:2d}"
    for d in dirs:
        p = Path(d) / f"episode_{ep:06d}.npz"
        if not p.exists(): line += f" | {d}: -"; continue
        z = np.load(p); pp = np.concatenate([z["pose"][fids], np.broadcast_to(z["scales"], (len(fids), 68))], 1); _, j = mhr.forward(pp, z["shape"])
        m = load_obj(Path(d) / f"episode_{ep:06d}_object.obj"); c = np.einsum("tij,j->ti", z["object_rotation"][fids], m.mean(0) * float(z["object_scale"])) + z["object_translation"][fids]
        dw = np.minimum(np.linalg.norm(c - j[:, 78], axis=-1), np.linalg.norm(c - j[:, 42], axis=-1)) * 100
        dp = np.linalg.norm(c - j[:, 1], axis=-1) * 100
        acc = np.mean([np.linalg.norm(c[s][:-2] - 2 * c[s][1:-1] + c[s][2:], axis=-1).mean() for s in stretches(fids) if s.stop - s.start >= 3]) * 100
        line += f" | {Path(d).name}: z {np.median(c[:,2]):.2f} wrist {np.median(dw):4.0f}/{dw.max():4.0f} pelvis {np.median(dp):4.0f} |a| {acc:.2f}"
    print("RESULT", line, flush=True)
