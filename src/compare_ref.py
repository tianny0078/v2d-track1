#!/usr/bin/env python3
"""Fast ours-vs-reference comparison the way the scorer looks at it.

Reference = another of OUR exports (A vs B ablation). NOTE: work/baseline_episodes is the Kaggle
sample submission -- one constant rest pose for every frame of every episode and a 30 cm cube at the
origin -- NOT the CARI4D baseline that sits on the leaderboard (CD-H 15.58, CD-O 23.16); the CARI4D
output is not downloadable. Never read a comparison against it as an error estimate. One Sim(3) from
our frame-0 alignment vertices onto the reference's, applied to our whole episode (human and
object), then per-frame: body-proxy Chamfer (cm), pelvis distance, object-centroid distance.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
from src.mhr_local import MHR
from v2dlb.mhr_metrics import fit_similarity_transform, apply_similarity_transform
from src.export_episodes import _sam3d_as_body

def load_obj(p):
    v = [list(map(float, l.split()[1:4])) for l in open(p) if l.startswith("v ")]
    return np.array(v, dtype=np.float32)

def chamfer(a, b):
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=ROOT / "work/baseline_episodes")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--eps", type=int, nargs="*", default=list(range(30)))
    a = ap.parse_args()
    mhr = MHR(a.device)
    ali, prox = mhr.alignment_vertices, mhr.body_proxy_vertices
    rows = []
    for ep in a.eps:
        fids = _sam3d_as_body(ROOT / f"work/sam3d/ep{ep:06d}.npz")["frame_ids"]
        o = np.load(a.ours / f"episode_{ep:06d}.npz"); r = np.load(a.ref / f"episode_{ep:06d}.npz")
        po = np.concatenate([o["pose"][fids], np.broadcast_to(o["scales"], (len(fids), 68))], 1)
        pr = np.concatenate([r["pose"][fids], np.broadcast_to(r["scales"], (len(fids), 68))], 1)
        vo, jo = mhr.forward(po, o["shape"]); vr, jr = mhr.forward(pr, r["shape"])
        tr = fit_similarity_transform(vo[0][ali], vr[0][ali])
        S = lambda x: apply_similarity_transform(x.reshape(-1, 3), tr).reshape(x.shape)
        vo, jo = S(vo), S(jo)
        cdh = np.mean([chamfer(vo[t][prox], vr[t][prox]) for t in range(len(fids))]) * 100
        pel = np.linalg.norm(jo[:, 1] - jr[:, 1], axis=-1) * 100
        # object centroid trajectory
        mo = load_obj(a.ours / f"episode_{ep:06d}_object.obj").mean(0) * float(o["object_scale"])
        mr = load_obj(a.ref / f"episode_{ep:06d}_object.obj").mean(0) * float(r["object_scale"])
        co = np.einsum("tij,j->ti", o["object_rotation"][fids], mo) + o["object_translation"][fids]
        cr = np.einsum("tij,j->ti", r["object_rotation"][fids], mr) + r["object_translation"][fids]
        co = S(co)
        od = np.linalg.norm(co - cr, axis=-1) * 100
        rows.append(dict(ep=ep, T=len(fids), scale=round(float(tr[0]), 3), cdh=round(float(cdh), 1),
                         pelvis=round(float(pel.mean()), 1), pelvis_last=round(float(pel[-1]), 1),
                         obj=round(float(od.mean()), 1), obj_dz=round(float((co[:, 2] - cr[:, 2]).mean() * 100), 1),
                         obj_ref_depth=round(float(cr[:, 2].mean()), 2), obj_our_depth=round(float(co[:, 2].mean()), 2)))
        print("ROW", json.dumps(rows[-1]), flush=True)
    print("RESULT mean body-proxy CD vs ref %.1f cm | pelvis %.1f cm | object centroid %.1f cm (median %.1f)" % (
        np.mean([x["cdh"] for x in rows]), np.mean([x["pelvis"] for x in rows]), np.mean([x["obj"] for x in rows]), np.median([x["obj"] for x in rows])))
    json.dump(rows, open(ROOT / f"work/compare_{a.ours.name}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
