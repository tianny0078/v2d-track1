#!/usr/bin/env python3
"""Image-space sanity of an export's objects: projected extent / mask extent and centroid offset
(px) per episode, plus budget-mesh watertightness. Usage: obj_image_sanity.py <export_dir>"""
import numpy as np, json, sys, trimesh
sys.path.insert(0, '.')
from src.compare_ref import load_obj
D = sys.argv[1]
sc = json.load(open('work/scored_frames_summary.json')); K = json.load(open('work/camera/intrinsics.json'))
meta = {json.loads(l)['episode_index']: json.loads(l) for l in open('data/hf/track_1/meta/episodes_metadata.jsonl')}
for ep in range(30):
    k = K[meta[ep]['camera']]; fr = np.concatenate([np.arange(a, b + 1) for a, b in sc[str(ep)]['spans']])
    z = np.load(f'work/masks/ep{ep:06d}.npz'); W = int(z['shape'][1]); mpos = {f: i for i, f in enumerate(z['frame_ids'])}
    e = np.load(f'{D}/episode_{ep:06d}.npz'); mesh = trimesh.load(f'{D}/episode_{ep:06d}_object.obj', force='mesh', process=False)
    v = np.asarray(mesh.vertices) * float(e['object_scale']); v = v[::8]
    ratios = []; offs = []
    for f in fr[::10]:
        if f not in mpos: continue
        m = np.unpackbits(z['packed'][mpos[f]], axis=-1)[0, :, :W].astype(bool); ys, xs = np.where(m)
        if xs.size < 30: continue
        P = v @ e['object_rotation'][f].T + e['object_translation'][f]; u = k['fx'] * P[:, 0] / P[:, 2] + k['cx']; vv = k['fy'] * P[:, 1] / P[:, 2] + k['cy']
        ratios.append(max(np.ptp(u), np.ptp(vv)) / max(np.ptp(xs), np.ptp(ys))); offs.append(np.hypot(u.mean() - xs.mean(), vv.mean() - ys.mean()))
    print(f"ROW ep{ep:2d} {meta[ep]['object']:18s} proj/mask extent {np.median(ratios):.2f}  centroid off {np.median(offs):4.0f} px (p90 {np.percentile(offs, 90):4.0f})  watertight={mesh.is_watertight}", flush=True)
