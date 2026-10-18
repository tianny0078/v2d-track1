#!/usr/bin/env python3
"""Sim(3) anchor risk: the scorer aligns on the FIRST scored frame only. Per episode, compare the
depth-ICP residual (MoGe person points -> visible body surface, cm) and the keypoint reprojection
at the first scored frame against the episode median. A first frame much worse than the median
means the whole episode is aligned on a bad pose."""
import sys, json, numpy as np, torch, trimesh
sys.path.insert(0, '.')
from src.mhr_local import MHR
from src.fit_mhr import COCO_TO_MHR
mhr = MHR('cuda:1'); K = json.load(open('work/camera/intrinsics.json')); meta = {json.loads(l)['episode_index']: json.loads(l) for l in open('data/hf/track_1/meta/episodes_metadata.jsonl')}
cs = sorted(COCO_TO_MHR); mi = [COCO_TO_MHR[c] for c in cs]; faces = mhr.faces
rows = []
for ep in range(30):
    k = K[meta[ep]['camera']]; b = np.load(f'work/body_depth/ep{ep:06d}.npz'); fids = b['frame_ids']; T = len(fids)
    mp = np.load(f'work/moge_pts/ep{ep:06d}.npz'); P = mp['pts']; n = mp['n']
    kz = np.load(f'work/kp2d/ep{ep:06d}.npz'); kp = np.nan_to_num(kz['kp'])
    sel = sorted(set([0, 1, 2, T - 1] + list(range(0, T, max(1, T // 25)))))
    p = np.concatenate([b['pose'][sel], np.broadcast_to(b['scales'], (len(sel), 68))], 1); v, j = mhr.forward(p, b['shape'])
    s_ep = json.loads(str(b['stats']))['s_ep'] if 'stats' in b.files else 1.15
    res = []
    for i, t in enumerate(sel):
        vt = v[i]; front = (np.asarray(trimesh.Trimesh(vt, faces, process=False).vertex_normals) * vt).sum(1) < 0
        vv = torch.tensor(vt[front][::3], device='cuda:1'); pts = torch.tensor(P[t, :n[t]] * s_ep, device='cuda:1')
        d = torch.cdist(pts, vv).min(1).values.median().item() * 100 if n[t] > 50 else np.nan
        Pj = j[i][mi]; u = k['fx'] * Pj[:, 0] / Pj[:, 2] + k['cx']; vvv = k['fy'] * Pj[:, 1] / Pj[:, 2] + k['cy']; c = kp[t, cs, 2]
        px = (np.hypot(u - kp[t, cs, 0], vvv - kp[t, cs, 1]) * c).sum() / max(c.sum(), 1)
        res.append((t, d, px))
    r = np.array(res); d0, px0 = r[0, 1], r[0, 2]; dm, pxm = np.nanmedian(r[:, 1]), np.nanmedian(r[:, 2])
    rows.append((ep, d0, dm, px0, pxm))
    print(f"ROW ep{ep:2d}: first scored frame {fids[0]}: depth resid {d0:4.1f} cm (episode median {dm:4.1f}) | px {px0:4.1f} (median {pxm:4.1f}) | frames 1-2: {r[1,1]:.1f}/{r[2,1]:.1f} cm", flush=True)
a = np.array(rows); bad = [int(e) for e, d0, dm, p0, pm in rows if d0 > 1.5 * dm or p0 > 1.5 * pm]
print(f"RESULT first-frame residual / median: depth {np.nanmedian(a[:,1]/a[:,2]):.2f}, px {np.nanmedian(a[:,3]/a[:,4]):.2f}; episodes with a clearly worse first frame: {bad}", flush=True)
