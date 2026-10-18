#!/usr/bin/env python3
"""Depth-ICP body refinement (the organisers' CARI4D stage-3/4 idea, our version).

Start from a body sequence (default: the FlowHMR fit with reprojection, work/body_flow_px) and pull
the VISIBLE body surface onto MoGe-2's per-frame person point cloud (work/moge_pts), while keeping
the 2D keypoints, a prior to the start pose and temporal smoothness.

  L = w_d * GM(dist(MoGe point -> nearest visible MHR vertex), 5 cm)       (one-sided, robust)
    + w_px * GM(reprojection / 100, 0.6)                                   (VitPose)
    + w_prior * |pose[:,6:] - pose0[:,6:]|^2 + w_acc * |acc(pose)|^2

MoGe's scale differs from SAM3D's (median 1.17 on this data): one scale per episode, s_ep, is set
from the start body (median over frames of body-depth / MoGe-depth) and the MoGe points are
multiplied by it. Visible vertices = inside the person mask after projection AND front-facing
(vertex normal . view ray < 0), recomputed from the start body (not per iteration).
Output: work/body_depth/epNNNNNN.npz (pose, scales, shape, frame_ids, stats)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, trimesh
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
from src.fit_mhr import FLIP, COCO_TO_MHR, gm_loss, ACC_JOINTS
from v2dlb.mhr_submission import stretches
from src.mhr_local import MHR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, nargs="+", required=True)
    ap.add_argument("--body", default="work/body_flow_px")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-d", type=float, default=1.0)
    ap.add_argument("--w-px", type=float, default=5.0)
    ap.add_argument("--w-prior", type=float, default=1.0)
    ap.add_argument("--w-acc", type=float, default=0.0, help="pose-parameter acceleration (Euler-wrap unsafe: raising it made joint |a| WORSE on ep2: 0.22->0.33)")
    ap.add_argument("--w-jacc", type=float, default=2.0, help="joint-space acceleration of the 22 ACC-H joints, cm/frame^2 (the metric itself)")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--nvis", type=int, default=2500)
    ap.add_argument("--out", default="work/body_depth")
    a = ap.parse_args(); dev = torch.device(a.device)
    m = torch.jit.load(str(ROOT / "work/models/mhr_model.pt"), map_location=dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    flip = FLIP.to(dev); mhr = MHR(a.device); faces = mhr.faces
    kjson = json.load(open(ROOT / "work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
    (ROOT / a.out).mkdir(exist_ok=True)

    def fwd(pose, scales, shape):
        T = pose.shape[0]
        v, j = m(shape.expand(T, 45), torch.cat([pose, scales.expand(T, 68)], 1), torch.zeros(T, 72, device=dev))
        return v / 100.0 * flip, j[..., :3] / 100.0 * flip

    for ep in a.ep:
        t0 = time.time(); K = kjson[meta[ep]["camera"]]
        b = np.load(ROOT / a.body / f"ep{ep:06d}.npz"); fids = b["frame_ids"]; T = len(fids)
        pose0 = torch.tensor(b["pose"], dtype=torch.float32, device=dev); scales = torch.tensor(b["scales"], dtype=torch.float32, device=dev); shape = torch.tensor(b["shape"], dtype=torch.float32, device=dev)
        mp = np.load(ROOT / f"work/moge_pts/ep{ep:06d}.npz"); assert (mp["frame_ids"] == fids).all()
        P = mp["pts"]; n = mp["n"]
        kz = np.load(ROOT / f"work/kp2d/ep{ep:06d}.npz"); kp = np.nan_to_num(kz["kp"]); assert (kz["frame_ids"] == fids).all()
        cs = sorted(COCO_TO_MHR); mhr_c = [COCO_TO_MHR[c] for c in cs]
        k2d = torch.tensor(kp[:, cs, :2], dtype=torch.float32, device=dev); kconf = torch.tensor(kp[:, cs, 2], dtype=torch.float32, device=dev)
        # masks for visibility
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); W = int(z["shape"][1]); mpos = {f: i for i, f in enumerate(z["frame_ids"])}
        with torch.no_grad():
            v0, _ = fwd(pose0, scales, shape)
        v0n = v0.cpu().numpy()
        # per-episode MoGe scale from the start body: median over frames of (body z at mask) / (MoGe z)
        ratios = []
        vis_idx = []
        for t in range(T):
            vt = v0n[t]
            msk = np.unpackbits(z["packed"][mpos[fids[t]]], axis=-1)[1, :, :W] if fids[t] in mpos else None
            u = K["fx"] * vt[:, 0] / vt[:, 2] + K["cx"]; vv = K["fy"] * vt[:, 1] / vt[:, 2] + K["cy"]
            inside = np.zeros(len(vt), bool)
            if msk is not None:
                iu = np.clip(u.astype(int), 0, W - 1); iv = np.clip(vv.astype(int), 0, msk.shape[0] - 1); inside = msk[iv, iu] > 0
            tm = trimesh.Trimesh(vt, faces, process=False); nrm = np.asarray(tm.vertex_normals)
            front = (nrm * vt).sum(1) < 0            # normal against the viewing ray (camera at origin)
            vis = np.flatnonzero(inside & front)
            if len(vis) > a.nvis: vis = np.random.RandomState(t).choice(vis, a.nvis, replace=False)
            vis_idx.append(vis)
            if n[t] > 50 and len(vis) > 50:
                ratios.append(np.median(vt[vis, 2]) / np.nanmedian(P[t, :n[t], 2]))
        s_ep = float(np.median(ratios)) if ratios else 1.17
        Pt = torch.tensor(np.nan_to_num(P) * s_ep, dtype=torch.float32, device=dev)
        valid = torch.zeros(T, P.shape[1], dtype=torch.bool, device=dev)
        for t in range(T): valid[t, :n[t]] = True
        # pad visible index lists to a fixed width for batched gather
        width = max(len(vi) for vi in vis_idx)
        vis_t = torch.zeros(T, width, dtype=torch.long, device=dev); vis_m = torch.zeros(T, width, dtype=torch.bool, device=dev)
        for t, vi in enumerate(vis_idx):
            vis_t[t, :len(vi)] = torch.tensor(vi, device=dev); vis_m[t, :len(vi)] = True

        def depth_loss(v, sl):
            """one-sided robust distance MoGe point -> nearest visible vertex, frames sl"""
            vv = torch.gather(v[sl], 1, vis_t[sl].unsqueeze(-1).expand(-1, -1, 3))      # [B,W,3]
            d = torch.cdist(Pt[sl], vv)                                                   # [B,N,W]
            d = d.masked_fill(~vis_m[sl].unsqueeze(1), 1e3)
            dmin = d.min(2).values * 100.0                                                # cm
            r = gm_loss(dmin.unsqueeze(-1), 5.0)                                          # GM on scalar distance
            return (r * valid[sl]).sum() / valid[sl].sum().clamp(min=1)

        def stats(pose):
            with torch.no_grad():
                v, j = fwd(pose, scales, shape); ds = []
                for s0 in range(0, T, a.chunk):
                    sl = slice(s0, min(T, s0 + a.chunk)); vv = torch.gather(v[sl], 1, vis_t[sl].unsqueeze(-1).expand(-1, -1, 3))
                    d = torch.cdist(Pt[sl], vv).masked_fill(~vis_m[sl].unsqueeze(1), 1e3).min(2).values * 100
                    ds.append(d[valid[sl]])
                d = torch.cat(ds); Pj = j[:, mhr_c]; zc = Pj[..., 2].clamp(min=0.3)
                p2 = torch.stack([K["fx"] * Pj[..., 0] / zc + K["cx"], K["fy"] * Pj[..., 1] / zc + K["cy"]], -1)
                px = float(((p2 - k2d).norm(dim=-1) * kconf).sum() / kconf.sum().clamp(min=1))
                return float(d.median()), float(d.mean()), px

        stretch_slices = [sl for sl in stretches(fids) if sl.stop - sl.start >= 3]
        d0_med, d0_mean, px0 = stats(pose0)
        pose = pose0.clone().requires_grad_(True); opt = torch.optim.Adam([pose], lr=a.lr)
        for it in range(a.iters):
            opt.zero_grad(set_to_none=True)
            v, j = fwd(pose, scales, shape)
            L = torch.zeros((), device=dev)
            for s0 in range(0, T, a.chunk):
                sl = slice(s0, min(T, s0 + a.chunk)); L = L + a.w_d * depth_loss(v, sl) * (sl.stop - sl.start) / T
            Pj = j[:, mhr_c]; zc = Pj[..., 2].clamp(min=0.3)
            p2 = torch.stack([K["fx"] * Pj[..., 0] / zc + K["cx"], K["fy"] * Pj[..., 1] / zc + K["cy"]], -1)
            L = L + a.w_px * (gm_loss((p2 - k2d) / 100.0, 0.6) * kconf).sum() / kconf.sum().clamp(min=1)
            L = L + a.w_prior * (pose[:, 6:] - pose0[:, 6:]).pow(2).mean()
            if a.w_acc > 0 and T >= 3: L = L + a.w_acc * (pose[2:] - 2 * pose[1:-1] + pose[:-2]).pow(2).mean()
            if a.w_jacc > 0:
                for sl in stretch_slices:
                    seg = j[sl][:, list(ACC_JOINTS)] * 100.0
                    L = L + a.w_jacc * (seg[:-2] - 2 * seg[1:-1] + seg[2:]).norm(dim=-1).mean() * (sl.stop - sl.start) / T
            L.backward(); opt.step()
        d1_med, d1_mean, px1 = stats(pose)
        res = dict(ep=ep, T=T, s_ep=round(s_ep, 3), depth_cm_median=(round(d0_med, 2), round(d1_med, 2)), depth_cm_mean=(round(d0_mean, 2), round(d1_mean, 2)),
                   px=(round(px0, 2), round(px1, 2)), root_shift_cm=round(float((pose[:, :3] - pose0[:, :3]).abs().mean() * 10), 2), seconds=round(time.time() - t0, 1))
        np.savez_compressed(ROOT / a.out / f"ep{ep:06d}.npz", pose=pose.detach().cpu().numpy(), scales=b["scales"], shape=b["shape"], frame_ids=fids, stats=json.dumps(res))
        print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
