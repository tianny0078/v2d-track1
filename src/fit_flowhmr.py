#!/usr/bin/env python3
"""Fit the MHR body sequence to FlowHMR's SMPL-H joints (route B).

FlowHMR gives a temporally coherent SMPL-H motion with a world root trajectory; SAM 3D Body gives
per-frame MHR parameters (identity included) that jitter in time (joint |a| 0.85 vs GT 0.15
cm/frame^2). We keep SAM3D's identity (shape, scales) and its pose as the initialisation, and pull
the whole sequence onto FlowHMR's joints:

  1. FlowHMR joints (SMPL world, y up) -> our camera frame by ONE Sim(3) fitted over all common
     frames between FlowHMR's 19 body joints and SAM3D-MHR's corresponding joints.
  2. Adam over pose [T,136]: robust 3D joint term (body + hands) + weak prior to the SAM3D body
     pose. Root translation/rotation follow FlowHMR freely (that is the trajectory we want).

Output: work/body_flow/epNNNNNN.npz with the same keys the exporter's sam3d body uses.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit")); sys.path.insert(0, str(ROOT / "work/flowhmr"))
from src.export_episodes import _sam3d_as_body
from src.fit_mhr import FLIP, ACC_JOINTS, COCO_TO_MHR, gm_loss
from v2dlb.mhr_metrics import fit_similarity_transform, apply_similarity_transform
from v2dlb.mhr_submission import stretches

# SMPL-H joint index -> MHR joint index. Spine joints are skipped: the two rigs place them
# differently and a wrong target there only bends the torso.
BODY_MAP = [(0, 1), (1, 2), (2, 18), (4, 3), (5, 19), (7, 4), (8, 20), (10, 8), (11, 24),
            (12, 110), (15, 113), (13, 74), (14, 38), (16, 75), (17, 39), (18, 76), (19, 40), (20, 78), (21, 42)]
# SMPL-H hand order per hand: index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3
_L = [92, 93, 94, 88, 89, 90, 80, 81, 82, 84, 85, 86, 97, 98, 99]
_R = [56, 57, 58, 52, 53, 54, 44, 45, 46, 48, 49, 50, 61, 62, 63]
HAND_MAP = [(22 + i, m) for i, m in enumerate(_L)] + [(37 + i, m) for i, m in enumerate(_R)]


def flowhmr_joints(ep: int, dev):
    os.chdir(ROOT / "work/flowhmr")
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLSkeleton
    os.chdir(ROOT)
    o = np.load(ROOT / f"work/flowhmr_out/ep{ep:06d}.npz")
    sk = SMPLSkeleton(model_path=str(ROOT / "work/flowhmr/assets/body_models/smplh/neutral/model.npz")).to(dev)
    T = o["rot6d"].shape[1]
    with torch.no_grad():
        out = sk({"rot6d": torch.tensor(o["rot6d"][0], device=dev),
                  "shapes": torch.tensor(o["shapes"][0].repeat(T, 0), device=dev),
                  "trans": torch.tensor(o["trans"][0], device=dev)})
    return out["keypoints3d"].cpu().numpy(), o["frame_ids"]


def acc_proxy(jm, fids):
    """mean second-difference magnitude of the 22 ACC-H joints in cm/frame^2 within stretches."""
    vals = []
    for sl in stretches(fids):
        if sl.stop - sl.start < 3:
            continue
        x = jm[sl][:, list(ACC_JOINTS)] * 100
        vals.append(np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=-1).mean())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--w-prior", type=float, default=1.0)
    ap.add_argument("--w-hand", type=float, default=0.3)
    ap.add_argument("--sigma-cm", type=float, default=5.0)
    ap.add_argument("--w-px", type=float, default=0.0, help="2D keypoint reprojection term (VitPose); keeps the root on the image where FlowHMR's trajectory drifts")
    ap.add_argument("--out", default="work/body_flow")
    a = ap.parse_args()
    dev = torch.device(a.device)
    m = torch.jit.load(str(ROOT / "work/models/mhr_model.pt"), map_location=dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    flip = FLIP.to(dev)
    kjson = json.load(open(ROOT / "work/camera/intrinsics.json"))
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
    (ROOT / a.out).mkdir(exist_ok=True)

    def fwd(pose, scales, shape):
        T = pose.shape[0]
        v, j = m(shape.expand(T, 45), torch.cat([pose, scales.expand(T, 68)], 1), torch.zeros(T, 72, device=dev))
        return j[..., :3] / 100.0 * flip

    for ep in a.ep:
        t0 = time.time()
        b = _sam3d_as_body(ROOT / f"work/sam3d/ep{ep:06d}.npz")
        fids = b["frame_ids"]
        Jf_all, fids_f = flowhmr_joints(ep, dev)
        pos = {f: i for i, f in enumerate(fids_f)}
        ok = np.array([f in pos for f in fids])
        Jf = np.stack([Jf_all[pos[f]] if f in pos else np.zeros((52, 3)) for f in fids])
        pose0 = torch.tensor(b["pose"], dtype=torch.float32, device=dev)
        scales = torch.tensor(b["scales"], dtype=torch.float32, device=dev)
        shape = torch.tensor(b["shape"], dtype=torch.float32, device=dev)
        with torch.no_grad():
            jm0 = fwd(pose0, scales, shape).cpu().numpy()
        # one Sim(3) over the episode, body joints only
        si = [s for s, _ in BODY_MAP]; mi = [q for _, q in BODY_MAP]
        tr = fit_similarity_transform(Jf[ok][:, si].reshape(-1, 3), jm0[ok][:, mi].reshape(-1, 3))
        Jt = apply_similarity_transform(Jf.reshape(-1, 3), tr).reshape(Jf.shape)
        err0 = np.linalg.norm(Jt[ok][:, si] - jm0[ok][:, mi], axis=-1).mean() * 100
        tgt = torch.tensor(Jt, dtype=torch.float32, device=dev)
        wmask = torch.tensor(ok, dtype=torch.float32, device=dev)[:, None]
        sb = [s for s, _ in BODY_MAP]; mb = [q for _, q in BODY_MAP]
        sh = [s for s, _ in HAND_MAP]; mh = [q for _, q in HAND_MAP]

        K = kjson[meta[ep]["camera"]]
        kz = np.load(ROOT / f"work/kp2d/ep{ep:06d}.npz"); kp = np.nan_to_num(kz["kp"]); assert (kz["frame_ids"] == fids).all()
        cs = sorted(COCO_TO_MHR); mhr_c = [COCO_TO_MHR[c] for c in cs]
        k2d = torch.tensor(kp[:, cs, :2], dtype=torch.float32, device=dev); kconf = torch.tensor(kp[:, cs, 2], dtype=torch.float32, device=dev)
        pose = pose0.clone().requires_grad_(True)
        for stage, (n, mask_root) in enumerate(((60, True), (a.iters, False))):
            opt = torch.optim.Adam([pose], lr=a.lr * (4 if mask_root else 1))
            for it in range(n):
                opt.zero_grad(set_to_none=True)
                jm = fwd(pose, scales, shape)
                rb = (jm[:, mb] - tgt[:, sb]) * 100
                rh = (jm[:, mh] - tgt[:, sh]) * 100
                data = (gm_loss(rb, a.sigma_cm) * wmask).sum() / (wmask.sum() * len(mb)) \
                     + a.w_hand * (gm_loss(rh, a.sigma_cm) * wmask).sum() / (wmask.sum() * len(mh))
                loss = data + a.w_prior * (pose[:, 6:] - pose0[:, 6:]).pow(2).mean()
                if a.w_px > 0:
                    P = jm[:, mhr_c]; z = P[..., 2].clamp(min=0.3)
                    p2 = torch.stack([K["fx"] * P[..., 0] / z + K["cx"], K["fy"] * P[..., 1] / z + K["cy"]], -1)
                    loss = loss + a.w_px * (gm_loss((p2 - k2d) / 100.0, 0.6) * kconf).sum() / kconf.sum().clamp(min=1)
                loss.backward()
                if mask_root:
                    pose.grad[:, 6:] = 0
                opt.step()
        with torch.no_grad():
            jm1 = fwd(pose, scales, shape).cpu().numpy()
        err1 = np.linalg.norm(Jt[ok][:, si] - jm1[ok][:, mi], axis=-1).mean() * 100
        # 2D check against VitPose
        def px(jm):
            cs = sorted(COCO_TO_MHR); P = jm[:, [COCO_TO_MHR[c] for c in cs]]
            u = K["fx"] * P[..., 0] / P[..., 2] + K["cx"]; v = K["fy"] * P[..., 1] / P[..., 2] + K["cy"]
            c = kp[:, cs, 2]; d = np.hypot(u - kp[:, cs, 0], v - kp[:, cs, 1])
            return float((d * c).sum() / max(c.sum(), 1))
        stats = dict(ep=ep, T=int(len(fids)), covered=int(ok.sum()), sim3_scale=float(tr[0]),
                     joint_err_cm=(round(float(err0), 2), round(float(err1), 2)),
                     acc_h=(round(acc_proxy(jm0, fids), 3), round(acc_proxy(jm1, fids), 3)),
                     px=(round(px(jm0), 2), round(px(jm1), 2)),
                     body_pose_change=float((pose[:, 6:] - pose0[:, 6:]).abs().mean()), seconds=round(time.time() - t0, 1))
        np.savez_compressed(ROOT / a.out / f"ep{ep:06d}.npz", pose=pose.detach().cpu().numpy(), scales=b["scales"], shape=b["shape"],
                            frame_ids=fids, covered=ok, stats=json.dumps(stats))
        print("RESULT", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
