#!/usr/bin/env python3
"""Assemble per-episode submission NPZs from whatever stages have finished.

The packer indexes `pose` by VIDEO FRAME (row t is frame t), but the body fit only solves the
scored frames, so unscored rows are filled from the nearest solved frame. Those rows are never
read by the scorer; they exist only so the array is dense.

Object pose is optional: with --object none the object keeps the format-floor placeholder, which
is what isolates a human-only change on the leaderboard.
"""
from __future__ import annotations
import os
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/kit/v2d_submission_kit"))
from v2dlb.mhr_submission import stretches  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fit_mhr import FOOT_JOINTS, POSE_UNIT_M, POSE_SIGN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _euler_to_mat(e):
    """Host's euler-xyz convention (metric_code/track_1/*.py::_euler_xyz_to_mat), transcribed."""
    cx, cy, cz = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
    sx, sy, sz = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
    return np.stack([np.stack([cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz], -1),
                     np.stack([cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz], -1),
                     np.stack([-sy, sx * cy, cx * cy], -1)], -2)


def _mat_to_euler(R):
    y = np.arcsin(np.clip(-R[..., 2, 0], -1, 1))
    return np.stack([np.arctan2(R[..., 2, 1], R[..., 2, 2]), y,
                     np.arctan2(R[..., 1, 0], R[..., 0, 0])], -1)


def _smooth_wrap_safe(seg: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a pose stretch WITHOUT blending the root rotation across the +-pi wrap.

    pose[3:6] is the root rotation as euler xyz. A subject who turns around (ep26/ep18: sit on
    the stool and rotate) sweeps yaw through +-pi; the raw sequence then jumps by 2*pi on two
    channels (ep26: 20 such frames). A plain Gaussian interpolates across that discontinuity and
    manufactures a half-turn -- measured: ep26's joint acceleration went 1.65 -> 9.58 cm/frame^2
    AFTER smoothing, and the leaderboard ACC-H came back 1.103 against a predicted ~0.12.
    Smoothing the first two columns of the rotation matrix (a 6D representation with no wrap),
    re-orthonormalising and converting back fixes it: ep26 -> 0.116 at sigma 3.
    Translation and the 130 body-pose params are small-angle and get the plain Gaussian.
    """
    from scipy.ndimage import gaussian_filter1d
    out = gaussian_filter1d(seg, sigma, axis=0, mode="nearest")
    R = _euler_to_mat(seg[:, 3:6])
    c = gaussian_filter1d(R[..., :2], sigma, axis=0, mode="nearest")
    a = c[..., 0] / np.linalg.norm(c[..., 0], axis=-1, keepdims=True)
    b = c[..., 1] - (a * c[..., 1]).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    out[:, 3:6] = _mat_to_euler(np.stack([a, b, np.cross(a, b)], -1))
    return out


_SIZE_CACHE: dict = {}


def _object_metric_size(ep, K, mesh) -> float:
    """Metric max-dimension of this episode's object: consensus over its THREE episodes.

    Each episode contributes its estimate at the frame where the object mask is largest (most
    visible, least occluded): size = ext_px * z_hand / f there, where z_hand is trusted only at
    that frame. Median over the three. Cached per object name."""
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    name = meta[ep]["object"]
    if name in _SIZE_CACHE:
        return _SIZE_CACHE[name]
    from src.mhr_local import MHR, L_WRIST, R_WRIST
    global _MHR
    try:
        mhr = _MHR
    except NameError:
        mhr = _MHR = MHR()
    Kall = json.load((ROOT / "work/camera/intrinsics.json").open())
    sc_all = json.load((ROOT / "work/scored_frames_summary.json").open())
    ests = []
    for e in [k for k, v in meta.items() if v["object"] == name]:
        Ke = Kall[meta[e]["camera"]]
        z = np.load(ROOT / f"work/sam3d/ep{e:06d}.npz")
        # the SAM3D body is SPARSE: one row per scored frame, in frame_ids order. Index it by
        # position; indexing it by video frame number was the IndexError (184 >= 169) that
        # crashed the first v3 run.
        bd = _sam3d_as_body(ROOT / f"work/sam3d/ep{e:06d}.npz")
        fr = bd["frame_ids"]
        pr = np.concatenate([bd["pose"], np.broadcast_to(bd["scales"], (len(fr), 68))], 1)
        _, j = mhr.forward(pr, bd["shape"])
        m = np.load(ROOT / f"work/masks/ep{e:06d}.npz"); W = int(m["shape"][1]); packed = m["packed"]
        mi = np.searchsorted(m["frame_ids"], fr)
        best = None
        for t, i in enumerate(mi):
            om = np.unpackbits(packed[i], axis=-1)[0, :, :W]; ys, xs = np.where(om)
            if len(xs) < 30:
                continue
            ext = max(xs.max() - xs.min(), ys.max() - ys.min())
            if best is None or len(xs) > best[0]:
                hz = j[t, [L_WRIST, R_WRIST], 2].min()
                best = (len(xs), ext * hz / Ke["fx"])
        if best:
            ests.append(best[1])
    S = float(np.median(ests)) if ests else float(np.max(mesh.extents))
    _SIZE_CACHE[name] = S
    print(f"  object {name}: metric size consensus {S:.2f} m from {np.round(ests, 2).tolist()}", flush=True)
    return S


OBJ_DEPTH = "mask"
OBJ_DEPTH_COL = 0
THIN_FILL = 0.2
CONTACT_ANCHOR = False
PUSH_MODE = "centroid"
OBJ_SIZE = "consensus"
OBJ_DEPTH_SCALE = 1.0
THIN_RULE = True
OBJ_NAME = ""
PUSH_ROUNDS = 4


def _place_object(ep, camera, pose, scales, shape, fids, mesh, sigma, T, R_seq=None):
    """Object translation at the hands, scale from the mask, identity rotation. See --object help."""
    from scipy.ndimage import gaussian_filter1d
    from src.mhr_local import MHR, L_WRIST, R_WRIST
    global _MHR
    try:
        mhr = _MHR
    except NameError:
        mhr = _MHR = MHR()
    K = json.load((ROOT / "work/camera/intrinsics.json").open())[camera]
    sc = json.load((ROOT / "work/scored_frames_summary.json").open())[str(ep)]
    fr = np.concatenate([np.arange(a_, b_ + 1) for a_, b_ in sc["spans"]])
    pr = np.concatenate([pose[fr], np.broadcast_to(scales, (len(fr), 68))], 1)
    v_all, j = mhr.forward(pr, shape)
    hands = j[:, [L_WRIST, R_WRIST]]                                     # [N,2,3] m
    m = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); W = int(m["shape"][1]); packed = m["packed"]
    mi = np.searchsorted(m["frame_ids"], fr)
    uv = np.full((len(fr), 2), np.nan); ext = np.full(len(fr), np.nan); fill = np.full(len(fr), np.nan)
    for t, i in enumerate(mi):
        om = np.unpackbits(packed[i], axis=-1)[0, :, :W]; ys, xs = np.where(om)
        if len(xs) > 30:
            uv[t] = [xs.mean(), ys.mean()]; ext[t] = max(xs.max() - xs.min(), ys.max() - ys.min())
            fill[t] = len(xs) / float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    hu = K["fx"] * hands[..., 0] / hands[..., 2] + K["cx"]; hv = K["fy"] * hands[..., 1] / hands[..., 2] + K["cy"]
    d = np.linalg.norm(np.stack([hu, hv], -1) - uv[:, None, :], axis=-1)
    near = np.where(np.isnan(d), np.inf, d).argmin(1)
    zh_hand = hands[np.arange(len(fr)), near, 2]
    ok = ~np.isnan(uv[:, 0])
    # DEPTH FROM THE OBJECT'S OWN APPARENT SIZE, not from the hand. The scored span is the
    # CONTACT span, which is not the same as holding: in the hula-hoop episodes the subject steps
    # into a hoop lying on the floor between the camera and themselves, and "hand depth" put a
    # 4.4 m hoop at 5.6 m (measured). With a per-object metric size S (consensus over its three
    # episodes, each estimated at the frame where the object is most visible, i.e. largest mask),
    # z = f * S / ext_px per frame; the hand depth is only a fallback when the mask is missing.
    S = _object_metric_size(ep, K, mesh)
    z_mask = np.where(ok & (ext > 0), K["fx"] * S / np.maximum(ext, 1), np.nan)
    # MoGe DEPTH RATIO (--obj-depth moge). Mask-size depth needs a clean silhouette and the
    # object's metric size; it put the object 1-2 m from the hand on the worst episodes (ep1 212
    # cm, ep8 157 cm, measured on place7). MoGe's dense depth has an unknown global scale, but the
    # ratio object/person inside ONE frame is scale-free, and the person's depth is SAM3D's. The
    # person reference is the depth of the body vertices that project inside the person mask
    # (35th percentile ~ the visible front surface, which is what MoGe measures).
    if OBJ_DEPTH == "moge" and (ROOT / f"work/objdepth/ep{ep:06d}.npz").exists():
        od = np.load(ROOT / f"work/objdepth/ep{ep:06d}.npz"); opos = {f: i for i, f in enumerate(od["frame_ids"])}
        z_moge = np.full(len(fr), np.nan)
        for t, f in enumerate(fr):
            if f not in opos:
                continue
            # column 0 = median, 1 = p20 (nearest surface: thin objects such as the hoop let the
            # floor behind them into the mask, which drags the median back), 2 = p80
            do, dp = od["d_obj"][opos[f], OBJ_DEPTH_COL], od["d_person"][opos[f], 0]
            if not (np.isfinite(do) and np.isfinite(dp)) or od["n_obj"][opos[f]] < 50:
                continue
            vz = v_all[t, :, 2]; vu = K["fx"] * v_all[t, :, 0] / vz + K["cx"]; vv = K["fy"] * v_all[t, :, 1] / vz + K["cy"]
            pm = np.unpackbits(packed[mi[t]], axis=-1)[1, :, :W]
            iu = np.clip(vu.astype(int), 0, W - 1); iv = np.clip(vv.astype(int), 0, pm.shape[0] - 1)
            inside = pm[iv, iu] > 0
            z_person = np.percentile(vz[inside], 35) if inside.sum() > 100 else np.percentile(vz, 35)
            z_moge[t] = z_person * do / dp * OBJ_DEPTH_SCALE
        # THIN objects (hula hoop: mask fill ratio 0.12 vs 0.8 for solid ones) let the background
        # through the mask, and MoGe then reports the wall behind the ring (ep1: ratio 1.5 while
        # the hoop is held up, object 1.5 m behind the hand). The scored span is the contact span,
        # so for a thin object the nearest hand's depth is the better estimate.
        thin = np.isfinite(fill) & (fill < THIN_FILL)
        if THIN_RULE and thin.sum() > 0.5 * np.isfinite(fill).sum():
            z_moge[:] = zh_hand
            print(f"  ep{ep} thin object (median fill {np.nanmedian(fill):.2f} < {THIN_FILL}): depth anchored to the nearest hand")
        if CONTACT_ANCHOR:
            # CONTACT ANCHORING. CD-O is measured after the Sim(3) fitted on the HUMAN, so what
            # matters is the object's depth RELATIVE to the body, not its absolute depth. At frames
            # where a wrist lies inside the object's mask the object is at the hand; MoGe's ratio
            # has a per-episode bias against the SAM3D hand (median |bias| 33 cm, within-episode
            # spread 24 cm; the mask-size depth: 33 / 46), so shift the whole MoGe series by the
            # median hand-minus-MoGe offset over the contact frames and keep its relative motion.
            dn = d[np.arange(len(fr)), near]
            contact = np.isfinite(z_moge) & np.isfinite(dn) & (dn < 0.6 * np.nan_to_num(ext, nan=0.0))
            if contact.sum() >= 10:
                delta = float(np.median(zh_hand[contact] - z_moge[contact]))
                z_moge = z_moge + delta
                print(f"  ep{ep} contact anchor: {int(contact.sum())} frames, shift {delta*100:+.0f} cm")
            else:
                print(f"  ep{ep} contact anchor: only {int(contact.sum())} contact frames, no shift")
        n_m = int(np.isfinite(z_moge).sum())
        print(f"  ep{ep} moge depth: {n_m}/{len(fr)} frames; median |z_moge - z_mask| "
              f"{np.nanmedian(np.abs(z_moge - z_mask)):.2f} m, median z_moge {np.nanmedian(z_moge):.2f} m")
        z_mask = np.where(np.isfinite(z_moge), z_moge, z_mask)
    # the mask's apparent size jitters frame to frame (thin objects edge-on, hands covering it),
    # and 1/ext turns that into depth jitter: ACC-O proxy hit 5.95 on the hoop. Smooth it first,
    # then keep it within reach of the hand that is touching it -- the scored span is the contact
    # span, so the object cannot be more than an arm away from the nearest hand.
    from scipy.ndimage import gaussian_filter1d as _g
    zf = z_mask.copy()
    for sl in stretches(fr):
        seg = zf[sl]
        if np.isnan(seg).all():
            continue
        idx_ = np.arange(len(seg)); good_ = ~np.isnan(seg)
        seg = np.interp(idx_, idx_[good_], seg[good_])
        zf[sl] = _g(seg, 6.0, mode="nearest") if len(seg) >= 3 else seg
    zh = np.where(np.isnan(zf), zh_hand, np.clip(zf, zh_hand - 1.5, zh_hand + 1.5))
    zh = np.clip(zh, 0.5, 12.0)
    if OBJ_SIZE == "moge" and OBJ_DEPTH == "moge":
        # METRIC SIZE FROM THE PLACEMENT DEPTH: S = ext_px * z / f on the least-occluded frames
        # (mask area >= 70% of the episode max). The hand-depth consensus was 10-20% off on
        # the desk (1.18 vs 1.35), stool (0.73 vs 0.64), paint roller (0.27 vs 0.19); this makes
        # the projected object match the mask at the depth we actually place it.
        area_ = np.array([np.count_nonzero(np.unpackbits(packed[i], axis=-1)[0, :, :W]) for i in mi])
        good_ = ok & np.isfinite(zh) & (area_ >= 0.7 * area_.max())
        sz_file = ROOT / "work/object_size_moge.json"
        per_obj = json.load(sz_file.open())["size_m"] if sz_file.exists() else {}
        if OBJ_NAME in per_obj:
            # per-OBJECT median over its three episodes (the pan's per-episode estimates were
            # 0.19/0.19/0.28 -- the handle comes and goes -- so the single-episode value is noisy)
            S_new = float(per_obj[OBJ_NAME])
            print(f"  ep{ep} object size: consensus {S:.2f} m -> moge-depth per-object {S_new:.2f} m")
            S = S_new
        elif good_.sum() >= 5:
            S_new = float(np.median(ext[good_] * zh[good_] / K["fx"]))
            print(f"  ep{ep} object size: consensus {S:.2f} m -> moge-depth {S_new:.2f} m ({good_.sum()} frames)")
            S = S_new
    # the clamp is applied AFTER the smoothing and re-introduces kinks where it engages (place7
    # ep18: z-dominated spikes on three consecutive frames with a stable mask). Smooth once more
    # after clamping so the trajectory the scorer differences is C1 everywhere.
    for sl in stretches(fr):
        if sl.stop - sl.start >= 3:
            zh[sl] = _g(zh[sl], 6.0, mode="nearest")
    # frames with no object mask: hold the last good centroid (object is then likely occluded by the body)
    for t in range(len(fr)):
        if not ok[t]:
            src_ = t - 1 if t > 0 and ok[:t].any() else int(np.where(ok)[0][0])
            uv[t] = uv[src_]; ext[t] = ext[src_]
    # occlusion hold: when the visible mask shrinks sharply the centroid is no longer the object's
    # centre (the hand/body is covering it) -- keep the previous estimate instead of chasing it
    ref = np.nanmedian(ext)
    for t in range(1, len(fr)):
        if ext[t] < 0.5 * ref:
            uv[t] = uv[t - 1]
    X = (uv[:, 0] - K["cx"]) / K["fx"] * zh; Y = (uv[:, 1] - K["cy"]) / K["fy"] * zh
    t3 = np.stack([X, Y, zh], 1).astype(np.float32)
    mesh_ext = float(np.max(mesh.extents))
    obj_s = np.float32(S / max(mesh_ext, 1e-6))
    # centre the mesh so its translation is its centroid, which is what the mask centroid estimates
    cen = (obj_s * np.asarray(mesh.vertices).mean(0)).astype(np.float32)
    # per-frame rotation (object_track.py): world centroid = R (s * mean(v)) + t  ->  t = t3 - R cen
    Rf = np.tile(np.eye(3, dtype=np.float32), (len(fr), 1, 1)) if R_seq is None else R_seq[fr].astype(np.float32)
    t3 = t3 - np.einsum("tij,j->ti", Rf, cen)
    # the object is rigid and the reference's own translation acceleration is only 0.074
    # cm/frame^2, so smooth it harder than the body (sigma 6 vs 3)
    obj_sigma = max(sigma, 6.0)
    for sl in stretches(fr):
        if sl.stop - sl.start >= 3:
            t3[sl] = gaussian_filter1d(t3[sl], obj_sigma, axis=0, mode="nearest")
    # PEN = 0 BY CONSTRUCTION. The scorer measures our hand points inside our own mesh; a centroid
    # placed at the hand puts the hand inside the object (measured: PEN 4.51 cm, max depth 19 cm).
    # Push the object away from the hand along the hand->object direction until no hand point is
    # inside, using the host's own penetration_depth. One smooth push per stretch (the max over
    # frames, then filtered) so the fix does not re-introduce jitter.
    sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
    from v2dlb.mhr_submission import penetration_depth
    vr = np.concatenate([pose[fr], np.broadcast_to(scales, (len(fr), 68))], 1)
    verts, _ = mhr.forward(vr, shape)
    hand_pts = verts[:, mhr.hand_vertices].astype(np.float64)              # [N,P,3] m
    # Run the SDF against the mesh THE HOST SCORES: the packer decimates to 4096 vertices/faces
    # (v2dlb.mesh_budget.budget_mesh) and PEN is measured on that. The raw SAM3D-Objects meshes
    # are 100k-400k faces -- pushing against them was both ~50x slower (3 episodes/hour) and
    # subtly wrong, since decimation moves the surface by a few mm.
    from v2dlb.mesh_budget import budget_mesh
    import tempfile, trimesh as _tm
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as _f:
        mesh.export(_f.name)
    try:
        mv, mf = budget_mesh(_f.name, 4096, 4096)
    finally:
        os.unlink(_f.name)                                                   # was leaking 8-57 MB per call into /tmp
    mv = np.asarray(mv, np.float64); mf = np.asarray(mf, np.int64)
    mf = mf[(mf[:, 0] != mf[:, 1]) | (mf[:, 1] != mf[:, 2])]        # drop zero-area padding faces
    hand_c = hands[np.arange(len(fr)), near]                                 # nearest hand, [N,3]
    if PUSH_MODE == "closest":
        # MINIMAL ESCAPE. The hand->centroid push pushed hollow/large objects by half their size
        # (stool ep25: 90 cm, cart 55 cm, desk 46 cm; image centroid 70-90 px off the mask) because
        # the hand gripping a seat edge sits at the object's centre. The way out of a thin slab is
        # a few cm along the surface normal: push along the mean (penetrating point -> closest
        # surface point) direction by the max depth, iterate until clear.
        pq = _tm.proximity.ProximityQuery(_tm.Trimesh(mv, mf, process=False))

        sub = np.random.RandomState(0).choice(hand_pts.shape[1], min(300, hand_pts.shape[1]), replace=False)

        def _escape(t, t3t, pts=None):
            """world-space push vector (m) that clears the hand at frame t from translation t3t."""
            P_ = hand_pts if pts is None else hand_pts[:, pts]
            total = np.zeros(3)
            for _ in range(8):
                local = ((P_[t] - (t3t + total)) @ Rf[t]) / float(obj_s)
                depth = penetration_depth(local, mv, mf)
                if depth.max() <= 0.0:
                    return total, True
                idx = np.flatnonzero(depth > 0)
                cp, _, _ = pq.on_surface(local[idx])
                vec = ((cp - local[idx]) @ Rf[t].T) * float(obj_s)             # local -> world
                w = depth[idx][:, None]; dvec = (vec * w).sum(0) / max(w.sum(), 1e-9)
                n = np.linalg.norm(dvec)
                dvec = dvec / n if n > 1e-9 else np.array([0, 0, 1.0])
                total = total + dvec * (depth.max() * float(obj_s) + 0.01)
            return total, False

        from scipy.ndimage import maximum_filter1d
        # SMOOTH rounds: per-frame escape vectors -> magnitude envelope (dilate + smooth + margin)
        # along the smoothed direction, applied; repeat while penetration remains. One round left
        # 40-70% of frames to the per-frame residual (<= 9 cm single-frame steps, |a| 3-5 on the
        # test episodes) because the escape direction turns as the hand moves round the object.
        push_s = np.zeros(len(fr), np.float32); mag0 = None
        # coarse-to-fine rounds (dilation window, direction sigma); a 300-point hand subsample
        # keeps the rounds affordable, the final per-frame residual uses every hand vertex
        for rnd, (win, dsig) in enumerate(((25, 6.0), (15, 4.0), (9, 2.0), (5, 1.0))[:PUSH_ROUNDS]):
            pv = np.zeros((len(fr), 3), np.float32)
            for t in range(len(fr)):
                pv[t], _ = _escape(t, t3[t], sub)
            mag = np.linalg.norm(pv, axis=1)
            if mag0 is None: mag0 = mag.copy()
            if mag.max() <= 0.0:
                break
            dirs = np.zeros_like(pv); env = np.zeros(len(fr), np.float32)
            for sl in stretches(fr):
                if sl.stop - sl.start >= 3:
                    env[sl] = gaussian_filter1d(maximum_filter1d(mag[sl], size=win, mode="nearest"), dsig, mode="nearest") + 0.01
                    dsm = gaussian_filter1d(pv[sl], dsig, axis=0, mode="nearest")
                    nrm = np.linalg.norm(dsm, axis=1, keepdims=True)
                    dirs[sl] = np.where(nrm > 1e-6, dsm / np.maximum(nrm, 1e-6), 0.0)
                else:
                    env[sl] = mag[sl]; dirs[sl] = pv[sl] / np.maximum(mag[sl][:, None], 1e-9)
            t3 = t3 + env[:, None] * dirs; push_s = push_s + env
        mag = mag0
        # per-frame residual, minimal escape again (small by construction)
        resid = np.zeros(len(fr), np.float32)
        for t in range(len(fr)):
            vec, ok_ = _escape(t, t3[t]); resid[t] = np.linalg.norm(vec); t3[t] = t3[t] + vec
        push = push_s + resid
        full = np.zeros((T, 3), np.float32)
        idx = np.clip(np.searchsorted(fr, np.arange(T)), 0, len(fr) - 1)
        left = np.clip(idx - 1, 0, len(fr) - 1)
        pick = np.where(np.abs(fr[idx] - np.arange(T)) <= np.abs(fr[left] - np.arange(T)), idx, left)
        full[:] = t3[pick]
        print(f"  ep{ep:02d} push-out(closest): raw max {mag.max()*100:.0f} cm (median of pushed {np.median(mag[mag>0])*100 if (mag>0).any() else 0:.0f}), "
              f"smoothed max {push_s.max()*100:.0f} cm, residual max {resid.max()*100:.1f} cm, frames with residual {(resid>0).sum()}/{len(fr)}", flush=True)
        return full, obj_s
    push = np.zeros(len(fr), np.float32); dirs = np.zeros((len(fr), 3), np.float32)
    for t in range(len(fr)):
        dirv = t3[t] - hand_c[t]; n = np.linalg.norm(dirv)
        dirv = dirv / n if n > 1e-6 else np.array([0, 0, 1.0])
        # coarse-to-fine: a 10->15->20->30 cm ladder overshot by a full object width (bowl ended
        # 29 cm from the hand, 40 px off its mask); refine to 0.5 cm so the hand ends ON the surface
        lo, hi = 0.0, None
        for step in (0.0, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55):
            local = ((hand_pts[t] - (t3[t] + step * dirv)) @ Rf[t]) / float(obj_s)
            if penetration_depth(local, mv, mf).max() <= 0.0:
                hi = step; break
            lo = step
        if hi is None:
            push[t] = 0.55
        else:
            while hi - lo > 0.005:
                mid = 0.5 * (lo + hi)
                local = ((hand_pts[t] - (t3[t] + mid * dirv)) @ Rf[t]) / float(obj_s)
                if penetration_depth(local, mv, mf).max() <= 0.0: hi = mid
                else: lo = mid
            push[t] = hi
        dirs[t] = dirv
    # TWO-PASS PUSH. Applying push[t] per frame put a 43 cm step between neighbouring frames
    # (ep15) on top of the smoothed trajectory: ACC-O proxy 2.21 vs the reference's 0.074.
    # Pass 1: the smoothed push (sigma 6, per stretch) carries the bulk and is continuous.
    # Pass 2: a per-frame residual push clears whatever penetration smoothing re-introduced;
    # it is bounded by the smoothing error, so the discontinuity it adds is small.
    # Smoothing a SPARSE spike train thins it out (a 30 cm push on 3 frames becomes a 2 cm bump
    # over 30 frames), so the residual pass had to redo the whole push per frame and the steps
    # came back (place5: residual max 25 cm on 99 frames in ep15, ACC-O proxy 1.68). The smooth
    # curve must be an UPPER ENVELOPE of the required push: max-filter (dilate) first, then
    # Gaussian, then one more max with the requirement. Near a contact the object is held a few
    # cm further from the hand for ~0.4 s -- cm-level CD-O cost against a 20x ACC-O penalty.
    from scipy.ndimage import maximum_filter1d
    push_s = push.copy()
    for sl in stretches(fr):
        if sl.stop - sl.start >= 3:
            env = maximum_filter1d(push[sl], size=25, mode="nearest")
            env = gaussian_filter1d(env, 6.0, mode="nearest")
            push_s[sl] = np.maximum(env, push[sl] * 0.0)   # env >= 0; residual pass handles any gap
            # The push DIRECTION was per frame: when the hand passes near the object's centre the
            # hand->centre vector is ~0 and its direction is noise, so a smooth magnitude times a
            # flipping direction still made single-frame xy spikes (place6: ep7 37.5 cm at one
            # frame, ep2 16 cm, with stable masks and stable depth). Smooth the direction too.
            dsm = gaussian_filter1d(dirs[sl], 6.0, axis=0, mode="nearest")
            nrm = np.linalg.norm(dsm, axis=1, keepdims=True)
            dirs[sl] = np.where(nrm > 1e-6, dsm / np.maximum(nrm, 1e-6), dirs[sl])
    t3 = t3 + push_s[:, None] * dirs

    def _need(t):
        """extra push along dirs[t] (m) that clears the hand at frame t from the current t3."""
        local = ((hand_pts[t] - t3[t]) @ Rf[t]) / float(obj_s)
        if penetration_depth(local, mv, mf).max() <= 0.0:
            return 0.0
        lo, hi = 0.0, None
        for step in (0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30):
            local = ((hand_pts[t] - (t3[t] + step * dirs[t])) @ Rf[t]) / float(obj_s)
            if penetration_depth(local, mv, mf).max() <= 0.0:
                hi = step; break
            lo = step
        if hi is None:
            return 0.30
        while hi - lo > 0.005:
            mid = 0.5 * (lo + hi)
            local = ((hand_pts[t] - (t3[t] + mid * dirs[t])) @ Rf[t]) / float(obj_s)
            if penetration_depth(local, mv, mf).max() <= 0.0: hi = mid
            else: lo = mid
        return hi
    # The first-pass push was measured along the PER-FRAME direction, then applied along the
    # SMOOTHED one, so the clearance it buys is not what was measured; the per-frame residual
    # then re-did up to 30 cm in single-frame steps (moge2: ep28 |a| 55 cm/frame^2 at one frame,
    # ACC-O proxy 1.63). Repeat the ENVELOPE pass against the actual residual (measured along
    # the smoothed direction) before the per-frame pass: the per-frame pass then only covers the
    # smoothing error, which is cm-level.
    for _ in range(3):
        need = np.array([_need(t) for t in range(len(fr))], np.float32)
        if need.max() <= 0.0:
            break
        env = np.zeros_like(need)
        for sl in stretches(fr):
            if sl.stop - sl.start >= 3:
                # per-frame rotation (track) makes the required push vary faster than a size-25
                # window follows: sub_track1 still had 26.8 cm single-frame residuals on ep25.
                # Wider dilation + a 2 cm margin, three rounds.
                env[sl] = gaussian_filter1d(maximum_filter1d(need[sl], size=37, mode="nearest"), 6.0, mode="nearest") + 0.02
            else:
                env[sl] = need[sl]
        env = np.where(need > 0, np.maximum(env, 0.0), env)
        push_s = push_s + env
        t3 = t3 + env[:, None] * dirs
    resid = np.zeros(len(fr), np.float32)
    for t in range(len(fr)):
        resid[t] = _need(t)
        t3[t] = t3[t] + resid[t] * dirs[t]
    push = push_s + resid
    full = np.zeros((T, 3), np.float32)
    idx = np.clip(np.searchsorted(fr, np.arange(T)), 0, len(fr) - 1)
    left = np.clip(idx - 1, 0, len(fr) - 1)
    pick = np.where(np.abs(fr[idx] - np.arange(T)) <= np.abs(fr[left] - np.arange(T)), idx, left)
    full[:] = t3[pick]
    print(f"  ep{ep:02d} push-out: smoothed max {push_s.max()*100:.0f} cm, residual max {resid.max()*100:.1f} cm, "
          f"frames with residual {(resid>0).sum()}/{len(push)}", flush=True)
    return full, obj_s


def _sam3d_as_body(path: Path) -> dict:
    """SAM 3D Body per-frame output -> the same dict layout the fitter writes."""
    from src.fit_mhr import POSE_UNIT_M, POSE_SIGN
    z = np.load(path)
    ok = z["ok"]
    if not ok.any():
        raise SystemExit(f"{path}: SAM 3D Body produced no frames")
    mp, ct = z["model_params"].copy(), z["cam_t"]
    # frames SAM3D missed: borrow the nearest good frame so the sequence is dense
    good = np.where(ok)[0]
    near = good[np.abs(good[None, :] - np.arange(len(ok))[:, None]).argmin(1)]
    mp, ct = mp[near], ct[near]
    pose = mp[:, :136].astype(np.float32)
    pose[:, :3] = (ct * (10.0 / (POSE_UNIT_M * 10.0)) * np.array(POSE_SIGN, np.float32)).astype(np.float32)
    scales = np.median(mp[good, 136:], axis=0).astype(np.float32)
    shape = np.median(z["shape"][good], axis=0).astype(np.float32)
    return dict(frame_ids=z["frame_ids"], pose=pose, scales=scales, shape=shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object", choices=["none", "mesh", "place"], default="mesh",
                    help="'none' keeps the unit-box placeholder; 'mesh' uses the reconstructed mesh")
    ap.add_argument("--body", type=Path, default=ROOT / "work/body")
    ap.add_argument("--body-source", choices=["fit", "sam3d"], default="fit",
                    help="'sam3d' reads work/sam3d/epNNNNNN.npz (SAM 3D Body per-frame output) and "
                         "builds pose from model_params[:136] with pose[0:3] taken from pred_cam_t "
                         "(verified in our camera frame by reprojection: 20 px from the VitPose "
                         "centroid; every flipped hypothesis was 485 px off or behind the camera). "
                         "shape and scales are the per-episode MEDIAN over good frames, because "
                         "the submission requires one identity per episode.")
    ap.add_argument("--reground", action="store_true",
                    help="After smoothing, slide each frame along the floor normal so the lowest "
                         "foot returns to the floor. Smoothing the pose averages the feet off the "
                         "ground it was fitted onto (0.44 cm -> 4.70 cm at sigma 6); the "
                         "correction is itself smoothed, so it restores contact (-> 2.09 cm) "
                         "without putting the jitter back (|a_pred| 0.3665 -> 0.3718).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: skip episodes whose npz already exists (the PEN push-out is CPU "
                         "heavy and the harness kills long foreground jobs)")
    ap.add_argument("--eps", type=int, nargs="*", default=list(range(30)))
    ap.add_argument("--obj-depth-stat", choices=["median", "p20"], default="p20")
    ap.add_argument("--mesh-dir", default="work/meshes", help="work/meshes_v2 has the re-run meshes with layout poses")
    ap.add_argument("--obj-rot", choices=["identity", "layout", "track"], default="identity")
    ap.add_argument("--obj-depth-scale", type=float, default=1.0, help="multiply the MoGe-ratio object depth (probe: nearest-surface p20 beat the median by 4.6 cm on CD-O, suggesting our depths are too far)")
    ap.add_argument("--no-thin-rule", action="store_true", help="disable hand-depth anchoring for thin objects (hoops)")
    ap.add_argument("--obj-size", choices=["consensus", "moge"], default="consensus", help="object metric size: 3-episode hand-depth consensus or from the MoGe placement depth")
    ap.add_argument("--push", choices=["centroid", "closest"], default="centroid", help="push-out direction: hand->centroid (old) or minimal escape to the closest surface")
    ap.add_argument("--contact-anchor", action="store_true", help="shift MoGe depth so contact frames put the object at the hand")
    ap.add_argument("--rot-tag", default="", help="suffix of the tracked rotation file (e.g. _ss)")
    ap.add_argument("--obj-depth", choices=["mask", "moge"], default="mask",
                    help="object depth source: mask size (default) or MoGe object/person depth ratio")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                    help="Gaussian sigma in frames, applied to pose WITHIN each scored stretch. "
                         "Depth is the one direction 2D keypoints do not constrain, so the limbs "
                         "swing in depth; that shows up as |a_pred| 11x the reference's own joint "
                         "acceleration and, we think, as the CD-H regression too. Post-hoc "
                         "filtering beats in-loop penalties here, which saturate: w_jacc=5 left "
                         "|a_pred| at 2.8x while driving reprojection error to 66 px.")
    a = ap.parse_args()
    global OBJ_DEPTH, OBJ_DEPTH_COL, CONTACT_ANCHOR, PUSH_MODE, OBJ_SIZE, OBJ_DEPTH_SCALE, THIN_RULE
    OBJ_DEPTH = a.obj_depth; OBJ_DEPTH_COL = 0 if a.obj_depth_stat == "median" else 1; CONTACT_ANCHOR = a.contact_anchor; PUSH_MODE = a.push; OBJ_SIZE = a.obj_size; OBJ_DEPTH_SCALE = a.obj_depth_scale; THIN_RULE = not a.no_thin_rule
    a.out.mkdir(parents=True, exist_ok=True)

    eps_len = {json.loads(l)["episode_index"]: json.loads(l)["length"]
               for l in (ROOT / "data/hf/track_1/meta/episodes.jsonl").open()}
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in (ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl").open()}
    scored = json.load((ROOT / "work/scored_frames_summary.json").open())
    ground = json.load((ROOT / "work/camera/ground.json").open())

    report = []
    for ep in a.eps:
        if a.skip_existing and (a.out / f"episode_{ep:06d}.npz").exists():
            continue
        T = max(eps_len[ep], scored[str(ep)]["last"] + 1)
        if a.body_source == "sam3d":
            b = _sam3d_as_body(ROOT / f"work/sam3d/ep{ep:06d}.npz")
        else:
            b = np.load(a.body / f"ep{ep:06d}.npz")
        fids, pose_s = b["frame_ids"], b["pose"].copy()
        if a.smooth > 0:
            from scipy.ndimage import gaussian_filter1d
            # per stretch: the metric never differences across a gap, and smoothing across one
            # would drag the two sides of a 360-frame gap together
            for sl in stretches(fids):
                if sl.stop - sl.start >= 3:
                    pose_s[sl] = _smooth_wrap_safe(pose_s[sl], a.smooth)
            if a.reground:
                from src.mhr_local import MHR
                global _MHR
                try:
                    mhr = _MHR
                except NameError:
                    mhr = _MHR = MHR()
                g = ground[meta[ep]["camera"]]
                gn = np.array(g["normal"])
                pr = np.concatenate([pose_s, np.broadcast_to(b["scales"], (len(pose_s), 68))], 1)
                _, j = mhr.forward(pr, b["shape"])
                # SCALE-FREE: restore the lowest foot to this stretch's own MEAN height, not to
                # the floor's measured distance. Pinning to the absolute distance mixes the body's
                # metric scale with MoGe's and is what sent CD-H from 82.68 to 96.92.
                h = (j[:, list(FOOT_JOINTS)] @ gn).min(1)
                for sl in stretches(fids):
                    if sl.stop - sl.start >= 3:
                        dh = gaussian_filter1d(h[sl].mean() - h[sl], a.smooth, mode="nearest")
                        pose_s[sl, :3] += (dh[:, None] * gn[None, :]) / (np.array(POSE_SIGN) * POSE_UNIT_M)
        # nearest-solved-frame fill for the rows the scorer never reads
        idx = np.clip(np.searchsorted(fids, np.arange(T)), 0, len(fids) - 1)
        left = np.clip(idx - 1, 0, len(fids) - 1)
        pick = np.where(np.abs(fids[idx] - np.arange(T)) <= np.abs(fids[left] - np.arange(T)),
                        idx, left)
        pose = pose_s[pick].astype(np.float32)

        obj_mesh = ROOT / a.mesh_dir / f"{meta[ep]['object']}.glb"
        R_seq = None
        if a.obj_rot == "track":
            # per-episode mesh + per-frame rotation from the silhouette tracker; non-scored rows
            # take the nearest tracked frame's rotation
            obj_mesh = ROOT / f"work/objtrack/ep{ep:06d}_mesh.glb"
            rp = ROOT / f"work/objtrack/ep{ep:06d}_rot{a.rot_tag}.npz"
            if not rp.exists(): rp = ROOT / f"work/objtrack/ep{ep:06d}_rot.npz"
            rz = np.load(rp); rf, RR = rz["frame_ids"], rz["R"]
            ii = np.clip(np.searchsorted(rf, np.arange(T)), 0, len(rf) - 1); ll = np.clip(ii - 1, 0, len(rf) - 1)
            pk = np.where(np.abs(rf[ii] - np.arange(T)) <= np.abs(rf[ll] - np.arange(T)), ii, ll)
            R_seq = RR[pk].astype(np.float32)
        if a.object in ("mesh", "place") and obj_mesh.exists():
            # "place" fell through to the 0.3 m placeholder box in v3: every object_scale came
            # out exactly 3.33x (=1/0.3) its metric size, the push-out ran against a box, and
            # the box would have been submitted. One condition, whole export poisoned.
            mesh = trimesh.load(str(obj_mesh), force="mesh", process=False)
            if a.obj_rot == "layout":
                # ORIENTATION, BAKED INTO THE EPISODE MESH. SAM 3D Objects' layout quaternion
                # (wxyz, matrix transposed -- verified against its own pointmap, median 0.066 m
                # vs 0.158 identity, and by silhouette IoU 0.65 vs 0.41) gives the object's
                # rotation in the crop camera; R_ray carries it to the full camera (a crop at the
                # image edge is ~30 deg off-axis). Constant over the episode (v1): rotating the
                # vertices once keeps object_rotation = I and leaves placement/push-out untouched.
                from scipy.spatial.transform import Rotation as _Rot
                pj = json.load((ROOT / a.mesh_dir / f"{meta[ep]['object']}_pose.json").open())
                q = np.array(pj["rotation_quat"]); R = _Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix().T
                Kc = json.load((ROOT / "work/camera/intrinsics.json").open())[meta[pj["view"]["ep"]]["camera"]]
                b_ = pj["view"]["bbox"]; ray = np.array([((b_[0] + b_[2]) / 2 - Kc["cx"]) / Kc["fx"], ((b_[1] + b_[3]) / 2 - Kc["cy"]) / Kc["fy"], 1.0]); ray /= np.linalg.norm(ray)
                ax = np.cross([0, 0, 1.0], ray); sn = np.linalg.norm(ax)
                R_ray = _Rot.from_rotvec(ax / sn * np.arctan2(sn, ray[2])).as_matrix() if sn > 1e-8 else np.eye(3)
                R_full = R_ray @ R
                mesh = trimesh.Trimesh(np.asarray(mesh.vertices) @ R_full.T, np.asarray(mesh.faces), process=False)
                print(f"  ep{ep} object rotation baked: rotvec {np.round(_Rot.from_matrix(R_full).as_rotvec(), 3).tolist()} (|θ| {np.degrees(np.linalg.norm(_Rot.from_matrix(R_full).as_rotvec())):.0f} deg)")
        else:
            mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        # Write the BUDGETED mesh (4096 v / 4096 f), i.e. exactly what the host scores after the
        # packer's own decimation (idempotent to re-budget). The raw SAM3D-Objects mesh has
        # 100k-400k faces: the local PEN self-check was doing SDF against it for 9877 frames
        # (place7: 35 min and counting at 800% CPU) and was not even testing the right surface.
        if len(mesh.faces) > 4096 or len(mesh.vertices) > 4096:
            sys.path.insert(0, str(ROOT / "work/kit/v2d_submission_kit"))
            from v2dlb.mesh_budget import budget_mesh
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as _f:
                mesh.export(_f.name)
            try:
                bv, bf = budget_mesh(_f.name, 4096, 4096)
            finally:
                os.unlink(_f.name)                                           # was leaking 8-57 MB per call into /tmp
            bf = np.asarray(bf, np.int64); bf = bf[(bf[:, 0] != bf[:, 1]) | (bf[:, 1] != bf[:, 2])]
            mesh = trimesh.Trimesh(np.asarray(bv, np.float64), bf, process=False)
        # CENTRE the submitted mesh on its vertex mean: object_translation is then the centroid
        # itself and a per-frame rotation (track) turns the object about its centroid instead of
        # about the canonical origin, so rotation jitter no longer moves the centroid (ACC-O went
        # 0.074 -> 0.134 on submission #9 with tracked rotations).
        mesh = trimesh.Trimesh(np.asarray(mesh.vertices) - np.asarray(mesh.vertices).mean(0), np.asarray(mesh.faces), process=False)
        mesh.export(a.out / f"episode_{ep:06d}_object.obj")

        obj_R = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
        obj_t = np.zeros((T, 3), np.float32); obj_s = np.float32(1.0)
        if a.object == "place":
            global OBJ_NAME
            OBJ_NAME = meta[ep]["object"]
            obj_t, obj_s = _place_object(ep, meta[ep]["camera"], pose, b["scales"], b["shape"],
                                         fids, mesh, a.smooth, T, R_seq=R_seq)
            if R_seq is not None:
                obj_R = R_seq
        np.savez(a.out / f"episode_{ep:06d}.npz",
                 pose=pose, scales=b["scales"].astype(np.float32), shape=b["shape"].astype(np.float32),
                 object_rotation=obj_R, object_translation=obj_t, object_scale=obj_s)
        report.append(dict(ep=ep, T=int(T), solved=int(len(fids)),
                           filled=int(T - len(fids)), object=meta[ep]["object"]))
    print(json.dumps(dict(episodes=len(report), total_rows=sum(r["T"] for r in report),
                          solved=sum(r["solved"] for r in report)), indent=1))


if __name__ == "__main__":
    main()
