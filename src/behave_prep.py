#!/usr/bin/env python3
"""BEHAVE -> local validation inputs + ground truth (one camera, one clip per sequence).

Reads data/behave/{params,calibs,objects,time,video}; writes
  work/behave/<seq>/frames/FFFFFF.jpg   undistorted colour frames of the chosen Kinect (native 2048x1536)
  work/behave/<seq>/meta.json           fps, pinhole intrinsics, object name/prompt, frame range, provenance
  work/behave_gt/<seq>/gt_smplh.npz     SMPL-H GT vertices/joints in the chosen camera frame (OpenCV, m)
  work/behave_gt/<seq>/episode_000000_object.obj   centred simplified BEHAVE template (the registration mesh)
  work/behave_gt/<seq>/object_pose.npz  per-frame object rotation/translation in the camera frame
  work/behave_gt/<seq>/scored_frames.json, overlay_check.jpg

Conventions (read from github.com/xiexh20/behave-dataset, not assumed):
  * 30 fps params `frame_times` are seconds on the per-camera timestamp clock (`time/*.time.json` / 1e6);
    the image for time t is picked exactly as tools/video2images.py does it (closest actual stamp over
    all 4 cameras, then the closest frame of each camera to that stamp).
  * World frame = Kinect 1 colour camera; calibs/<Date>/config/<k>/config.json holds local->world
    (p_w = R p_k + t); the colour cameras are OpenCV frames, so world->k is our submission frame directly.
  * SMPL-H fits: MANO v1.2 SMPL-H pkl (10 betas), full axis-angle hands, no hand mean (libsmpl/SMPL_Layer).
  * Object: angles (rotvec) / trans applied to the *centred* simplified template <obj>_fNNNN.ply.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2, torch, trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.behave_smplh import SMPLH  # noqa: E402

B = ROOT / "data/behave"
SMPLH_V12 = {g: ROOT / f"work/behave_gt/_assets/smplh_{g}_v12.npz" for g in ("male", "female")}
SIMPLIFIED = {"backpack": "backpack_f1000", "basketball": "basketball_f1000", "boxlarge": "boxlarge_f1000",
              "boxtiny": "boxtiny_f1000", "boxlong": "boxlong_f1000", "boxsmall": "boxsmall_f1000",
              "boxmedium": "boxmedium_f1000", "chairblack": "chairblack_f2500", "chairwood": "chairwood_f2500",
              "monitor": "monitor_closed_f1000", "keyboard": "keyboard_f1000", "plasticcontainer": "plasticcontainer_f1000",
              "stool": "stool_f1000", "tablesquare": "tablesquare_f2000", "toolbox": "toolbox_f1000",
              "suitcase": "suitcase_f1000", "tablesmall": "tablesmall_f1000", "yogamat": "yogamat_f1000",
              "yogaball": "yogaball_f1000", "trashbin": "trashbin_f1000"}
PROMPT = {"chairwood": "a wooden chair.", "stool": "a round wooden stool.", "trashbin": "a plastic trash bin.",
          "boxmedium": "a cardboard box.", "plasticcontainer": "a plastic storage container.",
          "boxsmall": "a small cardboard box.", "tablesmall": "a small square table.", "toolbox": "a metal toolbox."}
#: clip = (camera, first GT frame index, length). Camera = most frames facing it with the body inside
#: the image (measured from the GT, see notes/behave-valset.md) AMONG the cameras whose colour video is
#: a clean 30 fps stream: Kinects 0 and 3 drop 20-40% of their frames on Date01/02 (1100-1250 stamps
#: for ~50 s), so only k1/k2 are usable. Window = 15 s with the most object motion on a contiguous
#: 30 fps grid and no listed error frame.
CLIPS = {"Date01_Sub01_chairwood_hand": (1, 585, 450), "Date01_Sub01_stool_move": (1, 960, 450),
         "Date01_Sub01_trashbin": (1, 270, 450), "Date01_Sub01_boxmedium_hand": (1, 375, 450),
         "Date01_Sub01_plasticcontainer": (1, 90, 450), "Date02_Sub02_boxsmall_hand": (1, 105, 450),
         "Date02_Sub02_tablesmall_move": (1, 585, 450), "Date02_Sub02_toolbox": (1, 240, 450)}


def kinect(date: str, k: int):
    c = json.load((B / f"calibs/calibs/{date}/config/{k}/config.json").open())
    Rl = np.array(c["rotation"]).reshape(3, 3); tl = np.array(c["translation"])
    cal = json.load((B / f"calibs/calibs/intrinsics/{k}/calibration.json").open())["color"]
    K = np.array([[cal["fx"], 0, cal["cx"]], [0, cal["fy"], cal["cy"]], [0, 0, 1]])
    D = np.array(cal["opencv"][4:])
    return Rl.T, -Rl.T @ tl, K, D, (cal["width"], cal["height"])   # world->local


def frame_indices(seq: str, k: int, times: np.ndarray) -> np.ndarray:
    stamps = {j: np.array(json.load((B / f"time/{seq}.{j}.time.json").open())["color"], float) / 1e6 for j in range(4)}
    idx = []
    for t in times:
        actual = np.array([stamps[j][np.abs(stamps[j] - t).argmin()] for j in range(4)])
        a = actual[np.abs(actual - t).argmin()]
        idx.append(int(np.abs(stamps[k] - a).argmin()))
    return np.array(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=list(CLIPS))
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()
    for seq in a.seqs:
        k, s, n = CLIPS[seq]
        date, obj = seq.split("_")[0], seq.split("_")[2]
        info = json.load((B / f"params/{seq}/info.json").open())
        z = np.load(B / f"params/{seq}/smpl_fit_all.npz"); o = np.load(B / f"params/{seq}/object_fit_all.npz")
        assert (z["frame_times"] == o["frame_times"]).all() and info["cat"] == obj
        sl = slice(s, s + n)
        times = np.array([float(t[1:]) for t in z["frame_times"][sl]])
        assert np.abs(np.diff(times) - 1 / 30).max() < 2e-3, "clip is not a contiguous 30 fps grid"
        Rw2l, tw2l, K, D, (W, H) = kinect(date, k)

        # ---- human GT (camera frame) -----------------------------------------------------
        sm = SMPLH(info["gender"], path=SMPLH_V12[info["gender"]])
        with torch.no_grad():
            v, j = sm(z["poses"][sl], z["betas"][sl], z["trans"][sl])
        v, j = v.numpy() @ Rw2l.T + tw2l, j.numpy() @ Rw2l.T + tw2l
        gt = ROOT / f"work/behave_gt/{seq}"; gt.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gt / "gt_smplh.npz", verts=v.astype(np.float32), joints=j.astype(np.float32),
                            poses=z["poses"][sl], betas=z["betas"][sl], trans=z["trans"][sl], gender=info["gender"],
                            frame_times=z["frame_times"][sl], Rw2l=Rw2l, tw2l=tw2l, K=K, camera=k)

        # ---- object GT ----------------------------------------------------------------------
        tmpl = trimesh.load(str(B / f"objects/objects/{obj}/{SIMPLIFIED[obj]}.ply"), force="mesh", process=False)
        centre = np.asarray(tmpl.vertices).mean(0)
        local = np.asarray(tmpl.vertices) - centre
        Rw = Rotation.from_rotvec(o["angles"][sl]).as_matrix()
        Rc = np.einsum("ij,tjk->tik", Rw2l, Rw); tc = o["trans"][sl] @ Rw2l.T + tw2l
        trimesh.Trimesh(local, np.asarray(tmpl.faces), process=False).export(gt / "episode_000000_object.obj")
        np.savez(gt / "object_pose.npz", rotation=Rc.astype(np.float32), translation=tc.astype(np.float32), scale=np.float32(1.0),
                 template=SIMPLIFIED[obj], centre=centre)
        json.dump({"frames": list(range(n)), "stretches": [[0, n]]}, (gt / "scored_frames.json").open("w"))

        # ---- frames ---------------------------------------------------------------------------
        fidx = frame_indices(seq, k, times)
        out = ROOT / f"work/behave/{seq}"; (out / "frames").mkdir(parents=True, exist_ok=True)
        meta = dict(sequence=seq, object=obj, object_prompt=PROMPT.get(obj, f"a {obj}."), gender=info["gender"],
                    camera=k, fps=30, width=W, height=H, undistorted=True,
                    fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                    gt_frame_range=[int(s), int(s + n)], gt_frame_times=[str(t) for t in z["frame_times"][sl]],
                    video=f"data/behave/video/{seq}.{k}.color.mp4", video_frame_idx=fidx.tolist(),
                    source=dict(params="behave-30fps-params-v1.tar", smpl_fit=str(z["save_name"]), object_fit=str(o["save_name"]),
                                calib=f"calibs/{date}/config + calibs/intrinsics/{k}", template=SIMPLIFIED[obj]))
        json.dump(meta, (out / "meta.json").open("w"), indent=1)
        if not a.no_frames:
            cap = cv2.VideoCapture(str(ROOT / meta["video"]))
            want = {}
            for i, f in enumerate(fidx):
                want.setdefault(int(f), []).append(i)
            assert n - len(want) <= 2, f"{seq}: camera {k} dropped {n - len(want)} frames inside the clip"
            m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)
            fi, got = 0, 0
            while got < n:
                ok, img = cap.read()
                if not ok:
                    raise SystemExit(f"{seq}: video ended at {fi}, wanted up to {fidx.max()}")
                if fi in want:
                    und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
                    for i in want[fi]:
                        cv2.imwrite(str(out / f"frames/{i:06d}.jpg"), und, [cv2.IMWRITE_JPEG_QUALITY, 95]); got += 1
                fi += 1
            cap.release()
            # overlay check: GT body + object projected with the pinhole K onto the undistorted frame
            tiles = []
            for i in (0, n // 2, n - 1):
                img = cv2.imread(str(out / f"frames/{i:06d}.jpg"))
                for pts, col in ((v[i], (0, 255, 0)), (local @ Rc[i].T + tc[i], (0, 0, 255))):
                    p = pts[pts[:, 2] > 0.1]; uv = (K @ p.T).T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
                    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                    img[uv[ok, 1], uv[ok, 0]] = col
                tiles.append(cv2.resize(img, (W // 3, H // 3)))
            cv2.imwrite(str(gt / "overlay_check.jpg"), np.concatenate(tiles, 1))
        print(json.dumps(dict(seq=seq, cam=k, frames=n, video_idx=[int(fidx[0]), int(fidx[-1])],
                              body_z=round(float(v[..., 2].mean()), 2), obj_z=round(float(tc[:, 2].mean()), 2),
                              obj_faces=int(len(tmpl.faces)))), flush=True)


if __name__ == "__main__":
    main()
