#!/usr/bin/env python3
"""Which rotation convention does SAM 3D Objects' layout quaternion use? Test at each object's
best-view frame: rotate the mesh by each candidate, place it at the mask centroid / mask-size
depth, rasterise its silhouette, IoU with the SAM2 mask. The convention that wins across objects
is the one we export. Candidates: quaternion order (xyzw / wxyz) x transpose x camera flip
(OpenGL y-up/z-back -> our y-down/z-forward = diag(1,-1,-1))."""
import json, sys, numpy as np, trimesh
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot
ROOT = Path(__file__).resolve().parents[1]
K = json.load(open(ROOT / "work/camera/intrinsics.json")); meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(ROOT / "data/hf/track_1/meta/episodes_metadata.jsonl")}
F = np.diag([1.0, -1.0, -1.0])

def silhouette(v, f, k, uv, z, S, H, W):
    v = v - v.mean(0); v = v / max(np.ptp(v, axis=0).max(), 1e-6) * S           # metric, centred
    X = (uv[0] - k["cx"]) / k["fx"] * z; Y = (uv[1] - k["cy"]) / k["fy"] * z
    P = v + np.array([X, Y, z]); u = k["fx"] * P[:, 0] / P[:, 2] + k["cx"]; vv = k["fy"] * P[:, 1] / P[:, 2] + k["cy"]
    im = Image.new("1", (W, H), 0); d = ImageDraw.Draw(im)
    for tri in f: d.polygon([(u[i], vv[i]) for i in tri], fill=1)
    return np.asarray(im, bool)

def main():
    out = {}
    for pj in sorted((ROOT / "work/meshes_v2").glob("*_pose.json")):
        pose = json.load(open(pj)); name = pose["object"]; v_ = pose["view"]; ep, fr = v_["ep"], v_["frame"]
        m = trimesh.load(ROOT / f"work/meshes_v2/{name}.glb", force="mesh"); verts = np.asarray(m.vertices); faces = np.asarray(m.faces)
        if len(faces) > 30000: faces = faces[np.random.RandomState(0).choice(len(faces), 30000, replace=False)]
        z = np.load(ROOT / f"work/masks/ep{ep:06d}.npz"); W = int(z["shape"][1]); i = int(np.searchsorted(z["frame_ids"], fr)); mask = np.unpackbits(z["packed"][i], axis=-1)[0, :, :W].astype(bool); H = mask.shape[0]
        ys, xs = np.where(mask); uv = (xs.mean(), ys.mean()); ext = max(np.ptp(xs), np.ptp(ys)); k = K[meta[ep]["camera"]]
        S = 0.5  # placeholder metric size; the silhouette SHAPE is what we compare, so scale to the mask extent instead:
        q = np.array(pose["rotation_quat"])
        # The crop is a virtual camera looking along the ray through the crop centre; the full
        # camera looks along z. A crop near the image edge is ~30 deg off-axis at FOV 86, so the
        # layout rotation must be carried over by the rotation that maps z to that ray.
        b = v_["bbox"]; ray = np.array([((b[0] + b[2]) / 2 - k["cx"]) / k["fx"], ((b[1] + b[3]) / 2 - k["cy"]) / k["fy"], 1.0]); ray /= np.linalg.norm(ray)
        axis = np.cross([0, 0, 1.0], ray); sn = np.linalg.norm(axis); ang = np.arctan2(sn, ray[2])
        R_ray = Rot.from_rotvec(axis / sn * ang).as_matrix() if sn > 1e-8 else np.eye(3)
        cands = {}
        for order in ("xyzw", "wxyz"):
            R = Rot.from_quat(q if order == "xyzw" else np.r_[q[1:], q[0]]).as_matrix()
            for tr in (False, True):
                Rm = R.T if tr else R
                for flip in (False, True):
                    Rc = F @ Rm if flip else Rm
                    cands[f"{order}{'^T' if tr else ''}{'·flip' if flip else ''}"] = R_ray @ Rc
        cands["identity"] = np.eye(3)
        res = {}
        for cname, Rc in cands.items():
            vr = verts @ Rc.T
            # scale so the projected extent matches the mask extent (removes depth/size from the test)
            best = 0.0
            for zz in (2.0, 3.0, 4.0):
                vv = vr - vr.mean(0); P0 = vv / max(np.ptp(vv, axis=0).max(), 1e-6)
                # projected extent of a unit-size object at depth zz: choose S so it matches ext px
                u0 = k["fx"] * P0[:, 0] / zz; v0 = k["fy"] * P0[:, 1] / zz; e0 = max(np.ptp(u0), np.ptp(v0))
                S = ext / max(e0, 1e-6)
                sil = silhouette(vr, faces, k, uv, zz, S, H, W)
                iou = (sil & mask).sum() / max((sil | mask).sum(), 1); best = max(best, iou)
            res[cname] = round(float(best), 3)
        out[name] = res
        top = sorted(res.items(), key=lambda x: -x[1])[:3]
        print(f"RESULT {name:18s} ep{ep} f{fr}: identity {res['identity']:.3f} | best {top}", flush=True)
    json.dump(out, open(ROOT / "work/meshes_v2/rot_convention.json", "w"), indent=1)
    names = list(out); conv = [c for c in out[names[0]] if c != "identity"]
    print("RESULT mean IoU per convention:", {c: round(float(np.mean([out[n][c] for n in names])), 3) for c in conv}, "identity", round(float(np.mean([out[n]["identity"] for n in names])), 3))

if __name__ == "__main__":
    main()
