#!/usr/bin/env python3
"""Regenerate the pruned export H5s (images / human_masks / object_masks) of one sequence exactly as
step 02 (prep/prepare_mhr_wild_export.py) writes them, WITHOUT touching object_mesh/output_aligned.glb,
edex or wild_export.json (the FoundationPose poses are expressed in that aligned-mesh frame, so the
mesh must not be re-aligned). Run with the CARI4D PYTHONPATH from the cari4d source root.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import av
import h5py
import numpy as np

from prep.mhr_export_utils import MHR_CAMERA_NAMES
from prep.mhr_rgb_h5 import encode_rgb_jpeg, set_rgb_jpeg_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mask-h5", required=True)
    ap.add_argument("--export-seq", required=True)
    a = ap.parse_args()
    out = Path(a.export_seq); seq = out.name
    camera = MHR_CAMERA_NAMES[0]
    rgb_path = out / "images" / f"{camera}.h5"; human_path = out / "human_masks" / f"{camera}.h5"; object_path = out / "object_masks" / f"{camera}.h5"
    if all(p.is_file() for p in (rgb_path, human_path, object_path)):
        print("export h5 present"); return
    for p in (rgb_path, human_path, object_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(a.video); stream = container.streams.video[0]
    expected = int(stream.frames); height, width = int(stream.codec_context.height), int(stream.codec_context.width)
    tmp = [p.with_name(f".{p.name}.{os.getpid()}.tmp") for p in (rgb_path, human_path, object_path)]
    with h5py.File(a.mask_h5, "r") as src, h5py.File(tmp[0], "w") as rh, h5py.File(tmp[1], "w") as hh, h5py.File(tmp[2], "w") as oh:
        rd = rh.create_dataset("frames", shape=(expected,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        hd = hh.create_dataset("frames", shape=(expected, height, width), dtype=np.uint8, chunks=(1, height, width), compression="lzf", shuffle=True)
        od = oh.create_dataset("frames", shape=(expected, height, width), dtype=np.uint8, chunks=(1, height, width), compression="lzf", shuffle=True)
        set_rgb_jpeg_metadata(rd, (height, width, 3))
        for h in (rh, hh, oh):
            h.attrs["complete"] = False; h.attrs["sequence"] = seq; h.attrs["camera_id"] = 0
        n = 0
        for index, frame in enumerate(container.decode(stream)):
            name = f"{index:06d}"
            rgb = frame.to_ndarray(format="rgb24")
            human = np.asarray(src[f"{seq}/{name}-k0.person_mask.png"][()], dtype=bool); obj = np.asarray(src[f"{seq}/{name}-k0.obj_rend_mask.png"][()], dtype=bool)
            rd[index] = encode_rgb_jpeg(rgb); hd[index] = human.astype(np.uint8) * 255; od[index] = obj.astype(np.uint8) * 255
            n += 1
        assert n == expected, (n, expected)
        for h in (rh, hh, oh):
            h.attrs.modify("complete", True); h.flush()
    container.close()
    for t, final in zip(tmp, (rgb_path, human_path, object_path)):
        os.replace(t, final)
    print(f"regenerated export h5 for {seq}: {n} frames")


if __name__ == "__main__":
    main()
