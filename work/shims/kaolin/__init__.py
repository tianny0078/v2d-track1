"""Minimal kaolin stand-in.

SAM 3D Objects' mesh path touches kaolin exactly once -- flexicubes imports
`kaolin.utils.testing.check_tensor`, a pure-Python assertion helper. kaolin 0.17 has no wheel for
torch 2.9/cu128 and builds against CUDA 12.1, which does not support sm_120, so installing it to
get one assert is not worth it. The gaussian-splat path needs gsplat and is not used here.
"""
