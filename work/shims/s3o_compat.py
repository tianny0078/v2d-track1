"""Import this BEFORE sam3d_objects.

Two incompatibilities between the pinned SAM 3D Objects source and what we can install on a
Blackwell box, neither of which touches the reconstruction maths:

1. `utils3d.numpy.depth_edge` was renamed `depth_map_edge` upstream. Same signature
   (depth, rtol=, mask=), and it is only reached from scene-layout post-optimisation, which
   single-object mesh reconstruction does not use. We alias rather than pin an old utils3d,
   which would drag the rest of the environment backwards.
2. `sam3d_objects.init` is absent from the checkout on /mnt/nas3, so LIDRA_SKIP_INIT=1 is set
   here rather than relied on being exported by whoever runs this.
"""
from __future__ import annotations
import os

os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import utils3d.numpy as _u3n  # noqa: E402

if not hasattr(_u3n, "depth_edge"):
    from utils3d.numpy.maps import depth_map_edge as _dme
    _u3n.depth_edge = _dme
