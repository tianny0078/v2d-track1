#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
python3 src/export_episodes.py --out work/sub_rot_test --eps 22 25 8 12 --body-source sam3d --object place --obj-depth moge --mesh-dir work/meshes_v2 --obj-rot layout --smooth 3 2>&1 | grep -E "rotation baked|push-out|Traceback|Error"
echo "=== DONE"
