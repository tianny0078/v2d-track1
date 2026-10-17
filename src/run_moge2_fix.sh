#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
echo "=== re-export residual episodes $(TZ=Asia/Shanghai date +%H:%M)"
python3 src/export_episodes.py --out work/sub_moge2 --eps 1 4 10 15 24 25 27 28 --body-source sam3d --object place --obj-depth moge --smooth 3 2>&1 | grep -E "push-out|Traceback|Error"
python3 src/acc_gate.py work/sub_moge2 2>&1 | grep RESULT
echo "=== DONE $(TZ=Asia/Shanghai date +%H:%M)"
