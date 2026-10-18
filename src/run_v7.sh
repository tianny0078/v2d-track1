#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
bash src/run_export_parallel.sh work/sub_v7 6 --body work/body_flow_px --body-source fit --object place --obj-depth moge --obj-depth-stat median --obj-rot track --rot-tag _best --smooth 3 2>&1 | tail -n 1
python3 src/acc_gate.py work/sub_v7 2>&1 | grep RESULT
python3 src/pen_gate.py work/sub_v7 cuda:1 2>&1 | grep RESULT
echo "=== DONE $(TZ=Asia/Shanghai date +%H:%M)"
