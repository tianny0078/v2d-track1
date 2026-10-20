#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
bash src/run_export_parallel.sh work/sub_v8 6 --body work/body_flow_px --body-source fit --object place --obj-depth moge --obj-depth-scale 0.93 --obj-rot track --rot-tag _best --smooth 3 2>&1 | tail -n 1
python3 src/acc_gate.py work/sub_v8 2>&1 | grep RESULT; python3 src/pen_gate.py work/sub_v8 cuda:1 2>&1 | grep RESULT
bash src/run_export_parallel.sh work/sub_v9 6 --body work/body_flow_px --body-source fit --object place --obj-depth moge --no-thin-rule --obj-rot track --rot-tag _best --smooth 3 2>&1 | tail -n 1
python3 src/acc_gate.py work/sub_v9 2>&1 | grep RESULT; python3 src/pen_gate.py work/sub_v9 cuda:1 2>&1 | grep RESULT
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
