#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
bash src/run_export_parallel.sh work/sub_v4b 6 --body work/body_depth --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --smooth 5 2>&1 | tail -n 1
python3 src/body_proxy.py work/sub_track2 work/sub_v4b 2>&1 | grep RESULT
python3 src/acc_gate.py work/sub_v4b 2>&1 | grep RESULT
python3 src/pen_gate.py work/sub_v4b cuda:1 2>&1 | grep RESULT
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
