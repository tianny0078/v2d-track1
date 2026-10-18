#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
for ep in 22 29 8; do
python3 src/export_episodes.py --out work/sub_v3b --eps $ep --body work/body_flow_px --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --push closest --smooth 3 2>&1 | grep -E "push-out|Traceback|Error" | cut -c1-200 &
done; wait
python3 src/acc_gate_eps.py work/sub_v3b 22 29 8 2>&1 | grep RESULT
echo "=== DONE"
