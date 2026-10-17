#!/bin/bash
# Variant: FlowHMR fit WITH the 2D reprojection term (w_px 5), after run_flowhmr_all finishes.
cd /home/tianyuan/projects/v2d_track1
while pgrep -f "^bash src/run_flowhmr_all.sh" >/dev/null; do sleep 30; done
echo "=== flow2 fit start $(TZ=Asia/Shanghai date +%H:%M)"
python3 src/fit_flowhmr.py --ep $(seq 0 29 | tr '\n' ' ') --device cuda:2 --w-px 5 --out work/body_flow_px 2>&1 | grep -E "RESULT|Error|Traceback" | cut -c1-200
echo "=== fitted $(ls work/body_flow_px | wc -l)/30"
python3 src/export_episodes.py --out work/sub_flow2 --body work/body_flow_px --body-source fit --object place --obj-depth moge --smooth 3 > logs/export_flow2.log 2>&1
echo "=== exported $(ls work/sub_flow2 | wc -l) files"
python3 src/acc_gate.py work/sub_flow2 2>&1 | grep RESULT
echo "=== DONE $(TZ=Asia/Shanghai date +%H:%M)"
