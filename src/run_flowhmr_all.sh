#!/bin/bash
# Wait for the token workers, then: FlowHMR generation (all 30) -> MHR fit -> export (sigma 3) -> gates.
cd /home/tianyuan/projects/v2d_track1
while [ "$(ls work/flowhmr_tokens/ | wc -l)" -lt 30 ] || pgrep -f "^python3 src/flowhmr_tokens.py" >/dev/null; do sleep 60; done
echo "=== tokens complete $(date +%H:%M)"
EPS=$(seq 0 29 | tr '\n' ' ')
python3 src/flowhmr_generate.py --ep $EPS --device cuda:2 2>&1 | grep -E '"ep"|Error|Traceback' 
echo "=== generated $(ls work/flowhmr_out | wc -l)/30 $(date +%H:%M)"
python3 src/fit_flowhmr.py --ep $EPS --device cuda:2 2>&1 | grep -E "RESULT|Error|Traceback"
echo "=== fitted $(ls work/body_flow | wc -l)/30 $(date +%H:%M)"
python3 src/export_episodes.py --out work/sub_flow1 --body work/body_flow --body-source fit --object place --smooth 3 2>&1 | tail -n 5
bash src/check_export.sh work/sub_flow1 2>&1 | grep -E "RESULT|PEN|FAIL|PASS"
echo "=== DONE $(date +%H:%M)"
