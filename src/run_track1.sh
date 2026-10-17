#!/bin/bash
# After the rotation tracker finishes: export FlowHMR(px) body + MoGe depth + tracked rotation, gate it.
cd /home/tianyuan/projects/v2d_track1
while [ "$(ls state/objtrack/ 2>/dev/null | wc -l)" -lt 30 ]; do sleep 60; done
echo "=== tracking complete $(TZ=Asia/Shanghai date +%H:%M)"
python3 src/export_episodes.py --out work/sub_track1 --body work/body_flow_px --body-source fit --object place --obj-depth moge --obj-rot track --smooth 3 > logs/export_track1.log 2>&1
echo "=== exported $(ls work/sub_track1 | wc -l) files $(TZ=Asia/Shanghai date +%H:%M)"
grep -E "push-out" logs/export_track1.log | awk '{for(i=1;i<=NF;i++) if($i=="residual") print $1, $(i+2)}' | awk '$2+0>3' | tr '\n' ' '; echo
python3 src/acc_gate.py work/sub_track1 2>&1 | grep RESULT
python3 src/obj_hand_dist.py work/sub_moge2 work/sub_track1 2>&1 | grep RESULT
echo "=== DONE $(TZ=Asia/Shanghai date +%H:%M)"
