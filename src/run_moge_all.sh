#!/bin/bash
# Wait for MoGe object depth on all 30, then export with the depth-ratio placement and gate it.
cd /home/tianyuan/projects/v2d_track1
while [ "$(ls state/objdepth/ 2>/dev/null | wc -l)" -lt 30 ]; do sleep 60; done
echo "=== objdepth complete $(TZ=Asia/Shanghai date +%H:%M)"
python3 src/export_episodes.py --out work/sub_moge2 --body-source sam3d --object place --obj-depth moge --smooth 3 > logs/export_moge2.log 2>&1
echo "=== exported $(ls work/sub_moge2 | wc -l) files $(TZ=Asia/Shanghai date +%H:%M)"
grep -E "moge depth" logs/export_moge2.log | cut -c1-120
python3 src/acc_gate.py work/sub_moge2 2>&1 | grep RESULT
python3 src/obj_hand_dist.py work/sub_place7 work/sub_moge2 2>&1 | grep RESULT
echo "=== DONE $(TZ=Asia/Shanghai date +%H:%M)"
