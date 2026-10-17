#!/bin/bash
# sub_track2 = sub_track1 + contact anchoring + best-of rotation tracks; waits for track1 and the reruns.
cd /home/tianyuan/projects/v2d_track1
while pgrep -f "^bash src/run_track1.sh" >/dev/null || pgrep -f "^python3 src/object_track.py" >/dev/null; do sleep 60; done
echo "=== start $(TZ=Asia/Shanghai date +%H:%M)"
python3 src/select_rot.py 2>&1 | grep RESULT
python3 src/export_episodes.py --out work/sub_track2 --body work/body_flow_px --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --smooth 3 > logs/export_track2.log 2>&1
echo "=== exported $(ls work/sub_track2 | wc -l) files $(TZ=Asia/Shanghai date +%H:%M)"
grep -E "contact anchor" logs/export_track2.log | cut -c1-80
python3 src/acc_gate.py work/sub_track2 2>&1 | grep RESULT
python3 src/obj_hand_dist.py work/sub_track1 work/sub_track2 2>&1 | grep RESULT
echo "=== DONE $(TZ=Asia/Shanghai date +%H:%M)"
