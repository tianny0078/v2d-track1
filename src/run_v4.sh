#!/bin/bash
# v4 = depth-ICP body (fit_depth on body_flow_px) + track2 objects (MoGe depth, anchoring, best rotations, centred mesh, centroid push).
cd /home/tianyuan/projects/v2d_track1
while [ "$(ls state/moge_pts/ 2>/dev/null | wc -l)" -lt 30 ]; do sleep 60; done
echo "=== moge_pts complete $(TZ=America/Los_Angeles date +%H:%M)"
for g in 1 2 3; do
  EPS=$(python3 -c "print(' '.join(str(e) for e in range(30) if e % 3 == $((g-1))))")
  python3 src/fit_depth.py --ep $EPS --device cuda:$g > logs/fit_depth_$g.log 2>&1 &
done; wait
echo "=== fit_depth done $(ls work/body_depth | wc -l)/30 $(TZ=America/Los_Angeles date +%H:%M)"
grep -h RESULT logs/fit_depth_?.log | cut -c1-220
bash src/run_export_parallel.sh work/sub_v4 6 --body work/body_depth --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --smooth 3 2>&1 | tail -n 1
python3 src/body_proxy.py work/sub_track2 work/sub_v4 2>&1 | grep RESULT
python3 src/acc_gate.py work/sub_v4 2>&1 | grep RESULT
python3 src/obj_hand_dist.py work/sub_track2 work/sub_v4 2>&1 | grep RESULT | tail -n 2
python3 src/pen_gate.py work/sub_v4 cuda:1 2>&1 | grep RESULT
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
