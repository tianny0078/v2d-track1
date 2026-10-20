#!/bin/bash
# official CARI4D body (work/body_cari4d) + our v6 object pipeline; hy0 = body as is, hy3 = extra sigma3 body smoothing
cd /home/tianyuan/projects/v2d_track1
for S in 0 3; do
  CUDA_VISIBLE_DEVICES=$((S==0?1:2)) bash src/run_export_parallel.sh work/sub_hy$S 6 --body work/body_cari4d --body-source fit --object place --obj-depth moge --obj-rot track --rot-tag _best --smooth $S 2>&1 | tail -n 1 &
done; wait
for S in 0 3; do
  echo "--- sub_hy$S"
  python3 src/acc_gate.py work/sub_hy$S 2>&1 | grep RESULT
  python3 src/pen_gate.py work/sub_hy$S cuda:1 2>&1 | grep RESULT
  python3 src/body_proxy.py work/sub_track2 work/sub_hy$S 2>&1 | grep RESULT
done
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
