#!/bin/bash
# ACC-only candidates (GPU0/GPU3 only): heavier body smoothing (sigma 5) + object translation sigma 24, push-out x2.
cd /home/tianyuan/projects/v2d_track1
CUDA_VISIBLE_DEVICES=0 bash src/run_export_parallel.sh work/sub_acc_c 6 --body work/body_cari4d --body-source fit --object place --obj-depth moge --obj-rot track --rot-tag _best --smooth 5 | tail -1 &
CUDA_VISIBLE_DEVICES=3 bash src/run_export_parallel.sh work/sub_acc_f 6 --body work/body_flow_px --body-source fit --object place --obj-depth moge --obj-rot track --rot-tag _best --smooth 5 | tail -1 &
wait
for X in sub_acc_c sub_acc_f; do
  python3 src/smooth_obj_t.py work/$X work/${X}_t24 --sigma 24
  echo "--- $X"; SIG=24 bash src/smooth_push.sh work/${X}_t24
done
echo "=== ALL DONE $(TZ=America/Los_Angeles date +%H:%M)"
