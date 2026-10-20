#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
bash src/run_export_parallel.sh work/sub_v5 6 --body work/body_depth --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --obj-size moge --smooth 5 2>&1 | tail -n 1
python3 src/acc_gate.py work/sub_v5 2>&1 | grep RESULT
python3 src/obj_image_sanity.py work/sub_v5 2>&1 | grep ROW | awk '{print $2, $3, "ratio", $6, "off", $9}' | tr '\n' ';'; echo
python3 src/obj_hand_dist.py work/sub_v4b work/sub_v5 2>&1 | grep RESULT | tail -n 1
python3 src/pen_gate.py work/sub_v5 cuda:1 2>&1 | grep RESULT
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
