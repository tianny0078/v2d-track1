#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
python3 src/export_episodes.py --out work/sub_track3 --eps 25 9 27 8 --body work/body_flow_px --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --push closest --smooth 3 2>&1 | grep -E "push-out|Traceback|Error" | cut -c1-200
python3 src/obj_hand_dist.py work/sub_track2 work/sub_track3 25 9 27 8 2>&1 | grep RESULT
echo "=== DONE"
