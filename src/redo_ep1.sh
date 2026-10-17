#!/bin/bash
# ep1 downstream after the person mask was re-seeded onto the subject (15:50).
cd /home/tianyuan/projects/v2d_track1
set -o pipefail
echo "=== kp2d $(date +%H:%M)"; python3 src/keypoints2d.py --eps 1 --device cuda:2 2>&1 | tail -n 2
echo "=== sam3d $(date +%H:%M)"; CUDA_VISIBLE_DEVICES=2 python3 src/sam3d_body.py --eps 1 2>&1 | tail -n 2
echo "=== tokens $(date +%H:%M)"; CUDA_VISIBLE_DEVICES=2 python3 src/flowhmr_tokens.py --eps 1 2>&1 | tail -n 2
echo "=== objdepth $(date +%H:%M)"; python3 src/object_depth.py --eps 1 --device cuda:2 2>&1 | grep RESULT
echo "=== DONE $(date +%H:%M)"
