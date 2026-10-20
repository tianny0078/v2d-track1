#!/bin/bash
# Stop the CARI4D pruner loop and run one final prune pass.
cd /home/tianyuan/projects/v2d_track1 || exit 1
for p in $(pgrep -f "cari4d_prune.py --loop"); do kill "$p" 2>/dev/null; done
sleep 1
CUDA_VISIBLE_DEVICES=3 python3 src/cari4d_prune.py 2>&1 | grep -v arn | tail -n 1
