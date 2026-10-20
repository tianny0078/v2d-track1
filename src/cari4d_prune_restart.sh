#!/bin/bash
# Restart the CARI4D pruner loop (pattern lives in this file so the caller's shell never matches it).
cd /home/tianyuan/projects/v2d_track1 || exit 1
for p in $(pgrep -f "cari4d_prune.py --loop"); do kill "$p" 2>/dev/null; done
sleep 1
setsid nohup python3 src/cari4d_prune.py --loop --gpu 2 --sub work/sub_cari4d_cc --cc > logs/cari4d_prune.log 2>&1 < /dev/null & disown
echo "pruner restarted: $(pgrep -fc 'cari4d_prune.py --loop') process(es)"
