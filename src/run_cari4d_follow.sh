#!/bin/bash
# Follower: for each episode in the list, wait until stage 05 is done, then run stages 06+07 on $GPU.
# Usage: src/run_cari4d_follow.sh <gpu> <ep list...>
cd /home/tianyuan/projects/v2d_track1 || exit 1
GPU=$1; shift
for e in "$@"; do
  seq=$(printf "ep%06d" $e)
  until [ -f state/cari4d/05_object_pose/$seq.done ]; do sleep 60; done
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 06,07
done
echo "follower gpu$GPU finished"
