#!/bin/bash
# Parallel push-out: (re)run src/cari4d_pushout.py over episodes not yet complete in <dst>
# (complete = npz AND obj present), split into N CPU-heavy chunks (MHR forward on GPU $GPU).
# Usage: src/cari4d_pushout_par.sh <src> <dst> <N> <gpu> <tag>
cd /home/tianyuan/projects/v2d_track1 || exit 1
SRC=$1; DST=$2; N=$3; GPU=$4; TAG=$5
mkdir -p $DST
TODO=()
for e in $(seq 0 29); do
  f=$(printf "%s/episode_%06d" $DST $e)
  if [ -f $f.npz ] && [ -f ${f}_object.obj ]; then continue; fi
  rm -f $f.npz ${f}_object.obj
  [ -f $(printf "%s/episode_%06d.npz" $SRC $e) ] && TODO+=($e)
done
echo "$TAG todo: ${TODO[*]}"
for ((k=0; k<N; k++)); do
  CH=""
  for ((i=k; i<${#TODO[@]}; i+=N)); do CH="$CH ${TODO[$i]}"; done
  [ -z "$CH" ] && continue
  setsid nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU python3 src/cari4d_pushout.py $SRC $DST --eps $CH --device cuda > logs/cari4d_pushout_${TAG}_$k.log 2>&1; echo CHUNK_DONE >> logs/cari4d_pushout_${TAG}_$k.log" < /dev/null > /dev/null 2>&1 &
done
echo "$TAG launched"
