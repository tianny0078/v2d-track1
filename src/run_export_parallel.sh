#!/bin/bash
# Parallel export: run export_episodes.py over episode subsets in N worker processes.
# Usage: src/run_export_parallel.sh <out_dir> <n_workers> <extra args...>
cd /home/tianyuan/projects/v2d_track1
OUT="$1"; N="$2"; shift 2
mkdir -p "$OUT" logs
for ((w=0; w<N; w++)); do
  EPS=$(python3 -c "print(' '.join(str(e) for e in range(30) if e % $N == $w))")
  python3 src/export_episodes.py --out "$OUT" --eps $EPS "$@" > "logs/export_$(basename $OUT)_w$w.log" 2>&1 &
done
wait
echo "=== all workers done $(TZ=Asia/Shanghai date +%H:%M): $(ls $OUT | wc -l) files"
