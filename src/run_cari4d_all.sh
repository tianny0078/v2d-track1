#!/bin/bash
# Run the official CARI4D stages on all 30 episodes, two GPU workers (episodes interleaved by index).
# Usage: src/run_cari4d_all.sh "2 3" "01,02,03,04,05,06,07"   (gpus, stages)
cd /home/tianyuan/projects/v2d_track1 || exit 1
GPUS=(${1:-2 3}); STAGES=${2:-01,02,03,04,05,06,07}
N=${#GPUS[@]}
for ((w=0; w<N; w++)); do
  EPS=""
  for ((e=0; e<30; e++)); do if (( e % N == w )); then EPS="$EPS $e"; fi; done
  setsid nohup python3 src/cari4d_run.py --eps $EPS --gpu ${GPUS[$w]} --stages $STAGES \
    > logs/cari4d_worker_gpu${GPUS[$w]}.log 2>&1 < /dev/null & disown
  echo "worker gpu${GPUS[$w]}: eps$EPS"
done
