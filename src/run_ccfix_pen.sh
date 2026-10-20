#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
for w in 0 1 2 3 4 5; do
  EPS=$(python3 -c "print(' '.join(str(e) for e in range(30) if e % 6 == $w))")
  G=$(( w % 2 + 1 ))
  CUDA_VISIBLE_DEVICES=$G python3 src/cari4d_pushout.py work/sub_cari4d_cc_fix work/sub_cari4d_cc_fix_pen --eps $EPS --device cuda:0 > logs/ccfix_pen_w$w.log 2>&1 &
done; wait
echo "=== pushed $(ls work/sub_cari4d_cc_fix_pen | wc -l) files"
python3 src/acc_gate.py work/sub_cari4d_cc_fix_pen 2>&1 | grep RESULT
python3 src/pen_gate.py work/sub_cari4d_cc_fix_pen cuda:1 2>&1 | grep RESULT
python3 src/body_proxy.py work/sub_track2 work/sub_cari4d_cc_fix_pen 2>&1 | grep RESULT
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
