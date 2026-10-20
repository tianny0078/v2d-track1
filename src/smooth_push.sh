#!/bin/bash
# work/X_t6 -> pushout -> X_t6_pen -> sigma6 -> X_t6_pen_t6 -> pushout -> X_t6_pen_t6_pen, then gates.
# Usage: src/smooth_push.sh work/sub_hy3_t6
cd /home/tianyuan/projects/v2d_track1
push() { for w in 0 1 2 3 4 5; do EPS=$(python3 -c "print(' '.join(str(e) for e in range(30) if e % 6 == $w))")
  CUDA_VISIBLE_DEVICES=$(( w % 2 + 1 )) python3 src/cari4d_pushout.py $1 $2 --eps $EPS --device cuda:0 > logs/push_$(basename $2)_w$w.log 2>&1 & done; wait; }
A=$1
push $A ${A}_pen
python3 src/smooth_obj_t.py ${A}_pen ${A}_pen_t6 --sigma 6
push ${A}_pen_t6 ${A}_pen_t6_pen
F=${A}_pen_t6_pen
python3 src/acc_gate.py $F 2>&1 | grep RESULT
python3 src/pen_gate.py $F cuda:1 2>&1 | grep RESULT
python3 src/body_proxy.py work/sub_track2 $F 2>&1 | grep RESULT
echo "=== DONE $(TZ=America/Los_Angeles date +%H:%M)"
