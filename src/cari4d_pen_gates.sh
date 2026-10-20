#!/bin/bash
# Wait for all push-out chunks and for the non-pen gate runs (GATES_DONE markers), then gate the PEN variants.
cd /home/tianyuan/projects/v2d_track1 || exit 1
until [ "$(cat logs/cari4d_pushout_cc_[0-9].log logs/cari4d_pushout_s3_[0-9].log 2>/dev/null | grep -c CHUNK_DONE)" -ge 16 ]; do sleep 30; done
echo "pushouts done $(date +%H:%M); cc_pen $(ls work/sub_cari4d_cc_pen | grep -c npz) cc_s3_pen $(ls work/sub_cari4d_cc_s3_pen | grep -c npz)"
grep -h Traceback logs/cari4d_pushout_cc_[0-9].log logs/cari4d_pushout_s3_[0-9].log | head -n 3
until grep -q GATES_DONE logs/cari4d_gate_pen_cc.log 2>/dev/null && grep -q GATES_DONE logs/cari4d_gate_pen_cc_s3.log 2>/dev/null; do sleep 20; done
src/run_gates_gpu12.sh work/sub_cari4d_cc_pen cc_pen 1 2 &
src/run_gates_gpu12.sh work/sub_cari4d_cc_s3_pen cc_s3_pen 2 1 &
wait
for v in cc cc_s3 cc_pen cc_s3_pen; do
  echo "== $v"; grep -h "RESULT ACC-O" logs/cari4d_gate_acc_$v.log; grep -h "^RESULT sub_cari4d" logs/cari4d_gate_body_$v.log; grep -h "RESULT PEN" logs/cari4d_gate_pen_$v.log
done
