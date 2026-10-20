#!/bin/bash
# Four gates on an export dir using ONLY physical GPU1/GPU2 (body_proxy/obj_hand_dist hardcode cuda:3/cuda:2,
# so they are exec'd with the device string rewritten to cuda:0 under a single visible GPU).
# Usage: src/run_gates_gpu12.sh <export_dir> <tag> <gpu for body/pen> <gpu for obj>
cd /home/tianyuan/projects/v2d_track1 || exit 1
D=$1; TAG=$2; GB=$3; GO=$4
run_patched() { # script, gpu, from, args...
  local s=$1 g=$2 from=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g python3 -c "import sys; sys.argv=['$s']+sys.argv[1:]; src=open('$s').read().replace('\"$from\"','\"cuda:0\"'); exec(compile(src,'$s','exec'), {'__file__':'$s','__name__':'__main__'})" "$@"
}
python3 src/acc_gate.py $D > logs/cari4d_gate_acc_$TAG.log 2>&1
run_patched src/body_proxy.py $GB cuda:3 work/sub_track2 $D > logs/cari4d_gate_body_$TAG.log 2>&1
run_patched src/obj_hand_dist.py $GO cuda:2 work/sub_v6 $D > logs/cari4d_gate_obj_$TAG.log 2>&1
CUDA_VISIBLE_DEVICES=$GB python3 src/pen_gate.py $D cuda:0 > logs/cari4d_gate_pen_$TAG.log 2>&1
echo GATES_DONE >> logs/cari4d_gate_pen_$TAG.log
