#!/bin/bash
# TRUE official reproduction pass (with CoCoNet): per episode, regenerate the pruned 01/02/04
# intermediates, run official step 06 (CoCoNet) and step 07 on top of it.
# Usage: src/run_cari4d_cc.sh <gpu> <ep...>
cd /home/tianyuan/projects/v2d_track1 || exit 1
GPU=$1; shift
OFF=work/v2d_official; SRC=$OFF/reconstruction/modules/v2d_cari4d/lib/cari4d
for e in "$@"; do
  seq=$(printf "ep%06d" $e); run=$OFF/runs/$seq
  if [ -f state/cari4d/07_refinement/$seq.done ] && [ -f $run/inference/coconet.pth ] && [ $run/inference/refined.pth -nt $run/inference/coconet.pth ]; then echo "$seq: cc done (skip)"; continue; fi
  # keep the no-CoCoNet refinement result
  [ -f $run/inference/refined.pth ] && [ ! -f $run/inference/coconet.pth ] && mv -n $run/inference/refined.pth $run/inference/refined_noCoCoNet.pth
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 01 --force || { echo "$seq: 01 FAILED"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$OFF/shims:$SRC:$OFF/reconstruction/modules/v2d_sam3d_body/lib:$OFF/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose HDF5_USE_FILE_LOCKING=FALSE \
    python3 src/cari4d_regen_export.py --video $OFF/inputs/$seq/$seq.0.color.mp4 --mask-h5 $OFF/inputs/$seq/${seq}_masks_k0.h5 --export-seq $run/export/$seq || { echo "$seq: regen FAILED"; continue; }
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 04 --force || { echo "$seq: 04 FAILED"; continue; }
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 06,07 --force
done
echo "cc worker gpu$GPU finished"
