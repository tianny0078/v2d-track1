#!/bin/bash
# Lock-aware CoCoNet-pass worker: pulls episodes from a shared queue, claims each by an atomic
# mkdir under state/cari4d/cc_claim/, skips episodes whose cc pass is complete. Several workers
# (also several per GPU) can share the queue safely.
# Usage: src/run_cari4d_cc2.sh <gpu> <ep...>   (queue order)
cd /home/tianyuan/projects/v2d_track1 || exit 1
GPU=$1; shift
OFF=work/v2d_official; SRC=$OFF/reconstruction/modules/v2d_cari4d/lib/cari4d
CLAIM=state/cari4d/cc_claim; mkdir -p $CLAIM
for e in "$@"; do
  seq=$(printf "ep%06d" $e); run=$OFF/runs/$seq
  if [ -f state/cari4d/07_refinement/$seq.done ] && [ -f $run/inference/coconet.pth ] && [ $run/inference/refined.pth -nt $run/inference/coconet.pth ]; then continue; fi
  mkdir $CLAIM/$seq 2>/dev/null || continue
  echo "gpu$GPU claims $seq $(date +%H:%M:%S)"
  [ -f $run/inference/refined.pth ] && [ ! -f $run/inference/coconet.pth ] && mv -n $run/inference/refined.pth $run/inference/refined_noCoCoNet.pth
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 01 --force || { echo "$seq: 01 FAILED"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$OFF/shims:$SRC:$OFF/reconstruction/modules/v2d_sam3d_body/lib:$OFF/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose HDF5_USE_FILE_LOCKING=FALSE \
    python3 src/cari4d_regen_export.py --video $OFF/inputs/$seq/$seq.0.color.mp4 --mask-h5 $OFF/inputs/$seq/${seq}_masks_k0.h5 --export-seq $run/export/$seq || { echo "$seq: regen FAILED"; continue; }
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 04 --force || { echo "$seq: 04 FAILED"; continue; }
  python3 src/cari4d_run.py --eps $e --gpu $GPU --stages 06,07 --force
done
echo "cc2 worker gpu$GPU finished"
