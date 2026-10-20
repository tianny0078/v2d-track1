#!/usr/bin/env bash
# One-shot: export the SAM 3D Body reconstruction, self-check PEN, commit, pack, submit.
# Usage: src/submit_sam3d.sh <out_tag> "<submission message>" [extra export args...]
#   e.g. src/submit_sam3d.sh sub_sam3d "SAM 3D Body + sigma=2 smoothing, object unchanged" --object none --smooth 2.0
set -euo pipefail
cd /home/tianyuan/projects/v2d_track1
TAG="$1"; MSG="$2"; shift 2
python3 src/export_episodes.py --out "work/$TAG" --body-source sam3d "$@"
echo "--- local PEN self-check (must be 0.0) ---"
python3 src/score_local.py --episodes "work/$TAG" 2>&1 | tail -1
git add -A
git -c user.email=tianny007@gmail.com -c user.name=tianny0078 commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TMqTPD4NdJoeL2zDKuuhoB" || true
timeout 120 git push -q origin main
SHA=$(git rev-parse HEAD); echo "commit $SHA"
cd work/kit/v2d_submission_kit
../../venv-submit/bin/python eval_reconstruction.py --episodes "/home/tianyuan/projects/v2d_track1/work/$TAG" \
  --sample data/track_1_sample_submission.parquet \
  --commit "https://github.com/tianny0078/v2d-track1/commit/$SHA" \
  --out "/home/tianyuan/projects/v2d_track1/work/$TAG.parquet" 2>&1 | tail -1
../../venv-submit/bin/python submit/v2d_submit.py "/home/tianyuan/projects/v2d_track1/work/$TAG.parquet" \
  --track track_1 -m "$MSG" --wait 2>&1 | grep -E "COMPLETE|left today|ERROR|rejected"
