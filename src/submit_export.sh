#!/usr/bin/env bash
# Submit an ALREADY-EXPORTED episode dir: PEN gate -> commit -> push -> pack -> upload -> wait.
# Usage: src/submit_export.sh work/sub_refine "message for commit and Kaggle"
# Refuses to submit if local PEN != 0.0 (PEN is self-referential; we hold rank 2 at 0.00000).
set -euo pipefail
cd /home/tianyuan/projects/v2d_track1
D="$1"; MSG="$2"
PEN=$(python3 src/score_local.py --episodes "$D" 2>&1 | tail -1 | python3 -c "import sys,json;print(json.loads(sys.stdin.read().split('MEAN',1)[1])['pen_cm'])")
echo "local PEN = $PEN"
if [ "$PEN" != "0.0" ]; then echo "PEN gate FAILED — not submitting"; exit 2; fi
git add -A
git -c user.email=tianny007@gmail.com -c user.name=tianny0078 commit -q -m "$MSG

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TMqTPD4NdJoeL2zDKuuhoB" || true
timeout 120 git push -q origin main
SHA=$(git rev-parse HEAD); echo "commit $SHA"
cd work/kit/v2d_submission_kit
../../venv-submit/bin/python eval_reconstruction.py --episodes "/home/tianyuan/projects/v2d_track1/$D" \
  --sample data/track_1_sample_submission.parquet \
  --commit "https://github.com/tianny0078/v2d-track1/commit/$SHA" \
  --out "/home/tianyuan/projects/v2d_track1/$D.parquet" 2>&1 | tail -1
../../venv-submit/bin/python submit/v2d_submit.py "/home/tianyuan/projects/v2d_track1/$D.parquet" \
  --track track_1 -m "$MSG" --wait 2>&1 | grep -E "COMPLETE|left today|ERROR|rejected"
