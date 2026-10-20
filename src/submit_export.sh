#!/usr/bin/env bash
# Submit an ALREADY-EXPORTED episode dir: PEN gate -> commit -> push -> pack -> upload -> wait.
# Usage: src/submit_export.sh work/sub_refine "message for commit and Kaggle"
# Refuses to submit if local PEN != 0.0 (PEN is self-referential; we hold rank 2 at 0.00000).
set -euo pipefail
cd /home/tianyuan/projects/v2d_track1
D="$1"; MSG="$2"
if [ "${SKIP_PEN:-0}" != "1" ]; then
  # fast gate: the host's penetration_depth on scored frames only (src/pen_gate.py), same semantics as score_local's PEN
  python3 src/pen_gate.py "$D" 2>&1 | grep RESULT | tee /tmp/pen_gate.txt
  grep -q PASS /tmp/pen_gate.txt || { echo "PEN gate FAILED — not submitting"; exit 2; }
fi
git add -A
git -c user.email=tianny007@gmail.com -c user.name=tianny0078 commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
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
