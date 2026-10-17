#!/usr/bin/env bash
# Gates + sanity for an export dir, written to a log. Run DETACHED (nohup) -- the harness kills
# multi-minute foreground/background shells ("low memory", 6 times so far; the box has ~900 GB free).
# Usage: nohup src/check_export.sh work/sub_place5 > logs/check_sub_place5.log 2>&1 &
set -uo pipefail
cd /home/tianyuan/projects/v2d_track1
D="$1"
echo "=== push-out summary ==="; grep "push-out" "logs/export_$(basename "$D" | sed 's/^sub_//').log" 2>/dev/null | tail -30
echo "=== GATE 1: ACC-O proxy (object translation second difference, cm/frame^2; reference 0.074) ==="
python3 - "$D" <<'PY' 2>&1 | grep RESULT
import sys, json, numpy as np; sys.path.insert(0,'.'); sys.path.insert(0,'work/kit/v2d_submission_kit')
from v2dlb.mhr_submission import stretches
D=sys.argv[1]; sc=json.load(open('work/scored_frames_summary.json')); A=[]
for ep in range(30):
    z=np.load(f'{D}/episode_{ep:06d}.npz'); fr=np.concatenate([np.arange(a,b+1) for a,b in sc[str(ep)]['spans']]); t=z['object_translation'][fr]*100
    A.append(np.mean([np.linalg.norm(t[x][:-2]-2*t[x][1:-1]+t[x][2:],axis=-1).mean() for x in stretches(fr) if x.stop-x.start>=3]))
A=np.array(A); print(f"RESULT ACC-O proxy mean {A.mean():.3f} max {A.max():.3f} gate<0.3 -> {'PASS' if A.max()<0.3 else 'FAIL'}; worst {[(int(i),round(float(A[i]),2)) for i in np.argsort(-A)[:4]]}")
PY
echo "=== GATE 2: local PEN (must be 0.0) ==="
python3 src/score_local.py --episodes "$D" 2>&1 | tail -1 | sed 's/^/RESULT PEN /'
echo "=== SANITY (hand-obj cm | reproj off px | scale | size m) ==="
python3 src/place_sanity.py "$D" 2>&1 | grep -vE "Warning|warn|^$"
echo "=== DONE ==="
