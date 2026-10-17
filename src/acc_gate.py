
import sys, json, numpy as np; sys.path.insert(0,'.'); sys.path.insert(0,'work/kit/v2d_submission_kit')
from v2dlb.mhr_submission import stretches
D=sys.argv[1]; sc=json.load(open('work/scored_frames_summary.json')); A=[]
for ep in range(30):
    z=np.load(f'{D}/episode_{ep:06d}.npz'); fr=np.concatenate([np.arange(a,b+1) for a,b in sc[str(ep)]['spans']]); t=z['object_translation'][fr]*100
    A.append(np.mean([np.linalg.norm(t[x][:-2]-2*t[x][1:-1]+t[x][2:],axis=-1).mean() for x in stretches(fr) if x.stop-x.start>=3]))
A=np.array(A); print(f"RESULT ACC-O proxy mean {A.mean():.3f} max {A.max():.3f} gate<0.3 -> {'PASS' if A.max()<0.3 else 'FAIL'}; worst {[(int(i),round(float(A[i]),2)) for i in np.argsort(-A)[:4]]}")
