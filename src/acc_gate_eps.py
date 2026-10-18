import sys, json, numpy as np; sys.path.insert(0,'.'); sys.path.insert(0,'work/kit/v2d_submission_kit')
from v2dlb.mhr_submission import stretches
D=sys.argv[1]; eps=[int(e) for e in sys.argv[2:]]; sc=json.load(open('work/scored_frames_summary.json'))
for ep in eps:
    z=np.load(f'{D}/episode_{ep:06d}.npz'); fr=np.concatenate([np.arange(a,b+1) for a,b in sc[str(ep)]['spans']]); t=z['object_translation'][fr]*100
    A=np.mean([np.linalg.norm(t[x][:-2]-2*t[x][1:-1]+t[x][2:],axis=-1).mean() for x in stretches(fr) if x.stop-x.start>=3])
    print(f"RESULT ep{ep} ACC-O proxy {A:.3f}")
