#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
for ep in 9 18 24 12; do
python3 src/export_episodes.py --out work/sub_size_test --eps $ep --body work/body_depth --body-source fit --object place --obj-depth moge --contact-anchor --obj-rot track --rot-tag _best --obj-size moge --smooth 5 2>&1 | grep -E "object size|push-out|Traceback|Error" | cut -c1-160 &
done; wait
python3 - <<'PY'
import numpy as np, json, trimesh
sc=json.load(open('work/scored_frames_summary.json')); K=json.load(open('work/camera/intrinsics.json'))
meta={json.loads(l)['episode_index']:json.loads(l) for l in open('data/hf/track_1/meta/episodes_metadata.jsonl')}
for ep in (9,18,24,12):
  for D in ('work/sub_v4b','work/sub_size_test'):
    k=K[meta[ep]['camera']]; fr=np.concatenate([np.arange(a,b+1) for a,b in sc[str(ep)]['spans']])
    z=np.load(f'work/masks/ep{ep:06d}.npz'); W=int(z['shape'][1]); mpos={f:i for i,f in enumerate(z['frame_ids'])}
    e=np.load(f'{D}/episode_{ep:06d}.npz'); m=trimesh.load(f'{D}/episode_{ep:06d}_object.obj',force='mesh',process=False); v=np.asarray(m.vertices)*float(e['object_scale']); v=v[::8]
    r=[];o=[]
    for f in fr[::10]:
        if f not in mpos: continue
        mm=np.unpackbits(z['packed'][mpos[f]],axis=-1)[0,:,:W].astype(bool); ys,xs=np.where(mm)
        if xs.size<30: continue
        P=v@e['object_rotation'][f].T+e['object_translation'][f]; u=k['fx']*P[:,0]/P[:,2]+k['cx']; vv=k['fy']*P[:,1]/P[:,2]+k['cy']
        r.append(max(np.ptp(u),np.ptp(vv))/max(np.ptp(xs),np.ptp(ys))); o.append(np.hypot(u.mean()-xs.mean(),vv.mean()-ys.mean()))
    print(f"RESULT ep{ep} {D.split('/')[-1]:14s} proj/mask extent {np.median(r):.2f} centroid off {np.median(o):.0f} px")
PY
echo "=== DONE"
