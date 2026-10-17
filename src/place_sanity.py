import sys, json, glob, numpy as np, trimesh; sys.path.insert(0,'/home/tianyuan/projects/v2d_track1')
from pathlib import Path
from src.mhr_local import MHR, L_WRIST, R_WRIST
mhr=MHR('cuda:2'); K=json.load(open('work/camera/intrinsics.json')); sc=json.load(open('work/scored_frames_summary.json'))
meta={json.loads(l)['episode_index']:json.loads(l) for l in open('data/hf/track_1/meta/episodes_metadata.jsonl')}
print(f"{'ep':>3} {'object':<18}{'hand-obj cm':>12}{'reproj off px':>14}{'scale':>7}{'size m':>8}", flush=True)
for p in sorted(glob.glob(f'{sys.argv[1]}/episode_*.npz')):
    ep=int(Path(p).stem.split('_')[1]); k=K[meta[ep]['camera']]; z=np.load(p)
    fr=np.concatenate([np.arange(a,b+1) for a,b in sc[str(ep)]['spans']])
    pr=np.concatenate([z['pose'][fr],np.broadcast_to(z['scales'],(len(fr),68))],1); _,j=mhr.forward(pr,z['shape'])
    hands=j[:,[L_WRIST,R_WRIST]]
    mesh=trimesh.load(f"work/meshes/{meta[ep]['object']}.glb",force='mesh',process=False)
    s=float(z['object_scale']); R=z['object_rotation'][fr]; t=z['object_translation'][fr]
    cen=t+np.einsum('tij,j->ti',R,s*np.asarray(mesh.vertices).mean(0))
    dh=np.linalg.norm(hands-cen[:,None],axis=-1).min(1)*100
    m=np.load(f'work/masks/ep{ep:06d}.npz'); W=int(m['shape'][1]); mi=np.searchsorted(m['frame_ids'],fr); packed=m['packed']
    offs=[]
    for i,c in zip(mi[::4],cen[::4]):
        om=np.unpackbits(packed[i],axis=-1)[0,:,:W]; ys,xs=np.where(om)
        if len(xs)<30: continue
        offs.append(np.hypot(k['fx']*c[0]/c[2]+k['cx']-xs.mean(), k['fy']*c[1]/c[2]+k['cy']-ys.mean()))
    print(f"{ep:>3} {meta[ep]['object']:<18}{np.median(dh):>12.0f}{np.median(offs):>14.0f}{s:>7.2f}{s*mesh.extents.max():>8.2f}", flush=True)
