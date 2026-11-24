"""Pure CPU member reconstruction, independent of model training and random sampling."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from common import save_json
from morphology import track_shape
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
from coarse_3d_membership import propose,select


@lru_cache(maxsize=1)
def geometry(path):
    return np.load(path)


def seed_counts(tracks,truth,stereo,cells):
    eligible={int(t):int(((truth==t)&~stereo).sum()) for t in np.unique(truth) if t>=0 and ((truth==t)&~stereo).sum()>=5}
    targets={t for t in eligible if len(np.unique(cells[(truth==t)&stereo]))>=2};found=set();counts=Counter(truth_tracks=len(targets))
    for track in tracks:
        if not track['coarse_3d']['seed_ready']:continue
        counts['output_candidates']+=1;aa=np.array([i for i in track['indices'] if not stereo[i]],dtype=int);ss=np.asarray(track['confirmed_stereo_indices'],dtype=int)
        ids,ns=np.unique(truth[aa],return_counts=True);tid=int(ids[ns.argmax()]);good=ss[truth[ss]==tid]
        valid=tid in targets and ns.max()/len(aa)>=.8 and ns.max()/eligible[tid]>=.5 and len(np.unique(cells[good]))>=2 and len(good)/max(len(ss),1)>=.8
        if not valid:counts['fake_candidates']+=1
        elif tid in found:counts['clone_candidates']+=1
        else:found.add(tid)
    counts['matched_tracks']=len(found);return dict(counts)


def reconstruct(task):
    eid,data_path,prediction_path,output_path,cutoff=task
    data=Path(data_path)
    with np.load(prediction_path) as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(data/'events'/('event_%05d.npz'%eid)) as raw:cells,d,truth=raw['cell_id'],raw['d'],raw['truth']
    geo=geometry(str(data/'wire_geometry.npy'))[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    axial,_=axial_two_pass(geo,d,pairs,p);joint,_=attach_stereo(axial,geo,d,cells,pairs,p,DEFAULT_CONFIG)
    coarse,_=select(joint,propose(joint,geo,d,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    counts,_=score_tracks(joint,truth,stereo,DEFAULT_CONFIG)
    shapes=[track_shape(cells[(truth==t)&~stereo],geo[(truth==t)&~stereo,:2]) for t in np.unique(truth[~stereo]) if t>=0 and ((truth==t)&~stereo).sum()>=5]
    strata=['all']+(['high_hits'] if len(cells)>cutoff else [])+(['blob'] if any(t['blob'] for t in shapes) else [])+(['loop'] if any(t['loop'] for t in shapes) else [])
    if len(strata)==1:strata.append('ordinary')
    row=dict(event_id=eid,counts=counts,seed=seed_counts(coarse,truth,stereo,cells),strata=strata)
    save_json(Path(output_path),row)
    return row
