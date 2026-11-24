"""Same real-event predictions: compare prior serial logic with spawned CPU workers."""
from pathlib import Path
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'复杂事件归因_20261107'))
import experiment_common as baseline
sys.path.insert(0,str(ROOT))
from membership_cpu import reconstruct,seed_counts
from morphology import track_shape
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
from coarse_3d_membership import propose,select


def main():
    baseline.initialize_worker();folder=ROOT/'parallel_verification';folder.mkdir(exist_ok=True)
    eid=1506;data=ROOT.parent.parent/'joint_event_dataset'
    prediction=ROOT.parent.parent/'visual_diagnosis_20261121/joint_events_1500_2000/prediction_cache/event_01506.npz'
    cutoff=json.loads((ROOT/'difficulty_audit/selection.json').read_text(encoding='utf-8'))['hit_cutoff']
    with np.load(prediction) as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(data/'events'/('event_%05d.npz'%eid)) as raw:cells,d,truth=raw['cell_id'],raw['d'],raw['truth']
    geo=np.load(data/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    axial,_=axial_two_pass(geo,d,pairs,p);joint,_=attach_stereo(axial,geo,d,cells,pairs,p,DEFAULT_CONFIG)
    coarse,_=select(joint,propose(joint,geo,d,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    counts,_=score_tracks(joint,truth,stereo,DEFAULT_CONFIG)
    shapes=[track_shape(cells[(truth==t)&~stereo],geo[(truth==t)&~stereo,:2]) for t in np.unique(truth[~stereo]) if t>=0 and ((truth==t)&~stereo).sum()>=5]
    strata=['all']+(['high_hits'] if len(cells)>cutoff else [])+(['blob'] if any(t['blob'] for t in shapes) else [])+(['loop'] if any(t['loop'] for t in shapes) else [])
    if len(strata)==1:strata.append('ordinary')
    expected=dict(event_id=eid,counts=counts,seed=seed_counts(coarse,truth,stereo,cells),strata=strata)
    tasks=[(eid,str(data),str(prediction),str(folder/('worker_%d.json'%i)),cutoff) for i in range(2)]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(reconstruct,tasks))
    assert rows==[expected,expected]
    result=dict(status='passed',event_id=eid,independent_spawned_workers=2,serial_parallel_exact_match=True,
                counts=counts,seed=expected['seed'],strata=strata)
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
