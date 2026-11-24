"""Validation-only oracle pairing and conservative additive decoder controls."""
from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from common import save_json
P=ROOT/'evaluation_download';OUT=ROOT/'decoder_diagnosis_20261122';DATA=ROOT.parent.parent/'joint_event_dataset'

def one(eid):
    torch.set_num_threads(1)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from et_decode import E_edges,T_groups,geometry_extract,seeds
    from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,DEFAULT_CONFIG
    from coarse_3d_membership import propose,select
    with np.load(P/'cache/features'/('event_%05d.npz'%eid)) as s:f={k:s[k] for k in s.files}
    with np.load(P/'cache/indices'/('event_%05d.npz'%eid)) as s:t=s['t']
    with np.load(DATA/'events'/('event_%05d.npz'%eid)) as s:d=s['d'];truth=s['truth']
    geo=np.load(DATA/'wire_geometry.npy')[f['cells']]
    ep=np.load(P/'full/validation/epoch_035'/('prediction_%05d.npz'%eid))['E'];tq=np.load(P/'full/validation/epoch_025'/('prediction_%05d.npz'%eid))['T']
    base,_=axial_two_pass(geo,d,f['pairs'],f['p']);result={}
    for k in ['E_additive','T_additive','oracle_pair']:
        if k=='oracle_pair':
            a,b=f['pairs'];w=((truth[a]==truth[b])&(truth[a]>=0)).astype(np.float32)
            pairs=f['pairs'][:,w>0];labels=connected_components(coo_matrix((np.ones(pairs.shape[1]),pairs),shape=(len(truth),len(truth))),directed=False)[1]
            groups=[np.flatnonzero(labels==v).tolist() for v in np.unique(labels) if (labels==v).sum()>=5]
        elif k=='E_additive':groups,w=E_edges(f,ep)
        else:groups,w=T_groups(f,t,tq)
        proposed=geometry_extract(groups,w,f,geo,d,base)
        if k.endswith('additive'):
            tracks=list(base);used={i for c in base for i in c['indices']}
            for c in proposed:
                if not used.intersection(c['indices']):tracks.append(c);used.update(c['indices'])
        else:tracks=proposed
        joint,_=attach_stereo(tracks,geo,d,f['cells'],f['pairs'],f['p'],DEFAULT_CONFIG);counts,found=score_tracks(joint,truth,f['stereo'],DEFAULT_CONFIG)
        confirmed,_=select(joint,propose(joint,geo,d,f['cells'],f['pairs'],f['p'],refine=False),geo,f['cells'],'guarded_loose')
        result[k]=dict(counts=counts,found=found,seeds=seeds(confirmed,truth,f['stereo'],f['cells']))
    save_json(OUT/('event_%05d.json'%eid),dict(event_id=eid,methods=result));return eid

def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from STEP5_两级寻迹 import summarize
    from STEP2_训练双路线 import seed_summary
    OUT.mkdir(exist_ok=True);ids=json.loads((P/'full/contract.json').read_text(encoding='utf-8'))['validation_ids'][:16]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,eid) for eid in ids]):print('diagnosed',f.result(),flush=True)
    rows=[json.loads((OUT/('event_%05d.json'%eid)).read_text(encoding='utf-8')) for eid in ids]
    methods={k:dict(tracks=summarize([r['methods'][k]['counts'] for r in rows]),seeds=seed_summary([r['methods'][k]['seeds'] for r in rows])) for k in rows[0]['methods']}
    save_json(OUT/'summary.json',dict(events=16,scope='Development diagnostic subset, not confirmation; oracle_pair uses true edge labels and is not deployable',methods=methods));print(json.dumps(methods),flush=True)

if __name__=='__main__':main()
