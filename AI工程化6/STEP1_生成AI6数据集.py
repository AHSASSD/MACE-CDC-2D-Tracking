"""Rebuild all 10000 events from original CSV, compact storage, graphs built lazily."""
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DATA_ROOT);args=p.parse_args()
    out=args.output;out.mkdir(exist_ok=False);(out/'events').mkdir()
    if shutil.disk_usage(out).free<2*1024**3: raise OSError('Need 2 GiB free space')
    geometry=pd.read_excel(MAP_FILE).sort_values('CellID')
    assert np.array_equal(geometry.CellID,np.arange(len(geometry)))
    provenance=json.loads((SOURCE/'whole data/ENVIRONMENT.json').read_text(encoding='utf-8'))
    assert sha(MAP_FILE)==next(v for k,v in provenance['hashes'].items() if k.endswith('cdc_cell_map.xlsx'))
    geo=geometry[['x','y','u','v','w','l']].to_numpy(float)
    assert np.allclose(np.linalg.norm(geo[:,2:5],axis=1),1,atol=1e-6)
    np.save(out/'wire_geometry.npy',geo)
    split=json.loads((SOURCE/'whole_event_retraining/event_split.json').read_text(encoding='utf-8'))
    ids={k:split[v] for k,v in [('train','train_evt_ids'),('validation','val_evt_ids'),('test','test_evt_ids')]}
    assert not(set(ids['train'])&set(ids['validation']) or set(ids['train'])&set(ids['test']) or set(ids['validation'])&set(ids['test']))
    expected=set(sum(ids.values(),[]));seen=set();summary=[];sources={}
    for path in sorted((SOURCE/'whole data').glob('CDCSimHit_*.csv')):
        sources[path.name]=sha(path)
        f=pd.read_csv(path,usecols=['EvtID','TrkID','HitID','CellID','d','WireType','Good','CreatProc'])
        f=f.loc[f.Good.eq(True)&f.CreatProc.eq('ExtendedDecayWithSpin')]
        for eid,g in f.groupby('EvtID',sort=True):
            eid=int(eid)
            if eid in seen or eid not in expected: raise ValueError('Duplicate/unexpected event')
            cell=g.CellID.to_numpy(np.int64);d=g.d.to_numpy(float);hit=g.HitID.to_numpy(np.int64)
            assert np.isfinite(d).all() and len(np.unique(hit))==len(hit)
            assert np.array_equal(g.WireType.eq('Stereo'),np.hypot(geo[cell,2],geo[cell,3])>1e-9)
            np.savez_compressed(out/'events'/f'event_{eid:05d}.npz',cell_id=cell,d=d,hit_id=hit,truth=g.TrkID.to_numpy(np.int64))
            summary.append({'event_id':eid,'hits':len(g),'stereo_hits':int(g.WireType.eq('Stereo').sum())});seen.add(eid)
        print(path.name,'events',len(seen),flush=True)
    assert seen==expected and len(seen)==10000
    save_json(out/'event_split.json',split)
    save_json(out/'event_inventory.json',summary)
    save_json(out/'dataset_metadata_joint_no_time.json',{'status':'complete','profile':'AI6-joint-no-time-v1','event_ids':ids,
        'node_features':NODE_FEATURES,'edge_features':EDGE_FEATURES,'sources':sources,'geometry_sha256':sha(MAP_FILE),
        'time_used':False,'added_d_noise':0,'truth_is_input':False,'storage':'compressed per-event measurements; exact graph rebuilt lazily',
        'graph_rule':'minimum transverse wire separation at common finite z <=120mm; preserve repeated-cell hits',
        'angle':'atan2((-y*u+x*v)/r,w), radians; axial zero; signed stereo handedness',
        'reference':'wire midpoint (x,y,z=0), unit (u,v,w), physical full length l',
        'source_split_sha256':sha(SOURCE/'whole_event_retraining/event_split.json')})
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
