"""Small declared threshold grid, selected only on split validation events."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from common import MODEL_ROOT,DATA_ROOT,save_json,sha
from two_stage_core import run_two_pass
from STEP5_两级寻迹 import attach_stereo,DEFAULT_CONFIG,score_tracks,summarize
from STEP8_寻迹改进评估 import select_profile

PROFILES={'threshold_065_055':(.65,.55),'threshold_075_0575':(.75,.575),'threshold_075_060':(.75,.60),'threshold_070_055':(.70,.55)}


def threshold_tracks(geo,drift,pairs,p,thresholds):
    aa=np.flatnonzero(np.linalg.norm(geo[:,2:4],axis=1)<=1e-9);mapping=np.full(len(geo),-1,dtype=int);mapping[aa]=np.arange(len(aa))
    mask=(mapping[pairs[0]]>=0)&(mapping[pairs[1]]>=0)
    data=Data(wire_x=torch.tensor(geo[aa,0],dtype=torch.float64),wire_y=torch.tensor(geo[aa,1],dtype=torch.float64),
              drift_d=torch.tensor(drift[aa],dtype=torch.float64),edge_index=torch.tensor(mapping[pairs[:,mask]]),num_nodes=len(aa))
    tracks,_=run_two_pass(data,None,None,None,*thresholds,probabilities=p[mask])
    for t in tracks:t['indices']=aa[t['indices']].tolist()
    return tracks


def trial(eid,snapshot):
    snapshot=Path(snapshot);contract=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))
    old=json.loads((snapshot/'two_stage_validation_v1/events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with np.load(snapshot/('validation_prediction_cache_'+contract['weight_sha256'][:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    methods={'baseline':old['methods']['stereo_assisted']}
    for name,thresholds in PROFILES.items():
        axial=threshold_tracks(geo,drift,pairs,p,thresholds);tracks,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        methods[name],_=score_tracks(tracks,truth,stereo,DEFAULT_CONFIG)
    return dict(event_id=eid,methods=methods)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'threshold_validation_v1';out.mkdir(exist_ok=False)
    ids=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))['event_ids'];rows={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(trial,eid,str(snapshot)) for eid in ids]):
            row=future.result();rows[row['event_id']]=row;save_json(out/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%25==0:print(json.dumps(dict(events=len(rows),total=len(ids))),flush=True)
    def aggregate(selected):return {name:summarize([rows[eid]['methods'][name] for eid in selected]) for name in ['baseline']+list(PROFILES)}
    dev=aggregate(ids[:100]);check=aggregate(ids[100:]);selected=select_profile(dev);frozen=selected
    b=check['baseline'];m=check[selected]
    if m['track_efficiency']<b['track_efficiency'] or m['fake_fraction']>b['fake_fraction'] or m['clone_fraction']>b['clone_fraction']:frozen='baseline'
    save_json(out/'summary.json',dict(profiles=PROFILES,event_ids=ids,dev=dev,validation_check=check,selected_on_dev=selected,frozen=frozen,
              source_sha256=sha(Path(__file__)),weight_sha256=sha(snapshot/'best_interaction_network_joint_no_time.pth'),test_used_for_selection=False))
    print(json.dumps(dict(selected=selected,frozen=frozen,dev=dev,check=check)),flush=True)


if __name__=='__main__':main()
