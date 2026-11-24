"""Frozen tracking configuration on the complete fixed test partition."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import ROOT,MODEL_ROOT,DATA_ROOT,JointEventDataset,JointInteractionNetwork,sha,save_json
from server_runtime import loader,DevicePrefetch
from STEP5_两级寻迹 import undirected_predictions,axial_two_pass,attach_stereo,DEFAULT_CONFIG,score_tracks,summarize
from STEP8_寻迹改进评估 import repair_tracks


@torch.no_grad()
def cache_predictions(snapshot,cache,ids):
    data=JointEventDataset(DATA_ROOT,'test');index={eid:i for i,eid in enumerate(data.ids)}
    remaining=[eid for eid in ids if not (cache/f'event_{eid:05d}.npz').exists()]
    if not remaining:return
    device=torch.device('cuda');model=JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(snapshot/'best_interaction_network_joint_no_time.pth',weights_only=True,map_location=device));model.eval();torch.set_num_threads(2)
    for j,g in enumerate(DevicePrefetch(loader(torch.utils.data.Subset(data,[index[eid] for eid in remaining]),1),device)):
        with torch.autocast(device_type='cuda'):
            p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),p);eid=int(g.event_id.item())
        np.savez_compressed(cache/f'event_{eid:05d}.npz',pairs=pairs.astype(np.int32),probabilities=p)
        if (j+1)%100==0:print(json.dumps(dict(phase='test_predictions',events=j+1,total=len(remaining))),flush=True)


def test_event(eid,cache,profile):
    with np.load(Path(cache)/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    axial,_=axial_two_pass(geo,drift,pairs,p)
    base,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    base_counts,base_matched=score_tracks(base,truth,stereo,DEFAULT_CONFIG)
    selected_axial=axial if profile=='baseline' else repair_tracks(axial,geo,drift,cells,pairs,p,profile=='repair_recover')
    selected,_=attach_stereo(selected_axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    counts,matched=score_tracks(selected,truth,stereo,DEFAULT_CONFIG)
    used=[i for t in selected for i in t['indices']];assert len(used)==len(set(used))
    return dict(event_id=eid,baseline=base_counts,selected=counts,baseline_matched=base_matched,selected_matched=matched,
                candidates=selected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--events',type=int,default=1500);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;validation=snapshot/'tracking_repair_validation_v1/summary.json'
    val=json.loads(validation.read_text(encoding='utf-8'));profile=val['selected_on_dev']
    # Require the held validation half to support all stated non-regression constraints.
    b=val['validation_check']['baseline'];m=val['validation_check'][profile]
    if m['track_efficiency']<b['track_efficiency'] or m['fake_fraction']>b['fake_fraction'] or m['clone_fraction']>b['clone_fraction']:profile='baseline'
    weight=sha(snapshot/'best_interaction_network_joint_no_time.pth')
    metadata=json.loads((snapshot/'model_metadata_joint_no_time.json').read_text(encoding='utf-8'))
    if metadata['dataset_metadata_sha256']!=sha(DATA_ROOT/'dataset_metadata_joint_no_time.json'):raise ValueError('Dataset changed')
    if val['plan']['weight_sha256']!=weight:raise ValueError('Weight changed after validation selection')
    ids=list(map(int,np.random.default_rng(20261107).permutation(JointEventDataset(DATA_ROOT,'test').ids)[:args.events]))
    out=snapshot/'frozen_tracking_test_v1';out.mkdir(exist_ok=True)
    plan=dict(profile=profile,event_ids=ids,weight_sha256=weight,dataset_metadata_sha256=metadata['dataset_metadata_sha256'],
              validation_summary_sha256=sha(validation),source_hashes={name:sha(ROOT/name) for name in ['STEP9_固定方案测试.py','STEP8_寻迹改进评估.py','STEP5_两级寻迹.py','two_stage_core.py']},
              test_used_for_selection=False,scoring_config=DEFAULT_CONFIG)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Test contract changed')
    save_json(out/'contract.json',plan)
    cache=snapshot/('test_prediction_cache_'+weight[:12]);cache.mkdir(exist_ok=True)
    cache_predictions(snapshot,cache,ids)
    event_dir=out/'events';event_dir.mkdir(exist_ok=True);rows={};started=time.time()
    for eid in ids:
        if (event_dir/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((event_dir/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(test_event,eid,str(cache),profile) for eid in ids if eid not in rows]
        for future in as_completed(futures):
            row=future.result();rows[row['event_id']]=row;save_json(event_dir/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%50==0:
                progress=dict(phase='tracking',events=len(rows),total=len(ids),seconds=time.time()-started)
                save_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    baseline=summarize([rows[eid]['baseline'] for eid in ids]);selected=summarize([rows[eid]['selected'] for eid in ids])
    recovered=sum(len(set(row['selected_matched'])-set(row['baseline_matched'])) for row in rows.values())
    lost=sum(len(set(row['baseline_matched'])-set(row['selected_matched'])) for row in rows.values())
    denominator=np.array([rows[eid]['baseline']['truth_tracks'] for eid in ids]);gain=np.array([rows[eid]['selected']['matched_tracks']-rows[eid]['baseline']['matched_tracks'] for eid in ids])
    draws=np.random.default_rng(20261107).integers(len(ids),size=(2000,len(ids)))
    interval=np.quantile(gain[draws].sum(1)/denominator[draws].sum(1),[.025,.975]).tolist()
    report=dict(plan=plan,events=len(ids),baseline=baseline,selected=selected,recovered=recovered,lost=lost,
                paired_efficiency_difference_95_interval=interval,seconds=time.time()-started,
                scope='fixed test partition repeated assessment; previously inspected under epoch51, not a new untouched holdout')
    save_json(out/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
