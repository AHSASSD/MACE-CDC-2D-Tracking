import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
from common import MODEL_ROOT,DATA_ROOT,save_json,sha
from STEP5_两级寻迹 import score_tracks,summarize,DEFAULT_CONFIG
from STEP8_寻迹改进评估 import select_profile
from joint_candidate_tracking import reconstruct


def event_trial(eid,snapshot):
    snapshot=Path(snapshot);reference=snapshot/'two_stage_validation_v1'
    old=json.loads((reference/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    weight=json.loads((reference/'experiment_contract.json').read_text(encoding='utf-8'))['weight_sha256']
    with np.load(snapshot/('validation_prediction_cache_'+weight[:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    candidates,diag=reconstruct(geo,drift,cells,pairs,p,old['tracks']['axial_two_pass'],old['tracks']['stereo_assisted'])
    methods={'baseline':old['methods']['stereo_assisted']};matched={'baseline':old['matched_truth_ids']['stereo_assisted']}
    for name,tracks in candidates.items():methods[name],matched[name]=score_tracks(tracks,truth,stereo,DEFAULT_CONFIG)
    return dict(event_id=eid,methods=methods,matched=matched,candidates=candidates,diagnostics=diag)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch095_20261107');parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'joint_pool_validation_v1';out.mkdir(exist_ok=True)
    ids=json.loads((snapshot/'two_stage_validation_v1/experiment_contract.json').read_text(encoding='utf-8'))['event_ids']
    plan=dict(ids=ids,source=sha(Path(__file__).with_name('joint_candidate_tracking.py')))
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=plan:raise ValueError('Changed pool source')
    save_json(out/'contract.json',plan);rows={};started=time.time()
    for eid in ids:
        if (out/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((out/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(event_trial,eid,str(snapshot)) for eid in ids if eid not in rows]):
            row=future.result();rows[row['event_id']]=row;save_json(out/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%25==0:print(json.dumps(dict(events=len(rows),total=len(ids),seconds=time.time()-started)),flush=True)
    def aggregate(selected):return {name:summarize([rows[eid]['methods'][name] for eid in selected]) for name in ['baseline','joint_residual','joint_supported','joint_global']}
    dev=aggregate(ids[:100]);check=aggregate(ids[100:]);selected=select_profile(dev)
    save_json(out/'summary.json',dict(dev=dev,check=check,all=aggregate(ids),selected_on_dev=selected,plan=plan))
    print(json.dumps(dict(selected=selected,dev=dev,check=check)),flush=True)


if __name__=='__main__':main()
