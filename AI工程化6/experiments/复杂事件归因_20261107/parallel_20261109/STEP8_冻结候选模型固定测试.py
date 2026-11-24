"""One model frozen on validation, re-evaluated on the previously seen fixed300 test."""
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import STEP7_训练结果与同流程复测 as r

ROOT=r.OUT
OUT=ROOT/'fixed_test'
KEY='control_012_test'


@torch.no_grad()
def predictions(ids,weight):
    cache=ROOT/'prediction_cache'/KEY;cache.mkdir(parents=True,exist_ok=True)
    data=r.JointEventDataset(r.e.DATA_ROOT,'test');lookup={eid:i for i,eid in enumerate(data.ids)}
    todo=[i for i in ids if not (cache/('event_%05d.npz'%i)).exists()]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    model=r.JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(weight,map_location=device,weights_only=True));model.eval()
    for j,g in enumerate(r.DevicePrefetch(r.loader(torch.utils.data.Subset(data,[lookup[i] for i in todo]),1,workers=2),device)):
        with torch.autocast(device.type,enabled=device.type=='cuda'):prob=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,prob=r.undirected_predictions(g.edge_index.cpu().numpy(),prob);eid=int(g.event_id.item())
        path=cache/('event_%05d.npz'%eid);tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,pairs=pairs.astype(np.int32),probabilities=prob)
        tmp.replace(path)
        if (j+1)%50==0:print(json.dumps(dict(phase='prediction',events=j+1,total=len(todo))),flush=True)
    del model;torch.cuda.empty_cache()


def main():
    end=time.monotonic()+1800
    while not (ROOT/'summary.json').exists():
        if time.monotonic()>end:raise TimeoutError('Validation did not finish in 30 minutes')
        time.sleep(10)
    validation=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'));all_metrics=validation['stable_tracker_300']
    candidate=all_metrics['control_012'];base=all_metrics['initial126']
    # This confirmation model is frozen before reading any test-event measurements or labels.
    assert candidate['tracks']['track_efficiency']>=base['tracks']['track_efficiency']
    assert candidate['tracks']['fake_fraction']<=base['tracks']['fake_fraction']
    assert candidate['seed']['efficiency']>=base['seed']['efficiency']
    assert candidate['seed']['fake_fraction']<=base['seed']['fake_fraction']
    assert candidate['seed']['clone_fraction']<=base['seed']['clone_fraction']
    ids=json.loads((r.e.SNAPSHOT/'coarse_3d_v4_fast_test/contract.json').read_text(encoding='utf-8'))['event_ids']
    weight=ROOT/'cloud/control/epoch_012.pth';OUT.mkdir(exist_ok=True)
    plan=dict(model='control_012',weight_sha256=r.e.sha(weight),initial_weight_sha256=r.e.CONTRACT['weight_sha256'],
              event_ids=ids,validation_summary_sha256=r.e.sha(ROOT/'summary.json'),source_sha256={p.name:r.e.sha(p) for p in [Path(__file__),r.HERE/'STEP7_训练结果与同流程复测.py']},
              selection='Chosen only from validation for coarse3D efficiency/fake/clone trade-off and axial efficiency/fake; axial clone increased by one in validation and is explicitly checked on test',
              scope='Previously seen fixed300 test re-evaluation, no test-based model or parameter selection',stable_deployment_changed=False)
    if (OUT/'contract.json').exists():assert json.loads((OUT/'contract.json').read_text(encoding='utf-8'))==plan
    r.e.save_json(OUT/'contract.json',plan);started=time.perf_counter()
    baseline={}
    for eid in ids:
        with np.load(r.e.DATA_ROOT/'events'/('event_%05d.npz'%eid)) as f:truth,cells=f['truth'],f['cell_id']
        geo=np.load(r.e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
        tracks=json.loads((r.e.SNAPSHOT/'coarse_3d_inputs_test'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))['tracks']
        coarse=json.loads((r.e.SNAPSHOT/'coarse_3d_v4_fast_test/events'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))['candidates']['guarded_loose']
        counts,matched=r.score_tracks(tracks,truth,stereo,r.DEFAULT_CONFIG)
        baseline[eid]=dict(counts=counts,matched=matched,seed=r.seed_counts(coarse,truth,stereo,cells))
    predictions(ids,weight);folder=OUT/'events';folder.mkdir(exist_ok=True);rows={}
    for eid in ids:
        path=folder/('event_%05d.json'%eid)
        if path.exists():rows[eid]=json.loads(path.read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=8,initializer=r.initialize) as pool:
        for future in as_completed([pool.submit(r.event_evaluate,KEY,eid) for eid in ids if eid not in rows]):
            row=future.result();eid=row['event_id'];rows[eid]=row;r.e.save_json(folder/('event_%05d.json'%eid),row)
            if len(rows)%20==0:
                progress=dict(events=len(rows),total=len(ids),seconds=time.perf_counter()-started);r.e.save_json(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    samples=np.random.default_rng(20261109).integers(len(ids),size=(2000,len(ids)))
    den=np.array([baseline[i]['counts']['truth_tracks'] for i in ids]);intervals={}
    for kind in ['counts','seed']:
        delta=np.array([rows[i][kind]['matched_tracks']-baseline[i][kind]['matched_tracks'] for i in ids])
        intervals[kind]=np.quantile(delta[samples].sum(1)/den[samples].sum(1),[.025,.975]).tolist()
    results={}
    for label,source in [('initial126',baseline),('control_012',rows)]:
        results[label]=dict(tracks=r.summarize([source[i]['counts'] for i in ids]),seed=r.seed_summary([source[i]['seed'] for i in ids]))
    results['control_012'].update(gained=sum(len(set(rows[i]['matched'])-set(baseline[i]['matched'])) for i in ids),lost=sum(len(set(baseline[i]['matched'])-set(rows[i]['matched'])) for i in ids))
    result=dict(models=results,paired_efficiency_difference_95_ci=intervals,plan=plan,seconds=time.perf_counter()-started)
    r.e.verify_pins();r.e.save_json(OUT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
