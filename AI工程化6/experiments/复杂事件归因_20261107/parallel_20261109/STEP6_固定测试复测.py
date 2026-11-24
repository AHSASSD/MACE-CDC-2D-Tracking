"""Run the frozen extractor on the old 300-event test only after new validation passes."""
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import local_extractor as x
e=x.e
OUT=x.HERE/'extractor_results/test'


def evaluate(eid):
    start=time.perf_counter()
    with np.load(e.DATA_ROOT/'events'/('event_%05d.npz'%eid)) as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(e.SNAPSHOT/('test_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])/('event_%05d.npz'%eid)) as f:pairs,p=f['pairs'],f['probabilities']
    baseline=json.loads((e.SNAPSHOT/'coarse_3d_inputs_test'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))['tracks']
    coarse=json.loads((e.SNAPSHOT/'coarse_3d_v4_fast_test/events'/('event_%05d.json'%eid)).read_text(encoding='utf-8'))['candidates']['guarded_loose']
    pool=[dict(t,indices=[i for i in t['indices'] if not stereo[i]]) for t in baseline];old_count=len(pool)
    seen={tuple(sorted(t['indices'])) for t in pool}
    for t in x.proposals(geo,drift,pairs,p):
        key=tuple(sorted(t['indices']))
        if key not in seen:pool.append(t);seen.add(key)
    matrix=np.zeros((len(cells),len(cells)),dtype=np.float32);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    confidence=x._MODEL.predict_proba(np.array([x.features(t,geo,cells,matrix) for t in pool]))[:,1] if pool else np.empty(0)
    owned={i for t in pool[:old_count] for i in t['indices']};methods={'baseline':baseline};seeds={'baseline':coarse};diag={}
    for name,config in x._PROFILES.items():
        chosen,d=x.resolve(pool,confidence,old_count,owned,len(cells),config)
        methods[name],seeds[name]=x.materialize(chosen,pool,baseline,coarse,geo,drift,cells,pairs,p)
        diag[name]=dict(d,changed_groups=sum(i>=old_count for i in chosen))
    counts={};matched={}
    for name,tracks in methods.items():counts[name],matched[name]=x.score_tracks(tracks,truth,stereo,x.DEFAULT_CONFIG)
    return dict(event_id=eid,methods=counts,matched=matched,seeds={k:x.seed_counts(v,truth,stereo,cells) for k,v in seeds.items()},
                diagnostics=diag,all_baseline_axial_hits_preserved=True,total_seconds=time.perf_counter()-start,
                axial_members={k:[[i for i in t['indices'] if not stereo[i]] for t in v] for k,v in methods.items()})


def main():
    # Dependency wait is bounded; no independent recurring monitor is created.
    end=time.monotonic()+1800
    while time.monotonic()<end:
        status=json.loads((x.HERE/'local_runner_status.json').read_text())
        if status['status'] in ('completed','failed'):break
        time.sleep(10)
    else:raise TimeoutError('Local development/validation did not finish in 30 minutes')
    if status['status']!='completed':raise RuntimeError('Upstream validation failed')
    validation_file=x.HERE/'extractor_results/validation/summary.json'
    if not validation_file.exists():
        e.save_json(x.HERE/'extractor_results/test_skipped.json',dict(reason='No selected configuration passed development/check'));return
    dev=json.loads((x.HERE/'extractor_results/dev/summary.json').read_text());val=json.loads(validation_file.read_text())
    selected=dev['selected']
    if selected=='baseline' or not val['paired'][selected]['non_regression']:
        e.save_json(x.HERE/'extractor_results/test_skipped.json',dict(reason='New validation did not pass the frozen non-regression gate',selected=selected));return
    profiles={selected:x.PROFILES[selected]}
    ids=json.loads((e.SNAPSHOT/'coarse_3d_v4_fast_test/contract.json').read_text())['event_ids']
    OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'events';folder.mkdir(exist_ok=True)
    plan=dict(event_ids=ids,profiles=profiles,weight_sha256=e.CONTRACT['weight_sha256'],
              calibration_sha256=e.sha(e.MODEL_ROOT/'local_evaluation_epoch095_20261107/candidate_confidence_v2/candidate_confidence.pkl'),
              source_sha256={p.name:e.sha(p) for p in [Path(__file__),x.HERE/'local_extractor.py']},
              validation_summary_sha256=e.sha(validation_file),scope='Old fixed 300-event test re-evaluation, previously seen; no test parameter selection',stable_deployment_changed=False)
    if (OUT/'contract.json').exists():assert json.loads((OUT/'contract.json').read_text())==plan
    e.save_json(OUT/'contract.json',plan);started=time.perf_counter();rows={}
    for eid in ids:
        path=folder/('event_%05d.json'%eid)
        if path.exists():rows[eid]=json.loads(path.read_text())
    with ProcessPoolExecutor(max_workers=8,initializer=x.initialize,initargs=('test',profiles)) as pool:
        for f in as_completed([pool.submit(evaluate,eid) for eid in ids if eid not in rows]):
            row=f.result();eid=row['event_id'];rows[eid]=row;e.save_json(folder/('event_%05d.json'%eid),row)
            if len(rows)%10==0:
                progress=dict(events=len(rows),total=len(ids),seconds=time.perf_counter()-started);e.save_json(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    summary=x.aggregate([rows[i] for i in ids]);summary.update(plan=plan,seconds=time.perf_counter()-started)
    e.verify_pins();e.save_json(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
