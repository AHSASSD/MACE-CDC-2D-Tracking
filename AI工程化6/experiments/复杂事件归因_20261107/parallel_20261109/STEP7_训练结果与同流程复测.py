"""Collect paired training, then compare frozen selected weights in the stable tracker."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import experiment_common as e
import json
import hashlib
import pickle
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from common import JointEventDataset,JointInteractionNetwork
from server_runtime import loader,DevicePrefetch
from STEP5_两级寻迹 import axial_two_pass,attach_stereo,score_tracks,summarize,undirected_predictions,DEFAULT_CONFIG
from hybrid_joint_tracker import track
from coarse_3d_membership import propose,select
from 生成粗三维成员报告 import seed_counts
from STEP2_弱边候选校准 import seed_summary

OUT=HERE/'model_results_20261109'
MODELS={'control_006':('control',6),'control_012':('control',12),'balanced_006':('balanced',6),'balanced_012':('balanced',12)}
_CALIBRATOR=None


def initialize():
    global _CALIBRATOR
    e.initialize_worker()
    with (e.MODEL_ROOT/'local_evaluation_epoch095_20261107/candidate_confidence_v2/candidate_confidence.pkl').open('rb') as f:_CALIBRATOR=pickle.load(f)


def extract():
    archive=OUT/'results.tar.gz'
    assert e.sha(archive)=='e70366bf17fe65976ba0b8980fc3a3abbcc28982a82d736bd4e6dd39eb8f7dc9'
    dest=OUT/'cloud';dest.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            target=(dest/member.name).resolve()
            assert target.is_relative_to(dest.resolve()) and member.isfile(),member.name
        tf.extractall(dest)
    contract=json.loads((dest/'contract.json').read_text())
    assert contract['initial_weight_sha256']==e.CONTRACT['weight_sha256']
    assert contract['dataset_sha256']==e.sha(e.DATA_ROOT/'dataset_metadata_joint_no_time.json')
    return contract


def event_evaluate(model_key,eid):
    start=time.perf_counter()
    with np.load(e.DATA_ROOT/'events'/('event_%05d.npz'%eid)) as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    cache=(e.SNAPSHOT/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12])) if model_key=='initial126_raw' else OUT/'prediction_cache'/model_key
    with np.load(cache/('event_%05d.npz'%eid)) as f:pairs,p=f['pairs'],f['probabilities']
    axial,_=axial_two_pass(geo,drift,pairs,p);joint,_=attach_stereo(axial,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
    if model_key=='initial126_raw':
        counts,matched=score_tracks(joint,truth,stereo,DEFAULT_CONFIG)
        return dict(event_id=eid,model=model_key,counts=counts,matched=matched)
    final=track(geo,drift,cells,pairs,p,axial,joint,_CALIBRATOR,.02,.2)
    coarse,_=select(final,propose(final,geo,drift,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    counts,matched=score_tracks(final,truth,stereo,DEFAULT_CONFIG)
    return dict(event_id=eid,model=model_key,counts=counts,matched=matched,seed=seed_counts(coarse,truth,stereo,cells),seconds=time.perf_counter()-start,
                axial_members=[[i for i in t['indices'] if not stereo[i]] for t in final],
                confirmed_stereo_members=[t['confirmed_stereo_indices'] for t in coarse])


@torch.no_grad()
def predict(model_key,ids):
    arm,epoch=MODELS[model_key];weight=OUT/'cloud'/arm/('epoch_%03d.pth'%epoch)
    cache=OUT/'prediction_cache'/model_key;cache.mkdir(parents=True,exist_ok=True)
    data=JointEventDataset(e.DATA_ROOT,'validation');index={eid:i for i,eid in enumerate(data.ids)}
    remaining=[eid for eid in ids if not (cache/('event_%05d.npz'%eid)).exists()]
    if not remaining:return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    model=JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(weight,map_location=device,weights_only=True));model.eval()
    for j,g in enumerate(DevicePrefetch(loader(torch.utils.data.Subset(data,[index[eid] for eid in remaining]),1,workers=2),device)):
        with torch.autocast(device.type,enabled=device.type=='cuda'):prob=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,prob=undirected_predictions(g.edge_index.cpu().numpy(),prob);eid=int(g.event_id.item())
        path=cache/('event_%05d.npz'%eid);tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,pairs=pairs.astype(np.int32),probabilities=prob)
        tmp.replace(path)
        if (j+1)%50==0:print(json.dumps(dict(phase='prediction',model=model_key,events=j+1,total=len(remaining))),flush=True)
    del model;torch.cuda.empty_cache()


def main():
    contract=extract();e.verify_pins()
    ids=json.loads((HERE/'extractor_results/validation/contract.json').read_text())['event_ids']
    plan=dict(event_ids=ids,models={k:dict(arm=a,epoch=n,weight_sha256=e.sha(OUT/'cloud'/a/('epoch_%03d.pth'%n))) for k,(a,n) in MODELS.items()},
              source_sha256=e.sha(Path(__file__)),initial_weight_sha256=e.CONTRACT['weight_sha256'],
              selection='Epoch6 maximizes original two-pass axial efficiency; epoch12 is lower-fake/lower-clone checkpoint, both chosen only from old200 member validation',
              fixed_tracker='Stable hybrid .02/.2 + frozen epoch95 candidate calibrator + guarded_loose coarse3D',
              evaluation_scope='Same extra300 validation previously used for extractor study; no test events; all models use same local runtime',
              runtime=dict(torch=torch.__version__,device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'))
    if (OUT/'evaluation_contract.json').exists():assert json.loads((OUT/'evaluation_contract.json').read_text())==plan
    e.save_json(OUT/'evaluation_contract.json',plan)
    baseline={}
    for eid in ids:
        row=json.loads((HERE/'extractor_results/validation/events'/('event_%05d.json'%eid)).read_text())
        baseline[eid]=dict(event_id=eid,counts=row['methods']['baseline'],seed=row['seeds']['baseline'],matched=row['matched']['baseline'])
    rows={k:{} for k in ['initial126_raw']+list(MODELS)};tasks=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=6,initializer=initialize) as pool:
        for eid in contract['member_event_ids']:
            path=OUT/'events/initial126_raw'/('event_%05d.json'%eid)
            if path.exists():rows['initial126_raw'][eid]=json.loads(path.read_text())
            else:tasks.append(pool.submit(event_evaluate,'initial126_raw',eid))
        for key in MODELS:
            predict(key,ids)
            for eid in ids:
                path=OUT/'events'/key/('event_%05d.json'%eid)
                if path.exists():rows[key][eid]=json.loads(path.read_text())
                else:tasks.append(pool.submit(event_evaluate,key,eid))
        for future in as_completed(tasks):
            row=future.result();key=row['model'];eid=row['event_id'];rows[key][eid]=row
            folder=OUT/'events'/key;folder.mkdir(parents=True,exist_ok=True);e.save_json(folder/('event_%05d.json'%eid),row)
            count=sum(len(v) for v in rows.values())
            if count%25==0:
                progress=dict(phase='tracking',events=count,total=len(contract['member_event_ids'])+len(ids)*len(MODELS),seconds=time.perf_counter()-started)
                e.save_json(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    summaries={'initial126':dict(tracks=summarize([baseline[i]['counts'] for i in ids]),seed=seed_summary([baseline[i]['seed'] for i in ids]))}
    for key in MODELS:
        ordered=[rows[key][i] for i in ids];den=np.array([baseline[i]['counts']['truth_tracks'] for i in ids])
        delta=np.array([r['counts']['matched_tracks']-baseline[r['event_id']]['counts']['matched_tracks'] for r in ordered])
        samples=np.random.default_rng(20261109).integers(len(ids),size=(2000,len(ids)))
        ci=np.quantile(delta[samples].sum(1)/den[samples].sum(1),[.025,.975]).tolist()
        summaries[key]=dict(tracks=summarize([r['counts'] for r in ordered]),seed=seed_summary([r['seed'] for r in ordered]),
                            gained=sum(len(set(r['matched'])-set(baseline[r['event_id']]['matched'])) for r in ordered),
                            lost=sum(len(set(baseline[r['event_id']]['matched'])-set(r['matched'])) for r in ordered),paired_efficiency_difference_95_ci=ci)
    result=dict(initial126_original_two_pass_200=summarize([rows['initial126_raw'][i]['counts'] for i in contract['member_event_ids']]),
                stable_tracker_300=summaries,contract=plan,seconds=time.perf_counter()-started)
    e.verify_pins();e.save_json(OUT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
