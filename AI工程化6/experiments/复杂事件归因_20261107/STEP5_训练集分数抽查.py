"""Sample training scores without optimization; compare like-for-like topology."""
import experiment_common as e
import json
import time
import numpy as np
import torch
from common import JointEventDataset,JointInteractionNetwork
from server_runtime import loader,DevicePrefetch
from STEP5_两级寻迹 import undirected_predictions
from STEP4_真边分数与连通性诊断 import topology,BINS


@torch.no_grad()
def main():
    out=e.EXP/'training_score_probe_v1';out.mkdir(exist_ok=True);events=out/'events';events.mkdir(exist_ok=True)
    data=JointEventDataset(e.DATA_ROOT,'train');ids=list(map(int,np.random.default_rng(20261112).permutation(data.ids)[:200]))
    assert not set(ids).intersection(e.IDS)
    plan=dict(event_ids=ids,split='train',sampling_seed=20261112,model_sha256=e.CONTRACT['weight_sha256'],
              source_sha256=e.sha(__file__),topology_source_sha256=e.sha(e.EXP/'STEP4_真边分数与连通性诊断.py'),optimization_steps=0)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Probe contract changed')
    e.save_json(out/'contract.json',plan);lookup={eid:i for i,eid in enumerate(data.ids)};remaining=[eid for eid in ids if not (events/f'event_{eid:05d}.json').exists()]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    model=JointInteractionNetwork(chunk_size=65536).to(device);model.load_state_dict(torch.load(e.SNAPSHOT/'best_interaction_network_joint_no_time.pth',weights_only=True,map_location=device));model.eval()
    started=time.time()
    for index,g in enumerate(DevicePrefetch(loader(torch.utils.data.Subset(data,[lookup[eid] for eid in remaining]),1,workers=2),device)):
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,p=undirected_predictions(g.edge_index.cpu().numpy(),p);eid=int(g.event_id.item())
        truth=g.truth.cpu().numpy();stereo=g.stereo.cpu().numpy();same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
        kind=stereo[pairs[0]].astype(int)+stereo[pairs[1]].astype(int);bins=np.minimum((p*BINS).astype(int),BINS-1)
        hist=np.zeros((3,2,BINS),dtype=np.int64)
        for k in range(3):
            for j,mask in enumerate([same,~same]):hist[k,j]=np.bincount(bins[(kind==k)&mask],minlength=BINS)
        tracks=[];tids,n=np.unique(truth[(truth>=0)&~stereo],return_counts=True)
        for tid,size in zip(tids,n):
            if size<5:continue
            nodes=np.flatnonzero((truth==tid)&~stereo);mapping=np.full(len(truth),-1,dtype=int);mapping[nodes]=np.arange(len(nodes))
            keep=same&(kind==0)&(truth[pairs[0]]==tid);row=topology(len(nodes),mapping[pairs[:,keep]],p[keep]);row.pop('repair_local_edge_indices');row['truth_id']=int(tid);tracks.append(row)
        np.save(events/f'hist_{eid:05d}.npy',hist);e.save_json(events/f'event_{eid:05d}.json',dict(event_id=eid,tracks=tracks))
        if (index+1)%25==0:print(json.dumps(dict(events=index+1,total=len(remaining),seconds=time.time()-started)),flush=True)
    from collections import Counter
    hist=sum(np.load(events/f'hist_{eid:05d}.npy') for eid in ids);tracks=[t for eid in ids for t in json.loads((events/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['tracks']]
    np.save(out/'histogram.npy',hist)
    summary=dict(plan=plan,eligible_tracks=len(tracks),categories=dict(Counter(t['category'] for t in tracks)),
                 pure_match_scale_available=sum(t['largest_relaxed_component']>=t['pure_match_hits_required'] for t in tracks),
                 median_track_true_edge_score=float(np.median([t['median_true_score'] for t in tracks if t['median_true_score'] is not None])),
                 seconds=time.time()-started,interpretation='Pure labeled graph connectivity, not reconstructed-track efficiency; no optimization occurred')
    e.verify_pins();e.save_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
