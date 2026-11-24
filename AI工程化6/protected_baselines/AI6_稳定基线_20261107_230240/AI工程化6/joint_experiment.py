"""No-time axial/stereo association pilot. Truth is stored only for scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint as recompute
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'AI工程化4 - 副本'
spec = importlib.util.spec_from_file_location('ai4_common', SOURCE / 'common.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
MAP = Path(r'D:\SRTP\CDC_cell_map\cdc_cell_map.xlsx')
BASELINE = SOURCE / 'whole_event_retraining/stage2_no_time_assessment/last_epoch_62.pth'

class MeanInteractionNetwork(legacy.InteractionNetwork):
    def forward(self,x,edge_index,edge_attr):
        row,col=edge_index
        degree=torch.bincount(col,minlength=x.size(0)).clamp_min(1).to(x.dtype).unsqueeze(1)
        for _ in range(self.n_passes):
            effects=self.relational_model(torch.cat([x[row],x[col],edge_attr],dim=-1))
            messages=torch.zeros(x.size(0),effects.size(1),device=x.device,dtype=effects.dtype)
            messages.index_add_(0,col,effects)
            x=self.object_model(torch.cat([x,messages/degree],dim=-1))+x
        return self.edge_classifier(effects).squeeze(-1)

class MemoryEfficientInteractionNetwork(legacy.InteractionNetwork):
    """Identical sum architecture, recomputing edge MLP activations in backward."""
    def forward(self,x,edge_index,edge_attr):
        row,col=edge_index
        for _ in range(self.n_passes):
            inputs=torch.cat([x[row],x[col],edge_attr],dim=-1)
            effects=recompute(self.relational_model,inputs,use_reentrant=False) if self.training else self.relational_model(inputs)
            messages=torch.zeros(x.size(0),effects.size(1),device=x.device,dtype=effects.dtype)
            messages.index_add_(0,col,effects)
            x=self.object_model(torch.cat([x,messages],dim=-1))+x
        return self.edge_classifier(effects).squeeze(-1)

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def graph(frame):
    # Geometry x,y is the wire midpoint at z=0, never a reconstructed hit.
    xy = frame[['x', 'y']].to_numpy(float)
    direction = frame[['u', 'v', 'w']].to_numpy(float)
    length = frame.l.to_numpy(float)
    assert np.allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-6)
    slope = direction[:, :2] / direction[:, 2:3]
    halfz = length * np.abs(direction[:, 2]) / 2
    stereo = np.linalg.norm(direction[:, :2], axis=1) > 1e-9
    radius = np.linalg.norm(xy, axis=1)
    phi = np.arctan2(xy[:, 1], xy[:, 0])
    drift = frame.d.to_numpy(float)
    # Broad phase includes each wire's maximum transverse excursion.
    excursion = np.linalg.norm(direction[:, :2], axis=1) * length / 2
    pairs = cKDTree(xy).query_pairs(120 + 2 * excursion.max(), output_type='ndarray')
    a, b = pairs.T
    delta = xy[b] - xy[a]
    dslope = slope[b] - slope[a]
    hz = np.minimum(halfz[a], halfz[b])
    denom = (dslope ** 2).sum(1)
    z = np.clip(-(delta * dslope).sum(1) / np.maximum(denom, 1e-20), -hz, hz)
    minimum = np.linalg.norm(delta + dslope * z[:, None], axis=1)
    pairs = pairs[minimum <= 120]
    edges = np.vstack([pairs, pairs[:, ::-1]])
    a, b = edges.T
    delta = xy[b] - xy[a]
    dslope = slope[b] - slope[a]
    hz = np.minimum(halfz[a], halfz[b])
    z = np.clip(-(delta * dslope).sum(1) / np.maximum((dslope ** 2).sum(1), 1e-20), -hz, hz)
    minimum = np.linalg.norm(delta + dslope * z[:, None], axis=1)
    dist = np.linalg.norm(delta, axis=1) / 800
    dphi = (phi[b] - phi[a] + np.pi) % (2*np.pi) - np.pi
    x = np.column_stack([xy/800, radius/800, drift/10, np.sin(phi), np.cos(phi),
                         direction, length/1600, stereo, radius/410])
    e = np.column_stack([(radius[b]-radius[a])/800, dist, dphi, dphi/(dist+1e-6),
                         minimum/800, z/800, hz/800, (drift[b]-drift[a])/10,
                         (direction[a]*direction[b]).sum(1), stereo[a], stereo[b]])
    trk = frame.TrkID.to_numpy(np.int64)
    assert np.isfinite(x).all() and np.isfinite(e).all()
    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=torch.tensor(edges.T.copy()),
                edge_attr=torch.tensor(e, dtype=torch.float32),
                y=torch.tensor(trk[a] == trk[b], dtype=torch.float32),
                stereo=torch.tensor(stereo), truth=torch.tensor(trk),
                hit_id=torch.tensor(frame.HitID.to_numpy(np.int64)),
                cell_id=torch.tensor(frame.CellID.to_numpy(np.int64)),
                event_id=int(frame.EvtID.iloc[0]))

def prepare(out, counts):
    geometry = pd.read_excel(MAP)
    expected = json.loads((SOURCE/'whole data/ENVIRONMENT.json').read_text(encoding='utf-8'))['hashes']
    expected_hash = next(v for k,v in expected.items() if k.endswith('cdc_cell_map.xlsx'))
    assert sha(MAP) == expected_hash, 'Geometry differs from production manifest'
    split = json.loads((SOURCE/'whole_event_retraining/event_split.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(20261105)
    ids = {name: sorted(rng.choice(split[key], min(n,len(split[key])), replace=False).tolist())
           for name,key,n in zip(['train','validation','test'],
               ['train_evt_ids','val_evt_ids','test_evt_ids'], counts)}
    assert not(set(ids['train'])&set(ids['validation']) or set(ids['train'])&set(ids['test']) or set(ids['validation'])&set(ids['test']))
    partition = {eid:name for name,values in ids.items() for eid in values}
    graphs = {name:[] for name in ids}
    columns = ['EvtID','TrkID','HitID','CellID','Good','CreatProc','WireType','d']
    hashes = {}
    for path in sorted((SOURCE/'whole data').glob('CDCSimHit_*.csv')):
        hashes[path.name] = sha(path)
        frame = pd.read_csv(path, usecols=columns)
        frame = frame.loc[frame.EvtID.isin(partition) & frame.Good.eq(True) & frame.CreatProc.eq('ExtendedDecayWithSpin')]
        frame = frame.merge(geometry, on='CellID', how='left', validate='many_to_one')
        for eid, f in frame.groupby('EvtID', sort=True):
            assert not f.HitID.duplicated().any(), 'HitID must identify individual records'
            assert np.isfinite(f[['x','y','u','v','w','l','d']].to_numpy()).all()
            assert np.array_equal(f.WireType.eq('Stereo'), np.hypot(f.u,f.v)>1e-9)
            g = graph(f)
            # Shuffle raw row order to check that neither time nor source order defines adjacency.
            altered = f.copy()
            altered['t'] = np.arange(len(f))*1e9
            altered['TrkID'] = -999
            other = graph(altered)
            assert torch.equal(g.x,other.x) and torch.equal(g.edge_attr,other.edge_attr)
            assert torch.equal(g.edge_index,other.edge_index)
            graphs[partition[int(eid)]].append(g)
        print('prepared',path.name, {k:len(v) for k,v in graphs.items()}, flush=True)
    for name,values in graphs.items():
        assert len(values)==len(ids[name])
        torch.save(values,out/f'{name}.pt')
    save(out/'manifest.json', {'created':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'ids':ids,
         'source_sha256':hashes,'geometry_sha256':sha(MAP),'baseline_sha256':sha(BASELINE),
         'input_columns':columns,'node_dim':12,'edge_dim':11,'time_used':False,
         'd_added_noise':0,'candidate_rule':'minimum transverse separation at common finite-wire z <=120mm',
         'note':'Pilot subset of original event partitions; all-pair same-track edge labels.'})
    return graphs

def probabilities(model,g,device,baseline=False):
    if baseline:
        a,b=g.edge_index
        mask=~g.stereo[a]&~g.stereo[b]&(g.edge_attr[:,1]>0)&(g.edge_attr[:,1]<=.15)
        return model(g.x[:,:6].to(device),g.edge_index[:,mask].to(device),g.edge_attr[mask,:4].to(device)).sigmoid().cpu(), mask
    return model(g.x.to(device),g.edge_index.to(device),g.edge_attr.to(device)).sigmoid().cpu(), torch.ones(g.num_edges,dtype=torch.bool)

def metrics(graphs,predictions,threshold,baseline=False):
    totals={k:np.zeros(4,dtype=np.int64) for k in ['AA','AS','SS']}
    tracks=np.zeros(10,dtype=float)
    for g,(p,mask) in zip(graphs,predictions):
        a,b=g.edge_index[:,mask]; y=g.y[mask].bool(); selected=p>=threshold
        kind=g.stereo[a].int()+g.stereo[b].int()
        for k,name in enumerate(totals):
            use=kind==k; yes=selected&use
            totals[name]+=np.array([(yes&y).sum(),(yes&~y).sum(),((~selected)&use&y).sum(),use.sum()])
        adj=coo_matrix((np.ones(int(selected.sum())),(a[selected],b[selected])),shape=(g.num_nodes,g.num_nodes)).tocsr()
        _,component=connected_components(adj,directed=False)
        candidates=[np.where(component==c)[0] for c in np.unique(component)]
        candidates=[c for c in candidates if (~g.stereo[c]).sum()>=5]
        truth=g.truth.numpy(); axial=~g.stereo.numpy(); stereo=g.stereo.numpy()
        eligible=[t for t in np.unique(truth) if np.sum((truth==t)&axial)>=5]
        found={t:0 for t in eligible}; fake=0
        for c in candidates:
            best=None
            for t in eligible:
                n=np.sum((truth[c]==t)&axial[c]); purity=n/max(np.sum(axial[c]),1); recall=n/np.sum((truth==t)&axial)
                if purity>=.7 and recall>=.7 and (best is None or n>best[0]): best=(n,t)
            if best is None: fake+=1; continue
            t=best[1]; found[t]+=1
            tracks[6]+=best[0]/max(np.sum((truth==t)&axial),1)
            st=c[stereo[c]]; correct=np.sum(truth[st]==t)
            tracks[3]+=correct; tracks[4]+=len(st); tracks[5]+=np.sum((truth==t)&stereo)
        tracks[:3]+= [len(eligible),sum(v>0 for v in found.values()),fake]
        tracks[7]+=sum(max(v-1,0) for v in found.values())
        tracks[8]+=len(candidates)
        tracks[9]+=sum(found.values())
    result={}
    for name,(tp,fp,fn,n) in totals.items():
        precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
        result[name]={'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-12),'edges':int(n)}
    result['candidates']={'truth_tracks':int(tracks[0]),'matched_tracks':int(tracks[1]),'axial_efficiency':tracks[1]/max(tracks[0],1),'fake_candidates':int(tracks[2]),'fake_fraction':tracks[2]/max(tracks[8],1),'clone_candidates':int(tracks[7]),'clone_fraction':tracks[7]/max(tracks[8],1),'output_candidates':int(tracks[8]),'axial_completeness_matched':tracks[6]/max(tracks[9],1),'stereo_purity':tracks[3]/max(tracks[4],1),'stereo_completeness_matched':tracks[3]/max(tracks[5],1),'shared_output_hits':0}
    result['threshold']=threshold
    return result

def predict_all(model,graphs,device,baseline=False):
    model.eval()
    with torch.no_grad(): return [probabilities(model,g,device,baseline) for g in graphs]

def close_apps_after_delay():
    # Independent helper survives the training process and closes only the requested applications.
    code="import time,subprocess; time.sleep(1800); subprocess.run(['powershell','-NoProfile','-Command', \"Get-Process -Name Endfield,MaaEnd -ErrorAction SilentlyContinue | Stop-Process -Force\"] )"
    proc=subprocess.Popen([sys.executable,'-c',code],creationflags=subprocess.CREATE_NO_WINDOW)
    return {'helper_pid':proc.pid,'deadline_unix':time.time()+1800,'process_names':['Endfield','MaaEnd']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=int,default=256); p.add_argument('--validation',type=int,default=64); p.add_argument('--test',type=int,default=64); p.add_argument('--epochs',type=int,default=12); p.add_argument('--run',default='pilot'); p.add_argument('--close-apps',action='store_true'); p.add_argument('--reuse'); p.add_argument('--aggregation',choices=['sum','mean'],default='sum'); p.add_argument('--skip-test',action='store_true'); p.add_argument('--initialize'); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--memory-safe',action='store_true'); args=p.parse_args()
    random.seed(20261105); np.random.seed(20261105); torch.manual_seed(20261105); torch.set_num_threads(2)
    out=HERE/args.run; out.mkdir(exist_ok=False)
    if args.reuse:
        reuse=HERE/args.reuse
        data={name:torch.load(reuse/f'{name}.pt',weights_only=False) for name in ['train','validation','test']}
        save(out/'manifest.json',{'reuses':str(reuse),'manifest_sha256':sha(reuse/'manifest.json'),'arguments':vars(args),'test_status':'exploratory, previously observed pilot test'})
    else:
        data=prepare(out,[args.train,args.validation,args.test])
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    cls=MeanInteractionNetwork if args.aggregation=='mean' else legacy.InteractionNetwork
    if args.memory_safe:
        assert args.aggregation=='sum'
        cls=MemoryEfficientInteractionNetwork
        if device=='cuda': torch.cuda.set_per_process_memory_fraction(.3)
    model=cls(12,11,hidden=64,n_passes=3).to(device)
    if args.initialize:
        initial=torch.load(HERE/args.initialize,weights_only=True,map_location=device)
        model.load_state_dict(initial['model'])
    save(out/'configuration.json',{'arguments':vars(args),'seed':20261105,'architecture':{'hidden':64,'passes':3,'aggregation':args.aggregation},'initialization_sha256':sha(HERE/args.initialize) if args.initialize else None,'source_script_sha256':sha(Path(__file__))})
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    save(out/'training_started.json',{'time_unix':time.time(),'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'app_close':close_apps_after_delay() if args.close_apps else None})
    best=-1; history=[]
    for epoch in range(1,args.epochs+1):
        start=time.perf_counter(); model.train(); losses=[]
        for index in np.random.permutation(len(data['train'])):
            g=data['train'][index].to(device); optimizer.zero_grad(set_to_none=True)
            logits=model(g.x,g.edge_index,g.edge_attr)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,g.y)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); losses.append(loss.item()); g.cpu()
            if args.memory_safe and device=='cuda':
                del logits,loss
                torch.cuda.empty_cache()
        pred=predict_all(model,data['validation'],device)
        candidates=[metrics(data['validation'],pred,t) for t in [.3,.5,.7,.85,.95]]
        chosen=max(candidates,key=lambda v:sum(v[k]['f1'] for k in ['AA','AS','SS'])/3)
        score=sum(chosen[k]['f1'] for k in ['AA','AS','SS'])/3
        row={'epoch':epoch,'loss':float(np.mean(losses)),'validation_macro_f1':score,'seconds':time.perf_counter()-start}
        history.append(row); save(out/'history.json',history); print(row,flush=True)
        torch.save({'model':model.state_dict(),'epoch':epoch,'threshold':chosen['threshold'],'hidden':64,'passes':3},out/'last.pt')
        if score>best:
            best=score; torch.save({'model':model.state_dict(),'epoch':epoch,'threshold':chosen['threshold'],'hidden':64,'passes':3},out/'best.pt')
    if args.skip_test: return
    checkpoint=torch.load(out/'best.pt',weights_only=True,map_location=device); model.load_state_dict(checkpoint['model'])
    # Test is read for scoring only after model and threshold are frozen on validation.
    report={'joint':metrics(data['test'],predict_all(model,data['test'],device),checkpoint['threshold'])}
    baseline=legacy.InteractionNetwork(6,4).to(device); baseline.load_state_dict(torch.load(BASELINE,weights_only=True,map_location=device))
    val=predict_all(baseline,data['validation'],device,True)
    threshold=max([metrics(data['validation'],val,t,True) for t in [.3,.5,.7,.85,.95]],key=lambda r:r['AA']['f1'])['threshold']
    report['axial_baseline']=metrics(data['test'],predict_all(baseline,data['test'],device,True),threshold,True)
    report['limitations']=['Small pilot, not full-data convergence.','Connected-component candidate decoder with >=5 axial hits and 70% purity/completeness matching is exploratory, not the official two-pass benchmark.','No three-dimensional fit or momentum result claimed.','All-pair edge recall is conditional on geometric candidate graph.']
    save(out/'test_report.json',report); print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
