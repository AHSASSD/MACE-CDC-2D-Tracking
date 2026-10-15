"""Read-only leakage and timing-shortcut audit. Original data and weights are immutable."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'whole_event_retraining'
OUT = DATA / 'time_audit_reproduction'
sys.path.insert(0, str(ROOT))
from common import InteractionNetwork, model_dimensions
from STEP4_两级寻迹 import predict, truth_sets
from STEP6_事件切片可视化 import load_graphs
from assessment_utils import track, summary, init_worker


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def quantile(values):
    a = np.asarray(values)
    return {str(q): float(np.quantile(a, q)) for q in [0, .1, .5, .9, .99, 1]}


def fingerprint(a):
    a = np.ascontiguousarray(a[np.lexsort(a.T[::-1])], dtype='<f4')
    return hashlib.sha256(a.tobytes()).hexdigest()


def scan():
    split = read(DATA / 'event_split.json')
    expected = {name: set(split[key]) for name, key in [('train','train_evt_ids'), ('validation','val_evt_ids'), ('test','test_evt_ids')]}
    assert all(not expected[a] & expected[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    seen = defaultdict(list)
    track_seen = defaultdict(set)
    duplicate_events = []
    event_spans, track_spans, gaps = [], [], []
    counts = Counter()
    actual = defaultdict(set)
    for partition in expected:
        for path in sorted((DATA / f'dataset_{partition}.pt.parts').glob('part_*.pt')):
            graphs = torch.load(path, map_location='cpu', weights_only=False)
            for g in graphs:
                evt = int(g.event_id.item())
                assert evt in expected[partition] and evt not in actual[partition]
                actual[partition].add(evt)
                x = g.x.numpy()
                # Geometry+d signature detects repeated measured events independently of time origin and labels.
                sig = fingerprint(x[:, :4])
                for old_partition, old_evt in seen[sig]:
                    if old_partition != partition:
                        duplicate_events.append([old_partition, old_evt, partition, evt])
                seen[sig].append((partition,evt))
                associations = truth_sets(g)
                grouped = defaultdict(list)
                for i, keys in enumerate(associations):
                    assert len(keys)==1
                    grouped[next(iter(keys))].append(i)
                for key, indices in grouped.items():
                    track_seen[fingerprint(x[indices,:4])].add(partition)
                if partition == 'validation':
                    times = x[:,6].astype(float)*10000
                    event_spans.append(np.ptp(times))
                    medians = []
                    for indices in grouped.values():
                        track_spans.append(np.ptp(times[indices]))
                        medians.append(np.median(times[indices]))
                    gaps.extend(np.diff(np.sort(medians)))
                counts[partition] += 1
            del graphs
        assert actual[partition] == expected[partition]
        print('scanned',partition,counts[partition],flush=True)
    result = {'counts': dict(counts), 'event_ids_disjoint_and_complete':True,
              'cross_split_exact_measured_event_duplicates':duplicate_events,
              'cross_split_exact_measured_track_signatures':sum(len(v)>1 for v in track_seen.values()),
              'duplicate_check_limit':'Exact sorted geometry+d signatures; not a proof against regenerated or near-duplicate trajectories.',
              'validation_event_span_ns':quantile(event_spans), 'validation_track_span_ns':quantile(track_spans),
              'validation_neighbor_track_median_gap_ns':quantile(gaps),
              'neighbor_median_gap_gt_3ns_fraction':float(np.mean(np.array(gaps)>3)),
              'neighbor_median_gap_gt_100ns_fraction':float(np.mean(np.array(gaps)>100))}
    save('graph_audit.json', result)
    print(json.dumps(result),flush=True)


def set_times(g, times):
    h = g.clone()
    times = np.asarray(times, dtype=float)
    h.x[:,6] = torch.tensor((times-times.min())/10000, dtype=h.x.dtype)
    row,col = h.edge_index
    dt = h.x[col,6]-h.x[row,6]
    h.edge_attr[:,4] = dt
    h.edge_attr[:,5] = dt.abs()
    assert torch.equal(h.x[:,:6],g.x[:,:6]) and torch.equal(h.edge_attr[:,:4],g.edge_attr[:,:4])
    return h


def edge_counts(g, prob):
    labels = g.y.numpy().astype(bool)
    c = Counter()
    for threshold in [.5,.7]:
        pred = prob>=threshold
        c[f'tp{threshold}'] = int((pred&labels).sum())
        c[f'fp{threshold}'] = int((pred&~labels).sum())
        c[f'fn{threshold}'] = int((~pred&labels).sum())
    return c


def probes():
    torch.set_num_threads(2)
    ids = sorted(read(ROOT / 'reports/time_audit/invariance.json')['pilot_event_ids'])
    assert set(ids)<=set(read(DATA/'event_split.json')['val_evt_ids'])
    graphs = load_graphs({evt:'validation' for evt in ids})
    weight = DATA/'best_interaction_network_time_expert_whole_events_sigma05.pth'
    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert sha=='354747c584cdba0aa5ab019044cdcc29cc0b23562b0c1673fbd12b97591b39c8'
    model = InteractionNetwork(*model_dimensions('time'))
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval()
    original = {evt:predict(graphs[evt],model,device,'time') for evt in ids}
    g = graphs[ids[0]]
    reference = predict(g,model,device,'time')
    assert np.allclose(reference,original[ids[0]],atol=1e-5)
    polluted = g.clone()
    polluted.y = 1-polluted.y
    polluted.truth_pairs = torch.full_like(polluted.truth_pairs, -999)
    polluted.event_id = torch.tensor([-999])
    # CUDA index_add may vary in its last bits; test label independence exactly on CPU.
    cpu_model=InteractionNetwork(*model_dimensions('time'))
    cpu_model.load_state_dict(model.cpu().state_dict());cpu_model.eval()
    cpu_reference=predict(g,cpu_model,torch.device('cpu'),'time')
    assert np.array_equal(cpu_reference,predict(polluted,cpu_model,torch.device('cpu'),'time'))
    model=model.to(device)
    rng=np.random.default_rng(20261015)
    order=rng.permutation(g.num_nodes); inverse=np.argsort(order)
    permuted=g.clone();permuted.x=g.x[order];permuted.edge_index=torch.tensor(inverse[g.edge_index.numpy()],dtype=torch.long)
    difference=float(np.max(np.abs(reference-predict(permuted,model,device,'time'))))
    assert difference<1e-4, difference
    save('invariance.json',{'weight_sha256':sha,'truth_label_and_event_id_change_prediction_max_difference':0,'node_reordering_max_prediction_difference':difference,'pilot_event_ids':ids})
    modes=['clean','shuffle_within_event','zero_time','collapse_truth_track_medians','dt_gate_200ns','corrected_dt_gate_50ns','coupled_distance_smear','independent_14ns','truth_birth_subtracted']
    results={}
    delta_same,delta_other,corrected_same,corrected_other=[],[],[],[]
    for mode in modes:
        path=OUT/f'{mode}.json'
        if path.exists():
            saved=read(path)
            assert saved['weight_sha256']==sha and saved['event_ids']==ids
            results[mode]=saved
            continue
        probabilities={};counter=Counter()
        for evt in ids:
            g=graphs[evt];t=g.x[:,6].numpy().astype(float)*10000
            if mode=='clean':
                probability=original[evt]
            elif mode=='dt_gate_200ns':
                probability=np.where(g.edge_attr[:,5].numpy()*10000<=200,.99,.01)
            elif mode=='corrected_dt_gate_50ns':
                corrected=t-g.drift_d.numpy()/.035
                row,col=g.edge_index.numpy()
                probability=np.where(np.abs(corrected[col]-corrected[row])<=50,.99,.01)
            else:
                if mode=='shuffle_within_event':
                    t=np.random.default_rng(np.random.SeedSequence([20261015,evt])).permutation(t)
                elif mode=='zero_time':t=np.zeros_like(t)
                elif mode=='coupled_distance_smear':
                    coupled=torch.load(OUT/'coupled_times.pt',weights_only=False)
                    t=coupled[evt]
                elif mode=='truth_birth_subtracted':
                    t=torch.load(OUT/'drift_only_times.pt',weights_only=False)[evt]
                elif mode=='independent_14ns':
                    t=t+np.random.default_rng(np.random.SeedSequence([20261015,evt])).normal(0,.5/.035,len(t))
                else:
                    grouped=defaultdict(list)
                    for i, keys in enumerate(truth_sets(g)):grouped[next(iter(keys))].append(i)
                    for indices in grouped.values():t[indices]-=np.median(t[indices])
                probability=predict(set_times(g,t),model,device,'time')
            probabilities[evt]=probability
            counter.update(edge_counts(g,probability))
            if mode=='clean':
                row,col=g.edge_index.numpy();labels=g.y.numpy().astype(bool);dt=np.abs(t[col]-t[row])
                corr=t-g.drift_d.numpy()/.035;dc=np.abs(corr[col]-corr[row])
                delta_same.extend(dt[labels]);delta_other.extend(dt[~labels]);corrected_same.extend(dc[labels]);corrected_other.extend(dc[~labels])
        if mode=='clean':
            save('edge_timing_separation.json',{k:quantile(v) for k,v in [('same_track_abs_dt_ns',delta_same),('different_track_abs_dt_ns',delta_other),('same_track_abs_corrected_dt_ns',corrected_same),('different_track_abs_corrected_dt_ns',corrected_other)]})
        with ProcessPoolExecutor(max_workers=6,initializer=init_worker) as pool:
            rows=list(pool.map(track,((evt,graphs[evt],probabilities[evt]) for evt in ids),chunksize=1))
        metrics=summary(rows)
        metrics['edge']={}
        for threshold in [.5,.7]:
            a,b,c=[counter[f'{k}{threshold}'] for k in ['tp','fp','fn']]
            metrics['edge'][str(threshold)]={'precision':a/max(a+b,1),'recall':a/max(a+c,1),'f1':2*a/max(2*a+b+c,1)}
        result={'mode':mode,'weight_sha256':sha,'event_ids':ids,'metrics':metrics,'events':rows,
                'caution':'Truth-track centering is an oracle diagnostic, unavailable online; other ablations are out-of-distribution tests, not retrained no-time models.'}
        save(f'{mode}.json',result);results[mode]=result
        print('PROBE',mode,json.dumps(metrics),flush=True)
    save('probe_summary.json',{k:v['metrics'] for k,v in results.items()})


def raw():
    # Inspect source identity globally; timing diagnostics use only the predefined validation pilot.
    ids=set(read(ROOT/'reports/time_audit/invariance.json')['pilot_event_ids'])
    columns=['EvtID','TrkID','CellID','Good','WireType','CreatProc','t','tHit','t0','d','d_true','NativeEvtID','SourceShard','PoolEvtID']
    selected=[];sources=[]
    for p in sorted((ROOT/'whole data').glob('CDCSimHit_*.csv')):
        f=pd.read_csv(p,usecols=columns)
        sources.append(f[['EvtID','NativeEvtID','SourceShard','PoolEvtID']].drop_duplicates())
        f=f.loc[f.EvtID.isin(ids)&f.Good.eq(True)&f.WireType.eq('Axial')&f.CreatProc.eq('ExtendedDecayWithSpin')]
        selected.append(f)
        print('raw',p.name,len(f),flush=True)
    f=pd.concat(selected,ignore_index=True);source=pd.concat(sources).drop_duplicates()
    from common import load_cell_map
    from STEP1_完整事件数据集 import event_graph
    geometry=load_cell_map()[['CellID','wire_x','wire_y','wire_r']]
    frames=f.merge(geometry,on='CellID',how='left',validate='many_to_one')
    coupled={int(evt):(group.t.to_numpy()+(group.d.to_numpy()-group.d_true.to_numpy())/.035) for evt,group in frames.groupby('EvtID',sort=True)}
    torch.save(coupled,OUT/'coupled_times.pt')
    drift_only={int(evt):(group.t.to_numpy()-group.t0.to_numpy()) for evt,group in frames.groupby('EvtID',sort=True)}
    torch.save(drift_only,OUT/'drift_only_times.pt')
    check_ids=sorted(ids)[:10]
    stored=load_graphs({evt:'validation' for evt in check_ids})
    for evt in check_ids:
        frame=frames.loc[frames.EvtID.eq(evt)].copy()
        built=event_graph(frame)
        assert torch.equal(built.x,stored[evt].x) and torch.equal(built.edge_attr,stored[evt].edge_attr)
        assert torch.equal(built.edge_index,stored[evt].edge_index) and torch.equal(built.y,stored[evt].y)
        frame['TrkID']=np.arange(len(frame))
        changed=event_graph(frame)
        assert torch.equal(built.x,changed.x) and torch.equal(built.edge_attr,changed.edge_attr) and torch.equal(built.edge_index,changed.edge_index)
        assert not torch.equal(built.y,changed.y)
    source_reused=source.groupby(['SourceShard','NativeEvtID']).EvtID.nunique()
    pool_reused=source.groupby('PoolEvtID').EvtID.nunique()
    error=f.t-f.tHit-f.d_true/.035
    groups=f.groupby(['EvtID','TrkID'])
    birth=groups.t0.first()
    spans=groups.t0.agg(lambda x:x.max()-x.min())
    nearest=[]
    for evt,values in birth.groupby(level=0):nearest.extend(np.diff(np.sort(values.to_numpy())))
    result={'pilot_events':len(ids),'rows':len(f),'raw_event_mapping_rows':len(source),
        'raw_rebuild_exact_match_events':check_ids,'truth_id_changes_leave_graph_inputs_unchanged':True,
        'reused_native_source_events':int((source_reused>1).sum()),'reused_pool_events':int((pool_reused>1).sum()),
        't_minus_tHit_minus_dtrue_over_v_ns':{'mean':float(error.mean()),'std':float(error.std()),'abs_quantiles':quantile(np.abs(error))},
        'within_track_birth_time_span_ns':quantile(spans),'neighbor_truth_birth_gap_ns':quantile(nearest)}
    save('raw_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['scan','probes','raw']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    {'scan':scan,'probes':probes,'raw':raw}[args.stage]()
