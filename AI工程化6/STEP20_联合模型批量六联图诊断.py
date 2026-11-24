"""Complete-event joint A/S diagnosis; resumable batch with six-panel exports."""
from pathlib import Path
import argparse,json,time,csv,sys,traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import STEP19_六联图可视化诊断 as v
import numpy as np
import torch
from STEP4_真边分数与连通性诊断 import topology

BATCH=None
def initialize(folder):
    global BATCH
    BATCH=Path(folder);v.OUT=BATCH;v.e.initialize_worker();v.review.initialize()
    v.gallery=lambda:None  # Write the 501-event gallery once in the parent, not once per event.
    v.plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];v.plt.rcParams['axes.unicode_minus']=False


def process(eid,partition):
    started=time.time();r=v.review
    with np.load(v.e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,d,truth,hit=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=np.load(v.e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    with np.load(BATCH/'prediction_cache'/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    axial,_=r.axial_two_pass(geo,d,pairs,p)
    attached,_=r.attach_stereo(axial,geo,d,cells,pairs,p,r.DEFAULT_CONFIG)
    joint=r.track(geo,d,cells,pairs,p,axial,attached,r._CALIBRATOR,.02,.2)
    coarse,diag=r.select(joint,r.propose(joint,geo,d,cells,pairs,p,refine=False),geo,cells,'guarded_loose')
    counts,found=r.score_tracks(joint,truth,stereo,r.DEFAULT_CONFIG)
    sizes=Counter(truth[~stereo]);eligible={int(t) for t,n in sizes.items() if t>=0 and n>=5}
    seed_found=set()
    seed_eligible={t for t in eligible if len(np.unique(cells[(truth==t)&stereo]))>=2}
    for t in coarse:
        if not t['coarse_3d']['seed_ready']:continue
        aa=np.array([i for i in t['indices'] if not stereo[i]],dtype=int);ss=np.array(t['confirmed_stereo_indices'],dtype=int)
        tids,nums=np.unique(truth[aa],return_counts=True);tt=int(tids[nums.argmax()]);good=ss[truth[ss]==tt]
        if tt in seed_eligible and nums.max()/len(aa)>=.8 and nums.max()/sizes[tt]>=.5 and len(np.unique(cells[good]))>=2 and len(good)/max(len(ss),1)>=.8:seed_found.add(tt)
    missed=eligible-set(found);seed_only=(seed_eligible-seed_found)&set(found)
    focus=max(missed or seed_only or eligible,key=lambda t:(sizes[t],-t)) if eligible else None
    if focus is None:raise ValueError('事件没有满足至少5个直丝hit的可评径迹')
    groups=[];seen=set()
    for n,(t,c) in enumerate(zip(joint,coarse),1):
        aa=[i for i in t['indices'] if not stereo[i]];tid,num=Counter(truth[aa]).most_common(1)[0]
        purity=num/len(aa);completeness=num/sizes[tid];valid=tid>=0 and sizes[tid]>=5 and purity>=.8 and completeness>=.5
        status='fake' if not valid else 'clone' if tid in seen else 'matched'
        if valid:seen.add(tid)
        fit=t['fit'];error=np.abs(np.linalg.norm(geo[aa,:2]-np.array(fit[:2]),axis=1)-fit[2])-d[aa]
        groups.append(dict(number=n,indices=aa,fit=fit,confirmed=c['confirmed_stereo_indices'],seed_ready=bool(c['coarse_3d']['seed_ready']),truth=int(tid),purity=purity,completeness=completeness,status=status,rms=float(np.sqrt(np.mean(error**2)))))
    assert seen==set(found)
    target=np.flatnonzero((truth==focus)&~stereo);mapping=np.full(len(truth),-1,dtype=int);mapping[target]=np.arange(len(target))
    true_aa=(mapping[pairs[0]]>=0)&(mapping[pairs[1]]>=0)
    category=topology(len(target),mapping[pairs[:,true_aa]],p[true_aa])['category'] if focus in missed else 'matched'
    v.LABELS['matched']='该真径迹已匹配（无漏轨时选最长轨）'
    v.catalog=lambda:[dict(event=eid,truth=focus,category=category)]
    v.MODELS['baseline']=f'直丝+斜丝联合 epoch126 / {partition}'
    weight=v.e.CONTRACT['weight_sha256']
    v.load=lambda _eid,_model:(cells,d,truth,hit,geo,stereo,pairs,p,groups,weight)
    image=v.draw(eid,focus,'baseline')
    stem=image.stem;packet=v.read(BATCH/(stem+'.json'))
    assert packet['missed_axial_truth_ids']==sorted(missed)
    assert packet['missing_3d_only_truth_ids']==sorted(seed_only)
    packet.update(partition=partition,joint_tracks=joint,coarse_tracks=coarse,joint_metrics=counts,coarse_metrics=r.seed_counts(coarse,truth,stereo,cells),coarse_diagnostics=diag)
    v.e.save_json(BATCH/(stem+'.json'),packet)
    owners={i:[] for i in range(len(cells))};confirmed={i:[] for i in range(len(cells))};pending={i:[] for i in range(len(cells))}
    for number,(t,c) in enumerate(zip(joint,coarse),1):
        for i in t['indices']:owners[i].append(number)
        for i in c['confirmed_stereo_indices']:confirmed[i].append(number)
        for i in c['pending_stereo_indices']:pending[i].append(number)
    with (BATCH/(stem+'_hits.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['node','HitID','CellID','TrkID','is_stereo','d_mm','reference_x','reference_y','joint_groups','confirmed_S_groups','pending_S_groups'])
        w.writerows([i,int(hit[i]),int(cells[i]),int(truth[i]),bool(stereo[i]),float(d[i]),*geo[i,:2],';'.join(map(str,owners[i])),';'.join(map(str,confirmed[i])),';'.join(map(str,pending[i]))] for i in range(len(cells)))
    kinds=stereo[pairs[0]].astype(int)+stereo[pairs[1]].astype(int)
    result=dict(event_id=eid,partition=partition,focus_truth=focus,category=category,image=image.name,hits=len(cells),axial_hits=int((~stereo).sum()),stereo_hits=int(stereo.sum()),edge_counts={k:int((kinds==i).sum()) for i,k in enumerate(['AA','AS','SS'])},metrics=counts,seed=packet['coarse_metrics'],seconds=time.time()-started)
    v.e.save_json(BATCH/f'completed_{eid:05d}.json',result);return result


@torch.no_grad()
def predict(ids,folder):
    r=v.review;metadata=v.read(v.e.DATA_ROOT/'dataset_metadata_joint_no_time.json')
    class Events(torch.utils.data.Dataset):
        def __init__(self):
            self.sources={s:r.JointEventDataset(v.e.DATA_ROOT,s) for s in metadata['event_ids']}
            self.lookup={eid:(s,j) for s,ds in self.sources.items() for j,eid in enumerate(ds.ids)}
        def __len__(self):return len(ids)
        def __getitem__(self,i):
            s,j=self.lookup[ids[i]];return self.sources[s][j]
    # workers=0: nested local dataset class stays in this process, bounded graph memory.
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    model=r.JointInteractionNetwork(chunk_size=65536).to(device)
    model.load_state_dict(torch.load(v.e.SNAPSHOT/'best_interaction_network_joint_no_time.pth',map_location=device,weights_only=True));model.eval()
    for j,g in enumerate(r.DevicePrefetch(r.loader(Events(),1,workers=0),device)):
        with torch.autocast(device.type,enabled=device.type=='cuda'):p=model(g.x,g.edge_index,g.edge_attr).sigmoid().float().cpu().numpy()
        pairs,p=r.undirected_predictions(g.edge_index.cpu().numpy(),p);eid=int(g.event_id.item())
        path=folder/f'event_{eid:05d}.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,pairs=pairs.astype(np.int32),probabilities=p)
        tmp.replace(path)
        if (j+1)%25==0:v.e.save_json(folder.parent/'progress.json',dict(phase='联合模型推理',events=j+1,total=len(ids),unix=time.time()));print(f'预测 {j+1}/{len(ids)}',flush=True)
    del model;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--event-range',nargs=2,type=int,required=True);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    start,end=args.event_range
    if not 0<=start<=end:raise ValueError('事件闭区间无效')
    folder=v.ROOT/'visual_diagnosis_20261121'/f'joint_events_{start}_{end}';folder.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(folder/'batch.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    v.e.verify_pins();metadata=v.read(v.e.DATA_ROOT/'dataset_metadata_joint_no_time.json');part={i:s for s,values in metadata['event_ids'].items() for i in values};ids=list(range(start,end+1))
    assert all(i in part for i in ids),'事件范围包含数据集中不存在的EvtID'
    plan=dict(event_ids=ids,closed_interval=True,model='AI6 joint axial+stereo epoch126',weight_sha256=v.e.CONTRACT['weight_sha256'],dataset_sha256=v.e.sha(v.e.DATA_ROOT/'dataset_metadata_joint_no_time.json'),split_counts=dict(Counter(part[i] for i in ids)),reconstruction='AA两级 + AS/SS联合hybrid + guarded_loose粗三维',truth_used='offline evaluation/visualization only',script_sha256=v.e.sha(Path(__file__)),visualizer_sha256=v.e.sha(v.ROOT/'STEP19_六联图可视化诊断.py'))
    if (folder/'contract.json').exists():assert v.read(folder/'contract.json')==plan
    v.e.save_json(folder/'contract.json',plan);cache=folder/'prediction_cache';cache.mkdir(exist_ok=True)
    todo=[i for i in ids if not (cache/f'event_{i:05d}.npz').exists()]
    if todo:predict(todo,cache)
    rows={i:v.read(folder/f'completed_{i:05d}.json') for i in ids if (folder/f'completed_{i:05d}.json').exists()};errors=[];started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(folder),)) as pool:
        futures={pool.submit(process,i,part[i]):i for i in ids if i not in rows}
        for future in as_completed(futures):
            eid=futures[future]
            try:rows[eid]=future.result()
            except Exception:errors.append(dict(event_id=eid,error=traceback.format_exc()));v.e.save_json(folder/'errors.json',errors)
            progress=dict(phase='联合寻迹与六联图',completed=len(rows),failed=len(errors),total=len(ids),seconds=time.time()-started,unix=time.time());v.e.save_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
    result=dict(status='completed' if not errors and len(rows)==len(ids) else 'partial_failure',contract=plan,completed=len(rows),failed=errors,metrics=v.review.summarize([rows[i]['metrics'] for i in sorted(rows)]),coarse_metrics=v.review.seed_summary([rows[i]['seed'] for i in sorted(rows)]))
    v.e.save_json(folder/'summary.json',result)
    with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['EvtID','split','focus_TrkID','A_hits','S_hits','AA_edges','AS_edges','SS_edges','matched','fake','clone','image'])
        for i in sorted(rows):
            x=rows[i];w.writerow([i,x['partition'],x['focus_truth'],x['axial_hits'],x['stereo_hits'],*[x['edge_counts'][k] for k in ['AA','AS','SS']],x['metrics']['matched_tracks'],x['metrics'].get('fake_candidates',0),x['metrics'].get('clone_candidates',0),x['image']])
    v.OUT=folder;v.gallery();print(json.dumps(dict(status=result['status'],completed=len(rows),output=str(folder))),flush=True)


if __name__=='__main__':main()
