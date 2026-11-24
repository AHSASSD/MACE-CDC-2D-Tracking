"""Validation-only selection of observation-based track repair profiles."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch_geometric.data import Data
from common import ROOT, MODEL_ROOT, DATA_ROOT, sha, save_json
from two_stage_core import fit_2d_circle, run_two_pass
from STEP5_两级寻迹 import attach_stereo, DEFAULT_CONFIG, score_tracks, summarize


def fit_members(indices, geometry, drift, rms_limit):
    indices=sorted(set(indices))
    if len(indices)<5:return None
    xy=geometry[indices,:2];d=drift[indices]
    fitted=fit_2d_circle(xy[:,0],xy[:,1],d)
    if fitted is None or not np.isfinite(fitted).all():return None
    residual=np.abs(np.linalg.norm(xy-np.asarray(fitted[:2]),axis=1)-fitted[2])-d
    rms=float(np.sqrt(np.mean(residual**2)))
    return (list(fitted),rms) if rms<=rms_limit else None


def repair_tracks(baseline,geometry,drift,cells,pairs,p,recover=False):
    stereo=np.linalg.norm(geometry[:,2:4],axis=1)>1e-9;n=len(cells)
    matrix=np.zeros((n,n),dtype=float);matrix[pairs[0],pairs[1]]=p;matrix[pairs[1],pairs[0]]=p
    tracks=[dict(t,indices=list(t['indices'])) for t in baseline]
    # Merge only independently fitted axial fragments with reciprocal GNN support.
    changed=True
    while changed:
        changed=False
        for a in range(len(tracks)):
            for b in range(a+1,len(tracks)):
                left=np.array(tracks[a]['indices']);right=np.array(tracks[b]['indices'])
                block=matrix[np.ix_(left,right)];good=block>=.80
                if len(np.unique(cells[left[np.any(good,axis=1)]]))<2 or len(np.unique(cells[right[np.any(good,axis=0)]]))<2:continue
                if np.sort(block.ravel())[-min(4,block.size):].mean()<.85:continue
                proposed=sorted(set(left.tolist()+right.tolist()))
                limit=min(1.,max(tracks[a]['rms'],tracks[b]['rms'])+.10)
                fit=fit_members(proposed,geometry,drift,limit)
                if fit is None:continue
                tracks[a]=dict(tracks[a],indices=proposed,fit=fit[0],rms=fit[1],stage='fragment_merge')
                tracks.pop(b);changed=True;break
            if changed:break
    used={hit for t in tracks for hit in t['indices']}
    available=np.array([i for i in range(n) if not stereo[i] and i not in used],dtype=int)
    proposals={}
    for hit in available:
        scores=[]
        for j,t in enumerate(tracks):
            aa=np.array(t['indices']);values=matrix[hit,aa]
            if len(np.unique(cells[aa[values>=.65]]))<2:continue
            x,y,r=t['fit'];res=abs(abs(np.hypot(geometry[hit,0]-x,geometry[hit,1]-y)-r)-drift[hit])
            if res>1.:continue
            score=float(np.sort(values)[-min(3,len(values)):].mean())-.1*res
            if score>=.65:scores.append((score,j))
        scores.sort(reverse=True)
        if scores and (len(scores)==1 or scores[0][0]-scores[1][0]>=.10):proposals.setdefault(scores[0][1],[]).append(int(hit))
    for j,extra in proposals.items():
        t=tracks[j];proposed=t['indices']+extra
        fit=fit_members(proposed,geometry,drift,min(1.5,max(1.,t['rms']+.10)))
        if fit:
            tracks[j]=dict(t,indices=sorted(proposed),fit=fit[0],rms=fit[1],stage='hit_extension')
    if recover:
        used={hit for t in tracks for hit in t['indices']}
        aa=np.array([i for i in range(n) if not stereo[i] and i not in used],dtype=int)
        if len(aa)>=5:
            mapping=np.full(n,-1,dtype=int);mapping[aa]=np.arange(len(aa))
            keep=(mapping[pairs[0]]>=0)&(mapping[pairs[1]]>=0)
            data=Data(wire_x=torch.tensor(geometry[aa,0]),wire_y=torch.tensor(geometry[aa,1]),
                      drift_d=torch.tensor(drift[aa]),edge_index=torch.tensor(mapping[pairs[:,keep]]),num_nodes=len(aa))
            candidates,_=run_two_pass(data,None,None,None,.80,.55,probabilities=p[keep])
            for t in candidates:
                if t['mean_probability']<.65:continue
                t['indices']=aa[t['indices']].tolist();t['stage']='remaining_recovery';tracks.append(t)
    members=[i for t in tracks for i in t['indices']]
    assert len(members)==len(set(members))
    return tracks


def event_trial(eid,snapshot):
    snapshot=Path(snapshot);reference=snapshot/'two_stage_validation_v1'
    old=json.loads((reference/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    weight=json.loads((reference/'experiment_contract.json').read_text(encoding='utf-8'))['weight_sha256']
    with np.load(snapshot/('validation_prediction_cache_'+weight[:12])/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,drift,truth=f['cell_id'],f['d'],f['truth']
    geo=np.load(DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    rows={'baseline':old['methods']['stereo_assisted']};outputs={}
    for name,recover in [('repair',False),('repair_recover',True)]:
        aa=repair_tracks(old['tracks']['axial_two_pass'],geo,drift,cells,pairs,p,recover)
        tracks,_=attach_stereo(aa,geo,drift,cells,pairs,p,DEFAULT_CONFIG)
        rows[name],_=score_tracks(tracks,truth,stereo,DEFAULT_CONFIG);outputs[name]=tracks
    return dict(event_id=eid,methods=rows,tracks=outputs)


def select_profile(dev):
    baseline=dev['baseline']
    # Hard constraints prevent silently trading extra fakes/clones for recall.
    eligible=[name for name,m in dev.items() if m['fake_fraction']<=baseline['fake_fraction']+1e-12
              and m['clone_fraction']<=baseline['clone_fraction']+1e-12
              and m['track_efficiency']>=baseline['track_efficiency']-1e-12]
    return max(eligible,key=lambda name:(dev[name]['track_efficiency'],dev[name]['joint_track_efficiency'],-dev[name]['fake_fraction']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch051_20261106')
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;reference=snapshot/'two_stage_validation_v1'
    contract=json.loads((reference/'experiment_contract.json').read_text(encoding='utf-8'))
    if sha(snapshot/'best_interaction_network_joint_no_time.pth')!=contract['weight_sha256']:raise ValueError('Snapshot changed')
    ids=contract['event_ids'];out=snapshot/'tracking_repair_validation_v1';out.mkdir(exist_ok=True)
    source_hash=sha(Path(__file__));plan=dict(event_ids=ids,weight_sha256=contract['weight_sha256'],source_sha256=source_hash)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=plan:raise ValueError('Changed repair contract')
    save_json(out/'contract.json',plan);event_dir=out/'events';event_dir.mkdir(exist_ok=True);rows={};started=time.time()
    for eid in ids:
        if (event_dir/f'event_{eid:05d}.json').exists():rows[eid]=json.loads((event_dir/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(event_trial,eid,str(snapshot)) for eid in ids if eid not in rows]
        for f in as_completed(futures):
            row=f.result();rows[row['event_id']]=row;save_json(event_dir/f"event_{row['event_id']:05d}.json",row)
            if len(rows)%25==0:print(json.dumps(dict(events=len(rows),total=len(ids),seconds=time.time()-started)),flush=True)
    cut=len(ids)//2
    def aggregate(selected):return {name:summarize([rows[eid]['methods'][name] for eid in selected]) for name in ['baseline','repair','repair_recover']}
    dev=aggregate(ids[:cut]);check=aggregate(ids[cut:]);selected=select_profile(dev)
    save_json(out/'summary.json',dict(plan=plan,dev=dev,validation_check=check,selected_on_dev=selected,
              scope='first half validation for selection; second half validation for check, no test selection',seconds=time.time()-started))
    table='\n'.join(f"| {phase} | {name} | {m['track_efficiency']:.2%} | {m['fake_fraction']:.2%} | {m['clone_fraction']:.2%} | {m['joint_track_efficiency']:.2%} |" for phase,group in [('选择子集',dev),('复核子集',check)] for name,m in group.items())
    text=f'''# 寻迹碎片合并与遗漏hit回收

固定第{contract['selected_epoch']}轮模型、固定{len(ids)}个验证事件。前{cut}个选择方案，其余复核。仅使用hit观测、GNN概率与漂移圆几何，不用真值生成或筛选候选。

| 子集 | 方案 | 真轨匹配效率 | 假候选比例 | 克隆比例 | 直丝斜丝同时达标 |
|---|---|---:|---:|---:|---:|
{table}

repair要求片段之间有双向多hit高概率证据且共同拟合不恶化，再竞争回收有至少两个不同丝单元支持、残差<=1mm的遗漏直丝hit。repair_recover再在未用hit上做0.80/0.55的两级回收。斜丝归属沿用冻结方案。选择约束：相对baseline效率不降、假候选比例不升、克隆比例不升；在可行者中优先提高真轨和联合成员效率。

选择子集选出{selected}。若复核子集不支持同样取舍，不自动替换baseline。它们都是验证集诊断，不能当作独立测试保证。网络权重和服务器训练未修改。
'''
    (out/'寻迹改进验证报告.md').write_text(text,encoding='utf-8');print(json.dumps(dict(selected=selected,dev=dev,check=check)),flush=True)


if __name__=='__main__':main()
