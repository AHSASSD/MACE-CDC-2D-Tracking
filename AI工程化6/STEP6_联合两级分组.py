"""Joint hit membership without circle or helix fitting; validation experiment."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import ROOT, MODEL_ROOT, DATA_ROOT, sha, save_json
from STEP5_两级寻迹 import DEFAULT_CONFIG, score_tracks, summarize


def membership_groups(cells, stereo, pairs, probabilities, strict=.85, relaxed=.65, margin=.10):
    """Complete-link cores, reciprocal fragment support, exclusive hit expansion.

    No labels, time, fitted parameters or simulated positions are accepted.
    All AA/AS/SS probabilities come from the same full-event joint GNN.
    """
    n=len(cells)
    if n<2:return [], dict(cores=0, merged_fragments=0, attached_hits=0, unassigned=n)
    matrix=np.zeros((n,n),dtype=float)
    matrix[pairs[0],pairs[1]]=probabilities;matrix[pairs[1],pairs[0]]=probabilities
    np.fill_diagonal(matrix,1.)
    # Complete linkage requires every core pair to support membership. A single
    # false bridge cannot join two otherwise unrelated cores.
    tree=linkage(squareform(1-matrix,checks=False),method='complete')
    labels=fcluster(tree,1-strict,criterion='distance')
    cores=[]
    for label in np.unique(labels):
        group=np.flatnonzero(labels==label)
        na=int((~stereo[group]).sum());ns=int(stereo[group].sum())
        if na>=3 or (na>=2 and ns>=2):cores.append(group.tolist())
    initial=len(cores);merges=0
    # Only reciprocal support across at least two different cells per side may
    # merge fragments. Recheck after each merge instead of transitive flooding.
    for _ in range(2):
        merged=False
        for a in range(len(cores)):
            for b in range(a+1,len(cores)):
                left=np.array(cores[a]);right=np.array(cores[b]);block=matrix[np.ix_(left,right)]
                support=block>=strict
                left_good=np.any(support,axis=1);right_good=np.any(support,axis=0)
                if len(np.unique(cells[left[left_good]]))<2 or len(np.unique(cells[right[right_good]]))<2:continue
                observed=block[block>0]
                if len(observed)<4 or np.median(observed)<strict:continue
                if left_good.mean()<.5 or right_good.mean()<.5:continue
                cores[a].extend(cores[b]);cores.pop(b);merges+=1;merged=True;break
            if merged:break
        if not merged:break
    owners=np.full(n,-1,dtype=int)
    for i,group in enumerate(cores):owners[group]=i
    attached=0
    if cores:
        scores=np.zeros((n,len(cores)))
        for i,group in enumerate(cores):
            values=matrix[:,group];top=np.sort(values,axis=1)[:,-min(3,len(group)):].mean(1)
            witnesses=np.array([len(np.unique(cells[np.array(group)[row>=relaxed]])) for row in values])
            scores[:,i]=np.where(witnesses>=2,top,0)
        best=scores.argmax(1);top=scores[np.arange(n),best]
        second=np.partition(scores,-2,axis=1)[:,-2] if len(cores)>1 else np.zeros(n)
        select=np.flatnonzero((owners<0)&(top>=relaxed)&(top-second>=margin))
        # Frozen core evidence: new assignments do not vote in this pass.
        for hit in select:cores[int(best[hit])].append(int(hit));owners[hit]=best[hit]
        attached=len(select)
    tracks=[dict(indices=sorted(group),stage='joint_membership') for group in cores if (~stereo[group]).sum()>=5]
    return tracks,dict(cores=initial,merged_fragments=merges,attached_hits=attached,
                       unassigned=n-sum(len(t['indices']) for t in tracks))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch051_20261106')
    parser.add_argument('--reference',default='two_stage_validation_v1');args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot
    reference=snapshot/args.reference
    contract=json.loads((reference/'experiment_contract.json').read_text(encoding='utf-8'))
    weight=sha(snapshot/'best_interaction_network_joint_no_time.pth')
    if weight!=contract['weight_sha256']:raise ValueError('Snapshot changed')
    cache=snapshot/('validation_prediction_cache_'+weight[:12])
    out=snapshot/'joint_membership_validation_v1';out.mkdir(exist_ok=False)
    variants={'joint_085_065':(.85,.65),'joint_080_060':(.80,.60),'joint_090_070':(.90,.70)}
    totals={name:[] for name in variants};baseline={name:[] for name in ['simple_joint','axial_two_pass','stereo_assisted']}
    config=dict(DEFAULT_CONFIG);geometry=np.load(DATA_ROOT/'wire_geometry.npy');started=time.time()
    with (out/'events.jsonl').open('w',encoding='utf-8') as log:
        for j,eid in enumerate(contract['event_ids']):
            old=json.loads((reference/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
            for name in baseline:baseline[name].append(old['methods'][name])
            with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:
                cells,truth,hit_ids=f['cell_id'],f['truth'],f['hit_id']
            stereo=np.linalg.norm(geometry[cells,2:4],axis=1)>1e-9
            with np.load(cache/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
            rows={}
            for name,(strict,relaxed) in variants.items():
                tracks,diag=membership_groups(cells,stereo,pairs,p,strict,relaxed)
                counts,matched=score_tracks(tracks,truth,stereo,config);totals[name].append(counts)
                indices=[i for t in tracks for i in t['indices']]
                assert len(indices)==len(set(indices))
                rows[name]=dict(counts=counts,matched=matched,diagnostics=diag,
                                candidates=[dict(hit_ids=hit_ids[t['indices']].tolist(),cell_ids=cells[t['indices']].tolist()) for t in tracks])
            log.write(json.dumps(dict(event_id=eid,variants=rows))+'\n')
            if (j+1)%25==0:print(json.dumps(dict(events=j+1,total=len(contract['event_ids']),seconds=time.time()-started)),flush=True)
    methods={name:summarize(rows) for name,rows in {**baseline,**totals}.items()}
    # Validation selection is explicit; these are not untouched test results.
    def utility(name):
        m=methods[name];return (m.get('joint_matched_tracks',0)-.25*m.get('fake_candidates',0)-.25*m.get('clone_candidates',0))/max(m['truth_tracks'],1)
    selected=max(variants,key=utility)
    save_json(out/'summary.json',dict(events=len(contract['event_ids']),event_ids=contract['event_ids'],weight_sha256=weight,
              methods=methods,variants=variants,selected_validation_variant=selected,
              selection_objective='(joint matched - .25 fake - .25 clone) / axial eligible truth',
              source_sha256=sha(Path(__file__)),seconds=time.time()-started,
              scope='joint membership only, no 2D or 3D fitting; reference axial results reuse previous fits'))
    labels={'simple_joint':'简单连通分量','axial_two_pass':'原AA两级（含二维拟合）','stereo_assisted':'AA两级＋斜丝归属'}
    rows='\n'.join(f"| {labels.get(name,name)} | {m['track_efficiency']:.2%} | {m['fake_fraction']:.2%} | {m['clone_fraction']:.2%} | {m['joint_track_efficiency']:.2%} | {m['stereo_recall_all_eligible_tracks']:.2%} |" for name,m in methods.items())
    text=f'''# 联合两级成员分组实验（暂缓三维拟合）

目标是找到属于同一径迹的独立HitID集合。新分组流程不调用圆或螺旋拟合，不输出三维参数或动量；输入空间信息来自联合模型已有的有限丝几何与倾角特征。

同一第51轮权重、同一{len(contract['event_ids'])}个验证事件、同一双向均值预测缓存。合格真轨>=5直丝hit，匹配要求直丝纯度>=80%、完整度>=50%；联合达标另要求真轨>=3斜丝hit及斜丝纯度>=80%、完整度>=50%。

| 方案 | 直丝匹配效率 | 假候选比例 | 克隆比例 | 联合成员达标比例 | 全部合格真轨斜丝找回比例 |
|---|---:|---:|---:|---:|---:|
{rows}

新流程第一级用AA、AS、SS完整链接聚类生成高置信核心，防止单条错误桥边直接合并核心；第二级按双向多hit证据合并碎片，再按冻结核心证据竞争分配剩余hit，领先差至少0.10。每个hit最多归属一条径迹；重复CellID不合并成一个hit，但至少两个不同CellID才形成补点证据。未达标的小组保留为未分配。

三组工作点为预先声明的首级/次级阈值0.85/0.65、0.80/0.60、0.90/0.70；验证目标为(联合达标真轨-0.25假候选-0.25克隆)/合格直丝真轨，当前选择{selected}。这些阈值已使用验证结果选择，不能称为独立测试结论。

本次新分组及评分耗时{time.time()-started:.1f}秒，复用已缓存的GNN预测。原AA两级的对照来自此前同事件结果，其二维拟合开销未计入本次耗时。联合成员达标不等于三维参数拟合成功。

events.jsonl保存各方案逐事件评分及独立HitID/CellID候选；summary.json保存阈值、事件和权重哈希。三维拟合草稿_暂未启用.py已按用户要求搁置，未接入任何运行入口。
'''
    (out/'联合两级成员分组评估.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(selected=selected,methods=methods)),flush=True)


if __name__=='__main__':main()
