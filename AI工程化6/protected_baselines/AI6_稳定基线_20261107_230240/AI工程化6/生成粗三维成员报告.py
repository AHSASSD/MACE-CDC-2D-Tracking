"""Audit rough 3D membership independently of precision-fit parameter accuracy."""
import argparse
from collections import Counter
import json
import numpy as np
from common import ROOT,MODEL_ROOT,DATA_ROOT,save_json,sha
from STEP5_两级寻迹 import score_tracks,summarize,DEFAULT_CONFIG
from STEP17_保护直丝的三维辅助寻迹 import stereo_score,stereo_summary


def seed_counts(tracks,truth,stereo,cells):
    eligible={int(t):int(((truth==t)&~stereo).sum()) for t in np.unique(truth) if t>=0 and ((truth==t)&~stereo).sum()>=5}
    target={t for t in eligible if len(np.unique(cells[(truth==t)&stereo]))>=2}
    counts=Counter(truth_tracks=len(target));found=set()
    for t in tracks:
        if not t['coarse_3d']['seed_ready']:continue
        counts['output_candidates']+=1
        aa=np.array([i for i in t['indices'] if not stereo[i]],dtype=int)
        ss=np.array(t['confirmed_stereo_indices'],dtype=int);ids,n=np.unique(truth[aa],return_counts=True);tid=int(ids[n.argmax()])
        valid_a=tid in eligible and n.max()/len(aa)>=.8 and n.max()/eligible[tid]>=.5
        good_s=ss[truth[ss]==tid]
        valid_s=tid in target and len(np.unique(cells[good_s]))>=2 and len(good_s)/max(len(ss),1)>=.8
        if not valid_a or not valid_s:counts['fake_candidates']+=1;continue
        if tid in found:counts['clone_candidates']+=1
        else:found.add(tid)
    counts['matched_tracks']=len(found)
    return dict(counts)


def same(a,b):
    assert set(a)==set(b),(set(a),set(b))
    for k in a:assert np.isclose(a[k],b[k],rtol=1e-12,atol=1e-9),(k,a[k],b[k])


def audit(snapshot,run_tag,phase,profile):
    folder=snapshot/(run_tag+'_'+phase);summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    plan=summary['contract'];geometry=np.load(DATA_ROOT/'wire_geometry.npy')
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(snapshot/'best_interaction_network_joint_no_time.pth')==plan['weight_sha256']
    a=[];b=[];s=[];seed=[];changed=Counter();a_hits=0
    for eid in plan['event_ids']:
        row=json.loads((folder/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
        before=json.loads((snapshot/('coarse_3d_inputs_'+plan['split'])/f'event_{eid:05d}.json').read_text(encoding='utf-8'))['tracks']
        after=row['candidates'][profile]
        with np.load(DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:cells,truth,hits=f['cell_id'],f['truth'],f['hit_id']
        stereo=np.linalg.norm(geometry[cells,2:4],axis=1)>1e-9
        assert len(before)==len(after)
        used=[]
        for old,new in zip(before,after):
            old_a=[i for i in old['indices'] if not stereo[i]];new_a=[i for i in new['indices'] if not stereo[i]]
            assert old_a==new_a;a_hits+=len(old_a);used.extend(new['indices'])
            assert hits[new['indices']].tolist()==new['hit_ids']
            verified=set(new['confirmed_stereo_indices']);pending=set(new['pending_stereo_indices'])
            assert not verified.intersection(pending)
            assert verified|pending=={i for i in new['indices'] if stereo[i]}
            ids,n=np.unique(truth[old_a],return_counts=True);tid=int(ids[n.argmax()]);size=int(((truth==tid)&~stereo).sum())
            valid=tid>=0 and size>=5 and n.max()/len(old_a)>=.8 and n.max()/size>=.5
            if valid:
                previous={i for i in old['indices'] if stereo[i] and truth[i]==tid}
                current={i for i in verified if truth[i]==tid}
                changed['new_correct_confirmed_s']+=len(current-previous)
                changed['previous_correct_s_not_confirmed']+=len(previous-current)
        assert len(used)==len(set(used))
        bc,_=score_tracks(before,truth,stereo,DEFAULT_CONFIG);ac,_=score_tracks(after,truth,stereo,DEFAULT_CONFIG)
        same(bc,row['baseline']);same(ac,row['methods'][profile]);sc=stereo_score(after,truth,stereo,cells);same(sc,row['stereo_metrics'][profile])
        a.append(bc);b.append(ac);s.append(sc);seed.append(seed_counts(after,truth,stereo,cells))
    all_before=summarize(a);all_after=summarize(b);ss=stereo_summary(s);ready=Counter()
    for row in seed:ready.update(row)
    ready=dict(ready)
    ready.update(efficiency=ready['matched_tracks']/max(ready['truth_tracks'],1),
                 fake_fraction=ready.get('fake_candidates',0)/max(ready['output_candidates'],1),
                 clone_fraction=ready.get('clone_candidates',0)/max(ready['output_candidates'],1))
    for k in ['truth_tracks','output_candidates','matched_tracks','fake_candidates','clone_candidates']:
        assert all_before.get(k,0)==all_after.get(k,0)
    return dict(events=len(a),axial_hits_before=a_hits,axial_hits_after=a_hits,lost_axial_hits=0,lost_axial_candidates=0,
                baseline=all_before,associated=all_after,stereo=ss,genfit_ready_membership=ready,changes=dict(changed),
                all_event_memberships_and_scores_audited=True,all_frozen_hashes_match=True,
                baseline_stereo=summary['all']['baseline_stereo'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',default='local_evaluation_epoch126_20261107')
    parser.add_argument('--run-tag',default='coarse_3d_v4_fast');args=parser.parse_args();snapshot=MODEL_ROOT/args.snapshot
    validation=json.loads((snapshot/(args.run_tag+'_validation')/'summary.json').read_text(encoding='utf-8'))
    profile=validation['selected_on_development']
    if not profile:raise ValueError('No qualified development profile')
    results={'validation':audit(snapshot,args.run_tag,'validation',profile)}
    if (snapshot/(args.run_tag+'_test')/'summary.json').exists():results['test']=audit(snapshot,args.run_tag,'test',profile)
    out=snapshot/(args.run_tag+'_report');out.mkdir(exist_ok=True);save_json(out/'audit_summary.json',dict(profile=profile,results=results))
    rows=[]
    for phase,result in results.items():
        baseline=result['baseline_stereo'];stereo=result['stereo'];ready=result['genfit_ready_membership']
        rows.append(f"| {phase} / {result['events']} | {result['baseline']['track_efficiency']:.4%} → {result['associated']['track_efficiency']:.4%} | {result['lost_axial_hits']} / {result['lost_axial_candidates']} | {baseline['matched_associated_precision']:.4%} | {stereo['matched_confirmed_precision']:.4%} | {stereo['correct_confirmed_stereo_hits']:,} | {ready['matched_tracks']:,}/{ready['truth_tracks']:,} = {ready['efficiency']:.4%} |")
    selected=validation['development']['methods'][profile]['stereo'];checked=validation['check']['methods'][profile]['stereo']
    test_note=''
    if 'test' in results:
        measured=results['test']['stereo']['matched_confirmed_precision']
        if measured<.995:test_note=f'冻结测试的确认纯度为{measured:.4%}，略低于开发目标99.5%；未据此重新选择方案或调参。'
    details=[]
    for phase,r in results.items():
        s=r['stereo'];base=r['baseline_stereo'];ready=r['genfit_ready_membership'];change=r['changes']
        details.append(f"### {phase}：{r['events']}个事件\n\n原联合寻迹正确关联斜丝{base['correct_associated_stereo_hits']:,}个；新方案进入几何确认集的正确斜丝{s['correct_confirmed_stereo_hits']:,}个。其中新找回{change['new_correct_confirmed_s']:,}个原先未关联的正确斜丝；原先已正确关联、但未进入确认集的有{change['previous_correct_s_not_confirmed']:,}个，部分仍保存在待定列表，不能把它们都描述为彻底删除。\n\n确认斜丝在有效直丝候选内纯度为{s['matched_confirmed_precision']:.4%}；把无效直丝候选上的斜丝也计为错误后，整体确认精度为{s['confirmed_precision']:.4%}。确认正确斜丝占全部合格真轨斜丝的{s['confirmed_recall']:.4%}。\n\n可供GenFit使用的粗种子输出{ready['output_candidates']:,}条，按“直丝80%/50%＋至少两个不同CellID的正确斜丝、斜丝纯度≥80%”匹配{ready['matched_tracks']:,}条；假种子{ready.get('fake_candidates',0):,}条（{ready['fake_fraction']:.4%}）、克隆{ready.get('clone_candidates',0):,}条（{ready['clone_fraction']:.4%}）。这是粗三维成员种子指标，未使用斜丝完整度≥50%的旧门槛。原门槛下关联成员联合达标率为{r['baseline']['joint_track_efficiency']:.4%}→{r['associated']['joint_track_efficiency']:.4%}。\n\n所有{r['axial_hits_before']:,}个原直丝成员保留，成员顺序和候选归属逐事件一致；尚未得到可靠纵向种子的候选也未删除。")
    plan=validation['contract']
    text=f'''# 保护直丝的粗三维斜丝成员寻迹与GenFit交接

目标是保留二维直丝成果，通过初步三维几何寻找可靠斜丝子集与纵向种子；完整物理拟合由GenFit完成。新模型仍固定为第126轮，不重训GNN，不改服务器训练。

## 固定方案实测

| 样本 | 原直丝真轨效率→处理后 | 直丝hit/候选损耗 | 原关联斜丝纯度（有效A） | 确认斜丝纯度（有效A） | 正确确认斜丝数 | 有效粗三维种子效率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

确认集可以舍弃部分斜丝；确认斜丝纯度与正确数量、纵向种子覆盖率必须一起看。上述高纯度按已有效匹配的直丝候选计算；所有输出候选口径另在下文列出，避免隐藏直丝假候选的影响。

{chr(10).join(details)}

## 如何吸收元胞自动机的几何逻辑

已读取用户提供的`Z.L. YU元胞自动机/positron_momentum_analysis_multitrack_2dbranch_plotstats.py`中的`fit_sz_with_geometry`，源码哈希`{plan['geometry_reference']['sha256']}`，原目录未修改。

固定横向圆心和半径后，斜丝参数化为`wire_center + t * direction`；求它与R、R−d、R+d圆柱面的交点。t是沿丝距离，必须使用实际丝长检查端点；z=t*w，不能混用z与半丝长。R±d用于漂移侧分支，名义R没有交点时不能直接把一个非零漂移距离的hit判错。交点只是几何候选，不是真实三维击中位置。

将交点转换为圆周弧长对应的相位与z，两条独立丝的交点提出纵向直线假设，其他斜丝作一致性检验。采用有限直丝z范围检查种子、弧长80mm余量、独立丝支持；保留多个相近质量解。最终用螺旋到真实有限丝段的距离减去测量d做毫米级验收，避免把横向漂移误差直接当作同宽度的z误差。

原Yu模块的至少六层、双旋向、固定z窗口不直接照搬；我们记录层数、旋向和相位覆盖作为质量信息，粗种子只要求独立支持。独立CellID仅用于避免重复投票，输出仍保留每个独立HitID，重复击中同一丝不会合并。

## 直丝保护和斜丝处理

1. 固定原联合寻迹的直丝成员和二维圆，不用三维残差删直丝或整条候选。
2. 结合AS概率、有限丝段与圆柱关系提出斜丝池，两点提出初值，多点检验一致性；粗参数采用分支选择和稳健线性更新。当前部署配置不逐分支做非线性精修，精确拟合留给GenFit。
3. 多个候选竞争同一斜丝HitID，保证最终归属排他。可靠斜丝进入confirmed，仍有关联但未获几何确认的进入pending，剔除者另存身份。三维退化时保留直丝和待定斜丝。
4. GenFit输入列表为“全部受保护直丝＋几何确认斜丝”；pending独立提供，不自动塞入精确拟合。至少两个不同CellID的确认斜丝且存在纵向假设时标记seed_ready。

## 数据分工与工作点

前100个固定验证事件选方案，优先要求有效直丝候选内确认斜丝纯度≥99.5%，再最大化有两个正确确认斜丝的真轨数，随后最大化正确确认斜丝数量。选中`{profile}`，开发纯度{selected['matched_confirmed_precision']:.4%}；后100个验证事件复核纯度{checked['matched_confirmed_precision']:.4%}，通过状态为{validation['check_passed']}。

{test_note}

若报告包含test，其300个事件按固定种子20261109从既有1500测试事件中预先抽取，使用验证阶段冻结方案，不据测试结果调参；这是已暴露固定测试划分的复测，不是新建未暴露留出集，也不是全部1500事件的粗三维评估。GNN此前仍只按验证损失选第126轮。原候选校准器来自第95轮的专用验证校准子集，沿用不重训。

所有输入与重建仅使用丝几何、d、CellID/HitID和GNN概率，不使用时间、真实z、TrkID或动量。TrkID只在选方案和评价时生成标签。权重、数据、源码与参考文件哈希、逐事件成员及独立评分检查均保存在对应目录。

## GenFit交接与限制

入口`STEP18_粗三维寻迹与GenFit导出.py`输出每个独立hit的有限丝段端点和漂移距离、受保护直丝、确认/待定/剔除斜丝、二维圆、多个纵向初值、初始位置及两个可能方向。单位mm/rad。电荷、磁场、动量大小和协方差留给GenFit，不虚构这些精确拟合输入。

```powershell
python STEP18_粗三维寻迹与GenFit导出.py --snapshot {args.snapshot} --run-tag {args.run_tag} --event 4779 --output genfit_event04779.json
```

这里的“三维效率”只评价可用的粗三维成员种子；尚未按真实z、斜率或动量误差验证参数精度。单圆/单段螺旋近似、未知多圈展开和多解仍会留下待定候选。两个正确斜丝可产生纵向初值，但左右漂移分支与退化仍需其余点的支持。不能将几何通过率或二维直丝保留率当成完整GenFit重建成功率。
'''
    (out/'粗三维斜丝选点与GenFit交接.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(profile=profile,results=results),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
