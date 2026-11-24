"""Report score separation and extractor interactions, with coverage limits."""
import experiment_common as e
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from STEP5_两级寻迹 import summarize
from STEP4_真边分数与连通性诊断 import BINS,COHORTS,distribution

LABELS={'candidate_graph_insufficient':'纯候选图不足','pure_connected_at_existing_relaxed_cut':'宽松阈值下纯真图已足够连通',
        'broadly_low_scores':'真边普遍低分','few_weak_connectors':'少数弱连接','mixed_fragmentation':'混合碎裂'}


def auc(pos,neg):
    return float(pos@(np.cumsum(neg)-.5*neg)/max(float(pos.sum())*float(neg.sum()),1.))


def main():
    out=e.EXP/'edge_score_diagnosis_v1';r=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads((out/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8')) for eid in e.IDS]
    combined=sum(np.load(out/'events'/f'hist_{eid:05d}.npy') for eid in e.IDS)
    held=sum(np.load(out/'events'/f'hist_{eid:05d}.npy') for eid in e.IDS[100:])
    track_rows=[dict(t,event_id=row['event_id']) for row in rows for t in row['tracks']]
    with (out/'逐径迹分数与连通性.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(track_rows[0]));writer.writeheader();writer.writerows(track_rows)
    global_points={}
    for kind,k in [('AA',0),('AS',1),('SS',2)]:
        points=[]
        for point in r['held_operating_points'][kind]['all']:
            index=int(round(point['threshold']*BINS));record=dict(point)
            for name,pos,neg in [('found',2,4),('missed',3,5)]:
                record[name+'_true_recall']=float(held[k,pos,index:].sum()/max(held[k,pos].sum(),1))
                record[name+'_false_acceptance']=float(held[k,neg,index:].sum()/max(held[k,neg].sum(),1))
            points.append(record)
        global_points[kind]=points
    rank={kind:{'found_auc':auc(combined[k,2],combined[k,4]),'missed_auc':auc(combined[k,3],combined[k,5])} for k,kind in enumerate(['AA','AS','SS'])}
    actual_single=summarize([row['real_single_pass_counts'] for row in rows])
    validation_topology=dict(eligible_tracks=len(track_rows),pure_match_scale_available=sum(t['largest_relaxed_component']>=t['pure_match_hits_required'] for t in track_rows),
                             median_track_true_edge_score=float(np.median([t['median_true_score'] for t in track_rows if t['median_true_score'] is not None])))
    training_path=e.EXP/'training_score_probe_v1/summary.json';training=json.loads(training_path.read_text(encoding='utf-8')) if training_path.exists() else None
    train_text='训练集抽查尚未完成。'
    if training:
        train_hist=np.load(e.EXP/'training_score_probe_v1/histogram.npy')
        ts=distribution(train_hist[0,0]);vs=distribution(combined[0,0])
        train_text=f"固定随机抽取200个训练事件，仅前向推理、不优化参数。训练样本{training['eligible_tracks']}条合格真轨，纯真AA图达到匹配规模的比例为{training['pure_match_scale_available']/training['eligible_tracks']:.4%}；验证200事件相同口径为{validation_topology['pure_match_scale_available']/validation_topology['eligible_tracks']:.4%}。训练/验证AA真边平均分数为{ts['mean']:.4f}/{vs['mean']:.4f}，每径迹真边中位分数的中位数为{training['median_track_true_edge_score']:.4f}/{validation_topology['median_track_true_edge_score']:.4f}。这是抽样对比，不能单凭差异断定过拟合或数据缺失。"
    tables=[]
    for kind in ['AA','AS','SS']:
        found=r['score_distributions'][kind]['found_true'];missed=r['score_distributions'][kind]['missed_true']
        tables.append(f"| {kind} | {found['median']:.4f} | {missed['median']:.4f} | {found['fraction_above_0575']:.2%} | {missed['fraction_above_0575']:.2%} | {rank[kind]['found_auc']:.4f} / {rank[kind]['missed_auc']:.4f} |")
    budget_rows=[]
    for point in global_points['AA']:
        budget_rows.append(f"| {point['negative_edge_budget']:.0%} | {point['threshold']:.4f} | {point['held_false_edge_acceptance']:.2%} | {point['found_true_recall']:.2%} | {point['missed_true_recall']:.2%} | {point['missed_false_acceptance']:.2%} |")
    categories='\n'.join(f"| {LABELS[k]} | {n} | {n/223:.2%} |" for k,n in sorted(r['missed_categories'].items(),key=lambda kv:-kv[1]))
    fig,axes=plt.subplots(1,3,figsize=(14,4.4));centers=(np.arange(64)+.5)/64
    for j,label in [(2,'True edges: found tracks'),(3,'True edges: missed tracks'),(5,'False edges near missed tracks')]:
        counts=combined[0,j].reshape(64,-1).sum(1);axes[0].plot(centers,counts/max(counts.sum(),1),label=label)
    axes[0].axvline(.575,color='black',linestyle='--',alpha=.5);axes[0].set(title='AA score distributions',xlabel='GNN score',ylabel='Fraction within cohort');axes[0].legend(fontsize=7)
    names=['Connected at .575','Broadly low','Mixed','Few connectors','Graph insufficient'];keys=['pure_connected_at_existing_relaxed_cut','broadly_low_scores','mixed_fragmentation','few_weak_connectors','candidate_graph_insufficient']
    bars=axes[1].barh(names,[r['missed_categories'].get(k,0) for k in keys],color='#64748b');axes[1].bar_label(bars,padding=3);axes[1].set(title='223 missed tracks',xlabel='Track count');axes[1].invert_yaxis()
    for key,label in [('found','Found-track cohort'),('missed','Missed-track cohort')]:
        x=[];y=[]
        pos,neg=(2,4) if key=='found' else (3,5)
        for cut in np.linspace(0,1,200):
            index=int(cut*BINS);x.append(held[0,neg,index:].sum()/max(held[0,neg].sum(),1));y.append(held[0,pos,index:].sum()/max(held[0,pos].sum(),1))
        axes[2].plot(x,y,label=label)
    axes[2].set(xlim=(0,.1),ylim=(0,1.02),xlabel='False-edge acceptance',ylabel='True-edge recall',title='Held 100 events: AA ROC');axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'edge_score_diagnosis.png',dpi=180);plt.close(fig)
    text=f'''# 真边分数、连通性与提取器配合：进一步诊断

本轮没有训练第二个提取器网络，也没有重训GNN。固定第126轮模型和稳定备份，检查200个既有验证事件的缓存分数；另抽查200个训练事件，仅做前向推理。标签只用于离线诊断，原稳定代码、权重和阈值不改。

## 主要发现

| 原基线223条漏轨的纯真AA图特征 | 数量 | 占比 |
|---|---:|---:|
{categories}

先判断纯图是否已达到匹配规模；未达到者再细分。“少数弱连接”定义为：低于0.575的真边不足半数，且按分数降序的最大生成森林过程补至匹配规模最多需两条边。“普遍低分”定义为至少80%的候选真AA边低于0.575；其余断连列为混合碎裂。已足够连通仅是必要条件，不保证几何验收、竞争和80%/50%成员匹配成功。纯图匹配规模取至少5个直丝且覆盖真轨一半，不能当作允许少量错点时的严格性能上限。

在181条补强真边后可恢复的漏轨中，普遍低分有{r['boost_sensitive_categories'].get('broadly_low_scores',0)}条，已有足够纯真连通分量有{r['boost_sensitive_categories'].get('pure_connected_at_existing_relaxed_cut',0)}条；仅少数弱连接的有{r['boost_sensitive_categories'].get('few_weak_connectors',0)}条。因此，当前证据不支持只给少量“桥接边”加权就作为主要训练改进。

## 分数区分能力

| 边类型 | 已找到真轨的真边中位分数 | 漏轨真边中位分数 | 已找到真边≥0.575 | 漏轨真边≥0.575 | 近邻区分AUC：已找到 / 漏轨 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

边级统计按边计数加权，AUC使用4096个分数桶近似，不能代替径迹效率。近邻假边指候选图中不同真轨、且至少一端属于对应轨迹组的边；found/missed两组假边可能重叠，不相加。分数桶宽约0.000244。

## 相同工作点下的误接预算检查

用前100事件的全部AA假边选择阈值，然后固定阈值检查后100事件；表中同时列出实际误接率，不能把设定预算当作复核中已保证达到的预算。

| 前100事件假边接收预算 | 固定阈值 | 后100事件全部AA假边接收率 | 已找到组真边召回 | 漏轨组真边召回 | 漏轨附近假边接收率 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(budget_rows)}

这里的“假边接收率”不是“假轨率”。阈值校准不改变类型内部的分数排序；如果漏轨附近真、假边有明显重叠，仅降低阈值仍会增加混轨风险。

## 严格阶段锁定碎片的受控诊断

当前严格阶段先接受候选并占用hit，宽松阶段只读剩余hit。合成测试构造了同一真轨的三个5-hit片段：片段内部得分0.8、片段间0.6，二维几何完全正确。原两级提取没有匹配完整径迹；保持所有边分数不变、宽松阶段使用完整图，则能找回该径迹。

真实验证样本中，在“屏蔽所有错边、保留真边原分数”的诊断条件下，去掉严格阶段提前锁定、改为全图宽松一次提取，可额外找回{r['restored_by_denoised_nonlocking']}条原漏轨，其中{r['restored_by_denoised_nonlocking_among_boost_sensitive']}条属于之前181条补强真边敏感样本。这证明其中存在提取规则交互，不能将181条全部解释为网络没有学会。

**该对照使用真值去错边，不能部署。** 不去错边、直接全图宽松一次提取，实际真轨效率只有{actual_single['track_efficiency']:.4%}、假候选比例{actual_single['fake_fraction']:.4%}。所以本轮不替换提取器，更不能直接取消所有锁定规则。

## 训练集抽查

{train_text}

训练抽查没有反向传播、优化器更新或新增训练轮次，权重哈希保持不变。验证病例清单只用于归因，不会移入原7000训练事件进行训练。

## 下一次训练对照应怎样设定

1. 先保留原架构和Focal基线，针对训练集上的“整轨普遍低分/混合碎裂”病例做分组诊断；原损失已经有Focal难例聚焦，不能把它描述成完全没有难例机制。
2. 比较按真轨平衡的正边贡献、训练集内系统性弱评分轨迹的采样/加权，以及邻近高分假边约束。不能只抬高所有正边分数，也不优先仅补少数桥边。
3. 提取器侧另做“先提出严格和宽松候选，再处理冲突”的小对照，避免过早锁定碎片；需结合几何和斜丝支持，在固定假轨/克隆约束下验证。它不要求第二个神经网络。
4. 先固定提取器比较训练改变，再固定模型比较提取器改变，避免同时改动无法归因。全部以完整成员和粗三维种子指标选方案，边F1仅作辅助。

逐径迹分数与连通性.csv、逐事件记录、阈值迁移数据和图像保存在本目录。200验证事件已在先前研究中使用，后100事件仅用于本次分数工作点迁移复核，不称为全新独立测试。

![分数、连通性与排序](edge_score_diagnosis.png)
'''
    (out/'真边分数与提取器诊断报告.md').write_text(text,encoding='utf-8')
    e.save_json(out/'report_metrics.json',dict(global_operating_points=global_points,ranking_auc=rank,
               actual_single_pass=actual_single,validation_topology=validation_topology,training_probe=training,
               stable_model_unchanged=True,training_optimization_steps=0))
    print(json.dumps(dict(categories=r['missed_categories'],ranking=rank,AA_global_budget_points=global_points['AA'],
                         actual_single_pass_efficiency=actual_single['track_efficiency'],training=training),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
