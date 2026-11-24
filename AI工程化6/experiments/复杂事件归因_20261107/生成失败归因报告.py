"""Describe intervention evidence without presenting truth-oracle scores as deployable."""
import experiment_common as e
import csv
import json
from collections import Counter

BUCKETS={'pure_true_graph_fragmented':'纯真AA图碎裂/最小hit门槛',
         'single_circle_or_fitter_limit':'单圆表达或当前拟合器限制',
         'extractor_organization_limit':'当前候选提取组织限制',
         'sensitive_to_either_FP_or_FN':'去错边或补真边均能找回',
         'false_edge_interference':'去错边后找回',
         'weak_true_edge_scores':'补强真边后找回',
         'combined_score_interference':'同时纠正两类边后找回'}
NAMES={'actual_axial_stage':'原直丝两级阶段',
       'oracle_remove_false_edges':'诊断：仅屏蔽错连边',
       'oracle_boost_true_edges':'诊断：仅补强真边',
       'oracle_perfect_edges':'诊断：真边1/错边0'}


def main():
    out=e.EXP/'attribution_v1';r=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    rows=[]
    with (out/'失败径迹明细.csv').open('w',encoding='utf-8-sig',newline='') as f:
        names=['event_id','truth_id','bucket','axial_hits','rms','rms_after_up_to_two_trims','unused_true_axial_hits','restored_by_remove_FP','restored_by_boost_FN','restored_by_perfect_edges']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for eid in e.IDS:
            event=json.loads((out/'events'/f'event_{eid:05d}.json').read_text(encoding='utf-8'))
            for t in event['missed_tracks']:
                row={k:t.get(k) for k in names};row.update(event_id=eid,rms=t['geometry']['rms'],rms_after_up_to_two_trims=t['geometry']['rms_after_up_to_two_trims']);writer.writerow(row);rows.append(row)
    table=[]
    for name,m in [('完整稳定联合基线',r['baseline'])]+[(NAMES[k],v) for k,v in r['oracle_axial_stage'].items()]:
        table.append(f"| {name} | {m['matched_tracks']}/{m['truth_tracks']} | {m['track_efficiency']:.4%} | {m.get('fake_candidates',0)} | {m.get('clone_candidates',0)} |")
    buckets='\n'.join(f"| {BUCKETS[k]} | {n} | {n/max(r['missed_tracks'],1):.2%} |" for k,n in sorted(r['diagnostic_buckets'].items(),key=lambda v:-v[1]))
    h=r['high_occupancy_proxy'];hb=h['results']['baseline'];b=r['baseline'];recover=r['overlapping_oracle_recoveries']
    text=f'''# 稳定基线备份与复杂事件失败归因

## 备份与隔离

备份目录：`{e.PIN['backup']}`。完整工程23,503文件约2.47GiB，逐文件SHA256核对；manifest SHA256为`{e.PIN['manifest_sha256']}`。复制内容包括代码、10,000事件数据、冻结权重、候选校准器、概率缓存和逐事件评估。仅排除字节码缓存与临时文件。原CSV和几何XLSX外部源继续保留在原处。

已从备份直接运行STEP18无真值输入推理，事件9408的直丝、确认斜丝与待定斜丝身份均与备份前结果一致。恢复探针输出在实验目录，未写入备份。后续诊断导入备份代码，实验产物独立保存，稳定程序和权重不覆盖。

## 归因范围与对照

固定第126轮GNN、原候选校准器与几何，使用200个既有验证事件，不读取测试集进行本轮归因。200事件都属于本轮开发诊断材料；后续改进另用额外验证事件复核。评分仍为至少5直丝hit、候选直丝纯度≥80%、完整度≥50%，假候选/克隆分母为输出候选。

完整联合基线有{b['truth_tracks']:,}条目标真轨，找到{b['matched_tracks']:,}条，剩余{r['missed_tracks']}条；假候选{b.get('fake_candidates',0)}条、克隆{b.get('clone_candidates',0)}条。

| 对照 | 真轨匹配 | 效率 | 假候选 | 克隆 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

后三种oracle对照使用真值改边分数，只是固定直丝提取器的敏感性试验；不是可部署算法、性能承诺或整个联合流程的严格上限。去错边保留真边原评分；补真边保留错边原评分；perfect统一真边为1会抹掉原有边排序，也可能改变拆分行为。完整联合基线还含AS/SS联合候选与候选校准，因此应将干预与“原直丝两级阶段”比较，并单独看它能否找回完整联合基线漏掉的径迹。

## 漏轨分桶

| 根据干预与几何证据分桶 | 数量 | 占当前漏轨 |
|---|---:|---:|
{buckets}

在完整基线漏掉的径迹中，去错边可找回{recover.get('remove_FP',0)}条，补真边可找回{recover.get('boost_FN',0)}条，perfect边可找回{recover.get('perfect_edges',0)}条。三组会重叠，不能相加当作独立因果比例。

“纯真图碎裂”按至少5个纯真直丝hit且覆盖半条真轨检查；这是纯成员构成能力诊断，不是80%纯度允许少量错误点时的数学上限。另存AA图和联合AA/AS/SS图在全部候选边、0.5、0.575、0.7下的最大真轨直丝连通分量，便于区分缺边和评分压低。

每条漏轨用其真实直丝分组交给原多初值漂移圆拟合器，再试最多两个异常点，记录全组/处理后RMS。若仍不能在1.5mm解释，归为“单圆或当前拟合器限制”；这不直接证明某条轨迹发生多圈或反散射，也不能排除数值拟合问题。实际物理类型需进一步结合仅供离线诊断的真值轨迹验证。

## 假候选与斜丝损失

假候选按有效真轨、纯度和完整度分解：`{json.dumps(r['fake_causes'],ensure_ascii=False)}`。纯但不完整的碎片与混轨分别记录；逐候选另保存其真轨是否已被其他候选找到。

原正确关联、未进入粗三维确认集的斜丝分解：`{json.dumps(r['correct_stereo_not_confirmed'],ensure_ascii=False)}`。每次重放粗三维候选与原冻结结果逐一核对，再按有限交点/弧段、纵向共识和竞争进行诊断；这里不把所有未确认点都解释为网络错误。

斜丝“无有限弧内交点”项是几何症状分组，并非已证明的独立因果机制：线上最终验收使用有限丝段距离，并不要求每个已验收hit都有名义交点。要证明补充交点分支能减少损失，还需要专门的受控干预；本轮没有据此宣称收益。

## 复杂度分组

用可观测hit数、图平均度、重复CellID比例和纵向多解比例的等权经验分位构成占用/歧义代理，取最高四分位，共{len(h['event_ids'])}事件。该组真轨效率{hb['track_efficiency']:.4%}，总体为{b['track_efficiency']:.4%}。分组不使用真值轨迹数或真实位置；它是可复现的复杂度代理，不能等同于真实多圈标签。

完整逐事件诊断、几何残差、对照匹配ID、可观测特征和例子位于events；失败径迹明细.csv可逐条追查。后续方法的重建接口将不接收truth；真值只保留在离线诊断和评分中。
'''
    (out/'稳定基线与失败归因报告.md').write_text(text,encoding='utf-8');print(out/'稳定基线与失败归因报告.md')


if __name__=='__main__':main()
