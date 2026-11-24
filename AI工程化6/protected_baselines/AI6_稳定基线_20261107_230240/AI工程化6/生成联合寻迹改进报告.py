"""Auditable model/geometry/candidate-confidence ablations and frozen testing."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import MODEL_ROOT


def main():
    snapshot=MODEL_ROOT/'local_evaluation_epoch095_20261107';load=lambda p:json.loads((snapshot/p).read_text(encoding='utf-8'))
    fresh=load('hybrid_fresh_validation_v1/summary.json');test=load('hybrid_frozen_test_v1/summary.json')
    hybrid=load('hybrid_confidence_v1/summary.json');joint=load('joint_pool_validation_v1/summary.json')
    confidence=load('candidate_confidence_v2/summary.json');consensus=load('drift_consensus_validation_v1/summary.json')
    out=snapshot/'joint_tracking_improvement_report';out.mkdir(exist_ok=True)
    ablations=[('Baseline',joint['check']['baseline']),('Joint recovery',joint['check']['joint_residual']),
               ('Global pool',joint['check']['joint_global']),('Confidence only',confidence['check']['confidence_020']),
               ('Drift consensus',consensus['check']['consensus_raw']),('Protected hybrid',hybrid['check']['base_0.02_new_0.2'])]
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for name,m in ablations:
        axes[0].scatter(100*m['fake_fraction'],100*m['track_efficiency'],s=55)
        axes[0].annotate(name,(100*m['fake_fraction'],100*m['track_efficiency']),xytext=(4,4),textcoords='offset points',fontsize=7)
    axes[0].set(xlabel='Fake candidates (%)',ylabel='Axial truth-track efficiency (%)',title='Ablation: held 100 validation events')
    for ax,report,title in [(axes[1],fresh,'Frozen additional 300 validation events'),(axes[2],test,'Frozen 1,500 test events')]:
        b,m=report['baseline'],report['selected'];x=np.arange(3)
        vb=[100*b[k] for k in ['track_efficiency','fake_fraction','clone_fraction']];vm=[100*m[k] for k in ['track_efficiency','fake_fraction','clone_fraction']]
        bars=ax.bar(x-.18,vb,.36,label='Baseline',color='#94a3b8');ax.bar_label(bars,fmt='%.3f',fontsize=7)
        bars=ax.bar(x+.18,vm,.36,label='Protected hybrid',color='#16a34a');ax.bar_label(bars,fmt='%.3f',fontsize=7)
        ax.set(xticks=x,xticklabels=['Track recall','Fake rate','Clone rate'],ylabel='Percent',ylim=(0,105),title=title);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'joint_tracking_improvement.png',dpi=180);plt.close(fig)
    rows=[]
    for title,report in [('额外300验证事件',fresh),('完整1500测试事件',test)]:
        for label,key in [('原两级方案','baseline'),('保守联合方案','selected')]:
            m=report[key];rows.append(f"| {title} | {label} | {m['matched_tracks']:,}/{m['truth_tracks']:,} | {m['track_efficiency']:.4%} | {m.get('fake_candidates',0)} / {m['fake_fraction']:.4%} | {m.get('clone_candidates',0)} / {m['clone_fraction']:.4%} | {m['joint_track_efficiency']:.2%} |")
    b,m=test['baseline'],test['selected'];fb,fm=fresh['baseline'],fresh['selected'];lo,hi=test['efficiency_difference_95_ci']
    ablation_rows='\n'.join(f"| {name} | {v['track_efficiency']:.4%} | {v['fake_fraction']:.4%} | {v.get('clone_candidates',0)} |" for name,v in ablations)
    text=f'''# 联合候选池、可靠性校准与保守融合：实际改进结果

本轮未使用skills。固定第95轮最佳GNN、原数据与匹配条件，先诊断失败来源，再比较联合恢复、全局候选选择、候选可靠性筛选、漂移圆共识恢复和保守融合。本报告汇总冻结配置在额外300验证事件及完整1500测试事件上的成对结果。

## 最终结果

| 样本 | 方案 | 匹配真轨 | 效率 | 假候选数 / 比例 | 克隆数 / 比例 | 直丝斜丝成员同时达标 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

完整测试集净增加{m['matched_tracks']-b['matched_tracks']:+}条匹配真轨（新增找回{test['gained']}条，同时失去{test['lost']}条），假候选变化{m.get('fake_candidates',0)-b.get('fake_candidates',0):+}条、克隆变化{m.get('clone_candidates',0)-b.get('clone_candidates',0):+}条。效率差为{100*(m['track_efficiency']-b['track_efficiency']):+.4f}个百分点，按事件成对bootstrap的95%区间为{lo*100:+.4f}至{hi*100:+.4f}个百分点。该区间仅描述当前固定样本的抽样波动。

额外300验证事件净多匹配{fm['matched_tracks']-fb['matched_tracks']}条、假候选减少{fb.get('fake_candidates',0)-fm.get('fake_candidates',0)}条、克隆减少{fb.get('clone_candidates',0)-fm.get('clone_candidates',0)}条。该300事件未用于本轮后处理校准或选择，但属于GNN原来的验证划分。测试集曾被先前模型评估过，属于固定测试集复测，不称为新的未暴露留出集。

测试的三项比例均未恶化：{test['non_regression_rates']}；三项绝对计数取舍符合约定：{test['non_regression_counts']}。这些结果不构成任意数据下的全局最优保证。

## 为什么改变候选处理方式

200个开发验证事件中，原方案漏掉304条合格真轨，其中251条仍有至少5个未使用直丝hit；166条假候选中115条主要是同轨碎片覆盖不足、51条是混轨。直接放低阈值或增加剩余恢复会同时带入更多短碎片。

1. 联合候选生成：保留原直丝两级基线，同时用AA/AS/SS构造剩余图，在固定的多个图尺度上提出替代候选，直丝圆拟合继续检查几何；斜丝证据参与提出直丝成员假设。
2. 候选可靠性：用12个可观测特征综合评估，包括内部AA概率、斜丝支持与SS一致性、外部高概率延伸、拟合残差、候选大小及半径。原始事件ID、TrkID、真值完整度、时间和真实位置均不进入评分器。
3. 保守融合：基线候选仅在可靠性低于0.02时释放；新增候选要求至少0.20。先保护已有成员，再通过带hit排他约束的整数优化选择新增候选，最后统一分配斜丝。每个独立HitID最多归属一个输出候选，重复CellID保留独立hit。

这套策略把“提出更多可能性”和“最终接受哪些成员”分开。直接全局替换或直接筛掉全部低分候选的效果不及保守融合。

## 比较过的方案

以下同为第二批100验证事件，均与同一基线成对比较，不能与1500测试事件的数值直接相减。

| 方案 | 真轨效率 | 假候选比例 | 克隆数 |
|---|---:|---:|---:|
{ablation_rows}

漂移圆共识使用网络支持的三hit相切圆假设和稳健拟合，可找回额外成员，但在当前校准复核中带入过多碎片，未进入最终方案。候选评分器直接全量筛选能降低假轨，却损失真轨；最终只在保守融合中使用它。

## 数据分工与信息隔离

- GNN未重训，权重固定为第95轮：`{test['plan']['weight_sha256']}`。
- 从原验证集取前100个开发事件拟合候选可靠性校准器，TrkID仅产生训练标签。候选是否有效按直丝纯度>=80%、完整度>=50%标注。它是额外的有监督校准器，不应描述为完全没有训练。
- 在上述100事件内按事件分组做5折交叉预测来选工作点；来自同一事件的候选不会跨训练/交叉预测分组。
- 其余100开发验证事件复核后，冻结基线/新增阈值0.02/0.20。再在另取300验证事件复核，最后运行固定1500测试事件；未根据测试结果重新选择阈值。
- 该校准器来自验证集中的专用校准子集，因此不能把其训练内分数当作泛化性能。部署评估以额外300事件及测试结果为准。

## 评分与物理范围

合格真轨至少5个直丝hit；候选直丝纯度>=80%、完整度>=50%才匹配。同一真轨重复匹配计克隆；假候选和克隆比例以输出候选数为分母。联合成员达标另要求至少3个斜丝hit、斜丝纯度>=80%且完整度>=50%。

冻结测试方案的已匹配候选内斜丝纯度为{m['stereo_purity_matched']:.2%}、完整度为{m['stereo_completeness_matched']:.2%}；全部合格真轨斜丝hit正确找回比例为{m['stereo_recall_all_eligible_tracks']:.2%}。前两项是条件指标。

几何约束仍以二维漂移圆和有限斜丝必要条件为主；斜丝参考XY不作为真实击中位置。单段螺旋三维验收仍是独立原型，本轮未将多解或纵向约束不足直接当作假轨拒绝条件。本报告评价成员关联，不代表三维参数或动量精度。

## 实现、检查与复现

主入口STEP15_保守联合寻迹复核.py，推理模块hybrid_joint_tracker.py的track仅接收测量、几何、GNN概率、候选和冻结评分器，不接收真值。候选评分器保存为candidate_confidence_v2/candidate_confidence.pkl。联合候选、全局目标、成员排他、特征输入及部署重放检查通过。

hybrid_fresh_validation_v1及hybrid_frozen_test_v1保存完整事件、参数、GNN/校准器/源码哈希、逐事件候选、独立HitID、CellID及评分。每事件原子保存，重复执行可恢复已有结果；共享图概率缓存避免反复运行GNN。冻结测试后不再改源码与参数。

该新方案增加候选生成、可靠性推断和冲突优化的计算；本轮不能据此宣称推理速度也提升。原网络权重和服务器训练配置保持原样，旧方案和所有实验均保留。

![联合寻迹改进](joint_tracking_improvement.png)
'''
    (out/'联合寻迹最佳已验证方案.md').write_text(text,encoding='utf-8')
    print(out/'联合寻迹最佳已验证方案.md')


if __name__=='__main__':main()
