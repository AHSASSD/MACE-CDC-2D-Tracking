"""Render a fixed six-way model/decoder comparison without choosing thresholds."""
import argparse
import json
from common import MODEL_ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'direct_components':'网络直接分组（边≥0.5）','two_pass':'原两级寻迹','conservative_joint':'改进保守联合寻迹'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot;out=snapshot/'model_comparison_v1'
    report=json.loads((out/'paired_comparison.json').read_text(encoding='utf-8'))
    old,new=report['reference'],report['latest'];epochs=[old['plan']['epoch'],new['plan']['epoch']]
    table=[];deltas=[]
    for summary in [old,new]:
        for name,m in summary['methods'].items():
            table.append(f"| {summary['plan']['epoch']} | {LABELS[name]} | {m['matched_tracks']:,}/{m['truth_tracks']:,} | {m['track_efficiency']:.4%} | {m.get('fake_candidates',0):,} / {m['fake_fraction']:.4%} | {m.get('clone_candidates',0):,} / {m['clone_fraction']:.4%} | {m['joint_track_efficiency']:.4%} |")
    for name,comparison in report['model_differences'].items():
        a,b=old['methods'][name],new['methods'][name]
        ci=comparison['paired_difference_95_ci']['track_efficiency']
        deltas.append(f"| {LABELS[name]} | {100*(b['track_efficiency']-a['track_efficiency']):+.4f} | {b['matched_tracks']-a['matched_tracks']:+,} | {b.get('fake_candidates',0)-a.get('fake_candidates',0):+,} / {100*(b['fake_fraction']-a['fake_fraction']):+.4f} | {b.get('clone_candidates',0)-a.get('clone_candidates',0):+,} / {100*(b['clone_fraction']-a['clone_fraction']):+.4f} | [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] |")
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.7))
    x=np.arange(3)
    for ax,key,title in zip(axes,['track_efficiency','fake_fraction','clone_fraction'],['Track efficiency (%)','Fake candidates (%)','Clone candidates (%)']):
        for offset,summary,color in [(-.18,old,'#94a3b8'),(.18,new,'#2563eb')]:
            values=[100*summary['methods'][name][key] for name in LABELS]
            bars=ax.bar(x+offset,values,width=.36,label=f"Epoch {summary['plan']['epoch']}",color=color)
            ax.bar_label(bars,fmt='%.3f',fontsize=7)
        ax.set(title=title,xticks=x,xticklabels=['Raw graph','Two-pass','Joint hybrid'])
        ax.margins(y=.2);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(out/'model_tracking_comparison.png',dpi=180);plt.close(fig)
    b=new['methods']['two_pass'];m=new['methods']['conservative_joint'];raw=new['methods']['direct_components']
    previous_joint=old['methods']['conservative_joint'];difference=report['model_differences']['conservative_joint']
    uncertain=[]
    for key,label in [('fake_fraction','假轨率'),('clone_fraction','克隆率')]:
        ci=difference['paired_difference_95_ci'][key]
        if ci[0]<=0<=ci[1]:uncertain.append(label)
    uncertainty='、'.join(uncertain)+'差值的95%区间包含0，当前样本不支持宣称这些指标有显著改善。' if uncertain else ''
    selection=json.loads((snapshot/'best_model_selection.json').read_text(encoding='utf-8'))
    text=f'''# 第{epochs[1]}轮最佳模型与第{epochs[0]}轮：是否接入寻迹流程的固定对比

## 实测结果

同一固定测试划分的1500个事件、44,606条至少含5个直丝hit的真轨。两代GNN分别配合三种冻结分组方法；每个方法使用完全一致的80%纯度、50%完整度匹配规则。测试集此前已评估，本次属于固定测试集复测。

| GNN轮次 | 使用方式 | 匹配真轨 | 寻迹效率 | 假轨数 / 假轨率 | 克隆数 / 克隆率 | 直丝斜丝同时达标 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

同样使用改进联合流程，新模型相对第95轮净多匹配{m['matched_tracks']-previous_joint['matched_tracks']:+,}条真轨，假轨{previous_joint.get('fake_candidates',0)}→{m.get('fake_candidates',0)}条，克隆{previous_joint.get('clone_candidates',0)}→{m.get('clone_candidates',0)}条。效率提高并不意味着克隆也降低；应分别观察三项指标。{uncertainty}

第{epochs[1]}轮接入改进联合流程，相比直接分组，效率变化{100*(m['track_efficiency']-raw['track_efficiency']):+.4f}个百分点，假轨率变化{100*(m['fake_fraction']-raw['fake_fraction']):+.4f}个百分点。相比原两级流程，多匹配{m['matched_tracks']-b['matched_tracks']:+,}条真轨，假轨变化{m.get('fake_candidates',0)-b.get('fake_candidates',0):+,}条，克隆变化{m.get('clone_candidates',0)-b.get('clone_candidates',0):+,}条。

## 只更换GNN权重的变化

下表差值均为第{epochs[1]}轮减去第{epochs[0]}轮。“百分点”直接对比例相减；置信区间来自2000次按事件成对bootstrap，只描述当前固定测试样本的抽样波动。

| 相同流程 | 效率差/百分点 | 净增匹配真轨 | 假轨数差 / 比例差（百分点） | 克隆数差 / 比例差（百分点） | 效率差95%区间/百分点 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(deltas)}

完整成对结果和三个指标的置信区间见paired_comparison.json。新找到和同时失去的真轨分别保存，避免仅凭净增加数掩盖成员变化。

## “直接使用”与两种寻迹流程的区别

1. 网络直接分组：同一无向边的两个方向概率取平均，概率≥0.5的边连成分量；保留至少5个直丝hit的候选。没有漂移圆拟合、候选竞争、碎片处理或成员可靠性校准。网络本身输出边概率，仍需要这个最小解码步骤才能计算径迹指标。
2. 原两级寻迹：AA严格/宽松两级拆分和漂移圆几何检查，再通过AS/SS概率与有限丝段必要条件分配斜丝。
3. 改进联合寻迹：保留原两级候选，加入联合候选池、冻结候选可靠性评分和保守的排他成员选择，最后统一分配斜丝。

直接分量容易被少量错误跨轨边串联；克隆少可能来自合并和漏轨，不能单看克隆率判断优劣。此前STEP4简单分量报告采用70%/70%匹配和不同边对称方式，不能与本报告直接相减；第95轮直接分组的结果已按本次统一口径重新计算。

## 冻结模型、参数与数据隔离

当前下载的是服务器以最小验证损失选出的第{epochs[1]}轮，验证损失{selection['validation']['loss']:.8f}，验证边宏F1为{selection['validation']['macro_f1']:.4%}。它与训练日志最后完成的轮次不同。权重下载前后进行哈希核对，并核对原训练历史对应轮次记录。

- 第{epochs[0]}轮GNN SHA256：`{old['plan']['weight_sha256']}`。
- 第{epochs[1]}轮GNN SHA256：`{new['plan']['weight_sha256']}`。
- 候选校准器SHA256：`{new['plan']['confidence_model_sha256']}`。两代模型都沿用第95轮的校准器，基线保护阈值0.02、新候选阈值0.20；未重训或按测试结果调参。这项比较测量的是直接替换GNN的效果，不代表新模型经重新校准后的最优表现。
- 同一数据元信息与几何哈希、同一1500个事件ID、全部源码哈希保存在contract.json。第95轮两套寻迹成员结果复用既有冻结输出，并重新评分核对一致；其直接分量重新计算。
- 重建仅接收几何、CellID、独立HitID、漂移距离d和GNN概率；TrkID仅用于最终评分。不读取时间字段，不合并重复CellID，不给已有d重复加噪声。

## 评分与限制

候选直丝纯度≥80%、完整度≥50%才匹配真轨，同一真轨后续重复匹配计克隆。效率分母为44,606条合格真轨；假轨率和克隆率的分母分别为各方法输出的候选总数。表中联合达标还要求真轨至少3个斜丝hit、候选斜丝纯度≥80%及完整度≥50%。

这里的寻迹效率指成员识别召回率，不是每秒处理事件数。第95轮复用了既有成员结果，当前运行的墙钟时间不能充当两代模型公平速度比较。结果评价成员关联与二维几何，尚不能解释为三维参数或动量精度。

## 复现

```powershell
python STEP16_新旧模型寻迹对比.py --snapshot {args.snapshot} --workers 10
python 生成新旧寻迹对比报告.py --snapshot {args.snapshot}
python STEP4_联合寻迹推理.py --snapshot {args.snapshot} --event 4779 --output inference_epoch{epochs[1]}_04779.json
```

逐事件候选保存独立HitID，断言同一方法内无重复归属；完整报告、参数和成对区间保存在model_comparison_v1。原第95轮权重、评估文件和默认选择在本次比较中保持可复现。

![新旧模型和三种流程](model_tracking_comparison.png)
'''
    (out/'新旧模型与寻迹流程对比.md').write_text(text,encoding='utf-8')
    print(out/'新旧模型与寻迹流程对比.md')


if __name__=='__main__':main()
