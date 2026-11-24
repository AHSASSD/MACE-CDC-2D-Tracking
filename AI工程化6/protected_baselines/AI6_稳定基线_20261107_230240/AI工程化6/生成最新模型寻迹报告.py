"""Summarize frozen latest-model testing without selecting on test results."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import MODEL_ROOT,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);args=parser.parse_args()
    snapshot=MODEL_ROOT/args.snapshot
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    edge=load(snapshot/'snapshot_evaluation_test.json');val=load(snapshot/'two_stage_validation_v1/summary.json')
    repair=load(snapshot/'tracking_repair_validation_v1/summary.json');test=load(snapshot/'frozen_tracking_test_v1/summary.json')
    old=MODEL_ROOT/'local_evaluation_epoch051_20261106';oldedge=load(old/'snapshot_evaluation_test.json');oldval=load(old/'two_stage_validation_v1/summary.json')
    selected=test['plan']['profile'];b=test['baseline'];m=test['selected'];e=edge['result'];ov=oldval['methods']['stereo_assisted'];nv=val['methods']['stereo_assisted']
    assert sha(snapshot/'best_interaction_network_joint_no_time.pth')==edge['weight_sha256']==test['plan']['weight_sha256']
    assert val['config']['event_ids']==oldval['config']['event_ids']
    assert test['events']==1500 and not test['plan']['test_used_for_selection']
    rows='\n'.join(f"| {kind} | {oldedge['result']['types'][kind]['f1']:.2%} | {e['types'][kind]['f1']:.2%} | {e['types'][kind]['precision']:.2%} | {e['types'][kind]['recall']:.2%} |" for kind in ['AA','AS','SS'])
    stats='\n'.join(f"| {name} | {v['matched_tracks']:,}/{v['truth_tracks']:,} | {v['track_efficiency']:.4%} | {v.get('fake_candidates',0):,} / {v['fake_fraction']:.4%} | {v.get('clone_candidates',0)} / {v['clone_fraction']:.4%} | {v['joint_track_efficiency']:.2%} |" for name,v in [('原两级流程',b),('冻结验证选择方案',m)])
    low,high=test['paired_efficiency_difference_95_interval']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    x=np.arange(3)
    axes[0].bar(x-.18,[oldedge['result']['types'][k]['f1'] for k in ['AA','AS','SS']],.36,label='Epoch 51')
    axes[0].bar(x+.18,[e['types'][k]['f1'] for k in ['AA','AS','SS']],.36,label=f"Epoch {edge['selected_epoch']}")
    axes[0].set(xticks=x,xticklabels=['AA','AS','SS'],ylim=(0,1),title='Fixed 1,500-event test edge F1');axes[0].legend()
    axes[1].bar(['Epoch 51','Epoch 95'],[ov['track_efficiency'],nv['track_efficiency']],color=['#94a3b8','#2563eb'])
    axes[1].set(ylim=(0,1.05),title='Same 200 validation events: track recall')
    vals=[b['track_efficiency'],m['track_efficiency'],b['fake_fraction'],m['fake_fraction'],b['clone_fraction'],m['clone_fraction']]
    bars=axes[2].bar(['Recall\nbase','Recall\nselected','Fake\nbase','Fake\nselected','Clone\nbase','Clone\nselected'],vals,color=['#94a3b8','#16a34a']*3)
    axes[2].bar_label(bars,labels=[f'{v:.3%}' for v in vals],fontsize=7)
    axes[2].set(ylim=(0,1.05),title='Frozen tracking test: 1,500 events');axes[2].tick_params(axis='x',labelsize=7)
    fig.tight_layout();fig.savefig(snapshot/'latest_model_tracking.png',dpi=180);plt.close(fig)
    text=f'''# 最新最佳联合模型效果与寻迹改进

本次读取服务器验证损失最低的第{edge['selected_epoch']}轮模型，下载后SHA256一致并固定快照。原6小时作业1528054已到期，完整断点到95轮，第96轮未完成；目标200轮尚未达到，不能宣布已经收敛。最后学习率仍为0.001。测试和寻迹改进均在本机完成，未改变训练权重。

## 全部1500固定测试事件：边关联

| 边类型 | 第51轮F1 | 第95轮F1 | 第95轮Precision | 第95轮Recall |
|---|---:|---:|---:|---:|
{rows}

宏F1由{oldedge['result']['macro_f1']:.2%}提高到{e['macro_f1']:.2%}。边阈值均为0.5，输入、事件及评分口径相同；它是边分类指标，不是寻迹效率。边指标按有向候选边累计。

## 相同200验证事件：模型变化

原冻结两级＋斜丝归属配置下，第51轮与第95轮的直丝真轨匹配效率分别为{ov['track_efficiency']:.4%}、{nv['track_efficiency']:.4%}；假候选比例为{ov['fake_fraction']:.4%}、{nv['fake_fraction']:.4%}；克隆数为{ov.get('clone_candidates',0)}、{nv.get('clone_candidates',0)}。这些才是同事件跨模型比较。

## 改进方案选择

采用几何与网络证据共同约束的片段合并、独立hit竞争回收，并试验额外剩余子图恢复。前100验证事件选择方案，后100验证事件复核。要求真轨效率不降、假候选比例不升、克隆比例不升；复核不满足就回退baseline。前半选出{repair['selected_on_dev']}，最终冻结{selected}。

修复候选不使用TrkID、真实位置、时间或动量。合并要求片段间多hit高概率证据且共享漂移圆拟合不恶化；补点要求至少两个不同CellID支持、几何残差<=1mm及领先竞争候选至少0.10。所有hit保持独立HitID且至多属于一个候选。详细验证指标见tracking_repair_validation_v1/寻迹改进验证报告.md。

## 全部1500固定测试事件：冻结方案复测

| 方案 | 匹配真轨 | 效率 | 假候选数 / 比例 | 克隆数 / 比例 | 直丝斜丝联合成员达标 |
|---|---:|---:|---:|---:|---:|
{stats}

净增加匹配真轨{m['matched_tracks']-b['matched_tracks']:+,}条（新增找回{test['recovered']}条、失去{test['lost']}条）。效率差为{100*(m['track_efficiency']-b['track_efficiency']):+.4f}个百分点，按事件成对bootstrap的95%区间为{low*100:+.4f}至{high*100:+.4f}个百分点。区间若跨零，不能称为稳定提升。假轨与克隆数的变化应与效率同时阅读，测试结果未用于重新选择配置。

冻结方案已匹配候选内的斜丝纯度{m['stereo_purity_matched']:.2%}、斜丝完整度{m['stereo_completeness_matched']:.2%}；覆盖全部合格真轨的斜丝hit正确找回比例{m['stereo_recall_all_eligible_tracks']:.2%}。前两项是条件指标。

## 口径、限制和复现

真轨分母至少5个直丝hit；候选直丝纯度>=80%、完整度>=50%才匹配，第二次匹配同一真轨计克隆。联合成员达标另要求真轨至少3个斜丝hit、候选斜丝纯度>=80%、完整度>=50%。假候选/克隆率分母为输出候选数，均为微平均；不能与70%/70%的简单连通分量诊断混用。

两级使用AA的严格/宽松阈值0.70/0.575及1.0/1.5mm二维漂移圆验收。斜丝按AS/SS及有限丝段几何必要条件竞争归属，未将斜丝参考XY当成击中点。三维拟合仍为独立小样本原型，不把纵向约束不足或多解作为当前硬拒绝条件。本报告评价成员关联，未验证三维参数或动量精度。

固定测试集先前已被第51轮评估过；本次是相同测试集复测，不是新的未暴露留出集。当前模型按验证损失选取，后处理按验证选择/复核冻结，未扫描测试阈值。

权重SHA256：`{edge['weight_sha256']}`。frozen_tracking_test_v1/contract.json包含事件、权重、配置、源码及验证选择报告哈希；events/保存逐事件评分和候选，summary.json保存汇总与区间。保留预测缓存和逐事件结果，可从已有结果续跑。

![最新模型效果](latest_model_tracking.png)
'''
    (snapshot/'最新模型效果与寻迹改进报告.md').write_text(text,encoding='utf-8')
    print(snapshot/'最新模型效果与寻迹改进报告.md')


if __name__=='__main__':main()
