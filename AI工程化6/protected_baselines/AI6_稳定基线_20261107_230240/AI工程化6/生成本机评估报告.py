"""Render the fixed epoch-51 snapshot results without rerunning inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    run = root / 'joint_training/local_evaluation_epoch051_20261106'
    report = json.loads((run / 'snapshot_evaluation_test.json').read_text(encoding='utf-8'))
    history = json.loads((run / 'training_history.json').read_text(encoding='utf-8'))
    selection = json.loads((run / 'best_model_selection.json').read_text(encoding='utf-8'))
    result = report['result']
    c = result['candidate_diagnostic']
    fresh = report['historical_pilot_excluded_subset']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = [h['epoch'] for h in history]
    axes[0, 0].plot(epochs, [h['train_loss'] for h in history], label='Train')
    axes[0, 0].plot(epochs, [h['validation']['loss'] for h in history], label='Validation')
    axes[0, 0].set(xlabel='Completed epoch', ylabel='Focal loss', title='Scratch training through snapshot epoch')
    axes[0, 0].legend()
    for kind in ['AA', 'AS', 'SS']:
        axes[0, 1].plot(epochs, [h['validation']['types'][kind]['f1'] for h in history], label=kind)
    axes[0, 1].set(xlabel='Completed epoch', ylabel='Edge F1', title='Validation (model selection only)', ylim=(0, 1))
    axes[0, 1].legend()
    x = np.arange(3)
    for offset, metric in [(-.24, 'precision'), (0, 'recall'), (.24, 'f1')]:
        values = [result['types'][kind][metric] for kind in ['AA', 'AS', 'SS']]
        bars = axes[1, 0].bar(x + offset, values, .24, label=metric)
        axes[1, 0].bar_label(bars, fmt='%.3f', fontsize=7)
    axes[1, 0].set(xticks=x, xticklabels=['AA', 'AS', 'SS'], ylim=(0, 1.08),
                   ylabel='Score', title='Fixed test: 1,500 events, threshold=0.5')
    axes[1, 0].legend(fontsize=8, loc='lower right')
    bars = axes[1, 1].bar(['Truth-track\nmatch efficiency', 'Fake candidate\nfraction'],
                         [c['axial_efficiency'], c['fake_fraction']], color=['#2563eb', '#ef4444'])
    axes[1, 1].bar_label(bars, labels=[f"{c['axial_efficiency']:.2%}", f"{c['fake_fraction']:.2%}"])
    axes[1, 1].set(ylim=(0, 1), ylabel='Fraction', title='Connected components only; no track fit')
    fig.suptitle(f"AI6 joint axial/stereo network — frozen epoch {report['selected_epoch']}")
    fig.tight_layout()
    fig.savefig(run / 'evaluation_summary.png', dpi=180)
    plt.close(fig)
    rows = '\n'.join(f"| {name} | {result['types'][kind]['precision']:.2%} | {result['types'][kind]['recall']:.2%} | {result['types'][kind]['f1']:.2%} |"
                     for kind, name in [('AA', '直丝—直丝 AA'), ('AS', '直丝—斜丝 AS'), ('SS', '斜丝—斜丝 SS')])
    text = f'''# AI6 第51轮最佳模型：本机效果评估

本次固定的是任务开始时验证损失最低的第51轮权重。gpu14上的训练继续推进至总计200轮；本报告是阶段性快照，不是最终收敛结论。全新模型hidden=256、8次消息计算，直丝和斜丝联合输入，包含带符号倾斜角，不使用时间、真实位置或动量作为输入。

## 评估口径

- 固定原始划分：7000训练、1500验证、1500测试。模型仅按验证损失选取；本轮未按测试结果调参或挑选权重。
- 模型选择验证loss：{selection['validation']['loss']:.8f}，验证宏F1：{selection['validation']['macro_f1']:.2%}。
- 本机GPU：{report['gpu']}；阈值固定0.5，batch=1，FP16 AMP，分块65536，2个数据加载进程。完整评估约{report['seconds']:.1f}秒，包含加载、建图、前向及评分。
- 原始压缩事件惰性建图，所有候选边均参与。保留重复CellID的独立HitID，真值只用于评分。
- 边指标按有向候选边累计TP/FP/FN；宏F1是AA/AS/SS三个F1的等权均值，不是寻迹效率。

## 固定测试集边关联

共1500事件、{result['edges']:,}条有向候选边；测试loss为{result['loss']:.8f}。

| 类型 | Precision | Recall | F1 |
|---|---:|---:|---:|
{rows}

宏平均F1：**{result['macro_f1']:.2%}**。排除历史探索已查看的64个测试事件后，剩余{fresh['events']}事件宏F1为**{fresh['result']['macro_f1']:.2%}**，两者接近。该1436事件子集已被本次评估查看，后续不能再称为未暴露测试集；使用记录见上级evaluation_exposure_history.jsonl。

## 候选重建诊断

使用概率>=0.5的边取无向连通分量，至少5个直丝hit才输出候选。评分真轨也要求至少5个直丝hit；候选直丝纯度和完整度均>=70%才匹配。真值不参与连通分量构建。此口径沿用前期联合模型诊断，与AI4的圆拟合及80%/50%评分不同，不能直接比较其寻迹效率。

| 指标 | 结果 |
|---|---:|
| 合格真轨数 | {c['truth_tracks']:,} |
| 匹配真轨数 | {c['matched_tracks']:,} |
| 真轨匹配效率 | {c['axial_efficiency']:.2%} |
| 输出候选数 | {c['output_candidates']:,} |
| 假候选数及比例 | {c['fake_candidates']:,} / {c['fake_fraction']:.2%} |
| 克隆候选数 | {c['clone_candidates']} |
| 已匹配候选的直丝平均完整度 | {c['axial_completeness_matched']:.2%} |
| 已匹配候选的斜丝纯度 | {c['stereo_purity']:.2%} |
| 已匹配候选的斜丝完整度 | {c['stereo_completeness_matched']:.2%} |

**当前直接连通分量解码的整体匹配效率仍低。** 已成功匹配候选内的斜丝纯度/完整度接近98%，只描述匹配成功的部分，不能解释为全部斜丝已被正确重建。边级假阳性可能连接不同真轨；较好的边F1不能保证连通分量寻迹良好。本报告未额外统计假候选中的具体串轨原因，不能据此确定全部失败都由串轨导致。克隆数为零也不能证明完备的全局竞争能力。

后续训练继续仅使用训练/验证集；本次测试结果不用于挑选阈值或改动正在运行的训练配置。若进一步研究候选解码，应在验证集完成方案选择，再使用另外固定的评估口径。这里没有三维拟合、斜丝真实hit位置估计或动量精度结果。

## 文件与复现

权重SHA256：`{report['weight_sha256']}`。源模型元数据和第1—51轮训练历史均保存在本目录。STEP4_本机快照评估.py只读取快照；服务器继续保存新的最佳模型不会改变本报告权重。候选评分回归测试test_snapshot_evaluation.py已验证正确分离、错误合并、空候选及边计数。

- snapshot_evaluation_test.json：完整和1436事件子集指标。
- snapshot_events_test.jsonl：逐事件边计数与候选评分，便于复核。
- local_evaluation_test.log：本机完整运行日志。
- evaluation_summary.png：训练曲线、测试边指标及候选诊断。

![评估图](evaluation_summary.png)
'''
    (run / '本机模型效果评估.md').write_text(text, encoding='utf-8')
    print(run / '本机模型效果评估.md')


if __name__ == '__main__':
    main()
