from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'whole_event_retraining' / 'interrupted_no_time_assessment'


def main():
    report = json.loads((OUT / 'comparison.json').read_text(encoding='utf-8'))
    base, fine = report['summaries']['baseline'], report['summaries']['finetuned']
    history = report['training_history']
    best = min(history['Loss/validation'], key=lambda r: r['value'])
    last = max(r['epoch'] for r in history['Loss/validation'])
    assert abs(fine['edge']['loss'] - best['value']) < 1e-5
    for value in [base, fine]:
        assert value['output_tracks'] == value['matched_tracks'] + value['fake_tracks'] + value['clone_tracks']
        assert value['matched_tracks'] <= value['truth_tracks']
    assert base['truth_tracks'] == fine['truth_tracks']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, label in [('Loss/train', 'Training loss'), ('Loss/validation', 'Validation loss')]:
        axes[0].plot([r['epoch'] for r in history[tag]], [r['value'] for r in history[tag]], label=label)
    axes[0].axvline(best['epoch'], linestyle='--', color='black', alpha=.6, label=f'Best checkpoint: {best["epoch"]}')
    axes[0].set(xlabel='Epoch', ylabel='Focal loss', title=f'Training stopped after logged epoch {last}')
    axes[0].legend(fontsize=8)
    bins = ['0_1', '1_5', '5_15', '15_60', '60_plus']
    for value, label in [(base, 'Original weights'), (fine, 'Fine-tuned weights')]:
        axes[1].plot(range(len(bins)), [value['by_truth']['radius_bin'][b]['efficiency'] * 100 for b in bins], marker='o', label=label)
    axes[1].set_xticks(range(len(bins)), ['0-1', '1-5', '5-15', '15-60', '60+'])
    axes[1].set(xlabel='Birth radius (mm)', ylabel='Track efficiency (%)', title='Same 1500 validation events, thresholds 0.70 / 0.575', ylim=(0, 100))
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / 'training_and_tracking.png', dpi=160)
    plt.close(fig)
    lines = ['# 重启后no-time微调权重评估', '',
             '2026-10-14。仅使用1500个验证事件，独立测试组未评估；没有恢复或继续训练。', '',
             f'保存权重可完整加载，参数均有限。日志最后完整记录到第{last}轮；保存文件时间与第{best["epoch"]}轮对应，重新计算验证loss={fine["edge"]["loss"]:.8f}，与该轮日志一致。故当前评估的是第{best["epoch"]}轮最佳权重，而非最后一轮权重。', '',
             '## 完整寻迹结果', '',
             '两模型使用相同sigma=0.5mm测量d、相同完整事件、相同图和原两级寻迹。固定历史阈值0.70/0.575，严格/宽松拟合RMS上限为1.0/1.5mm，不为某个模型另调阈值；这不代表任一模型在该阈值下达到最优。', '',
             '| 指标 | 原始no-time | 微调no-time |', '|---|---:|---:|',
             f'| 寻迹效率 | {base["efficiency"]:.3%} | {fine["efficiency"]:.3%} |',
             f'| 假轨率 | {base["fake_rate"]:.3%} | {fine["fake_rate"]:.3%} |',
             f'| 克隆率 | {base["clone_rate"]:.3%} | {fine["clone_rate"]:.3%} |',
             f'| 找回真轨/可评估真轨 | {base["matched_tracks"]}/{base["truth_tracks"]} | {fine["matched_tracks"]}/{fine["truth_tracks"]} |',
             f'| 假轨数量 | {base["fake_tracks"]} | {fine["fake_tracks"]} |',
             f'| 克隆数量 | {base["clone_tracks"]} | {fine["clone_tracks"]} |',
             f'| 每事件假轨 | {base["fake_per_event"]:.3f} | {fine["fake_per_event"]:.3f} |', '',
             '真轨至少5hit；候选纯度至少0.8、完整度至少0.5。假轨率/克隆率分母都是输出径迹总数；寻迹效率分母是可评估真轨数。', '',
             f'效率提升{100 * (fine["efficiency"] - base["efficiency"]):.3f}个百分点。同事件成对bootstrap的95%区间为' +
             ' -- '.join(f'{100*x:.3f}' for x in report['efficiency_gain_95pct_event_bootstrap']) + '个百分点（1000次事件重抽样）。这是当前验证集上的区间，不是独立测试结论。', '',
             '## 按出生半径分层', '', '| 半径mm | 真轨数 | 原始效率 | 微调效率 |', '|---|---:|---:|---:|']
    for key in bins:
        b, f = base['by_truth']['radius_bin'][key], fine['by_truth']['radius_bin'][key]
        lines.append(f'| {key} | {b["truth"]} | {b["efficiency"]:.3%} | {f["efficiency"]:.3%} |')
    lines += ['', '## 按hit数量分层', '', '| hit数 | 真轨数 | 原始效率 | 微调效率 |', '|---|---:|---:|---:|']
    for key, b in base['by_truth']['hit_bin'].items():
        f = fine['by_truth']['hit_bin'][key]
        lines.append(f'| {key} | {b["truth"]} | {b["efficiency"]:.3%} | {f["efficiency"]:.3%} |')
    lines += ['', f'超过20hit的长轨仍是明显短板：微调后仅找回{fine["by_truth"]["hit_bin"]["21_plus"]["matched"]}/{fine["by_truth"]["hit_bin"]["21_plus"]["truth"]}，效率{fine["by_truth"]["hit_bin"]["21_plus"]["efficiency"]:.3%}。该分层不能独自判定问题来自边分类、组轨还是拟合，后续需要针对漏检做分解。', '', '## 训练及边分类', '',
              f'固定0.5边阈值下，边F1由{base["edge"]["at_0p5"]["f1"]:.3%}提升到{fine["edge"]["at_0p5"]["f1"]:.3%}。这不是寻迹效率。微调权重在已扫描0.30--0.95边阈值中以0.50取得最高边F1；边F1最优阈值不等于两级寻迹最优阈值。', '',
              f'第{best["epoch"]}轮验证loss最低；最后记录轮的验证loss为{history["Loss/validation"][-1]["value"]:.8f}。后期训练loss继续下降但验证loss未改善，不必因为未跑满200轮就假定必须立即恢复。下一步应先在验证组检查寻迹阈值和失败类别，再决定是否继续微调。', '',
              '## 范围和限制', '',
              '当前样本已通过Good真值预筛选，不含额外电子噪声；不能直接代表完整读出条件下的效率。与旧原点训练集约98%的结果存在输入物理域、距离误差、占用率差异，不能直接作同条件比较。尚未确认重启原因，也没有证据仅凭重启现象将其归因于训练负载。', '',
              '权重、概率缓存、逐事件指标和比较JSON都保存在本目录；原始与微调权重快照按SHA256命名，主模型权重和训练日志未修改。', '',
              '![训练曲线与分层效率](training_and_tracking.png)', '']
    (OUT / '重启后微调效果报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Report assertions passed; report and PNG saved', OUT)


if __name__ == '__main__':
    main()
