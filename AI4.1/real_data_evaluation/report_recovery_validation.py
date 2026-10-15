"""Checked paired results, confidence interval and case images for recovery experiments."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_recovery_validation import summarize

OUT = ROOT / 'whole_event_retraining/recovery_validation'
LABELS = {'baseline': '原两级寻迹', 'extend_geometry': '几何延长', 'merge_geometry': '几何延长+合并', 'merge_time': '时间约束延长+合并', 'hough_geometry': '几何Hough兜底', 'hough_time': '时间分峰Hough兜底'}


def table(summaries):
    lines = ['| 方案 | 效率 | 假轨率 | 克隆率 | 找回 | 假轨 | 克隆 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in summaries.items():
        lines.append(f'| {LABELS[name]} | {s["efficiency"]:.3%} | {s["fake_rate"]:.3%} | {s["clone_rate"]:.3%} | {s["matched_tracks"]} | {s["fake_tracks"]} | {s["clone_tracks"]} |')
    return lines


def main():
    pilot = json.loads((OUT / 'pilot.json').read_text(encoding='utf-8'))
    confirm = json.loads((OUT / 'confirm.json').read_text(encoding='utf-8'))
    hp = json.loads((OUT / 'hough-pilot.json').read_text(encoding='utf-8'))
    hc = json.loads((OUT / 'hough.json').read_text(encoding='utf-8'))
    parent = pilot['selected_profile']
    selected = hp['selected_profile']
    assert not set(pilot['event_ids']) & set(confirm['event_ids'])
    assert set(hp['event_ids']) == set(pilot['event_ids']) and set(hc['event_ids']) == set(confirm['event_ids'])
    assert pilot['first_phase_function_sha256'] == confirm['first_phase_function_sha256'] == hp['first_phase_function_sha256'] == hc['first_phase_function_sha256']
    all_rows = pilot['events']+confirm['events']
    all_hough = hp['events']+hc['events']
    overall = {name: summarize(all_rows if name in ['baseline', parent] else all_hough, name) for name in list(dict.fromkeys(['baseline', parent, selected]))}
    reference_parent = {r['EvtID']: r['metrics'][parent] for r in all_rows}
    for row in all_hough:
        assert row['metrics'][parent] == reference_parent[row['EvtID']]
    rng = np.random.default_rng(20261014)
    rows = hc['events']
    truth = np.array([r['metrics']['baseline']['truth_tracks'] for r in rows])
    changes = np.array([r['metrics'][selected]['matched_tracks']-r['metrics']['baseline']['matched_tracks'] for r in rows])
    indices = rng.integers(0, len(rows), (1000, len(rows)))
    ci = np.quantile(changes[indices].sum(axis=1)/truth[indices].sum(axis=1), [.025, .975])
    lines = ['# 片段回收与Hough按顺序验证', '', '2026-10-14。第62轮no-time权重固定，0.70/0.575两级阈值、匹配纯度0.8/完整度0.5均未改变。', '',
             '203个验证事件用于选择，1297个验证事件用于内部确认；模型训练期间用过整个验证组，不能把该确认组宣称为新的独立测试。独立1500测试事件未使用。', '',
             '回收只读取观测量，不读取TrkID、t0、d_true、边标签；时间代理为观测t减去测量d/漂移速度。新增时间约束是额外后处理信息，改进结果不能称为全流程不使用时间。', '',
             '## 全1500验证事件', '']+table(overall)
    lines += ['', '## 第一阶段预实验（203事件）', '']+table(pilot['summaries'])
    lines += ['', '只做几何延长/合并会吸入附近轨迹hit，未通过假轨约束。预先规定：假轨和克隆数量均不增加时，选择找回真轨最多的方案；平局保留基线。', '',
              '## 第一阶段内部确认（1297事件）', '']+table(confirm['summaries'])
    lines += ['', f'选定{LABELS[parent]}：新增找回{confirm["summaries"][parent]["newly_recovered"]}条，同时新增丢失{confirm["summaries"][parent]["newly_lost"]}条。退化事件没有删除，逐事件missed_tracks可核查。', '',
              '## 第二阶段预实验（203事件）', '']+table(hp['summaries'])
    lines += ['', 'Hough先检查剩余hit，并尝试重分解观测时间跨度异常的候选。几何投票与时间分峰分别对照；选择规则相对冻结的第一阶段父方案要求假轨、克隆不增加。参数空间不要求轨迹经过原点。', '',
              '## 第二阶段内部确认（1297事件）', '']+table(hc['summaries'])
    s = hc['summaries'][selected]
    parent_summary = hc['summaries'][parent]
    parent_recovered = parent_lost = 0
    for row in rows:
        before = set(map(tuple, row['metrics'][parent]['missed_tracks']))
        after = set(map(tuple, row['metrics'][selected]['missed_tracks']))
        parent_recovered += len(before-after)
        parent_lost += len(after-before)
    lines += ['', f'选定{LABELS[selected]}：相对原两级基线新增找回{s.get("newly_recovered", 0)}条、新增丢失{s.get("newly_lost", 0)}条。成对事件bootstrap的效率增量95%区间为{100*ci[0]:.3f}～{100*ci[1]:.3f}个百分点（1000次重抽样）。该区间是当前确认组内的抽样区间，不包含参数选择和新数据域变化的全部不确定性。', '',
              f'相对第一阶段父方案，Hough确认组新增找回{parent_recovered}条、丢失{parent_lost}条，但假轨数量增加{s["fake_tracks"]-parent_summary["fake_tracks"]}、克隆增加{s["clone_tracks"]-parent_summary["clone_tracks"]}。混轨候选减少{parent_summary["mixed_candidates"]-s["mixed_candidates"]}，完整度不足片段增加{s["incomplete_candidates"]-parent_summary["incomplete_candidates"]}。因此Hough并未在确认组继续满足“假轨/克隆数量均不增加”，应作为可选的效率优先模式保留，不能将预实验的精度结论直接外推。默认绘图和正式STEP4仍不启用回收。', '',
              '## 第三阶段', '', '已准备STEP7_双专家验证.py及time专家训练命令，原始time权重通过新输入接口前向检查。尚未在新数据上训练time专家，因此未给出双专家性能结论。先验证空间/时间双专家的固定权重融合与两个单专家，再决定是否需要训练门控MoE。见回收实验与时间专家启动说明.md。', '',
              '## 限制', '', '原型尚未验收真实电子噪声、场/漂移速度失配以及独立测试性能。Hough有固定采样与分辨率上限，近重合且时间也相近的轨迹仍可能无法分开；多圈或回散射后的不同曲率轨迹仍受单圆输出和整粒子匹配口径约束。正式STEP4没有被替换，实验结果不能自动视为已部署。', '',
              '所有负面对照、退化事件和新增候选均保存在pilot/confirm/hough-pilot/hough.json；baseline_tracks.pt是签名校验后的原流程缓存。', '',
              '![验证对照](recovery_comparison.png)', '']
    (OUT / '回收验证报告.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps({'overall_validation': overall, 'confirmation': hc['summaries'], 'paired_efficiency_gain_ci95': ci.tolist(), 'third_stage': 'prepared, not trained or performance-validated'}, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    names = list(overall)
    captions = [{'baseline': 'Baseline', 'merge_time': 'Fragments + time', 'hough_time': 'Hough + time', 'hough_geometry': 'Hough geometry'}.get(name, name) for name in names]
    axes[0].bar(range(len(names)), [overall[n]['efficiency']*100 for n in names])
    axes[0].set(ylim=(0, 101), ylabel='Tracking efficiency (%)', title='1500 validation events; frozen no-time network')
    for i, name in enumerate(names):
        axes[0].text(i, overall[name]['efficiency']*100+1, f'{overall[name]["efficiency"]:.2%}', ha='center')
    axes[1].bar(np.arange(len(names))-.17, [overall[n]['fake_rate']*100 for n in names], width=.34, label='Fake rate')
    axes[1].bar(np.arange(len(names))+.17, [overall[n]['clone_rate']*100 for n in names], width=.34, label='Clone rate')
    axes[1].set(ylabel='Rate among output tracks (%)', title='Same truth-matching requirements')
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(len(names)), captions)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'recovery_comparison.png', dpi=160)
    plt.close(fig)
    print('Checked recovery report saved', json.dumps(overall, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
