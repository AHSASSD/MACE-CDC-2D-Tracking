"""Render verified stage-two tracking measurements and bounded diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'whole_event_retraining'
OUT = DATA / 'stage2_no_time_assessment'


def main():
    report = json.loads((OUT / 'comparison.json').read_text(encoding='utf-8'))
    probes = json.loads((OUT / 'miss_probes.json').read_text(encoding='utf-8'))
    original = json.loads((DATA / 'interrupted_no_time_assessment/comparison.json').read_text(encoding='utf-8'))['summaries']['baseline']
    summaries = report['summaries']
    names = {'baseline': '第一阶段loss最佳（77轮）', 'finetuned': '第二阶段loss最佳（34轮）', 'last': '第二阶段末轮（62轮）', 'oracle': '真实连边诊断对照'}
    assert report['checkpoint']['epoch'] == 62 and report['checkpoint']['best_epoch'] == 34
    assert abs(summaries['finetuned']['edge']['loss'] - report['checkpoint']['best_loss']) < 1e-6
    assert probes['probability_signature']['weight_sha256'] == summaries['finetuned']['weight_sha256']
    for value in summaries.values():
        assert value['truth_tracks'] == 45165
        assert value['output_tracks'] == value['matched_tracks'] + value['clone_tracks'] + value['fake_tracks']
    ci = report['efficiency_gain_95pct_event_bootstrap']
    best = summaries['finetuned']
    baseline = summaries['baseline']
    history = report['training_history']
    losses = {r['epoch']: r['value'] for r in history['Loss/validation']}
    assert abs(losses[34] - best['edge']['loss']) < 1e-6
    lines = ['# 第二阶段no-time：两级寻迹评估与停滞诊断', '',
             '2026-10-14。1500个验证事件，45165条至少5hit真轨；独立1500测试事件未评估。未继续训练或修改主权重、两级寻迹算法。', '',
             f'第二阶段不是完全无改善：最佳权重效率从{baseline["efficiency"]:.3%}提高到{best["efficiency"]:.3%}，增加{100*(best["efficiency"]-baseline["efficiency"]):.3f}个百分点（净找回{best["matched_tracks"]-baseline["matched_tracks"]}条）。同事件成对bootstrap的95%区间为{100*ci[0]:.3f}～{100*ci[1]:.3f}个百分点；是验证组内的抽样区间，不是独立测试保证。', '',
             f'第62轮的寻迹效率反而更高，为{summaries["last"]["efficiency"]:.3%}；比第一阶段净多找回{summaries["last"]["matched_tracks"]-baseline["matched_tracks"]}条真轨。它的验证loss较差，所以没有成为主best文件；这是本次发现的训练选模目标与实际寻迹目标不一致。', '',
             '## 相同两级流程的实测', '',
             '固定0.70/0.575阈值，拟合RMS门槛1.0/1.5mm。真轨至少5hit，匹配要求纯度≥0.8、完整度≥0.5；假轨率和克隆率分母均为输出轨数。原始预训练模型在同一新验证组效率为'+f'{original["efficiency"]:.3%}。', '',
             '| 权重/对照 | 寻迹效率 | 假轨率 | 克隆率 | 找回真轨 |', '|---|---:|---:|---:|---:|']
    for name, value in summaries.items():
        lines.append(f'| {names[name]} | {value["efficiency"]:.3%} | {value["fake_rate"]:.3%} | {value["clone_rate"]:.3%} | {value["matched_tracks"]}/{value["truth_tracks"]} |')
    lines += ['', '真实连边对照把已有候选边的概率替换为真实标签；其余图、d和寻迹流程不变。它使用不可部署的真值信息，且全1边概率会改变排序、切树和层级拆分行为，因此是瓶颈诊断对照，不是数学意义上的最优效率上限。', '',
              '## 第62轮为什么不如继续下降的训练loss那样乐观', '',
              f'主best权重与完整checkpoint中的best_model逐张量一致；重新推理验证loss={best["edge"]["loss"]:.8f}，对应第34轮。第62轮训练loss={history["Loss/train"][-1]["value"]:.8f}、验证loss={losses[62]:.8f}。训练拟合仍在改善而验证损失回升，证据支持过拟合；保存器保留34轮而非62轮是正常行为。', '',
              f'边F1@0.5：第一阶段{baseline["edge"]["at_0p5"]["f1"]:.3%}、第二阶段最佳{best["edge"]["at_0p5"]["f1"]:.3%}、第62轮{summaries["last"]["edge"]["at_0p5"]["f1"]:.3%}。Focal loss、边F1和整条径迹指标不是同一目标；错误连边可合并两条轨，缺边可把轨迹打碎，随后还有拟合和hit冲突验收，不能从训练loss直接推出98%寻迹效率。', '',
              '## 漏检位置', '', '| 真轨hit数 | 数量 | 第一阶段 | 第二阶段最佳 | 真实连边对照 |', '|---|---:|---:|---:|---:|']
    for key, value in best['by_truth']['hit_bin'].items():
        lines.append(f'| {key} | {value["truth"]} | {baseline["by_truth"]["hit_bin"][key]["efficiency"]:.3%} | {value["efficiency"]:.3%} | {summaries["oracle"]["by_truth"]["hit_bin"][key]["efficiency"]:.3%} |')
    strict_best = next(r for r in best['edge']['thresholds'] if r['threshold'] == .7)
    strict_last = next(r for r in summaries['last']['edge']['thresholds'] if r['threshold'] == .7)
    lines += ['', f'实际首级阈值0.70下，正边召回率34轮为{strict_best["recall"]:.3%}、62轮为{strict_last["recall"]:.3%}，精确率分别{strict_best["precision"]:.3%}/{strict_last["precision"]:.3%}。这解释了为什么62轮在loss和F1@0.5较差时，固定两级流程仍能找回更多轨迹：生产工作点的连通性变化与0.5分类评价不同。该统计说明工作点变化，并非逐条漏检的独立因果证明。']
    long = best['by_truth']['hit_bin']['21_plus']
    lines += ['', f'长轨虽差，但只占{long["truth"]}/{best["truth_tracks"]}。即使单独补回所有长轨漏检、保持其他结果不变，总效率也只升到{(best["matched_tracks"]+long["truth"]-long["matched"])/best["truth_tracks"]:.3%}；因此不能把与旧98%的差距全部归因于长轨。']
    model_rows = json.loads((OUT / 'tracking_finetuned.json').read_text(encoding='utf-8'))['events']
    oracle_rows = json.loads((OUT / 'tracking_oracle.json').read_text(encoding='utf-8'))['events']
    model_misses = {tuple(k) for row in model_rows for k in row['missed_tracks']}
    oracle_misses = {tuple(k) for row in oracle_rows for k in row['missed_tracks']}
    lines += ['', f'最佳模型漏检{len(model_misses)}条，其中{len(model_misses-oracle_misses)}条在真实连边对照中恢复，{len(model_misses&oracle_misses)}条两者都未恢复；另有{len(oracle_misses-model_misses)}条仅对照失败。这一交叉统计同时说明连边和后处理的作用，也提醒不能将对照直接当上限。', '',
              f'真实连边下总体效率{summaries["oracle"]["efficiency"]:.3%}，说明当前图和两级流程能够接受绝大部分普通真轨，并没有统一卡在93%左右的几何门槛；主要改进空间在预测连边及其引起的组轨变化。长轨真实连边效率仍仅{summaries["oracle"]["by_truth"]["hit_bin"]["21_plus"]["efficiency"]:.3%}，同时存在后处理/几何瓶颈。真实标签不可观测，因此该结果不保证no-time网络一定能达到99%。', '',
              '固定随机种子20261014抽取200个验证事件；以下为规则判别的失败表象，分类有先后顺序，不能当作相互独立的因果贡献。', '', '| 漏检表象 | 条数 |', '|---|---:|']
    labels = {'complete_but_contaminated': '完整度够但纯度不足', 'edge_fragmentation_below_min_hits': '宽松连边仍碎成不足5hit', 'strict_threshold_fragmentation': '严格阈值碎裂', 'merged_with_other_tracks': '与其他真轨连在一起', 'geometry_or_candidate_rejection': '几何/候选拒绝', 'partial_candidate_loss': '候选部分丢失'}
    for key, count in probes['causes'].items():
        lines.append(f'| {labels[key]} | {count} |')
    lines += ['', '## 单圆拟合的独立检查', '',
              '把同一条真轨的全部hit交给原拟合器，分别使用测量d和d_true，不删hit；这比生产寻迹中允许取片段、删最多2hit更严格，不能将失败比例直接等同实际寻迹漏检率。', '',
              '| 已知真轨分组 | 真轨数 | RMS中位数mm | RMS>1.5mm |', '|---|---:|---:|---:|']
    for key, value in probes['truth_group_circle_fits'].items():
        lines.append(f'| {key} | {value["tracks"]} | {value["median_rms_mm"]:.4f} | {value["rms_gt_1p5mm"]} |')
    for hit_bin in ['5_20', '21_plus']:
        rows = [r for r in probes['alternate_initialization'] if r['hit_bin'] == hit_bin]
        if not rows:
            lines += ['', f'{hit_bin}在本次抽样中没有需要进一步检查初值的d_true拟合失败轨。']
            continue
        recovered = sum(r['alternate_rms_mm'] <= 1.5 for r in rows)
        lines += ['', f'对{hit_bin}的{len(rows)}条d_true拟合失败轨，用线性圆估计给出新初值并尝试3个半径初值，有{recovered}条降到RMS≤1.5mm，原中位数{np.median([r["original_rms_mm"] for r in rows]):.4f}mm，改初值后中位数{np.median([r["alternate_rms_mm"] for r in rows]):.4f}mm。普通轨仅选抽样中前200条失败轨，长轨检查该抽样所有失败轨；不是代表性总体成功率，也不是模型/生产算法已修复。']
    n_recovered_long = sum(r['hit_bin'] == '21_plus' and r['alternate_rms_mm'] <= 1.5 for r in probes['alternate_initialization'])
    lines += ['', f'同一hit、同一d_true，仅改优化初值有{n_recovered_long}条长轨通过门槛。仍不通过的轨迹不能据此认定为物理非圆，也可能还有左右漂移解、优化初值等问题；需进一步结合模拟坐标、动量演化核查。', '', '更正说明：2026-10-14后续核查发现旧的真轨全hit拟合探针混用了float32坐标和float64距离，会改变数值差分行为，放大部分RMS。本节已用全float64重新计算，旧版的“62条长轨全部失败、19条因初值恢复”统计撤回。实际两级寻迹效率、假轨率、克隆率、真实连边对照以及漏检分类不受此探针问题影响；寻迹算法和权重均未修改。']
    lines += ['', '## 为什么旧98%不能保证新数据98%', '',
              '原STEP1默认每图也混合约30条粒子，旧源粒子在原点出生；新数据含离轴、管壁附近衰变。旧距离误差σ=0.075mm，新文件已添加σ=0.5mm；旧数据还执行300ns窗口和100ns每Cell死时间处理，新完整事件没有这些处理。旧完整度分母基于电子学处理后hit，新组基于保留的原始筛选hit。这些都是已核查的任务差异，不能只归因于“随机混合变成完整事件”。', '',
              '此前核查旧源train/validation/test的EvtID集合互不相交，未发现用户担心的跨划分源粒子重用。因此旧98%不能被简单解释为题库泄漏，也不应据此保证新域必能达到98%。当前证据没有单独分配出生半径、误差、电子学和优化初值各自的因果贡献。', '',
              '## 后续建议', '',
              f'同时保留第34轮与第62轮作为候选，先暂停继续加轮次。第62轮效率{summaries["last"]["efficiency"]:.3%}和假轨率{summaries["last"]["fake_rate"]:.3%}在当前验证组优于34轮，但克隆率略高，不能仅凭验证loss回升就淘汰62轮。末轮权重只从完整checkpoint导出到诊断目录last_epoch_62.pth，主best文件仍是34轮。验证组校准两级阈值，并用整条径迹效率、假轨率和克隆率联合选择候选；阈值扫描和按寻迹指标选择训练checkpoint本次未实施。', '',
              '对已经定位的碎裂、混轨和长轨分别改进并做固定事件的成对实验：连边侧检查困难正边和短轨的训练覆盖；后处理侧检查圆拟合初值、层级拆分与长轨片段恢复。若要改网络，再与相同新数据、相同预算的从头训练作对照；本次尚不足以判定网络容量已经达到极限。', '',
              '当前Good、Axial、ExtendedDecayWithSpin筛选依赖真值，不含额外电子噪声；指标不能直接视为完整实验读出性能。没有评估time模型，没有在独立测试组调参或验收。', '',
              '![第二阶段曲线与寻迹对照](stage2_training_tracking.png)', '']
    (OUT / '第二阶段微调与两级寻迹诊断.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for tag, label in [('Loss/train', 'Training'), ('Loss/validation', 'Validation')]:
        axes[0].plot([r['epoch'] for r in history[tag]], [r['value'] for r in history[tag]], label=label)
    axes[0].axvline(34, color='black', linestyle='--', label='Best loss: epoch 34')
    axes[0].set(xlabel='Stage 2 epoch', ylabel='Focal loss', title='Validation rises after best checkpoint')
    axes[0].legend(fontsize=8)
    for name, label in [('baseline', 'Stage 1'), ('finetuned', 'Stage 2 best'), ('oracle', 'Truth-edge control')]:
        bins = ['5_6', '7_10', '11_20', '21_plus']
        axes[1].plot(range(4), [summaries[name]['by_truth']['hit_bin'][b]['efficiency'] * 100 for b in bins], marker='o', label=label)
    axes[1].set_xticks(range(4), ['5-6', '7-10', '11-20', '21+'])
    axes[1].set(xlabel='Hits per truth track', ylabel='Tracking efficiency (%)', ylim=(0, 101), title='Long tracks remain a weak point')
    axes[1].legend(fontsize=8)
    keys = ['baseline', 'finetuned', 'last', 'oracle']
    axes[2].bar(range(4), [summaries[k]['efficiency'] * 100 for k in keys])
    axes[2].set_xticks(range(4), ['Stage 1', 'Best 34', 'Last 62', 'Truth edges'])
    axes[2].set(ylabel='Tracking efficiency (%)', ylim=(0, 101), title='1500 identical validation events')
    for i, k in enumerate(keys):
        axes[2].text(i, summaries[k]['efficiency'] * 100 + 1, f'{summaries[k]["efficiency"]:.2%}', ha='center', fontsize=9)
    for axis in axes:
        axis.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / 'stage2_training_tracking.png', dpi=170)
    plt.close(fig)
    print('Report checks passed', OUT)


if __name__ == '__main__':
    main()
