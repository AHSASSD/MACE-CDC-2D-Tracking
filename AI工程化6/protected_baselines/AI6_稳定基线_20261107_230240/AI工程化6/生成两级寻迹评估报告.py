"""Summarize paired validation tracking and export independent hit identities."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import MODEL_ROOT, DATA_ROOT, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='local_evaluation_epoch051_20261106')
    parser.add_argument('--experiment', default='two_stage_validation_v1')
    args = parser.parse_args()
    out = MODEL_ROOT / args.snapshot / args.experiment
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    config = summary['config']
    events = [json.loads((out / 'events' / f'event_{eid:05d}.json').read_text(encoding='utf-8')) for eid in config['event_ids']]
    methods = summary['methods']
    names = ['simple_joint', 'axial_two_pass', 'stereo_assisted']
    labels = ['简单联合连通分量', 'AA两级寻迹', 'AA两级＋斜丝归属']
    # Paired bootstrap is a within-validation sampling diagnostic, not a test guarantee.
    denominator = np.array([event['methods']['simple_joint']['truth_tracks'] for event in events])
    matched = np.array([[event['methods'][name]['matched_tracks'] for event in events] for name in names])
    rng = np.random.default_rng(20261106)
    draws = rng.integers(len(events), size=(2000, len(events)))
    differences = (matched[1, draws].sum(1) - matched[0, draws].sum(1)) / denominator[draws].sum(1)
    interval = np.quantile(differences, [.025, .975])
    recovered = lost = 0
    for event in events:
        before = set(event['matched_truth_ids']['simple_joint'])
        after = set(event['matched_truth_ids']['axial_two_pass'])
        recovered += len(after - before); lost += len(before - after)
    save_json(out / 'paired_comparison.json', dict(recovered_truth_tracks=recovered, lost_truth_tracks=lost,
              net_gain=recovered - lost, efficiency_gain=methods[names[1]]['track_efficiency'] - methods[names[0]]['track_efficiency'],
              validation_paired_bootstrap_95_interval=interval.tolist(), draws=2000, seed=20261106))
    # Export HitID rather than collapsing repeated CellID measurements.
    with (out / 'joint_track_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for event in events:
            eid = event['event_id']
            with np.load(DATA_ROOT / 'events' / f'event_{eid:05d}.npz') as f:
                hit_ids, cells = f['hit_id'], f['cell_id']
            candidates = []
            for track in event['tracks']['stereo_assisted']:
                indices = np.asarray(track['indices'], dtype=int)
                ss = np.asarray(track.get('stereo_indices', []), dtype=int)
                candidates.append(dict(hit_ids=hit_ids[indices].tolist(), cell_ids=cells[indices].tolist(),
                                       stereo_hit_ids=hit_ids[ss].tolist(), axial_circle=track['fit'],
                                       axial_rms_mm=track['rms'], stage=track['stage']))
            stream.write(json.dumps(dict(event_id=eid, candidates=candidates)) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    english = ['Simple joint\ncomponents', 'Axial\ntwo-pass', 'Axial two-pass\n+ stereo']
    for ax, metric, title in zip(axes,
            ['track_efficiency', 'fake_fraction', 'stereo_recall_all_eligible_tracks'],
            ['Axial truth-track efficiency', 'Fake / output candidates', 'Correct stereo / all eligible truth stereo']):
        values = [methods[name][metric] for name in names]
        bars = ax.bar(english, values, color=['#94a3b8', '#2563eb', '#16a34a'])
        ax.bar_label(bars, labels=[f'{value:.2%}' for value in values], fontsize=9)
        ax.set(title=title, ylim=(0, 1.08), ylabel='Fraction')
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle(f"AI6 frozen epoch {config['selected_epoch']}: {summary['events']} paired validation events")
    fig.tight_layout(); fig.savefig(out / 'two_stage_comparison.png', dpi=180); plt.close(fig)
    rows = '\n'.join(f"| {label} | {methods[name]['track_efficiency']:.2%} | {methods[name]['fake_fraction']:.2%} | {methods[name]['clone_fraction']:.2%} |"
                     for label, name in zip(labels, names))
    assisted = methods['stereo_assisted']
    simple = methods['simple_joint']; axial = methods['axial_two_pass']
    assignment = {key: sum(event['stereo_assignment'][key] for event in events)
                  for key in ['direct_assignments', 'ss_assisted_assignments', 'unassigned_stereo']}
    text = f'''# AI6 联合模型嵌入两级寻迹：验证实验

使用固定第{config['selected_epoch']}轮权重，在{summary['events']}个完整验证事件上对比三种后处理。事件从原1500验证事件中以随机种子20261106无放回抽取，列表已固定；测试集没有参与本次阈值选择或试验。此结果用于可行性判断，不是独立测试或最终收敛结论。服务器训练继续推进，当前实验不改训练配置。

## 同口径结果

三种方案使用同一次联合模型预测，将双向边概率取均值，所有输入hit、候选边及事件保持一致。合格真轨要求至少5个直丝hit；直丝匹配要求纯度>=80%、完整度>=50%。效率以共同的{axial['truth_tracks']:,}条真轨为分母；假轨/克隆比例以各方案输出候选为分母。

| 方案 | 直丝真轨匹配效率 | 假候选比例 | 克隆比例 |
|---|---:|---:|---:|
{rows}

两级流程比简单连通分量净增加{recovered - lost:,}条匹配真轨（新增找回{recovered:,}条、同时失去{lost:,}条），效率变化{(axial['track_efficiency'] - simple['track_efficiency']) * 100:+.2f}个百分点。同事件成对bootstrap的95%区间为{interval[0] * 100:+.2f}至{interval[1] * 100:+.2f}个百分点，仅反映本验证样本内抽样波动。

两级＋斜丝方案保留AA两级的直丝成员，因此两者直丝效率相同是实现约定，不是斜丝改善直丝效率的证据。简单连通分量使用0.5阈值；两级使用0.70/0.575阈值及1.0/1.5mm圆拟合门槛，整体差异同时包含阈值和几何后处理作用，本实验未分离各环节贡献。

## 斜丝归属

| 指标 | 两级＋斜丝归属 |
|---|---:|
| 已匹配真轨候选内斜丝纯度 | {assisted['stereo_purity_matched']:.2%} |
| 已匹配真轨候选内斜丝完整度 | {assisted['stereo_completeness_matched']:.2%} |
| 全部合格真轨斜丝hit的正确找回比例 | {assisted['stereo_recall_all_eligible_tracks']:.2%} |
| 直丝和斜丝同时达标的真轨比例 | {assisted['joint_track_efficiency']:.2%} |
| AS直接归属hit数 | {assignment['direct_assignments']:,} |
| SS辅助补充hit数 | {assignment['ss_assisted_assignments']:,} |
| 保留未分配的斜丝hit数 | {assignment['unassigned_stereo']:,} |

“同时达标”分母为至少5个直丝hit且至少3个斜丝hit的{assisted['stereo_eligible_tracks']:,}条真轨，要求直丝和斜丝各自纯度>=80%、完整度>=50%。这是成员关联指标，不是三维拟合成功率。已匹配候选内的斜丝指标是条件指标，应结合覆盖全部合格真轨的斜丝找回比例阅读。

## 实现范围

1. 整个事件的直丝、斜丝先共同进入256隐藏维、8次消息计算的AI6模型，随后只取AA边建立直丝核心。
2. 复用冻结的AI4两级算法：严格首级、层级拆分、弱树边拆分、最多剔除2个异常hit，以及剩余hit宽松恢复。拟合器仍采用原8初值及soft_l1漂移圆拟合，输入统一float64。
3. 斜丝AS归属要求至少两个不同CellID的直丝支持，取最高两条AS概率均值>=0.70，且比其他合格候选领先至少0.10。每个独立hit最多属于一条轨迹。
4. SS只做一次补充：至少两个不同CellID的已归属斜丝支持，并有AS弱证据；分数为0.75×AS最高两边均值＋0.25×SS最高两边均值，至少0.65，竞争差至少0.10。不会沿SS边无限扩展，也不会合并两个直丝核心。
5. 对整根有限斜丝计算其XY投影与已拟合圆的最小可能距离，并要求不超过测量d＋2mm。它是保守几何必要条件，不是三维残差验收；没有把斜丝参考XY当成真实击中点。
6. TrkID只在全部候选重建后进入评分。不读取任何时间字段，不用真实位置/动量，也不按CellID合并独立HitID。

当前尚未包含纵向参数、左右解/多圈分支联合求解或完整有限丝段三维距离拟合。这里验证的是“二维直丝核心＋联合网络引导斜丝归属”。斜丝尚不反过来恢复新的直丝核心。

## 复测与文件

权重SHA256：`{config['weight_sha256']}`。experiment_contract.json保存事件列表、全部阈值、几何精度、数据集及源代码哈希。预测缓存只包含边与概率，可在相同权重下复用；新权重使用独立缓存。

后续模型先下载成新的冻结快照目录（含权重、model_metadata_joint_no_time.json和best_model_selection.json），然后执行相同命令替换快照名即可。默认随机种子、事件及配置一致，便于跨模型比较。新模型是否收敛应依据验证曲线判断，不能由日期或训练轮数保证。

```powershell
& 'D:\\amancoda\\envs\\pytorch\\python.exe' '.\\STEP5_两级寻迹.py' --snapshot 新快照目录名 --events 200 --workers 2
& 'D:\\amancoda\\envs\\pytorch\\python.exe' '.\\生成两级寻迹评估报告.py' --snapshot 新快照目录名
```

- summary.json：汇总与完整约定；paired_comparison.json：逐真轨得失及成对区间。
- events/：逐事件三种评分、直丝候选、斜丝归属和算法诊断。
- joint_track_candidates.jsonl：按事件导出的独立HitID、CellID、斜丝HitID、直丝圆参数及RMS。
- two_stage_comparison.png：同口径比较图。

该实验与之前1500测试事件上的31.27%诊断具有不同事件、概率对称化和80%/50%匹配口径，不能将数值直接相减。本文三种方案之间才是成对比较。

![两级寻迹比较](two_stage_comparison.png)
'''
    (out / '两级寻迹可行性评估.md').write_text(text, encoding='utf-8')
    print(out / '两级寻迹可行性评估.md')


if __name__ == '__main__':
    main()
