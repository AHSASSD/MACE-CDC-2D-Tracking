"""汇总冻结权重成对诊断，不修改默认配置。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "paired_diagnosis"
PARENT = OUT.parent
VARIANTS = ["no-time", "time"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summaries, metric_rows = {}, []
    for path in sorted(OUT.glob("n*_*_seed*/**/*.json")):
        report = read(path)
        if "scenario" not in report:
            continue
        summaries[f"{path.parent.name}/{path.name}"] = report["summary"]
        metric_rows.append({"events": len(report["events"]), "model": report["variant"],
                            "scenario": report["scenario"], "context": report["context"], "seed": report["seed"],
                            **{k: v for k, v in report["summary"].items() if not isinstance(v, dict)}})
    pd.DataFrame(metric_rows).to_csv(OUT / "paired_metrics.csv", index=False, encoding="utf-8-sig")
    source = read(OUT / "source_physics_comparison.json")
    probe = read(OUT / "origin_translation_probe.json")
    circle = read(OUT / "circle_shape_summary.json")
    physics = pd.read_csv(OUT / "new_particle_physics.csv")
    for variant in VARIANTS:
        report = read(PARENT / f"{variant}_historical_frozen_all.json")
        missed = {(e["EvtID"], int(k[1])) for e in report["events"] for k in e["missed_tracks"]}
        physics[variant] = [(int(e), int(t)) not in missed for e, t in zip(physics.EvtID, physics.TrkID)]
        sample = read(OUT / "n200_single_seed20261013" / f"{variant}_raw.json")
        expected = {e["EvtID"]: e for e in report["events"]}
        assert all(expected[e["EvtID"]]["matched_tracks"] == e["matched_tracks"] and
                   expected[e["EvtID"]]["fake_tracks"] == e["fake_tracks"] for e in sample["events"])
    physics["radius_bin"] = pd.cut(physics.birth_radius_mm, [-1, 1, 5, 15, 60, 100],
                                    labels=["0_1", "1_5", "5_15", "15_60", "60_plus"])
    bins = physics.groupby("radius_bin", observed=True).agg(truth=("TrkID", "size"),
          no_time=("no-time", "mean"), time=("time", "mean"), mean_hits=("hit_count", "mean"))
    ordinary = physics.loc[physics.hit_count.between(7, 20)].groupby("radius_bin", observed=True).agg(
          truth=("TrkID", "size"), no_time=("no-time", "mean"), time=("time", "mean"))
    bins.to_csv(OUT / "efficiency_by_birth_radius.csv", encoding="utf-8-sig")
    ordinary.to_csv(OUT / "ordinary_efficiency_by_birth_radius.csv", encoding="utf-8-sig")
    missed_share = {}
    for variant in VARIANTS:
        failed = physics.loc[~physics[variant]]
        missed_share[variant] = {"total_missed": len(failed),
                                "missed_birth_radius_over_15": int(failed.birth_radius_mm.gt(15).sum()),
                                "fraction_missed_radius_over_15": float(failed.birth_radius_mm.gt(15).mean())}
    complete = {v: {s: read(OUT / "n1000_single_seed20261013" / f"{v}_{s}.json")["summary"]
                   for s in ["deadtime_only", "full_old_contract"]} for v in VARIANTS}
    raw = {v: read(PARENT / f"{v}_historical_frozen_all.json")["summary"] for v in VARIANTS}
    result = {"source_physics": source, "efficiency_by_birth_radius": bins.to_dict("index"),
              "ordinary_efficiency_by_birth_radius": ordinary.to_dict("index"), "missed_share": missed_share,
              "full_batch_electronics": complete, "raw_fixed_threshold": raw,
              "diagnostic_cases": summaries, "origin_probe": probe, "circle_shapes": circle,
              "conclusion": "Primary observed generalization failure is concentrated in displaced/large-impact-parameter trajectories, absent or scarce in old training; electronics changes explain only part; upstream simulation causes not independently controlled"}
    (OUT / "diagnosis_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = lambda x: f"{x * 100:.3f}%"
    lines = ["# 寻迹效率下降归因报告", "", "2026-10-13。所有诊断使用旧冻结权重、0.70/0.575 阈值及原两级寻迹，未重训、未修改生产默认配置。",
             "", "## 结论", "", "主要失配集中在新批次的偏心出生/大冲击参数径迹。旧筛选粒子的出生点全部为原点，"
             "新粒子的横向出生半径覆盖约 0--70 mm，进入探测器时的轨迹圆相对原点的几何形状分布明显改变。"
             "新批次近中心径迹仍接近 98%；远离中心的径迹明显下降。电子学、测量误差及较低占用率都不足以单独解释整个降幅。",
             "", "这是对模型失败所在物理域的定位，尚不是对模拟器中产生点、材料、磁场等上游配置的独立因果拆分。"
             "出生半径、z、飞行历史相互关联，不能把全部差值归到一个参数。",
             "", "## 1. 可复现的固定条件", "", "主评估为同一新批次 1000 个完整事件、10927 条源可评估真轨。"
             "单因素筛查从 800 个开发事件中固定抽取 200 个事件（抽样种子 73013）；噪声/误差种子为 20261013+1009*EvtID。"
             "全旧处理条件与原 common.prepare_trigger 对相同随机种子逐列完全一致，已通过自检。"
             "raw 子集与上一轮缓存报告的逐事件匹配/假轨计数一致。",
             "", "固定观察口径：物理可见率=观测后 >=5 hit 真轨/源 >=5 hit 真轨；条件算法效率=成功匹配/观测可评估真轨；"
             "源端到端效率=按观测匹配条件成功匹配/固定源可评估真轨。额外报告源 hit 完整度：候选纯度 >=0.8、"
             "重合 hit/源完整 hit >=0.5，源真轨 >=5 hit。后两种口径的分母始终固定，不以删点提高分数。",
             "", "## 2. 1000 事件成对电子学实验", "", "| 模型 | 输入 | 观测可评估真轨 | 条件算法效率 | 源端到端效率 | 固定源 hit 完整度效率 | 假轨/事件 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        r = raw[v]
        lines.append(f"| {v} | 原始 hit | 10927 | {pct(r['track_efficiency'])} | {pct(r['track_efficiency'])} | {pct(r['track_efficiency'])} | {r['fake_per_event']:.3f} |")
        for s, label in [("deadtime_only", "仅100ns死时间"), ("full_old_contract", "完整旧电子学契约")]:
            m = complete[v][s]
            lines.append(f"| {v} | {label} | {m['observable_truth_tracks']} | {pct(m['conditional_algorithm_efficiency'])} | "
                         f"{pct(m['source_end_to_end_efficiency'])} | {pct(m['fixed_source_hit_efficiency'])} | {m['fake_per_event']:.3f} |")
    lines += ["", "死时间可减少假轨、部分改善 time，但完整旧处理后仍未恢复旧测试约98%的水平。"
              "完整旧处理淘汰 5819 个真实 hit，25 条源真轨变成少于5 hit；因此必须同时看源分母。",
              "", "## 3. 200 事件单因素筛查", "", "| 输入 | no-time 源效率 | time 源效率 |", "|---|---:|---:|"]
    labels = {"raw": "原始", "deadtime_only": "仅死时间", "noise_only": "仅100kHz噪声", "distance_only": "仅0.075mm距离误差",
              "time_only": "仅3ns时间误差及窗口裁切", "measurement_only": "距离+时间误差", "full_without_deadtime": "噪声+误差，无死时间", "full_old_contract": "完整旧处理"}
    for s, label in labels.items():
        m = [read(OUT / "n200_single_seed20261013" / f"{v}_{s}.json")["summary"] for v in VARIANTS]
        lines.append(f"| {label} | {pct(m[0]['source_end_to_end_efficiency'])} | {pct(m[1]['source_end_to_end_efficiency'])} |")
    lines += ["", "时间误差条件包含原协议的 [0,300)ns 裁切，因此 no-time 也会受到删除越窗 hit 的影响；"
              "这不是把时间特征误传给 no-time。所有条件共享源 hit、误差随机数与噪声抽样，不能将不同条件的效应简单相加。",
              "", "## 4. 提高新事件占用率没有恢复旧表现", "", "在同一200个锚事件上，每个锚事件加入两个新批次其他事件，"
              "保持锚粒子不变，仅改变周围径迹。每图约33条真轨，完整旧处理后平均361节点、平均入度39，接近但并非精确等于旧训练的318节点/入度35。"
              "这是诊断用人工上下文，不作为真实新事件评估；加入的伙伴事件会重复出现，故只比较不重复的锚粒子。",
              "", "| 模型 | 单事件原始锚效率 | 三事件原始锚效率 | 单事件完整旧处理锚效率 | 三事件完整旧处理锚效率 |", "|---|---:|---:|---:|---:|"]
    for v in VARIANTS:
        get = lambda context, s: read(OUT / f"n200_{context}_seed20261013" / f"{v}_{s}.json")["summary"]["anchor_end_to_end_efficiency"]
        lines.append(f"| {v} | {pct(get('single','raw'))} | {pct(get('triple','raw'))} | {pct(get('single','full_old_contract'))} | {pct(get('triple','full_old_contract'))} |")
    lines += ["", "增加上下文加重了混轨和碎片化，未恢复旧表现。此实验排除了简单补足图规模的修复思路，"
              "不表示占用率或 sum 聚合在任何条件下都无影响。",
              "", "## 5. 旧、新粒子的实际物理域变化", "", "旧源记录与原划分元信息对应的筛选行数同为81883；"
              "旧8109条源真轨的 x0=(0,0,0)。新10927条真轨横向出生半径最小0.078mm、最大70.222mm，z从约-985到1469mm。",
              "", "初始动能均值约37.10/37.39MeV，KS距离0.033；初始方向 cos(theta) KS距离0.038。"
              "相比之下出生半径KS距离为1.0，z为0.573。出生至首个有效直丝hit的能量损失中位数为0.185/0.398MeV，"
              "均值0.311/1.139MeV，说明到达有效直丝hit前的历史也不同；该量包含材料损失、路径和有效hit选择效应，不能独自证明材料配置变了。",
              "", "同一新批次、同一模型的分层效率：", "", "| 横向出生半径(mm) | 真轨数 | no-time | time |", "|---|---:|---:|---:|"]
    for b, m in bins.iterrows():
        lines.append(f"| {b} | {int(m.truth)} | {pct(m.no_time)} | {pct(m.time)} |")
    lines += ["", "限制到7--20 hit后，60mm以上出生半径仍为 no-time83.816%、time84.597%；"
              "1--5mm分别为98.691%、97.677%。这不是只由超长轨组成不同造成。"]
    for v in VARIANTS:
        m = missed_share[v]
        lines.append(f"- {v}：{m['total_missed']}条漏检中，{m['missed_birth_radius_over_15']}条出生半径>15mm，占{pct(m['fraction_missed_radius_over_15'])}。")
    lines += ["", "出生半径>15mm的真轨占全批约42.18%；漏检向这一群体显著集中。这是关联证据，不能称为独立的因果贡献百分比。",
              "", "## 6. 在真值单轨上排除其他粒子干扰", "", "将200事件中的2153条5--20 hit真轨分别建图，"
              "此时所有边都是同轨正边，没有噪声和其他粒子竞争。仍然观察到偏心轨正边概率明显偏低：",
              "", "| 模型 | 5--15mm出生半径正边召回@0.70 | >=60mm出生半径正边召回@0.70 |", "|---|---:|---:|"]
    for v in VARIANTS:
        q = probe["models"][v]["original_coordinates"]["by_birth_radius"]
        lines.append(f"| {v} | {pct(q['5_15']['positive_edge_recall_070'])} | {pct(q['60_plus']['positive_edge_recall_070'])} |")
    lines += ["", "这进一步定位到旧分类器对新轨迹几何域的适配不足，而非全图噪声或其他粒子竞争单独造成。"
              "但真值分组是离线诊断，不能作为可部署寻迹流程，也不能将正边召回当作完整寻迹效率。",
              "", "另一个离线实验保持单轨图拓扑、d、时间和标签不变，将全部wire坐标平移使该真轨出生点位于原点。"
              "预测并未改善。该平移同时改变了相对探测器层的坐标分布，并不能还原模拟器中的原点出生历史；"
              "因此不支持用简单坐标平移修复，更不能将真值x0提供给在线模型。",
              "", "## 7. 实际圆形状与拟合", "", "对旧验证源粒子和新成对子集的7--20 hit真轨直接拟合漂移圆，"
              "原点冲击参数定义为 |sqrt(a²+b²)-R|。所有拟合均成功返回有限值：",
              "", "| 样本 | 真轨数 | 冲击参数中位数(mm) | 90%分位(mm) | 冲击参数>15mm比例 | RMS中位数(mm) |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in circle.items():
        lines.append(f"| {name} | {m['tracks']} | {m['origin_impact_quantiles_mm'][2]:.3f} | {m['origin_impact_quantiles_mm'][3]:.3f} | {pct(m['fraction_impact_over_15'])} | {m['median_rms_mm']:.4f} |")
    lines += ["", "旧、新普通真轨的拟合RMS分布相近，圆相对原点的冲击参数分布却明显改变。"
              "结合真值边诊断能找回99.09%，普通径迹的主要问题位于学习到的连边，长轨仍有独立的拟合/组轨限制。",
              "", "## 8. 对训练与下一步的影响", "", "训练数据应覆盖目标场景的出生位置、位移/冲击参数及到达探测器前的真实飞行历史，"
              "以完整事件划分并保持目标电子学协议。先做同架构重训/微调对照，按出生半径、冲击参数、长度和整体假轨率验收；"
              "不能删除远离原点的hit来恢复98%，不能把x0或TrkID作为在线输入，也不建议继续只调全局边阈值。",
              "", "需要新旧模拟配置进一步确认：产生位置、母粒子的运动状态、靶材、磁场和材料路径是否改变。"
              "本轮能够定位到失败的物理域与连边机制，不能用观察数据独自判断上游哪项变化造成了这些位移轨。",
              "", "## 复现文件", "", "- paired_input_diagnosis.py：200事件八条件单因素、三事件上下文、1000事件关键条件确认。",
              "- compare_source_physics.py：原始物理分布及出生到首个有效直丝hit能量损失。",
              "- origin_translation_probe.py：仅供离线的真轨分组与平移诊断。",
              "- circle_shape_diagnosis.py：3368条已知成员普通径迹的圆形状诊断。",
              "- diagnosis_summary.json、paired_metrics.csv、efficiency_by_birth_radius.csv 保存可复核数值。",
              "", "本轮没有重训或修改模型架构，默认寻迹配置未改变。随机误差/噪声仅使用一个共同种子；"
              "小于约一个百分点的单项差异仍需多种子验证，较大的物理域差异与漏检集中现象需要新的模拟对照确认因果。"]
    (OUT / "效率下降归因报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    old = pd.read_csv(OUT / "old_particle_physics.csv")
    new = physics
    for label, values, color in [("Old source", old.birth_radius_mm, "#8796a8"), ("New source", new.birth_radius_mm, "#1a80b6")]:
        x = np.sort(values.to_numpy())
        axes[0].plot(x, np.arange(1, len(x) + 1) / len(x), label=label, color=color, linewidth=2)
    axes[0].set(xlabel="Birth radius (mm)", ylabel="Cumulative fraction", title="Source position distribution")
    axes[0].legend(fontsize=8)
    x = np.arange(len(ordinary))
    for key, label, color in [("no_time", "no-time", "#2367ac"), ("time", "time", "#df792c")]:
        axes[1].plot(x, ordinary[key].to_numpy() * 100, marker="o", label=label, color=color)
    axes[1].set_xticks(x, ["<1", "1-5", "5-15", "15-60", ">=60"])
    axes[1].set(xlabel="Birth radius (mm)", ylabel="Tracking efficiency (%)", title="New batch: 7-20 hit truth tracks", ylim=(75, 100))
    axes[1].legend(fontsize=8)
    for v, color in [("no-time", "#2367ac"), ("time", "#df792c")]:
        values = [raw[v]["track_efficiency"], complete[v]["deadtime_only"]["source_end_to_end_efficiency"],
                  complete[v]["full_old_contract"]["source_end_to_end_efficiency"]]
        axes[2].plot(range(3), np.array(values) * 100, marker="o", color=color, label=v)
        for i, value in enumerate(values):
            axes[2].annotate(f"{value * 100:.2f}", (i, value * 100), xytext=(0, 7 if v == 'no-time' else -14), textcoords="offset points", ha="center", fontsize=8)
    axes[2].set_xticks(range(3), ["Raw", "Dead time", "Full contract"])
    axes[2].set(ylabel="Source efficiency (%)", title="Paired electronics: 1000 events", ylim=(90, 94))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "diagnosis_overview.png", dpi=180)
    plt.close(fig)
    print({"missed_share": missed_share, "full_confirmed": {v: complete[v]["full_old_contract"]["conditional_algorithm_efficiency"] for v in VARIANTS}})


if __name__ == "__main__":
    main()
