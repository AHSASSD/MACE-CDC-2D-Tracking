"""先 lock（只看开发集），再运行保留集评估，最后 report。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from STEP4_真实数据评估 import summarize

OUT = Path(__file__).resolve().parent
VARIANTS = ["no-time", "time"]
NAMES = ["higher_threshold", "short_core", "residual_recursive", "conservative_residual"]


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def paired(base, improved):
    base_by_id = {r["EvtID"]: r for r in base}
    assert set(base_by_id) == {r["EvtID"] for r in improved}
    recovered = lost = 0
    for row in improved:
        old_missed = set(map(tuple, base_by_id[row["EvtID"]]["missed_tracks"]))
        new_missed = set(map(tuple, row["missed_tracks"]))
        recovered += len(old_missed - new_missed)
        lost += len(new_missed - old_missed)
    return {"recovered_truth_tracks": recovered, "newly_lost_truth_tracks": lost}


def acceptable(base, candidate):
    return (candidate["matched_tracks"] > base["matched_tracks"]
            and candidate["fake_rate"] <= base["fake_rate"]
            and candidate["clone_rate"] <= base["clone_rate"])


def lock():
    target = OUT / "selected_config.json"
    if target.exists():
        raise FileExistsError("已有锁定配置，请勿根据保留结果重新选规则")
    models = {}
    for variant in VARIANTS:
        baseline = load(f"{variant}_baseline.json")
        dev = [r for r in baseline["events"] if r["partition"] == "development"]
        base_summary = summarize(dev)
        choices = []
        comparisons = {}
        for name in NAMES:
            path = OUT / f"{variant}_{name}_development.json"
            if not path.exists():
                continue
            report = load(path.name)
            assert all(r["partition"] == "development" for r in report["events"])
            summary = report["summary"]
            comparisons[name] = {"metrics": summary, **paired(dev, report["events"]),
                                 "passes_development_gate": acceptable(base_summary, summary)}
            if acceptable(base_summary, summary):
                choices.append(report)
        selected = max(choices, key=lambda r: r["summary"]["matched_tracks"]) if choices else baseline
        models[variant] = {"recipe": selected["recipe"], "development_baseline": base_summary,
                           "development_candidates": comparisons}
    config = {"locked_date": "2026-10-13", "selection_data": "development events only",
              "selection_criteria": "efficiency increases; fake/clone rates do not increase; denominator all outputs",
              "input_stamp": load("input_manifest.json")["stamp"], "models": models}
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print({v: models[v]["recipe"] for v in VARIANTS})


def pct(value):
    return f"{value * 100:.3f}%" if value is not None else "N/A"


def make_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    config = load("selected_config.json")
    manifest = load("input_manifest.json")
    results = {}
    table_rows = []
    lines = ["# 新批次 test.xlsx 寻迹评估与改进", "", "评估日期：2026-10-13。",
             "", "输入由用户确认独立于旧训练数据。筛选 Good=True、WireType=Axial、CreatProc=ExtendedDecayWithSpin，"
             "每个 EvtID 整体处理，不混合事件。原表 280275 行，筛选后 116935 hit、1000 事件、10927 条至少 5 hit 的真实径迹。"
             "保留重复 CellID 的命中，不添加噪声、高斯测量误差，不执行额外死时间抑制。",
             "", "time 模型按训练约定使用逐行 t-t0 / 300；全体时间在 [0,300) ns 内。模型不接收 TrkID、Good 或 CreatProc；"
             "后二者仅用于输入选择，TrkID 仅用于离线评价。",
             "", "开发/保留事件为 800/200，固定种子 20261013；切片事件 0/1/2 已用于诊断，固定留在开发集。"
             "锁定配置只根据开发结果选择，以下保留结果不用于继续修改参数。",
             "", "## 主评估口径", "", "真实径迹身份为 (EvtID,TrkID)，真轨至少 5 hit。候选纯度 >=80%、完整度 >=50% 才能成功匹配。"
             "同一真轨首次成功匹配计有效重建，其余合格候选计克隆，其他计假轨。效率=不同成功真轨/合格真轨；"
             "假轨率、克隆率各以所有输出候选为分母。这一假轨定义也包含纯度高但完整度不足的真实径迹碎片，不能全都解释为无物理来源的虚假圆。",
             "", "## 200 事件保留集结果", "", "| 模型 | 流程 | 真轨/匹配 | 效率 | 假轨率 | 克隆率 | 假轨/事件 | 克隆/事件 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        baseline = load(f"{variant}_baseline.json")
        recipe = config["models"][variant]["recipe"]
        name = recipe["name"]
        dev_report = load(f"{variant}_{name}_development.json") if name != "baseline" else baseline
        hold_report = load(f"{variant}_{name}_holdout.json") if name != "baseline" else baseline
        combined = ([r for r in dev_report["events"] if r["partition"] == "development"]
                    + [r for r in hold_report["events"] if r["partition"] == "holdout"])
        assert len(combined) == manifest["events"] and len({r["EvtID"] for r in combined}) == len(combined)
        improved = {"variant": variant, "recipe": recipe, "summary": summarize(combined), "events": combined}
        (OUT / f"{variant}_selected_all.json").write_text(json.dumps(improved, ensure_ascii=False, indent=2), encoding="utf-8")
        partitions = {}
        for part in ["development", "holdout", "all"]:
            before = baseline["events"] if part == "all" else [r for r in baseline["events"] if r["partition"] == part]
            after = combined if part == "all" else [r for r in combined if r["partition"] == part]
            b, a = summarize(before), summarize(after)
            partitions[part] = {"baseline": b, "selected": a, **paired(before, after)}
            for label, metrics in [("baseline", b), (name, a)]:
                table_rows.append({"model": variant, "partition": part, "pipeline": label,
                                   **{k: v for k, v in metrics.items() if k not in ["by_truth_hits", "miss_causes"]}})
        hold = partitions["holdout"]
        for label, m in [("原基线", hold["baseline"]), ("保守恢复", hold["selected"])]:
            lines.append(f"| {variant} | {label} | {m['truth_tracks']}/{m['matched_tracks']} | {pct(m['track_efficiency'])} | "
                         f"{pct(m['fake_rate'])} | {pct(m['clone_rate'])} | {m['fake_per_event']:.4f} | {m['clone_per_event']:.4f} |")
        accepted = acceptable(hold["baseline"], hold["selected"])
        config["models"][variant]["holdout_accepted"] = accepted
        results[variant] = {"recipe": recipe, "holdout_accepted": accepted, "partitions": partitions}
    lines += ["", "保留集逐轨对比：", ""]
    for v, r in results.items():
        h = r["partitions"]["holdout"]
        lines.append(f"- {v}：新增找回 {h['recovered_truth_tracks']} 条，新丢失 {h['newly_lost_truth_tracks']} 条；"
                     f"保留集验收{'通过' if r['holdout_accepted'] else '未通过，默认入口回退原基线'}。")
    lines += ["", "## 全部 1000 事件结果（包含已用于选规则的开发集）", "",
              "此表用于描述整批数据，不应当作独立泛化验证；独立比较以上述保留集为准。", "",
              "| 模型 | 流程 | 匹配/10927 | 效率 | 假轨数/率 | 克隆数/率 |", "|---|---|---:|---:|---:|---:|"]
    for v, r in results.items():
        for label, m in [("原基线", r["partitions"]["all"]["baseline"]), ("保守恢复", r["partitions"]["all"]["selected"])]:
            lines.append(f"| {v} | {label} | {m['matched_tracks']} | {pct(m['track_efficiency'])} | "
                         f"{m['fake_tracks']}/{pct(m['fake_rate'])} | {m['clone_tracks']}/{pct(m['clone_rate'])} |")
    lines += ["", "## 为什么改这些", "", "冻结基线的全批次漏检诊断（启发式分类，并非因果证明）：", ""]
    for v, r in results.items():
        lines.append(f"- {v}：{r['partitions']['all']['baseline']['miss_causes']}。")
    lines += ["", "原拆分只试最弱 5 条树边的 1/2 边切，并要求切出子块立即通过几何拟合；"
              "多个真轨混合时，第一刀的子块可能仍混合，从而没有机会继续拆。新增恢复冻结首轮径迹，"
              "只处理剩余点，以 0.70 连边、0.20 mm 残差接受、最多 4 层和 40 次候选检查、最弱 3 边递归拆分。"
              "开发选择的最终平均边概率下限为 0.92；没有使用真值选择推理时的 hit，也没有重训权重。",
              "", "提高阈值、短核心，以及概率下限 0.80 的宽松恢复均因开发假轨率上升未采用。"
              "0.92 是开发集校准参数，不是预先指定的参数；它在保留集锁定验证。",
              "", "## 长径迹限制", "", "| 模型 | 基线真轨 hit 范围 | 真轨数 | 匹配数 | 效率 |", "|---|---|---:|---:|---:|"]
    for v, r in results.items():
        for b, m in r["partitions"]["all"]["baseline"]["by_truth_hits"].items():
            lines.append(f"| {v} | {b} | {m['truth']} | {m['matched']} | {pct(m['efficiency'])} |")
    lines += ["", "超过 20 hit 的 114 条真轨是明显弱项；最长有 238 hit。单圆拟合能力、碎片完整度门槛及模型连边均可能影响，"
              "本次没有修改 TrkID 或真值定义来提高分数。不能据此直接断言所有长轨都不符合圆模型。",
              "", "## 文件与复现", "", "- `STEP4_真实数据评估.py`：筛选、建图、两个模型的推理和评估；缓存输入与概率，避免重复读大表。",
              "- `STEP4_剩余混轨恢复.py`：受限递归恢复；`selected_config.json` 保存锁定配置。",
              "- 两个 `STEP4_test_xlsx_*.py` 入口已增加三条件筛选和 time 支持；默认读取已验收配置，`--baseline` 可比较原流程。",
              "- `metrics.csv`、`comparison.json` 和每事件 JSON 含详细匹配、圆参数与漏检真轨；`holdout_comparison.png` 是指标图。",
              "", "```powershell", '& "D:\\amancoda\\envs\\pytorch\\python.exe" .\\STEP4_真实数据评估.py',
              '& "D:\\amancoda\\envs\\pytorch\\python.exe" .\\STEP4_真实数据评估.py --features no-time --recipe conservative_residual --partition holdout',
              '& "D:\\amancoda\\envs\\pytorch\\python.exe" .\\STEP4_真实数据评估.py --features time --recipe conservative_residual --partition holdout',
              '& "D:\\amancoda\\envs\\pytorch\\python.exe" .\\STEP4_test_xlsx_圆参数.py --features time --event-ids 0 1 2', "```",
              "", "验证：Excel 接口筛选/行号检查、逐行时间与指标分母检查、人工四轨混合及首轮冻结检查均通过；"
              "全批次两个基线、800 开发事件的候选和 200 保留事件的锁定方案实际运行完成。common.py、原 STEP4 与两份权重的哈希仍与备份一致。",
              "", "本次未重新训练模型，未做噪声/高斯误差/电子学压力测试。并发评估的单事件耗时不用于模型速度排名；"
              "增加恢复步骤会带来额外拟合耗时。小幅提升需要更多独立批次验证稳定性。"]
    pd.DataFrame(table_rows).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "selected_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "评估报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(axes, ["track_efficiency", "fake_rate", "clone_rate"], ["Track efficiency (%)", "Fake rate (%)", "Clone rate (%)"]):
        x = np.arange(2)
        for offset, name, label, color in [(-.18, "baseline", "Baseline", "#8b9bad"), (.18, "selected", "Conservative recovery", "#1779ba")]:
            values = [results[v]["partitions"]["holdout"][name][key] * 100 for v in VARIANTS]
            bars = ax.bar(x + offset, values, .35, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xticks(x, VARIANTS)
        ax.set_title(title)
        ax.set_ylim(0, max(b.get_height() for b in ax.patches) * 1.18)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Independent generated batch: locked comparison on 200 holdout events")
    fig.tight_layout()
    fig.savefig(OUT / "holdout_comparison.png", dpi=180)
    plt.close(fig)
    print({v: results[v]["partitions"]["holdout"] for v in VARIANTS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["lock", "report"])
    args = parser.parse_args()
    lock() if args.mode == "lock" else make_report()
