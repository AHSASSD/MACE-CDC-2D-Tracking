"""历史冻结流程在新批次上的固定配置比较；不选模或改默认配置。"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from STEP4_真实数据评估 import summarize

OUT = Path(__file__).resolve().parent
OLD = ROOT.parent / "AI工程化4"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    frozen = {v: load(OLD / f"step4_ai4_{v.replace('-', '_')}_clean_physics_v2_test_frozen.json")
              for v in ["no-time", "time"]}
    new = {}
    for v in frozen:
        dev = load(OUT / f"{v}_higher_threshold_development.json")["events"]
        hold = load(OUT / f"{v}_higher_threshold_holdout.json")["events"]
        rows = sorted(dev + hold, key=lambda r: r["EvtID"])
        assert len(rows) == 1000 and len({r["EvtID"] for r in rows}) == 1000
        new[v] = {"recipe": {"strict": .70, "relaxed": .575}, "summary": summarize(rows), "events": rows}
        (OUT / f"{v}_historical_frozen_all.json").write_text(json.dumps(new[v], ensure_ascii=False, indent=2), encoding="utf-8")
    rerun = load(OUT / "time_historical_strongest.json")
    historical = load(OLD / "step4_short_rerun_experiment" / "result_time_calibration_3000.json")
    # 历史报告保存每真轨漏检列表，用原测试图的真值计数得到严格一致的 hit 分箱。
    import torch
    from collections import Counter
    torch.set_num_threads(1)
    old_bins = Counter()
    old_misses = Counter()
    old_rows = frozen["time"]["rounds"]
    offset = 0
    for path in sorted((OLD / "dataset_test_noise100k_sigma3.pt.parts").glob("part_*.pt")):
        graphs = torch.load(path, weights_only=False, map_location="cpu")
        for data in graphs:
            sizes = Counter(map(tuple, data.truth_pairs.tolist()))
            missed = set(map(tuple, old_rows[offset]["missed_tracks"]))
            for key, size in sizes.items():
                if size >= 5:
                    bucket = "5_6" if size <= 6 else "7_10" if size <= 10 else "11_20" if size <= 20 else "21_plus"
                    old_bins[bucket] += 1
                    old_misses[bucket] += key in missed
            offset += 1
        del graphs
    assert offset == 1000 and sum(old_bins.values()) == frozen["time"]["summary"]["total_truth_tracks"]
    old_rates = {b: 1 - old_misses[b] / old_bins[b] for b in old_bins}
    new_bins = new["time"]["summary"]["by_truth_hits"]
    composition_only = sum(old_rates[b] * m["truth"] for b, m in new_bins.items()) / new["time"]["summary"]["truth_tracks"]
    result = {"historical_thread": {"title": "AI工程化4开发", "thread_id": "01a042bd-291c-7131-814e-e5192121a55c"},
              "fixed_threshold": {v: {"old_test": frozen[v]["summary"], "new_batch": new[v]["summary"]} for v in frozen},
              "historical_strongest": {"old_calibration": historical["short_graph_rerun"], "new_batch": rerun["summary"]},
              "old_time_hit_bins": {b: {"truth": old_bins[b], "missed": old_misses[b], "efficiency": old_rates[b]} for b in old_bins},
              "composition_only_expected_efficiency": composition_only,
              "composition_note": "Reweights old within-bin efficiencies to new hit-count composition; descriptive counterfactual, not causal proof"}
    (OUT / "historical_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    def pct(x):
        return f"{x * 100:.3f}%"
    lines = ["# 与 AI工程化4开发 的 98% 结果对齐核查", "", "2026-10-13。已通过应用读取该对话，并核查磁盘报告。",
             "", "旧结果确实存在：0.70/0.575 两级流程在独立旧测试图上，time 微平均效率为 98.005%；"
             "短核心+原模型剩余子图二次推理在 3000 个校准图上为 98.331%。后者来自验证源池重新组合并选过参数，不是新的独立粒子测试。",
             "", "此前新批次默认基础结果取 metadata 的 0.50/0.35。这与旧冻结寻迹阈值不同，不能直接把此前 90.748% 与旧 98.005% 当作完全同流程比较。本次补齐固定配置，未根据新测试结果调参或改默认验收方案。",
             "", "## 同一 0.70/0.575 两级流程", "", "| 模型 | 旧独立测试图 | 新批次原始事件 | 新批次假轨/事件 | 新批次克隆/事件 |",
             "|---|---:|---:|---:|---:|"]
    for v in frozen:
        m = new[v]["summary"]
        lines.append(f"| {v} | {pct(frozen[v]['summary']['micro_track_efficiency'])} | {pct(m['track_efficiency'])} | {m['fake_per_event']:.3f} | {m['clone_per_event']:.3f} |")
    lines += ["", f"旧增强流程在新批次上的效率为 **{pct(rerun['summary']['track_efficiency'])}**，"
              f"假轨/事件 {rerun['summary']['fake_per_event']:.3f}、克隆/事件 {rerun['summary']['clone_per_event']:.3f}。"
              "首轮沿用缓存的 GPU 边概率，剩余子图用相同权重在 CPU float32 推理；没有重训。",
              "", "## 输入与分母差异", "", "旧图包含噪声、距离/时间高斯误差、300 ns 窗口裁切及 100 ns 死时间竞争。"
              "旧基本效率以电子学处理后仍有至少 5 hit 的真轨为分母；新批次主评估保留原始筛选 hit，无这些处理，"
              "以原始至少 5 hit 的真轨为分母。两者 purity>=0.8、completeness>=0.5 门槛一致，但完整度参照的 hit 集合不同。",
              "", "旧审计已给出具体例子：条件算法效率 98.197%，乘以物理可见率 98.933%，端到端效率为 97.149%。"
              "因此应分别比较电子学可见效率和原始源真值效率，不能只拿一个 98% 数值外推所有输入场景。",
              "", "## 按真轨长度对齐", "", "| hit 数 | 旧 time 真轨数 | 旧效率 | 新真轨数 | 新同流程效率 |", "|---|---:|---:|---:|---:|"]
    for b in ["5_6", "7_10", "11_20", "21_plus"]:
        m = new_bins[b]
        lines.append(f"| {b} | {old_bins[b]} | {pct(old_rates[b])} | {m['truth']} | {pct(m['efficiency'])} |")
    lines += ["", f"如果仅改变 hit 分箱比例，保留旧的分箱内效率，预期汇总效率为 {pct(composition_only)}。"
              "这是假设性重加权，不是因果证明；它说明短/长轨比例变化不足以解释实际降幅，普通 7--20 hit 径迹的识别能力也明显下降。",
              "", "当前已确认：旧、新输入图规模与邻居数不同；旧训练每图约 318 节点、平均入度 34.65，新图约 117 节点、入度 19.88。"
              "新批次保留同 Cell 多次原始 hit，旧图做死时间抑制。旧权重在新图上的边 F1 已降到约 84.45%/86.07%。"
              "这些共同支持输入分布适配问题，尚不能分配各因素的独立因果贡献。",
              "", "## 下一步", "", "先在同一新批次上使用准确的电子学观测契约做成对诊断，分别记录源真值和观测真值效率，"
              "再比较旧权重、同架构重训、微调。时间零点、噪声、误差、死时间和输入 hit 语义必须固定；"
              "完整事件建图后不再随机将多个新事件叠加。最终在新的独立生成批次验收，不能将本轮已研究的 test.xlsx 宣称为全新未见测试。",
              "", "本次没有新增电子学模拟或重训；补跑的是旧冻结阈值和旧增强流程。旧报告可复核，数据分布差异的因果拆分仍待实验。"]
    (OUT / "98%结果差异核对.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"fixed": {v: m["summary"]["track_efficiency"] for v, m in new.items()},
           "historical_strongest_new": rerun["summary"]["track_efficiency"], "composition_only_expected": composition_only})


if __name__ == "__main__":
    main()
