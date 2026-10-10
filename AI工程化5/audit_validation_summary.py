"""Summarize an offline validation provenance audit without model inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(records: list[dict]) -> dict:
    reasons = Counter()
    real_reasons = Counter()
    for record in records:
        reasons.update(record.get("candidate_counts_by_reason", {}))
        real_reasons.update(record.get("real_candidate_counts_by_reason", {}))

    def total(name: str) -> int:
        return sum(int(record.get(name, 0)) for record in records)

    graphs = len(records)
    source_tracks = total("source_tracks")
    visible_nodes = total("accepted_nodes")
    return {
        "graphs": graphs,
        "source_tracks": source_tracks,
        "source_real_hits": total("source_real_hits"),
        "accepted_nodes": visible_nodes,
        "noise_nodes": total("noise_nodes"),
        "shared_nodes": total("shared_nodes"),
        "shared_node_rate": total("shared_nodes") / max(visible_nodes, 1),
        "truth_hits_lost_total": total("truth_hits_lost_total"),
        "tracks_with_dead_time_loss": total("tracks_with_dead_time_loss"),
        "mean_source_tracks_per_graph": source_tracks / max(graphs, 1),
        "mean_accepted_nodes_per_graph": visible_nodes / max(graphs, 1),
        "candidate_counts_by_reason": dict(reasons),
        "real_candidate_counts_by_reason": dict(real_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 AI5 validation provenance 审计")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = json.loads((args.audit_dir / "audit_records.json").read_text(encoding="utf-8"))
    report = summarize(records)
    output = args.output or args.audit_dir / "validation_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("报告：", output.resolve())


if __name__ == "__main__":
    main()
