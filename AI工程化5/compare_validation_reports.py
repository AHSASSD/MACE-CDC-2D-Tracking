import argparse
import json
from pathlib import Path


METRICS = [
    "physics_track_survival",
    "algorithm_efficiency_given_visible",
    "end_to_end_efficiency_from_source",
    "algorithm_efficiency_5_6_visible",
    "mean_fake_tracks",
    "mean_clone_tracks",
]


def main():
    parser = argparse.ArgumentParser(description="比较两个 AI5 固定 validation 寻迹报告")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["conventions"]["exclusive_winner"]
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))["conventions"]["exclusive_winner"]
    result = {
        key: {"baseline": baseline[key], "candidate": candidate[key], "delta": candidate[key] - baseline[key]}
        for key in METRICS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
