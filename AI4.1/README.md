# AI4.1: whole-event tracking and timing audit

[English](./README.md) | [简体中文](./README.zh-CN.md)

AI4.1 is a separate research snapshot of the local AI4 working copy, covering whole-event retraining, two-pass tracking, fragment recovery and time-separated circle voting as of **2026-10-15**. The existing frozen AI4 and experimental AI5 directories are preserved.

**Current simulation contract: the user explicitly supplies known noisy d, retaining sigma_d=0.5mm. Times are in ns; the TTC reference is approximately 100ps and the reported extrapolation error is within about 5ns, with its statistical definition unresolved. Initial distance reconstruction and TTC iteration are outside the current scope. Existing models and timing recovery remain comparable under the same supplied-d assumption.** See the [latest update](docs/MEASUREMENT_UPDATE_20261015.md).

## Reading order

1. [Training, data contracts and measured results](docs/TRAINING_AND_RESULTS.md) (Chinese).
2. [Two-pass tracking, extension/merging and Hough principles](docs/TRACKING_AND_RECOVERY.md) (Chinese).
3. [Time-model shortcuts and measurement risks](docs/TIME_MODEL_AUDIT.md) (Chinese).
4. [Data and reproduction instructions](docs/REPRODUCTION.md) (Chinese, with commands).

## Current results

Eligible truth tracks have at least 5 hits. A match requires purity ≥0.8 and completeness ≥0.5. Efficiency uses truth tracks as denominator; fake and clone rates use output candidates. These are neither edge accuracy nor complete hit recovery.

| Model/pipeline | Dataset | Efficiency | Fake rate | Clone rate |
|---|---|---:|---:|---:|
| no-time finetune epoch62 + original two-pass | 1500 validation events | 93.7762% | 4.4130% | 0.1285% |
| Same weights + time-constrained extension/merging | Same validation set | 94.4957% | 3.4250% | 0.0882% |
| Same weights + recovery + time-separated Hough | Same validation set | 95.8951% | 3.4069% | 0.0891% |
| no-time scratch epoch88 + original two-pass | 1500 test events | 93.6578% | 4.9328% | 0.1000% |
| time finetune epoch33 + original two-pass | Same 1500 test events | 98.7334% | 1.8907% | 0.4899% |

Recovery improves reconstruction with frozen network weights; it does not retrain the network. A no-time network followed by timing recovery is not a timing-free pipeline. No accepted results exist for time+Hough, expert fusion or gated MoE; the gains above must not be added together.

## Contents

- `STEP1_完整事件数据集.py` / `STEP2_训练模型.py`: whole-event preprocessing, training and strict resume checks.
- `common.py` / `STEP4_两级寻迹.py`: input variants, message-passing network, drift-circle fitting and two-pass assembly.
- `fragment_recovery.py`: optional timing-constrained fragment recovery and three-drift-circle voting.
- `STEP6_事件切片可视化.py` / `STEP6_绘图界面.py`: inclusive EvtID slices, PNGs and circle/hit JSON.
- `STEP7_双专家验证.py`: fixed probability fusion experiment; **not a trained MoE**.
- `real_data_evaluation/`, `tools/`: research scripts and reproducible timing-leakage diagnostics.
- `whole_event_retraining/`: inference weights, metadata and event partitions; [weight hashes](MODEL_CATALOG.json).
- `reports/`: historical reports, summary evidence and figures. The topic documents above supersede outdated status statements in historical reports.

## Quick start

Install the repository requirements plus this directory's additions, configure the wire map, and provide the external CSV/Excel or graph shards described in the [reproduction instructions](docs/REPRODUCTION.md).

```powershell
cd AI4.1
$env:MACE_CDC_CELL_MAP = 'D:/your-data/cdc_cell_map.xlsx'
python STEP4_两级寻迹.py --features time --dataset-root whole_event_retraining --run-tag expert_whole_events_sigma05 --split test --rounds 1500 --strict-threshold 0.70 --relaxed-threshold 0.575
```

## Release scope and limitations

Source, small inference weights and summary evidence are included. Raw experimental data, graph shards, optimizer checkpoints, prediction caches, IDE state and SSH/cluster account configuration are excluded. Full reconstruction requires external data.

No direct label leakage or exact cross-split measurement duplicates were found. The time model strongly depends on simulated inter-track timing. Known noisy distance is an explicit simulation assumption, so the unimplemented initial distance-reconstruction stage does not invalidate the current comparisons. Actual TTC association follows tracking; updating distance after a coarse track is future work. The reported 5ns is not yet a Gaussian sigma, and current scores are not full experimental end-to-end efficiencies.
