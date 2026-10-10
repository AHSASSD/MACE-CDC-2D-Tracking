# MACE CDC Tracking

[English](./README.md) | [简体中文](./README.zh-CN.md)

This repository contains the complete reproducible releases of the CDC track-finding project.

## Versions

- `AI工程化4/`: frozen, mature baseline with complete STEP1/STEP2/STEP3/STEP4 training and evaluation flow.
- `AI工程化5/`: independent development line with separate data contracts, node-state classification, residual recovery, and training scripts.
- `AI工程化5/iterations/v2_edge_graph/`: experimental shared-hit edge-graph and triplet-classifier branch. It is not promoted to the default pipeline.
- `工程化2/`: small classic CA/Hough components reused by the AI5 residual experiment.

Original datasets, audit graphs, TensorBoard runs, caches, and temporary reports are intentionally excluded. Generate them with the STEP1 scripts or use the published summary reports in `reports/`.

## Reproduction order

1. Run `AI工程化4/STEP1_生成AI4数据集.py`.
2. Train with `AI工程化4/STEP2_训练模型.py`.
3. Run inference with `AI工程化4/STEP3_推理评估.py`.
4. Run tracking and evaluation with `AI工程化4/STEP4_两级寻迹.py`.
5. Use the AI5 scripts only after reading `AI工程化5/DATA_CONTRACT.md` and `AI工程化5/WORK_HANDOFF.md`.

## Results

The tested configurations and locked validation metrics are summarized in [`reports/RESULTS.md`](reports/RESULTS.md). AI4 is the stable reference. AI5 edge-graph results are included as experimental evidence and are not a production recommendation.

## Requirements

Python 3.10+, PyTorch 2.x, PyTorch Geometric, NumPy, pandas, SciPy, scikit-learn, joblib, and tqdm.

Set `MACE_CDC_CELL_MAP` and `MACE_CDC_HITS` to the local input Excel files before running STEP1.
