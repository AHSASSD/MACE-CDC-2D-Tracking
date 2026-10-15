# MACE CDC 径迹重建

[English](./README.md) | [简体中文](./README.zh-CN.md)

本仓库包含 CDC 径迹重建项目的完整可复现实验版本。

## 版本

- `AI工程化4/`：冻结的成熟基线，包含完整 STEP1/STEP2/STEP3/STEP4 数据生成、训练、推理和评估流程。
- [`AI4.1/`](AI4.1/)：AI4副本的完整事件研究快照，包含time/no-time重训、两级寻迹、可选片段/Hough回收及时间信息审计。请先阅读其README；time的98%是筛选模拟任务结果，尚非真实探测器性能。
- `AI工程化5/`：独立开发版本，包含独立数据契约、节点状态分类、残差回收和训练程序。
- `AI工程化5/iterations/v2_edge_graph/`：shared 命中 edge-graph 与 triplet 分类器实验分支，尚未晋级为默认流程。
- `工程化2/`：AI5 残差实验复用的经典 CA/Hough 小型组件。

原始数据、审计图、TensorBoard 日志、缓存和临时报告均不上传。可使用 STEP1 程序重新生成数据，或查看 `reports/` 中的汇总结果。

## 复现顺序

1. 运行 `AI工程化4/STEP1_生成AI4数据集.py`。
2. 使用 `AI工程化4/STEP2_训练模型.py` 训练。
3. 使用 `AI工程化4/STEP3_推理评估.py` 推理。
4. 使用 `AI工程化4/STEP4_两级寻迹.py` 寻迹和评估。
5. 使用 AI5 前先阅读 `AI工程化5/DATA_CONTRACT.md` 和 `AI工程化5/WORK_HANDOFF.md`。

## 测试结果

已测试配置和锁定 validation 指标见 [`reports/RESULTS.md`](reports/RESULTS.md)。AI4 是稳定参考版本，AI5 edge-graph 结果仅用于实验说明，不代表生产推荐。

AI4.1的新完整事件结果及口径单独记录于[训练与表现](AI4.1/docs/TRAINING_AND_RESULTS.md)、[回收原理](AI4.1/docs/TRACKING_AND_RECOVERY.md)、[time模型风险](AI4.1/docs/TIME_MODEL_AUDIT.md)。原始数据和优化器断点不随仓库发布。

[最新测量与范围约定](AI4.1/docs/MEASUREMENT_UPDATE_20261015.md)：AI4.1本阶段明确假设d已知且带误差，TTC关联及迭代求d留作后续工作；“5ns以内”尚未定义为高斯σ。已有模拟比较在这一明确假设下继续有效。

## 依赖

Python 3.10+、PyTorch 2.x、PyTorch Geometric、NumPy、pandas、SciPy、scikit-learn、joblib 和 tqdm。

运行 STEP1 前，请将环境变量 `MACE_CDC_CELL_MAP` 和 `MACE_CDC_HITS` 分别指向本地输入 Excel 文件。
