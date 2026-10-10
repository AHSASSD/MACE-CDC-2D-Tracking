# AI工程化5

AI5 从 AI4 的冻结基线出发，只改变数据契约和评估口径。

- AI4 模型、权重、特征和 STEP4 寻迹实现保持不变。
- AI5 不复制模型权重或大型数据文件，直接引用 AI4 的只读产物。
- 第一阶段同时报告电子学物理损失与算法损失。
- `is_noise`、`shared_hit` 和命中来源仍只用于离线审计，不进入模型输入。

## 当前入口

```powershell
python "STEP4_物理算法损失分析.py" `
  --audit-dir "../AI工程化4/step4_hit_audit\audit_test_3000"
```

正式开发或调参应使用 validation 来源的审计集。当前 test 审计集只用于一次冻结诊断。

## AI4 基础流程入口

AI5 提供 AI4 基础程序的本地入口，源码、模型权重和大型数据仍只读复用
`../AI工程化4`，不会在 AI5 产生第二份基线。可从本目录运行：

```powershell
python "STEP3_推理评估.py" --features time --run-tag clean_physics_v2 --split validation
python "STEP4_两级寻迹.py" --features time --run-tag clean_physics_v2 --split validation
python "STEP5_噪声时间矩阵.py" --run-tag clean_physics_v2 --graphs 50
python "test_ai4.py"
```

这些入口仅转发到 AI4 冻结实现。STEP1 和 STEP2 会写入 AI4 自身目录，
因此在 AI5 中已明确禁用；在 AI5 独立数据生成和训练输出目录准备好之前，
不要从 AI5 运行这两个入口。AI5 的数据契约实验应继续使用独立的
validation 审计集和 AI5 自己的实验输出。

## AI5 独立训练

AI5 独立数据写入 `data/`，模型和元数据写入 `models/`，日志写入 `runs/`。
普通数据训练契约为 `exclusive_winner`；shared compatibility 仅在下方独立契约数据中接入损失权重。

```powershell
python "STEP1_生成AI5数据集.py" `
  --tag noise100k_sigma3 --workers 4 --chunk-size 100

python "STEP2_训练AI5模型.py" `
  --features time --data-tag noise100k_sigma3 --epochs 200 --run-tag ai5_exclusive_v1
```

no-time 不再重复训练，直接使用 AI4 正式权重：

```text
../AI工程化4/best_interaction_network_no_time_clean_physics_v2.pth
```

AI5 已完成的 time 模型只作为独立训练链路复现对照；默认冻结基线仍为 AI4。

## Shared Compatibility 样本权重对照

不要在其他训练或评估占用资源时生成完整契约数据。现有
`ai5_exclusive_time_v1` 已使用相同物理图和 exclusive 标签完成训练，直接作为
基线，不再重复训练。契约数据只用于 shared 权重实验组。

```powershell
python "STEP1_生成AI5契约数据集.py" `
  --train-graphs 4000 --val-graphs 1000 `
  --tag contract_noise100k_sigma3 --chunk-size 100

python "STEP2_训练AI5模型.py" `
  --features time --data-tag contract_noise100k_sigma3 `
  --shared-compat-weight 0.25 --epochs 200 `
  --run-tag ai5_contract_shared025_v1
```

固定 validation 评估示例：

```powershell
python "STEP4_AI5固定评估.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "models\best_interaction_network_time_ai5_contract_shared025_v1.pth" `
  --features time --pipeline strongest `
  --output "reports\validation_ai5_contract_shared025_v1.json"
```

模型选择以最终寻迹报告中的条件算法效率、5--6 点效率、假径迹和 clone 为准，
不使用边 F1 单独选模。

当前 shared025 对照已完成但未通过门槛：5--6 点效率下降到 75.5102%，
假径迹上升到 0.5367/事件。因此该模型仅作为负结果存档，不作为默认模型。

`shared_compat_weight=0.75` 单变量对照命令：

```powershell
python "STEP2_训练AI5模型.py" `
  --features time --data-tag contract_noise100k_sigma3 `
  --shared-compat-weight 0.75 --epochs 200 `
  --run-tag ai5_contract_shared075_v1
```

训练完成后仍使用固定 strongest 配置评估：

```powershell
python "STEP4_AI5固定评估.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "models\best_interaction_network_time_ai5_contract_shared075_v1.pth" `
  --features time --pipeline strongest `
  --output "reports\validation_ai5_contract_shared075_v1.json"
```

该对照已经完成但未晋级：相对 AI4 少匹配 29 条独占真径迹，5--6 点效率下降
0.1806 个百分点，clone 增加 47 条；假径迹只减少 3 条。停止继续扫描 shared
损失权重，主线转入节点状态分类与局部重连。

全量拟合消融也已完成：29,522 条候选含 shared 点，其中 11,218 条去除 shared
后不足 5 点；完整拟合虽平均降低 RMS 0.1012，但参数差异 P95 为 12,548.6。
因此 shared 点不能等权决定轨迹参数，后续按 exclusive 锚定、shared 验证处理。

候选冲突图全量报告已生成：88,528 条候选中 29,522 条含 shared 点，诊断得到
2,135 条真值冲突边。该图读取真值，只能离线分析，不能直接作为在线裁决器；
在线版本必须改用命中数、边概率、几何 RMS 和候选重叠等可观测量。

## Shared 冲突图与拟合后验证

该分析只读取 AI4 strongest 输出，构建候选间冲突统计，并比较包含/排除
shared 点的圆拟合参数；不会修改在线模型或 STEP4。

```powershell
python "STEP4_shared_conflict_analysis.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "..\AI工程化4\best_interaction_network_time_clean_physics_v2.pth" `
  --features time --rounds 3000 `
  --output "reports\shared_conflict_validation_3000.json"
```

解释结果时，shared 点只作为候选兼容性和拟合后验证信息；首次轨迹参数拟合
仍只由 exclusive 点主导。

纯可观测多阈值候选冲突重排（实验性，不改变 AI4 默认流程）：

```powershell
python "STEP4_observable_conflict_rerank.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "..\AI工程化4\best_interaction_network_time_clean_physics_v2.pth" `
  --features time --rounds 3000 --overlap-penalty 0.25 `
  --output "reports\observable_conflict_validation_3000.json"
```

该报告中的 `baseline` 是同一多阈值候选池的无惩罚贪心结果，
`observable_rerank` 才是重排结果；最终是否晋级还必须与
`reports/validation_ai4_time_strongest.json` 比较。

## 节点状态分类

该支线冻结 AI4 time strongest，使用完整候选图、AI4 边概率和首轮候选归属，
训练 `P(noise)` 与 `P(shared)`。真值字段只生成监督标签，不存入模型输入。

先生成派生数据（该步骤会运行 AI4 strongest，耗时较长）：

```powershell
python "STEP1_生成节点分类派生数据.py" `
  --tag ai4_strongest_v1 --chunk-size 100 `
  --workers 6 --worker-torch-threads 1
```

`--workers 1` 保留原来的单进程 GPU/CPU 路径；`--workers > 1` 使用 CPU 多进程。
不要把 `workers × worker-torch-threads` 设置得明显高于 CPU 逻辑核心数。

再训练轻量节点分类器：

```powershell
python "STEP2_训练节点分类器.py" `
  --data-tag ai4_strongest_v1 --run-tag node_state_v1 `
  --epochs 100 --batch-size 4
```

固定 validation 评估节点分类和“仅噪声硬清理”消融：

```powershell
python "STEP4_评估节点分类器.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --features time --noise-threshold 0.95 --shared-threshold 0.45 --rounds 3000 `
  --output "reports\node_classifier_validation_3000.json"
```

报告中的 `baseline` 必须与 `validation_ai4_time_strongest.json` 一致；
`hard_noise_filter` 只检验去噪，不代表最终 shared 局部重连方案。此阶段只验证
节点分类和去噪是否有净收益，分类结果通过后才接入局部第二轮重连。

全量结果显示节点分类稳定，但单独硬去噪只净增 2 条匹配径迹并增加 1 条 fake。
因此继续使用两个节点概率作为局部重连信号，不将硬去噪单独晋级。

shared 非破坏性局部重连固定评估：

```powershell
python "STEP4_shared局部重连.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --features time --noise-threshold 0.95 --shared-threshold 0.45 `
  --fragment-threshold 0.55 --borrow-edge-threshold 0.55 `
  --anchor-rms-limit 1.0 --shared-residual-limit 1.0 --rounds 3000 `
  --output "reports\shared_local_reconnect_validation_3000.json"
```

该实验冻结首轮径迹，只从未使用的 2--6 点碎片出发；至少 3 个未占用点先建立
参数锚点，预测 shared 点只能被借用并接受锚定轨迹残差检查。

候选级机制审计与锁定裁决（固定参数，完整 3000 图）：

```powershell
python "STEP4_shared候选审计.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --rounds 3000 --development-events 1500 `
  --output "reports\shared_candidate_adjudication_validation_3000.json"
```

前 1500 图只用于从预注册物理规则中选择最多两个条件，后 1500 图只评估一次。
no-borrow 和 borrowed 分支独立选择、独立报告。锁定 gate 要求新增匹配和新增
5--6 点匹配均大于 0，同时新增 fake 和 clone 都为 0。

此前 300 图只有 12 条候选：no-borrow 为 6 真/3 fake，borrowed shared 为
1 真/1 fake/1 clone。该批次及其扫描规则不参与新的锁定裁决。

只做拟合消融、不改变候选生成：

```powershell
python "STEP4_shared_fit_ablation.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "..\AI工程化4\best_interaction_network_time_clean_physics_v2.pth" `
  --features time --rounds 3000 `
  --output "reports\shared_fit_ablation_validation_3000.json"
```

## 经典残差联合寻迹

论文启发的局部 edge graph 迭代版位于
`iterations/v2_edge_graph/`。该目录与主线隔离，首版规则实验只作为结构
诊断，不改变 AI4 默认结果；下一版将从 shared articulation 直接触发局部
Interaction Network 重建。

当前残差恢复主线复用 `工程化2` 的 CA、Hough 与 IoU 共识，并在联合候选图上按
跨片段边支持或联合拟合质量合并片段。300 图实验中，联合分支新增 20 条匹配，
高于 CA 的 16 条和 Hough 的 12 条；片段合并后新增匹配为 21，fake 从 48 降至
42，clone 保持 0。

快速验证：

```powershell
python "STEP4_经典残差寻迹.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --rounds 300 `
  --output "reports\classic_residual_validation_300.json"
```

完整训练来源候选审计耗时较长，由人工运行：

```powershell
python "STEP4_经典残差候选审计.py" `
  --dataset-file "data\ai5_dataset_train_contract_noise100k_sigma3.pt" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --rounds 4000 `
  --output "reports\classic_residual_candidate_audit_train_4000.json"
```

该完整训练审计完成前，不启动 Interaction Network 后训练；AI4 权重继续只读冻结。

完整训练审计已生成后，可训练轻量三分类候选裁决器：

```powershell
python "STEP4_训练经典候选裁决器.py" `
  --candidate-csv "reports\classic_residual_candidate_audit_train_4000.csv" `
  --model-output "models\classic_candidate_adjudicator_v1.joblib" `
  --report-output "reports\classic_candidate_adjudicator_train_4000.json"
```

该模型的 accept 分支在 validation 300 图只得到 6 真/5 fake，fragment merge 分支
为 0 真/2 fake，均不单独晋级。fragment 概率只保留为后续补点/扩图许可。

下一次长任务是完整 3000 图机制对照：

```powershell
python "STEP4_经典残差寻迹.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --adjudicator-model "models\classic_candidate_adjudicator_v1.joblib" `
  --rounds 3000 `
  --output "reports\classic_residual_validation_3000.json"
```

该对照已完成。`accept only` 新增 66 真/51 fake/0 clone，其中 46 条为 5--6 点；
`hybrid merged` 虽新增 175 真，但同时新增 478 fake 和 4 clone。当前不扩大连接，
也不让 fragment merge 直接输出。

下一步生成完整 validation 候选明细，用于归因 accept 分支的错误类型：

```powershell
python "STEP4_经典残差候选审计.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --rounds 3000 `
  --output "reports\classic_residual_candidate_audit_validation_3000.json"
```

该报告只用于误差归因和设计下一版特征，不在 validation 上重新选择阈值。
