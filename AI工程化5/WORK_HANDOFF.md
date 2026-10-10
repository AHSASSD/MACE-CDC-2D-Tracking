# MACE CDC AI工程化5 工作交接

更新日期：2026-09-01

## 1. 项目目标

AI工程化5（AI5）用于继续研究 MACE CDC 多径迹重建，但不再直接改动已经稳定的 AI工程化4（AI4）。AI5 的第一阶段只改变数据契约、真值口径和评估方式，模型结构与寻迹规则继续复用 AI4，以便明确每次改动究竟改善了哪一层。

工作目录：

```text
AI工程化5
```

Python 环境：

```text
python
```

原始模拟命中：

```text
$env:MACE_CDC_HITS
```

## 2. 当前原则

1. AI4 是冻结基线。不要修改其正式模型、数据集、STEP1--STEP4 或已校准阈值。
2. AI5 暂不复制大型数据和模型权重，通过相对路径只读复用 AI4。
3. 一次只改变一类变量：先数据契约，再模型，最后才考虑寻迹规则。
4. `is_noise`、`shared_hit`、真值身份和淘汰原因仅供离线审计，禁止作为模型输入。
5. test 审计结果已经用于一次冻结诊断，不能继续用于调参或选模。后续开发必须生成 validation 来源审计集。

详细契约见 [DATA_CONTRACT.md](DATA_CONTRACT.md)，冻结基线配置见 [baseline_manifest.json](baseline_manifest.json)。

## 3. 物理与数据假设

- 径迹身份为 `(EvtID, TrkID)`。
- 时间单位 ns，长度单位 mm，能量单位 MeV。
- 每张图对应 300 ns 固定硬件触发窗口。
- 观测上升沿时间为 `t_obs = (t - t0) + Normal(0, 3 ns)`。
- 漂移距离为 Excel 真值距离加 `0.075 mm` 高斯误差后取绝对值。
- 每 Cell 使用约 100 ns 非延长死时间。
- 噪声和真实信号在同一 `t_obs` 时间轴排序并竞争死时间。
- 默认噪声率为每 Cell 100 kHz，单窗口泊松均值为 0.03。
- 模型可使用线丝几何、漂移距离以及可选的固定触发窗时间。
- 禁止输入 `is_noise`、`shared_hit`、`EvtID`、`TrkID`、`t0`、`tHit`、`t_window`、`t_rel`、`dt_rel`。

## 4. AI4 冻结基线

主要文件位于：

```text
AI工程化4
```

正式 time 模型：

```text
best_interaction_network_time_clean_physics_v2.pth
```

正式 no-time 模型：

```text
best_interaction_network_no_time_clean_physics_v2.pth
```

原正式寻迹入口：

```text
STEP4_两级寻迹.py
```

目前最强但仍属于实验流程的配置：

```text
strict_threshold = 0.70
relaxed_threshold = 0.575
rerun_threshold = 0.55
rerun_rms = 0.20
```

它在短候选核心提取后，对剩余小碎片子图再次运行原 time 模型。实现位于：

```text
..\AI工程化4\step4_short_rerun_experiment\run_experiment.py
```

3000 张 calibration 图上的关键结果：

| 流程 | 微平均效率 | 假径迹/事件 | clone/事件 |
|---|---:|---:|---:|
| 短核心基线 | 97.9745% | 0.3187 | 0.0353 |
| 原模型残差子图二次推理 | 98.3306% | 0.3767 | 0.0363 |

二次推理净恢复 315 条真径迹，新增损失 0；其中 251 条为 5--6 点径迹，64 条为 7--10 点径迹。报告：

```text
..\AI工程化4\step4_short_rerun_experiment\result_time_calibration_3000.json
```

训练过的 bridge residual expert v2 将桥接边 F2 从 0.6272 提升至 0.6502，但候选级表现仍弱于直接重复运行原模型，因此暂不作为默认专家。

## 5. 命中来源审计数据

生成器：

```text
..\AI工程化4\step4_hit_audit\generate_audit_dataset.py
```

已生成的冻结 test 审计集：

```text
..\AI工程化4\step4_hit_audit\audit_test_3000
```

包含：

- `graphs.pt`：电子学处理后的图，并保留离线 provenance；
- `audit_records.json`：事件级死时间、窗口和命中来源统计；
- `metadata.json`：生成参数。

审计原因包括：

```text
accepted
dead_time_by_other_track
dead_time_by_noise
dead_time_by_same_track
outside_window
```

注意两套真值口径：

- `exclusive_winner`：一个 Cell 只属于最早被接受的上升沿，是默认物理口径。
- `shared_credit`：获胜观测点也关联被它抑制的其他径迹，只用于估计共享兼容性的理论上限。

## 6. 第一阶段严格拆分结果

分析入口：

```text
STEP4_物理算法损失分析.py
```

实际实现复用：

```text
..\AI工程化4\step4_hit_audit\analyze_physics_algorithm_loss.py
```

完整冻结报告：

```text
..\AI工程化4\step4_hit_audit\audit_test_3000\physics_algorithm_loss_time_strongest.json
```

### Exclusive winner 默认口径

| 指标 | 数值 |
|---|---:|
| source 可评估径迹 | 89,698 |
| 电子学可见径迹 | 88,741 |
| 物理层不可重建径迹 | 957 |
| STEP4 成功重建 | 87,141 |
| STEP4 算法漏检 | 1,600 |
| 物理层径迹存活率 | 98.9331% |
| 给定电子学可见径迹的算法效率 | 98.1970% |
| 源真值端到端效率 | 97.1493% |
| 独占真实命中保留率 | 91.0951% |
| 假径迹/事件 | 0.4713 |
| clone/事件 | 0.0337 |

拆分恒等式已经验证：

```text
89,698 = 957 + 88,741
88,741 = 87,141 + 1,600
98.9331% * 98.1970% = 97.1493%
```

### 物理层命中损失

源真实命中共 907,714 个，电子学独占保留 826,883 个，损失 80,831 个：

| 原因 | 命中数 | 损失占比 |
|---|---:|---:|
| 被其他真实径迹抢占 | 70,250 | 86.91% |
| 被噪声抢占 | 5,780 | 7.15% |
| 抖动后落出 300 ns 窗口 | 4,439 | 5.49% |
| 被同一径迹更早命中抑制 | 362 | 0.45% |

结论：短径迹的第一物理来源是多径迹之间的 Cell 竞争，不是噪声。

### 算法层损失

算法漏掉 1,600 条已经电子学可见的径迹，其中：

| 可见命中数 | 漏检径迹 |
|---|---:|
| 5--6 | 1,169 |
| 7--10 | 320 |
| 11--20 | 31 |
| 21+ | 80 |

短径迹问题因此是两段叠加：电子学先夺走命中，算法又最容易漏掉只剩 5--6 点的径迹。

### Shared credit 敏感性

共享记分把物理层径迹存活率按定义提高到 100%，但条件算法效率下降到 97.3221%。原因是共享关联引入了更强的标签歧义和候选污染。因此它只能作为辅助兼容标签，不能替代 `exclusive_winner` 主真值。

阶段结果摘要另见 [PHASE1_RESULTS.md](PHASE1_RESULTS.md)。

## 7. 已确认的问题

1. AI4 原数据集的 `shared_hit` 始终为 false，不能用于分析 Cell 共享或死时间归因。
2. 原困难事件报告只能分析电子学处理之后的算法现象，不能判断一个点在进入 GNN 前为何消失。
3. time 模型困难候选中的污染主要来自其他真实径迹，而非噪声。
4. 普通边 F1 或桥接边 F2 的提升不保证最终候选级效率提升，选模目标必须最终落到寻迹效用。
5. 直接使用 shared-credit 主标签会混淆“一个电子学观测”与“多个可能径迹归属”。

## 8. 下一步推荐顺序

### 第一步：生成 validation 审计开发集

不要用当前 test 审计集继续开发。建议生成 3000 张 validation 审计图：

```powershell
python `
  "../AI工程化4/step4_hit_audit\generate_audit_dataset.py" `
  --split validation `
  --graphs 3000 `
  --output-dir "data\audit_validation_3000"
```

### 第二步：只做数据契约对照

保持 AI4 模型结构、输入特征、训练设置和 STEP4 不变，比较：

1. `exclusive_winner` 主边标签；
2. `exclusive_winner` 主边标签 + shared compatibility 辅助权重或辅助头。

不要直接将 shared-credit 作为主边标签。第一轮应先用样本权重实现，只有它显示稳定收益后才增加辅助输出头。

### 第三步：固定评估

所有实验同时报告：

- 物理层径迹存活率；
- 条件算法效率；
- 源真值端到端效率；
- 5--6 点可见径迹效率；
- 假径迹/事件；
- clone/事件。

模型选择优先使用 validation 上的最终寻迹效用，而不是边分类 F1。

### 第四步：满足条件后再改模型

只有数据契约实验证明 shared compatibility 有独立价值后，才考虑：

- 独占重建主专家 + shared compatibility 辅助专家；
- 对可见 5--6 点径迹单独加权；
- 候选级裁决者或多专家融合。

在此之前不要同时改 GNN、数据生成和 STEP4，否则无法归因。

## 9. 验证状态

已通过：

```text
AI4 self-check passed
```

以下程序已完成语法检查与实际 3000 图运行：

```text
..\AI工程化4\step4_hit_audit\generate_audit_dataset.py
..\AI工程化4\step4_hit_audit\analyze_physics_algorithm_loss.py
STEP4_物理算法损失分析.py
```

完整 3000 图分析耗时约 2300 秒，使用 NVIDIA GeForce RTX 4060 Laptop GPU；主要耗时仍来自 CPU 候选圆拟合和 STEP4 后处理。

## 10. 建议技能

下一位代理建议使用：

- `ponytail`：维持单变量、最小改动，优先复用 AI4；
- `diagnose`：当边指标提升但最终寻迹下降时做严格归因；
- `tdd`：为新数据契约和标签生成建立最小回归检查；
- `code-review`：AI5 首次训练改动完成后，检查真值泄露和是否意外修改 AI4。

## 11. 接手时先做什么

1. 阅读本文件、[DATA_CONTRACT.md](DATA_CONTRACT.md) 和 [PHASE1_RESULTS.md](PHASE1_RESULTS.md)。
2. AI4 time strongest 继续作为冻结主基线；AI4 no-time 直接复用，不再训练 AI5 no-time。
3. `shared_compat_weight=0.25` 已在固定 validation 上失败，不得晋级。
4. `shared_compat_weight=0.75` 已完成并失败：少匹配 29 条独占真径迹，5--6 点
   效率下降 0.1806 个百分点，clone 增加 47 条。停止继续扫描 shared 权重。
5. 节点分类派生数据及训练程序已经准备：完整候选图输入，输出 `P(noise)`、
   `P(shared)`，uncertainty 由熵推导。先验证分类，不立即修改 STEP4。
6. 节点分类模型已训练完成：噪声 F1=99.6547%，shared F1=94.2002%。先运行
   全量独立评估为噪声 F1=99.6627%、shared F1=94.4007%；单独硬去噪只净增
   2 条匹配径迹并增加 1 条 fake，不单独晋级。
7. `STEP4_shared局部重连.py` 已实现：只处理疑似 shared 点、未使用点和 2--6 点
   碎片；长径迹冻结，shared 点允许非破坏性借用，但不等权决定首次拟合参数。
   全量 validation 新恢复 55 条，其中 49 条为 5--6 点；但新增 49 条 fake、21 条
   clone，未晋级。125 条候选只借用 84 个 shared 点，至少 41 条未实际借用 shared，
   下一步必须先做候选级分层审计和裁决，不能直接扫描阈值。
8. 300 图候选审计得到 12 条：no-borrow 为 6 真/3 fake，borrowed 为
   1 真/1 fake/1 clone。v1 主要增益来自残余碎片自行成轨，不是 shared 借用。
   样本过小，任何扫描所得规则都不晋级。`STEP4_shared候选审计.py` 已升级为
   3000 图两阶段裁决：前 1500 图选择预注册简单规则，后 1500 图锁定验证，
   no-borrow 与 borrowed 独立报告。
9. 完整两阶段裁决已完成但锁定 gate 失败：no-borrow 为 9 真/4 fake/0 clone，
   borrowed 为 1 真/2 fake/2 clone。no-borrow 的 4 条 fake 都由只有 4 个真实可见点
   的径迹混入其他径迹点后凑成；下一版需要 leave-one-out 稳定性和节点级边支持。
   本锁定区不得继续用于阈值选择。
10. observable overlap rerank 仅保留为诊断支线，不继续围绕 overlap penalty 调参。
11. 经典残差 CA/Hough 联合生成已完成 300 图验证：hybrid 新增 20 条匹配，
    其中 16 条为 5--6 点，高于 CA 单独的 16 条和 Hough 单独的 12 条；但新增
    48 条 fake，单候选简单过滤在训练 300 图上没有正效用。
12. 已加入候选对片段图合并：训练 300 图与 validation 300 图中，规则
    `P(edge)>=0.5 的跨片段边数 >=30，或联合 RMS<=1.0` 均未合并异真径迹候选。
    validation 上 hybrid_merged 比 hybrid 多 1 条匹配、少 6 条 fake、clone 不变，
    5--6 点新增仍为 16。报告为 `reports/classic_residual_validation_300.json`。
13. 下一步由人工运行完整 4000 训练图候选审计，命令见 README。先扩充候选级和
    候选对样本，再设计接收/合并策略；不要在 validation 300 图继续选阈值。
    只有经典联合生成仍显示明显召回上限时，才后训练残差图专用 Interaction Network。
14. 完整 4000 训练图审计已完成：923 候选中 206 真、713 fake、4 clone；142 条
    为未匹配真径迹的不完整片段。片段合并规则选中 102 对，100 对同真径迹；加上
    fragment 分类许可后为 81/81 同真径迹。
15. 轻量三分类裁决器已训练。锁定段 accept 为 24 真/16 假，fragment 为 33 真/1 假。
    但 validation 300 图端到端 accept only 只有 6 真/5 fake，fragment merge 为
    0 真/2 fake，组合不晋级。fragment 只作为补点/扩图种子。
16. 下一步由人工运行完整 3000 validation 机制对照，命令见 README；主要比较
    `hybrid_merged`、`hybrid_accept_only`、`hybrid_fragment_merge`。不要再调当前
    三分类阈值。
17. 完整 3000 validation 已完成。accept only 为 66 真/51 fake/0 clone，其中
    46 条为 5--6 点，简单效用 +15；fragment merge 为 2 真/11 fake，停止直接输出；
    hybrid merged 为 175 真/478 fake/4 clone，召回高但不可晋级。
19. 已完成 shared 点全量回收 300 图测试：残差节点从 2190 增至 7510，hybrid merged
    仅新增 7 条匹配，却增加 609 条 fake 和 39 条 clone，5--6 点效率下降。该方案废弃；
    下一版只允许残差候选附近的一跳 shared 点进入局部图，并要求供体边支持低、候选边支持高。
18. 当前不后训练 Interaction Network 扩大连接。下一步由人工运行完整 3000 图
    候选明细审计，分析 accept 的错误构成并设计节点拓扑、单点删除质量变化等特征。
    validation 明细不得用于直接选择最终阈值。
20. 局部 shared 并集测试已完成：保留原始 hybrid 候选，再加入不重复的一跳 shared
    候选。300 图 `hybrid_shared_augment_merged` 新增 23 条匹配、约 17 条 5--6 点，
    fake 相对基线减少 17 条，clone 不增加。这是当前优先验证方案，需人工运行完整
    3000 validation；不要直接替换 AI4 默认流程。
