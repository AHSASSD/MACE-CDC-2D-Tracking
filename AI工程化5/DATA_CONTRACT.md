# AI5 数据契约 v1

## 三层真值

1. `source truth`：电子学与时间窗处理之前的模拟命中。
2. `exclusive electronics truth`：每个 Cell 在非延长死时间内只有最早上升沿拥有观测命中。
3. `shared-credit truth`：同一观测 Cell 可离线关联到被它抑制的其他真实径迹，用于研究共享点上限。

## 在线输入

沿用 AI4：线丝几何、带测量误差的漂移距离，以及可选的固定触发窗观测时间。

禁止输入：`is_noise`、`shared_hit`、`EvtID`、`TrkID`、`t0`、`tHit`、`t_window`、命中淘汰原因。

## 固定评估

- 物理层效率：电子学可见径迹数 / source 可评估径迹数。
- 算法层效率：成功重建径迹数 / 电子学可见径迹数。
- 端到端效率：成功重建径迹数 / source 可评估径迹数。
- 默认主口径为 `exclusive_winner`；`shared_credit` 只作为共享点敏感性上限。

## Shared compatibility 第一轮实验

- 主边标签始终为 `exclusive_winner`。
- `shared-compatible` 但非 exclusive 的边仍是负样本，不改成正样本。
- 第一轮只降低这些歧义负边的损失权重；推理阶段不读取 shared 真值。
- 基线权重为 `1.0`，首个候选权重为 `0.25`。
- shared 点只用于候选兼容性和拟合后验证，不参与首次轨迹参数拟合。

## 节点状态分类 v1

- 输入完整候选点图和边图，并加入冻结 AI4 time strongest 的边概率。
- `edge_same_first_pass_track` 只表示边两端被首轮归入同一最终候选，不读取真值。
- 节点输出为两个独立概率：`P(noise)` 和 `P(shared)`。
- `shared` 训练标签定义为非噪声且 `truth_tracks` 数量大于 1；真值仅作监督。
- uncertainty 不作为第三类训练，而由两个输出的归一化二元熵推导。
- 第二轮重连只处理疑似 shared 点、未使用点和 2--6 点碎片；首轮长径迹冻结，
  shared 点只能被非破坏性借用。
- 候选裁决将 `borrowed_shared_hits=0` 与 `>0` 拆为独立机制。
- validation 图 `0--1499` 选择预注册的最多两条件规则，`1500--2999` 只做一次
  锁定评估；禁止用锁定结果继续修改规则。
