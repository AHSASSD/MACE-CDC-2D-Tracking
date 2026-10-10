# AI工程化4

AI4 在 AI3 的独立副本基础上加入探测器死时间、单元噪声和带误差的可观测上升沿时间。

运行顺序：

```powershell
python STEP1_生成AI4数据集.py
python STEP2_训练模型.py --features no-time --run-tag clean_physics_v2
python STEP2_训练模型.py --features time --run-tag clean_physics_v2
python STEP3_推理评估.py --features no-time --run-tag clean_physics_v2 --split validation
python STEP3_推理评估.py --features time --run-tag clean_physics_v2 --split validation
python STEP4_两级寻迹.py --features no-time --run-tag clean_physics_v2 --split validation
python STEP4_两级寻迹.py --features time --run-tag clean_physics_v2 --split validation
python STEP5_噪声时间矩阵.py --run-tag clean_physics_v2 --graphs 50
```

默认数据配置：平均 30 条轨迹的截断 Poisson 分布、噪声率 100 kHz、事例宽度 300 ns、时间测量误差 `sigma_t=3 ns`。真实命中以 `t-t0` 放入固定触发窗口，距离真值加入 `sigma_d=0.075 mm` 高斯误差；所有 Cell 的噪声与真实命中按 `t_obs` 共同参与 100 ns 非延长死时间，同一 Cell 可保留多个间隔足够远的命中。STEP1 默认使用最多 4 个 CPU 进程，可通过 `--workers` 或 `MACE_DATA_WORKERS` 调整。

STEP2 默认使用低显存配置：`BATCH_SIZE=2`、`VAL_BATCH_SIZE=2`、`ACCUMULATION_STEPS=16`、CUDA AMP 开启，并固定训练种子 `20260823`。可通过 `MACE_BATCH_SIZE`、`MACE_VAL_BATCH_SIZE`、`MACE_ACCUMULATION_STEPS`、`MACE_USE_AMP`、`MACE_TRAIN_SEED` 覆盖。

STEP1 默认按 100 张图一个分片写入 `*.pt.parts` 目录，不再把整个训练集、验证集和测试集同时保存在内存中；可通过 `--chunk-size` 或 `MACE_DATA_CHUNK_SIZE` 调整分片大小。STEP2 会自动读取这些分片。

模型输入只使用在线可观测的几何、带误差的 `d` 和 `t_obs`，不包含 `t0`、`tHit`、`t-t0`、`is_noise`、`shared_hit` 或窗口全局最小时间。评价按 `(EvtID, TrkID)` 逐条轨迹统计。`time` 输入维度为节点 7、边 6，旧 AI4 数据和权重必须废弃并重新生成、训练。

先用验证集比较 `no-time` 与 `time`，再按 `10/100/300/1000 kHz` 和 `sigma_t=1/3/5/10 ns` 生成独立配置复验；测试集只在参数冻结后验收。

单独生成扫描数据时使用 `--tag noise300k_sigma5` 一类标签，避免覆盖中心配置。
