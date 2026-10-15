# AI4.1 数据、权重与复现

**以下命令复现已记录的模拟实验。最新实验反馈表明精确模拟时间不可直接测得；运行成功不等于通过真实可测输入验收。** 参见[测量口径更新](MEASUREMENT_UPDATE_20261015.md)。

## 安装与外部输入

Python 3.10+，先安装仓库根目录requirements，再安装本目录补充依赖。PyTorch请按实际CUDA版本安装；V100使用FP16，不要求BF16。

```powershell
python -m pip install -r ../requirements.txt -r requirements.txt
$env:MACE_CDC_CELL_MAP = 'D:/your-data/cdc_cell_map.xlsx'
```

命令默认在`AI4.1/`下执行。几何表至少包含CellID、x、y、u、v并与数据对应。原始CSV/Excel、训练图分片、优化器断点和预测缓存不随仓库分发。

新数据可选两种准备方式：

- 将已生成的`dataset_train.pt.parts/`、`dataset_validation.pt.parts/`、`dataset_test.pt.parts/`放到`whole_event_retraining/`，同时保留对应元信息和事件划分；须与已发布模型的数据契约一致。
- 将原始`CDCSimHit_*.csv`、`mother_decays.csv`、`SimMACE.yaml`放到`whole data/`，审计后重新建图：

```powershell
python real_data_evaluation/audit_whole_data.py
python STEP1_完整事件数据集.py --workers 3
```

第二种方式会在本地生成event_inventory.csv等文件。审计脚本面向当前完整生产字段；STEP1图输入只使用几何、d、t及生成离线标签所需的ID。已有图分片非空时拒绝覆盖，避免破坏数据。

`STEP1_生成AI4数据集.py`保留的是旧AI4合成混合数据流程，另需原单粒子Excel和旧源事件划分。不要把它与完整事件建图命令混用。

## 已发布权重

[MODEL_CATALOG.json](../MODEL_CATALOG.json)列出逐文件SHA256。全部为小型推理state_dict，不包含优化器。根目录`*_clean_physics_v2.pth`是旧域初始化；`residual_*`是早期残差实验，未用于当前主结果。

| 用途 | 路径，均相对whole_event_retraining/ |
|---|---|
| no-time第一阶段77轮 | `best_interaction_network_no_time_finetune_whole_events_sigma05.pth` |
| no-time第二阶段loss最佳34轮 | `best_interaction_network_no_time_finetune_stage2_lr3e4.pth` |
| no-time第二阶段末轮62，绘图入口 | `stage2_no_time_assessment/last_epoch_62.pth` |
| 同一62轮权重的STEP4 CLI别名 | `best_interaction_network_no_time_finetune_stage2_last62.pth` |
| no-time从头最佳88轮 | `best_interaction_network_no_time_scratch_whole_events_sigma05.pth` |
| no-time从头末轮91 | `no_time_scratch_assessment/last_epoch_91.pth` |
| time完整事件33轮 | `best_interaction_network_time_expert_whole_events_sigma05.pth` |

发布时为部分no-time权重补写了推理元信息，并不代表它们完成了原计划全部轮数。62轮别名与绘图路径逐字节相同，不是另一套训练。

## 训练：使用新run-tag

先完成数据准备，再使用新的实验名以免覆盖已发布候选。

```powershell
$env:MACE_BATCH_SIZE = '16'
$env:MACE_VAL_BATCH_SIZE = '32'
$env:MACE_ACCUMULATION_STEPS = '2'
$env:MACE_USE_AMP = '1'
$env:MACE_TRAIN_SEED = '20260823'
python STEP2_训练模型.py --features no-time --dataset-root whole_event_retraining --run-tag scratch_example --lr 0.001 --epochs 100
python STEP2_训练模型.py --features time --dataset-root whole_event_retraining --run-tag time_example --init-model best_interaction_network_time_clean_physics_v2.pth --lr 0.0001 --epochs 100
```

Linux容器使用`export MACE_BATCH_SIZE=16`等方式设置相同变量。上述batch是V100 32GB测量起点，不保证适配其他显卡；本地低显存可减batch并增加累积步数。

若拥有自己这次训练的完整断点，可续训；epochs是目标总轮数：

```powershell
python STEP2_训练模型.py --features time --dataset-root whole_event_retraining --run-tag time_example --resume whole_event_retraining/training_state_time_time_example.pth --epochs 100
```

resume不能同时传`--init-model`或`--lr`。断点严格核对路径、数据文件大小/mtime、划分、批量等配置；不能把云端Linux完整断点直接当作Windows可迁移断点。发布权重仅用于推理或新阶段初始化。

## 两级寻迹评估

显式指定两个阈值：正式CLI宽松默认值并不自动读取元信息中的0.575。

```powershell
python STEP4_两级寻迹.py --features no-time --dataset-root whole_event_retraining --run-tag finetune_stage2_last62 --split validation --rounds 1500 --strict-threshold 0.70 --relaxed-threshold 0.575
python STEP4_两级寻迹.py --features no-time --dataset-root whole_event_retraining --run-tag scratch_whole_events_sigma05 --split test --rounds 1500 --strict-threshold 0.70 --relaxed-threshold 0.575
python STEP4_两级寻迹.py --features time --dataset-root whole_event_retraining --run-tag expert_whole_events_sigma05 --split test --rounds 1500 --strict-threshold 0.70 --relaxed-threshold 0.575
```

不要根据测试结果反复选择阈值、checkpoint或恢复规则。指标定义和验收限制见[训练报告](TRAINING_AND_RESULTS.md)。

## 可视化与回收

```powershell
python STEP6_事件切片可视化.py --event-range 100 200 --model last --split validation --recovery none
python STEP6_事件切片可视化.py --event-range 100 200 --model last --split validation --recovery fragments
python STEP6_事件切片可视化.py --event-range 100 200 --model last --split validation --recovery hough
python STEP6_绘图界面.py
```

区间包含两端，按EvtID选择，非表格行号；PNG与圆参数/hit关联JSON同时保存。当前STEP6选择器主要面向已验证的no-time候选，不是通用time绘图入口。默认不开启回收。

`real_data_evaluation/run_recovery_validation.py`与`STEP7_双专家验证.py`是带签名的研究驱动器，依赖本地重新生成的概率/基线缓存；这些缓存未发布，不是下载后直接可运行的无数据演示。不要移除签名检查来复用旧机器缓存。双专家只实现固定概率融合，尚无门控MoE训练或融合验收结论。

## 时间审计复核

原始数据与相同图分片就绪后依次运行：

```powershell
python tools/audit_time_leakage.py scan
python tools/audit_time_leakage.py raw
python tools/audit_time_leakage.py probes
```

输出在`whole_event_retraining/time_audit_reproduction/`，不覆盖发布报告；原始权重固定SHA256。raw阶段建立仅供诊断的时间变体，probes阶段真值干预只用于判断依赖信息，不是新的部署特征。实验采用已发布报告中的固定203验证事件。

## 历史接口警告

`STEP4_真实数据评估.py`和`STEP4_test_xlsx_绘图.py`/圆参数入口的旧time分支仍按历史`t−t0`约定处理原始Excel。这里的CSV真t0不是已验证可测参考；这些历史分支**不能用于当前time33完整事件的验收**，也不是98.73%测试结果的产生路径。当前模型应通过完整事件STEP1和上述STEP4命令评估。

部分早期研究脚本依赖未分发的临时实验缓存，保留用于审计历史，不承诺所有归档脚本可无数据一键运行。完整事件数据与旧Excel数据的时间尺度也不同，不能直接交叉喂入模型。

## 发布检查

无需大数据的接口/合成检查：

```powershell
python test_ai4.py
python test_xlsx_interface.py
python test_real_data_evaluation.py
python -m unittest test_fragment_recovery test_event_visualization
python STEP1_完整事件数据集.py --self-check
```

`test_event_visualization`的真实图库检查在未提供图库时跳过。`test_training_resume.py`、`test_whole_event_contract.py`等还需要本地训练图和已有权重，不属于纯合成检查。

发布副本仅做运行路径环境变量化、GUI默认几何路径适配、双专家首次创建父目录修正，以及缺失可选图库fixture时跳过。网络、拟合、回收的数值算法不改动；原本地实验目录保持不变。历史报告源码签名记录的是原实验源码，发布版路径适配会改变文件哈希，需重新生成缓存。
