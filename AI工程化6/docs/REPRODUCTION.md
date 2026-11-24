# 获取、复现和路径迁移

## 获取完整权重和图片

```bash
git lfs install
git clone https://github.com/AHSASSD/MACE-CDC-2D-Tracking.git
cd MACE-CDC-2D-Tracking
git lfs pull
cd AI工程化6
```

模型和图片可能由Git LFS管理。GitHub网页上的指针文本不是权重；需要完成LFS下载。仅加载自己信任的检查点；包含优化器/RNG的旧恢复状态可能需要`torch.load(..., weights_only=False)`。

## 环境

建议独立Python 3.10+环境。实际本机版本见ENVIRONMENT_LOCAL.json，云端使用过pytorch-24.08容器；这两者不是严格相同环境。依赖见requirements-handoff.txt，先运行结构检查再训练。若无法安装相同CUDA版本，可先用CPU加载检查点和运行小图测试。

```bash
python -m pip install -r requirements-handoff.txt
python experiments/联合归属训练_20261123/check_repair.py
python experiments/联合归属训练_20261123/check_batch.py
python tools/verify_handoff.py
```

`verify_handoff.py`校验所有目录内模型文件的SHA256与MODEL_CATALOG，不启动训练，不连接服务器。

## 数据

`joint_event_dataset/events/event_XXXXX.npz`是逐事件观测和离线真值；`wire_geometry.npy`给出几何，dataset_metadata_joint_no_time.json给出特征、来源哈希和7000/1500/1500划分。已附这份轻量事件数据，训练无需重新读原Windows CSV。部署时只提供观测字段，真值用于评价。

大型旧train.pt/test.pt/validation.pt是预展开图数据集，不是模型权重，未当作网络权重上传。数十GB图分片与冻结latent缓存需要再生成或取得原服务器副本；EXCLUDED_REGENERABLE_FILES.json列出本机排除项。需要训练可先不启用MACE_GRAPH_CACHE，以原始事件惰性构图（较慢），或运行graph_cache.py生成校验过的分片。

## 权重入口

按MODEL_CATALOG中的路径和SHA256选择，不按最新修改日期选择：

- 稳定126主干：SHA256 `5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac`。
- T25：SHA256 `46a9d457a19e3e81112534f52e5a76e0437baa26b06c84f98fcd29f0e3b3eed2`；需配匹配的主干和T结构。
- 新Membership：使用对应实验目录的membership.py/fast_model.py；旧InteractionNetwork不能加载其state_dict。
- 云端OC全量最新恢复状态：`cloud_artifacts/ai6_membership_20261123/full_data_stage1/run/state.pth`。它包含第10轮部分状态，不是“训练完成权重”。
- 云端原始各轮、best及state全部保留在cloud_artifacts下；文件名best不保证跨流程最好，必须读对应选择记录。

## 稳定推理和六面板

在AI工程化6目录先查看相应入口的`--help`，然后按原报告选择快照。例如：

```bash
python STEP4_联合寻迹推理.py --help
python STEP18_粗三维寻迹与GenFit导出.py --help
```

旧代码中仍有Windows绝对路径、原服务器路径或冻结contract内的路径引用。发布快照保存原源码以保持既有哈希，不批量改写它们。迁移时优先使用入口的dataset/snapshot/weight/output参数，并根据错误定位明确修改实验副本。修改后生成新contract；不要绕过旧哈希断言再宣称完全复现。

重点旧位置：本机源目录`D:\pycharm\root操作\AI工程化6`，Python环境`D:\amancoda\envs\pytorch`，几何原表`D:\SRTP\CDC_cell_map\cdc_cell_map.xlsx`；服务器根目录见CLOUD_FILE_INVENTORY的source字段。这些是历史位置，不要求同学建立相同盘符或获得原账号。仓库相对路径才是交接入口。

## 服务器训练

Slurm脚本保存了当时的账户、队列、挂载、容器和绝对目录，仅作为可审计模板。接收者应改成自己的分配与路径，不使用原账号凭据。发布包不包含密钥或登录令牌。

全量OC阶段源代码为experiments/联合归属训练_20261123/train_full_stage.py；部署时的精确副本和USE_TWO_WORKERS标记在cloud_artifacts中。训练恢复必须同时匹配数据、代码、初始权重和数学配置摘要；唯一已记录的加载进程调整有明确迁移记录。不要把热启动扩量等同于C/D从头训练200轮。

## 两人协作建议

共享同一版本的元数据、固定开发事件与匹配规则；模型侧输出原HitID、边/T分数和可选实例嵌入，提取器侧只消费这些观测输出。各自独立Git分支/实验目录；交付时提交源码、权重SHA、阈值来源、逐事件计数与找回/损失清单，再按事件配对复核。GenFit同学消费明确标注的确认成员与未定成员，不默认接受全部斜丝。
