# Transformer 联合寻迹与课程学习实验交接

日期：2026-11-07，北京时间。本文用于交给另一个对话开展独立的 Transformer 实验。本次交接只创建此文档，没有启动新训练、修改AI6程序或调整其服务器作业。

## 1. 用户目标与实验边界

用户希望在现有AI6直丝/斜丝联合GNN训练继续进行的同时，尝试Transformer架构，探索课程学习：先简单事件、再复杂事件、最终覆盖完整分布。目标是找到同一条径迹的独立hit成员，提高直丝/斜丝联合寻迹的纯度、完整度和效率。

拟合是成员合理性的证据之一，可以保留。当前效果较好的基线是二维直丝几何验收加联合斜丝归属；完整三维拟合仍为小样本原型。不要把边F1、成员匹配效率、三维拟合成功率、动量精度混为一谈。

Transformer与课程学习均是待实现/待验证方向，本文没有宣称已经训练过Transformer，也不保证一定优于GNN。新对话应实际实现、检查、训练及评估，而不是重复小规模hidden64试跑来替代正式模型。

为保护并行实验，建议新建独立项目目录，例如 `D:\pycharm\root操作\AI工程化7_Transformer`；这是建议名称，尚未创建。模型、断点、日志、守护锁、运行状态必须独立。AI6数据可只读复用，AI6现有代码、训练断点、权重与作业不应被新实验覆盖。目录或命名另有用户要求时，以用户要求为准。

## 2. 先阅读哪些地方

主工程：`D:\pycharm\root操作\AI工程化6`。历史基线：`D:\pycharm\root操作\AI工程化4 - 副本`。

| 优先级 | 本地文件/目录（相对AI6主工程） | 内容与用途 |
|---|---|---|
| 1 | `模型与数据约定.md` | 输入、倾角、网络、划分、训练和命名约定 |
| 1 | `common.py` | 正式12维节点/12维边、有限丝候选图、GNN及损失的实际实现 |
| 1 | `STEP1_生成AI6数据集.py` | 原CSV筛选、事件划分来源和压缩数据生成 |
| 1 | `STEP2_训练模型.py` | 按事件等权损失、有效批量、完整断点、验证与选模 |
| 1 | `joint_event_dataset/dataset_metadata_joint_no_time.json` | 固定事件ID、源文件/几何哈希、特征顺序 |
| 2 | `服务器训练加速说明.md`、`graph_cache.py`、`server_runtime.py` | 分片图缓存、数据加载、异步传输与显存经验 |
| 2 | `远程训练说明.md`、`无人值守训练守护.md` | SSH、Slurm、恢复及授权边界，部分正文为历史记录 |
| 2 | `STEP5_两级寻迹.py`、`two_stage_core.py` | 当前二维核心＋联合斜丝归属基线 |
| 2 | `STEP4_本机快照评估.py` | 冻结权重的完整测试边评分与简单候选诊断 |
| 3 | `STEP6_联合两级分组.py` | 无拟合分组的失败/不足对照，不能替换较好基线 |
| 3 | `STEP7_三维几何验收试验.py`、`三维拟合草稿_暂未启用.py` | 仅小样本三维几何原型，后者虽保留草稿名，已由STEP7试验性加载 |

历史聊天《AI工程化5副本》曾用于确认hidden256、8次消息计算、无时间等约定。优先读取已经落实的代码和约定文件；如需查聊天，可用应用的聊天检索/读取工具，不要要求用户重述所有已定内容。

注意历史污染：README后半、`实验结果.md`、`pilot_v1`、`mean_ablation`、`extended_sum`、`cpu_extension`、`memory_safe_gpu`主要属于早期hidden64/3次传递的小规模探索。正式AI6是从原始CSV重新制集并从随机初始化训练的256/8模型，不继承这些旧权重。文档中“1436个未暴露测试事件”的旧说法也已过时，见第7节。

## 3. 已完成的正式数据工作及不可破坏的约定

原CSV在 `D:\pycharm\root操作\AI工程化4 - 副本\whole data`，10个CDCSimHit文件、10000个完整事件。几何源：`D:\SRTP\CDC_cell_map\cdc_cell_map.xlsx`。

正式压缩数据已生成于 `AI工程化6\joint_event_dataset`，约71MB。结构包括 `events/event_00000.npz`、`wire_geometry.npy`、`event_inventory.json`、`event_split.json`和元数据。事件文件保存 `cell_id,d,hit_id,truth`；几何数组列为 `[x,y,u,v,w,l]`，按CellID索引。几何参考点为z=0的丝中点，方向单位向量，全长l。

固定划分7000训练、1500验证、1500测试；ID来自原AI4完整事件划分，不能重洗或按模型效果重选事件。生成器保留 `Good==True` 且 `CreatProc=='ExtendedDecayWithSpin'` 的原筛选。此训练/评估样本不是未经筛选的所有原始探测器hit，不能忽略输入总体差异。

必须遵守：

- 一个EvtID整体处理，不重新拼粒子、不按时间分峰；事件间不连边或共享注意力。
- 不使用t、t0、tHit或派生时间；不使用真实位置、真实方向、动量等作为输入。
- d直接使用已有模拟误差版本，不重复加噪。TrkID仅用于标签和离线评分。
- CellID只定位丝几何，HitID用于保持独立记录身份，不把整数ID当作物理位置编码。
- 重复CellID的不同HitID必须保留，不按CellID去重；不预先删折返、回旋或重复击中。
- 斜丝参考XY不是真实击中点，禁止直接当直丝漂移圆圆心送入二维拟合。

节点特征按顺序为12维：

```text
wire_x/800, wire_y/800, wire_r/800, d/10,
sin_phi, cos_phi, u, v, w, length/1600, is_stereo, signed_tilt_rad
```

`signed_tilt_rad = atan2((-y*u+x*v)/r, w)`，弧度，直丝为0，斜丝保留旋向。它是丝几何倾角，不是粒子的真实方向。

边特征按顺序为12维：

```text
dr/800, reference_xy_distance/800, dphi, dphi/(distance+1e-6),
finite_common_z_min_xy/800, min_xy_z/800, common_half_z/800,
delta_d/10, direction_dot, source_stereo, target_stereo, delta_signed_tilt
```

候选边规则：两根有限丝在共同z区间内的最小横向距离<=120mm。保留双向边，AA/AS/SS分别是直直、直斜、斜斜。它是宽松局部几何候选，不等于已确定真实hit的z或轨迹兼容性。标签为同事件同TrkID，不能用于建图、注意力mask或在线组轨。

Transformer可直接复用这套正式数据和候选图；改变架构并不要求再复制或重新清洗数据。若改变图/特征，须单独版本化，不能悄悄改变对照输入。

## 4. 当前AI6模型、训练与最新状态

AI6是独立 `JointInteractionNetwork`：共享hidden256，8次边消息计算（7次节点更新＋最后边分类），物理节点残差状态仍为12维；共享主干后有AA/AS/SS条件输出头，一次联合训练。它不是Transformer。

训练约定：AdamW，初始lr=0.001，weight_decay=0.0001，FocalLoss alpha=.5/gamma=2，clip_grad_norm=5，FP16 AMP，有效批量32。实际损失公式为 `0.5*(1-exp(-BCE))**2*BCE`。训练按每个事件先平均边损失、再按事件等权平均，最后不足32事件的累积组也正确归一；不要因图密度不同而把训练目标改成全批边等权。历史日志的train_loss展示仍按边数累计，不能与优化用的每事件等权目标混淆。

调度器：ReduceLROnPlateau按验证loss，patience10、factor0.5、min_lr=1e-6。目标200轮不是必须跑满或必然收敛的保证。断点保存模型、优化器、调度器、AMP、CPU/CUDA/Python/NumPy RNG、loader generator、轮次及历史。

本次只读检查，服务器时钟2026-11-07 00:20:43 CST的快照：

- 已完成78轮，正在79轮（当时4500/7000事件），lr仍0.001。
- 第78轮也是当前最低验证loss：0.0117751134；验证宏F1=87.7905%。
- 第78轮AA/AS/SS F1分别90.7258% / 86.6377% / 86.0081%。
- 第51轮验证loss=0.01316597、宏F1=86.5171%；目前仍有改善，不能说已经最终收敛。
- 服务器状态持续变化，本地训练目录可能仍停留在迁移前轮次。必须以远端实时状态为准，不把本地旧断点当最新权重。

当前远程项目：`/seu_share/home/zhangyongchao/213230965/network_datasets/ai6`。

当前作业1528054，gpu14，V100-SXM2 32GB，截止北京时间2026-11-07 02:30:26。检查时训练步骤为1528054.2；最新squeue显示CPU3、内存字段20156M，不要照抄早期作业的60GB内存配置。配额结束后需用户提供后续可用作业，不能假定进程能跨Slurm到期。

远端关键文件（相对远程项目根）：

```text
joint_training/joint_no_time_scratch_v1/training_progress.json
joint_training/joint_no_time_scratch_v1/training_history.json
joint_training/joint_no_time_scratch_v1/best_model_selection.json
joint_training/joint_no_time_scratch_v1/best_interaction_network_joint_no_time.pth
joint_training/joint_no_time_scratch_v1/training_state_joint_no_time.pth
joint_training/joint_no_time_scratch_v1/server_runtime.json
joint_training/watchdog_1528054.json
joint_training/training_health.jsonl
```

## 5. 数据加速、稳定性与并行资源

已在服务器一次性生成全量10000事件图缓存：

`/seu_nvme/home/zhangyongchao/213230965/ai6_graph_cache_v1`

约51.2GB，32事件/分片、313分片；float32特征无损，局部边int32落盘、读取恢复int64。缓存绑定原始元数据和构图函数源码哈希，不匹配就拒绝；不要绕过校验。每worker最多映射2个分片，返回tensor有独立buffer，淘汰映射时关闭并作文件页缓存建议回收。缓存是只读可复用资源，禁止修改正在被AI6使用的manifest/数组。

AI6当前训练batch4、累积8、验证batch8、chunk131072、workers2，固定内存和异步传输，自适应部分重算。短测端到端约16.34事件/秒，完整近期轮次约402秒，不能把短测当整轮吞吐。最大图和重算切换边界均测试过；Transformer必须重新测，不能继承batch结论。

用户希望并行以利用资源，但“显存空余”不等于“GPU算力空闲”。AI6过去有高GPU利用率记录；本次没有测量远端整卡实时利用率，不能声称gpu14存在可观空闲算力。

**首选可执行路径：AI6继续远程训练，Transformer先在本机RTX4060 Laptop 8GB独立试训。** 本机Python为 `D:\amancoda\envs\pytorch\python.exe`，Torch2.11.0+cu128；交接检查时显存1855/8188MiB、利用率9%，仅瞬时快照，启动前再查。可从batch1＋累积32、分块边解码开始，实际显存和速度决定规模。若有另一块已分配GPU，再将独立Transformer迁移过去。

若确需共享同一块远程GPU，先在不影响AI6的前提下做受控资源测量，评估两作业总吞吐、AI6轮时、GPU/CPU/RAM峰值；AI6只有3核，两个各自workers2的程序会竞争。不能因为32GB显存大就直接再开大batch。也不要直接套用AI6的训练锁或Slurm启动脚本启动Transformer，否则可能被锁阻塞或加载错误断点。并行请求不意味着可以停止AI6给新模型让资源。

SSH登录别名 `seu-gateway`，用户213230965；可复用已有SSH配置。登录节点python3为3.6，容器Python3.10，远程Torch2.5.0a0 nv24.08、NumPy1.24.4、PyG2.6.1。注意跨NumPy版本随机状态兼容。登录查询squeue需显式jupyter分区：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=yes seu-gateway '/usr/bin/squeue -h -p jupyter -u 213230965 -o "%i|%T|%N|%M|%e"'
```

曾反复出现SIGKILL与退出清理时的“28 leaked semaphore”告警。可见OOM计数和退出前内存不足以证明原因为OOM，信号发送者未确认。不能把清理警告当根因或声称已经解决。已使用正式srun步骤、完整断点和内外层守护恢复；新模型应复用思路而不是共用状态文件。

用户明确：作业1528054只允许自动恢复，取消配额由用户操作。旧作业1527110曾有连续3次无进展自动取消授权，不能扩大到新作业或Transformer。当前守护连续3次无新整轮进展会needs_attention并停止重试，不取消1528054；不要擅自更改授权。

## 6. 已完成的效果评估（都不是Transformer结果）

本机冻结权重目录：`D:\pycharm\root操作\AI工程化6\joint_training\local_evaluation_epoch051_20261106`。

权重SHA256：`83ff990062669a99bd20f1214d84fb2ff6411f0e8d034cf72c886e8ed6321a76`。这些结果均来自第51轮，不是远端第78轮。

### 6.1 固定1500测试事件的边关联

报告：上述目录下 `本机模型效果评估.md`、`snapshot_evaluation_test.json`、`snapshot_events_test.jsonl`。

阈值0.5；AA/AS/SS F1=90.03%/85.47%/84.77%，宏F1=86.76%。历史pilot排除后的1436事件子集宏F1=86.71%。简单连通分量诊断效率31.27%、假候选38.17%，但当时匹配条件为直丝纯度/完整度均70%，不能直接与下列80%/50%的验证试验相减。

### 6.2 固定200验证事件的两级寻迹

报告：`two_stage_validation_v1/两级寻迹可行性评估.md`。确切事件和参数：`two_stage_validation_v1/experiment_contract.json`。随机种子20261106，从1500验证事件无放回抽200；共同真轨分母5969条，要求至少5个直丝hit，匹配要求直丝纯度>=80%、完整度>=50%。假轨与克隆分母为输出候选。

| 方案 | 直丝真轨匹配效率 | 假候选比例 | 克隆比例 |
|---|---:|---:|---:|
| 简单联合连通分量 | 39.74% | 33.76% | 0.00% |
| AA两级几何寻迹 | 93.95% | 3.72% | 0.03% |
| AA两级＋联合斜丝归属 | 93.95% | 3.72% | 0.03% |

最后一项斜丝归属不反改直丝核心，因此直丝效率相等是约定，不证明斜丝无用或已经提高直丝效率。GNN前向本身始终共同处理直/斜丝，尚未做去掉斜丝输入的因果消融。

最后一项已匹配候选内斜丝纯度99.16%、完整度79.38%；全部合格真轨斜丝hit正确找回72.87%；直/斜丝成员同时达标83.56%。联合达标另要求真轨>=3斜丝hit，且斜丝纯度>=80%、完整度>=50%，分母5968。不能只引用条件纯度99%而忽略覆盖率。

基线实现：双向概率取均值；AA首级/次级阈值0.70/0.575，二维拟合RMS1.0/1.5mm，层级/弱树边拆分与最多2个异常hit剔除；AS/SS多证据和候选竞争分配斜丝，有限斜丝XY投影与圆的距离只作为必要条件，不是完整三维验收。整个事件约1.13秒累计CPU事件处理量（200事件总计226.6秒；两worker墙钟更短），不要据此声称重拟合没有成本。

### 6.3 两个额外探索

- `joint_membership_validation_v1/联合两级成员分组评估.md`：完全不拟合的完整链接核心＋补点，最佳所试阈值0.80/0.60，直丝效率63.09%、假候选18.13%、克隆1.93%、联合达标27.14%，尚弱于几何基线；不能推广成所有无拟合方法都不行。
- `geometry_3d_gate_005_v1/三维几何验收评估.md`：5验证事件，固定横向圆后拟合单段螺旋的纵向两参数，以实际有限丝段三维距离减d作残差。129原候选中115通过；过滤2假候选，同时使12条原本正确匹配真轨未通过。约87.3秒。原因包括残差/退化、多解、hit不足、旋向不足。通过者假轨0仅为小样本，不外推。未使用真实z或动量验证参数精度。

三维距离实现已与独立标量最小化器比较，并做非退化合成参数恢复；仍受固定圆、单段螺旋、相位区间、有限初值搜索等限制。当前保留全部129原候选，单独记录compatible/pending，见 `membership_candidates_with_geometry_flags.jsonl`。三维失败不能一律判作成员错误。用户最终意见是保留必要拟合，让拟合服务于成员合理性判断。

## 7. 验证、测试及公平对照

完整1500测试事件已经在第51轮模型上查看，1436事件子集也已暴露，不能再称“全新未见测试”。记录在 `AI工程化6\joint_training\evaluation_exposure_history.jsonl`。新Transformer/课程阈值只在训练/验证侧设计和选择；如后续在原test作比较，必须披露它已被查看，需要新的独立数据才能确认泛化。不要不断测试再调参。

先比较同一模型输出接口与同一STEP5后处理，固定200验证事件、几何阈值和80%/50%评分。若针对Transformer重新校准阈值，在验证集完成并明确标为“各自校准”对照，与固定阈值对照分开。三种边F1、precision/recall、轨迹效率、假轨/克隆、斜丝全局找回及联合达标一起报告。

建议至少形成：AI6 GNN基线、Transformer普通随机训练、相同Transformer课程学习。不要同时改数据、图、损失、采样和后处理，再把收益全部归因于Transformer或课程学习。硬件不同的运行时间不直接比较；同GPU对照需记录同等资源。

## 8. Transformer第一版建议（待实施，不是历史定论）

优先试几何感知Graph Transformer：每个独立hit为token，投影到hidden256，多头注意力真正使用Q/K/V、归一化、残差及FFN；边几何进入相对位置/attention bias或edge message。可以从4层、8头、FFN宽度1024起步，若资源支持再对照6/8层。这是起点建议，不是已定必须参数；GNN的“8次消息计算”不能等同Transformer的8层。

第一版输出仍为候选边二分类、AA/AS/SS条件头，与既有评分/后处理对接。不要一开始同时改成轨迹ID分类、端到端集合预测或三维参数回归。固定数据下从随机初始化训练，不加载AI6 GNN权重。

注意力必须隔离不同事件；保持hit顺序置换等变，禁止把任意输入行序/整数HitID作为绝对位置编码。可沿现有候选边做稀疏注意力，优先控制N平方和边MLP内存。若用密集注意力，padding mask和事件mask要检查；若缩小邻域、截断token或采样边，要单独记录并验证，不能悄悄丢掉密集困难事件。V100不具备与新GPU相同的BF16/FlashAttention条件，不能假定某个加速后端必然可用；先测当前库实际支持的FP16路径。

初版尽量复用事件等权Focal目标、有效批量32、clip5及固定评估。Transformer可能需要较小lr或warmup，允许新实验单独记录优化策略，不能改AI6的调度。hidden相同不代表参数量相同，需同时报告参数量、优化步数、处理事件数和实际训练时长。

先完成：真实简单/中等/最密集训练事件前向反向、有限梯度、跨事件隔离、置换等变、无truth/time输入、重复CellID保持、padding/缓存一致性，以及一次完整训练+验证+断点恢复。不要只做GPU前向短测就宣布可稳定训练。

## 9. 课程学习建议（用户同意探索的方向）

“分层训练”首选理解为同一网络的训练课程，而不是拆成独立直丝/斜丝专家。简单阶段也共同输入直/斜丝，避免早期只见直丝而后期突然更换输入分布。

从7000训练事件内部，用可观测复杂度固定排序，例如hit数N、候选边数E、平均度/局部占用密度、直斜丝比例；原始清单的hit数可先用于粗分，后续丰富指标。不能按测试表现定义难度；TrkID数量、真实动量或真实轨迹形状不得作为线上输入。训练标签可用于离线诊断困难样本，但首版难度分级优先使用可观测量以便复现与部署解释。

一个可落地起点：

| 阶段 | 采样范围 | 初始预算建议 |
|---|---|---|
| 简单 | 按固定复杂度排名较低的30%训练事件，仍包含两种丝 | 约10个该子集epoch |
| 混合 | 扩到较低70%，混合重放简单事件 | 约20个该子集epoch |
| 完整 | 全7000事件，逐步过渡到均匀采样 | 占主要训练预算，按完整验证曲线判断停止 |

以上比例/轮数未验证，可由新对话按吞吐和验证曲线调整，并在开跑前写进配置。不要让验证集也随课程变简单，不要将困难事件从验证分母移除。最终必须充分覆盖最复杂事件，且持续重放简单事件，检查遗忘。

课程子集的epoch大小不同，不能直接说“都200轮所以预算相同”。记录累计处理事件数、optimizer steps和GPU时长，普通Transformer对照尽量匹配实际更新/事件预算；全量阶段单独报告。每个事件在有效batch内等权，阶段内末尾累积组处理仍正确。

阶段切换默认保留同一模型、optimizer、AMP及RNG状态；不自动重置优化器。学习率、warmup与scheduler是否按阶段变化需预先写清。新增断点字段至少包括课程阶段、阶段内轮次/采样位置、难度清单与哈希、采样器状态、累计事件数/步数，防止重启后重复简单阶段或跳过困难阶段。

每轮监测完整固定验证集的loss/AA-AS-SS，并按固定复杂度分桶报告。阶段边界用固定200验证事件评估寻迹，不要每阶段重新选一批“好看的”事件。课程是否有效应以完整分布、尤其困难事件的最终指标为准，不以简单阶段低loss为证据。

## 10. 实现和命名要求

沿用 `common.py`、`STEP1_…数据集.py`、`STEP2_训练模型.py`、`STEP3_推理评估.py`、`test_*.py` 的命名风格。新架构可单独 `transformer_model.py`、课程采样可单独模块，保持明确入口。使用独立run-tag（如 `transformer_joint_uniform_v1`、`transformer_joint_curriculum_v1`）。避免“final/best_new2”式无法追溯名称。

显式传入dataset-root和output-root；直接复制AI6脚本后其common.ROOT、MODEL_ROOT及启动脚本路径可能仍指向AI6，开写前逐项核对。复用评估函数时避免Python同名common模块冲突；不要意外导入AI4旧time模型或不同输入维度。

AI6已具备的合约/恢复检查：`test_joint_contract.py`、`test_training_supervisor.py`、`test_slurm_watchdog.py`、`test_snapshot_evaluation.py`、`test_two_stage_joint.py`、`test_joint_membership.py`、`test_helix_geometry.py`。测试范围不同，不能把旧GNN测试通过当作新Transformer已验证。

当前最新权重在远端；取模型做比较时要下载到新的冻结目录，保证权重、best_model_selection和model_metadata对应同一版本并核对SHA256。禁止覆盖本机第51轮快照。原STEP5对模型类有硬编码，新Transformer应做明确适配或独立预测缓存，不可仅替换pth路径就认为可用。

## 11. 给新对话的建议执行顺序

1. 读取本交接和第2节核心文件，只读确认AI6最新作业与可用本机/远程资源。
2. 建立独立项目与配置，绑定AI6正式数据的只读路径；保存完整数据/代码/环境清单。
3. 实现最小Graph Transformer，先确认输入输出、掩码、内存、梯度和完整恢复；做真正数据到优化器的吞吐测试。
4. 先启动可与AI6共存的独立普通Transformer基线；课程学习用独立run-tag、相同初始种子及可比较预算。资源不足时先准备第二实验而不是在同卡上强行并跑。
5. 保存完整曲线/断点，训练期间监测完整验证分布和复杂度分桶，不依赖测试调参。
6. 用同一200验证事件、同口径两级成员寻迹对照，并明确模型、阈值、后处理和几何验收版本。
7. 报告已完成的实际训练规模、当前收益、失败样本、显存/内存/吞吐和下一步。继续依据真实曲线推进，而不是承诺次日一定收敛。

本交接没有要求新对话替主线程管理AI6训练，也没有自动批准取消现有GPU配额。训练并行、数据复用与模型对照应保持可追溯且互不覆盖。
