# 参考文献与适配范围

1. A. Correia et al., **Graph Neural Network-Based Pipeline for Track Finding in the Velo at LHCb**, arXiv:2406.12869。[原文](https://arxiv.org/html/2406.12869v3)。借鉴共享命中和三元组兼容图、边与T联合监督。VELO提供前向平面序和直接位置；本项目CDC存在盘绕/返回轨，不能直接采用全局z或半径单调规则。
2. L. Reuter et al., **End-to-End Multi-Track Reconstruction using Graph Neural Networks at Belle II**, arXiv:2411.13596。[原文](https://arxiv.org/html/2411.13596v2)。CAT Finder在漂移室wire hits上学习实例空间和凝聚置信度。其TDC/ADC输入及运动学任务与本项目不同，不借用论文效率作为本项目预期。
3. J. Kieseler, **Object condensation: one-stage grid-free multi-object reconstruction in physics detectors, graph and image data**, Eur. Phys. J. C 80, 886 (2020), arXiv:2002.03605。[论文](https://arxiv.org/abs/2002.03605)。本项目OC-v1/v2/hard_v1是有明确公式和负结果记录的适配，不声称完整复现。

本项目内部参考：原AI4/AI5版本、用户提供的Z.L. Yu元胞自动机三维几何逻辑及历史对话。内部代码来源以仓库和文件记录为准；精确拟合交给GenFit，当前报告仅评价成员与粗三维种子。

中文最新结果和所有数值以RESULTS.md链接的本项目原始报告为证据，外部文献不是这些数值的来源。第三方论文只提供阅读链接，不额外转载全文，也不替第三方代码授予许可。
