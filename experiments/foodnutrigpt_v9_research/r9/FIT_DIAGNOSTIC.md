# 首组完成后的训练拟合诊断

## 目的与边界

在线训练loss包含187轴、来源校准和正则，且在训练模式及不断更新的模型上计算；验证主指标是推理模式下142轴的来源/食品组宏平均误差。两者不能直接相减归因于过拟合，也不能仅凭在线训练loss判断需要更大模型。

本诊断在两组完整训练、独立审计、预算分析和学习率直接对照都结束后，使用各自已选定的检查点重建全部训练家族任务。采用与验证相同的source-free推理和评分规则，得到更可解释的训练拟合、正值/零值及逐轴支持信息。诊断不更新参数、不拟合变换、不训练基线，也不改变冻结数据或验证任务。

## 实现与覆盖

独立入口`diagnose_foodnutrigpt_r9_fit.py`复用已有完整训练拟合诊断的评价逻辑，改用R9专用公开加载器；原MLP诊断和所有冻结执行文件不改。

仅在内存中浅复制评价对象，用训练分区实际已观测的187轴单元构建评分请求，共1,828,536个。显式零保留，未观测标签排除；逐档案完整隐藏目标家族，名称和数值输入仍遵循原协议。保存训练逐轴指标、候选组支持数与主指标；验证分数直接复用已独立审计的保存结果，不额外重跑验证选点。

数字预测保存在`data/local/research_diagnostics/v9_r9_first_group_fit_v1/<run>/completion_predictions.parquet`，汇总及逐轴表保存在`reports/v9_r9_first_group_fit_v1/`。目录已经存在时拒绝覆盖。开始/结束检查冻结输入和检查点身份；记录查询面板、预测、代码及数据哈希。

## 解释规则

两个模型的训练误差差异是在相同训练数据与任务上的实际优化设置比较。训练与验证仍有不同食品、来源构成和轴支持，误差差距本身不能唯一证明过拟合、确定容量瓶颈，或证明增加容量没有价值。结合完整曲线及验证正值/零值、稀疏轴和来源分解，再决定下一项有控制的方法实验。

此处仅诊断source-free预测；不将已知来源校准路径的训练内拟合当成模型的部署能力。所有数字仍属于训练/内部验证分析，不打开历史测试，不改变Transformer优于固定RF的完成条件。

## 执行与当前状态

```powershell
.venv/Scripts/python.exe scripts/diagnose_foodnutrigpt_r9_fit.py --check-only --output-dir reports/v9_r9_first_group_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_first_group_fit_v1
.venv/Scripts/python.exe scripts/diagnose_foodnutrigpt_r9_fit.py --output-dir reports/v9_r9_first_group_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_first_group_fit_v1
```

编译和真实依赖预检通过：当前父分析尚未完成，预检正确返回未就绪且没有写输出。正式全量诊断尚未执行；完成后仍需要检查实际结果并纳入首组八节研究报告。
