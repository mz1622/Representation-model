# 无 food name 的 masked-axis Transformer 架构搜索报告

## 结论

按当前“先不使用蒸馏，只训练一个模型”的约束，推荐版本更新为 **v16 ReGLU＋按 axis 对学习的 attention bias**：营养 142 轴 `0.231242`、食品代谢物 45 轴 `0.598959`、全部 187 轴 `0.319730`。相对父模型 v7 ReGLU，三项汇总指标分别改善 `0.002883 / 0.000379 / 0.002280`，187 个轴中有 138 个改善。

本轮还测试了 QK-Norm（v15）和 Talking-Heads（v17）：前者与 v7 基本持平，后者明显退化，因此没有与 v16 叠加。v16 没有使用教师、软目标、food name、source ID 或多模型预测；最终和训练中的模型数都为 1。完整测试集保持关闭。v12 蒸馏和 v11 ensemble 只保留为历史实验。

## 补全接口

所有正式 Transformer 都遵守同一核心原则：目标值由目标 axis token 补全，context token 不连接 187 维输出头。

```text
可见 token = axis embedding + numeric encoder(value)
目标 token = target axis embedding + [MASK_VALUE]
                  ↓
        无位置编码 Transformer
                  ↓
        每个目标 axis token 的隐藏状态
                  ↓
          共享标量 expression decoder
```

每个任务加入该成分家族中固定的全部可监督目标轴；单条记录是否有真实标签只控制 loss mask，不控制目标 token 是否出现，避免标签存在性泄漏。`[FOOD]`/context token 只保留样本表示。MVC 实验曾让它承担训练辅助重建，但结果变差；当前 v16 不使用 MVC。

## 固定协议

| 项目 | 固定设置 |
| --- | --- |
| 训练输入 | axis ID、可见连续值、固定目标 axis token；无 food name/ID/source |
| 数值目标 | 训练正值中位数尺度下的 `log1p(raw/s_axis)` |
| 数据 | 64,700 个训练 profile；11,175 个冻结验证 profile；323,809 个验证 job |
| 任务 | 现有成分家族遮蔽；187 个轴宏平均 |
| 损失 | 只在观测目标上计算的逐轴加权 MAE |
| 主干 | 维度 128、4 heads、2 blocks、FFN 256、batch 128 |
| 优化 | AdamW，学习率 `3e-4`，weight decay `1e-4`，gradient norm 1 |
| 训练与选点 | 20 轮；按营养 142 轴 scaled log MAE 选 checkpoint |
| 数据清单 SHA-256 | `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923` |

本轮没有把修改 seed、dropout、学习率或训练轮数当作架构搜索。种子只用于 v3 的三个独立重复；后续候选统一使用 `20261005` 做单因素筛选。

## 架构实验结果

| 版本 | 主要改变 | 参数/成员 | 营养142 | 代谢物45 | 全部187 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v3，seed 20261005 | masked axis＋普通 GELU FFN | 347,649 | 0.243748 | 0.586440 | 0.326214 | 父控制 |
| v4-A | 训练分位点 PLE 数值嵌入 | 588,929 | 0.239170 | 0.624719 | 0.331949 | 代谢物退化 |
| v4-B | visible encoder＋target cross-attention | 613,377 | 0.236287 | 0.611643 | 0.326613 | 与父控制相当 |
| v4-C | 16 个 learned latent set | 615,425 | 0.262898 | 0.631444 | 0.351586 | 信息瓶颈明显 |
| v4-D / v7 | 参数量匹配 ReGLU FFN | 348,077 | 0.234125 | 0.599338 | 0.322010 | 本轮父模型 |
| v4-E | 显式 natural-missing axis token | 347,777 | 0.246167 | 0.596247 | 0.330411 | 更慢且无收益 |
| v4-F | ReGLU＋scGPT MVC 辅助 | 380,973 | 0.236566 | 0.623069 | 0.329575 | context 辅助干扰主任务 |
| v4-G | ReGLU＋每轴 decoder residual | 380,585 | 0.238775 | 0.637489 | 0.334722 | 代谢物过拟合 |
| 3×v3 等权 | 三个 masked-axis seed | 3 个成员 | 0.225859 | 0.574525 | 0.309762 | 已略低于目标 |
| v4-H / v11 | 3×v3＋1×ReGLU，scaled-log 等权 | 4 个成员 | 0.223433 | 0.571067 | 0.307088 | 训练教师；不作为最终部署方式 |
| v12 | 四教师稠密软目标蒸馏至一个 ReGLU student | 348,077；1 个模型 | 0.228699 | 0.572968 | 0.311545 | 暂停：当前不采用蒸馏 |
| v13 | ReGLU gate 改为 SwiGLU/SiLU | 348,077；1 个模型 | 0.236900 | 0.619182 | 0.328893 | 淘汰：差于 ReGLU 和普通 GELU |
| v15 | ReGLU＋QK-Norm attention | 348,085；1 个模型 | 0.233167 | 0.602714 | 0.322096 | 与 v7 持平，不叠加 |
| **v16** | **ReGLU＋按 axis 对学习的 attention bias** | **860,149；1 个模型** | **0.231242** | **0.598959** | **0.319730** | **当前最佳无蒸馏单模型** |
| v17 | ReGLU＋Talking-Heads attention | 348,141；1 个模型 | 0.239960 | 0.624023 | 0.332382 | 淘汰：两组轴均退化 |

### 为什么 ReGLU 有效

v3 block 的普通 FFN 是 `Linear(128,256) → GELU → Linear(256,128)`。ReGLU 版本把它换成 `Linear(128,342) → split → left×ReLU(gate) → Linear(171,128)`；`171≈256×2/3`，因此门控后的参数量与普通 FFN近似相同。模型总参数只从 347,649 增至 348,077，收益不能用明显增大容量解释。

门控让网络按样本和隐藏维度选择哪些特征通过 FFN。结果相对同种子 v3：营养改善 0.009623，代谢物恶化 0.012898，但 142 个营养轴占全部轴的多数，因此全轴改善 0.004204。[FT-Transformer](https://arxiv.org/abs/2106.11959)及其[官方实现](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py)是该门控设计的依据。

### 为什么 ensemble 达标

四个成员的单次误差并不相同，尤其代谢物轴存在较强随机波动。直接在 `z` 空间平均相当于对非负原值做尺度一致的几何型融合，能抵消各模型方向不同的误差。三个 v3 seed 的等权平均已经把单模型均值 0.326706 降到 0.309762；加入结构不同的 ReGLU 后进一步降到 0.307088。

采用等权是为了保持规则简单且可复现。四个权重在运行前固定为 0.25，没有用验证集拟合 stacking 模型。近期表格深度学习也发现 parameter-efficient ensemble 可显著改善稳定性；本轮只使用已有独立训练模型的普通 deep ensemble，不改各成员训练配置。[TabM 论文](https://openreview.net/pdf?id=Sd4wYYOhmY)提供了相关公开证据。

### 如何把多个教师压成一个模型

v12 把 v11 从“推理规则”改成“训练监督”。对 337,048 个训练 row-family task，四个冻结教师分别输出该家族全部目标 axis token 的 scaled-log 预测，再在 `z=log1p(raw/s_axis)` 空间等权平均。这样即使某一训练记录只观测到家族内少数目标轴，学生仍能在固定的全部目标 query 上得到稠密软目标。软目标只为训练 partition 生成；验证标签没有进入蒸馏。

学生从随机初始化训练，结构完全沿用 v7：同一个 ReGLU Transformer、同一个目标 axis token 标量 decoder。每个 batch 的损失为：

```text
0.5 × 真实观测标签的逐轴宏平均 MAE
+ 0.5 × 教师稠密软目标的逐轴宏平均 MAE
```

两个损失都按轴总支持量归一化，避免 60,536 个 mineral task 压过只有 35 个 task 的 lignan 家族。训练 20 轮，按既定营养指标选中第 18 轮。学生对教师的训练 MAE 从第 1 轮 `0.244778` 降到第 20 轮 `0.082184`；验证全轴结果从 v7 的 `0.322010` 降到 `0.311545`。导出文件不含任何教师 state dict，已通过只加载学生 checkpoint 的独立推理检查。

### SwiGLU 单模型结果

SwiGLU 保持 v7 的输入、masked target axis token、attention、decoder、hidden width、优化器和 20 轮预算不变。FFN 唯一变化为：

```text
ReGLU: left × ReLU(gate)
SwiGLU: left × SiLU(gate) = left × gate × sigmoid(gate)
```

二者的 gated hidden width 都是 171，总参数均为 348,077。SwiGLU 按既定营养选点规则选择第 18 轮，营养 `0.236900`、代谢物 `0.619182`、全轴 `0.328893`。相对 ReGLU，全轴恶化 `0.006883`；相对普通 GELU v3 也恶化 `0.002679`。

逐轴看，SwiGLU 在 119/187 个轴上改善，中位变化为 `-0.002221`，但少数轴产生严重离群退化。例如 isomeric linolenic acids (18:3) 增加 `0.894792`、benzoic acid 增加 `0.511245`、oxalic acid 增加 `0.412453`。由于正式指标是逐轴宏平均，这些稀疏轴的不稳定性超过了多数轴的小收益。当前不保留 SwiGLU。

### ReGLU 上的非 trivial attention 优化

本轮固定 v7 的 ReGLU FFN、数值编码、decoder、训练损失、学习率、dropout、20 轮预算和种子，只改变 attention 的信息交互方式。

1. **QK-Norm（v15）**：每个 head 的 query 和 key 投影先做 L2 归一化，再乘每个 head 独立学习的温度：`softmax(tau_h * normalize(Q) * normalize(K)^T)`。它限制 attention logit 的大小由向量夹角和温度决定，减少 query/key 范数漂移。设计依据来自 [QKNorm](https://aclanthology.org/2020.findings-emnlp.379/)。结果为 `0.233167 / 0.602714 / 0.322096`，与 v7 的总体误差只差 `+0.000085`，没有形成稳定收益。
2. **按 axis 对学习 attention bias（v16）**：对每层、每个 head、每个有序 `(query axis, key axis)` 学一个标量 `B`，attention 变为 `softmax(QK^T/sqrt(d) + B_h[a_q,a_k])`。它让模型直接表达“某个成分轴在预测另一个成分轴时通常应被多看或少看”，同时仍由样本数值决定 Q/K/V。该设计借鉴 [Graphormer](https://arxiv.org/abs/2106.05234)把结构关系作为 attention bias 注入的思路。bias 全零初始化，因此初始函数和 v7 完全一致。结果为 `0.231242 / 0.598959 / 0.319730`，是本轮唯一同时改善营养、代谢物和全轴指标的候选。
3. **Talking-Heads（v17）**：在 softmax 前后分别用一个跨 head 的线性映射，让不同 attention head 交换 logit 和概率信息；两组矩阵均从单位阵初始化。设计依据来自 [Talking-Heads Attention](https://arxiv.org/abs/2003.02436)。结果为 `0.239960 / 0.624023 / 0.332382`，说明当前任务中的 head 独立性更合适。

v16 的代价是参数量从 348,077 增至 860,149，正式训练耗时从约 747 秒增至约 1,620 秒。它距离 `0.31` 目标仍有 `0.009730`，而且目前只有一个种子，因此应视为新的结构候选，而不是已经确认的稳定提升。

## 负结果解释

1. **PLE 改善营养、损害代谢物。**逐轴分段数值嵌入增加了 24 万参数，在稀疏代谢物任务上泛化更差；不与后续结构叠加。
2. **cross-attention 没有解决瓶颈。**目标 query 独立读取可见 memory 的结果与 v3 相当，说明目标 token 在联合 self-attention 中互相出现并不是主要误差来源。[Perceiver IO](https://arxiv.org/abs/2107.14795)是该候选的依据。
3. **latent set 丢失细粒度轴信息。**把可见组成压入 16 个无轴身份 latent 后，两组轴都退化。[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)式瓶颈不适合当前补全任务。
4. **显式 missing token 无收益。**当前稀疏 set 已能通过 token 是否出现表达缺失；加入所有缺失轴使每轮从约 37 秒增至约 68 秒且精度变差。
5. **MVC 支持不让 context token 承担补全。**按 [scGPT 官方 MVCDecoder](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py)加入等权辅助损失后，代谢物误差从 ReGLU 的 0.599338 升至 0.623069。即使 inference 只使用 axis token，context 全轴压缩任务仍会干扰主表示。
6. **每轴输出头加重负迁移。**零初始化的 axis-specific residual 没有改善；代谢物误差升至 0.637489，说明当前共享 decoder 的跨轴统计共享是有用约束。
7. **QK-Norm 只改变了注意力标度。**归一化 Q/K 后，营养略好但代谢物略差，全轴与 v7 持平；不继续组合。
8. **Talking-Heads 引入了有害的 head 混合。**即使从标准 attention 的恒等函数开始，正式结果仍在两组轴上退化；不继续组合。

## 三个传统 baseline 的技术细节

三个 baseline 与 Transformer 使用相同可见单元格。每个目标轴单独构造 504 维 dense 输入：前 252 维是可见轴的 scaled-log 数值，缺失或目标家族位置填 0；后 252 维是 observation mask，因此真实 0 与缺失可区分。

- **Random Forest**：每个目标轴独立训练 200 棵回归树；bootstrap sampling，平方误差分裂，`min_samples_leaf=2`，每次分裂最多查看 50% 特征。营养/代谢物/全轴为 `0.178649 / 0.628245 / 0.286841`。
- **XGBoost**：每轴 300 棵深度 6 的 histogram tree；learning rate 0.05，`min_child_weight=5`，row/column subsample 都为 0.8，L2 regularization 1。结果为 `0.190612 / 0.633525 / 0.297195`。
- **KNN**：每轴在有该目标标签的训练 profile 中找 16 个欧氏近邻；邻居贡献为 `训练权重/(distance+1e-3)`，对 scaled-log 目标加权平均。结果为 `0.239212 / 0.677675 / 0.344725`。

RF 和 XGBoost 的全轴结果仍优于当前 v16，主要来自营养轴；v16 的代谢物误差 `0.598959` 则优于 RF、XGBoost 和 KNN。v16 全轴优于 KNN，但与 RF 相差 `0.032889`、与 XGBoost 相差 `0.022535`。

## 完整性与限制

- v12 学生和四个教师均记录 `food_name_model_input=false`、`source_model_input=false`、`complete_test_opened=false`，并共享同一数据清单哈希。
- v12 只在已有冻结内部验证集验证，尚无独立 external validation；0.311545 不能外推为外部数据性能。
- 三个 v3 成员来自独立 seed，ReGLU 只有 seed `20261005`；ensemble 没有第二组四成员重复。
- v12 目前只有 seed `20261005`，尚未做独立种子复验；它距离 0.31 仍有 0.001545，不能写成已经达标。
- 蒸馏降低推理成本，但训练阶段仍需要四个教师 checkpoint 和约 73 MB 的软目标缓存。正式发布只需保存学生 checkpoint、轴映射和训练尺度。
- v13 SwiGLU 只有 seed `20261005`；因为同种子、同参数量实验已经明确差于 ReGLU，本轮不追加种子复验。
- v15、v16、v17 均只有 seed `20261005`。v16 的提升较小，必须在独立种子上复验后才能判断稳定性。
- v16 的 axis-pair bias 新增 512,072 个参数，正式训练约为 v7 的 2.17 倍耗时；收益需要与这部分成本一并评价。
- 本轮没有重新设计成分家族；所有实验沿用既定 family mask。

## 复现

单模型训练入口分别位于：

- `scripts/run_foodnutrigpt_no_foodname_v4.py`：PLE
- `scripts/run_foodnutrigpt_no_foodname_v5.py`：target cross-attention
- `scripts/run_foodnutrigpt_no_foodname_v6.py`：latent set
- `scripts/run_foodnutrigpt_no_foodname_v7.py`：ReGLU
- `scripts/run_foodnutrigpt_no_foodname_v8.py`：missing-aware tokens
- `scripts/run_foodnutrigpt_no_foodname_v9.py`：MVC auxiliary
- `scripts/run_foodnutrigpt_no_foodname_v10.py`：axis-specific residual
- `scripts/run_foodnutrigpt_no_foodname_v13.py`：SwiGLU 单模型
- `scripts/run_foodnutrigpt_no_foodname_v15.py`：QK-Norm attention
- `scripts/run_foodnutrigpt_no_foodname_v16.py`：按 axis 对学习 attention bias
- `scripts/run_foodnutrigpt_no_foodname_v17.py`：Talking-Heads attention

最终 ensemble：

```powershell
.venv\Scripts\python.exe -X utf8 scripts/ensemble_no_foodname_transformers.py `
  --output-dir output/no_foodname_v11_equal_log_ensemble_20261005
```

当前单模型蒸馏版本：

```powershell
.venv\Scripts\python.exe scripts/run_foodnutrigpt_no_foodname_v12.py `
  --output-dir output/no_foodname_v12_distilled_student_20ep_20261005 `
  --epochs 20 --device cuda
```

v12 输出另含 `transformer.pt`、逐轮 `history.csv` 和训练专用 `teacher_targets.npy`。部署只需要单个 `transformer.pt` 及数据预处理元数据。逐版本因果说明和失败理由见同目录 `ITERATIONS_ZH.md`。

SwiGLU 单模型：

```powershell
.venv\Scripts\python.exe scripts/run_foodnutrigpt_no_foodname_v13.py `
  --output-dir output/no_foodname_v13_swiglu_20ep_20261005 `
  --epochs 20 --device cuda
```

当前推荐的 v16 单模型：

```powershell
.venv\Scripts\python.exe scripts/run_foodnutrigpt_no_foodname_v16.py `
  --output-dir output/no_foodname_v16_axis_pair_bias_20ep_20261005 `
  --epochs 20 --device cuda
```
