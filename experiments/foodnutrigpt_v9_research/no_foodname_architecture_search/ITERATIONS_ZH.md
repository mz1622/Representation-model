# 无 food name Transformer 架构搜索记录

## 固定目标与约束

目标是把冻结验证协议下全部 187 轴的 scaled log MAE 降到 **0.31 以下**。完整测试集保持关闭。food name、食物 ID 和来源 ID 不进入模型；现有数据划分、目标、成分家族遮蔽、训练尺度、MAE 损失、20 轮预算、检查点选择规则和三个传统基线保持不变。

本轮只探索有公开架构依据的 Transformer、数值 token 和 attention 改动。不把随机种子、dropout、学习率、训练轮数或同类超参数搜索当作候选改进。单种子 `20261005` 用于顺序筛选；达到 0.31 后再用固定新增种子复验。

当前父控制为 [masked-axis v3](../no_foodname_v3/REPORT_ZH.md)：目标轴以 `[MASK_VALUE]` token 进入 Transformer，并从目标 token 输出预测。三种子全轴均值 `0.326706±0.002674`，最好单次运行 `0.324312`；RF 为 `0.286841`。

## 网络检索与候选顺序

### A. 分段线性数值嵌入（优先）

[On Embeddings for Numerical Features in Tabular Deep Learning](https://openreview.net/pdf?id=pfI7u0eJAIr)指出，数值特征进入 Transformer 前的嵌入方式是重要设计维度；quantile-based piecewise-linear encoding（PLE）在多项表格任务上可显著优于普通线性/ReLU数值映射，并能缩小与 GBDT 的差距。作者的[官方实现](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py)把连续值表示为分位区间上的连续分段线性基函数，再由每个特征自己的可训练权重映射成 token。

本项目此前在旧 `[FOOD]` 输出模型上，8 段数值编码把全轴误差从 `0.3394` 降到 `0.3298`，已有任务内证据。因此第一项把正确的 PLE 与新的 masked-axis 补全结合：训练集逐轴分位点、重复分位点去重、每轴专属分段权重、连续饱和编码；其余结构与 v3 固定。

### B. 目标查询 cross-attention（第二）

[Perceiver IO](https://arxiv.org/abs/2107.14795)将输入编码与结构化输出查询解耦，用 output queries cross-attend 到编码后的输入；论文在匹配设置中报告 query-based attention decoder 相对平均池化解码器有小而一致的改善。它适合本任务“一个可见成分集合、多个语义不同的目标轴”。

候选结构为：可见 axis/value token 先独立 self-attention 编码；目标 axis query 再通过两层 cross-attention 读取可见表示，最后共享标量解码。这样目标 token 不会在输入编码阶段互相影响，同时仍保留每轴目标身份。

### C. Set Transformer 式诱导表示（第三）

[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)针对无顺序、长度可变的集合设计了 attention encoder/decoder，并以 inducing points 在建模元素交互的同时形成固定潜变量。若直接 cross-attention 仍不足，将测试少量 learned latent tokens：latent 先读取可见组成，目标轴再读取 latent。当前序列最多 252 个轴，计算效率不是主要动机；该候选只检验潜变量瓶颈是否改善泛化。

### D. FT-Transformer 风格 block（后置）

[FT-Transformer](https://arxiv.org/abs/2106.11959)在多类表格任务中是强神经基线，其关键包括 feature-level tokenization、PreNorm 和门控 FFN；[官方实现](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py)使用 GLU 类激活。当前模型已经具备 feature token、PreNorm 和无位置编码，因此这一项只会在前述输入/输出结构仍不足时，单独替换 FFN 为 gated FFN，避免把多个已存在的共同点重复包装成“新模型”。

### E. 显式 natural-missing axis token（追加）

前三项完整实验没有改善，ReGLU 虽改善但仍未达到目标。进一步审计输入后发现，RF/XGBoost 的 dense feature 明确同时输入逐轴数值和 missing mask；Transformer 只让观测轴出现，缺失只能从 token 缺席间接推断。[Missing Feature Attention Network](https://doi.org/10.1145/3729241)在回归和分类缺失表格任务中使用特征 embedding 加 missing-mask embedding，并报告显式建模 observed/missing 关系的收益。因此追加一个单因素版本：每个非目标家族轴都出现，观测值使用 numeric token，自然缺失使用共享 `[MISSING]` value embedding，目标家族仍使用独立 `[MASK_VALUE]` target token。真实 0 与自然缺失严格分开，预测仍只从目标 axis token 输出。

### F. scGPT MVC 辅助头（追加）

scGPT 的[官方实现](https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py)除逐 token 的 expression decoder 外，还提供 MVCDecoder：用 cell/`<cls>` 表示与 gene embedding 的 inner product 辅助重建 masked value；官方预训练代码把 MVC loss 与主 masked-value loss直接相加。本候选以当前最好的 ReGLU 版本为父模型，加入同形状的 context-axis inner-product MVC 和等权宏平均 MAE。MVC 只作为训练辅助，验证和导出预测仍只使用目标 axis token 的主解码器，避免让 context token 接管补全。

### G. 目标轴专属 decoder residual（追加）

现有 187 个回归任务共享同一个 scGPT 标量 decoder，轴身份只能通过 hidden state 间接调节输出；而强 RF 基线为每个轴单独拟合模型。为检验共享 decoder 的多任务负迁移，在 ReGLU 父模型的共享 decoder 后增加一个每轴专属的线性 residual：`shared(h_target) + w_axis·h_target/√d + b_axis`。新增权重和偏置全部从 0 初始化，所以训练开始时与 v4-D 完全相同；context token 不参与输出，预测仍来自对应目标 axis token。

### H. 固定等权 scaled-log Transformer ensemble（达标）

单模型搜索中 ReGLU 有确定改善但仍高于目标。表格深度学习的近期结果也显示 ensemble 是稳定降低方差的强方向；本项目已有三颗独立 masked-axis 种子，因此先计算预定义的三种子等权平均，再把结构不同且单模型最好的 ReGLU 作为第四个等权成员。四个成员都无 food name，预测统一转换为训练目标空间 `z=log1p(raw/s_axis)` 后取算术平均，再反变换；权重固定为 `0.25`，没有连续权重搜索或按轴选模型。

### I. 单学生知识蒸馏（最终部署约束）

因最终方案要求只训练和部署一个推理模型，把 H 的四成员平均改为训练监督：四个冻结教师只在 training row-family tasks 上生成 scaled-log 稠密软目标，一个随机初始化的 ReGLU student 同时学习真实观测标签和教师平均，各占损失的 0.5。验证标签不参与蒸馏，学生仍只从目标 axis token 输出；导出的 checkpoint 不含教师参数，推理模型数为 1。

### J. SwiGLU 单模型（暂停蒸馏后）

不读取任何教师或软目标，回到 v7 的单模型真实标签训练。唯一改变是把 ReGLU 的 `left×ReLU(gate)` 换成 `left×SiLU(gate)`；gated hidden width 仍为 171，总参数不变。该实验检验更平滑、非零负半轴的 Swish gate 能否改善数值回归。

## 迭代记录

| 版本 | 唯一主要改变 | 参数量 | 最佳轮次 | 营养142 | 代谢物45 | 全部187 | 决定 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v3 父控制（种子20261005） | masked axis token＋普通共享数值 MLP | 347,649 | 18 | 0.243748 | 0.586440 | 0.326214 | 父控制 |
| v4-A | 正确 PLE 替代共享数值 MLP | 588,929 | 18 | 0.239170 | 0.624719 | 0.331949 | 淘汰：全轴比 v3 差 0.005736 |
| v4-B | visible encoder＋target cross-attention | 613,377 | 18 | 0.236287 | 0.611643 | 0.326613 | 淘汰：全轴比 v3 差 0.000399 |
| v4-C | learned latent inducing set | 615,425 | 18 | 0.262898 | 0.631444 | 0.351586 | 淘汰：latent bottleneck 丢失细粒度轴信息 |
| v4-D | 参数量匹配的 ReGLU gated FFN | 348,077 | 16 | 0.234125 | 0.599338 | 0.322010 | 保留：当前最好，但未到 0.31 |
| v4-E | 显式 natural-missing axis token | 347,777 | 20 | 0.246167 | 0.596247 | 0.330411 | 淘汰：显式缺失 token 无收益且更慢 |
| v4-F | ReGLU＋scGPT MVC 训练辅助 | 380,973 | 20 | 0.236566 | 0.623069 | 0.329575 | 淘汰：context 辅助任务干扰主补全 |
| v4-G | ReGLU＋目标轴专属 residual decoder | 380,585 | 20 | 0.238775 | 0.637489 | 0.334722 | 淘汰：每轴残差加重代谢物过拟合 |
| v4-H / v11 | 3×v3＋1×ReGLU，scaled-log 等权集成 | 4个成员 | — | **0.223433** | **0.571067** | **0.307088** | **达标：全部187轴 < 0.31** |
| v12 | 四教师稠密软目标蒸馏至一个 ReGLU student | 348,077 | 18 | 0.228699 | 0.572968 | 0.311545 | 历史最好单 checkpoint；当前暂停蒸馏 |
| v13 | ReGLU gate 改为 SwiGLU/SiLU；无蒸馏 | 348,077 | 18 | 0.236900 | 0.619182 | 0.328893 | 淘汰：稀疏轴离群退化 |

每次完整运行必须保存配置、代码与数据哈希、逐轮曲线、最佳检查点、逐轴预测和指标，并在本表追加结果与是否继续的理由。

### v4-A 结果说明

固定种子 `20261005`、20 轮和既定的营养轴检查点选择规则。第 17 轮即时全轴指标最低为 `0.324619`，但营养轴指标继续改善，因此预先固定的选择规则选中第 18 轮；该检查点营养 142 轴为 `0.239170`、代谢物 45 轴为 `0.624719`、全部 187 轴为 `0.331949`。相较同种子的 v3 父控制，营养轴改善 `0.004578`，代谢物轴恶化 `0.038279`，使全轴恶化 `0.005736`。结论是 PLE 对营养值有效，但不能解决当前代谢物泛化瓶颈，因此不带入 v4-B，以便继续隔离架构因果。

### v4-B 结果说明

可见 axis/value token 先独立经过两层 self-attention；每个目标 axis token 作为 query，经过两层 cross-attention 读取可见 memory，再由自身的共享标量头输出。context token 只保证 memory 非空并保留样本表示，不承担 187 维预测。固定种子 `20261005` 的第 18 轮被选中：营养 `0.236287`、代谢物 `0.611643`、全部 `0.326613`。结果与 v3 接近但没有改善，说明阻止无值目标 token 参与输入编码本身不能解决误差；下一项检验 learned latent set 是否能形成更稳健的组成摘要。

### v4-C 结果说明

16 个 learned latent token 先 cross-attend 可见 axis/value set，经过两层 latent self-attention 后，再由每个目标 axis query 读取 latent。第 18 轮被固定规则选中：营养 `0.262898`、代谢物 `0.631444`、全部 `0.351586`。两组轴均显著退化，说明把全部可见组成压入 16 个无轴身份的潜变量会丢失补全所需的细粒度信息。本候选不再组合或扩展。

### v4-D 结果说明

保持 v3 的完整 masked-axis 序列和共享标量解码，只把每个 block 的普通 GELU FFN 替换为参数量匹配的 ReGLU FFN，gated hidden width 为 `round(256×2/3)=171`。第 16 轮被选中：营养 `0.234125`、代谢物 `0.599338`、全部 `0.322010`；参数量 `348,077`，与 v3 的 `347,649` 基本相同。相对 v3 同种子全轴改善 `0.004204`，说明收益来自门控结构而非明显增加容量。结果仍高于 `0.31`，所以保留为当前最好结构，并继续检验显式 missingness；只有后者独立有效才组合两项。

### v4-E 结果说明

所有非目标家族轴均进入序列，观测轴使用数值 token，自然缺失轴使用共享 learned missing-value embedding；目标家族仍用 masked target axis token。第 20 轮被选中：营养 `0.246167`、代谢物 `0.596247`、全部 `0.330411`。每轮由约 37 秒增至约 68 秒，且全轴指标比稀疏 v3 和 ReGLU 都差。结论是当前稀疏 set 中的 token 缺席已能表达缺失模式，显式缺失 token 增加的长度和噪声没有带来泛化收益。

### v4-F 结果说明

以 ReGLU 为主干，MVC 使用与 scGPT 官方 `MVCDecoder(inner product)` 一致的 axis-to-query、Sigmoid、无偏置投影和 context inner product；MVC 宏平均 MAE 与主 axis-token MAE 等权相加。第 20 轮被选中：营养 `0.236566`、代谢物 `0.623069`、全部 `0.329575`。MVC 损失在全程均高于主损失，尤其损害代谢物轴；即使验证和导出不使用 MVC 输出，context 压缩任务仍会干扰主表示。该结果支持让目标 axis token 独立承担正式补全，不保留 MVC。

### v4-G 结果说明

共享 decoder 后的每轴 residual 从全零开始，因此初始函数与 ReGLU 相同。固定选择规则最终选中第 20 轮：营养 `0.238775`、代谢物 `0.637489`、全部 `0.334722`。第 16 轮即时全轴曾为 `0.326813`，但营养指标之后继续改善，固定规则选择的模型在代谢物轴显著过拟合。结果说明 decoder 共享不是主要瓶颈，单独放开每轴输出反而削弱跨轴共享。

### v4-H / v11 结果说明

先把三个 v3 种子的预测在 scaled-log 空间等权平均，全部 187 轴为 `0.309762`，已经略低于目标；再加入单模型最好的 ReGLU 作为第四个等权成员，营养 `0.223433`、代谢物 `0.571067`、全部 `0.307088`。相比三种子 v3 均值，营养改善 `0.002425`、代谢物改善 `0.003457`、全轴改善 `0.002674`。四成员固定等权，不使用 RF/XGBoost/KNN、不使用 food name、不按轴挑选成员，也没有在验证集搜索连续权重。完整测试集仍未打开。

### v12 结果说明

共为 337,048 个训练 task 缓存四教师平均软目标，最大 family query 宽度为 54；两个损失都做逐轴宏平均后等权相加。固定种子 `20261005`、20 轮，既定营养选点规则选择第 18 轮：营养 `0.228699`、代谢物 `0.572968`、全部 `0.311545`。相对 v7 单模型全轴改善 `0.010466`（约 3.25%），并保留 v11 大部分收益；相对 v11 仍差 `0.004456`，因此不能宣称单模型已达到 0.31。最终 checkpoint 只有 348,077 参数的一个 ReGLU student，已验证无需教师即可独立推理。完整测试集仍未打开。

### v13 结果说明

固定种子 `20261005`、真实标签 MAE、20 轮和原营养选点规则，选择第 18 轮：营养 `0.236900`、代谢物 `0.619182`、全部 `0.328893`。相对等参数 ReGLU v7 全轴恶化 `0.006883`，也比普通 GELU v3 差 `0.002679`。虽然 119/187 个轴有小幅改善，但少数稀疏轴出现大误差，逐轴宏平均因此变差。结论是 SwiGLU 的平滑负半轴在当前任务中没有提供稳健收益，当前无蒸馏单模型继续采用 ReGLU。完整测试集仍未打开。
