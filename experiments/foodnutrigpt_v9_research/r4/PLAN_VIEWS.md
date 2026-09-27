# R4第二项：额外上下文删除下的表征一致性（训练前登记）

首项查询残差未改善补全，父控制保持原MAE/512/60线性头。现在登记两条完整60轮轨迹：`mlp60_views_weight0` 和 `mlp60_views_weight01`。本版预算由2增至4/12，不增加宽度、查询头、名称标准化、name-only任务或额外标签。本方案目前是待验证假设，不是已发生的改善。

## 问题与可证伪预期

当前食品编码器只接受逐轴监督，未直接约束同一档案在不同可见营养上下文下的表示。假设适度的一致性约束能够降低对具体观测模式的依赖，并改善未见食品的补全。如果成立，固定验证主指标应降低，正值可加误差贡献应改善；训练表示更一致本身不足以接受假设。更强不变性也可能丢掉有效营养信息、使表征方向趋同，故必须报告补全代价和表征分散度。不能从内部结果直接宣称跨来源迁移。

这里约束的是名称＋营养的融合表征：两视图名称相同，模型也可能通过更多依赖名称来降低距离。因此不能将C下降解释为独立营养表征已经更好；name-only、补全取舍和未来营养分支专项验证需分开报告。

## 两条分支唯一差异

view A是原始共同家族任务，完整目标家族（含关联context-only形式）已隐藏。对同一任务，在全部252轴上独立生成Bernoulli(0.3)额外隐藏标记，view B只在A基础上进一步隐藏；已经隐藏的值永不恢复。名称保持相同原始PCA32，不加入来源等字段。只改变可见性mask，不改target、positive、值、来源权重或任务数量。B中保留下来的显式零仍是观察，缺失不会变成观察。

随机数来自独立NumPy `SeedSequence([20260922, epoch, 4104])`；按冻结家族task ID及轴顺序生成float32均匀数，两个候选共用相同额外mask。与任务排列及模型CPU/CUDA随机数独立；保存每轮额外mask哈希、实际删除的可见单元数、原始可见数和任务数。空上下文的A和B都合法。

两条分支均计算同一个编码器的 `hA`、`hB`，A走原线性头和原MAE187监督。B不增加回归目标、不会把额外隐藏值作为新标签。以 `u=h/max(||h||2,1e-12)` 定义每个任务的距离 `d=0.5*sum((uA-uB)^2)`，两侧均有梯度，无teacher、stop-gradient、投影头、负样本或额外参数。

一致性权重沿用原轴/来源平衡目标：对任务i，`q_i=sum_a(target_ia * cell_weight_ia / train_axis_total_a)`；完整训练目标 `C=(1/187)*sum_i(q_i*d_i)`。uniform家族任务minibatch使用 `T/(B*187)`，与原MAE完全相同的无偏估计因子。每个观测轴仍按原训练来源/候选权重计入，不把重复原始记录视作额外一致性样本。该权重依赖已登记的监督支持，用于loss，绝不进入编码器；不使用隐藏数值计算距离或视图。它是轴诱导任务权重，不能简称每个食品任务等权。

总目标明确固定为 `L=MAE_A + lambda*C`。control的lambda=0，candidate=0.1。两者执行相同的编码/距离和反向图，不在lambda=0时跳过第二视图；监督项系数保持1，不除以1+lambda。新增一致性梯度会改变总范数、clip1以及有效更新，这是这项损失干预的组成部分，不声称只改变方向或保持整体梯度尺度。保存每轮MAE、未乘系数的C、总loss、裁剪前梯度范数均值及触发比例，以便检查该替代解释。

## 固定设置及公平边界

数据/尺度/划分/外部PCA32输入、原337,048家族任务和1,828,536回归目标不变。父MLP667,900参数、LayerNorm、共享头、MAE187、0%额外name-only、batch256、lr0.001、AdamW wd0.0001、60轮cosine至0.00001、clip1、seed20260922全部不变。无提前停止，两条轨迹分别按同一142轴补全主指标最早严格最小值选点。

新增B只包含共同A输入的子集，没有额外原始观测、真实目标、来源字段或候选营养信息。模型优化目标确有变化，不能声称各算法优化流程相同。监督训练面板、目标/权重和最终评价输入保持共同协议，因此现有强树与名称专项基线可以复用。若将来改变有标签训练遮蔽任务、输入字段、数据、划分或指标，则需重建共同协议并重算基线，不沿用本结论。

运行0是新的相同计算控制，正式因果对照0.1对0，不把旧父控制替换0。另报告0对旧父控制的兼容性/训练波动，不将其作为一致性效果。训练前先做：默认损失与0分支的真实训练输入、多步参数/RNG比较；两视图完整目标家族隐藏；改隐藏值/标签/来源不改变编码或预测；未观测不监督、零仍监督；来源权重/batch聚合正确；非有限数失败；保存重载及三个API。有限步CPU一致不能外推完整CUDA轨迹。

## 评价、诊断和决定规则

每条轨迹保留完整补全、name-only、固定名称库完整/30%检索，逐轴/来源、正值/零值可加贡献、raw/log/142/45/187指标，以及全训练面板拟合和实际曲线。比较0.1对0的三个任务；两者均对同seed22 RF400、XGB800，name-only对名称近邻、检索对独立名称MLP；所有结果使用同一个补全选出的检查点。整食品组1000次配对重采样，seed20260922，明确不包含选点/训练种子/标签真实性不确定性，不作多重比较校正后的确认解释。

每轮记录A的平均表示范数和单位化表示逐维batch标准差均值（任务数加权的batch诊断），并比较C曲线。它不是全数据表示协方差、不是坍塌的充分判据、不是迁移性能。训练后共同探针固定为8192个训练家族task ID（无放回，NumPy default_rng(20260925)），额外mask使用同一函数、seed20260922、epoch1001、p0.3，再按所选task ID索引；不重用训练60轮mask。报告这8192样本上的单位表示逐维总体标准差均值、均值中心平方距离、均值范数、零范数比例、A/B距离，以及用原q权重计算的距离。两个模型使用同一输入，指标只作机制诊断，不选择超参数或检查点。

若主指标无支持收益或收益仅来自零值、正值退步，按现象决定拒绝/保留分支，不能仅凭C下降接受模型。任何被接受的改进需三个固定种子；超过较强树5%、配对区间支持改善且原营养log-MAE退步不超过2%的门槛不变。当前只是两个预登记候选，后续权重或扰动率不得在中途改动。历史测试始终关闭，FooDB原始证据限制不变。

## 已执行命令（不改变登记设置）

```powershell
.\.venv\Scripts\python.exe scripts/audit_foodnutrigpt_views.py --output-dir reports/v9_r4_views_functional_audit_v1
.\.venv\Scripts\python.exe -u scripts/train_foodnutrigpt_v9_r4_views.py --consistency-weight 0 --view-drop-probability 0.3 --seed 20260922 --output-dir output/v9_r4/mlp60_views_weight0
.\.venv\Scripts\python.exe -u scripts/train_foodnutrigpt_v9_r4_views.py --consistency-weight 0.1 --view-drop-probability 0.3 --seed 20260922 --output-dir output/v9_r4/mlp60_views_weight01
.\.venv\Scripts\python.exe scripts/audit_foodnutrigpt_views_zero_replay.py --output-dir reports/v9_r4_views_zero_replay_v1
.\.venv\Scripts\python.exe scripts/diagnose_foodnutrigpt_view_geometry.py --checkpoint output/v9_r4/mlp60_views_weight0/best_model.pt output/v9_r4/mlp60_views_weight01/best_model.pt --output-dir reports/v9_r4_views_geometry_v1
```

这些目录现已存在，复跑必须使用新目录，程序拒绝覆盖。训练环境、完整hash/config和逐轮曲线在各run目录；所有比较、诊断及版本决定见 [README.md](README.md)。
