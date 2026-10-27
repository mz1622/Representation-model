# 显式依赖 mask 下的食物组成基础模型证据

**版本：** 2026-10-28

**主模型：** V5+CCE（dependency-mask v1）

**范围：** 新 mask 训练审计、UMAP 表征、台湾 TFDA 零样本与少样本、少观测补全、全验证集敏感性

## 摘要

这份报告重做了 commit `5eb7aeb` 的证据链，但所有补全任务都使用新的显式依赖 mask。这个 mask **同时影响训练和预测**：训练集中每一个 profile-target 样本先隐藏目标轴，以及依赖图中与它相连的祖先和后代；验证、complete test、台湾 zero-shot/few-shot、少观测与 sensitivity 使用同一个规则。新 V4、V5、V5+CCE 均从头按此规则训练，绝不是只在推理时替换 mask。

主要结果：

- 新 mask 只登记 FAO/INFOODS 定义可支持的 37 条有向关系，涉及 44 个轴。187 个监督轴中有 144 个是 singleton；平均每个目标隐藏 1.465 个轴，而旧语义 family 平均会隐藏 27.717 个轴。兄弟成分仍可见。
- complete test 上，V5+CCE 的 macro scaled-log MAE 为 **0.3016**，在本轮四个正式方法中最低。
- 台湾 strict zero-shot 上，V5+CCE 为 **0.1922**，相对 XGBoost、RF、KNN 分别降低 **17.3%、18.9%、32.3%**。
- 在相同架构、目标和优化器下，预训练初始化在 8/32/128-shot 的台湾适配中分别比随机初始化降低 **52.0%、45.7%、25.1%**。
- UMAP 和定量检查显示，V5+CCE 的食品类别结构强于同一训练链中的 V4/V5，但原始数值加缺失标记仍更强；来源和观测覆盖是明显混杂因素。
- sensitivity 显示模型对 ±10% 单输入扰动平滑；删除一个可见正值后的退化需要与 RF 的干净误差一起解释。prediction shift、error delta、perturbed error 不能混为同一结论。

旧 family-mask 与新 dependency-mask 定义了不同难度的任务，因此本报告只在新 mask 内比较模型；数值不能直接用来宣称模型本身比旧报告提高了多少。

## 1. 共同设定与新 mask

### 1.1 mask 的依据和作用位置

唯一分类依据是 [FAO/INFOODS component identifiers (tagnames)](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/) 对显式总量、分量、子总量和结构成员的定义。规则是：对目标轴隐藏 `target + all directed ancestors + all directed descendants`。没有被登记的轴独立预测；兄弟轴不因为同属一个宽泛营养类别而被隐藏。`mask_family` 列不再参与训练或评价。

实现审计：

- 37 条有向边，44 个登记轴；
- 144 个 singleton target，43 个 dependency-connected target；
- mask size 最小 1、最大 15、平均 1.465；
- 234 个 sibling-visibility 检查通过；
- food name、source、food family label、profile ID 均不是模型输入。

训练时共有 337,048 个 profile-target 任务/epoch。每个任务先由目标轴调用同一个 `DependencyMaskPolicy` 构造可见 context，再进入 V4/V5/CCE 的训练 batch。推理也调用同一对象，因此不存在“训练仍用旧 family、预测才换新 mask”的情况。

### 1.2 模型与指标

V5+CCE 保持 3 层、hidden 192、6 heads、ReGLU 256、有序 axis-pair bias。V5 使用两次共享参数前向：第一遍产生 `[FOOD]`，第二遍以 `detach(h_food)` 作为 prompt；CCE 在第一遍 `[FOOD]` 的两个 dropout view 上增加对比损失。正式预测来自第二遍目标 token。

每个轴使用原训练集正值中位数 `s_a`：`z = log(1 + y/s_a)`。先在轴内计算 MAE/RMSE，再对评价轴等权平均。scaled-log MAE 是主要指标，raw MAE 保留原含量尺度。

### 1.3 新 mask 的 complete-test 基准

| Method | SLMAE | SLRMSE | Raw MAE |
|---|---|---|---|
| V5+CCE | 0.3016 | 0.4910 | 0.3137 |
| XGBoost | 0.3270 | 0.4825 | 0.2624 |
| RF | 0.3322 | 0.4841 | 0.2838 |
| KNN | 0.4047 | 0.5937 | 0.4729 |

V5+CCE 的主要指标最好；XGBoost 的 raw MAE 和 scaled-log RMSE 更低。这说明 Transformer 更均衡地改善不同量级轴上的典型误差，而树模型对原单位误差或较大对数误差仍有优势。

## 2. UMAP 与表征结构

对 75,875 个 train+validation profile 输入全部已观测数值，并使用空 target query，提取第一遍 `[FOOD]`。空 query 本身不触发某个目标的 mask；它检验的是**由新 mask 训练得到的 encoder** 的全 profile 表征。UMAP 只用于二维展示，不参与训练、选 checkpoint 或性能评价。

![新版 V5+CCE UMAP](figures/umap_v5_cce.png)

| Representation | Macro purity@10 | Macro-F1 | Balanced acc. | Cosine silhouette |
|---|---|---|---|---|
| Nutrients_mask | 0.756 | 0.861 | 0.860 | 0.091 |
| PCA50 | 0.725 | 0.723 | 0.740 | 0.113 |
| V4 | 0.642 | 0.730 | 0.751 | -0.012 |
| V5 | 0.677 | 0.759 | 0.781 | 0.009 |
| V5+CCE | 0.697 | 0.790 | 0.810 | 0.083 |

V5+CCE 的 macro purity@10 为 0.697，高于 V4 的 0.642；linear macro-F1 为 0.790。但 raw nutrients+mask 仍达到 0.861 macro-F1，说明 FOOD embedding 没有普遍超过原始组成表示。

missingness-only 的类别 macro-F1 为 0.597，来源 balanced accuracy 为 0.971。图中的来源簇和观测轴数量梯度因此是实质混杂，而不能全部解释为食品生物学类别。

三个 UMAP seed 相对首个 seed 的 10 邻域平均重合为 0.779。随机隐藏 10%、30%、50% 已观测输入后，V5+CCE 与完整输入的平均 cosine similarity 分别为 0.981、0.933、0.857。UMAP 适合做表征审计，不单独构成 foundation-model 优势证据。

![表征类别定量](figures/representation_category_metrics.png)

## 3. 台湾 TFDA zero-shot

台湾数据没有参与训练、调参、checkpoint 选择或 `s_a` 估计。strict panel 含 2,212 个食品、128,470 个 food-axis 任务和 80 个轴；name-novel panel 含 1,789 个食品。所有方法都先应用新 dependency mask。RF/XGBoost/KNN 只在原训练集拟合。

![台湾 zero-shot](figures/taiwan_zero_shot.png)

| Method | SLMAE | SLRMSE | Raw MAE |
|---|---|---|---|
| V5+CCE | 0.1922 | 0.3703 | 0.3253 |
| XGBoost | 0.2324 | 0.3731 | 0.3185 |
| RF | 0.2370 | 0.3729 | 0.3332 |
| KNN | 0.2838 | 0.4519 | 0.4968 |

V5+CCE 在 strict panel 的 scaled-log MAE 最低。2,000 次台湾 profile 配对 bootstrap 中，V5+CCE 相对每个 baseline 的差值区间均由 `tables/taiwan_zero_shot_paired_bootstrap.csv` 给出。name-novel panel 得到相同排序。raw MAE 与 scaled-log 指标仍可能给出不同最优模型，应按各自定义解释。

## 4. 台湾 TFDA few-shot

只使用 name-novel 食品。固定 400 个食品作为测试集，其余 1,389 个组成适配池；每个 seed 内 8 subset 32 subset 128。比较预训练 V5+CCE 与完全相同架构的随机初始化，均训练 20 epochs、AdamW、lr `1e-4`、batch 128；固定最后一轮，不使用测试集选模型。

![台湾 few-shot](figures/taiwan_fewshot_curve.png)

| Init. | Shots | SLMAE mean | SLMAE SD | Raw MAE mean |
|---|---|---|---|---|
| pretrained | 8 | 0.2033 | 0.0026 | 0.3586 |
| scratch | 8 | 0.4237 | 0.0093 | 1.0460 |
| pretrained | 32 | 0.1974 | 0.0038 | 0.3561 |
| scratch | 32 | 0.3634 | 0.0166 | 0.7284 |
| pretrained | 128 | 0.1829 | 0.0004 | 0.3001 |
| scratch | 128 | 0.2443 | 0.0034 | 0.4850 |

固定测试集的冻结 zero-shot SLMAE 是 0.2000。8-shot 平均为 0.2033，略差于 zero-shot；32-shot 为 0.1974，小幅改善；128-shot 为 0.1829，改善 8.5%。预训练相对 scratch 的优势在每个样本量和每个 seed 都成立，这是当前最直接的样本效率证据。

## 5. 少量可见 observation

从验证集选择支持量最高的 12 个目标，每个目标固定 200 个任务，共 2,400 个任务。每个任务先隐藏目标及其依赖祖先/后代，再随机保留 5、10、20、40 或全部 context；有限预算重复三次。RF/KNN 使用正式 dependency-mask 训练特征，没有额外 mask-mix augmentation。

![少观测补全](figures/sparse_observation_curve.png)

| Method | Budget | Mean visible | SLMAE |
|---|---|---|---|
| KNN | 5 | 5.000 | 0.563 |
| RF | 5 | 5.000 | 0.492 |
| V5+CCE | 5 | 5.000 | 0.380 |
| KNN | 10 | 9.850 | 0.445 |
| RF | 10 | 9.850 | 0.374 |
| V5+CCE | 10 | 9.850 | 0.294 |
| KNN | 20 | 16.728 | 0.345 |
| RF | 20 | 16.728 | 0.294 |
| V5+CCE | 20 | 16.728 | 0.243 |
| KNN | 40 | 27.421 | 0.224 |
| RF | 40 | 27.421 | 0.189 |
| V5+CCE | 40 | 27.421 | 0.203 |
| KNN | all | 34.106 | 0.188 |
| RF | all | 34.106 | 0.132 |
| V5+CCE | all | 34.106 | 0.188 |

V5+CCE 在 10 和 20 个可见成分时分别得到 0.294 和 0.243，优于 RF/KNN；5 个成分时 V5 与 V5+CCE 接近。到 40 个和全部 context 时，RF 分别达到 0.189 和 0.132，优于 Transformer。这表明新任务下的优势集中在 context 真正稀疏的区间，而充分观测时 RF 很强。由于不再隐藏宽泛 nutrition family，“all” 条件可见信息通常多于旧报告，因此不能把两份 sparse 数值直接相减解释为架构提升。

## 6. 全验证集 sensitivity

分析覆盖 11,175 个 validation profile、323,809 个真实 target job 和 187 个目标。每个任务先应用 dependency mask，再从可见正值中按固定哈希选择一个输入；所有模型对同一输入作 ×0.001 至 ×1000、±5%/10%/20% 以及 deletion。另对可见零值作 `+0.1*s_a` 探针。这个设计保留全部 target job，但每个 job 只抽一个输入，因此是**全目标覆盖的配对 sampled-input sensitivity**，不是对全部 input-target pair 的穷举。

三个量分别是：prediction shift（预测移动）、error delta（扰动后误差减干净误差）、perturbed error（扰动后的绝对误差）。后两者必须结合 clean error 解读。

![局部敏感性](figures/sensitivity_dose_response.png)

| Method | Condition | Prediction shift | Error delta | Perturbed error |
|---|---|---|---|---|
| V5+CCE | factor_0.9 | 0.00422 | -0.00050 | 0.22163 |
| V5+CCE | factor_1.1 | 0.00382 | 0.00071 | 0.22284 |
| V5+CCE | factor_1000 | 0.12913 | 0.06848 | 0.29062 |
| V5+CCE | deletion | 0.05234 | 0.00882 | 0.23095 |
| RF | factor_0.9 | 0.00331 | 0.00049 | 0.22867 |
| RF | factor_1.1 | 0.00311 | 0.00095 | 0.22913 |
| RF | factor_1000 | 0.09537 | 0.07196 | 0.30014 |
| RF | deletion | 0.05463 | 0.02830 | 0.25648 |

在 ±10% 下，V5+CCE 的 prediction shift 和 error delta 都很小。deletion 条件下，V5+CCE 的 error delta 为 0.00882，RF 为 0.02830；扰动后误差分别为 0.23095 和 0.25648。这些结果描述模型响应，不能视为真实营养干预或因果效应。

![极端 stress](figures/sensitivity_stress.png)

×1000 用于发现数值失稳，不代表合理食品。所有结果均为有限值。500 次 profile 配对 bootstrap 的 CCE-RF 区间在 `tables/sensitivity_paired_bootstrap.csv` 中。

## 7. 综合判断

1. **新 mask 确实进入训练。** 任务生成、validation、complete test、台湾和扰动分析共享同一 target-specific policy；checkpoint 也记录 policy hash 和 `legacy_mask_family_used_for_masking=false`。
2. **外部迁移优势仍成立。** 台湾 zero-shot 中 V5+CCE 的主要指标优于三种 baseline，且 name-novel 子集排序一致。
3. **预训练提高样本效率。** 相同架构 scratch 对照显著更差；128-shot 还明显优于冻结 zero-shot。
4. **UMAP 是描述性证据。** CCE 表征比 V4/V5 更有类别结构，但 raw 数值+mask 更强，来源缺失模式明显。
5. **稳健性取决于条件和指标。** 小幅数值扰动总体平滑；删除与极端倍数下需要分别看预测移动、误差增量和最终误差。

最稳妥的陈述是：

> 在只屏蔽 FAO/INFOODS 可明确支持的依赖关系、并让其余轴独立预测后，重新训练的 V5+CCE 仍在台湾外部数据库上表现出清楚的 zero-shot 优势；同架构 few-shot 消融进一步表明预训练显著提高样本效率。表征和 sensitivity 分析支持模型利用了跨轴结构，同时显示数据库缺失模式和输入覆盖仍是重要混杂。

## 8. 局限

1. 新旧 mask 定义不同任务；新 mask 更窄，绝对误差不可直接用于比较架构进步。
2. 台湾只覆盖 80/187 个监督轴；名称未精确匹配不等于语义上完全未见。
3. few-shot 只有一个固定测试 split 和三个 seed；scratch 与 pretrained 使用统一超参数，而非分别调到最优。
4. UMAP 使用空 target query，展示 encoder 的全 profile 状态，不对应某一个具体补全 query。
5. sensitivity 每个 job 抽一个输入，并非穷举所有 input-target pair；扰动可能破坏真实食品的质量守恒。
6. 所有神经结果来自一条单 seed 训练链，未估计预训练 seed 间不确定性。

## 9. 复现与附件

报告只包含聚合表、图和哈希；不包含 checkpoint、原始台湾数据或逐条预测。主要附件位于 `tables/` 与 `figures/`。完整本地分析 manifest 记录数据、policy、checkpoint 和代码哈希。
