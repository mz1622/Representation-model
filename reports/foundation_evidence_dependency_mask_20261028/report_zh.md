# 显式依赖 mask 下的食物组成基础模型证据：修订分析

**版本：** 2026-10-28 r2

**主模型：** V5+CCE（dependency-mask v1）

**范围：** mask 训练审计、food/database UMAP、台湾 zero/few-shot、少观测补全、INFOODS 组件组 sensitivity、dependency 暴露消融

## 摘要

当前 dependency mask **同时影响训练和预测**。训练中的每个 profile-target 任务先隐藏目标轴及其有向祖先和后代；validation、complete test、台湾外部验证和以下分析复用同一 policy。V4、V5、V5+CCE 均在这个 mask 下从头训练。

本次按要求修订四部分：UMAP 只保留 food category 与 database source；台湾 few-shot 只比较冻结 V5+CCE 与台湾少样本 RF/XGBoost；少 observation 只比较 V5+CCE、RF、XGBoost、KNN；sensitivity 增加 FAO/INFOODS 组件组矩阵和 dependency-family 推理暴露消融。

主要结果：

- 台湾固定测试集上，冻结 V5+CCE 为 **0.2000**。RF 的 8/32/128-shot 为 **0.4551/0.3266/0.2430**；XGBoost 为 **0.5287/0.3262/0.2369**。冻结模型在三个样本量都更好。
- +10% 单输入扰动中，INFOODS 同组 input-target 的平均 shift 是 **0.00550**，跨组是 **0.00179**；删除输入时分别是 **0.08410** 与 **0.02230**。
- dependency 暴露消融没有证明严格父子/包含轴在当前模型中比普通轴影响更强。暴露全部通常隐藏值后，SLMAE 从 **0.1541** 变为 **0.1628**；依赖轴扰动的 shift 也整体低于 matched control。这是重要的负结果。

## 1. 共同设置与 mask

mask 图只登记 FAO/INFOODS tagname 定义能够明确支持的 totals、components、subtotals 和 structural members。37 条有向边覆盖 44 个轴；187 个监督轴中 144 个是 singleton。目标 mask 为 `目标 + 全部有向祖先 + 全部有向后代`，siblings 仍可见，旧 `mask_family` 不参与训练或评估。

| Method | SLMAE | SLRMSE | Raw MAE |
|---|---|---|---|
| V5+CCE | 0.3016 | 0.4910 | 0.3137 |
| XGBoost | 0.3270 | 0.4825 | 0.2624 |
| RF | 0.3322 | 0.4841 | 0.2838 |
| KNN | 0.4047 | 0.5937 | 0.4729 |

## 2. UMAP：food category 与 database

UMAP 使用 4,000 个 validation profile 的 V5+CCE 第一遍 `[FOOD]` 表征；类别和 database 只用于着色，不进入模型。图中不再展示 observation count 或单个 nutrient 梯度。

![Food category 与 database UMAP](figures/umap_food_category_database.png)

| Representation | Macro purity@10 | Macro-F1 | Balanced acc. |
|---|---|---|---|
| Values+mask | 0.7563 | 0.8614 | 0.8598 |
| V4 | 0.6424 | 0.7296 | 0.7505 |
| V5 | 0.6767 | 0.7592 | 0.7814 |
| V5+CCE | 0.6973 | 0.7895 | 0.8096 |

database code 对应如下：

| Code | Source key | Database | UMAP n |
|---|---|---|---|
| D01 | afcd | Australian Food Composition Database | 107 |
| D02 | anfood_2_0 | FAO/INFOODS AnFooD 2.0 | 22 |
| D03 | bangladesh_fct_2013 | Bangladesh Food Composition Table 2013 | 15 |
| D04 | biofoodcomp_4_0 | FAO/INFOODS BioFoodComp 4.0 | 51 |
| D05 | bls_4_0 | German Nutrient Database BLS 4.0 | 405 |
| D06 | ciqual | ANSES-Ciqual | 194 |
| D07 | cnf | Canadian Nutrient File | 326 |
| D08 | cofid | UK CoFID | 141 |
| D09 | efsa_eu_fcdb_2013 | EFSA EU Food Composition Database | 830 |
| D10 | fndds | USDA FNDDS | 273 |
| D11 | foodb | FooDB | 622 |
| D12 | frida | Frida FoodData, Denmark | 79 |
| D13 | lesotho_fct_2006 | Lesotho Food Composition Table 2006 | 12 |
| D14 | mext_japan_2023 | Japan MEXT Food Composition Tables | 133 |
| D15 | norway | Norwegian Food Composition Database | 87 |
| D16 | phyfoodcomp_1_0 | FAO/INFOODS/IZiNCG PhyFoodComp 1.0 | 112 |
| D17 | smiling_cambodia_2013 | SMILING Cambodia FCT 2013 | 6 |
| D18 | smiling_indonesia_2013 | SMILING Indonesia FCT 2013 | 6 |
| D19 | smiling_laos_2013 | SMILING Laos FCT 2013 | 6 |
| D20 | smiling_thailand_2013 | SMILING Thailand FCT 2013 | 7 |
| D21 | smiling_vietnam_2013 | SMILING Vietnam FCT 2013 | 11 |
| D22 | swiss_fcdb_7_1 | Swiss Food Composition Database 7.1 | 73 |
| D23 | usda_sr_legacy | USDA Standard Reference Legacy | 431 |
| D24 | wafct_2019 | FAO/INFOODS West Africa FCT 2019 | 51 |

缺失模式本身对 source 的 balanced accuracy 仍达到约 0.971，因此 database 分离包含明显的 coverage/missingness 混杂，不能单独作为语义表征证据。

## 3. 台湾 zero-shot

台湾数据不参与训练、尺度估计、调参或 checkpoint 选择。strict panel 含 2,212 个 food、128,470 个 food-axis job 和 80 个轴。

![台湾 zero-shot](figures/taiwan_zero_shot.png)

| Method | SLMAE | SLRMSE | Raw MAE |
|---|---|---|---|
| V5+CCE | 0.1922 | 0.3703 | 0.3253 |
| XGBoost | 0.2324 | 0.3731 | 0.3185 |
| RF | 0.2370 | 0.3729 | 0.3332 |
| KNN | 0.2838 | 0.4519 | 0.4968 |

V5+CCE 的主要 scaled-log MAE 为 **0.1922**，仍优于三个 source-trained baseline；raw MAE 最优仍可能是 XGBoost。

## 4. 台湾 few-shot：冻结模型与树模型少样本拟合

固定 400 个 name-novel food 为测试集，1,389 个为 adaptation pool。8-shot 子集先按缺失模式做 axis coverage 集合覆盖，再随机填满；不查看数值或测试误差。子集保持 `8 ⊂ 32 ⊂ 128`，三个 seed 使用相同协议。V5+CCE 完全冻结；RF/XGBoost 对每个轴只用相应台湾 adaptation profiles 拟合正式配置。这里没有 scratch Transformer，也没有 Transformer fine-tuning。

![台湾 few-shot](figures/taiwan_fewshot_curve.png)

| Method | Profiles | SLMAE mean | SD |
|---|---|---|---|
| RF | 8 | 0.4551 | 0.0179 |
| V5+CCE frozen | 8 | 0.2000 | 0.0000 |
| XGBoost | 8 | 0.5287 | 0.0243 |
| RF | 32 | 0.3266 | 0.0058 |
| V5+CCE frozen | 32 | 0.2000 | 0.0000 |
| XGBoost | 32 | 0.3262 | 0.0074 |
| RF | 128 | 0.2430 | 0.0017 |
| V5+CCE frozen | 128 | 0.2000 | 0.0000 |
| XGBoost | 128 | 0.2369 | 0.0010 |

128 个台湾 food 后，XGBoost 已到 0.2369、RF 到 0.2430，但仍未达到冻结 V5+CCE 的 0.2000。这比 pretrained-vs-scratch Transformer 更直接地显示跨数据库预训练的样本效率。

## 5. 少量可见 observation

固定 12 个高支持目标，每轴 200 个 validation job。先应用 dependency mask，再随机保留 5、10、20、40 或全部 context；有限预算重复三次。只比较四个指定方法。

![少观测补全](figures/sparse_observation_curve.png)

| Method | Visible | SLMAE mean | SD |
|---|---|---|---|
| KNN | 5 | 0.5633 | 0.0069 |
| RF | 5 | 0.4918 | 0.0077 |
| V5_CCE | 5 | 0.3799 | 0.0078 |
| XGBoost | 5 | 0.4572 | 0.0043 |
| KNN | 10 | 0.4453 | 0.0029 |
| RF | 10 | 0.3738 | 0.0099 |
| V5_CCE | 10 | 0.2940 | 0.0042 |
| XGBoost | 10 | 0.3693 | 0.0086 |
| KNN | 20 | 0.3452 | 0.0025 |
| RF | 20 | 0.2943 | 0.0021 |
| V5_CCE | 20 | 0.2427 | 0.0061 |
| XGBoost | 20 | 0.3035 | 0.0020 |
| KNN | 40 | 0.2241 | 0.0022 |
| RF | 40 | 0.1888 | 0.0027 |
| V5_CCE | 40 | 0.2033 | 0.0018 |
| XGBoost | 40 | 0.2142 | 0.0005 |
| KNN | all | 0.1883 | nan |
| RF | all | 0.1316 | nan |
| V5_CCE | all | 0.1879 | nan |
| XGBoost | all | 0.1731 | nan |

V5+CCE 在 5/10/20 个可见值时分别为 0.380/0.294/0.243，均优于三个 baseline。40 个和 all 时 RF 最强，XGBoost 次之。这说明 foundation advantage 集中在真正稀疏的 context，而不是充分观测条件。

## 6. FAO/INFOODS 组件组 sensitivity

### 6.1 分类依据

只采用一个来源：[FAO/INFOODS 与 EuroFIR 组件命名 workshop report](https://www.fao.org/fileadmin/templates/food_composition/documents/upload/9IFDCReportfinal.pdf) Annex 1 的组件层级，并用 [INFOODS tagnames](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/)核对名称。Annex 的层级标为 draft proposition，因此只登记氨基酸、脂肪酸、糖、膳食纤维、矿物质、维生素、有机酸七个清楚的大组；其余轴保持未分类。本分析共分类 135/252 个轴，其中 132 个为监督目标。这个分类只用于汇总 sensitivity，不参与 mask 或模型输入。

### 6.2 跨组矩阵

每个 target job 先应用 dependency mask，再在每个可用 input group 中固定抽一个正值轴。共 58,197 个配对组合。矩阵数值是 target-axis macro 的 normalized-log prediction shift。

![+10% 组件组 sensitivity](figures/group_sensitivity_plus10.png)

![删除一个输入的组件组 sensitivity](figures/group_sensitivity_deletion.png)

| Condition | Relation | Prediction shift | Error delta | Pairs |
|---|---|---|---|---|
| deletion | between_group | 0.0223 | 0.0013 | 45269 |
| deletion | within_group | 0.0841 | 0.0349 | 12928 |
| minus_10pct | between_group | 0.0020 | -0.0002 | 45269 |
| minus_10pct | within_group | 0.0061 | -0.0002 | 12928 |
| plus_10pct | between_group | 0.0018 | 0.0002 | 45269 |
| plus_10pct | within_group | 0.0055 | 0.0010 | 12928 |

对角线总体高于非对角线：+10% 时约为跨组的 3.1 倍，deletion 时约为 3.8 倍。糖与膳食纤维的组内响应最强，但膳食纤维部分 cell 支持量很低，必须结合 `tables/infoods_group_sensitivity.csv` 的 jobs/axes 阅读。这些是预测关联，不是营养干预因果效应。

## 7. Dependency-family 暴露与扰动消融

本实验只使用当前 policy 本来会隐藏的 target-specific ancestors/descendants；siblings 本来就可见。共 6,889 个 job、32 个目标和 5 个 dependency components。标准输入依次改为暴露一个、一半、全部已观测依赖亲属。

![Dependency 暴露消融](figures/dependency_exposure_ablation.png)

| Condition | SLMAE | Shift vs masked | Axes | Jobs |
|---|---|---|---|---|
| standard_mask | 0.1541 | 0.0000 | 32 | 6889 |
| expose_one | 0.1612 | 0.0237 | 32 | 6889 |
| expose_half | 0.1620 | 0.0256 | 32 | 6889 |
| expose_all | 0.1628 | 0.0290 | 32 | 6889 |

暴露越多并没有改善准确率，反而使整体误差上升。这符合一个关键事实：模型在训练时从未见过“预测目标同时看到其祖先/后代”的任务，推理时解开 mask 是分布外输入。

随后分别扰动一个已暴露依赖轴和一个按 INFOODS 组/normalized value 匹配的非依赖可见 control：

![Dependency 与 matched control 扰动](figures/dependency_perturbation_ablation.png)

| Factor | Dependency shift | Control shift | Dependency Δerror | Control Δerror |
|---|---|---|---|---|
| 0.9000 | 0.0028 | 0.0046 | -0.0007 | 0.0003 |
| 1.1000 | 0.0026 | 0.0042 | 0.0007 | 0.0004 |
| 10.0000 | 0.0548 | 0.0865 | 0.0177 | 0.0400 |

单纯 reveal 一个值时，dependency shift 为 0.02371，control reveal shift 为 0.02991；profile bootstrap 的 dependency-minus-control 95% 区间为 [-0.01124, 0.00091]。三种数值扰动下 dependency shift 也均较小，bootstrap 区间见附件。

因此本消融**不能证明**当前模型在推理时强烈依赖这些父子轴。它反而支持继续 mask：一旦只在推理时开放，误差变差。父子/包含关系的科学依据来自 INFOODS 定义；若要检验“不 mask 会带来多大预测泄漏”，需要重新训练一个 matched no-dependency-mask 模型，不能用这次 inference-only OOD 消融替代。

## 8. 结论与限制

1. 新 mask 确实进入训练与所有预测路径。
2. 冻结 V5+CCE 在台湾 fixed test 上优于仅用 8–128 个台湾 food 拟合的 RF/XGBoost，支持预训练的外部样本效率。
3. 少 observation 下 V5+CCE 在 5–20 个可见值最强；40/all 时树模型更强。
4. FAO/INFOODS 大组矩阵显示组内响应整体大于组间，但这是关联结构。
5. inference-only dependency 暴露是负结果，不能作为依赖关系已被模型学会的证据。
6. UMAP 的 database 分离受 missingness 强烈混杂；台湾只覆盖 80/187 个目标；所有神经结果仍来自单一预训练 seed。

## 9. 复现与附件

报告只包含聚合表、图和哈希，不包含 checkpoint、原始数据或逐条预测。完整脚本、manifest、表和图随报告提交。
