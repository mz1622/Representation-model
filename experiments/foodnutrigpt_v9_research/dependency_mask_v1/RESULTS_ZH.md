# 明确依赖关系 mask：实现、自动审计与训练结果

## 1. 结论

新的 mask 协议已经实现并完成全量实验。旧的 `mask_family` 不再参与训练或预测；它也不作为模型输入。对目标轴 `t`，新规则只隐藏：

```text
M(t) = {t} ∪ ancestors(t) ∪ descendants(t)
```

兄弟轴保持可见，未登记明确关系的轴只隐藏自己。V5+CCE 使用 validation 选择的 CCE 第 17 轮 checkpoint，在 659 个 complete-test profiles、19,690 个 profile-axis 任务上的 macro scaled-log MAE 为 **0.301589**，优于本轮最好的 baseline XGBoost（0.327034）。相对改善为 **7.78%**。

三项指标并未选出同一个最优模型：V5+CCE 的 scaled-log MAE 最低；XGBoost 的 scaled-log RMSE 和 raw MAE 最低。因此，本轮证据支持 V5+CCE 对“典型、按轴平衡的相对量级误差”更强，但不支持它在大误差控制或原始浓度误差上全面优于树模型。

## 2. 关系定义

关系表只使用 [FAO/INFOODS Food Component Identifiers](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/) 的成分定义作为主权威来源。接受的关系包括：

- Vitamin D total 与 D2、D3；
- Sugars,total 与明确的游离单糖、双糖；
- Dietary fibre,total、insoluble fibre、cellulose、lignin 的层级；
- Phytosterols 与 beta-sitosterol、campesterol、stigmasterol；
- Lutein+zeaxanthin 与两个单项；
- Saturated fatty acids total 与明确的饱和脂肪酸；
- Total trans fatty acids、trans-monoenoic、trans-polyenoic 与明确的 trans 单项。

未接受的候选关系包括 folate、choline、niacin、available carbohydrate、protein、total fat，以及未保留分析 tag 时的 mono/polyunsaturated total。原因不是这些成分毫无关系，而是当前规范化轴名不足以证明每条来源记录使用同一个数值表达式。未接受时统一退回单轴 mask。

冻结后的注册表有 37 条有向边，涉及 44 个轴。187 个监督轴中，144 个是单轴 mask，43 个使用依赖 mask。新协议平均隐藏 1.465 个轴；旧语义 family 平均隐藏 27.717 个轴。

## 3. 自动审计

由于本轮不依赖人工逐条复核，实现中加入了 fail-closed 检查：

- 轴名必须与冻结数据中的 canonical name 精确匹配；
- 拒绝重复边、自环、缺失轴和有向环；
- 自动计算传递祖先和后代；
- 检查 child target 不会隐藏 sibling，共完成 234 次 sibling-visibility 检查；
- Transformer 和三类 baseline 调用同一个 mask 实现；
- 验证可见 token 顺序置换不改变预测；
- 验证增加 padding token 不改变预测；
- 验证 checkpoint 保存、加载后预测逐值一致；
- 小规模三阶段训练和完整 complete-test 读取通过后，才启动正式训练。

测试结果为 `4 passed`。依赖表在任何正式训练结果产生前冻结；训练结果没有用于增加、删除或修改关系。

## 4. 训练协议

所有方法使用同一训练、validation、complete-test 划分，同一 187 个监督轴，同一正值训练中位数 scale，以及同一逐 profile-axis 直接评分。

Baselines：

- KNN：16 个邻居，欧氏距离，`1/(distance+0.001)` 加权；
- RF：每轴 200 棵树，`min_samples_leaf=2`，`max_features=0.5`；
- XGBoost：每轴 300 棵树，depth 6，learning rate 0.05，row/column subsample 0.8。

V5+CCE 是一个 2,057,125 参数的单模型：3 层、hidden 192、6 heads、ReGLU、ordered axis-pair attention bias。训练由三段组成：

1. V4 从随机初始化训练 20 轮，学习率 `3e-4`；
2. 从 V4 第 20 轮继续 V5 detached FOOD prompt 训练 20 轮，学习率 `1e-4`；
3. 从 V5 第 20 轮继续 CCE 训练 20 轮，CCE weight 0.25、temperature 0.5。

每轮从 1,828,536 个可用训练标签中按轴均衡抽取 337,048 个任务。这个任务数等于旧协议每轮的 family-task 前向预算。batch size 128，seed 20261005。阶段父模型固定使用前一阶段第 20 轮，避免用 validation 选择父模型；最终只在 CCE 阶段按 validation macro scaled-log MAE 选 checkpoint。

## 5. 结果

### 5.1 Validation

| 模型 | Scaled-log MAE | Scaled-log RMSE | Raw MAE |
|---|---:|---:|---:|
| **V5+CCE** | **0.223342** | 0.427485 | 0.249135 |
| RF | 0.228900 | **0.390471** | 0.215873 |
| XGBoost | 0.237969 | 0.399867 | **0.208768** |
| KNN | 0.289384 | 0.484461 | 0.366831 |

V5+CCE 相对 validation 上最好的 baseline RF 降低 scaled-log MAE 2.43%。

### 5.2 Complete test

| 模型 | Scaled-log MAE | Scaled-log RMSE | Raw MAE |
|---|---:|---:|---:|
| **V5+CCE** | **0.301589** | 0.491016 | 0.313714 |
| XGBoost | 0.327034 | **0.482543** | **0.262373** |
| RF | 0.332210 | 0.484127 | 0.283838 |
| KNN | 0.404747 | 0.593722 | 0.472923 |

在 primary metric 上，V5+CCE 相对 XGBoost 改善 7.78%，相对 RF 改善 9.22%。XGBoost 在两个 secondary metrics 上更好。

### 5.3 神经训练阶段

| 阶段 | Validation 最佳轮 | 最佳 scaled-log MAE | 第 20 轮 scaled-log MAE |
|---|---:|---:|---:|
| V4 | 17 | 0.241260 | 0.242385 |
| V5 | 14 | 0.223905 | 0.230059 |
| V5+CCE | 17 | **0.223342** | 0.226385 |

CCE 阶段的最佳值比 V5 阶段最佳值低 0.000563，即 0.25%。这只能说明最终 CCE continuation 在 validation 选择上略优；由于本轮没有同轮数的 continuation-only control，不能把这 0.25% 全部解释为 CCE 的因果收益。

## 6. 解释与限制

新协议比旧 family mask 明显更窄，所以本轮绝对数值与旧 broad-family 结果属于不同任务定义，不能把下降全部解释为模型进步。新协议的价值在于：只阻止有明确数值依赖的直接泄漏，不再因为“同属营养、维生素或脂肪酸类别”而大量删除可能合法的上下文。

本轮仍是单 seed。Complete test 中每轴支持量为 5 到 566，稀疏轴的不确定性仍较大。模型权重、原始数据和逐样本预测没有提交到 Git；仓库只保存关系审计、聚合指标、阶段摘要和本报告。
