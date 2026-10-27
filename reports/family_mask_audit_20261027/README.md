# Family-aware masking audit / 家族整体遮蔽审计

## 结论先行

当前 `mask_family` 是项目在建模前人工设计的任务分组。外部标准支持“这些成分属于哪一类营养或化学物质”，但没有任何一个标准规定“训练 Transformer 时必须把这一整类一起遮蔽”。因此必须区分两件事：

1. **语义分类依据**：INFOODS、WHO/FAO、NIH DRI、LIPID MAPS、ChEBI、Phenol-Explorer 和 USDA 专业数据库支持成分身份与大类归属。
2. **共同 mask 的任务设计**：为了减少同类成分、总量和分量之间的直接泄漏，本项目选择按 family 整组隐藏。这是本项目的工程决定，不是科学界统一标准。

现有结果应表述为 **family-block completion**：给定一个食品在其他 family 中的组成，重建被整体移除的一个 family。它比“只缺一个成分时补一个成分”更严格，也不是同一个任务。

## 实际执行方式

- 训练时，每个任务是 `(profile, mask_family)`。只要该食品在这个 family 内至少有一个监督标签，就生成一次任务。
- 输入中删除这个 family 的所有已观测轴，包括该 family 中不参与 loss 的 context-only 轴。
- Transformer 同时放入该 family 的全部监督轴作为 `[MASK]` 查询 token；loss 只计算该食品实际有标签的轴。
- RF、XGBoost 和 KNN 使用同一 family mask，只是输入被编码为定长数值与可见性向量。
- 内部验证、complete test 和台湾外部数据都复用同一套冻结 family。
- 台湾分析项先人工核对成分身份与每 100 g 单位，然后映射到内部轴；`mask_family` 直接从内部轴复制，没有为台湾数据重新分类。

代码位置：

- [family 定义及科学依据](../../src/foodcomp/final_prediction_axes.py#L206-L299)
- [可见输入中移除整个 family](../../src/foodcomp/research_no_foodname_v1.py#L111-L151)
- [固定 family 查询轴和 masked-axis batch](../../src/foodcomp/research_no_foodname_v3.py#L15-L72)
- [训练任务及 loss](../../scripts/run_foodnutrigpt_no_foodname_v3.py#L70-L107)
- [台湾分析项继承冻结 family](../../scripts/audit_taiwan_tfda_external_validation.py#L140-L155)
- [台湾推理按 family 分组](../../scripts/run_foodnutrigpt_no_foodname_v38.py#L87-L120)

## 22 个实际 family 的分类依据

| Family | 内部轴：监督/全部 | 台湾计分轴 | 分类依据 | 为什么整组 mask | 主要审计意见 |
|---|---:|---:|---|---|---|
| `food_matrix` | 2/2 | 2 | 水和灰分属于食品基质/近似分析组成；FAO/INFOODS、USDA FDC | 避免水或灰分直接帮助恢复同组近似组成 | 只有两个轴；但宏量营养仍在其他 family，可保留强质量平衡信息 |
| `macronutrient` | 4/4 | 3 | 可利用碳水、总膳食纤维、总脂肪、总蛋白是定义明确的营养总量；INFOODS、NIH DRI、FDC | 让模型不能用一个宏量总量直接推另一个，形成真正跨类重建 | 四者并非相互可换算，整组 mask 很严格；与水/灰分的质量平衡仍跨 family |
| `carbohydrate` | 9/13 | 6 | 单糖、双糖、糖醇、淀粉和总糖等明确碳水表达；INFOODS、NIH DRI | `Sugars, total` 与各糖分量关系紧密，整组隐藏可防明显泄漏 | 与 `Available carbohydrate` 分属不同 family；仍可能存在总量—分量提示 |
| `dietary_fibre` | 3/3 | 0 | 纤维化学成分或明确纤维分级：纤维素、不溶性纤维、木质素；INFOODS、NIH DRI | 避免纤维分量之间互相提示 | `Dietary fibre, total` 在 `macronutrient`，没有一起隐藏，是当前最明确的跨 family 边界问题之一 |
| `mineral` | 14/14 | 9 | 具有营养作用的元素/离子；WHO/FAO 微量营养、NIH DRI、INFOODS | 构造“由非矿物组成重建矿物谱”的任务 | 所有矿物整组隐藏不是防代数泄漏所必需；属于较强的任务假设 |
| `vitamin` | 30/30 | 16 | 维生素、维生素化学形式和前体；WHO/FAO、NIH DRI、INFOODS；活性当量被排除 | 同一维生素的不同形式和总量很相关，整组隐藏可防形式间泄漏 | 把 B1、C、D、E 等全部一起隐藏过宽；`Lutein + zeaxanthin` 与独立 carotenoid 轴跨 family；`Choline, total` 与 free choline 也跨 family |
| `amino_acid` | 20/20 | 18 | 单个氨基酸及直接测量的蛋白相关分子；FAO 蛋白质量与 INFOODS | 防止氨基酸谱内强相关或总和关系带来简单复制 | 全部氨基酸整组隐藏适合“恢复氨基酸谱”，但不是预测单一氨基酸所必需；总蛋白在另一 family |
| `fatty_acid` | 54/55 | 25 | 明确链长、不饱和度、顺反/异构体及定义明确的脂肪酸总量；FAO 脂肪酸、LIPID MAPS、INFOODS | 此 family 内同时有单体与饱和/单不饱和/多不饱和/反式总量，整组 mask 的防泄漏理由最强 | `Fat, total` 在 `macronutrient`，仍是强代理但不是脂肪酸总和的严格同义量；该 family 占内部 54/187 个目标，权重很大 |
| `sterol` | 5/6 | 1 | 胆固醇、植物甾醇及定义明确的甾醇总量；LIPID MAPS、INFOODS | `Phytosterols` 总量与 beta-sitosterol、campesterol、stigmasterol 等分量关系紧密 | 台湾只计分胆固醇一个轴，因此在台湾相当于单轴 mask；内部任务则有总量—分量防泄漏意义 |
| `ethanol` | 1/1 | 0 | 乙醇是独立、可直接测量的食品成分；INFOODS、FDC | 单轴 family，实际等同于只隐藏目标轴 | 台湾仅 5 个 sample-backed 食品，未进入最终 80 轴评分 |
| `organic_acid` | 7/14 | 0 | 结构明确的食品有机酸/小分子；INFOODS、ChEBI | 形成恢复有机酸谱的任务 | 多数酸之间没有代数关系；整组 mask 主要是化学分类驱动，而不是防直接泄漏 |
| `nitrogenous_metabolite` | 7/10 | 0 | 胺类、胆碱、牛磺酸等结构明确的含氮食品代谢物；INFOODS、ChEBI | 形成恢复含氮小分子谱的任务 | 化学和生物角色较异质；free choline 与 `Choline, total` 被分开 |
| `carotenoid` | 1/4 | 0 | 膳食类胡萝卜素及相关异戊二烯化合物；INFOODS、ChEBI | 隐藏同类色素，避免近邻形式提示 | 只有 Lutein 是监督目标；alpha/beta-carotene、lycopene、lutein+zeaxanthin 在 `vitamin`，边界不一致 |
| `phenolic_acid` | 8/13 | 0 | 结构明确的食品多酚/酚酸，不包括无化学身份的“总酚”检测；Phenol-Explorer、ChEBI | 形成酚酸谱补全 | family 名称比实际范围窄；与 flavonoid、anthocyanin、isoflavone、lignan 的边界是项目级子类切分 |
| `flavonoid` | 9/19 | 0 | 明确黄酮及其化学形式；糖苷与苷元保留为不同轴；Phenol-Explorer、USDA flavonoid、ChEBI | 减少同一黄酮骨架/不同形式之间的直接提示 | 同类内仍有多个化学亚类；共同 mask 是谱恢复选择，不是统一标准 |
| `anthocyanin` | 4/6 | 0 | 明确花青素/花青苷元；Phenol-Explorer、ChEBI | 防止高度近缘花青素形式互相提示 | 实际监督轴以 cyanidin、delphinidin 等苷元为主；代码标签把 anthocyanidin/anthocyanin 合并为一个任务族 |
| `isoflavone` | 3/9 | 0 | 明确异黄酮及相关植物雌激素；USDA isoflavone、Phenol-Explorer、ChEBI | 防止苷元与糖苷等近缘形式互相提示 | 化学依据较清楚，但整组 mask 仍是项目任务设计 |
| `lignan` | 4/6 | 0 | 明确植物木脂素/相关苯丙烷类；Phenol-Explorer、ChEBI | 形成木脂素谱补全 | 轴较少，整组 mask 主要基于化学类属 |
| `alkaloid_or_bioactive` | 2/9 | 0 | 咖啡因、可可碱等结构明确的生物碱或其他非营养活性物；INFOODS、ChEBI | 避免近缘生物碱互相提示 | 这是最宽泛的 family，包含 capsaicin、curcumin、piperine 等不同结构；应视为工作性分组，不是严格单一化学家族 |
| `nucleotide_or_purine` | 0/5 | 0 | 核苷/嘌呤相关结构明确小分子；INFOODS、ChEBI | 原计划用于同类谱补全 | 当前没有监督轴，所以不会生成该 family 的 mask 任务；这些轴只作为其他任务的可见 context |
| `organosulfur_or_glucosinolate` | 0/8 | 0 | 硫代葡萄糖苷、异硫氰酸酯和有机硫分子；INFOODS、ChEBI | 原计划用于含硫代谢物谱补全 | 当前没有监督轴，不会被作为一个 family 整体 mask |
| `stilbene` | 0/1 | 0 | 结构明确的食品 stilbene；Phenol-Explorer、ChEBI | 原计划用于同类补全 | 当前只有 context-only 的 pterostilbene，没有监督任务，因此从未触发整组 mask |

完整轴归属见 [`axis_membership.csv`](axis_membership.csv)，family 汇总及所有轴名见 [`family_summary.csv`](family_summary.csv)。

## 台湾外部验证到底用了哪些 family

台湾审计手工核对了 86 个分析项与内部轴的成分身份和单位。其中 80 个轴满足“至少 20 个 sample-backed 食品”的计分门槛，覆盖 8 个 family：

| Family | 台湾计分轴 | 占 80 轴比例 |
|---|---:|---:|
| `fatty_acid` | 25 | 31.25% |
| `amino_acid` | 18 | 22.50% |
| `vitamin` | 16 | 20.00% |
| `mineral` | 9 | 11.25% |
| `carbohydrate` | 6 | 7.50% |
| `macronutrient` | 3 | 3.75% |
| `food_matrix` | 2 | 2.50% |
| `sterol` | 1 | 1.25% |

前三个 family 合计 59/80，即 73.75% 的最终 macro-axis 指标。因此“台湾 zero-shot 更好”主要是这 8 个 family，尤其是脂肪酸、氨基酸和维生素上的结论，不能外推到 22 个 family。

台湾数据还有 6 个已映射但未计分的轴：`Sugars, total`、乙醇、三个脂肪酸总量和 `Vitamin D (D2 + D3)`。它们可以作为数字 context，但预测同一 family 时仍会一起被隐藏。

## 台湾逐 family 结果

下表是 2,212 个食品的 strict panel，数值为 scaled-log MAE。最终总分按轴等权，不按 family 等权。

| Family | 轴 | V5+CCE | RF | XGBoost | KNN | 最低者 |
|---|---:|---:|---:|---:|---:|---|
| amino acid | 18 | 0.1056 | 0.1058 | **0.1047** | 0.1923 | XGBoost |
| carbohydrate | 6 | **0.2754** | 0.4534 | 0.4307 | 0.4421 | V5+CCE |
| fatty acid | 25 | **0.2265** | 0.3670 | 0.3489 | 0.3951 | V5+CCE |
| food matrix | 2 | 0.1242 | **0.0899** | 0.0955 | 0.1835 | RF |
| macronutrient | 3 | 0.1451 | 0.1416 | **0.1353** | 0.2207 | XGBoost |
| mineral | 9 | **0.2767** | 0.3043 | 0.3474 | 0.3728 | V5+CCE |
| sterol | 1 | 0.2648 | 0.2434 | **0.2224** | 0.2865 | XGBoost |
| vitamin | 16 | **0.3569** | 0.4688 | 0.4511 | 0.4781 | V5+CCE |

所以总体优势并不是“每个 family 都胜出”。V5+CCE 在 4/8 个 family 最好；它在 carbohydrate、fatty acid、mineral 和 vitamin 上的优势推动了总指标。完整数值见 [`taiwan_family_metrics.csv`](taiwan_family_metrics.csv)。

## 当前分类的科学性判断

### 可以 defend 的部分

- family 在训练前冻结，不根据台湾测试结果调整。
- 台湾、内部数据和所有模型使用同一 family，模型之间的比较没有因 mask 规则不同而失衡。
- 脂肪酸、糖、甾醇以及同一维生素的不同形式确实存在总量、分量或近似表达泄漏风险。
- 每个轴仍保留独立身份；同一 family 不代表把数值合并或认为成分相同。

### 不能过度声称的部分

- 22 个 family 不是正式国际 masking 标准。
- `mineral`、`vitamin`、`amino_acid` 等整组隐藏远超防止直接泄漏的最低要求。
- 当前边界没有完全封住跨 family 的总量/分量提示。
- family 大小悬殊，最终 macro-axis 指标会让大 family 获得更多总权重。
- 台湾只覆盖其中 8 个 family；其结论不是对完整 187 轴或 22 个 family 的证明。

## 建议的验证实验

为了判断 foundation-model 优势是否依赖当前人工 family，下一步应固定同一 80 个台湾轴，比较三种预先定义的 mask：

1. **Target-only**：只隐藏正在预测的单个轴。
2. **Dependency block**：隐藏目标轴以及明确的总量/分量、同一化学表达或可直接换算的关系轴。
3. **Current broad family**：保持现在的整 family mask。

每种协议都让 Transformer、RF、XGBoost 和 KNN 使用完全相同的可见输入，并分别报告总体与逐 family 结果。若 V5+CCE 在三种协议下都保持优势，foundation-model 结论会明显更稳；若优势只出现在 broad-family mask 下，则结论应限定为“跨家族重建优势”。

## 外部分类依据

- [FAO/INFOODS food-component identifiers](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/)
- [WHO/FAO vitamin and mineral requirements](https://www.who.int/publications/i/item/9241546123)
- [NIH Dietary Reference Intakes](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations/)
- [FAO dietary protein quality](https://www.fao.org/4/i3124e/i3124e.pdf)
- [FAO fats and fatty acids](https://www.fao.org/4/i1953e/i1953e00.pdf)
- [LIPID MAPS classification](https://www.lipidmaps.org/data/classification/LM_classification.php)
- [ChEBI chemical ontology](https://www.ebi.ac.uk/chebi/)
- [Phenol-Explorer compound classes](https://phenol-explorer.eu/compounds)
- [USDA isoflavone database](https://www.ars.usda.gov/ARSUserFiles/80400525/Data/isoflav/Isoflav_R2.pdf)

这些来源支持成分命名、营养角色或化学类别；它们不直接支持本项目的整组 masking 边界。
