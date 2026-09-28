# 营养表征模型综合研究报告：R0–R9

对应英文：[REPORT_EN.md](REPORT_EN.md)。本报告整理完整实验与验证证据；内部验证结论不等于外部泛化或标签真实性证明。

版本范围：截至 R9 固定 MAE 配方三种子复验的完整阶段报告。后续方法实验独立登记，不将尚未运行的分支写成结果；本报告可随研究进展被新版本取代。

## 1. 研究目标与主要结论

当前目标是在数据和既有RF/XGBoost结果冻结的前提下，通过有验证依据的单阶段Transformer方法改善营养补全，并持续报告仅名称预测和营养→名称检索。历史MLP结果保留为方法探索，不替代Transformer的结果。

固定配方lr=3e-4、60轮的三个种子补全主误差为 **0.189184 ± 0.006925**。相对冻结RF，平均改善 **-0.081%**，食品组条件性95%区间 **[-2.173%, 2.074%]**；旧log-MAE相对改善为 **-2.103%**。本配方**未通过预登记的RF复验条件**。完整基线、单种子与三种子结果见第5–6节，不能将条件性区间表述为对任意随机种子和新数据的保证。

采用原登记条件：三固定种子的平均主误差低于固定RF，食品组配对区间支持改善，旧营养log-MAE平均相对退步不超过2%。没有5%最低改善要求，也不要求超过XGBoost。全部三项任务使用各自种子补全选中的同一个检查点，不拼接任务最佳模型。

## 2. 数据来源、实际训练数据与标签可信度

### 2.1 直接数据来源与上游来源

代码起点为[Representation-model](https://github.com/mz1622/Representation-model)，父提交`1289d38129dffb7d3490239fb516328fa5c905e3`。本次直接使用用户提供的`foodnutrigpt_v8_source_native_baseline.tar.gz`，保持原包和V8产物不变。该包记录25个来源、76915个源档案、2259178条原始观测；这些数目不等于本次训练样本数。

实际研究视图只载入训练与验证：64700/11175档案，分别42282/7409个完全同名候选组、24个来源。USDA Foundation的395个保留档案属于原包测试来源；原包完整测试含659档案，另有381个Foundation同名重叠档案排除。以上测试信息仅来自原包划分元数据，本轮没有加载测试标签作模型评价。原历史测试已有上游研究结果，因此即便未来打开也不能称为从未被研究过的新外部数据。

以下训练/验证数量由隔离视图重新统计。来源链接用于识别数据库家族；后续下载清单可能对应更新版本，**链接不证明V8使用了该网站当前版本，也不替代逐记录原始溯源**。

|数据源 / Source|训练档案 / Train profiles|验证档案 / Validation profiles|训练营养观测单元 / Observed nutrition cells|
|---|---:|---:|---:|
|[Australian Food Composition Database](https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/data-files)|1314|247|35245|
|[FAO/INFOODS AnFooD 2.0](https://www.fao.org/food-composition/tables-and-databases/detail/%28global--2017%29-fao-infoods-analytical-food-composition-database---version-2.0-%28anfood2.0%29/en)|601|72|3979|
|[Bangladesh Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|437|57|7122|
|[FAO/INFOODS BioFoodComp 4.0](https://www.fao.org/infoods/infoods/food-biodiversity/en/)|959|140|964|
|[German Nutrient Database BLS 4.0](https://blsdb.de/download)|6051|1089|94523|
|[ANSES-Ciqual](https://ciqual.anses.fr/)|2945|521|92130|
|[Canadian Nutrient File](https://food-nutrition.canada.ca/cnf-fce/)|5044|922|298207|
|[UK Composition of Foods Integrated Dataset](https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid)|2414|423|34126|
|[EFSA EU Food Composition Database (package label: 2013)](https://www.efsa.europa.eu/en/data-report/food-composition)|14621|2324|147914|
|[USDA Food and Nutrient Database for Dietary Studies](https://fdc.nal.usda.gov/data-documentation.html)|4612|790|212152|
|[FooDB](https://foodb.ca/about)|9467|1780|239188|
|[Frida FoodData, Denmark](https://frida.fooddata.dk/)|1167|188|50658|
|[Lesotho Food Composition Table 2006](https://www.fao.org/infoods/infoods/tables-and-databases/africa/en/)|250|40|6153|
|[Japan MEXT Food Composition Tables (package label: 2023)](https://www.mext.go.jp/a_menu/syokuhinseibun/index.htm)|2148|390|80926|
|[Norwegian Food Composition Database](https://www.matvaretabellen.no/en/api/)|1808|287|61542|
|[FAO/INFOODS/IZiNCG PhyFoodComp 1.0](https://www.fao.org/food-composition/tables-and-databases/detail/%28global--2018%29-fao-infoods-izincg-global-food-composition-database-for-phytate---version-1.0-%28phyfoodcomp1.0%29/en)|1879|307|4956|
|[SMILING Cambodia Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|78|11|1631|
|[SMILING Indonesia Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|91|17|1182|
|[SMILING Laos Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|128|12|1790|
|[SMILING Thailand Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|122|17|1921|
|[SMILING Vietnam Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|130|32|1562|
|[Swiss Food Composition Database 7.1](https://valeursnutritives.ch/en/downloads/)|1078|168|25800|
|[USDA Standard Reference Legacy](https://fdc.nal.usda.gov/data-documentation.html)|6481|1191|363116|
|[FAO/INFOODS Western Africa Food Composition Table 2019](https://www.fao.org/food-composition/tables-and-databases/detail/food-composition-tables/en)|875|150|28346|
|USDA Foundation Foods（保留来源 / held out）|0|0|0|

来源之间可能存在转载、借用值、配方计算与分析定义差异。来源标签不等于独立实验室测量，24来源也不等于24个统计独立数据集。当前完整别名、翻译与转载链审查尚未完成。

原包身份核验：用户下载包为135,711,396字节，SHA256为`83ee2b50f04963943797aa1818a91f66d0ffaeb7a5bc3c5b2db7a11be6dff792`。10个原生载荷文件与包内清单、本地V8及R0冻结输入指纹一致。此项只验证文件身份，不证明上游发行真实性、授权或营养标签正确。

### 2.2 单位、聚合与隔离

原包数值以g/100g表示；本轮不因常识猜测改动异常值。对同一源档案×轴，统一在原始单位空间取中位数，保留原始观测，不跨来源自动合并。食品候选组仅是划分和权重单元，不当作已确认食品身份。

FooDB存在同档案、同轴正值相差10³/10⁶的记录；Goose fat/Cholesterol案例仍缺原始Content.csv/staging证据，无法确定错误发生位置。隔离64个单元（训练53、验证11，涉及145条原始记录），同时保留未隔离视图用于既有敏感性分析。当前单位解析器对mg/100 g与mg/100g的换算一致，不能因此判定历史加工链无误。湿重/干重、可食部分及化学形式的原始证据仍未全部闭环。

研究视图隔离前含2221276条原始观测、2153204个档案×轴单元；隔离后训练有1829199个已观测252轴单元，其中187轴监督1828536个、142营养轴1795133个。训练营养中显式零388743个。验证有323941个已观测单元，323809个监督目标，其中营养317616个、显式零69271个。

缺失没有进入目标，显式零仍是观测和目标。计算张量中隐藏位置填0，同时使用单独可见标记；这是输入表示，不是把未知标签设成真实零。异常隔离不能证明剩余标签全真，也不能解决所有系统性单位错误。

### 2.3 变换、划分和统一评分

每轴变换为 `t_a(y)=log(1+y/s_a)`，其中`s_a`是仅训练正值拟合、候选组内来源平衡的加权中位数。尺度、词嵌入投影和所有归一化均不使用验证/测试标签拟合。尺度相差极大，所以主要指标在模型比较前固定为142轴宏平均缩放log-MAE；不按某模型获胜选择指标。

评分在候选组×轴×来源内分别取标签和预测的档案中位数，计算该来源误差；再对组内来源等权、食品候选组等权，最后对轴等权。另报告旧log1p(g/100g) MAE、原始单位MAE、正值/显式零条件误差、45轴和187轴。条件正/零MAE分母不同，不能直接相加；机制分解使用原主指标分母。

验证标签、查询、家族遮蔽和权重共享。全部完全同名候选组跨训练/验证隔离通过；近似名称候选不自动合并，现有隔离不等于所有语义近重复均已排除。主指标配对区间用7344个有营养评价支持的食品组，不能把每个营养单元当独立样本。

## 3. 当前 Transformer 训练方法及参数依据

### 数据、目标与输入

沿用R0的`foodnutrigpt_v9_r0_v1 / quarantined`视图：训练64,700个来源原生档案、42,282个名称候选组；验证11,175个档案、7,409个候选组。各部分均覆盖24个来源。当前不修改数据、划分、隔离规则、尺度、任务或名称缓存；RF/XGBoost不再拟合，完整测试保持关闭。原始单位疑点和未确认别名仍是研究局限，不因冻结而得到解决。

252个轴中，142个nutrition轴和45个metabolome轴接受监督，65个轴仅提供已观测上下文。每个档案内同轴的重复记录在原始单位中取中位数，不跨来源合并。缺失值不监督；显式零仍是观测。训练目标为`t_a(y)=log(1+y/s_a)`，`s_a`是训练正值按名称候选组、来源和档案平衡的加权中位数。训练187轴共有1,828,536个已观测目标，其中142个营养轴有1,795,133个目标、388,743个显式零。

名称分支仅使用`original_name`。冻结MiniLM提供384维向量，再使用已存在、仅由训练数据拟合的PCA；保留32个有效方向，放入128个槽位，其余96个为零。选择该输入是为了匹配已经完成的最强共同协议RF32，不是宣称32维最优。数值分支使用252轴的变换值与可见状态；不输入来源、食物分组或加工元数据。

每个训练任务指定一个食品档案和一个化学家族，隐藏整个家族，包括关联形式；损失仅使用其中已观测且可监督的目标。每轮遍历337,048个任务，每个已观测训练目标恰好覆盖一次。没有额外的name-only混合训练；name-only是本组必须报告的推理评估情景，不能描述成已针对其优化的模型。

### 模型结构与数值输出

序列包含1个可学习CLS token、1个名称token和固定的252个轴token，共254个token，不作截断。名称经过LayerNorm及两层GELU投影；每个可见数值经过`1→192→192`的GELU MLP，再加轴嵌入。缺失或隐藏位置用可学习mask向量替代数值编码。所有轴始终存在，查询轴不由真实标签是否存在决定。没有位置嵌入、类型嵌入或来源token。

编码器使用pre-norm Transformer及最终LayerNorm。每个轴的隐藏状态经过共享非线性标量头，并加rank-16的轴残差和轴偏置。直接回归变换后的数值`z_hat`；推理还原为`y_hat=s_a*expm1(max(z_hat,0))`。训练损失使用未经该非负截断的`z_hat`。当前预测不乘正值概率；父实现的presence分支仍构造和调用以保持初始化及dropout随机数消耗顺序，但其参数冻结，输出不进入损失或预测。

|配置字段|值|
|---|---|
|`d_model`|192|
|`n_layers`|3|
|`n_heads`|6|
|`feedforward_dim`|768|
|`dropout`|0.15|
|`axis_residual_rank`|16|
|`parameter_count`|1548594|
|`trainable_parameter_count`|1515929|
|`active_name_dimensions`|32|
|`name_input_slots`|128|
|`batch_size`|64|
|`epochs`|60|
|`schedule_epochs`|60|
|`learning_rate_screening / selected`|0.0001, 0.0003 / 0.0003|
|`weight_decay`|0.0001|
|`gradient_clip`|1.0|
|`source_weight`|1.0|
|`source_residual_l2`|0.0001|
|`eta_min_fraction`|0.01|
|`screening_seed`|20260922|
|`confirmation_seeds`|20260922, 20260923, 20260924|

### 损失、来源校准与优化

令`w_ia=1/(n_sources(g,a)*n_profiles(g,a,s))`，即每个名称候选组与轴内，各来源等总权重，同来源的多个档案再平分；`W_a`是训练轴权重之和。全数据直接回归目标为：

`L_base = (1/187) * sum_a [sum_i w_ia*abs(z_hat_ia-t_a(y_ia)) / W_a]`。

小批量仅累加当前任务的已观测目标，并乘`N_tasks/(batch_tasks*187)`，使用全训练集轴分母；不是对每个小批量内出现的轴重新取平均。训练与评分的聚合不完全相同：评分先在候选组、来源、轴内对档案标签和预测分别取中位数，再计算误差；训练对档案误差加权求和。因此“训练loss等于验证主指标”不是本实验的假设。

来源只参与训练校准：`z_calibrated=z_hat+b_sa`。每轴偏置在训练来源上做算术均值中心化；未见来源偏置为零。总损失为`(L_base + lambda*L_calibrated)/(1+lambda) + rho*mean(R_train^2)`，其中`lambda=1`、`rho=0.0001`，正则项作用于中心化前的训练来源参数表。来源偏置不进入推理。除以`1+lambda`控制两项损失的整体尺度；来源开关的实际收益仍需单独消融。

优化器为AdamW，两个候选仅初始学习率不同；batch64、weight decay 0.0001、梯度范数裁剪1。单阶段60轮，余弦日程`T_max=60`，最低学习率为初始值的0.01。每轮打乱由`seed+epoch`确定。使用float32，不启用混合精度。每轮核对完整任务覆盖、目标数、顺序指纹和冻结文件指纹，并记录裁剪比例及裁剪前梯度范数。NaN/Inf触发失败。

正式优化前，对32个训练任务进行50步过拟合检查；随后恢复模型参数与CPU/CUDA RNG，再创建正式优化器。此检查仅证明实现可以降低小样本训练损失，不证明泛化能力。

### 选点、三项能力与比较

每轮评价固定的完整验证面板，以142轴宏平均scaled-log MAE选择最早严格最小值；同时保存前20轮最优和前60轮最优检查点。20/60比较来自同一60轮余弦轨迹，包含训练预算增加和更多验证选点机会，不能称作独立的纯时长因果实验。

补全隐藏目标整个家族；name-only隐藏全部数值上下文。两项都只在有真实观测的目标上评分，同时报告142/45/187轴、旧log误差、原始单位误差及正值/零值误差。候选组内来源等权，再对食品候选组和轴取平均。

营养→名称检索采用同一检查点的“名称→预测营养→匹配”方法，不是本轮另行训练的对比学习头。49,913个固定候选名称仅从名称生成142轴预测；查询只使用已观测营养，在可见轴上计算变换空间均方距离。报告全可见及固定30%可见两种情景，至少有3个可见营养轴；距离相同按候选顺序确定名次。评价Recall@1/5/10与MRR，正确答案仅为精确原名称；尚无已确认别名表，不能称为语义身份准确率，也不把距离称为概率。

首组筛选种子为20260922；固定配方随后以20260922、20260923、20260924完成三种子复验。食品组配对重采样区间不消除反复验证选点的偏差，也不包含固定RF自身的种子不确定性。接受目标是Transformer主误差均值低于固定RF且配对区间支持改善，旧营养log-MAE相对退步不超过2%；不是要求超过XGBoost。

### 参数依据与尚未回答的问题

|选择|依据|当前允许的解释|
|---|---|---|
|192维、3层、6头、FF768|沿用直接Transformer父实现，功能核验精确匹配|用于建立同输入控制；尚未证明容量最优|
|学习率0.0001/0.0003|预登记单因素比较|同种子、同初值和同任务顺序的完整对照支持选择0.0003；具体收益见第7节|
|60轮及20轮快照|预登记预算诊断|完整嵌套窗口对照支持60轮窗口；不能分离训练时长和额外选点机会|
|直接MAE|旧R2直接回归研究提供动机，当前重建控制|旧输入协议不同，不能直接迁移历史收益归因|
|source weight1、残差正则0.0001|沿用来源校准控制|尚未在当前输入、目标和时长下证明优于关闭校准|
|batch64、裁剪1、weight decay0.0001|固定工程选择及父训练设置|可复现，不称为充分调参后的最优值|
|dropout0.15、rank16|父结构与PDF方法提供动机|尚无当前协议的独立消融支持|
|PCA32、数据及评分|冻结共同输入与比较协议|本轮不调整；不属于可用来求胜的超参数|

PDF给出256维、3层、8头、FF1024及宏轴MSE等方法，但数据合并、监督标签、归一化和评价均不同，初始学习率也未明确。当前只借鉴方法；没有复现其成绩，也不恢复两阶段。后续容量组或MSE干预须先依据首组完整结果另行登记。任何新分支都不自动成为最终配方。

### 复现与证据边界

原始运行实现提交为`b4fd4c1`。后续文档提交不同于训练实现变更；精确代码身份以每个运行的源码快照和哈希为准。环境为Python3.10.19、PyTorch2.7.1+cu128，GPU为RTX5070Ti16GB，训练使用4个CPU线程。实际计算成本以完整运行manifest与history为准，不用计划估时充当实测。

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr1e4_60 --output-dir output/v9_r9/tf192_mae_lr1e4_60
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr3e4_60 --output-dir output/v9_r9/tf192_mae_lr3e4_60
```

上述是已登记运行命令；存在的输出目录拒绝覆盖，不是要求再次启动。完整运行后另有独立评分重放、预算比较、学习率比较及全训练拟合诊断。功能验收已完成，但不能代替这些性能证据。逐轮记录、三项任务、固定配方复验及选择依据汇总在下文；这些证据仍不建立foundation-model迁移能力。

证据索引：[配置](../r9/config.json)、[预登记](../r9/PLAN.md)、[冻结凭据](../../../reports/v9_r9_freeze_v1/manifest.json)、[功能核验](../../../reports/v9_r9_functional_v1/verification.json)、[数据汇总](../../../reports/v9_final_data_evidence_v1/summary.json)、[基线方法](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_ZH.md)。数值预测与检查点保存在本地忽略目录，本说明不包含原始食品营养值。

### 三个公共推理接口

`TransformerNutritionModel`提供`predict(food_name, observed_profile, target_axes)`、`encode(food_name=None, observed_profile=None, modality="fused")`和`retrieve_names(observed_profile, candidate_names, top_k=10)`。预测目标由调用者指定，必须是未放入已知上下文的监督轴；空营养上下文即name-only。65个context-only轴没有有效预测保证，接口拒绝将其作为输出。

`encode`支持name/nutrition/fused，当前返回192维轴token隐藏状态均值；名称模式隐藏数值，营养模式不看名称。这是可复用探针接口，不是经迁移验证的对比表征。检索只接收营养查询，候选向量来自名称预测，返回负变换空间MSE及明确的非概率标记。接口至少需要一个营养观测，正式检索面板另要求至少三个。

已完成检查点20260923的3个缓存训练名称和1个训练档案通过12项接口检查，包括模态隔离、未观测轴、候选顺序/重复不变性、零值查询和分数重算；仅为此检查点及这些输入提供功能证据，不作为新的性能或迁移实验。完整记录：[Public interface audit](../../../reports/v9_r9_public_interfaces_seed23_v1/README.md)。

## 4. 逐轮研究路径与归因

历史完整配置、失败、逐轴/来源及三任务均保留于每轮报告。下表不是把跨版本最好分数串成一个因果曲线；R0→R1训练协议和R7/R8名称输入变化必须分别解释。

|版本|问题、干预与关键证据|解释边界与决定|
|---|---|---|
|[R0](../r0/README.md)|统一原始空间聚合、训练尺度、name-only文本、查询/遮蔽；12配置基准与异常隔离|建立可比评价；标签溯源未闭环，训练遮蔽当时仍不同，不能纯归因架构|
|[R1](../r1/README.md)|全量共同家族任务及全局轴权重；固定日程8→20轮，MLP主误差.246757→.222459；V9 amount1→2→3与source开关|时长对照支持增加预算；amount提高改善该V9分支；去来源未获明确补全收益。面板与损失修复共同变化，不作单因素归因|
|[R2](../r2/README.md)|12预算：SmoothL1→MAE .222459→.209484；256→512 .209484→.202428；同60轮日程内20→60 .209506→.184491；另试1024、LayerNorm、直接V9、选点与名称标准化|保留MAE/512/60及LayerNorm。MAE主要改善零值；加宽1024补全区间跨零且name-only变差；hurdle→direct改动多项，不能只归因概率乘法|
|[R3](../r3/README.md)|4配置：name-only比例10%/20%、独立任务头及相应诊断|部分名称收益伴随补全代价，未采用到历史补全MLP；未把所有取舍都称为梯度冲突|
|[R4](../r4/README.md)|8预算，5完整含复用、3数值失败；查询残差、视图一致性、代谢物权重.5；修正共同逆变换/查询契约|无足够补全收益；失败保留，不用NaN跳过或宽容差包装成功；组件修复和性能变化分开|
|[R5](../r5/README.md)|12配置：名称预测、营养—名称对齐与部分视图；完整输入独立检索三种子R@10 .298199±.003720，相对当时旧KNN .192192|仅完整输入专项模型优势，30% R@10仅.002252；不是补全模型，也未证明相对后来PCA128近邻优势|
|[R6](../r6/README.md)|固定30%及混合30/60/90额外上下文删除；补全.197232/.210462，父.184491|固定评价面板上均退步，不采用；名称改善不能替代补全结果|
|[R7](../r7/README.md)|6配置；公共PCA32→128，名称MLP/KNN主误差改善8.84%/6.87%|保留128输入方向；MLP仍比同128近邻差5.11%，不能与旧输入树直接比较|
|[R8](../r8/SCOPE_CHANGE_CLOSURE.md)|原 12 条目中 10 完整、1 部分、1 未运行；6 个树配置完整，MLP32→128 补全改善 .931% 且区间跨零，名称改善 18.909%|按用户约束冻结；不再拟合树，不宣称完整 RF 搜索或树三种子确认|
|[历史 MLP 三种子](../final_report_v1/NEURAL_REPLICATION.md)|固定 MLP 配方三种子已完成，全量重放通过；补全 0.183435 ± 0.001222|历史 PCA128 辅助结果，不作为当前 Transformer 完成证据|
|[R9](../r9/README.md)|共同 PCA32、192维直接 Transformer，MAE、source1，两个学习率各完整60轮；3e-4追加两个固定种子|1e-4未胜RF；3e-4筛选种子0.183553；固定配方三种子结果及决定见第6–8节|

历史 MLP/PCA128 配方三种子补全为 0.183435 ± 0.001222，name-only 为 0.484042 ± 0.010458；这些数值不是本轮 PCA32 Transformer 的结果。其名称清空/置换及常见轴正值低估诊断也不能直接转移到 Transformer。完整参数、逐种子结果及失败见[旧阶段稿](../final_report_v1/REPORT_ZH.md)与各轮记录。

R9 数值编码组件此前仅做过训练集可行性检查，没有投入本轮训练。R9 首组仅筛选两个学习率，追加种子为固定配方重复，没有把可行性检查计作新模型性能证据。

## 5. 冻结 RF、XGBoost 的具体方法和完整结果

### 共同输入与监督

每项配置独立拟合187个逐轴回归器，覆盖142营养轴及45代谢组轴；65个其他轴只提供已观测上下文。每轴使用全部合格训练档案，无5000行上限。共有1,828,536个训练目标，最大轴58,958行。显式零是标签，缺失不进训练目标，不跨来源合并数值。

632个输入槽位为128名称槽位、252变换后数值和252可见标记。32维设置只启用相同训练集PCA基底的前32个名称方向，其余96槽置零；128维设置启用全部。名称严格只取food name。补全时目标家族全部隐藏，name-only时数值与可见标记全部隐藏。变换t=log(1+y/s)中的s只由训练集拟合，逐轴来源权重按均值归一后传入sample_weight。

### 拟合方法及实际调参范围

RF使用scikit-learn 1.5.2：400棵bootstrap树、平方误差、无深度上限、min_samples_leaf=1、max_features=0.5；32/128输入各完成一项。XGBoost 2.1.3使用800棵树、hist、learning_rate=0.03、min_child_weight=5、subsample=0.8、colsample_bytree=0.8、reg_lambda=1、reg:squarederror，不使用early stopping；深度及输入见表。两类逐轴random_state=20260922+axis_index、拟合4线程。RF推理按固定树顺序串行累加。完整get_params保存在summary.json，不把未显式设置的库默认参数冒充搜索结果。

原登记8项树配置中6项完整；RF leaf3/feature0.5/name128在123/187轴中断，RF leaf1/feature1.0/name128未运行，原因是用户冻结树实验。二者无完整分数，不参加选优，不能宣称RF的3配置搜索完成。没有新增树种子；下列都是seed20260922的固定参照，不提供虚构的种子标准差。

|Run|Name dim|Trees|Depth|Min leaf|Feature fraction|Preparation + fit sum (s)|Run elapsed (s)|
|---|---|---|---|---|---|---|---|
|rf400leaf1half_name32|32|400|unlimited|1|0.5|5457.2|5807.5|
|rf400leaf1half_name128|128|400|unlimited|1|0.5|16821.1|17318.3|
|xgb800d10_name32|32|800|10|N/A|N/A|1607.6|1686.6|
|xgb800d10_name128|128|800|10|N/A|N/A|2753.6|2833.4|
|xgb800d6_name128|128|800|6|N/A|N/A|1359.1|1437.2|
|xgb800d14_name128|128|800|14|N/A|N/A|5733.2|5861.3|

时间为各作业实测：逐轴准备与拟合阶段耗时之和，以及包括预测/核验的整次作业耗时。原始fit_seconds计时从特征构造之前开始，包含准备和检查，不能称为纯fit调用耗时。此前存在并发计算，也不能当作严格控制的算法速度对照。

### 营养预测结果

全部是内部验证结果。主指标是142轴宏平均缩放log-MAE，先在食品组×来源层面等权。误差越低越好；不同名称维度分开识别，不将跨维度差异全部归因于架构。

|Run|Completion 142|Legacy log 142|Name-only 142|Completion 45|Completion 187|
|---|---|---|---|---|---|
|rf400leaf1half_name32|0.189031|0.056270|0.659518|0.674196|0.305782|
|rf400leaf1half_name128|0.199422|0.060131|0.619922|0.698854|0.319606|
|xgb800d10_name32|0.174487|0.052593|0.656148|0.674273|0.294756|
|xgb800d10_name128|0.181550|0.055256|0.612496|0.676969|0.300768|
|xgb800d6_name128|0.189261|0.057610|0.597474|0.675860|0.306357|
|xgb800d14_name128|0.182485|0.055811|0.621506|0.677074|0.301504|

|Run|Raw MAE (g/100g)|Positive scaled MAE|Zero scaled MAE|
|---|---|---|---|
|rf400leaf1half_name32|0.321219|0.248086|0.168645|
|rf400leaf1half_name128|0.340156|0.261008|0.182566|
|xgb800d10_name32|0.298575|0.230310|0.146588|
|xgb800d10_name128|0.306447|0.240916|0.160178|
|xgb800d6_name128|0.322449|0.249631|0.164759|
|xgb800d14_name128|0.309121|0.241528|0.161849|

正值与零值条件误差分母不同，不能相加得到整体主指标。全部任务和子集的原始精度指标见summary.json。

### 营养到名称检索

同一套已拟合逐轴模型只根据候选名称生成营养向量；查询不含名称，候选不使用真实营养档案。候选库固定49,913个名称。0.3/1.0可见比例分别有8,610/10,479个合格查询，要求至少3个已知营养轴。名称正确性按完全相同原名，尚无已确认别名映射。下表全部为0–1比例；不同可见比例的查询群体不同，不是纯可见性对照。

|Run|Visible fraction|MRR|R@1|R@5|R@10|
|---|---|---|---|---|---|
|rf400leaf1half_name32|0.3|0.001477|0.000310|0.001239|0.001987|
|rf400leaf1half_name32|1.0|0.002528|0.000176|0.001566|0.004457|
|rf400leaf1half_name128|0.3|0.001149|0.000000|0.000310|0.001510|
|rf400leaf1half_name128|1.0|0.002827|0.000000|0.001474|0.004721|
|xgb800d10_name32|0.3|0.001559|0.000310|0.001471|0.001819|
|xgb800d10_name32|1.0|0.003417|0.000423|0.002468|0.006275|
|xgb800d10_name128|0.3|0.001408|0.000000|0.000968|0.001936|
|xgb800d10_name128|1.0|0.004482|0.000306|0.003852|0.008533|
|xgb800d6_name128|0.3|0.001779|0.000155|0.000929|0.001768|
|xgb800d6_name128|1.0|0.005945|0.000796|0.006092|0.011207|
|xgb800d14_name128|0.3|0.001439|0.000000|0.001007|0.002671|
|xgb800d14_name128|1.0|0.004076|0.000494|0.003046|0.006381|

### 解释与复现边界

R9同输入32维主要参照固定为RF leaf1/feature0.5及XGB depth10。它们是已完成集合中的参照，不代表完整搜索或所有随机种子的最优结果。本轮Transformer完整三种子结果见第6节；基线附录本身不证明标签全部可靠或foundation能力。

逐轴训练时已检查反序、子批次和内存pickle重载，完成审计也核对了数据、特征、权重及保存的预测/检索排名。森林在预测后释放，没有保存完整森林，因此不能说当前产物支持任意新名称在线树推理。数字预测与候选矩阵保留在本地忽略目录；附录仅含汇总。

Data manifest SHA256: `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`

Freeze manifest SHA256: `1ec58002d76beb329004db77ef512622f57b40e142725799918d20027b5fddf3`

各运行代码、环境、完整参数、运行及拟合记录哈希见[机器结果](../../../reports/v9_r9_frozen_baseline_appendix_v2/summary.json)。两种语言所有数值表格一致；已完成附录不等于整体终稿。

## 6. 全部R9候选、固定配方复验及结果对比

### 6.1 筛选与预算对照（单种子）

下表每一行的三个任务来自同一检查点；误差越低越好，R@10越高越好。正值/零值为条件误差，不能直接相加重构总体。全部候选仅一个训练种子，故不报告虚构的种子标准差。

| 配方／窗口 | 所选轮次 | 补全142 | 旧log | 正值 | 显式零 | 代谢物45 | 全187 | Name-only142 | R@10 全可见／30% |
|---|---|---|---|---|---|---|---|---|---|
| 1e-4 / 20 | 18 | 0.219576 | 0.066269 | 0.358686 | 0.093135 | 0.564114 | 0.302486 | 0.407723 | 0.8061% / 0.3807% |
| 1e-4 / 60 | 60 | 0.194543 | 0.058632 | 0.324743 | 0.086138 | 0.577952 | 0.286807 | 0.414298 | 1.2212% / 0.5618% |
| 3e-4 / 20 | 18 | 0.213405 | 0.063450 | 0.348083 | 0.110594 | 0.587303 | 0.303380 | 0.418662 | 0.5665% / 0.4491% |
| 3e-4 / 60 | 60 | 0.183553 | 0.054362 | 0.302773 | 0.092039 | 0.598022 | 0.283292 | 0.426411 | 1.0603% / 0.6781% |

冻结 RF32：补全142=0.189031，旧log=0.056270，name-only=0.659518，全可见/30% R@10=0.4457%/0.1987%。冻结 XGB32：0.174487、0.052593、0.656148、0.6275%/0.1819%。固定名称KNN32：name-only=0.262260，R@10=19.3572%/9.6722%；此处不冒用为同家族补全基线。六个已完成树配置的具体超参数与成本见[冻结基线附录](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_ZH.md)。

3e-4相对RF主误差改善2.898%，食品组95%区间[0.681%,5.061%]；旧log改善3.390% [0.237%,6.528%]。相对XGB主误差退步5.196%，改善区间[-8.081%,-2.340%]。1e-4相对RF主误差退步2.916%，区间跨零，旧log退步4.198%，不满足2%保护条件。

学习率直接对照：3e-4相对1e-4主误差改善5.649% [0.407%,10.135%]，旧log改善7.282% [2.255%,12.134%]。Name-only点退步2.924%，改善区间[-6.356%,0.366%]，不足以确认退步或等效；完整检索R@10点下降0.1609个百分点，差区间[-0.5006,0.1476]个百分点。30%检索R@10点上升0.1163个百分点，差区间[-0.1465,0.3616]个百分点。

20→60窗口：1e-4补全改善11.401% [10.060%,12.683%]；3e-4改善13.988% [12.260%,15.684%]。这两个嵌套窗口共享60轮日程，且60轮提供更多验证选点机会，不是独立20轮日程与60轮日程的纯训练时长效应。

逐轴、来源、不同可见信息量及正值/零值贡献见机器分析与两个gap目录。所有区间为固定模型条件下1000次食品组配对重采样，不将营养单元或不同种子重复当成独立食品，不覆盖训练随机性、反复验证选择、原始标签或外部泛化的不确定性。

失败记录：原首项独立审计曾因 Windows CP936 解码 UTF-8 名称文件失败。原失败记录保留，独立版本仅增加显式UTF-8读取；未改模型或重训首项。恢复队列及本阶段所有实际分析已成功完成。未发生训练NaN/Inf。

### 6.2 固定配方三个种子的独立结果

以下是同一配方的重复实验，只有种子变化。先评分各模型，再对误差取均值；没有平均预测构造集成，也没有挑选最好种子。均值±样本标准差描述这三个种子的波动，无法用三个值充分刻画所有训练随机性。

|Seed|Selected epoch|Completion142|Legacy log142|Name-only142|R@10 full|R@10 30%|
|---|---|---|---|---|---|---|
|20260922|60|0.183553|0.054362|0.426411|0.010603|0.006781|
|20260923|60|0.187083|0.057067|0.409478|0.009537|0.004091|
|20260924|57|0.196917|0.060931|0.451395|0.009037|0.003394|

|Model|Completion 142|Legacy log 142|Name-only 142|Completion 45|Completion 187|
|---|---|---|---|---|---|
|transformer|0.189184 ± 0.006925|0.057454 ± 0.003301|0.429095 ± 0.021087|0.591929 ± 0.008974|0.286102 ± 0.005996|
|rf|0.189031|0.056270|0.659518|0.674196|0.305782|
|xgb|0.174487|0.052593|0.656148|0.674273|0.294756|
|name_knn|—|—|0.262260|—|—|

固定树各只有一个先前训练的实现，故不填写种子SD；KNN是固定名称基线，没有在此建立营养上下文补全比较。树的name-only来自补全训练后的全部营养输入遮蔽，并非专门优化的name-only树。KNN以每轴已观测训练名称为邻居池，使用相同PCA32空间的KDTree欧氏10近邻；权重为来源平衡权重除以距离加0.001，在变换空间加权平均后还原。未重新调K。

### 6.3 子集、配对区间及检索

|Task|Axes|Scaled MAE|Legacy log MAE|Raw MAE|Positive MAE|Zero MAE|
|---|---|---|---|---|---|---|
|completion|142|0.189184 ± 0.006925|0.057454 ± 0.003301|0.348136 ± 0.021134|0.310514 ± 0.011019|0.092668 ± 0.002517|
|completion|45|0.591929 ± 0.008974|0.035336 ± 0.000486|0.064796 ± 0.001569|0.802609 ± 0.011679|0.224642 ± 0.008340|
|completion|187|0.286102 ± 0.005996|0.052131 ± 0.002397|0.279952 ± 0.016300|0.428933 ± 0.009461|0.116876 ± 0.002874|
|name_only|142|0.429095 ± 0.021087|0.151743 ± 0.005202|0.937501 ± 0.046174|0.552585 ± 0.018126|0.410400 ± 0.045874|
|name_only|45|0.824736 ± 0.085107|0.057004 ± 0.006606|0.104080 ± 0.022302|0.977394 ± 0.100290|0.519521 ± 0.042180|
|name_only|187|0.524302 ± 0.033943|0.128945 ± 0.004151|0.736945 ± 0.029735|0.654812 ± 0.036603|0.430416 ± 0.035985|

正值与零值是条件误差，分母不同。原始单位MAE为g/100g。以下正改善表示Transformer误差更低；区间来自1000次食品组配对重采样，条件于三个已训练模型及固定树结果，不包括选择偏差、树随机性或标签有效性。

|Reference|Task|Metric|Gain (%)|95% conditional interval (%)|
|---|---|---|---|---|
|rf|completion|scaled_log_mae|-0.081|[-2.173, 2.074]|
|rf|completion|log_mae|-2.103|[-6.178, 1.689]|
|rf|name_only|scaled_log_mae|34.938|[33.048, 36.845]|
|rf|name_only|log_mae|29.034|[26.112, 31.515]|
|xgb|completion|scaled_log_mae|-8.423|[-11.026, -5.622]|
|xgb|completion|log_mae|-9.242|[-14.649, -3.638]|
|xgb|name_only|scaled_log_mae|34.604|[32.594, 36.637]|
|xgb|name_only|log_mae|28.999|[25.978, 31.717]|
|name_knn|name_only|scaled_log_mae|-63.614|[-68.879, -58.393]|
|name_knn|name_only|log_mae|-70.329|[-81.946, -58.985]|

检索均为0–1比例。全可见/30%分别有10,479/8,610查询及7,090/6,458食品组；候选名称49,913个。不同可见率的合格食品不同，不能把差异全部归因于可见率。

|Model|Visible fraction|MRR|R@1|R@5|R@10|
|---|---|---|---|---|---|
|transformer|0.3|0.002837 ± 0.000637|0.000413 ± 0.000089|0.002398 ± 0.001140|0.004755 ± 0.001789|
|transformer|1.0|0.005138 ± 0.000647|0.000889 ± 0.000289|0.005164 ± 0.000821|0.009726 ± 0.000800|
|rf|0.3|0.001477|0.000310|0.001239|0.001987|
|rf|1.0|0.002528|0.000176|0.001566|0.004457|
|xgb|0.3|0.001559|0.000310|0.001471|0.001819|
|xgb|1.0|0.003417|0.000423|0.002468|0.006275|
|name_knn|0.3|0.034512|0.005149|0.054796|0.096722|
|name_knn|1.0|0.068879|0.013005|0.117573|0.193572|

逐轴、来源、稀疏支持数及探索性区间：[Summary](../../../reports/v9_r9_three_seed_confirmation_v1/summary.json) / [Per-axis by seed](../../../reports/v9_r9_three_seed_confirmation_v1/axis_metrics_by_seed.csv) / [Per-source by seed](../../../reports/v9_r9_three_seed_confirmation_v1/source_metrics_by_seed.csv) / [Axis paired intervals](../../../reports/v9_r9_three_seed_confirmation_v1/axis_paired_intervals.csv)。逐轴区间未校正多重比较；来源覆盖不同，来源均值不用于评判数据库质量。原始预测留在本地忽略目录。

### 6.4 预登记判定

|Registered gate|Result|
|---|---|
|three_registered_neural_seeds|PASS|
|primary_mean_improvement_over_fixed_rf|FAIL|
|food_group_interval_supports_improvement|FAIL|
|legacy_mean_regression_at_most_2_percent|FAIL|

## 7. 机制、归因与为什么选择这一配方

以下7.1–7.2是完整筛选组的解释，数字均属于种子20260922；不可冒充三种子机制确认。固定配方后续只改变随机种子，新的总体分解见7.3。

### 7.1 筛选组机制诊断

已实际查看[四面板学习曲线](../../../reports/v9_r9_first_group_v1/learning_curves.png)。两组最佳点均为第60轮；后期下降趋缓，未见主误差持续回升或数值发散。3e-4后10轮平均裁剪比例0.350883，1e-4为0.482988；这只是优化轨迹现象，不单独证明裁剪是收益原因。

以同一source-free评分在全部1,828,536训练目标上重新评价：1e-4训练主误差0.143092、验证0.194543；3e-4训练0.134937、验证0.183553。较大学习率同时降低两者。训练与验证的食品及支持不同，差值不能唯一归因为过拟合，在线187轴校准loss也不能直接与验证142轴误差相减。

相对1e-4，3e-4主指标差=-0.010990，由正值贡献-0.014577与零值贡献+0.003587组成：正值收益超过零值代价。相对RF则仍有正值额外误差+0.017036，其中低估+0.012476、高估+0.004560；零值收益-0.022514，净差-0.005478。因此总体获胜不表示正值或所有营养轴都更好。

3e-4对RF有76/142轴点值更好。训练支持<100组的7轴贡献-0.002453，100–999组的20轴贡献-0.005975，≥1000组的115轴贡献+0.002951。1e-4的稀疏7轴原本贡献+0.008669，学习率对照收益很大部分来自这些轴；它们验证支持少，应保留逐轴区间并优先检验种子稳定性，不据此修改标签、指标或采样。脂肪酸家族整体贡献-0.006294；氨基酸、矿物质仍有额外误差。

40个极端成功/失败数值案例保留在忽略的本地数据目录，不混入公开报告，也不作为代表性抽样案例。来源/上下文分层是观察性分层，覆盖不同，不能解释为来源或遮蔽的干预效应。

### 7.2 受控证据与解释边界

已观察事实：固定条件下3e-4取得较低训练与验证补全误差；更长选择窗口得到更低补全误差；name-only和检索未同步改善。受控制实验支持的有限解释：在当前架构、数据、种子、日程及选点规则下，学习率设置影响了优化结果，不能把此前与RF的差距全部解释为Transformer容量不足。

未排除的解释：稀疏轴及随机初始化敏感性、验证选择偏差、来源校准与MAE的交互、容量/损失对正值误差的影响。尚未做本架构MSE、PDF256容量、source_weight的消融，不声称这些因素已最优或没有价值。也没有由训练/验证差值证明过拟合，或由单种子区间证明跨种子稳定胜出。

### 7.3 三种子轨迹与固定分母分解

![Three-seed learning curves](../../../reports/v9_r9_three_seed_curves_v1/learning_curves.png)

[Figure review](../../../reports/v9_r9_three_seed_curves_v1/visual_review.json)记录实际查看图像后的判断。训练187轴含来源校准的loss和验证142轴指标定义不同，不能直接相减判断过拟合。星号均标记同一补全选点，零值或正值图没有另行选点。

相对RF，下列三项使用主指标共同分母，能够相加还原总体误差差；负值表示Transformer有利。它们描述误差来自哪里，不是独立的因果干预。

|Primary-error contribution: Transformer minus RF|Difference|
|---|---|
|positive_under|+0.018121239|
|positive_over|+0.004878008|
|explicit_zero|-0.022845780|

### 7.4 极端案例与标签限制

三种子的补充分解见[逐轴种子诊断](../../../reports/v9_r9_seed_variation_v1/summary.json)及[分区贡献](../../../reports/v9_r9_seed_variation_v1/partition_contributions.csv)。平均误差在64/142轴低于RF；53轴在所有种子下更好，62轴在所有种子下更差。训练支持少于100、100–999、至少1000个候选组的7/20/115轴，分别贡献−0.001095/−0.005428/+0.006676的主误差差，使用固定142轴分母。这说明差距并不只在稀疏轴。

种子24相对22的主误差增加0.013363；支持至少1000组的轴贡献0.008705，少于100组的轴贡献0.003346。Lignin单轴贡献0.002837，但仅17个验证候选组，不能据此单独选择模型或改动权重。20个氨基酸轴中19个在三个种子下均比RF差。共9个营养轴验证支持少于30组，逐轴区间仅作探索解释。此分解支持检验数值损失及优化机制，但尚不能区分MAE目标、来源校准、容量和初始化的各自作用。

已核对筛选父模型40个本地极端档案×轴案例。更差20例全部来自FooDB，其中17例是Cholesterol；这些17例的Transformer预测均比保留的正值标签至少低500倍，而RF更接近记录值。这不证明单位错误、RF泄漏或Transformer更符合营养真值。缺少原始加工链，仍有标签定义、系统校准、正值低估等替代解释。极端案例不是随机食品样本，计数也不等于宏平均贡献；详见[Extreme-case review](../../../reports/v9_r9_parent_extreme_cases_review_v1/README.md)。本轮没有据此改标签、排除更多样本或调整指标。

### 7.5 配方决定与未被验证的参数

选择lr3e-4而非1e-4的直接依据，是同初始化、同任务顺序和相同预算下补全改善；60轮窗口优于前20轮窗口。保留固定配方进行三个种子复验，使优化因素与随机性分开检查。最终条件判定为：**未通过预登记的RF复验条件**。

192维、3层、6头、dropout0.15、rank16、来源校准及MAE没有在当前协议下各自完成全部消融；合理的实现与可复现的设置不等于所有参数已证明最优。学习率对照支持优化方法的局部选择，不能证明attention比所有简单模型更有优势。当前未完成未见来源、类别留出、少样本或冻结表征迁移验证，故仅称为营养表征研究模型。

## 8. 可复现信息、失败记录、版本决定与下一步

数据、划分、文本缓存、尺度及面板沿用冻结R0/R8；测试保持关闭。训练与基线细节见第2–5节，首组登记及决策见[First-group decision and evidence](../r9/first_group_v1/README.md)。每种子的代码与检查点身份如下；文档提交变化不等于数值训练代码变化，精确源码哈希以运行清单为准。

|Seed|Run code commit|Checkpoint SHA256|Elapsed (s)|
|---|---|---|---|
|20260922|ca403e61a1495924bbe8c1f13da3138fb5d91dc5|bf96ea0abf4f68ab77d24adf4e043ffc01472d85ff067e7d8b7b26a5e4fb8c44|12254.359|
|20260923|c46284ceecd6b544da791143c1df9b36a32efa18|09f728d3bb161c22d942d0d617895007a6a850df679d7e6ccb94aae796bd8092|10481.047|
|20260924|4c539761e85a74c69da2b66b022377ba187ffc5f|7c1fec8110ecf13f270d6f891af583cfa92b2563c9a3efa7cc4a18eef2e91684|10516.812|

|Method|Seed|Recorded elapsed (s)|
|---|---|---|
|transformer|20260922|12254.359|
|transformer|20260923|10481.047|
|transformer|20260924|10516.812|
|rf|fixed|5807.484|
|xgb|fixed|1686.562|
|name_knn|fixed|600.109|

复验、独立审计与汇总命令如下。种子22复用首组已审计模型；每个新增种子在统计前完成独立审计及固定参照比较，运行清单保留配置与源码哈希。下面是既有运行的复现记录，已有输出目录拒绝覆盖，不应再次启动同名作业。

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_confirmation.py --plan experiments/foodnutrigpt_v9_research/r9/first_group_v1/confirmation_plan.json --seed 20260923 --output-dir output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260923
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_r9_completed_utf8.py --run output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260923 --output-dir reports/v9_r9_confirmation_seed20260923_audit_v1
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_confirmation.py --plan experiments/foodnutrigpt_v9_research/r9/first_group_v1/confirmation_plan.json --seed 20260924 --output-dir output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260924
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_r9_completed_utf8.py --run output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260924 --output-dir reports/v9_r9_confirmation_seed20260924_audit_v1
.venv/Scripts/python.exe scripts/confirm_foodnutrigpt_r9_fixed_references.py --output-dir reports/v9_r9_three_seed_confirmation_v1
.venv/Scripts/python.exe scripts/plot_foodnutrigpt_r9_confirmation.py --output-dir reports/v9_r9_three_seed_curves_v1
```

实测时长包括各运行内评价与检查，不是隔离的纯训练速度比较；先前并发负载不同。筛选1e-4额外10,551.985秒，首组全量训练拟合诊断110.75秒；复验表已包含复用种子22，不能再把它作为新增训练成本重复相加。历轮成本和失败保留于对应版本README及机器清单。

独立审计已核对每个完整60轮的顺序、曝光与学习率，精确重放两份各323,809条预测、49,913个候选向量及19,089条检索排名。原审计曾因Windows CP936读取UTF-8名称失败，依赖队列停止；独立修正版仅指定UTF-8读取，原失败记录保留，未用重训掩盖故障。历史R4数值失败、KNN批次依赖修复和R8部分/未运行树均保留，不作为成功配置计算。

版本决定：本固定 MAE 配方未通过预登记的 RF 复验条件，保留为已验证的训练控制，不接受为优于 RF 的最终方法。后续将登记仅改变训练损失的 MSE 对照；本报告不包含该分支的结果。来源留出和低标签迁移仍是后续研究问题，测试集继续关闭。

所有结果受重复使用验证集、稀疏轴、有限种子、冻结单次树、单位/别名/转载溯源未闭环的限制。R0的标签有效性限制依然成立。模型比较结论和数据真值核验是不同证据层次。机器可读引用与文档哈希见[evidence.json](evidence.json)。
