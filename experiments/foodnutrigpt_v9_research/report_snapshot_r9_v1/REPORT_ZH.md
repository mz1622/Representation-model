# 营养表征模型研究报告：R9 阶段稿

**状态：研究进行中，非最终交付。** 本版纳入首个完成独立核验的 R9 Transformer、冻结 RF/XGBoost、名称近邻和 R0–R8 研究路径。第二个学习率候选、完整预算对照、训练拟合诊断及 Transformer 多种子确认尚未完成；不把中间轮次分数放入最终结果表。对应英文为 [REPORT_EN.md](REPORT_EN.md)。

## 1. 当前问题与阶段结论

当前目标是在数据不变、RF/XGBoost 结果冻结的条件下，通过有对照依据的 Transformer 方法改善超过既有 RF。先前“不要求超过树”的报告终点及 MLP 结果保留为历史，不替代当前目标。仍要求交付训练合理性、每次迭代及归因、参数选择和三个任务的中英文报告。

首个完整 Transformer 的补全主误差为 **0.194543**，固定 RF32 为 **0.189031**。相对改善点估计为 −2.916%，食品组 95% 区间 [−7.503%, +2.033%]；目前不支持优于 RF。其 name-only 优于树，但弱于名称近邻；检索的绝对表现仍低。没有最终获选的 Transformer，也没有已证实的通用 nutrition foundation model。

接受一个固定 Transformer 配方要求种子 20260922、20260923、20260924 的平均主误差低于固定 RF，食品组配对区间支持改善，且旧营养 log-MAE 的相对退步不超过 2%。不要求超过 XGBoost。旧 5% 最低改善要求没有沿用；指标、标签、任务和固定参照没有因此改动。

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
|`learning_rates`|0.0001, 0.0003|
|`weight_decay`|0.0001|
|`gradient_clip`|1.0|
|`source_weight`|1.0|
|`source_residual_l2`|0.0001|
|`eta_min_fraction`|0.01|
|`seed`|20260922|
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

首组种子为20260922。若后续固定配方显示希望，再用20260922、20260923、20260924确认。食品组配对重采样区间不消除反复验证选点的偏差，也不包含固定RF自身的种子不确定性。接受目标是Transformer主误差均值低于固定RF且配对区间支持改善，旧营养log-MAE相对退步不超过2%；不是要求超过XGBoost。

### 参数依据与尚未回答的问题

|选择|依据|当前允许的解释|
|---|---|---|
|192维、3层、6头、FF768|沿用直接Transformer父实现，功能核验精确匹配|用于建立同输入控制；尚未证明容量最优|
|学习率0.0001/0.0003|预登记单因素比较|两组完成后才判断该结构下的优化差异|
|60轮及20轮快照|预登记预算诊断|当前尚无完整预算比较结果|
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

上述是已登记运行命令；存在的输出目录拒绝覆盖，不是要求再次启动。完整运行后另有独立评分重放、预算比较、学习率比较及全训练拟合诊断。功能验收已完成，但不能代替这些性能证据。完整研究仍需各轮八节报告、三个任务结果、多种子确认及最终中英文归因；本说明不提前宣布最终选择或foundation-model迁移能力。

证据索引：[配置](../r9/config.json)、[预登记](../r9/PLAN.md)、[冻结凭据](../../../reports/v9_r9_freeze_v1/manifest.json)、[功能核验](../../../reports/v9_r9_functional_v1/verification.json)、[数据汇总](../../../reports/v9_final_data_evidence_v1/summary.json)、[基线方法](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_ZH.md)。数值预测与检查点保存在本地忽略目录，本说明不包含原始食品营养值。

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
|[R9](../r9/README.md)|共同 PCA32、192维直接 Transformer，MAE、source1，学习率1e-4/3e-4；首项60轮完成，第二项待完整分析|首项0.194543，未超过RF；保留未改善结论，尚无最终配方|

历史 MLP/PCA128 配方三种子补全为 0.183435 ± 0.001222，name-only 为 0.484042 ± 0.010458；这些数值不是本轮 PCA32 Transformer 的结果。其名称清空/置换及常见轴正值低估诊断也不能直接转移到 Transformer。完整参数、逐种子结果及失败见[旧阶段稿](../final_report_v1/REPORT_ZH.md)与各轮记录。

R9 数值编码组件此前仅做过训练集可行性检查，没有投入本轮训练。当前 R9 首组只登记两个学习率，没有把可行性检查计作新模型性能证据。


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

R9同输入32维主要参照固定为RF leaf1/feature0.5及XGB depth10。它们是已完成集合中的参照，不代表完整搜索或所有随机种子的最优结果。本轮仅首项Transformer完整结果已产生，见第6节；基线附录本身不证明标签全部可靠或foundation能力。

逐轴训练时已检查反序、子批次和内存pickle重载，完成审计也核对了数据、特征、权重及保存的预测/检索排名。森林在预测后释放，没有保存完整森林，因此不能说当前产物支持任意新名称在线树推理。数字预测与候选矩阵保留在本地忽略目录；附录仅含汇总。

Data manifest SHA256: `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`

Freeze manifest SHA256: `1ec58002d76beb329004db77ef512622f57b40e142725799918d20027b5fddf3`

各运行代码、环境、完整参数、运行及拟合记录哈希见[机器结果](../../../reports/v9_r9_frozen_baseline_appendix_v2/summary.json)。两种语言所有数值表格一致；已完成附录不等于整体终稿。

## 6. 已完成结果：共同 PCA32 内部验证

下表全部使用同一冻结评价协议。误差越低越好；树和 Transformer 均只有筛选种子 20260922。KNN 为确定性参照，不虚构种子标准差。KNN 未作为本轮补全对照，“—”表示没有在此建立该比较。Transformer 三任务使用同一所选检查点，不能把不同专项模型的最好值合并。

|方法|补全142|旧log142|Name-only142|补全45|补全187|
|---|---|---|---|---|---|
|Transformer lr1e-4|0.194543|0.058632|0.414298|0.577952|0.286807|
|RF32|0.189031|0.056270|0.659518|0.674196|0.305782|
|XGB32|0.174487|0.052593|0.656148|0.674273|0.294756|
|KNN32|—|—|0.262260|—|—|

以下相对改善为百分比，正值表示 Transformer 误差更小；区间条件于已选单种子模型。

|Reference|Task|Metric|Gain (%)|95% interval (%)|
|---|---|---|---|---|
|rf32|completion|scaled_log_mae|-2.916|[-7.503, 2.033]|
|rf32|completion|log_mae|-4.198|[-9.931, 0.724]|
|rf32|name-only|scaled_log_mae|37.182|[34.601, 39.705]|
|rf32|name-only|log_mae|31.636|[27.825, 34.985]|
|xgb32|completion|scaled_log_mae|-11.495|[-16.915, -6.036]|
|xgb32|completion|log_mae|-11.483|[-19.664, -3.924]|
|xgb32|name-only|scaled_log_mae|36.859|[34.135, 39.436]|
|xgb32|name-only|log_mae|31.602|[27.791, 34.955]|
|knn32|name-only|scaled_log_mae|-57.972|[-63.536, -52.804]|
|knn32|name-only|log_mae|-64.083|[-73.198, -55.274]|

同一 Transformer 的完整子集指标：

|Task|Axes|Scaled MAE|Legacy log MAE|Raw MAE|Positive MAE|Zero MAE|
|---|---|---|---|---|---|---|
|completion|142|0.194543|0.058632|0.357728|0.324743|0.086138|
|completion|45|0.577952|0.032940|0.059523|0.808772|0.176678|
|completion|187|0.286807|0.052449|0.285968|0.441221|0.102746|
|name_only|142|0.414298|0.146179|0.901488|0.544090|0.406688|
|name_only|45|0.760225|0.056509|0.113176|0.922643|0.440394|
|name_only|187|0.497543|0.124600|0.711787|0.635186|0.412871|

检索数值为 0–1 比例，越高越好；候选库固定 49,913 名称。不同可见率使用不同合格查询群，不能作为可见率的纯因果对照。

|Method|Visible fraction|MRR|R@1|R@5|R@10|
|---|---|---|---|---|---|
|Transformer lr1e-4|0.3|0.003381|0.000336|0.003141|0.005618|
|Transformer lr1e-4|1.0|0.006852|0.001622|0.006617|0.012212|
|RF32|0.3|0.001477|0.000310|0.001239|0.001987|
|RF32|1.0|0.002528|0.000176|0.001566|0.004457|
|XGB32|0.3|0.001559|0.000310|0.001471|0.001819|
|XGB32|1.0|0.003417|0.000423|0.002468|0.006275|
|KNN32|0.3|0.034512|0.005149|0.054796|0.096722|
|KNN32|1.0|0.068879|0.013005|0.117573|0.193572|

完整与30%检索分别覆盖7,090/6,458食品组。逐指标食品组配对区间全部保留于 evidence.json 所引用的八份比较记录。Name-only 虽优于树，仍比 KNN 主误差高57.972%；全可见 R@10 约为1.22%，KNN约19.36%，不能宣称名称能力已经良好。


## 7. 机制、归因与当前选择

首项选中第 60 轮，完整运行耗时 10,551.985 秒，约 2.93 小时，包含运行内评价与检查。前 20 轮保存快照选中第 18 轮；正式三任务预算比较仍在等待两组完整运行。不能把第 60 轮仍较好的现象直接写成“继续训练必然有效”，也不能把含来源校准的 187 轴在线 loss 与 142 轴验证指标直接相减判断过拟合。

首项相对 RF 的主误差净差为 +0.005512。固定主指标分母的分解中，正值额外低估贡献 +0.025472，额外高估 +0.006141，显式零收益 −0.026100。八种分区均在 1e-12 内重构主指标。72/142 轴点误差更低，不等于这些轴各自均已确认获胜。

7 个训练支持少于 100 食品组的轴贡献 +0.008669；100–999 组的 20 轴抵消 −0.004691，至少 1000 组的 115 轴贡献 +0.001534。Isomeric linolenic acids (18:3) 仅有 68 个训练组和 17 个验证组，轴误差差 +0.944753，食品组区间 [−0.184128, 1.997180]；全部 7 个稀疏轴区间均跨零。该结果使“差距主要是常见轴”的旧 MLP 解释不适用于此 Transformer，但仍不能证明稀疏轴采样、容量或 MAE 是原因。

**观察事实**是当前正零取舍和支持度差异。**受控证据**目前包括功能核验；学习率和预算的完整控制结果尚未齐备。**未排除的解释**包括优化不足、共享表示、稀疏监督波动、来源校准与泛化差异。先完成两个学习率及 source-free 全训练集拟合诊断，再登记一个方法改动；不会同时改损失、容量、采样或指标。

首项补全未达到接受条件，旧 log 点退步 4.198% 也超出保护条件；保留完整结果，但不接受为最终改进。只有筛选后固定配方再完成三种子，才能报告最终选择。未见来源、类别留出、少样本迁移和冻结表征探针尚无足够证据。

## 8. 失败记录、复现与未完成事项

第一项训练成功完成后，原独立审计因 Windows CP936 读取 UTF-8 的 candidate_names.json 失败，三个依赖队列随后停止。修正仅显式指定四处 UTF-8 读取；AST 核验确认训练及评分逻辑未改变。完整首项审计随后成功，重放两个各 323,809 条预测表、49,913 个候选向量和 19,089 条排名精确一致。没有重训首项，原失败记录保留，第二项按原配方开始。详见[编码修复记录](../r9/AUDIT_UTF8_NOTE.md)。这是审计 I/O 故障，不作为方法失败或性能改进。

第一项执行实现提交为 b4fd4c1，后续记录提交为 186ea75。检查点 SHA256 为 `b9ee82eda2edeec17c62ac6e9b67df5b2dded0b6bbf53ab976e4e1319527f2a0`；数据、缓存、面板、源码与结果的完整指纹在本目录 [evidence.json](evidence.json) 及所链接运行凭据中。环境为 Windows、Python 3.10.19、PyTorch 2.7.1+cu128、RTX5070Ti16GB。实际耗时不视为受控算法速度比较。

完整逐轴、来源和配对结果见[首项对 RF](../../../reports/v9_r9_tf192_mae_lr1e4_60_vs_rf32_completion_v1/summary.json)、[独立审计](../../../reports/v9_r9_tf192_mae_lr1e4_60_audit_v1/verification.json)和[固定分母诊断](../../../reports/v9_r9_tf192_lr1e4_rf32_gap_v1/summary.json)。来源切片的轴覆盖不同，不根据不同来源的平均值直接判定数据库质量。所有原始数值案例、预测、缓存和模型留本地忽略目录。

本版已纠正旧稿中追加树拟合、MLP 最终胜出和 R9 尚无训练的过时状态。最终报告仍须加入第二组及其后方法迭代、完整曲线与训练拟合、最终固定 Transformer 的三种子确认和最终选择理由。所有现有结果来自反复使用的内部验证；食品组区间不涵盖训练种子、选择偏差、固定树随机性或标签真实性。测试保持关闭，单位溯源、别名与转载问题仍未解决。完整双语稿的存在不表示研究目标已经完成。
