# 营养表征模型研究报告

**本文件保留为此前MLP阶段草稿。** [R9 Transformer实验](../r9/PLAN.md)正在执行，数据不变、树不再训练；旧草稿中的待跑树计划已取消。最新[交付要求](../r9/EVIDENCE_AND_REPORTING.md)是验证训练方法和参数并完成中英文报告，不要求超过RF/XGBoost。终稿须纳入新增Transformer证据，当前不能作为完成报告。

**报告状态：草稿，训练与比较收尾中（2026-09-28）。** 数据、方法、R0–R7及神经网络三种子结果已写入；R8最后两项RF、选定树模型的三种子结果、最终配对区间及正式版本决定尚未完成。本文件不是最终交付。英文对应稿为[REPORT_EN.md](REPORT_EN.md)。

## 1. 研究问题与结论范围

本项目研究三个任务：名称加部分营养的缺失值补全、只凭原始food name预测营养、营养到训练未见名称的候选排序。主要评价对象是142个nutrition轴，45个metabolome轴单列，保留187轴结果；65个context-only轴可以提供已知上下文，但不宣称可靠预测。

用户于2026-09-28调整终点：**不再要求超过RF/XGBoost，要求训练合理、方法和参数有验证过程、比较公平、报告完整**。目标修订发生于R8部分结果已知之后，见[计划及时间边界](PLAN.md)，不能称为最初预登记。旧5%主改善和2%旧log退步门槛作为历史记录保留，不再用于阻止复核或交付。

当前保留的补全神经网络是冻结名称编码器加两层MLP，不是最终胜出的Transformer，也不是已证实的foundation model。三种子补全主误差为**0.183435 ± 0.001222**，其训练和输出可重放。仅名称主误差为0.484042 ± 0.010458，仍弱于同输入名称近邻。最终RF/XGBoost比较在第7节待补齐，不能提前宣布胜负。

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

## 3. 模型与训练方法

### 3.1 从原Transformer到当前补全MLP

原V8使用来源token；V9研究分支使用source-free Transformer和仅训练使用的来源×轴残差。早期hurdle头同时学习正值存在概率及正值含量。R0–R2检查了共同输入/任务、来源校准、amount权重、hurdle与直接回归及选点规则；保留全部结果，没有把历史协议的V8/RF数值直接拿来判定架构优劣。

共同研究协议下，V9输入为CLS、投影后的名称及252个固定轴槽位；轴槽位结合轴嵌入、数值编码或隐藏标记。Transformer为3层、隐藏192维、6个注意力头、前馈768维、dropout .15、GELU和pre-norm；逐轴预测头含rank 16残差。名称投影与标量值编码均为两层网络。来源×轴残差在训练来源间逐轴去中心化，验证只用不含来源的主预测。R1来源损失统一为`(L_base + w_source L_calibrated)/(1+w_source) + 1e-4 L_residual`，控制移除来源项时的整体损失尺度。

R1代表性V9训练为batch 64、AdamW学习率.0001、20轮及固定20轮日程；amount权重1/2/3、来源校准开关分别实验。R2直接数值头分支在同20轮预算下比较SmoothL1与MAE，补全主误差.225345→.213960；正值条件误差却由.318757变为.354390，零值误差由.152736降至.087285。它说明总误差改善伴随正零取舍，不支持“所有营养预测均改善”。这些历史运行使用当时的32维名称缓存；与最终PCA128 MLP之间还改变了输入、优化和训练预算，不能视为纯架构消融。

最终固定代表采用更简单的MLP，因为已有受控实验支持其损失、容量和训练预算选择，当前内部补全表现也较好。它的512维隐藏向量可以导出供后续研究，但冻结探针、少样本迁移和未见来源泛化尚未完成。因此“表征研究模型”是恰当定位，“通用营养基础模型已建立”没有证据。

### 3.2 名称、数值及可见性输入

名称分支只用original_name，没有来源、分组、加工或科学名称等额外字段。冻结[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)修订`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，按attention-mask均值池化、L2归一得到384维向量。本地token上限128；既有名称审计最长54，没有因该上限截断名称。

R7/R8仅在64700训练档案上拟合float64精确公共PCA，不白化。前32方向保留54.533%方差，128方向保留87.846%。最终用128方向；32维对照仍占128槽位，后96列为0，使两臂结构/参数数目一致。PCA方差不等于营养预测效用；增加维度对不同算法有不同表现。

最终输入632维：128名称＋252缩放营养值＋252可见性。隐藏整个目标家族，含所有关联形式；隐藏标签及其真实存在性不决定模型收到的查询轴。模型输出网格固定，由调用者指定需要返回的轴。

### 3.3 结构、损失、优化与选点

结构为`Linear(632,512) → GELU → LayerNorm(512) → Linear(512,512) → GELU → Linear(512,252)`，717052个可训练参数，不计冻结文本编码器。没有额外name-only头、来源残差、存在概率乘法或新增对比损失。

每轮遍历337048个档案×家族任务、19个家族，每个已观测187轴目标恰好监督一次。对轴a的训练权重按食品候选组内来源及同来源档案归一，完整数据轴权重和为`Z_a`。总体目标为：

`L=(1/187) Σ_a [Σ_i w_ia |f_a(x_i)-t_a(y_ia)| / Z_a]`。

小批量采用全训练集归一常数和任务数量校正，不在批内重新定义轴权重。45个metabolome轴系数为1；主要模型选择指标仍只含142营养轴。MAE改善了零值误差但可能牺牲正值，这个取舍明确保留。训练回归头不乘存在概率，推理将负变换值截至0后反变换；非有限输出使实验失败，不悄悄跳过。

AdamW，lr=.001、weight_decay=.0001、batch256、gradient clip=1，60epoch；CosineAnnealingLR的T_max=60、eta_min=.00001。每轮完整固定验证面板选取142轴主误差严格最小的检查点，同分最早。没有增加name-only样本比例或额外上下文dropout。选中轮次和全部曲线保存，未证明60轮或该学习率全局最优。

### 3.4 三任务推理

`predict(food_name, observed_profile, target_axes)`支持指定目标，数值上下文为空时即name-only；`encode(...)`导出明确模态表征；`retrieve_names(...)`给出排序而非概率。检索候选向量是同一模型仅由候选名称预测的142轴营养，查询侧不输入名称，候选侧不读取真实营养。固定49913名称；完整与30%保留面板分别10479/8610档案，仅评价至少3个营养观测的查询。两个面板覆盖不同，不能直接把分数差归因于遮蔽率。

## 4. 逐轮研究路径与归因

历史完整配置、失败、逐轴/来源及三任务均保留于每轮报告。下表不是把跨版本最好分数串成一个因果曲线；R0→R1训练协议和R7/R8名称输入变化必须分别解释。

|版本|问题、干预与关键证据|解释边界与决定|
|---|---|---|
|[R0](../r0/README.md)|统一原始空间聚合、训练尺度、name-only文本、查询/遮蔽；12配置基准与异常隔离|建立可比评价；标签溯源未闭环，训练遮蔽当时仍不同，不能纯归因架构|
|[R1](../r1/README.md)|全量共同家族任务及全局轴权重；固定日程8→20轮，MLP主误差.246757→.222459；V9 amount1→2→3与source开关|时长对照支持增加预算；amount提高改善该V9分支；去来源未获明确补全收益。面板与损失修复共同变化，不作单因素归因|
|[R2](../r2/README.md)|12预算：SmoothL1→MAE .222459→.209484；256→512 .209484→.202428；同60轮日程内20→60 .209506→.184491；另试1024、LayerNorm、直接V9、选点与名称标准化|保留MAE/512/60及LayerNorm。MAE主要改善零值；加宽1024补全区间跨零且name-only变差；hurdle→direct改动多项，不能只归因概率乘法|
|[R3](../r3/README.md)|4配置：name-only比例10%/20%、独立任务头及相应诊断|部分名称收益伴随补全代价，未采用到最终补全模型；未把所有取舍都称为梯度冲突|
|[R4](../r4/README.md)|8预算，5完整含复用、3数值失败；查询残差、视图一致性、代谢物权重.5；修正共同逆变换/查询契约|无足够补全收益；失败保留，不用NaN跳过或宽容差包装成功；组件修复和性能变化分开|
|[R5](../r5/README.md)|12配置：名称预测、营养—名称对齐与部分视图；完整输入独立检索三种子R@10 .298199±.003720，相对当时旧KNN .192192|仅完整输入专项模型优势，30% R@10仅.002252；不是补全模型，也未证明相对后来PCA128近邻优势|
|[R6](../r6/README.md)|固定30%及混合30/60/90额外上下文删除；补全.197232/.210462，父.184491|固定评价面板上均退步，不采用；名称改善不能替代补全结果|
|[R7](../r7/README.md)|6配置；公共PCA32→128，名称MLP/KNN主误差改善8.84%/6.87%|保留128输入方向；MLP仍比同128近邻差5.11%，不能与旧输入树直接比较|
|[R8](../r8/README.md)|12配置的相同632维输入比较；MLP32→128补全改善.931%，区间跨零，name-only改善18.909%；XGB固定深度增加名称维度使补全退步4.048%|完整搜索仍收尾；两树类别各3个128配置，保留32控制；R8完成后形成最终选择记录|
|[固定配置三种子](NEURAL_REPLICATION.md)|不新增超参数，MLP seeds22/23/24全部完成且全量重放通过；主指标0.183435 ± 0.001222|支持限定条件下重复性；不是新测试泛化证据，树重复和最终比较待完成|

固定名称干预表明网络实际使用名称；清空或交换名称损害补全。正值/零值分解显示多数补全差距来自常见轴的正值低估，零值收益部分抵消；这不支持仅以稀疏轴欠拟合解释总体差距。分解是描述性证据，不能证明MAE、结构或尺度中某单项是唯一原因。下一轮数值编码只做了训练集可行性与组件核验，未训练R9，不计性能改进。

## 5. 最终参数为什么这样选

|设置|验证过程|当前选择及尚未证明的内容|
|---|---|---|
|共同家族任务/权重|R1全量特征、行、目标、权重对齐；每目标每轮一次|保留，避免训练问题不一致；不等于优化器/目标函数完全相同|
|回归损失|R2同配置SmoothL1→MAE、XGB损失对照|神经用MAE，树仍平方误差；保留正零取舍，不说MAE普遍最好|
|隐藏宽度|256→512有效；512→1024收益不确定且名称任务退步|512；并非穷举容量|
|训练预算|固定60轮日程的20/60窗口比较、三个种子曲线|60轮，主指标选点；更长预算未进一步验证|
|LayerNorm|匹配初始化移除后补全退步3.69%，区间支持|保留；其name-only代价明确记录|
|名称维度|R7共同基底32/128；R8相同槽位/参数数目|128；不宣称补全维度收益已获区间支持|
|任务混合/独立头|R3与R6有独立对照|最终0%额外name-only、共享头、无额外dropout；不是同时最优三任务|
|代谢物权重/复杂读出|R4系数.5、一致性与查询残差|系数1，保留简单头；未支持复杂分支收益|
|lr、batch、weight_decay、clip、GELU|固定于受控实验，轨迹及梯度日志检查，三种子稳定|不是每项都做过独立调参；合理工程设置不等于最优超参数|
|随机种子|预定22/23/24，独立运行全部保留|报告均值、SD和范围，不挑最佳种子|

## 6. RF、XGBoost和名称参照的具体方法

基线逐轴训练187个独立回归器，使用该轴全部合格训练档案，没有5000行上限，最大轴58958行。名称/数值/可见标记与MLP相同；目标家族全部隐藏；目标和变换相同。每轴来源权重按均值归一后传入sample_weight，不自动跨来源合并标签。树以缩放log值回归，预测后同样非负反变换。

[Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)采用scikit-learn1.5.2，400棵bootstrap树、无深度上限、平方误差；候选叶子大小/特征比例如下。其余库默认参数以每轴get_params凭据为准。拟合4线程，预测按固定树顺序串行累加，避免并行归约产生几个ULP的批次/顺序差。

[XGBoost](https://arxiv.org/abs/1603.02754)采用2.1.3，800棵树、hist、learning_rate=.03、min_child_weight=5、subsample=.8、colsample_bytree=.8、reg_lambda=1、平方误差；max_depth在6/10/14中筛选。每轴random_state=运行seed+axis_index；拟合4线程。

|方法 / Method|R8 配置 / Configuration|名称维度 / Name dimensions|训练目标 / Objective|
|---|---|---:|---|
|RF control|400 trees, leaf=1, max_features=0.5|32|Squared error|
|RF A|400 trees, leaf=1, max_features=0.5|128|Squared error|
|RF B|400 trees, leaf=3, max_features=0.5|128|Squared error|
|RF C|400 trees, leaf=1, max_features=1.0|128|Squared error|
|XGB control|800 trees, max_depth=10|32|reg:squarederror|
|XGB A|800 trees, max_depth=6|128|reg:squarederror|
|XGB B|800 trees, max_depth=10|128|reg:squarederror|
|XGB C|800 trees, max_depth=14|128|reg:squarederror|

128输入下每个方法固定3个搜索配置，另1个32控制；这是一份明确预算的调参，不是穷尽优化。按固定142补全验证主指标选择各类别配置，其他任务不参与选优；随后冻结参数跑其余种子。最终三种子比较不会用旧随机PCA32树替代新输入树。

同一套拟合树生成补全、全数值隐藏的name-only、候选名称的预测营养和检索。每轴拟合后进行反序/子批次/内存pickle重载检查，再释放森林以控制存储；保存预测矩阵和凭据。不能声称被丢弃的森林已经提供任意新名称在线推理。名称参照另有全训练数据KNN，K=10，KDTree、来源与逆距离权重；其数据输入严格仅名称。

## 7. 当前已完成结果与最终比较待办

三个固定神经种子如下，全部来自同一补全训练配置。误差越低越好。

|Seed|Selected epoch|Completion142 MAE|Legacy log-MAE|Name-only142 MAE|
|---|---:|---:|---:|---:|
|20260922|60|0.184290|0.059214|0.493993|
|20260923|59|0.183980|0.059999|0.484992|
|20260924|57|0.182035|0.058565|0.473141|

补全旧log-MAE 0.059259 ± 0.000718，原始单位MAE 0.368583 ± 0.004683；正值/零值误差分别0.279095 ± 0.001755和0.111908 ± 0.001335。45轴主误差0.578353 ± 0.014242，187轴0.278469 ± 0.002567。

同一模型的完整检索R@10 0.003182 ± 0.000271，30%输入0.002232 ± 0.000454，均为0–1比例。name-only和检索表现仍是局限；不能把R5专项模型的较好排序结果写进本MLP能力行。

![三种子训练与验证曲线](../../../reports/v9_final_neural_summary_v1/learning_curves.png)

与同PCA128名称近邻的食品组比较已完成：MLP的name-only主误差相对增加98.186%，95%条件区间为[92.594%, 104.248%]；完整检索R@10差值为−0.246558，区间[−0.256954, −0.236429]。这支持把名称任务与检索列为当前补全模型的明确短板。KNN是单个确定性参照，没有虚构三种子波动。全部任务、逐轴/来源及失败记录见[阶段报告](NEURAL_REFERENCE_COMPARISON.md)，统计细节见[口径与实现](STATISTICS.md)。

**最终共同协议比较表：待完成。** 必须加入选定RF/XGB各三种子、两个任务完整均值/SD/范围、检索MRR/R@1/5/10、逐轴支持与来源、食品组配对区间及成本。目前不给尚未完成的树重复填入单种子代替值。

食品组区间将先平均三个独立模型的误差，再整体重采样食品组1000次；不平均预测形成集成。种子SD与食品组区间回答不同问题，均不能消除反复调参的验证选择偏差。所有结果仍为内部验证，测试未打开。

## 8. 最终选择、局限与复现

当前固定MLP512/PCA128代表是可复现、训练稳定且有多轮取舍依据的实用配置；并非因已证明优于两类树才保留。最终树选优及版本决定尚未闭合，因此本草稿不提供最终模型排名。

未解决事项包括原始标签单位/定义溯源、语义别名与转载泄漏、新来源/食物类别与少样本迁移、名称任务较差、查询稀疏条件、参数搜索有限、只有三个种子以及验证集重复使用。不能据内部成绩宣称广泛生物学或健康应用能力。

数据清单SHA256：`48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`；训练执行提交`6ff6f36d4f32c9c2dbf12eac6d32d0f3c21d6df4`；128缓存`4132b3623da37df2f02aedc6dde7d06b5c1e87a6b65e71d87f58ebbed5d2f768`；128任务面板`384b66c10142aefa801a5d0fbed2772b2e128150c9bb1575e2c2fe9bc513fbf8`。环境Windows、Python3.10.19、PyTorch2.7.1+cu128、NumPy2.1.3、RTX5070Ti16GB。

数据统计与来源证据为`reports/v9_final_data_evidence_v1/summary.json`；神经核验及完整结果为`reports/v9_final_neural_audit_v1/`与`reports/v9_final_neural_summary_v1/`。命令见[三种子记录](NEURAL_REPLICATION.md)、[R8执行记录](../r8/README.md)及各轮PLAN。最终统计入口、全部树指纹、成本和双语表格一致性检查将在成稿时补入；原始值、预测、模型和缓存留本地数据/输出目录，不纳入公开实验正文。

