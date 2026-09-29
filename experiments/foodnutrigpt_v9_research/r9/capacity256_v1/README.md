# V9-R9 / PDF容量组合：筛选完成，拒绝此固定升级配方

完整[中文报告](../../report_snapshot_r9_capacity256_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_capacity256_v1/REPORT_EN.md)包含数据来源、历次迭代、冻结RF/XGBoost方法与全部结果。

## 1. 研究问题与预先假设

关闭来源校准的上一配方未通过筛选，因此仍以192维、source weight=1的MAE为控制。原PDF采用256维、8头、FF1024，本轮检验这个容量组合能否改善营养上下文表示。假设成立时无来源训练拟合和验证误差可能同时改善；更大模型也可能产生优化或泛化代价。PDF原始数据和评分不同，其历史分数不能直接比较。

## 2. 父版本与受控改动

第6/12个候选仅改变登记的容量组：d_model 192→256、n_heads 6→8、feedforward_dim 768→1024，每头维度仍为32。保持3层、dropout 0.15、rank16、MAE、来源校准权重1、来源L2=1e-4、按(1+w)归一化的损失、batch64、AdamW lr3e-4/weight decay1e-4、clip1和60轮余弦日程。数据、名称缓存和有效32个文本方向、家族遮蔽、任务顺序、评分及RF/XGB均冻结。没有引入MSE、轴×数值残差或two-stage。

## 3. 可复现信息

|Item|Value|
|---|---|
|Code commit|f19d545c33272906b334db3ed63351f69707892e|
|Checkpoint SHA256|5017bca4ea3a0fbdeac3de0241667003b3a447536d56edbd6f245ee3a97eb229|
|Capacity initial-state SHA256|6c0443083d8752e5fca82e5dfeb648c93bf03fb48d2ac6e0443243421f41c7bb|
|Parent initial-state SHA256|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Capacity / parent parameters|2696626 / 1548594|
|Capacity / parent requires_grad parameters|2648409 / 1515929|
|Seed|20260922|
|Selected epoch|59|
|Run elapsed (s)|13419.829|
|Training-fit inference elapsed (s)|74.890|
|Pre-run estimated elapsed (s)|15402.150|
|Measured 256 / 192 step-time ratio|1.331463|
|Registered first follow-up delay (min)|267|

独立审计重建本容量模型的初值，核对参数量及全部60轮任务顺序、目标暴露量和学习率，并精确重放两份各323,809条预测、49,913个名称候选向量和19,089条排名。环境为Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB，保持父实验CUDA后端。短时估时使用相同104个训练batch，8步预热与96步计时、每种容量两轮交替次序，诊断权重丢弃；不评价验证结果。正式启动后设置267分钟回访，实际耗时单独报告。

[Registration](PLAN.md) · [Configuration](config.json) · [Version](README.md) · [Analysis](../../../../reports/v9_r9_capacity256_analysis_v1/summary.json) · [Training fit](../../../../reports/v9_r9_capacity256_fit_v1/summary.json) · [Functional checks](../../../../reports/v9_r9_capacity256_functional_v1/verification.json) · [Runtime estimate](../../../../reports/v9_r9_capacity256_runtime_v1/summary.json)

运行中断记录：原进程和会话消失，最后完整保存为48轮，具体退出原因未知。原目录保持原样，在独立恢复目录从49轮继续至60轮；完整恢复模型、优化器、调度器及随机状态，未改方法。上表运行耗时仅为原已保存耗时与恢复段耗时之和，不包括未知的未保存工作或等待时间。

|Recovery item|Value|
|---|---|
|Original run|output/v9_r9_methods/tf256_mae_lr3e4_60|
|Recovery code commit|3aa6c8bca7e490e9a93f8dd99120866b3d648352|
|Original saved elapsed (s)|10908.641|
|Resumed segment elapsed (s)|2511.188|
|Resume started (UTC)|2026-09-29T07:06:16.427620+00:00|
|Resume finished (UTC)|2026-09-29T07:48:07.518157+00:00|

[Recovery registration](recovery1/PLAN.md) · [Recovery configuration](recovery1/config.json)

恢复后完整60轮、原48轮历史、最优检查点、两份完整预测和检索结果的独立审计均通过，11项比较全部完成。恢复训练实际耗时2511.188秒，原保存耗时10908.641秒，合计13419.829秒；这不是包含未知丢失工作和停机时间的完整墙钟成本。训练拟合另耗时74.890秒，没有优化器更新。训练恢复前16项合成检查通过，真实状态重载与训练行前向核验通过。案例读取有pandas混合类型提示，但逐值/元数据验证成功。原中断原因仍不明确，不能省略该失败记录。

以下命令记录已完成运行，路径均相对于仓库根目录；已有产物目录拒绝覆盖。恢复前16项合成测试和32项后处理分析测试均已通过。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_capacity256.py --candidate tf256_mae_lr3e4_60 --output-dir output/v9_r9_methods/tf256_mae_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/resume_foodnutrigpt_v9_r9_capacity256.py --candidate tf256_mae_lr3e4_60 --output-dir output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_capacity256_recovery.py --run output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1 --output-dir reports/v9_r9_capacity256_lr3e4_60_audit_v1
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_capacity256_recovery.py --output-dir reports/v9_r9_capacity256_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_capacity256_recovery_fit.py --output-dir reports/v9_r9_capacity256_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_capacity256_fit_v1
```

[Recovery execution record](recovery1/README.md)

## 4. 完整结果

MAE和CAPACITY256均为种子20260922；RF/XGB/KNN沿用冻结结果。每个神经模型三任务使用同一个按补全主指标选择的检查点。综合报告第6节的历史MAE三种子汇总与这里单种子容量对照不同。

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|CAPACITY256|0.201520|0.062202|0.378797|0.328890|0.094971|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|CAPACITY256|0.432889|0.147108|0.853721|0.543452|0.446587|
|RF32|0.659518|0.213824|1.134940|0.695565|0.730742|
|XGB32|0.656148|0.213719|1.173381|0.693838|0.718390|
|KNN32|0.262260|0.089088|0.502782|0.338107|0.246578|

**45/187 axes**

|Method|Task|Subset|Axes|Scaled-log MAE|
|---|---|---|---|---|
|MAE|completion|food_metabolome|45|0.598022|
|MAE|completion|all|187|0.283292|
|MAE|name_only|food_metabolome|45|0.749716|
|MAE|name_only|all|187|0.504212|
|CAPACITY256|completion|food_metabolome|45|0.636716|
|CAPACITY256|completion|all|187|0.306247|
|CAPACITY256|name_only|food_metabolome|45|0.836458|
|CAPACITY256|name_only|all|187|0.530005|
|RF32|completion|food_metabolome|45|0.674196|
|RF32|completion|all|187|0.305782|
|RF32|name_only|food_metabolome|45|0.889240|
|RF32|name_only|all|187|0.714798|
|XGB32|completion|food_metabolome|45|0.674273|
|XGB32|completion|all|187|0.294756|
|XGB32|name_only|food_metabolome|45|0.851545|
|XGB32|name_only|all|187|0.703168|
|KNN32|name_only|food_metabolome|45|0.702450|
|KNN32|name_only|all|187|0.368188|

**检索 / Retrieval**

|Method|Visible fraction|Recall@1|Recall@5|Recall@10|MRR|
|---|---|---|---|---|---|
|MAE|0.3|0.000465|0.003452|0.006781|0.003381|
|MAE|1.0|0.001023|0.006003|0.010603|0.005720|
|CAPACITY256|0.3|0.000697|0.002568|0.004516|0.003039|
|CAPACITY256|1.0|0.001034|0.004099|0.008510|0.005148|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

检索固定49,913个名称候选，候选向量仅由名称预测，不能使用候选真实营养档案；营养查询不含名称。正确答案按原始名称精确匹配，没有已确认的别名映射。食品组条件区间不覆盖种子总体、标签有效性和重复选型的不确定性。

|Reference|Task|Metric|CAPACITY256 gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-9.788383|-12.942591|-6.495415|
|mae_parent|completion|log_mae|-14.421456|-19.747409|-9.709883|
|mae_parent|name_only|scaled_log_mae|-1.519171|-4.296305|1.363567|
|mae_parent|name_only|log_mae|4.593294|-0.030119|9.076375|
|rf32|completion|scaled_log_mae|-6.606981|-9.954789|-3.281495|
|rf32|completion|log_mae|-10.542188|-15.754437|-5.945467|
|rf32|name_only|scaled_log_mae|34.362755|32.117923|36.719105|
|rf32|name_only|log_mae|31.201522|27.159454|34.529478|
|xgb32|completion|scaled_log_mae|-15.493081|-18.876326|-11.876650|
|xgb32|completion|log_mae|-18.271110|-25.113878|-11.043024|
|xgb32|name_only|scaled_log_mae|34.025623|31.717179|36.433260|
|xgb32|name_only|log_mae|31.167807|27.217757|34.427602|
|knn32|name_only|scaled_log_mae|-65.061135|-70.554154|-59.778766|
|knn32|name_only|log_mae|-65.126113|-76.210468|-54.934067|

|Reference|Visible fraction|Metric|CAPACITY256 minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|-0.000343|-0.001363|0.000642|
|mae_parent|0.3|recall_at_1|0.000232|-0.000542|0.001084|
|mae_parent|0.3|recall_at_5|-0.000884|-0.002794|0.000839|
|mae_parent|0.3|recall_at_10|-0.002265|-0.004607|0.000226|
|mae_parent|1.0|mrr|-0.000572|-0.001915|0.000602|
|mae_parent|1.0|recall_at_1|0.000012|-0.000988|0.001058|
|mae_parent|1.0|recall_at_5|-0.001904|-0.004173|0.000148|
|mae_parent|1.0|recall_at_10|-0.002094|-0.005216|0.000662|
|rf32|0.3|mrr|0.001562|0.000693|0.002475|
|rf32|0.3|recall_at_1|0.000387|-0.000310|0.001161|
|rf32|0.3|recall_at_5|0.001329|-0.000078|0.002659|
|rf32|0.3|recall_at_10|0.002529|0.000838|0.004337|
|rf32|1.0|mrr|0.002620|0.001681|0.003617|
|rf32|1.0|recall_at_1|0.000858|0.000164|0.001681|
|rf32|1.0|recall_at_5|0.002533|0.000802|0.004194|
|rf32|1.0|recall_at_10|0.004052|0.001527|0.006534|
|xgb32|0.3|mrr|0.001480|0.000663|0.002367|
|xgb32|0.3|recall_at_1|0.000387|-0.000310|0.001161|
|xgb32|0.3|recall_at_5|0.001097|-0.000426|0.002518|
|xgb32|0.3|recall_at_10|0.002697|0.000813|0.004646|
|xgb32|1.0|mrr|0.001731|0.000649|0.002888|
|xgb32|1.0|recall_at_1|0.000611|-0.000165|0.001528|
|xgb32|1.0|recall_at_5|0.001630|-0.000156|0.003499|
|xgb32|1.0|recall_at_10|0.002235|-0.000546|0.005069|
|knn32|0.3|mrr|-0.031474|-0.034027|-0.029179|
|knn32|0.3|recall_at_1|-0.004452|-0.006323|-0.002800|
|knn32|0.3|recall_at_5|-0.052228|-0.057598|-0.047133|
|knn32|0.3|recall_at_10|-0.092206|-0.098889|-0.085462|
|knn32|1.0|mrr|-0.063731|-0.067261|-0.060598|
|knn32|1.0|recall_at_1|-0.011971|-0.014633|-0.009506|
|knn32|1.0|recall_at_5|-0.113474|-0.121155|-0.106535|
|knn32|1.0|recall_at_10|-0.185062|-0.194554|-0.176731|

## 5. 机制诊断

|Fixed-denominator component|CAPACITY256 minus MAE|
|---|---|
|positive_under|0.016194|
|positive_over|0.002098|
|explicit_zero|-0.000326|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|CAPACITY256|train|0.155628|0.273913|0.065631|
|CAPACITY256|validation|0.201520|0.328890|0.094971|

|Method|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|CAPACITY256|0.409132|0.366907|59|13419.829|

![Capacity 192/6/768 versus 256/8/1024 learning curves](../../../../reports/v9_r9_capacity256_analysis_v1/learning_curves.png)

共同分母分解精确重构主误差增加0.017966898：正值低估+0.016194201、正值高估+0.002098269、显式零−0.000325572。主要代价来自正值低估。条件零值宏平均反而从0.092039升至0.094971；它与共同分母零贡献的方向不同，因为轴内分母和有效权重不同。不能把条件统计和主误差贡献混为一谈。对全部1,828,536个训练目标的无来源推理主误差由0.134937升至0.155628（差+0.020690469），条件正值与零值误差也都升高，未显示更大模型改善拟合。

已实际查看六面板曲线。容量模型选第59轮，192控制选第60轮；容量模型的主误差、旧log和正值误差在后期仍更高，曲线趋缓但不证明全局收敛。末10轮裁剪比例为0.366907，对照为0.350883；平均梯度范数有波动，没有可见发散。较高裁剪率是描述性现象，参数量也不同，不能单独证明裁剪导致退步。第48轮中断及49–60轮恢复已单独审计，不把曲线平滑当作未中断轨迹等价的证明。

13/142轴点改善，只有2个未校正逐轴区间支持改善，96个支持退步；9轴验证支持不足30个食品组。两个有改善区间的轴为Tocotrienol, gamma和Vitamin D2；20个氨基酸轴全部点退步。115个至少1000训练组的轴贡献+0.013257662，20个100–999组轴贡献+0.001768189，7个少于100组轴贡献+0.002941046，因此代价并非只集中于稀疏轴。脂肪酸、维生素、膳食纤维家族分别贡献+0.005717328、+0.004269019、+0.003190012。24个来源分区均点退步，FooDB贡献+0.006363348。最大单轴代价Lignin仅有17个验证组，区间宽；来源、家族与支持度分区不能相加，也不能据此认定唯一生物机制。

已从冻结任务和保存预测重建并核对40个RF对比极端案例。20个退步案例均为FooDB正值，来自18个食品组，含15个Cholesterol和5个Biotin案例；其中15个Cholesterol预测均至少比保留标签低500倍。这既不确认单位错误，也不证明标签正确。20个改善案例含7个正值、13个零值，覆盖5个来源和18个食品组。尾部案例不是代表性独立样本，不据此更改数据；逐条名称、浓度和预测保留在本地。

仅名称主误差0.432889，较同种子MAE点退步1.519%，改善区间[−4.296%,+1.364%]跨零；其旧log-MAE点改善4.593%，相应区间仍跨零。相对冻结RF/XGB，仅名称预测较好，但明显弱于名称KNN的0.262260。全可见/30%可见检索R@10为0.008510/0.004516。两种可见度下R@1略有点改善，MRR/R@5/R@10均点退步，8个差值区间全部跨零；不能宣称检索升级。45轴和187轴补全也点退步。全部三任务使用同一个第59轮检查点。

[Visual review](../../../../reports/v9_r9_capacity256_analysis_v1/visual_review.json) · [Axis contrasts](../../../../reports/v9_r9_capacity256_axis_changes_v1/summary.json) · [Case review](../../../../reports/v9_r9_capacity256_cases_v1/summary.json)

|Functional exercise|Value|
|---|---|
|Training tasks|32|
|Steps|50|
|Loss before|0.156038|
|Loss after|0.057187|

训练前12项共享/合成功能测试通过。真实训练行核验复现192控制初值，并验证256构造器与显式替换配置的独立构造在权重、同宽构造RNG、前向及损失上精确匹配。隐藏标签和来源不影响基础前向，未观测轴可查询，8个训练名称×187轴输出有限。50步后来源残差已学习为非零，仍不进入基础推理，初始和已学模型均可精确重载。上述只证明实现和可学习性，不作为性能证据。

## 6. 因果分析边界

已观察事实是：在这组固定训练设置下，容量组合增加后训练与验证主误差都更高，退步覆盖大多数营养轴和所有来源分区。对照支持拒绝直接替换为此256/8/FF1024配方，不能推出更大Transformer普遍无益。宽度、头数、前馈规模、参数量、初值形状和随机数消耗共同改变，相同种子不是相同初始化。训练拟合也变差，与单纯“训练更好、验证更差”的过拟合叙述不一致；优化设置与容量的相互作用仍是未验证假设。中断前48轮已保存，恢复状态和数值循环通过核验，但未中断GPU反事实不可获得。单种子食品组区间不覆盖种子总体、重复选型、标签有效性或外部迁移。

## 7. 筛选与版本决定

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

未满足登记的筛选条件；单种子不能确认为稳定改进或优于RF。

版本决定：拒绝此固定容量配方作为补全升级，不追加20260923/20260924复验，保留原48轮、中断记录、恢复运行及全部结果。继续以192维MAE、source weight=1为方法控制；保留控制不等于证明其参数最优。整体RF目标尚未达到。[机器决定](decision.json)记录筛选依据。

## 8. 下一轮问题与测试状态

下一轮回到现有192维控制，优先检验初始学习率3e-4→6e-4的优化假设，其余结构、损失、来源权重、采样和60轮日程保持不变。此前学习率筛选及本轮训练拟合退步说明应继续验证优化设置，不能仅靠扩容；这不预设更大学习率会获胜。该具体候选尚未登记或启动，必须另行登记、核验并先估时再回访。数据和RF/XGB继续冻结，测试关闭，来源留出与少样本迁移能力仍未验证。