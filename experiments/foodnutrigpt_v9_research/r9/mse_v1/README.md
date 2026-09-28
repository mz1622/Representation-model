# V9-R9 / MSE 单因素对照：完成，拒绝此升级配方

第3/12个方法候选已完成60轮、独立重放、11项配对比较、全量训练拟合和机制诊断。MSE的零值代价超过正值收益，补全及name-only退步；三个筛选条件全部失败，不追加其23/24种子。Transformer优于冻结RF的整体目标仍未达到。

完整[中文综合报告](../../report_snapshot_r9_mse_v1/REPORT_ZH.md)和[英文综合报告](../../report_snapshot_r9_mse_v1/REPORT_EN.md)保留数据来源、R0–R9历轮和基线方法。原[计划](PLAN.md)及[机器配置](config.json)保持冻结。

## 1. 研究问题与假设

假设平方残差训练能减轻MAE控制的正值低估，同时检查显式零、旧log-MAE及两个辅助任务的代价。PDF的宏轴MSE仅提供动机，其数据与评分不同，历史成绩不作为本轮对照。

## 2. 父版本与受控改动

第3/12个方法候选，仅MAE→MSE；种子20260922、初值、60轮样本顺序、学习率日程、数据、来源权重和评分均核对一致。模型192维/3层/6头/FF768/rank16/dropout0.15；AdamW lr3e-4、weight decay1e-4、batch64、clip1。来源校准和base损失取均值，残差L2系数1e-4。无两阶段、无树重拟合。

## 3. 复现信息

|Item|Value|
|---|---|
|Code commit|aa909f767c877d16945e4acef8a0d9bb430f97ae|
|Checkpoint SHA256|578cce5bd43d7b214b0f1368f97f1b54ab47984089665811bc2cf86abd2f2d85|
|Initial-state SHA256|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Selected epoch|56|
|Training elapsed (s)|10389.016|
|Training-fit inference elapsed (s)|57.047|

环境及精确源码身份保存在运行清单。耗时包含运行内核验和评价，不是纯训练速度；MAE为复用父运行，不能重复计入新增训练成本。

[Registration](PLAN.md) · [Configuration](config.json) · [Analysis](../../../../reports/v9_r9_mse_analysis_v1/summary.json) · [Training fit](../../../../reports/v9_r9_mse_fit_v1/summary.json)

9项功能/独立梯度测试通过；真实32训练任务50步MSE从0.557977降至0.011301，正式训练恢复初值和全部RNG。隐藏标签/来源前向不变、未观测监督轴查询及保存重载核验通过。模型共1,548,594参数，其中1,515,929可训练。完整独立重放核对60轮日程，两份各323,809条预测、49,913候选向量和19,089条排名均精确一致，未跳过NaN/Inf。

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_mse.py --candidate tf192_mse_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mse_lr3e4_60
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_r9_completed_utf8.py --run output/v9_r9_methods/tf192_mse_lr3e4_60 --output-dir reports/v9_r9_mse_lr3e4_60_audit_v1
.venv/Scripts/python.exe scripts/analyze_foodnutrigpt_r9_mse.py --output-dir reports/v9_r9_mse_analysis_v1
.venv/Scripts/python.exe scripts/diagnose_foodnutrigpt_r9_mse_fit.py --output-dir reports/v9_r9_mse_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_mse_fit_v1
```

这些命令已执行，已有目录拒绝覆盖。环境为Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB。完整运行约2.89小时，独立分析耗时1.125秒；全量训练拟合推理57.047秒。逐轴/来源和11项比较绑定在分析清单中，原始预测只留在本地数据目录。

## 4. 完整结果

MAE和MSE行均为种子22；与第6节MAE三种子均值严格区分。RF/XGB/KNN均是冻结结果。每个模型三项任务使用补全选中的同一检查点。主误差/条件正值/条件零值的分母不同，后两者不能直接相加。

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|MSE|0.193006|0.060136|0.365023|0.268409|0.122075|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|MSE|0.457182|0.157425|0.905488|0.517463|0.504336|
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
|MSE|completion|food_metabolome|45|0.590152|
|MSE|completion|all|187|0.288576|
|MSE|name_only|food_metabolome|45|0.815017|
|MSE|name_only|all|187|0.543292|
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
|MSE|0.3|0.000728|0.002573|0.006702|0.003591|
|MSE|1.0|0.001760|0.006705|0.011745|0.006941|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

固定49,913个名称候选，候选向量仅来自名称预测；查询不含名称。正确答案为原始名称精确匹配，尚未建立已确认别名映射。食品组条件性区间不覆盖种子总体、标签真实性或多轮选型的不确定性。

|Reference|Task|Metric|MSE gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-5.149879|-9.256798|-2.316294|
|mae_parent|completion|log_mae|-10.621548|-15.172193|-6.586417|
|mae_parent|name_only|scaled_log_mae|-7.216188|-9.913146|-4.638114|
|mae_parent|name_only|log_mae|-2.098213|-6.474531|2.049716|
|rf32|completion|scaled_log_mae|-2.102890|-5.179971|0.569103|
|rf32|completion|log_mae|-6.871111|-11.032837|-3.098267|
|rf32|name_only|scaled_log_mae|30.679347|28.460938|32.780659|
|rf32|name_only|log_mae|26.376228|23.344967|29.121762|
|xgb32|completion|scaled_log_mae|-10.613556|-14.127177|-7.494001|
|xgb32|completion|log_mae|-14.343356|-19.397318|-9.133662|
|xgb32|name_only|scaled_log_mae|30.323296|27.940966|32.473146|
|xgb32|name_only|log_mae|26.340148|23.192573|28.969658|
|knn32|name_only|scaled_log_mae|-74.323978|-80.280923|-68.396460|
|knn32|name_only|log_mae|-76.707504|-89.422073|-64.245350|

|Reference|Visible fraction|Metric|MSE minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|0.000209|-0.000789|0.001170|
|mae_parent|0.3|recall_at_1|0.000263|-0.000511|0.001038|
|mae_parent|0.3|recall_at_5|-0.000879|-0.002726|0.000916|
|mae_parent|0.3|recall_at_10|-0.000079|-0.002615|0.002568|
|mae_parent|1.0|mrr|0.001221|-0.000224|0.002672|
|mae_parent|1.0|recall_at_1|0.000737|-0.000383|0.001966|
|mae_parent|1.0|recall_at_5|0.000702|-0.001817|0.003189|
|mae_parent|1.0|recall_at_10|0.001142|-0.001947|0.004206|
|rf32|0.3|mrr|0.002114|0.001237|0.003036|
|rf32|0.3|recall_at_1|0.000418|-0.000279|0.001177|
|rf32|0.3|recall_at_5|0.001334|-0.000039|0.002847|
|rf32|0.3|recall_at_10|0.004715|0.002706|0.006942|
|rf32|1.0|mrr|0.004413|0.003286|0.005708|
|rf32|1.0|recall_at_1|0.001583|0.000701|0.002689|
|rf32|1.0|recall_at_5|0.005139|0.003218|0.007283|
|rf32|1.0|recall_at_10|0.007288|0.004714|0.010051|
|xgb32|0.3|mrr|0.002032|0.001195|0.002969|
|xgb32|0.3|recall_at_1|0.000418|-0.000279|0.001223|
|xgb32|0.3|recall_at_5|0.001102|-0.000377|0.002539|
|xgb32|0.3|recall_at_10|0.004883|0.002722|0.007013|
|xgb32|1.0|mrr|0.003524|0.002268|0.004878|
|xgb32|1.0|recall_at_1|0.001337|0.000349|0.002492|
|xgb32|1.0|recall_at_5|0.004236|0.002177|0.006385|
|xgb32|1.0|recall_at_10|0.005470|0.002570|0.008425|
|knn32|0.3|mrr|-0.030922|-0.033227|-0.028571|
|knn32|0.3|recall_at_1|-0.004421|-0.006232|-0.002821|
|knn32|0.3|recall_at_5|-0.052223|-0.057609|-0.047063|
|knn32|0.3|recall_at_10|-0.090020|-0.096773|-0.083183|
|knn32|1.0|mrr|-0.061938|-0.065575|-0.058669|
|knn32|1.0|recall_at_1|-0.011245|-0.014051|-0.008741|
|knn32|1.0|recall_at_5|-0.110869|-0.118453|-0.103745|
|knn32|1.0|recall_at_10|-0.181826|-0.191254|-0.173341|

## 5. 机制诊断

|Fixed-denominator component|MSE minus MAE|
|---|---|
|positive_under|-0.005673|
|positive_over|0.003953|
|explicit_zero|0.011173|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|MSE|train|0.130030|0.185514|0.077244|
|MSE|validation|0.193006|0.268409|0.122075|

|Objective|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|MSE|0.394570|0.141807|56|10389.016|

![MAE and MSE learning curves](../../../../reports/v9_r9_mse_analysis_v1/learning_curves.png)

三项共分母贡献相加精确重构主误差差值；全部1,828,536个训练目标的source-free拟合已核验。六面板图已实际查看，文字、图例及MAE第60轮/MSE第56轮选点清晰；原生成凭据保留，另存[视觉审阅凭据](../../../../reports/v9_r9_mse_analysis_v1/visual_review.json)。全部本轮营养预测配对比较的1000次食品组重采样均保留142轴，无因轴缺失丢弃的抽样。

训练集共同评分主误差从0.134937降至0.130030，验证却从0.183553升至0.193006。这表明训练拟合收益没有迁移到当前验证面板，不足以唯一证明过拟合或容量不足。训练与验证的食品、支持和选点不同；在线MAE/MSE目标数值不作跨损失大小比较。

共同分母下，正值低估减少0.005673，但正值高估增加0.003953，正值净收益仅0.001720；显式零代价0.011173使总误差增加0.009453。MSE后10轮裁剪比例0.141807，低于MAE的0.350883；因此“后期裁剪更频繁导致退步”不符合观测。其早期平均梯度范数较高也不等于所有轮次裁剪更频繁，尚无梯度尺度控制实验。

142轴中31轴点估计改善，7轴未调整的逐轴区间支持改善、63轴支持退步；20个氨基酸轴有19个点估计退步。逐轴探索未校正多重比较，9轴验证食品组少于30，不能把这些计数当成独立确认。七个训练支持少于100组的轴和115个至少1000组的轴分别贡献+0.004495和+0.004529；退步同时涉及稀疏轴与常见轴。来源贡献包含覆盖差异，不是数据库质量排名。

|Partition|Stratum|Supported axes|MSE minus MAE contribution|
|---|---|---|---|
|mask_family|amino_acid|20|0.001165|
|mask_family|fatty_acid|54|0.005487|
|mask_family|vitamin|30|-0.001197|
|source_key|bls_4_0|17|-0.000283|
|source_key|cnf|98|0.001132|
|source_key|foodb|89|0.004807|
|source_key|frida|72|0.000998|
|source_key|usda_sr_legacy|103|0.001233|
|support_bin|100_to_999|20|0.000428|
|support_bin|at_least_1000|115|0.004529|
|support_bin|below_100|7|0.004495|

完整[142轴变化与支持数](../../../../reports/v9_r9_mse_axis_changes_v1/axis_changes.csv)及[全部来源/家族/浓度分区](../../../../reports/v9_r9_mse_analysis_v1/mse_minus_mae_partitions.csv)保留。最大的两个轴贡献为Isomeric linolenic acids (18:3) +0.001882和Lignin +0.001455，各仅17个验证组；前者的未调整区间跨零。不可据这两个轴修改标签、删样本或改主指标。

已重建并核对MSE对RF的40条本地极端案例：[核验记录](../../../../reports/v9_r9_mse_cases_review_v2/summary.json)。较差20例含13个显式零、16例来自FooDB；轴为Biotin10例、Dodecanoic acid7例及其他3例。较好20例含17个正值、3个零值，涉及6个来源。它们是档案×轴的两端选择，存在同食品重复，不是随机样本，也不代表宏指标贡献。MSE较差尾部中没有Cholesterol，不能据此断言先前问题已解决或标签正确。

数值训练、重放和正式分析均成功。依赖作业曾因UTC时间二次解析偏移8小时而在启动前失败，修复后沿用原训练。案例核验v1对空Cholesterol集合输出了逻辑上的all-true；v2改为明确不适用/null，保留v1记录，40例选择与所有预测/指标均未改变。原始浓度和逐条预测仍只保存在本地数据目录。

## 6. 因果解释边界

损失替换是受控干预，但同时改变梯度大小、裁剪频率和优化轨迹。即使正值或主指标改善，也不能唯一归因为均值/中位数目标；失败只能否定此固定优化设置下的配方。训练/验证食品、支持数和选点不同，拟合差距本身不能证明过拟合。逐轴/来源分解为探索结果，未据其改数据、权重或门槛。

## 7. 筛选结果与版本决定

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

未满足进入独立登记种子复验的筛选条件；当前仍只有一个MSE种子，不能据此接受为最终模型。

版本决定：拒绝此固定MSE配方作为补全升级，不追加23/24种子，保留全部负结果及MAE控制。MAE三种子本身也未确认优于RF，不能把本轮损失选择写成最终目标已经实现。Name-only主指标比MAE退步7.216%，8个检索对照区间均跨零；较高的检索点值不能补偿补全退步。

## 8. 下一步与测试集状态

保留MAE作为已验证控制，下一问题是：当前共享数值编码器加轴标识的输入方式，是否限制了轴与数值的交互学习？拟单独检验一个零初始化的轴×数值线性残差，保持现有Transformer、损失、训练预算和全部数据指纹不变。这是待登记、待训练的假设，不是本轮已实现收益；参数增加与归纳结构也须区分，不能仅凭改善证明某一机制。特征专属数值向量的动机来自[FT-Transformer原论文](https://arxiv.org/html/2106.11959v5)，本项目的残差改造不等于复现其模型或成绩。测试继续关闭，来源留出及低标签迁移仍未验证。
