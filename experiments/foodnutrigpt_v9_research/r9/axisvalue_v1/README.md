# V9-R9 / 营养轴×数值残差：筛选完成，拒绝此固定升级配方

完整[中文报告](../../report_snapshot_r9_axisvalue_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_axisvalue_v1/REPORT_EN.md)包含数据来源、此前迭代及冻结RF/XGB方法。

## 1. 研究问题与预先假设

假设显式营养轴×数值交互能够改善共享数值编码器的条件表示。此前MSE配方被拒绝，本轮保留MAE，单独改变数值token。氨基酸、脂肪酸及正/零误差只作机制诊断，不替代142轴主指标。

## 2. 父版本与受控改动

第4/12个配方以同种子MAE为父控制，从头初始化。输入由e_axis+g(t)改为e_axis+g(t)+t*r_axis，新增252×192=48,384个零初始化参数；隐藏值不贡献残差。旧参数初值、构造RNG及零残差下的前向均精确匹配。192维、3层、6头、FF768、dropout0.15、rank16、MAE、来源损失权重1、来源L2=1e-4、batch64、AdamW lr3e-4/weight decay1e-4、clip1和60轮余弦日程不变。无树重拟合、无数据变化。

## 3. 可复现信息

|Item|Value|
|---|---|
|Code commit|89698ddeeda3377286c180e028119f00f4f06852|
|Checkpoint SHA256|512d963a4053262d90accaeae98f1c8d267f8501e9f7e1b69c6dac157e38a70c|
|Full initial-state SHA256|225b0385dd3ef60215abcf3543c00f63406c56ba8cf9836ef1ac1026244e6bbd|
|Shared parent initial-state SHA256|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Parameters|1596978|
|Trainable parameters|1564313|
|Added parameters|48384|
|Selected epoch|60|
|Run elapsed (s)|10564.781|
|Fit inference elapsed (s)|57.672|

全部60轮任务顺序、目标暴露量和学习率经独立重放核对。两份各323,809条预测、49,913个候选向量和19,089条排名精确重放。环境和命令见版本记录及运行清单；运行耗时含评价，MAE父运行不重复计入新增成本。

[Registration](PLAN.md) · [Configuration](config.json) · [Version record](README.md) · [Analysis](../../../../reports/v9_r9_axisvalue_analysis_v1/summary.json) · [Training fit](../../../../reports/v9_r9_axisvalue_fit_v1/summary.json)



```powershell
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_axisvalue.py --candidate tf192_mae_axisvalue_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_axisvalue.py --run output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60 --output-dir reports/v9_r9_axisvalue_lr3e4_60_audit_v1
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_axisvalue.py --output-dir reports/v9_r9_axisvalue_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_axisvalue_cases.py --output-dir reports/v9_r9_axisvalue_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_axisvalue_axes.py
```

环境：Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB。上述命令已执行，已有目录拒绝覆盖；完整命令及依赖哈希见运行/分析清单。正式训练、重放及比较无失败；案例读取出现pandas混合类型提示，但已用精确值与元数据逐项核对通过。

## 4. 完整结果

MAE与AXISVALUE均为种子22，RF/XGB/KNN为冻结结果；每个神经模型三任务使用同一补全选点。条件正值/零值误差的分母不同，不能相加重构主指标。历史MAE三种子均值见综合报告第6节，不与这里的单种子混用。

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|AXISVALUE|0.187792|0.056027|0.337126|0.306668|0.085240|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|AXISVALUE|0.419489|0.150305|0.909482|0.544610|0.375529|
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
|AXISVALUE|completion|food_metabolome|45|0.579883|
|AXISVALUE|completion|all|187|0.282145|
|AXISVALUE|name_only|food_metabolome|45|0.878140|
|AXISVALUE|name_only|all|187|0.529859|
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
|AXISVALUE|0.3|0.000632|0.002013|0.004341|0.002922|
|AXISVALUE|1.0|0.001481|0.004060|0.008550|0.005392|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

检索固定49,913个候选名称，候选向量仅由名称预测，不使用候选真实营养值；查询不含名称。正确答案仍为原始名称精确匹配，尚无确认别名映射。食品组条件性区间不覆盖种子总体、标签有效性或重复选型的不确定性。

|Reference|Task|Metric|AXISVALUE gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-2.309245|-5.099278|0.262962|
|mae_parent|completion|log_mae|-3.062976|-7.865762|0.976248|
|mae_parent|name_only|scaled_log_mae|1.623499|-0.920789|4.037533|
|mae_parent|name_only|log_mae|2.519862|-2.238321|7.260391|
|rf32|completion|scaled_log_mae|0.655429|-1.887559|3.016789|
|rf32|completion|log_mae|0.431201|-4.232860|4.237859|
|rf32|name_only|scaled_log_mae|36.394649|34.184365|38.709253|
|rf32|name_only|log_mae|29.706355|26.084824|33.035853|
|xgb32|completion|scaled_log_mae|-7.625321|-11.094921|-4.308238|
|xgb32|completion|log_mae|-6.530480|-13.040331|0.095379|
|xgb32|name_only|scaled_log_mae|36.067954|33.814848|38.440340|
|xgb32|name_only|log_mae|29.671907|25.932994|33.012904|
|knn32|name_only|scaled_log_mae|-59.951434|-65.062756|-54.786306|
|knn32|name_only|log_mae|-68.714726|-79.771306|-58.301770|

|Reference|Visible fraction|Metric|AXISVALUE minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|-0.000459|-0.001386|0.000453|
|mae_parent|0.3|recall_at_1|0.000168|-0.000581|0.000890|
|mae_parent|0.3|recall_at_5|-0.001439|-0.003195|0.000291|
|mae_parent|0.3|recall_at_10|-0.002440|-0.004658|-0.000073|
|mae_parent|1.0|mrr|-0.000329|-0.001599|0.000835|
|mae_parent|1.0|recall_at_1|0.000458|-0.000588|0.001528|
|mae_parent|1.0|recall_at_5|-0.001943|-0.004215|0.000165|
|mae_parent|1.0|recall_at_10|-0.002054|-0.004857|0.001027|
|rf32|0.3|mrr|0.001446|0.000635|0.002257|
|rf32|0.3|recall_at_1|0.000323|-0.000374|0.001032|
|rf32|0.3|recall_at_5|0.000774|-0.000607|0.002104|
|rf32|0.3|recall_at_10|0.002354|0.000521|0.004142|
|rf32|1.0|mrr|0.002864|0.001880|0.003965|
|rf32|1.0|recall_at_1|0.001305|0.000470|0.002233|
|rf32|1.0|recall_at_5|0.002494|0.000897|0.004173|
|rf32|1.0|recall_at_10|0.004093|0.001612|0.006595|
|xgb32|0.3|mrr|0.001363|0.000591|0.002164|
|xgb32|0.3|recall_at_1|0.000323|-0.000348|0.001071|
|xgb32|0.3|recall_at_5|0.000542|-0.000826|0.001910|
|xgb32|0.3|recall_at_10|0.002521|0.000877|0.004290|
|xgb32|1.0|mrr|0.001974|0.000900|0.003135|
|xgb32|1.0|recall_at_1|0.001058|0.000141|0.002045|
|xgb32|1.0|recall_at_5|0.001592|-0.000098|0.003409|
|xgb32|1.0|recall_at_10|0.002275|-0.000344|0.004973|
|knn32|0.3|mrr|-0.031590|-0.033910|-0.029264|
|knn32|0.3|recall_at_1|-0.004516|-0.006349|-0.002864|
|knn32|0.3|recall_at_5|-0.052783|-0.058309|-0.047599|
|knn32|0.3|recall_at_10|-0.092382|-0.099002|-0.085861|
|knn32|1.0|mrr|-0.063487|-0.067102|-0.060062|
|knn32|1.0|recall_at_1|-0.011524|-0.014273|-0.009174|
|knn32|1.0|recall_at_5|-0.113513|-0.121315|-0.106566|
|knn32|1.0|recall_at_10|-0.185022|-0.194445|-0.176533|

## 5. 机制诊断

|Fixed-denominator component|AXISVALUE minus MAE|
|---|---|
|positive_under|0.004091|
|positive_over|0.000922|
|explicit_zero|-0.000774|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|AXISVALUE|train|0.135435|0.247870|0.051219|
|AXISVALUE|validation|0.187792|0.306668|0.085240|

|Method|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|AXISVALUE|0.400266|0.344902|60|10564.781|

![MAE control and axis-value learning curves](../../../../reports/v9_r9_axisvalue_analysis_v1/learning_curves.png)

共同分母分解精确重构主误差增加0.004238696：正值低估增加0.004090529，正值高估增加0.000921943，零值收益为−0.000773776。六面板曲线已实际查看，两个模型均选第60轮；本轮末10轮裁剪比例0.344902，低于控制0.350883，未支持“裁剪更频繁导致退步”的解释。全量1,828,536个训练目标的无来源推理主误差也从0.134937略增至0.135435；不能据此单独区分优化、容量或泛化机制。条件正值/零值宏平均有各自分母，即使两项条件均值较低，也不保证总体宏平均较低。

相对同种子MAE，60/142轴点改善，14个逐轴区间支持改善、20个支持退步（均未作多重比较校正）；9轴验证支持不足30个食品组。7个训练支持少于100组的轴贡献+0.004304441，20个100–999组轴贡献+0.000573374，115个至少1000组轴贡献−0.000639120。稀疏轴中的Lignin和Isomeric linolenic acids贡献较大，支持仅17个验证食品组；后者区间跨零，不能把稀疏轴总体视为已确定的因果瓶颈。脂肪酸家族贡献+0.002138810、膳食纤维+0.001829756；FooDB来源分区贡献+0.004233540。来源、家族、支持度是同一误差的重叠分解，不能相加。

已从保存预测重建并核对40个RF对比极端案例。20个退步案例均来自FooDB，其中19个为正值Cholesterol，预测至少比保留标签低500倍；这不证明单位错误，也不证明这些标签正确。20个改善案例含14个正值、6个零值，覆盖5个来源、13个食品候选组。极端案例并不代表总体分布或40个独立样本，不能据此修改标签或排除数据。逐条食物名称、浓度与预测仍只存本地数据目录。

补全的142轴正值条件误差从0.302773增至0.306668，零值误差从0.092039降至0.085240。仅名称142轴主误差点改善1.623%，区间[−0.921%,+4.038%]跨零；其45轴和187轴误差反而变差。全可见/30%可见的R@10分别为0.008550/0.004341，均低于MAE；30%可见R@10差值区间为[−0.004658,−0.000073]，其余7个同种子检索指标差值区间均跨零。这些条件性区间未校正多指标选择，不能由某一辅助指标替代补全筛选。

[Visual review](../../../../reports/v9_r9_axisvalue_analysis_v1/visual_review.json) · [Axis contrasts](../../../../reports/v9_r9_axisvalue_axis_changes_v1/summary.json) · [Case review](../../../../reports/v9_r9_axisvalue_cases_v1/summary.json)

功能预检11项测试通过。真实32任务50步训练损失从0.217358降至0.044068，只证明可学习性；正式运行重置参数及RNG。最初pytest因缺少PYTHONPATH在收集阶段失败，修正执行环境后通过，没有借此重训正式候选。

## 6. 因果分析边界

观察事实是：此固定干预在单种子下没有满足改善条件，零值收益被正值代价抵消，训练主误差也略高。对照支持拒绝当前完整配方，不能证明所有轴专属编码均无效。新增48,384参数、交互形式和优化轨迹一同改变，无法唯一归因；未验证另一学习率、正则或等容量控制。条件区间不包括训练种子总体、反复选型、标签有效性或外部迁移不确定性。数据、尺度、指标、门槛均未随结果修改。

## 7. 筛选结果与版本决定

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

未满足预先规定的筛选条件。只有种子20260922的结果，尚不能接受为稳定改进或宣称优于RF。

版本决定：拒绝当前轴×数值残差配方作为补全升级；保留全部产物，不追加其20260923/20260924种子，继续使用原MAE作为方法控制。区间跨零不证明等效或普遍有害；本决定遵循预先规定的接受条件。该结果不完成Transformer优于冻结RF的整体目标。详见[机器决定](decision.json)。

## 8. 下一轮问题与测试状态

下一项优先核验训练中的来源校准：在原MAE控制上单独比较source weight 1→0，保持按(1+w)归一化的损失尺度及其余训练设置，先验证零残差初始时共享参数梯度一致。该问题尚未登记或训练；已有来源偏置稳定性不证明其收益，当前来源分区差异也不构成校准的因果证据。后续长训练将根据既有约3小时耗时安排定时回访，停止逐epoch观察。数据和RF/XGB继续冻结，测试保持关闭，来源留出、少样本迁移及可靠foundation model能力仍未验证。