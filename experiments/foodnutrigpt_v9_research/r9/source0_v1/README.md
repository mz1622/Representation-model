# V9-R9 / 关闭来源校准：筛选完成，拒绝此固定升级配方

完整[中文报告](../../report_snapshot_r9_source0_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_source0_v1/REPORT_EN.md)包含数据来源、全部历史迭代及冻结RF/XGB方法。

## 1. 研究问题与预先假设

训练同时拟合基础输出与已知来源的校准输出，推理只评价基础输出。本轮检验关闭校准目标能否改善基础预测。此前来源偏置稳定性和来源分区误差均不证明校准有益或有害。

## 2. 父版本与受控改动

第5/12个候选以同种子MAE为控制，仅将source weight从1改为0。损失仍为(L_base+w*L_calibrated)/(1+w)+1e-4*mean(R_train²)，保留归一化，避免同时改变基础损失尺度。模型、零初始化来源表、L2、优化器和样本来源平衡权重均保留；来源表在w=0时保持零。192维、3层、6头、FF768、dropout0.15、rank16、MAE、batch64、AdamW lr3e-4/weight decay1e-4、clip1及60轮余弦日程均不变。

## 3. 可复现信息

|Item|Value|
|---|---|
|Code commit|92f80d90b798822d39af4644d65ea9441e52d042|
|Checkpoint SHA256|172d5af07d978672824daa3ba9e376d5c7fb94504e81d39b54ef3cd83f107c73|
|Initial-state SHA256 (same as parent)|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Parameters|1548594|
|Parameters with requires_grad|1515929|
|Source weight|0.0|
|Selected epoch|58|
|Run elapsed (s)|10516.218|
|Fit inference elapsed (s)|56.656|

全部60轮任务顺序、目标暴露量和学习率与父控制一致；两份各323,809条预测、49,913个候选向量和19,089条排名独立精确重放。最优与最终来源表均为零；该表计入requires_grad参数但有效梯度为零。环境为Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB。正式训练保持父CUDA后端，确定性设置仅用于功能参考进程。训练前估时约3小时并设置190分钟回访；估时不替代实际成本。

[Registration](PLAN.md) · [Configuration](config.json) · [Version](README.md) · [Analysis](../../../../reports/v9_r9_source0_analysis_v1/summary.json) · [Training fit](../../../../reports/v9_r9_source0_fit_v1/summary.json) · [Functional checks](../../../../reports/v9_r9_source0_functional_v1/verification.json)



```powershell
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_source0.py --candidate tf192_mae_source0_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_source0_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_source0.py --run output/v9_r9_methods/tf192_mae_source0_lr3e4_60 --output-dir reports/v9_r9_source0_lr3e4_60_audit_v1
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_source0.py --output-dir reports/v9_r9_source0_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_source0_fit.py --output-dir reports/v9_r9_source0_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_source0_fit_v1
```

上述运行已完成。独立审计和11项比较均通过，正式训练无失败；案例读取的pandas混合类型提示未影响逐值和元数据核验。功能预检失败及恢复见下文。额外训练拟合推理耗时56.656秒，没有优化器更新。

## 4. 完整结果

MAE和SOURCE0均为种子22；RF/XGB/KNN沿用冻结结果。各神经模型三任务使用同一个补全选中的检查点。综合报告第6节的历史MAE三种子均值与本节单种子比较不同。

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|SOURCE0|0.187741|0.056708|0.344686|0.304187|0.090369|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|SOURCE0|0.444413|0.156721|0.974482|0.554378|0.424196|
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
|SOURCE0|completion|food_metabolome|45|0.592878|
|SOURCE0|completion|all|187|0.285234|
|SOURCE0|name_only|food_metabolome|45|0.845787|
|SOURCE0|name_only|all|187|0.541000|
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
|SOURCE0|0.3|0.000426|0.002839|0.004762|0.002786|
|SOURCE0|1.0|0.000404|0.005040|0.009677|0.004908|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

检索固定49,913个名称候选，候选向量仅由名称预测，不能用其真实营养值；营养查询不含名称。正确答案仍按原始名称精确匹配，无确认别名映射。食品组条件区间不包括种子总体、标签有效性或重复选型的不确定性。

|Reference|Task|Metric|SOURCE0 gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-2.281351|-5.218685|0.429649|
|mae_parent|completion|log_mae|-4.314022|-9.464234|0.242264|
|mae_parent|name_only|scaled_log_mae|-4.221660|-7.037420|-1.251824|
|mae_parent|name_only|log_mae|-1.641299|-7.268159|3.944402|
|rf32|completion|scaled_log_mae|0.682515|-2.544394|3.790670|
|rf32|completion|log_mae|-0.777430|-6.360213|4.283938|
|rf32|name_only|scaled_log_mae|32.615460|30.424656|34.933673|
|rf32|name_only|log_mae|26.705711|22.994630|30.025658|
|xgb32|completion|scaled_log_mae|-7.595977|-11.287282|-4.005461|
|xgb32|completion|log_mae|-7.823616|-15.191958|-1.165845|
|xgb32|name_only|scaled_log_mae|32.269354|29.886638|34.557613|
|xgb32|name_only|log_mae|26.669792|22.731187|30.075311|
|knn32|name_only|scaled_log_mae|-69.455141|-74.486058|-64.052129|
|knn32|name_only|log_mae|-75.916697|-85.459950|-66.514116|

|Reference|Visible fraction|Metric|SOURCE0 minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|-0.000596|-0.001522|0.000277|
|mae_parent|0.3|recall_at_1|-0.000039|-0.000698|0.000619|
|mae_parent|0.3|recall_at_5|-0.000613|-0.002459|0.001174|
|mae_parent|0.3|recall_at_10|-0.002019|-0.004355|0.000381|
|mae_parent|1.0|mrr|-0.000812|-0.001971|0.000283|
|mae_parent|1.0|recall_at_1|-0.000618|-0.001422|0.000150|
|mae_parent|1.0|recall_at_5|-0.000963|-0.003221|0.001481|
|mae_parent|1.0|recall_at_10|-0.000927|-0.003826|0.002342|
|rf32|0.3|mrr|0.001309|0.000516|0.002114|
|rf32|0.3|recall_at_1|0.000116|-0.000504|0.000774|
|rf32|0.3|recall_at_5|0.001600|0.000155|0.002994|
|rf32|0.3|recall_at_10|0.002774|0.000916|0.004633|
|rf32|1.0|mrr|0.002380|0.001600|0.003162|
|rf32|1.0|recall_at_1|0.000228|-0.000207|0.000776|
|rf32|1.0|recall_at_5|0.003474|0.001831|0.005313|
|rf32|1.0|recall_at_10|0.005220|0.002609|0.007660|
|xgb32|0.3|mrr|0.001227|0.000476|0.001969|
|xgb32|0.3|recall_at_1|0.000116|-0.000466|0.000774|
|xgb32|0.3|recall_at_5|0.001368|-0.000052|0.002840|
|xgb32|0.3|recall_at_10|0.002942|0.001329|0.004777|
|xgb32|1.0|mrr|0.001491|0.000640|0.002447|
|xgb32|1.0|recall_at_1|-0.000019|-0.000583|0.000569|
|xgb32|1.0|recall_at_5|0.002572|0.000780|0.004500|
|xgb32|1.0|recall_at_10|0.003402|0.000728|0.006095|
|knn32|0.3|mrr|-0.031727|-0.034128|-0.029425|
|knn32|0.3|recall_at_1|-0.004723|-0.006465|-0.003071|
|knn32|0.3|recall_at_5|-0.051957|-0.057276|-0.047090|
|knn32|0.3|recall_at_10|-0.091961|-0.098826|-0.085333|
|knn32|1.0|mrr|-0.063971|-0.067440|-0.060863|
|knn32|1.0|recall_at_1|-0.012601|-0.015250|-0.010235|
|knn32|1.0|recall_at_5|-0.112533|-0.120300|-0.105451|
|knn32|1.0|recall_at_10|-0.183895|-0.193454|-0.175241|

## 5. 机制诊断

|Fixed-denominator component|SOURCE0 minus MAE|
|---|---|
|positive_under|0.002142|
|positive_over|0.001669|
|explicit_zero|0.000377|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|SOURCE0|train|0.136431|0.245846|0.052563|
|SOURCE0|validation|0.187741|0.304187|0.090369|

|Method|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|SOURCE0|0.402164|0.347503|58|10516.218|

![Source-loss weight1 versus0 learning curves](../../../../reports/v9_r9_source0_analysis_v1/learning_curves.png)

共同分母分解精确重构主误差增加0.004187495：正值低估+0.002141826、正值高估+0.001668563、显式零+0.000377106。注意：条件零值宏平均从0.092039降至0.090369，但零值对总体主误差的贡献反而增加，因为两种统计的轴内分母和有效权重不同；不能将条件零值均值降低直接称为总体收益。全量1,828,536个训练目标的无来源推理主误差从0.134937增至0.136431，也没有显示基础预测拟合改善。训练条件正值/零值均值虽降低，也不能相加推出主误差改善。

已实际查看六面板曲线。SOURCE0选第58轮，MAE控制选第60轮，两个模型后期曲线趋缓；这不证明训练已达全局最优。末10轮梯度裁剪比例0.347503，控制为0.350883，未支持“关闭校准导致更多后期裁剪”的解释。均值梯度范数有波动，但未见发散走势。训练目标在去掉校准项后语义不同，因此机制比较使用共同的无来源拟合指标，不直接以两条训练loss的高低证明优劣。

58/142轴点改善；11个逐轴区间支持改善、26个支持退步，均未作多重比较校正，9轴验证支持不足30个食品组。7个训练支持少于100组的轴贡献+0.003714232，20个100–999组轴贡献+0.000491940，115个至少1000组轴贡献−0.000018677。膳食纤维家族贡献+0.002794349、脂肪酸+0.001034044；FooDB来源分区贡献+0.003903528。Lignin和Isomeric linolenic acids各只有17个验证食品组，其逐轴区间均跨零；不能把这些稀疏轴认定为已证实的因果瓶颈。来源、家族和支持度分区相互重叠，不能相加。

从保存预测重建并核对40个RF对比极端案例。20个退步案例均来自FooDB，含19个正值、1个零值，来自16个食品候选组；其中10个Cholesterol案例的预测至少比保留标签低500倍。这既不证明单位错误，也不证明标签正确。20个改善案例含15个正值、5个零值，覆盖5个来源、15个食品组。极端尾部不是代表性或独立样本，不据此修改数据；逐条名称、浓度和预测留在本地数据目录。

仅名称142轴主误差0.444413，比同种子MAE高4.222%，改善区间[−7.037%,−1.252%]支持这一条件性退步。相对冻结RF/XGB，它的仅名称表现仍较好，但明显弱于名称KNN（0.262260），不能因此宣称名称任务已解决。全可见/30%可见的检索R@10为0.009677/0.004762，八个同种子检索指标均点退步且其差值区间均跨零。45轴补全点改善而187轴补全变差，辅助指标不替代142轴筛选。所有结果使用同一个第58轮检查点。

[Visual review](../../../../reports/v9_r9_source0_analysis_v1/visual_review.json) · [Axis contrasts](../../../../reports/v9_r9_source0_axis_changes_v1/summary.json) · [Case review](../../../../reports/v9_r9_source0_cases_v1/summary.json)

|Functional exercise|Value|
|---|---|
|Training tasks|32|
|Steps|50|
|Loss before|0.217358|
|Loss after|0.058560|

14项共享/合成功能测试通过。普通GPU预检在优化器步前因共享梯度逐位比较失败；重复同一目标也出现微小差异。CPU和确定性GPU参考在各自设备内精确匹配，未放宽原断言。完整50步检查通过，来源表保持零，学习后public loader重载精确。失败与恢复单独留档；这些只支持实现与可学习性，不是验证性能证据。

## 6. 因果分析边界

观察事实是：删除校准目标未满足当前配方的任何筛选条件，基础训练主误差也更高。对照支持保留source weight=1，拒绝这个固定的关闭方案；不证明校准总是有益，也不证明所有关闭方案必然有害。参数量和初值相同，干预改变校准梯度及后续优化轨迹；真实来源偏差校正、隐式正则和优化效应仍不能唯一分离。普通GPU存在微小非确定性，而单种子条件区间不包含种子总体、反复选型、标签真实性或外部迁移的不确定性。没有改数据、指标或门槛来追求获胜。

## 7. 筛选与版本决定

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

未满足登记的筛选条件；单种子不能确认为稳定改进或优于RF。

版本决定：拒绝此SOURCE0配方作为补全升级，不追加20260923/20260924复验，保留全部产物和失败记录，继续以MAE、source weight=1为方法控制。区间跨零不证明等效；保留控制是遵循预先接受条件，不是证明其所有参数最优。本轮不完成Transformer优于冻结RF的整体目标。见[机器决定](decision.json)。

## 8. 下一轮问题与测试状态

下一项优先问题是原PDF的容量组合：256维、8头、FF1024，相对于当前192维、6头、FF768，保持MAE、来源权重1及其余训练方法不变。假设更大的条件表示容量可能改善拟合，但也可能增加泛化代价；宽度、头数和前馈维度共同变化，不能称为纯宽度因果实验。该方向已列于R9总计划，具体候选尚未登记或训练。数据和RF/XGB继续冻结、测试保持关闭，后续仍先估时并定时回访。来源留出、少样本迁移及可靠foundation model能力仍未验证。