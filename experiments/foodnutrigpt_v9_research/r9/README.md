# V9-R9：Transformer方法研究（首组运行中）

## 1. 问题与预先假设

当前目标是固定数据与已完成RF/XGBoost后，获得有验证依据且优于固定RF的Transformer，并交付完整中英文报告；见[目标与报告标准](EVIDENCE_AND_REPORTING.md)。首组假设是现有直接Transformer的优化速度和预算可能限制补全；两个学习率的完整60轮轨迹用于检验。[预登记计划](PLAN.md)保留原假设和方法，不倒写运行中的配置。

## 2. 父版本与控制

模型父实现为R2的192维source-free直接回归Transformer；输入采用已存在的R8公共PCA32，并保持与固定RF32一致。旧R2名称缓存不同，故必须重建同输入Transformer控制。首组仅lr .0001/.0003不同；固定3层、6头、FF768、dropout .15、rank16、batch64、MAE、source weight1、AdamW及60轮余弦日程。旧MLP作为辅助历史对照，不作为本目标的完成模型。

## 3. 复现和功能核验

数据和基线冻结凭据为`reports/v9_r9_freeze_v1/manifest.json`，SHA256 `1ec58002d76beb329004db77ef512622f57b40e142725799918d20027b5fddf3`。67个文件指纹和6个已完成树结果只读核对，两项营养预测评分全部重算一致。未改数据、缓存、任务或旧源码。

6项针对性测试通过。真实训练行核验`reports/v9_r9_functional_v1/verification.json`证明：与父192直接Transformer同初始化参数、前向与MAE损失精确一致；改隐藏标签、目标掩码、来源不改变前向；公共接口保存重载精确；可预测调用者指定的未观测轴；8个训练名称全部187查询有限。小样本50步损失.217358→.066450。参数1548594，其中可训练1515929。此处无验证性能结论。

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m pytest tests/test_research_transformer_r9.py -q
.venv/Scripts/python.exe scripts/freeze_foodnutrigpt_r9_inputs.py --output-dir reports/v9_r9_freeze_v1
.venv/Scripts/python.exe scripts/verify_foodnutrigpt_r9_functional.py --output-dir reports/v9_r9_functional_v1
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr1e4_60 --output-dir output/v9_r9/tf192_mae_lr1e4_60
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr3e4_60 --output-dir output/v9_r9/tf192_mae_lr3e4_60
```

所有存在目录拒绝覆盖。PDF解析曾遇控制台GBK字符输出异常，UTF-8文本已成功保存，六页渲染全部检查；不涉及训练或数据加工失败。正式命令、源码快照、环境及耗时由每个运行manifest记录。

首项训练后期发现待执行审计的Windows默认编码问题，已用全部49,913个候选名称复现并准备显式UTF-8修正版，见[编码问题与条件恢复记录](AUDIT_UTF8_NOTE.md)。原冻结训练源码、数据及队列保持不变；编码预检不能代替实际全量模型审计，也不表示训练失败。

## 4. 结果

首组长训练已顺序启动，运行实现提交`b4fd4c1`，尚无完整结果。队列先tf192_mae_lr1e4_60，随后tf192_mae_lr3e4_60；每项结束自动独立审计并与冻结RF/XGB/KNN比较。状态以各run_manifest和`work/r9-first-transformers-status.json`为准。固定RF32主误差.1890309379，旧log .0562700618；同输入固定XGB32主误差.1744867960。禁止用部分epoch、功能过拟合或旧MLP分数宣称Transformer获胜。

六个冻结树参照的[中英文方法与结果附录](BASELINE_APPENDIX.md)已完成，所有数值表格由同一机器结果生成并核对一致。两组训练完成后的20轮窗口评估及20/60配对比较已排队，独立控制器只等待与评估，不启动训练。阶段依赖未完成时不产生比较数值。

首组Transformer的[中文方法说明](../../../reports/v9_r9_transformer_methods_v1/METHODS_ZH.md)及[英文方法说明](../../../reports/v9_r9_transformer_methods_v1/METHODS_EN.md)记录实际结构、损失公式、遮蔽、参数依据及选点规则，区分工程固定项与待完成的控制实验。这是可用于最终报告的方法部分，尚不是最终配方或完整研究结果。

2026-09-28 03:34 UTC，首项lr1e-4已完成前20轮并保存预算快照，所选轮次18，历史主误差0.2195760950。CPU读取检查确认72个保存状态张量均有限，配置、种子、数据/缓存/任务身份一致，所选轮次是前20轮最早的主指标最小点；SHA256 `2642e315578f1f14f63bf7563a17c82b4a88745e36e1eb4bb42d303c38dd3e5d`。凭据为`reports/v9_r9_tf192_mae_lr1e4_60_budget20_ready_v1/verification.json`。这只是保存文件与历史的对应检查，没有进行前向或指标重放，也不是完整三任务结果；训练继续至60轮，正式预算评价仍等待两个父运行及审计结束。

## 5. 机制诊断

计划检查同日程20/60窗口、完整学习曲线、梯度裁剪频率、正值低估/高估和零值误差。若两个lr终点仍进步，只能提出预算假设；若训练误差低而验证停滞，需要进一步区分表示与泛化问题。具体解释须待实际曲线。

已补充[完整训练拟合诊断](FIT_DIAGNOSTIC.md)，在两组正式分析完成后，以相同source-free评分重建全部1,828,536个训练目标。在线loss含187轴及来源校准，不能直接与验证142轴主指标相减；该诊断用于辅助下一项方法选择，不重拟合参数或改动冻结数据。当前只完成入口和依赖预检，尚无训练拟合结果。

## 6. 因果边界

首组学习率干预只检验当前架构、目标及输入下的优化设置，不证明Transformer家族上限。PDF与当前协议差异大；借鉴架构不等于复现PDF优势。source残差只在训练损失中使用；所有方法推理保持不含来源。

## 7. 版本决定

功能检查通过，允许登记的两个完整实验进入执行；暂无模型改进接受或最终胜负结论。R8树搜索按用户冻结指示终止，不能称其3配置RF搜索已经完成。

## 8. 下一轮与测试状态

首组完整后独立重放、食品组配对区间及三任务报告，再依据证据登记PDF容量或MSE等单因素比较。有希望的固定Transformer配方做三个神经种子复核；固定RF不重跑。[20/60预算窗口分析](BUDGET_WINDOW_ANALYSIS.md)已落实为独立评估入口，仅在两个父运行及审计完成后执行。尚未证明优于RF时保持研究中；旧测试关闭，数据问题和外部迁移局限继续报告。
