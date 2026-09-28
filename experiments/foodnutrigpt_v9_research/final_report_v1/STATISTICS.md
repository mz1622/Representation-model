# 最终比较统计口径与执行记录

本文件落实[目标修订计划](PLAN.md)。实现发生在三个神经种子及部分树探索分数已知之后，不称为结果出现之前的统计预登记；未修改既有主指标、选点、种子或树配置选择规则。

## 比较口径

- 固定MLP及各自按R8全搜索选定的RF/XGBoost均需20260922/23/24三个种子；树配置只由seed22的补全主指标选择，其余种子不重新选配置。
- 对每个模型分别重算来源等权的食品组×轴误差，再平均三个种子的误差；不先平均预测，也不挑最好种子。另报种子均值、样本SD、范围与个体分数。
- 主区间对食品候选组整体配对重采样1000次，固定bootstrap seed20260922；同一组权重用于全部轴。主指标及旧log指标均报相对改善与绝对差值区间，展示所有树参照。
- 检索先在食品组×来源内平均查询，再组内来源等权；每个可见率分别平均三个种子的组分数，再做整体食品组区间。MRR及R@1/5/10报绝对差值，稀疏零参照不计算无定义的相对增益。
- 名称KNN是一个确定性参照，count=1、SD未定义；不复制三份制造种子重复性。其补全参照忽略数值上下文，名称任务与检索是主要用途。
- 全187轴保留每种子误差、正/零支持及逐轴配对区间。无该条件观测的正/零误差留空，不当作0。预测、主指标或排名的NaN/Inf直接失败。逐轴区间无多重检验校正，仅用于探索诊断。
- 来源表给出142轴范围内有支持轴的宏平均及支持数；不同来源覆盖不同轴，来源排名不解释为纯地域难度。
- 正值低估、正值高估、显式零使用原主指标分母，三项变化必须相加为主误差差值；条件正/零MAE本身不可这样相加。

区间条件于已选配置和三个实际模型；种子SD单列。区间不能量化种子总体全部不确定性，或消除反复验证及选点偏差。旧测试关闭。统计程序没有5%/2%胜负门槛，不自动宣称foundation能力或标记报告完成。

## 实现、验证与依赖

实现提交`63c64e1`：`scripts/compare_foodnutrigpt_final_models.py`及`src/foodcomp/research_final_statistics.py`，复用既有食品组bootstrap实现。13项针对性测试通过：缺失种子/覆盖变化、NaN与非法排名拒绝、来源等权、单参照SD语义、正零可加分解，以及神经checkpoint和树cache两种检索凭据格式。

真实阶段重算三个神经模型和KNN的两项323809行营养预测及19089检索排名，保存逐轴/来源支持与区间。只读完整模式预检在树证据未完成时报告ready=false且不写结果，不能生成缺少基线的“完整比较”。

第一次真实阶段`reports/v9_final_neural_reference_comparison_v1`因元数据格式差异失败：神经结果保存checkpoint SHA而非冗余name-cache SHA。保持失败目录；修复为通过已审计checkpoint/manifest校验缓存，未变更训练、预测或评分。v2完整通过，见[阶段报告](NEURAL_REFERENCE_COMPARISON.md)。

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m pytest tests/test_research_final_statistics.py -q
.venv/Scripts/python.exe scripts/compare_foodnutrigpt_final_models.py --neural-only --output-dir reports/v9_final_neural_reference_comparison_v2
.venv/Scripts/python.exe scripts/compare_foodnutrigpt_final_models.py --check-only --output-dir reports/v9_final_comparison_v1
```

完整模式等待R8全部12配置及选定树新增四次重复/独立审计：

```powershell
.venv/Scripts/python.exe scripts/compare_foodnutrigpt_final_models.py --output-dir reports/v9_final_comparison_v1
```

所有正式输出使用新目录；拒绝覆盖部分或完整结果。完整目录在依赖结束前应不存在。依赖状态以`work/final-statistics-status.json`为准，不能只根据本文件判断任务已启动或完成。
