# 固定树参照下的三种子统计

首组完整报告已选择lr3e-4、60轮进行复验，种子20260923正在运行，20260924串行接续；20260922复用已审计父模型。旧三种子确认入口要求重新训练三个RF/XGB并比较较强树、至少5%收益，与当前冻结树及R9判据不同，不能直接沿用。

新增独立入口 `scripts/confirm_foodnutrigpt_r9_fixed_references.py`，统计核心为 `src/foodcomp/research_fixed_reference_confirmation.py`。没有修改当前训练、审计、数据或任何树结果。

## 输入和拒绝条件

入口读取实际选择计划，要求全部三个完整Transformer及独立审计。逐种子核对完整配方、源码、环境、数据/文本/面板、所有轮次任务顺序与曝光、余弦学习率和最早最小选点。新增种子必须绑定原计划、父模型及选择证据，且仅种子改变。历史MLP或不完整运行不能替代。

RF/XGB读取R9冻结清单中的指定32维参照，各只有一个实际固定运行，不制造三个树种子。名称KNN32只进入name-only和检索，不冒用为同家族补全对照。两类预测表重新用共同评分函数评分；检索按来源等权形成食品组指标，查询、名称候选库及输入身份必须相同。NaN/Inf、负误差、缺失/重复面板或错误指纹使流程失败。

## 统计及接受规则

先对每个模型分别评分，再对相同食品组×轴的三个模型误差取均值；检索先计算每个模型的排名指标再平均。禁止先平均预测或排名，也不选择最佳种子。报告三个种子的均值、样本SD、范围和全部单次结果。固定RF/XGB/KNN的种子数量为1、SD不可用。

食品组配对重采样1000次，保持组内所有轴、来源以及三个种子共同取样。对RF和XGB都报告主指标、旧log及逐轴区间；name-only和两种可见率检索也有对应比较。接受门槛仅针对固定RF：主误差均值改善严格大于0、改善区间下界严格大于0、旧log相对退步不超过2%。不要求5%或超越XGB，沿用原R9登记。

保留187/142/45轴、正值/零值、原单位误差、分轴与分来源表；所有三个任务都使用每个种子的补全选点模型。食品组区间条件于这三个神经模型及固定树，不能估计总体训练随机性、树随机性、反复验证选择、标签真实性或外部泛化。逐轴区间是探索性结果，未校正多重比较。即使统计门槛通过，完整中英文报告仍须完成，统计脚本不会把整体目标标为完成。

## 检查及实际执行状态

38项针对性与复用统计测试通过，覆盖严格/非严格门槛边界、低于5%的有效收益、不必超过XGB、旧log保护条件、食品组计数、三个种子误差平均、有限值/面板守卫、完整轨迹及最早同分选点。第一次pytest命令遗漏PYTHONPATH，收集失败；显式设置src后全部通过。四类真实模型的名称缓存/候选/查询元数据兼容检查通过，活动训练队列30项指纹未改变。

真实依赖预检正确返回未就绪：种子23未完成、其审计缺失，种子24及其审计尚未生成；没有创建正式统计输出。见[准备核验](../../../reports/v9_r9_confirmation_statistics_preflight_v1/verification.json)。这些是实现和接口检查，不是实际三种子统计结果。完整数据重算只能在两次新运行及审计完成后执行。

```powershell
$env:PYTHONPATH=(Resolve-Path -LiteralPath src).Path
.venv/Scripts/python.exe -m pytest tests/test_research_fixed_reference_confirmation.py tests/test_research_r9_confirmation_statistics.py tests/test_research_final_statistics.py -q
.venv/Scripts/python.exe scripts/confirm_foodnutrigpt_r9_fixed_references.py --check-only --output-dir reports/v9_r9_three_seed_confirmation_v1
.venv/Scripts/python.exe scripts/confirm_foodnutrigpt_r9_fixed_references.py --output-dir reports/v9_r9_three_seed_confirmation_v1
```

已有输出目录拒绝覆盖；失败时保留失败凭据，用新版本处理恢复。
