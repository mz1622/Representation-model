# 冻结基线的双语附录

已从六个完整、冻结的RF/XGBoost运行生成[中文附录](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_ZH.md)及[英文附录](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_EN.md)。两份文档从[同一机器结果](../../../reports/v9_r9_frozen_baseline_appendix_v2/summary.json)生成，包含实际参数、输入和权重、训练支持数、三任务指标、142/45/187轴结果、正值/零值误差及计算成本。

六个完整参照保留全部配置，未仅摘录最优值；中断的RF leaf3和未运行的RF feature1另行明示。全部参照为seed20260922，不能伪称完成了新增树种子复核或完整RF搜索。R9同输入32维的固定补全主误差为RF 0.1890309379、XGB 0.1744867960，Transformer仍在训练，不在该附录中提前填入阶段最低值。

生成程序读取冻结指标和187条逐轴拟合凭据，核对各轴种子规则、公共参数、完整支持数、输入文件哈希以及数值有限性；不重新训练、评价新测试或改写任何基线结果。两种语言全部数值表格已逐行核对一致，英文无残留中文。版本v1是未交付草稿，v2只修正成本描述及一处存储措辞，所有预测与检索数值不变：原记录fit_seconds包括特征准备和检查，不能称纯fit调用耗时。

```powershell
.venv/Scripts/python.exe scripts/report_foodnutrigpt_frozen_baselines.py --output-dir reports/v9_r9_frozen_baseline_appendix_v2
```

已有目录拒绝覆盖；复现时使用新输出目录。机器结果保存生成代码、各运行与拟合记录哈希。此附录已经完成，但仍不是用户要求的中英文整体终稿；终稿还需新增Transformer训练、方法选择、多种子及归因分析。
