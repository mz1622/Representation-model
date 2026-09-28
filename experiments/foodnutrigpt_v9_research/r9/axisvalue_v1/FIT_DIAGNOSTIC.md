# 完整训练拟合诊断：等待正式分析成功

本候选增加48,384个参数。完整训练拟合用于判断在相同source-free评分下，训练收益是否迁移到验证集；即便发现差距，也不能单独证明过拟合或容量不足。

`scripts/diagnose_foodnutrigpt_r9_axisvalue_fit.py`只在60轮训练、独立重放、11项比较及分析成功后运行。复用已审计MAE父模型的训练预测；使用同一1,828,536个训练家族目标、相同标签、来源权重和142/45/187评分，无优化、无新训练面板文件或数据变换。新模型采用其补全选中的检查点。逐条预测只存本地忽略目录，输出分轴训练/验证指标供解释。

编译、依赖未完成时拒绝正式执行及代码差异检查已经通过。父拟合凭据和预测哈希复核通过，已有完整重算证据被复用，不称为本次再次执行。正式候选推理仍未执行。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_axisvalue_fit.py --check-only
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_axisvalue_fit.py --output-dir reports/v9_r9_axisvalue_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_axisvalue_fit_v1
```

主训练和分析队列完成前等待；检测到仍有GPU训练则拒绝并行竞争。最终结果、曲线、逐轴/来源解释、案例及双语版本报告仍需完成。不得根据拟合诊断反过来更改此次选择规则。
