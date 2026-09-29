# 容量候选6：运行中断与恢复记录

本页是同一候选的执行记录，研究协议仍见[原计划](../PLAN.md)。原48轮产物保持不变，恢复产物使用新目录。完整60轮和全部后处理已完成，结果与拒绝该固定容量配方的决定见[完整版本报告](../README.md)。

恢复于2026-09-29北京时间15:06启动，代码提交`3aa6c8bca7e490e9a93f8dd99120866b3d648352`，按原日程完成49–60轮；训练结束于15:48，后台全部分析及草稿生成结束于15:50。[启动回执](launch.json)保留实际进程身份和74分钟回访安排；16:20回访已确认进程退出且任务完整。没有重新运行前48轮，也没有根据部分分数修改配方。

[恢复计划](PLAN.md)、[恢复配置](config.json)和[实际中断证据](../../../../../reports/v9_r9_capacity256_interruption_v1/record.json)已保存。原PIDs及终端会话已确认不存在；日志没有明确退出原因，不据状态文件的running字样判断存活。

16项合成测试通过（3.77秒），包括CPU dropout网络中断前后连续三个AdamW/余弦步骤精确一致。[真实保存状态核验](../../../../../reports/v9_r9_capacity256_recovery_functional_v1/verification.json)也通过：模型、优化器、调度器、CPU/CUDA随机状态、下一段随机抽样及训练行前向/损失一致；完整每轮数值语句和checkpoint函数AST不变。未运行新的正式优化步或候选验证评价。

恢复只执行49–60轮，仍为MAE、source weight1、256/8/FF1024和原60轮余弦日程。最后10轮耗时中位数211.0935秒，剩余12轮乘1.5保守系数估计3799.683秒（约63分钟），首次回访74分钟。原未知未保存工作及停机时间不计入“原保存耗时＋恢复段耗时”，报告必须明确这个成本口径。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/resume_foodnutrigpt_v9_r9_capacity256.py --candidate tf256_mae_lr3e4_60 --output-dir output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_capacity256_recovery.py --run output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1 --output-dir reports/v9_r9_capacity256_lr3e4_60_audit_v1
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_capacity256_recovery.py --output-dir reports/v9_r9_capacity256_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_capacity256_recovery_fit.py --output-dir reports/v9_r9_capacity256_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_capacity256_fit_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_capacity256_axes.py
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_capacity256_recovery_cases.py --output-dir reports/v9_r9_capacity256_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_capacity256_recovery.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_capacity256_v1
```

新控制器会在恢复、原审计和11项比较完成后串行执行分析、训练拟合、逐轴/案例核验及双语证据草稿。生成草稿不代表实际曲线审阅或科学版本决定完成；不得据自动状态宣称超越RF或完成研究目标。
