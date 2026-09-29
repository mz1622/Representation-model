# 候选8完成后的处理步骤

本文件于训练期间准备，仅涉及未加入队列的新报告代码。不得修改控制器、训练和已排队的审计/比较/分析/拟合脚本。作业位置见工作区 `work/r1-active-jobs.json`；该文件不是进程存活证据。已安排2026-09-30 00:36:50北京时间首次回访，不按epoch轮询。

1. 到回访时间核对控制器9252、虚拟环境启动器17416、实际训练进程15800及各自创建时间，再读取 `work/r9-lr2e4-v1-status.json`。CIM与Get-Process时间显示精度不同，比较共同微秒精度并保留原值，不用字符串完全相等判断进程更换。检查16个队列命令的退出状态与产物。此作业没有统一终端session，不能使用write_stdin；观察超时不等于训练失败。
2. 队列达到 `machine_evidence_complete_actual_review_and_bilingual_reports_required` 后，运行以下三个新入口。缺少完整分析时不能生成报告；不得覆盖原有产物，失败记录必须保留。
3. 实际查看曲线PNG、142轴、全部来源/家族/支持数分区及40条本地案例。脚本成功或生成标志不能代替实际审阅。不要在报告中输出原始食物级数值。
4. 完成八节README与版本决定；更新新中文/英文综合报告的概览及第15节。第2–14节保留既有历史证据。分别陈述事实、对照支持的解释和替代解释，并实际校对两种语言、数值表、百分比方向和链接。
5. 必须保留本轮首次默认GPU梯度预检失败及独立确定性诊断的范围。精确梯度结论仅适用于诊断后端，不能升级为正式训练轨迹逐位相同。正式训练后端与父保持一致；此处不是上一轮PowerShell启动失败的复发。
6. 仍按登记的三个筛选条件决定是否另行登记20260923/24复验。单种子筛选不代表稳定改善；失败同样形成完整版本。完成当前报告后停用回访，下一项正式训练按新估时启用；不将阶段报告误标为整体研究目标完成。

从仓库根目录运行：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_lr2e4_axes.py
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_lr2e4_cases.py --output-dir reports/v9_r9_lr2e4_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_lr2e4.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_lr2e4_v1
```

三个入口均支持 `--check-only`，仅检查依赖状态，不生成结果。新报告工具未追加至正在运行的控制器，回访完成后再执行。不得提前照搬上一轮负结果或预先决定本轮归因。
