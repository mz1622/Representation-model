# 候选7完成后的处理步骤

此文件于训练期间准备，仅涉及未加入运行队列的新报告代码。不得修改当前控制器、训练或已排队的审计/比较/分析/训练拟合脚本。最新作业位置见工作区 work/r1-active-jobs.json；它不是进程存活证据。现有回访预计2026-09-29 20:46北京时间执行，勿按epoch轮询。

1. 到回访时间只核对一次控制器7700、实际训练进程22932的创建时间、实际存活状态及 work/r9-lr6e4-v1-status.json。已完成时检查队列的16个命令（训练、独立审计、11项比较、gap、analysis、fit）的退出状态与结果。不存在统一终端session，不能调用write_stdin。观察超时不代表失败，不能直接重启。
2. 若队列完成并且状态为 machine_evidence_complete_actual_review_and_bilingual_reports_required，执行以下三个新入口。它们要求完整分析已经存在，禁止覆盖既有产物；若中途失败，保存目录，定位原因后再版本化处理。
3. 实际查看学习曲线PNG，逐轴142行、全部来源/家族/支持数分区及40条本地案例的核验。不能用 generated、check-only、tests passed 或报告标志代替实际阅读。
4. 完成版本决定和八节README，改写新综合报告概览与第14节解释，清楚区分事实、受控解释和替代解释；此前已审阅章节保留历史语义。中文和英文报告都需要实际校对；逐表验证数值与方向一致、相对百分比符号正确、文件链接可用。
5. 依照原三个筛选条件决定是否另行登记两个固定种子；单种子未通过不追加复验。停止当前回访；下一轮正式训练再按估时重新启用。整体目标尚未验证完成，不将本阶段报告视作目标完成。

从仓库根目录运行：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_lr6e4_axes.py
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_lr6e4_cases.py --output-dir reports/v9_r9_lr6e4_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_lr6e4.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_lr6e4_v1
```

三个入口支持 --check-only，仅检查完成依赖，不能生成结果或报告。这些步骤未追加到当前控制器；由定时回访后继续执行，避免修改正在运行的已锁定队列。
