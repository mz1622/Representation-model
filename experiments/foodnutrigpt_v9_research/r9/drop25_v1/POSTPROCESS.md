# 候选9完成后的处理步骤

本文件在训练期间准备，未改变正在运行的16项队列。现有首次回访时间为北京时间2026-09-30 04:13:21，不逐轮查看epoch。工作区 `work/r1-active-jobs.json` 仅定位任务，不是存活证据。

1. 回访时一次核对控制器10720、虚拟环境启动器23172、实际训练进程7568及各自出生时间，再检查 `work/r9-drop25-v1-status.json`。出生时间比较Windows接口共同微秒精度。无统一终端session，不调用write_stdin；观察超时不代表失败，不据此重启。
2. 队列16项退出成功且状态为 `machine_evidence_complete_actual_review_and_bilingual_reports_required` 后执行以下新入口。完整分析缺失时不得生成报告；输出目录禁止覆盖。失败要保留并另行版本化修复。
3. 实际查看全部六面板曲线、142轴、所有来源/家族/支持数分区、40条本地极端案例和低支持轴训练/验证拟合。脚本生成标志不代替阅读；个体数值仅保存在本地。
4. 完成八节README、版本决定和新综合中文/英文报告概览及第16节。第2–15节保留此前历史证据。实际校对两种语言、数字表、效应方向与链接，区分观察事实、受控解释及替代解释。
5. 本轮是共享dropout .15→.25：14处配置同时变化，初始参数与评估函数相同、训练函数与梯度预期不同。学习率日程保持相同，不是上一轮2e-4对照；本轮没有前轮的梯度精确相等预检失败，不照搬其叙述。
6. 三个筛选条件全部通过才另行登记20260923/24；单种子不宣称稳定超过RF。成败均形成完整记录，完成本轮后暂停回访，下一项正式训练才按新估时启用。整体研究目标未被本轮报告完成所替代。

从仓库根目录运行：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_drop25_axes.py
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_drop25_cases.py --output-dir reports/v9_r9_drop25_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_drop25.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_drop25_v1
```

三个入口提供 `--check-only`，只核对完成依赖；没有增加新的训练观察器，也未写入性能结论。
