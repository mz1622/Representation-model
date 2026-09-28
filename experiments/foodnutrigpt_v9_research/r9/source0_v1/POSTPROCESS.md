# 来源校准消融：完成后的分析与报告流程

这份流程在正式运行启动后、尚未读取本轮中途验证分数时准备。它只增加独立分析和报告工具，不修改登记的训练方法、接受条件、冻结数据、父模型或树基线。准备完成不等于训练完成或结果通过。

先在已安排的定时回访中核对 `work/r9-source0-v1-status.json` 对应控制器及实际进程身份。正常控制器负责训练、专用审计及完整独立重放、11项比较和RF差距诊断。若这些阶段未结束，不运行下面的正式分析；若失败，保留原产物定位原因，不重启或覆盖已有目录。不要按epoch反复查看。

控制器完成后，从代码库根目录依次执行：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_source0.py --output-dir reports/v9_r9_source0_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_source0_fit.py --output-dir reports/v9_r9_source0_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_source0_fit_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_source0_axes.py
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_source0_cases.py --output-dir reports/v9_r9_source0_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_source0.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_source0_v1
```

这些命令尚未执行。新的分析程序要求完整60轮记录、相同初值和参数量、相同任务/日程、关闭校准后的零来源表，以及已完成的专用和通用重放。比较的是冻结MAE父控制、RF/XGB和名称KNN。训练拟合只推理，不优化；1828536条预测保存在本地数据目录。

报告生成器保留上一版已审阅历史，增加第12节及中英文完全相同的新增数值表。生成器只输出草稿，不能自动宣称视觉审阅完成、接受版本或达成RF目标。报告完整交付前必须实际完成：

1. 查看六面板学习曲线，核对最优轮选择、裁剪和误差走势。
2. 解读共同分母的正值低估、正值高估、零值贡献，以及逐轴/来源支持与不确定性；重叠分区不能相加。
3. 核对全部训练目标的源无关拟合结果，区分训练拟合与验证泛化，不把拟合差距单独当作过拟合证明。
4. 审阅40个极端案例的汇总与逐轴区间；极端案例不是代表性样本，逐轴区间没有多重比较校正，不能据此改标签。
5. 根据原登记门槛保存 `decision.json`，明确事实、对照支持的解释、替代解释和是否进入另行登记的23/24种子复验。
6. 补齐八节README和中英文综合报告，保留普通CUDA严格预检失败及确定性参考恢复的记录。
7. 实际校对两种语言的解释、所有数值、引用文件身份及本地链接，另存最终review回执；草稿生成回执不能替代审阅。
8. 完成后停用或为下一个已登记运行更新同一 `nutrition` 定时任务；整个研究仍未达到目标时保持目标未完成。

分析层21项合成检查已通过：筛选必须同时满足三个条件、NaN/Inf不能通过、额外方法变化会拒绝、共同分母分解重构主差异，并拒绝不一致树贡献和权重。测试不使用本轮验证结果。

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_research_transformer_r9_source0_analysis.py
```

实现和核验清单见 `postprocessing_preparation.json`。上述准备不会改变正在运行的控制器，也不会创建新的训练或观察进程。
