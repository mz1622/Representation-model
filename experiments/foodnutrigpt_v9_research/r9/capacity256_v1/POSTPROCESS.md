# 容量组合实验的结果处理准备

以下流程在本轮验证结果尚未查看时准备。它不改变已冻结的训练程序、配置、输入、指标或筛选门槛，也不构成实验结果。正式版本仍须完成实际曲线审阅、机制解释和八节README。

新增容量专用分析入口核对256模型自己的初值指纹和参数量，并与训练前功能记录一致；同种子192控制的初值只作为独立参照。三项结构配置必须作为一组变化，其余数值设置必须相同。不能沿用来源权重消融中的“同初值、同参数量、来源残差恒零”断言。

32项合成测试通过（2.66秒）：筛选三条件及边界、NaN/Inf拒绝、额外干预拒绝、错误同初值声明拒绝、与功能记录不符的参数量/初值拒绝，以及共同分母的误差分解守恒。六个新模块编译通过。这些均不证明候选的验证效果。

运行后先由现有控制器完成专用审计、完整重放、11项比较及对RF的误差诊断；以下命令只应在控制器终止且全部必要记录完整后运行。所有目录拒绝覆盖，发生错误需保留失败记录并判断恢复方式，不能直接重跑训练。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_capacity256.py --check-only
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_capacity256.py --output-dir reports/v9_r9_capacity256_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_capacity256_fit.py --output-dir reports/v9_r9_capacity256_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_capacity256_fit_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_capacity256_axes.py
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_capacity256_cases.py --output-dir reports/v9_r9_capacity256_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_capacity256.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_capacity256_v1
```

三个带`--check-only`的入口（分析、训练拟合、双语报告）只检查完成状态及所需文件；证据未齐时不能生成正式分析和报告。训练拟合只是对完整1,828,536个训练目标推理，与旧MAE预测重评分后比较，不进行梯度更新。逐条预测继续保存在忽略的本地目录。

双语生成器保留上一份完整来源校准版本的第1–12节作为历史证据，在第13节加入本次容量组实验。两种语言共用同一组数值表；在实际审阅前明确标为草稿，不自动宣称视觉审阅、版本决定、稳定改进或研究目标完成。后续须核对每个链接和数值表，更新摘要为当前结论，并分别记录事实、支持的解释和未排除的替代解释。

准备核验的机器记录将保存在[verification.json](../../../../reports/v9_r9_capacity256_postprocess_preparation_v1/verification.json)。训练继续由[原启动记录](launch.json)对应的控制器负责；仍按既有267分钟定时回访安排处理，不读取中途epoch或候选分数。
