# 轴×数值残差：训练完成后的共同分析

正式分析入口为`scripts/analyze_foodnutrigpt_r9_axisvalue.py`，只读取已完成的训练、重放、11项配对比较和两个对RF的误差分解。它不训练、不改数据、不重新抽取任务、不打开测试。

训练尚未完成，`--check-only`正确返回依赖未齐且不写结果。17项合成检查通过：三个筛选条件、NaN/Inf拒绝、共同初始化身份和参数增加48,384的核对；单一字段偏离控制即失败。原训练及评分脚本保持冻结，分析入口不影响在跑配置。

完成后输出同种子MAE与新模型的三任务/142-45-187指标、60轮轨迹CSV、六面板PNG/SVG、正值低估/高估及零值的共同分母贡献、全部来源/家族/支持/浓度分区及筛选条件。所有分区须重构同一个主指标差值；稀疏轴与来源差异仍是描述性证据。新增参数数目及零初始化与父参数一致分别检查，不能误要求整模型哈希相等。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_axisvalue.py --check-only
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_axisvalue.py --output-dir reports/v9_r9_axisvalue_analysis_v1
```

仅预检已执行。正式分析等待原队列完整成功，失败时保留现场、不重启训练。图片生成后仍须实际查看，版本决定及中文/英文报告仍须人工/代理审阅；自动分析不能接受最终模型。
