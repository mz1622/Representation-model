# 完整训练拟合诊断：MSE 完成后执行

`scripts/diagnose_foodnutrigpt_r9_mse_fit.py`要求正式MSE训练、独立重放、11项配对比较及完整分析先成功。该入口尚未运行正式评估，不用在跑检查点推断最终成绩。

沿用首组的完整训练家族面板：1,828,536个已观测目标，查询面板指纹必须与已完成MAE诊断相同。只在内存中建立训练评分任务，原数据、划分、尺度、缓存、mask、验证任务文件不修改。缺失仍不监督，显式零仍是观测。

读取MAE控制既有的训练预测，验证文件及所对应检查点身份，并用共同评分重算，要求与原指标精确一致。随后对已完成、已审计、按补全选中的MSE模型做无梯度推理，记录142/45/187轴与逐轴结果。训练和验证都使用无来源残差的公开推理路径，不能把带来源校准的训练分数冒充推理能力。

```powershell
.venv/Scripts/python.exe scripts/diagnose_foodnutrigpt_r9_mse_fit.py --check-only
.venv/Scripts/python.exe scripts/diagnose_foodnutrigpt_r9_mse_fit.py --output-dir reports/v9_r9_mse_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_mse_fit_v1
```

第二条应在原队列全部完成后执行，避免与训练竞争GPU。本入口没有自动排队；检查--check-only为ready后才执行。已有目录拒绝覆盖。数值预测只写到本地忽略目录，公开报告只含汇总误差和支持数。

问题是：在同一评分定义下，MSE的训练集拟合和验证表现分别如何变化？这些证据可指导后续最小实验，但训练/验证食品和支持不同，检查点又由验证选择；差值本身不能证明过拟合，更不能唯一归因于容量、损失或校准。它不会修改预登记筛选条件，也不会自动接受模型或增加种子。
