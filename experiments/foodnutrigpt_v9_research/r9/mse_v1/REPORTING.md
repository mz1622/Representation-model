# MSE 完成后的双语证据草稿

当前状态：已生成并补充[中文报告](../../report_snapshot_r9_mse_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_mse_v1/REPORT_EN.md)，纳入完整MSE结果、曲线查看、逐轴诊断及40项案例核验。最初的evidence.json保持为生成时凭据；最终审阅另存review_verification.json，不把自动生成当成审阅。以下保留原工作流。

`scripts/report_foodnutrigpt_r9_mse.py`仅在原训练、独立重放、11项比较、完整分析和全量训练拟合诊断成功后生成新报告快照。它保留已审阅的R0–R9 MAE阶段报告作为第1–8节历史，增加第9节MSE单因素结果和第10节来源残差参数诊断；原报告不覆盖。

```powershell
.venv/Scripts/python.exe scripts/report_foodnutrigpt_r9_mse.py --check-only
.venv/Scripts/python.exe scripts/report_foodnutrigpt_r9_mse.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_mse_v1
```

入口核对证据哈希、冻结参照检索文件、相同验证指标、完整训练拟合和双语表格一致性，只读取汇总结果及哈希，不加载模型或原始食品数值。新增表格覆盖补全、仅名称、142/45/187轴、两种可见比例的Recall@1/5/10和MRR、食品组配对区间、正值/零值共分母贡献、全量训练拟合、裁剪频率和计算成本。

在正式MSE结果尚未完成时，`--check-only`已正确返回not ready。使用既有MAE结果构建的内存格式预检已验证11张中英文共用表格；其中代用角色和零占位仅用于格式检查，没有写入任何MSE报告或实验结果。本地预检凭据保存在work目录。

生成器必须把输出标为草稿，`manual_proofreading_complete`、`visual_review_complete`、`version_decision_complete`和`report_complete`均为false。它不会自动接受模型、生成已审阅凭据或追加种子。还需要实际查看六面板图，解释轴/来源差异与失败案例，完成本版本八节README，逐句审阅中英文报告，并另存绑定最终文件哈希的审阅凭据。

双语整理排在全量拟合诊断之后，控制器与状态保存于本地work目录；任一前置步骤失败即停止，保留原故障，不通过重训或覆盖结果绕过检查。
