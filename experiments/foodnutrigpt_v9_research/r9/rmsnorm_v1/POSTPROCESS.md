# 定时唤醒后的动作

先读workspace的work/r9-rmsnorm-v1-status.json，并一次检查其中登记的实际进程身份，PID需结合创建时间；不能只信work/r1-active-jobs.json。没有unified terminal session，不要调用write_stdin查询detach进程。

若队列还在运行：一次健康检查，保留日志及状态，重新估时更新既有nutrition；不要逐epoch观察。若失败：读异常、保护原目录、定位修复，不因等待超时重启训练。原config_v2.json、绑定代码和冻结输入不能在运行时改变。

若状态machine_evidence_complete_actual_review_and_bilingual_reports_required：

1. 检查完整60轮、独立重放、11项配对比较、gap分解、分析和训练拟合结果；all60学习率/顺序/曝光与父一致，检查结构规范与checkpoint kind。
2. 读取reports/v9_r9_rmsnorm_analysis_v1/summary.json和reports/v9_r9_rmsnorm_fit_v1/summary.json，实际查看learning_curves.png。审阅142轴、来源和支持数、局部成功失败案例。严格区分条件误差和共同分母贡献。
3. 完成README八节，出具独立中文和英文结构实验报告。对照父模型、MLP32和历史MLP128（三种子，注明不同名称维度）、RF/XGB及名称KNN。所有任务使用同一个检查点；报告所有失败和筛选条件，不拼接最好成绩。
4. 旧综合报告在experiments/foodnutrigpt_v9_research/report_snapshot_r9_lr2e4_v1/；补齐drop25_v1/POSTPROCESS.md要求的旧报告后再建立最新双语综合快照。不得将未审阅的草稿标为完成。
5. 区间仅包含食品组不确定性，不追加种子、dropout、学习率、损失或宽度搜索。通过结构筛选只能称固定种子有希望；下一SwiGLU候选先独立登记，不能自动混合RMSNorm或缩放FF维度。
6. 完成本轮报告后暂停现有nutrition回访，更新active-jobs并向用户报告结果和下一项最小结构实验。原始预测保持本地，代码/协议/聚合结果可本地提交，不自动push。

新脚本：audit_foodnutrigpt_r9_rmsnorm.py、analyze_foodnutrigpt_r9_rmsnorm.py、diagnose_foodnutrigpt_r9_rmsnorm_fit.py。初次融合桥接预检失败已记录，修订后配方相同。启动回访延迟以launch.json和automation实际更新时间为准。
