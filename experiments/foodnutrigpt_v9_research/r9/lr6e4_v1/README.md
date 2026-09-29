# V9-R9 / Candidate7：learning rate 6e-4

**状态：2026-09-29 16:50（北京时间）已启动60轮训练；预计20:46定时回访。** 本页不是完整结果报告。配置、计划及预检证据在训练前冻结；启动回执和结果按完成阶段追加，不改写预注册结论。

## 1. 研究问题与预先假设
192维MAE控制能否在相同60轮预算下受益于更高的学习率？上一轮容量扩展使训练与验证均变差，先继续检查192维模型的优化设置。改善是待验证假设。

## 2. 父版本与改动
父运行 tf192_mae_lr3e4_60；仅初始learning_rate从3e-4到6e-4，完整绝对余弦日程加倍，最低比例保持.01。AdamW的每步解耦衰减幅度也随学习率改变，不能把作用仅归因于“训练更快”。其他模型、损失、校准、初始化、种子、任务、曝光、数据与评分不变。
详细假设、控制和门槛见 [PLAN.md](PLAN.md)，冻结参数见 [config.json](config.json) 与 [registered_spec.json](registered_spec.json)。

## 3. 可复现信息
预检共42项测试通过，21项训练/数值/隔离检查用时3.82秒，21项分析、区间门槛和统一分母检查用时2.33秒。最初测试收集失败因未设置PYTHONPATH=src；修正命令后通过，未改变模型或数据。控制器草稿曾有JavaScript字符串插值错误，未写文件或启动进程；修正后PowerShell语法解析通过。
真实训练行GPU预检验证父初始化、初始损失/梯度/随机数状态完全一致，小批次误差0.2173580974→0.0608443543；验证仅证明实现和学习能力，不是当前候选性能。
- [功能检查](../../../../reports/v9_r9_lr6e4_functional_v1/verification.json)
- [测试与执行准备](../../../../reports/v9_r9_lr6e4_preparation_v1/verification.json)

PowerShell复现入口（仓库根目录；现有输出存在时禁止重复运行）：

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_research_transformer_r9.py tests/test_research_transformer_r9_source0.py tests/test_research_transformer_r9_lr6e4.py tests/test_research_transformer_r9_lr6e4_analysis.py
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_lr6e4.py --candidate tf192_mae_lr6e4_60 --output-dir output/v9_r9_methods/tf192_mae_lr6e4_60
```

同机历史耗时保守估算13479.7949秒（约3小时45分）；另预留10分钟，首次回访间隔235分钟。实际成本由完成清单给出，不以估算冒充实测。训练独立后台进程和后处理队列，见[启动回执](launch.json)。实际代码提交43165c07755b4e91af3b69e479ed7b3568efd119；控制器7700、启动器9568、实际训练进程22932，已核对创建时间及初始配置；进程号仅为历史回执，后续不能单凭号码判断存活。

首次使用powershell.exe的控制器26592因子进程无法识别Get-FileHash，在训练目录/状态创建前退出，优化步数为0。确认无训练进程后，保持控制器脚本和方法登记不变，改用已验证的PowerShell7.6.5路径成功启动，日志另存attempt2。见[启动失败记录](../../../../reports/v9_r9_lr6e4_launch_failure_v1/record.json)。启动回执检查曾因PowerShell返回大写SHA而误报不匹配；将摘要大小写统一后校验通过，文件本身没有变化。此问题不影响训练中的PowerShell校验。

## 4. 完整结果
待正式训练、60轮完整曲线、独立重放、11项比较及训练拟合诊断完成。不得用预检或局部epoch替代结果。RF/XGB仅读取冻结预测；完整测试集关闭。

## 5. 机制诊断
队列完成后检查学习曲线、梯度裁剪、正值低估/高估、显式零统一分母贡献、完整训练拟合、分轴/来源与典型案例。补全、name-only及两种可见率检索全部持续记录。

## 6. 因果分析
目前只有方法实现和功能检查事实，无本候选的性能解释。结果需区分观察、受控学习率干预的证据以及优化路径/单种子/验证复用等未排除解释。

## 7. 版本决定
待完整单种子筛选。三个预注册门槛全部通过后才另行登记20260923/24；登记和运行不等于接受。既有256维失败版本和中英文报告保持可追溯。

## 8. 下一轮问题
由本轮结果决定。完成后新增独立中文与英文报告快照，包含数据来源、各迭代、RF/XGB具体方法与所有比较；不覆盖此前已复核报告。
