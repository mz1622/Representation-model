# V9-R9 / PDF容量组合：预检完成，正式结果待完成

这是第6/12个候选，父控制仍为同种子MAE、source weight1。[计划](PLAN.md)和[配置](config.json)已冻结；上一项完整版本见[来源校准消融报告](../source0_v1/README.md)。这里尚无本候选的正式验证结论。

## 1. 研究问题与假设

原PDF的256维、8头、FF1024组合是否能改善当前192维、6头、FF768的条件营养表示？可能改善拟合，也可能增加优化或泛化代价。本轮检验容量组合，不将多个参数共同变化称为纯宽度效应。

## 2. 父版本与控制

只改变d_model、n_heads、feedforward_dim三项结构参数。保留MAE、source weight1、3层、dropout0.15、rank16、来源L2、全局轴/来源权重、学习率3e-4、batch64、clip1与60轮日程。数据、名称缓存、划分、查询、任务、指标和RF/XGB不变。种子均为20260922，但跨宽度初值和RNG消耗不同；每头维度仍为32。

## 3. 可复现信息

配置SHA256：`da41444ffd447f70de2a74303591f29708499028f607a620a72d1058052f5100`，绑定17项实现/证据及原67项冻结输入。256模型参数总数2696626，requires_grad参数2648409；初值SHA256为 `6c0443083d8752e5fca82e5dfeb648c93bf03fb48d2ac6e0443243421f41c7bb`。环境沿用Windows、Python3.10.19、PyTorch2.7.1+cu128和RTX5070Ti16GB。

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_research_transformer_r9_capacity256.py tests/test_research_transformer_r9.py
.venv/Scripts/python.exe -X utf8 scripts/verify_foodnutrigpt_r9_capacity256_functional.py --output-dir reports/v9_r9_capacity256_functional_v1
.venv/Scripts/python.exe -X utf8 scripts/estimate_foodnutrigpt_r9_capacity256_runtime.py --output-dir reports/v9_r9_capacity256_runtime_v1
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_capacity256.py --candidate tf256_mae_lr3e4_60 --output-dir output/v9_r9_methods/tf256_mae_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_capacity256.py --run output/v9_r9_methods/tf256_mae_lr3e4_60 --output-dir reports/v9_r9_capacity256_lr3e4_60_audit_v1
```

前三项已完成；正式训练已于2026-09-29北京时间10:04启动，代码提交 `f19d545c33272906b334db3ed63351f69707892e`，训练及审计结果仍待完成。产物目录拒绝覆盖。[启动回执](launch.json)保存实际进程身份和心跳工具回执；267分钟定时回访已启用，预计北京时间14:32左右检查。这个时间是估算，不是完成承诺。

## 4. 已有结果与失败记录

12项合成/共享检查通过（3.02秒），真实训练行[功能核验](../../../../reports/v9_r9_capacity256_functional_v1/verification.json)通过。32任务50步损失从0.156038降至0.057187，仅证明可学习性；不用于与192模型的验证性能比较。现有来源校准表在训练后非零，基础前向仍与来源索引无关。预检无数值失败，只有既有Torch norm_first/nested_tensor提示。

[训练行测速](../../../../reports/v9_r9_capacity256_runtime_v1/summary.json)中192/256每步中位耗时分别0.031617/0.042097秒，比值1.331463。按预登记公式估计完整运行15402.150秒（约4小时17分钟），首次回访为启动后267分钟。测速优化步全部丢弃；没有评价本候选的验证集指标。

## 5. 机制与功能核验

已验证：原192控制初值可复现；256构造器与显式修改配置的独立构造在初值、同宽构造RNG、前向和损失上精确匹配；配置仅三项不同；数值训练内循环AST不变；隐藏标签、target/positive标志和来源不影响基础前向；未观测轴可查询；8个训练名称×187轴有限；初始及学习后检查点public loader重载精确。正式训练另做小批检查并恢复自身初值与CPU/CUDA RNG。

## 6. 因果边界

容量组合共同改变宽度、头数、前馈规模、参数数目及随机轨迹。即使未来改善，也不能只归因于某一项。相同整数种子不是跨宽度相同初始化；功能核验和测速不是性能或泛化证据。数据标签、别名、来源转载及历史验证反复使用的局限仍然存在。

## 7. 当前决定

允许进入预登记的单种子筛选，尚未接受为升级。相对同种子MAE的主误差点改善、改善区间下界>0、旧log退步≤2%必须全部成立；通过后才另行登记23/24复验。整体RF目标仍需三种子确认，不能靠单种子点分数完成。

## 8. 后续与测试状态

完成训练后进行专用审计、完整重放、11项固定比较和机制分析，再补齐本README与中英文综合报告。任何结果均保留，原始值和逐条预测只在本地数据目录。测试继续关闭，RF/XGB不重拟合；尚未证明来源留出或少样本迁移能力。

训练进行期间已完成[容量专用后处理准备](POSTPROCESS.md)：新增分析、完整训练拟合推理、逐轴/案例核验与双语报告入口；32项合成分析测试通过，六个模块编译通过。三个就绪检查均正确等待完整实验及审计，17项登记输入和67项冻结输入的哈希仍一致。本轮中途曲线和验证指标未查看，也未提前生成结果报告；上述仅是准备完成，正式结论仍待回访处理。
