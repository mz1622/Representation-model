# V9-R9 / 轴×数值残差：预检通过，正式结果待完成

第20轮阶段核验已完成：[快照凭据](../../../../reports/v9_r9_axisvalue_epoch20_snapshot_v1/verification.json)。前20轮的任务顺序、目标数及学习率与MAE控制一致；阶段最优选中第18轮，快照在CPU上可读取且全部状态有限，新残差已有非零学习参数。这是局部状态和选点核验，尚未执行完整预测重放，不构成最终成绩或接受决定。原60轮训练继续运行。

## 1. 研究问题与预先假设

共享数值编码器加轴标识可能限制轴与数值的交互。只增加一个零初始化的轴专属线性斜率，检验同一Transformer预算下的补全表现。完整[预登记](PLAN.md)在任何候选训练前保存；这是第4/12个方法配方，不假定有效。

## 2. 父版本与改动

父模型是同种子20260922的MAE控制，保留192维、3层、6头、FF768、dropout0.15、rank16、MAE、来源校准、lr3e-4、AdamW和60轮调度。仅数值token由 `e_axis+g(t)` 改为 `e_axis+g(t)+t*r_axis`，隐藏值不贡献残差。新增48,384参数；总参数1,596,978，可训练1,564,313。所有旧参数初值及随机状态一致。

## 3. 可复现信息

[配置](config.json)绑定13项实现/父运行/决定证据及67项冻结数据与基准输入。配置SHA256为`1a01380bb2f6ae9b711d91da53287ebf15129550c1052b4e7a8b09b94a74e32c`。沿用Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB，源码提交和实际成本由正式run_manifest记录。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_research_transformer_r9_axisvalue.py tests/test_research_transformer_r9.py
.venv/Scripts/python.exe -X utf8 scripts/verify_foodnutrigpt_r9_axisvalue_functional.py --output-dir reports/v9_r9_axisvalue_functional_v1
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_axisvalue.py --candidate tf192_mae_axisvalue_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_axisvalue.py --run output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60 --output-dir reports/v9_r9_axisvalue_lr3e4_60_audit_v1
```

前两项已完成；训练/重放结果仍待正式完成。各产物目录拒绝覆盖，不得重复启动。原始预测与检查点留本地忽略目录。

## 4. 完整结果

正式三任务结果尚无。11项合成/回归检查通过；真实训练行预检见[凭据](../../../../reports/v9_r9_axisvalue_functional_v1/verification.json)。32任务50步MAE目标从0.217358降至0.044068；仅验证可学习性，不是验证集成绩。正式训练重新初始化，热身后也重置所有权重和RNG。

冻结参照不改：RF补全主误差0.189031，XGB0.174487，同种子MAE0.183553；MAE三种子均值0.189184±0.006925，未确认优于RF。历史详情见[中文综合报告](../../report_snapshot_r9_mse_v1/REPORT_ZH.md)及[英文综合报告](../../report_snapshot_r9_mse_v1/REPORT_EN.md)。

## 5. 机制诊断

预检确认：零残差时旧参数/CPU随机状态、CUDA带dropout前向及RNG精确匹配；合成输入旧参数梯度精确匹配；非零残差对可见值起作用；隐藏标签和来源更改不影响前向。公共加载器在非零学习残差下重载前向和encode精确；未观测监督轴可查询，name encode不依赖营养值、nutrition encode不依赖名称。内层训练循环AST与原MAE相同。

首个pytest调用因未设置PYTHONPATH而在收集阶段失败，未执行模型；显式指定src后11项通过。正式训练尚无失败记录。必须在完整结果中补充学习曲线、正/零代价、逐轴/来源及典型案例。

## 6. 因果分析

当前仅能证明实现与预登记的单项改动一致及可学习性，尚不能证明方法收益。参数增加、特征交互结构和后续优化轨迹一起改变；改善也不能单独识别某个机制。功能预检不能证明标签可信、外部泛化或最终优于RF。

## 7. 版本决定

允许进入单种子60轮筛选，不接受为最终模型。完成后按登记规则比较同种子MAE；全部筛选通过才另外登记23/24复验。禁止根据早期结果调整数据、门槛或在跑配置。

## 8. 下一轮问题与测试使用状态

等待完整训练、独立重放、11项固定参照比较及分解，再决定接受筛选、拒绝或证据不足。测试继续关闭，RF/XGB不重训，数据/尺度/缓存/面板不改。整体Transformer优于冻结RF的目标仍未达到；完整版本将补齐双语报告。
