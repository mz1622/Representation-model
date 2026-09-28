# V9-R9 / 来源校准权重1→0：预检通过，正式结果待完成

这是第5/12个方法候选，原[计划](PLAN.md)与[配置](config.json)在训练前登记。上一项完整版本见[轴×数值报告](../axisvalue_v1/README.md)；当前没有本候选的正式验证结果。

## 1. 研究问题与假设

训练同时优化base与已知来源的校准输出，而推理只评价base。检验关闭校准目标是否有助于base补全；已有来源偏置稳定性及来源误差分解均不证明该因素的收益或损害。

## 2. 父版本与改动

父控制是20260922的 `tf192_mae_lr3e4_60`。仅source weight从1变为0，沿用 `(L_base+w*L_calibrated)/(1+w)+1e-4*mean(R_train^2)`，避免缩小base损失尺度。模型、零初始化来源表、来源L2、样本来源权重、MAE、192/3/6/FF768、dropout0.15、rank16、AdamW lr3e-4/weight decay1e-4、batch64、clip1及60轮余弦日程均保留。来源表在w=0时保持零，每轮断言；数据和RF/XGB不改。

## 3. 可复现信息

配置SHA256：`c98d64d187a37396cc51b6145843930ba43f195ef86c9bbc64c28c6c925e79fa`，绑定14项实现/父运行/完整决定证据与67项冻结输入。参数总数1,548,594，可训练1,515,929（按requires_grad计；关闭的来源表梯度为零）。初值指纹与MAE父控制一致。环境仍为Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -X utf8 -m pytest -q tests/test_research_transformer_r9_source0.py tests/test_research_transformer_r9.py
.venv/Scripts/python.exe -X utf8 scripts/verify_foodnutrigpt_r9_source0_reference.py --output-dir reports/v9_r9_source0_functional_v1
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_source0.py --candidate tf192_mae_source0_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_source0_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_source0.py --run output/v9_r9_methods/tf192_mae_source0_lr3e4_60 --output-dir reports/v9_r9_source0_lr3e4_60_audit_v1
```

前两项已完成；正式训练已于2026-09-29 06:27（北京时间）启动，代码提交为 `92f80d90b798822d39af4644d65ea9441e52d042`，训练与审计结果尚待完成。所有产物目录拒绝覆盖。正式运行估计约3小时，已启用原任务内的190分钟定时回访，预计北京时间09:38左右回访；不运行逐epoch观察器。[启动及工具回执](launch.json)记录实际进程身份和估时依据，时间是估算，不是完成承诺。实际成本由run_manifest记录。

## 4. 结果与失败记录

14项合成/共享功能测试通过。[真实训练行功能核验](../../../../reports/v9_r9_source0_functional_v1/verification.json)通过，32任务50步MAE从0.217358降至0.058560。这里的GPU功能检查使用[单独记录的确定性参考进程](../../../../reports/v9_r9_source0_functional_v1/reference_execution.json)，只证明可学习性与实现一致性，不构成验证集或优化方法优势。正式训练使用父模型原有后端设置并独立做小批检查，随后恢复初值和全部RNG。

首个普通GPU预检在共享梯度精确比较处失败，最大差异约1.82e-12，发生在优化器步和产物目录创建之前。重复同一目标也出现浮点差异；[诊断](../../../../reports/v9_r9_source0_gradient_reference_v1/summary.json)显示CPU和确定性GPU参考在各自设备内精确匹配。[失败记录](../../../../reports/v9_r9_source0_preflight_failure_v1/summary.json)保留，未放宽断言、未修改已登记训练代码或模型设置。

## 5. 机制与功能诊断

核验了独立MAE公式和梯度尺度、初值/构造RNG、零残差参考状态的共享梯度及dropout RNG、来源表在50步中的零值不变、来源索引不影响关闭校准后的损失、隐藏标签/来源不影响前向、显式零监督、未观测轴查询、8个名称×187轴有限，以及学习后检查点public loader重载精确。数值内循环AST与原MAE一致。

完整结果须补齐学习曲线、正/零共同分母分解、逐轴/来源支持和典型案例。通用独立重放之外，专用审计还核对唯一改动、全60轮与父控制任务/日程一致，以及最优和最终来源表均为零。

## 6. 因果边界

初始梯度在数学目标和确定性参考下相同，不意味着普通GPU的每次反向运算逐位一致。正式训练保留父后端，后续差异包含校准干预引起的优化轨迹变化；必须报告单种子与后续多种子的不确定性。功能预检不能证明性能改善、标签真实性或外部泛化。

## 7. 版本决定

预检允许进入单种子筛选，尚未接受为升级。相对同种子MAE的主误差点改善、改善区间下界>0、旧log退步≤2%须全部成立，才另行登记23/24种子。最终RF目标仍需三种子确认，不采用5%最低提升要求。

## 8. 下一轮问题与测试状态

等待完整训练、独立重放与11项比较，再依据完整证据决定复验或拒绝，并补齐本README和中英文综合报告。RF/XGB、数据、尺度、mask、查询与评分不改，历史测试继续关闭；整体研究目标仍未达到。
