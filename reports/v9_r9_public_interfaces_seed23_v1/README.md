# R9 公共接口核验 / Public interface check

已完成固定配方种子20260923的公共接口核验，12项检查通过，实际检查耗时1.282秒。使用已完成的补全选点检查点、3个已缓存训练名称和1个训练营养档案，在CPU上运行2线程；没有训练、新文本编码或新的性能评价。

The completed, completion-selected seed 20260923 checkpoint passed 12 public-interface checks in 1.282 seconds. The audit used three cached training names and one training nutrition profile on CPU with two threads. It performed no training, new text encoding or new performance evaluation.

| 接口 / Interface | 已核对的行为 / Verified behavior |
| --- | --- |
| `predict(food_name, observed_profile, target_axes)` | 调用者指定的未观测监督轴得到有限预测；拒绝同时作为已观测输入的目标及65个无可靠监督的context-only输出。 / Finite predictions for caller-requested unobserved supervised axes; visible targets and context-only outputs are rejected. |
| `encode(..., modality)` | name/nutrition/fused均返回192维有限向量；名称表示不受营养上下文改变影响，营养表示不受传入名称改变影响；无效模态被拒绝。 / All three modes return finite 192-dimensional vectors; name encoding is independent of numeric context, nutrition encoding is independent of supplied name, and invalid modes are rejected. |
| `retrieve_names(observed_profile, candidate_names, top_k)` | 查询接口不接收名称；排序和分数与仅候选名称预测营养后的负MSE匹配精确一致，候选顺序及重复名称不影响结果；显式零可以查询，空营养查询被拒绝。 / No query-name argument; exact ranking/score agreement with name-predicted-profile negative-MSE matching, candidate-order and duplicate-name invariance, explicit-zero retention, and rejection of empty nutrition queries. |

检索分数明确标为`negative_scaled_log_mse; not probability`，不解释为概率。公共接口允许至少一个已观测营养轴，本研究正式检索面板另外要求至少三个；二者不能混为同一覆盖范围。

Retrieval scores explicitly identify negative scaled-log MSE, not probability. The public interface accepts at least one observed nutrition axis; the formal evaluation panel separately requires at least three. These have different coverage.

当前Transformer的`encode`是轴token隐藏状态的均值，用作表征探针，未训练成对比学习嵌入。本次只证明一个已完成检查点在这些输入上的接口行为，不能据此宣称迁移有效、检索准确或所有名称覆盖。隐藏标签、全量预测重放及完整检索排名仍由先前功能核验和独立运行审计分别提供证据。

The current Transformer exposes mean axis-token hidden states as a representation probe, not a trained contrastive embedding. This audit establishes interface behavior for one completed checkpoint and these inputs; it does not establish transfer, retrieval quality or coverage of all names. Earlier functional and completed-run audits separately cover hidden-label invariance, full prediction replay and all evaluation ranks.

```powershell
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_r9_public_interfaces.py --run output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260923 --output-dir reports/v9_r9_public_interfaces_seed23_v1
```

这是实际已执行命令的记录；现有输出目录拒绝覆盖。检查点、源码和运行身份及逐项结果见[verification.json](verification.json)。原始名称、营养值、预测与表征向量不写入此目录；当前训练、数据、冻结树与测试状态没有改变。

This records the executed command; existing outputs cannot be overwritten. Checkpoint/source/run identities and individual checks appear in [verification.json](verification.json). This directory contains no raw names, composition values, predictions or representation vectors. Active training, data, frozen trees and test status are unchanged.
