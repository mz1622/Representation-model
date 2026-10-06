# 从 v18 出发的 Attention 结构实验报告（v34–v36）

## 结论

本轮三个 attention 候选均完成固定 20 epochs 训练，但都没有超过 v18。因此继续保留 **v18**，不采用 v34、v35 或 v36，也不使用 v33 的事后校准。

三个候选中 v36 最接近 v18，但其 all187 scaled-log MAE 为 `0.312203`，仍比 v18 的 `0.309882` 高 `0.002321`。新增机制都学到了明显的非零信号，所以负结果不是因为新增分支没有被训练，而是这些交互方式没有转化为更好的验证集泛化。

## 固定实验协议

三个候选均以 v18 为直接父版本并独立从头训练，不互相叠加：

- 输入不含 foodname、food ID、source 或 family label；目标值继续由 masked target axis token 补全。
- 保留完整双向 self-attention、目标 token 之间的交互，以及每层、每 head 独立的完整有序 axis-pair bias。
- 主干固定为 3 层、hidden 192、6 heads、ReGLU FFN 256、dropout 0.15。
- 优化固定为 AdamW、学习率 `3e-4`、batch 128、20 epochs、seed `20261005`。
- 数据、187 轴面板、训练任务、损失、归一化、训练/验证划分与 v18 相同。
- 每次训练完整运行 20 轮，以 all187 macro-axis scaled-log MAE 选择 checkpoint；没有提前停止或根据中间结果调参。
- 完整测试集保持关闭。

## 三种结构

### v34：目标 token 专用 Query residual

每层在 v18 的共享 Query 上增加只作用于目标 token 的投影：

```text
Q_i = Wq(x_i) + I(target_i) × ΔWq_target(x_i)
```

`ΔWq_target` 从全零开始，因此初始函数和 v18 相同。Key、Value、softmax、pair bias 以及全部 attention 通路均未改变。

结果说明：目标 token 获得专用“提问方式”没有帮助。它们原本已经通过 axis embedding、mask value 和 axis-pair bias 获得明确身份；额外 Query 自由度更可能削弱共享投影提供的正则化。

### v35：内容与 axis 身份的解耦 Q/K

在 v18 的 content-content score 和完整 pair bias 之外，增加 content-to-axis 与 axis-to-content 两个交叉项：

```text
score(i,j) =
    Qc(x_i) · Kc(x_j)
  + Qc(x_i) · Ka(axis_j)
  + Qa(axis_i) · Kc(x_j)
  + pair_bias(axis_i, axis_j)
```

`Qa` 和 `Ka` 从全零开始；context token 的 axis 特征为零。设计借鉴 [DeBERTa disentangled attention](https://arxiv.org/abs/2006.03654)，但把位置身份替换为成分 axis 身份。

结果说明：该结构学到了很强的额外交互，但验证误差反而最高。v18 已经在 token 输入中包含 axis embedding，并在 score 中加入完整 axis-pair bias；继续加入两种显式 axis/content 交叉项形成了冗余约束。

### v36：Relation-aware Value

保持 v18 的 attention score 不变，在 Value 消息中加入有序 axis 关系：

```text
output_i = Σ_j attention(i,j) × [V(x_j) + R(axis_i, axis_j)]
R(i,j) = [1 + U(i)] ⊙ Vrelation(j)
```

`U` 和 `Vrelation` 均从零开始。该设计依据 [Shaw 等人的 relation-aware self-attention](https://arxiv.org/abs/1803.02155)，并用因子化形式避免为全部有序 axis 对保存完整 Value 向量。

结果说明：这是本轮最接近 v18 的方案，说明“关系决定传递什么信息”比继续修改 Query/Key 更有潜力；但在当前 20 轮、单 seed 协议下仍没有形成净收益。

## all187 结果

| 模型 | 唯一结构变化 | 最佳 epoch | 参数量 | 相对 v18 参数 | all187 MAE | 相对 v18 | 训练时间 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **v18** | 完整 axis-pair score bias | 20 | 2,057,125 | — | **0.309882** | — | 47分26秒 |
| v34 | 目标 token 专用 Query residual | 18 | 2,168,293 | +111,168（+5.40%） | 0.315622 | +0.005740（误差 +1.85%） | 48分27秒 |
| v35 | content-axis 双向 Q/K 交叉项 | 16 | 2,278,309 | +221,184（+10.75%） | 0.319729 | +0.009847（误差 +3.18%） | 53分33秒 |
| v36 | 有序 axis relation-aware Value | 18 | 2,348,581 | +291,456（+14.17%） | 0.312203 | +0.002321（误差 +0.75%） | 1小时15分16秒 |

这里数值越低越好。三个候选都增加了参数和训练成本，但没有降低 all187 MAE。

## 新机制是否真正工作

最佳 checkpoint 中新增分支均明显偏离零初始化：

| 模型 | 诊断量 | 结果 |
| --- | --- | ---: |
| v34 | target Query residual 参数 RMS | 0.058979 |
| v35 | axis Q/K projection RMS | 1.004920 |
| v35 | 固定 128-task batch 的三层 cross-logit RMS | 2.653822 / 2.175174 / 2.329318 |
| v35 | 固定 batch 的平均 cross-logit RMS | 2.386105 |
| v36 | factorized relation Value 有效 RMS | 0.088338 |
| v36 | relation Query / Value 参数 RMS | 0.143746 / 0.086214 |

v35 的交叉 logit 已达到不可忽略的量级，但它带来的不是信息不足修复，而是更差的泛化。v36 的关系 Value 也确实参与了消息传递，因此 `0.312203` 应视为有效但不够好的结构结果。

## 工程验证与完整性

- 新增 14 项结构测试全部通过：零初始化时与 v18 输出完全一致、可见 token 排列不变、padding 隔离、checkpoint 往返一致、新增参数量符合预注册设计、新增分支能够获得梯度。
- 排除两个缺少历史外部 fixture 的既有测试文件后，仓库其余测试为 `702 passed`。完整测试集中的另外 10 个 setup error 和 1 个 failure 均来自本地缺少 `composition_classification_v5_1.json` 或 `prediction_axis_registry.csv`，与本轮代码无关。
- 三个正式 checkpoint 重新加载后的验证预测最大绝对差均为 `0.0`。
- 三个 manifest 与 v18 具有相同的数据哈希 `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`、相同 187 轴列表和固定训练配置。
- 三个版本均记录 `food_name_model_input=false`、`source_model_input=false`、`family_labels_used_by_model=false`、`distillation=false`、`posthoc_calibration=false`、`deployed_model_count=1`、`complete_test_opened=false`。

## 最终决定

当前继续以 v18 作为无 foodname、单模型 Transformer 的保留版本。本轮结果不支持把目标专用 Query、DeBERTa 式 axis/content QK 解耦或 factorized relation-aware Value 合入 v18。

该结论只对应预先指定的单 seed 内部验证实验。按照本轮约定，不自动追加 normalized sigmoid attention 或 differential attention。

## 产物

- `output/no_foodname_v34_target_query_unified187_20ep_20261007/`
- `output/no_foodname_v35_disentangled_axis_content_unified187_20ep_20261007/`
- `output/no_foodname_v36_relation_value_unified187_20ep_20261007/`

每个目录均包含 `transformer.pt`、`history.csv`、`validation_predictions.parquet`、`axis_metrics.csv`、`metrics.json` 和 `manifest.json`。
