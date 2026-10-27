# Foundation-model evidence bundle / 基础模型证据包

This directory publishes the conclusions and aggregate analysis data behind the bilingual reports. It contains no model weights, checkpoints, raw food-composition rows, food identifiers, or profile-level predictions.

本目录同步中英文结论及其聚合分析数据，不含模型权重、checkpoint、原始食品数据、食品 ID 或逐样本预测。

## Reports / 报告

- [中文 PDF](food_foundation_evidence_zh.pdf)
- [English PDF](food_foundation_evidence_en.pdf)
- [中文 Markdown](report_zh.md)
- [English Markdown](report_en.md)

## Main conclusions / 主要结论

1. Without food names or source identifiers, V5+CCE transfers to the independent Taiwan TFDA database and has lower zero-shot scaled-log MAE than RF, XGBoost, and KNN.
2. Pretraining improves external-data sample efficiency: at 8, 32, and 128 adaptation profiles, the pretrained model outperforms the same architecture trained from scratch.
3. Very small adaptation is not automatically useful. Eight-shot fine-tuning is worse than zero-shot; the clear gain appears at 128 shots.
4. With sparse visible inputs, V5+CCE leads from 10 to 40 visible components, while KNN is best at five.
5. UMAP and sensitivity analyses are descriptive. They show learned cross-axis structure, but also source and missingness confounding; they do not establish causality.
6. CCE improves over a matched continuation control, but much of the V5-to-CCE gain also comes from extra continuation training. Current evidence does not support attributing the whole gain to CCE.

## Aggregate tables / 聚合表格

| File | Granularity |
|---|---|
| `complete_test_model_metrics.csv` | One row per model on the 659-profile, 187-axis complete-test panel |
| `cce_vs_continuation_validation_test.csv` | V5, continuation-only, and V5+CCE validation/test metrics |
| `taiwan_zero_shot_metrics.csv` | Model-level Taiwan zero-shot metrics and confidence intervals |
| `taiwan_zero_shot_paired_bootstrap.csv` | Aggregate paired bootstrap model differences |
| `taiwan_fewshot_summary.csv` | Mean and SD across three adaptation seeds |
| `taiwan_fewshot_runs.csv` | One row per training run; no weights or predictions |
| `sparse_observation_summary.csv` | Metrics by visible-component budget |
| `routine_panel_summary.csv` | Aggregate routine-panel comparison |
| `umap_category_metrics.csv` | Representation-level category metrics |
| `umap_robustness.csv` | Aggregate repeated-subsample UMAP checks |
| `sensitivity_clean_accuracy.csv` | Clean full-validation metrics |
| `sensitivity_key_results.csv` | Aggregate perturbation results |
| `sensitivity_verdicts.csv` | Aggregate paired-bootstrap sensitivity comparisons |

Figures are in `figures/`. The CSV files contain only derived statistics and experiment-level metadata.

## Provenance

The bundle consolidates completed artifacts from:

- `v9_representation_p0_p1_20261020_v1`
- `taiwan_tfda_external_completion_v1_20261024`
- `taiwan_tfda_fewshot_completion_v1_20261026`
- `v9_foundation_124_20261021_v1`
- `v9_axis_sensitivity_full_validation_20261024_v1`
- `v9_sensitivity_baseline_comparison_full_validation_20261025_v1`
- `no_foodname_complete_test_profile_macro_v1_20261007`
- `no_foodname_v42_v5_cce_vs_continuation_summary_20261014`

The Taiwan few-shot experiment is the only newly trained experiment in this report bundle. The reports document metrics, protocols, limitations, and the interpretation of each evidence source.
