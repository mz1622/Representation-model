# FoodNutriGPT V8/V2 Optimization Summary

## Scope and protocol

This experiment series uses the fixed `global_foodnutrigpt_v8_single_stage_v2_complete_test`
dataset and split. The complete held-out panel contains 659 food profiles, 187
prediction axes, and 19,690 profile-axis cells under the leave-one-mask-family-out
protocol. All new variants used the same seed, architecture, masks, source-unknown
validation procedure, and train/validation data. During optimization, each variant
was run with `--skip-test`; the complete test was opened once, only for the
validation-selected final checkpoint.

The primary selection metric was validation macro-axis MAE in
`log1p(g/100 g)`. Each axis contributes equally, preventing frequent nutrients
from determining model selection by themselves. Missing values remain missing;
explicit zero values remain observed labels.

## Frozen references

| Method | Complete-test macro log MAE | Macro log RMSE | Macro raw MAE (g/100 g) |
| --- | ---: | ---: | ---: |
| FoodNutriGPT, unified all-axis, 8 epochs | 0.08720 | 0.25298 | 0.54921 |
| Random Forest, per-axis | 0.07708 | 0.22185 | 0.43770 |

The RF model and eight-epoch unified Transformer were frozen before this search.

## Validation-only causal search

| Variant | One change from parent | Validation macro log MAE | Interpretation |
| --- | --- | ---: | --- |
| Unified all-axis baseline | 8-epoch schedule | 0.08329 | Frozen starting reference. |
| 20-epoch schedule | 8 to 20 epochs | 0.07341 | Strong underfitting evidence: 11.9% lower MAE. |
| No text-only examples | text-only probability 0.15 to 0 | 0.07123 | Pure-text batches competed with the fixed partial-context reconstruction task. |
| Source unknown | source dropout 0.30 to 1.0 | 0.07082 | Small primary-metric gain; raw MAE was mixed. |
| Rank 32 head | residual rank 16 to 32 | 0.07110 | Rejected: larger axis-specific residual did not improve MAE. |
| Amount weight 2 | positive amount loss 1 to 2 | 0.06635 | Largest single improvement after training duration. |
| Amount weight 3 | positive amount loss 2 to 3 | **0.06506** | Selected by the predeclared primary metric. |

The absolute training objective is not compared across amount-loss weights because
the amount term has a different coefficient. The validation reconstruction metrics,
which use a common evaluation definition, are comparable.

## One-time complete-test result

| Method | All 187 axes: log MAE | Log RMSE | Raw MAE (g/100 g) | Nutrition log MAE | Food-metabolome log MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unified Transformer baseline | 0.08720 | 0.25298 | 0.54921 | 0.10074 | 0.04447 |
| Random Forest | 0.07708 | 0.22185 | 0.43770 | 0.08655 | 0.04721 |
| **Selected FoodNutriGPT** | **0.06981** | **0.21238** | **0.41666** | **0.07747** | **0.04565** |

On the common 19,690-cell panel, the selected Transformer reduced macro log MAE
by 9.4%, log RMSE by 4.3%, and raw MAE by 4.8% relative to RF. The overall
gain is driven chiefly by the 142 nutrition axes: the selected model has lower
log MAE, log RMSE, and raw MAE for that subset. On the 45 food-metabolome axes,
it has lower log MAE (`0.04565` vs `0.04721`) and wins on 30/45 axes by that
metric, but RF retains lower log RMSE (`0.13741` vs `0.16319`) and raw MAE
(`0.07805` vs `0.08528`). Thus the metabolome result is not uniformly better.

The selected model has lower per-axis log MAE than RF on 82/142 nutrition axes
and 30/45 food-metabolome axes. It is not a claim that every axis improves.

## Final model configuration

- Shared set Transformer: dimension 192, 3 layers, 6 heads, dropout 0.15.
- Axis residual calibration rank: 16.
- 20 epochs with the same cosine schedule and early-stopping policy as the
  validation search.
- Single unified all-axis masked hurdle objective.
- Positive amount SmoothL1 term weight: 3.0; presence BCE remains weight 1.0.
- Text-only training probability: 0.0 for this partial-profile reconstruction
  objective.
- Source token is always `SOURCE_UNKNOWN` during training and evaluation.

## Limits and next direction

This is a single-seed comparison without confidence intervals, so the estimated
advantage should be confirmed using split/family-cluster bootstrap intervals or
additional seeds before being treated as a final research claim. The most direct
remaining optimization target is conditional amount reconstruction for the
metabolome axes, where the model improved typical log error but not tail-sensitive
log RMSE or raw-unit MAE. The frozen complete test must remain closed until a
separately documented follow-up protocol is defined.

## Artifacts

- `experiment_ledger.csv`: every variant, changed factor, and selection status.
- `frozen_baselines.json`: hashes and fixed baseline metrics.
- `output/global_foodnutrigpt_v8_single_stage_v2_final_selected_amount3_complete_test/`:
  one-time final checkpoint evaluation.
- `output/global_foodnutrigpt_v8_single_stage_v2_final_amount3_comparison/`:
  common-cell method and per-axis results.
