# FoodNutriGPT V9 Source-Calibrated Validation Run

## Scope

This V9 baseline reuses the immutable V8/V2 corpus and grouped split rather
than rebuilding or pooling any source-native values. The checkpoint was selected
on validation only; its complete-test result was run once afterwards and is a
baseline reference, not an input to later model selection.

| Item | Value |
| --- | ---: |
| Source-native profiles | 76,915 |
| Direct-mass tokens | 2,259,178 |
| Active axes | 252 |
| Masked-loss axes | 187 (142 nutrition, 45 food-metabolome) |
| Source-free validation candidate cells | 268,842 |
| Complete test | 659 profiles, 19,690 profile-axis masks, 187 axes |

## V9 model

The encoder sequence is `[CLS], [TEXT], [AXIS_ID + VALUE]...`. It has no source
token and no source embedding. A zero-centred, regularized source-by-axis
residual is used only while fitting source-native labels:

```text
known-source training label fit = source-free base prediction + source-axis residual
validation/inference prediction = source-free base prediction
```

Each exact-name-candidate / axis / independent-source cell has equal total loss
weight. Duplicate observations within a source-native profile divide that
source's weight rather than adding supervision. The shared model uses an
all-axis macro-averaged masked hurdle loss: BCE for explicit-zero versus
positive values, plus SmoothL1 loss for positive `log1p(g/100 g)` values scaled
from training rows only.

The selected checkpoint was trained for eight epochs, with a 192-dimensional,
three-layer Transformer, rank-16 axis calibration heads, batch size 64,
learning rate `1e-4` cosine decay, and no text-only examples. Text-only
cold-start prediction remains a separate downstream validation task; the
primary run evaluates partial-profile reconstruction.

## Validation protocol

For every validation food profile and every eligible observed chemical family,
the entire family is masked. This prevents direct aggregate/form shortcuts. The
source-free base head predicts the missing cells. Scoring first collapses
duplicate measurements within a source-native profile, then gives each source
equal weight within an exact-name-candidate / axis cell. Source keys do not
enter either model input or released candidate-cell metric table.

The RF baseline receives the same frozen MiniLM text encoding (32 train-fitted
PCA components), visible normalized composition values, and observedness
indicators. It receives no source feature. It hides the exact same chemical
family for each target axis. This run uses 20 trees, depth 12, 0.35 feature
subsampling, minimum leaf size 5, and a deterministic cap of 5,000 train
profiles per axis; all sparse axes retain their complete training support.

## Validation Results

| Method | Macro log-MAE | Macro log-RMSE | Macro raw MAE (g/100 g) |
| --- | ---: | ---: | ---: |
| FoodNutriGPT V9, source-free base | 0.07572 | 0.22958 | 0.46050 |
| Random Forest, source-free | **0.06258** | **0.18816** | **0.33924** |

| Subset | Method | Axes | Macro log-MAE | Macro log-RMSE | Macro raw MAE (g/100 g) |
| --- | --- | ---: | ---: | ---: | ---: |
| Nutrition | FoodNutriGPT V9 | 142 | 0.08650 | 0.25074 | 0.58331 |
| Nutrition | Random Forest | 142 | **0.06962** | **0.20247** | **0.42489** |
| Food-metabolome | FoodNutriGPT V9 | 45 | 0.04172 | 0.14368 | 0.07295 |
| Food-metabolome | Random Forest | 45 | **0.04036** | **0.13328** | **0.06897** |

RF is lower by 17.5% in macro log-MAE relative to V9
(`(0.07572 - 0.06258) / 0.07572`). This is a validation result only, not a
claim that would support later model selection. The experiment shows that
removing source from the encoder and equalizing source-level supervision is
feasible, but does not by itself surpass a strong tabular baseline.

## One-Time Complete-Test Baseline

The validation-selected checkpoint was then evaluated once. Both methods use
the same 19,690 profile-axis jobs and 16,017 source-free candidate food-axis
cells, cover all 187 axes, receive no source input, and use the same equal-source
offline aggregation.

| Method | Macro log-MAE | Macro log-RMSE | Macro raw MAE (g/100 g) |
| --- | ---: | ---: | ---: |
| FoodNutriGPT V9, source-free base | 0.09263 | 0.26664 | 0.60152 |
| Random Forest, source-free | **0.07987** | **0.22497** | **0.46760** |

On nutrition axes, V9 has log-MAE `0.10853` and RF `0.09201`; on food-metabolome
axes, V9 has `0.04245` and RF `0.04158`. RF is lower by 13.8% relative to V9 on
the overall complete-test macro log-MAE. No subsequent candidate may use this
test result for selection; all optimization below returns to validation only.

## Reproduction

```bash
python scripts/train_global_foodnutrigpt_v9_source_calibrated.py --skip-test
python scripts/evaluate_global_foodnutrigpt_v9_validation.py
python scripts/evaluate_global_foodnutrigpt_v9_rf_baseline.py \
  --output-dir output/global_foodnutrigpt_v9_source_calibrated_rf20_max5000_validation \
  --trees 20 --max-train-rows-per-axis 5000
python scripts/evaluate_global_foodnutrigpt_v9_complete_test.py
python scripts/evaluate_global_foodnutrigpt_v9_complete_test_rf.py
```

Generated artifacts are intentionally excluded from Git because they include
source-licensed values, text caches, and checkpoints. The commands above write
them under `output/` without overwriting V8 artifacts.
