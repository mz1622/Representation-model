# FoodNutrition Agent Guide

## Purpose and Current State

This repository develops a source-free representation model for food
composition. The model receives standardized food text and optionally observed
composition values, then reconstructs masked composition axes.

The current immutable baseline is
`global_foodnutrigpt_v8_single_stage_v2_complete_test`. V8/V2 is a
**source-native baseline**, not the final scientific benchmark. It preserves raw
source observations and places a source embedding in the training Transformer.
Do not modify its data, split, masks, normalizer, checkpoints, or reported
results. New work must use a separately versioned V9 dataset and output folder.

## Current Data Composition

| Item | Count / policy |
| --- | --- |
| Source-native food profiles | 76,915 |
| Direct-mass source-native tokens | 2,259,178 |
| Integrated data sources | 25 |
| Active axes | 252 |
| Masked-loss axes | 187: 142 nutrition and 45 food-metabolome |
| Context-only axes | 65 |
| Complete test | 659 source-native profiles covering all 187 loss axes |
| Unit | `g/100 g edible portion, fresh weight` after legal mass conversion |
| Missing value | Absent token; never converted to zero |
| Explicit zero | Observed value and valid target |

The source-native corpus includes AFCD, ANFOOD, Bangladesh FCT, BioFoodComp,
BLS, CIQUAL, CNF, CoFID, EFSA EU FCDB, FNDDS, FooDB, Frida, Lesotho FCT,
MEXT Japan, Norway, PhyFoodComp, Cambodia/Indonesia/Laos/Thailand/Vietnam
SMILING tables, Swiss FCDB, USDA SR Legacy, WAFCT, and USDA Foundation. USDA
Foundation is held out from V8 training.

## Data and Audit Rules

1. Never fill missing composition values with zero.
2. Preserve explicit zero as an observed target.
3. Do not silently convert censored, range-only, dry-weight, activity-equivalent,
   molar, percentage, or incompatible-definition records into mass labels.
4. Keep raw source observations and provenance in the data layer. A source may
   be used for training-only calibration, but it is not food semantics.
5. Exact-name groups are candidate duplicate groups, not confirmed food-identity
   merges. Do not make fuzzy matching a merge decision.
6. Do not automatically pool cross-source values. Identical names can hide food
   facets, samples, regions, analytical definitions, or source expression.
7. All same exact-name candidate records must stay in one active partition. V8
   has zero active train/validation/test exact-name overlap; 122 Foundation
   overlap groups are explicitly excluded from the benchmark.
8. Fit transformations using training data only. Test values must never influence
   normalization, model selection, source calibration, masks, or hyperparameters.

## Source-Aware V9 Design

V9 retains source-native measurements but does not require manual cross-source
pooling. Its design is:

```text
food text + visible composition tokens -> source-free Transformer -> base prediction
training-only source × axis residual -> source-specific observed label
inference / validation / test -> base prediction only
```

For food `f`, axis `a`, and source `s`:

```text
known-source training prediction = base(f, a) + residual(s, a)
source-free inference prediction = base(f, a)
```

The residual is zero-centred per axis and regularized. Source must not be
concatenated to the Transformer encoder sequence in V9. Repeated observations
for one exact-name candidate and axis must have equal total loss weight per
independent source; duplicate rows must not multiply supervision.

## Training Plan and Rationale

Use one single-stage run with a shared Transformer and an all-axis
macro-averaged hurdle loss. Nutrient-versus-metabolome two-stage training is not
required.

The training objective has three reported paths:

1. **Known-source calibration:** fit raw observed labels with the training-only
   source-axis residual.
2. **Partial-profile reconstruction:** mask a component family while retaining
   visible context. This is the primary composition task.
3. **Text-only zero shot:** mask all numeric context and predict from food text.
   This is an auxiliary task, not removal of the text embedding.

Inference contains food text plus visible axis/value tokens only. Tune the
partial-profile/text-only balance on validation. The complete test remains closed
until one validation-selected checkpoint is final.

## Loss, Metrics, and Fine-Tuning

Amount targets use train-only robust scaling of `log1p(g/100 g)`. Use a masked
hurdle objective: BCE for positive-versus-explicit-zero plus weighted SmoothL1
amount loss for positive observations. Macro-average error by axis.

Report macro-axis log MAE, log RMSE, raw `g/100 g` MAE, per-axis support, and
nutrition/food-metabolome subsets. RF, XGBoost, text kNN, food-group baseline,
and global baseline must use identical splits, masks, visible context, and
source-free inputs.

Fine-tuning tasks, after the composition benchmark is frozen:

1. Contextual composition reconstruction in a specialist or regional corpus.
2. Text-only cold-start food-profile prediction.
3. Low-observation completion with validation-tested sequential masking.
4. Separate downstream phenotype heads for diet, microbiome, or health studies.

V9 may reduce source-expression noise but is not guaranteed to outperform RF or
XGBoost. Claims require predeclared metrics and replicated uncertainty estimates.

## Audit and Release

Run before V9 work:

```bash
python scripts/audit_v8_source_aware_source_free.py
```

Use Hugging Face Datasets for a gated working release and Zenodo for a frozen,
DOI-bearing archive. GitHub contains code, manifests, checksums, and download
instructions only. Never commit raw databases, derived numeric matrices, cached
text embeddings, checkpoints, or source-licensed data without explicit review.

## Engineering Rules

- Code must run in Google Colab using relative paths or `/content` paths.
- Use `pathlib`, pandas, NumPy, PyTorch, scikit-learn, and minimal dependencies.
- Raise explicit errors for missing files, invalid values, shape mismatches, and
  stale artifacts. Do not suppress errors broadly.
- Use `apply_patch` for manual edits. Keep changes surgical and versioned.
- Every experiment records data/split hashes, seed, configuration, validation
  metrics, test-open status, and the causal change relative to its parent.
- Never overwrite artifacts; create a new versioned directory.
