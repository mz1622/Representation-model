# RF and XGBoost trained only on the paper DAE dataset

## Question

This experiment trains the project's numerical RF and XGBoost implementations
only on the 2022 DAE paper's released food-composition dataset, then evaluates
them zero-shot on the current dependency-masked completion task. It separates
the effect of the paper training corpus from the effect of the published DAE
architecture.

The paper source is Gjorshoska et al., [Missing value imputation in food
composition data with denoising autoencoders](https://doi.org/10.1016/j.jfca.2022.104638),
and the released data/code are from
[MIDA-in-FCDBs](https://github.com/gjorshoskaivana/MIDA-in-FCDBs).

## Frozen protocol

- The same 10,484 paper training foods used in the DAE reproduction are used.
  The 2,621 paper holdout foods are not used.
- The 198 paper nutrients are mapped by strict chemical identity and unit
  compatibility into the current 252-axis grid. This yields 130 mapped axes,
  including 127 supervised targets.
- Each model receives the project's normal 504-dimensional input: 252
  normalized values plus 252 visibility flags.
- Unmapped current axes are always hidden. Paper `G`, `MG`, and `UG` values are
  converted into current `g/100g` units.
- Each axis is normalized with the median of its strictly positive values in
  the paper training rows. Current training values are not used to fit model
  normalization.
- For each target axis, the current dependency closure is masked in the paper
  training profiles. This trains the tree models for the current task rather
  than for the paper's random-cell MCAR task.
- Current validation and complete-test labels are used only for evaluation.
  Current training labels are not supplied to either model. The current
  training-derived metric scale is used only by the frozen evaluator, as it is
  for every method in the comparison.
- No food name is provided because the project's RF and XGBoost baselines are
  numerical-composition-only methods.
- No validation selection, fine-tuning, calibration, or checkpoint reuse is
  performed.

### RF

One independent regressor is fitted per target axis: 200 trees,
`min_samples_leaf=2`, squared-error splits, bootstrap sampling, 50% random
feature candidates per split, and seed `20261005 + axis_index`.

### XGBoost

One independent regressor is fitted per target axis: 300 trees, depth 6,
learning rate 0.05, `min_child_weight=5`, row and column subsampling 0.8,
`reg_lambda=1`, histogram splits, and seed `20261005 + axis_index`.

## Results

All methods below are scored on the same 127 common target axes. Validation
contains 300,380 profile-axis jobs; complete test contains 18,430 jobs from the
fixed 659-profile panel.

### Source-only methods

| Method | Partition | Scaled-log MAE | Scaled-log RMSE | Raw MAE |
|---|---|---:|---:|---:|
| **Paper-data RF** | Validation | **0.394228** | **0.642196** | **0.830909** |
| Paper-data XGBoost | Validation | 0.405171 | 0.662875 | 0.878031 |
| Paper DAE, no name | Validation | 0.546920 | 0.794806 | 1.059076 |
| Paper DAE + FoodText | Validation | 0.552384 | 0.810123 | 1.090366 |
| **Paper-data RF** | Complete test | **0.397546** | **0.646479** | **0.811568** |
| Paper-data XGBoost | Complete test | 0.399866 | 0.663625 | 0.817879 |
| Paper DAE, no name | Complete test | 0.616934 | 0.877443 | 1.290308 |
| Paper DAE + FoodText | Complete test | 0.619005 | 0.901362 | 1.314242 |

RF is the best paper-data-only method on all three metrics. On complete test,
its primary error is 0.58% lower than paper-data XGBoost and 35.56% lower than
the paper DAE. XGBoost is 35.19% lower than the paper DAE. Thus the published
dataset contains transferable composition relationships, and task-aligned
tree regressors extract them considerably better than the reproduced DAE.
Paper-data RF beats the paper DAE on 117 of the 127 individual axes.

### Complete-test comparison with current-data models

| Method | Training corpus | Scaled-log MAE | Scaled-log RMSE | Raw MAE |
|---|---|---:|---:|---:|
| **V5+CCE** | Current corpus | **0.196371** | 0.353243 | 0.372027 |
| XGBoost | Current corpus | 0.214957 | **0.342926** | **0.312990** |
| RF | Current corpus | 0.217158 | 0.346190 | 0.331422 |
| KNN | Current corpus | 0.319885 | 0.490872 | 0.581787 |
| Paper-data RF | Paper corpus only | 0.397546 | 0.646479 | 0.811568 |
| Paper-data XGBoost | Paper corpus only | 0.399866 | 0.663625 | 0.817879 |
| Paper DAE, no name | Paper corpus only | 0.616934 | 0.877443 | 1.290308 |
| Paper DAE + FoodText | Paper corpus only | 0.619005 | 0.901362 | 1.314242 |

Training-data alignment remains more important than the choice among these
three source-only architectures. Current-data RF lowers primary error by 45.38%
relative to paper-data RF, and current-data XGBoost lowers it by 46.24%
relative to paper-data XGBoost. V5+CCE is 50.60% lower than the strongest
paper-data-only method. Current-data RF beats paper-data RF on 109 of 127 axes;
current-data XGBoost beats its paper-data counterpart on 110 of 127 axes.

The source-only tree comparison is task-aligned but is not a pure architecture
ablation against DAE: RF/XGBoost are explicitly trained with the current
dependency mask and per-axis objective, whereas the DAE preserves the paper's
random-corruption reconstruction objective. The defensible conclusion is that
the original DAE result is limited by both its training objective/architecture
and cross-database domain shift.

## Versioned artifacts

- `artifacts/manifest.json`: hashes, leakage controls, hyperparameters, and
  summary results.
- `artifacts/paper_tree_metrics.csv`: RF/XGBoost validation and test metrics.
- `artifacts/source_only_common_127_axis_metrics.csv`: direct source-only
  comparison with both DAE variants.
- `artifacts/all_methods_common_127_axis_metrics.csv`: complete matched-panel
  comparison with current-data baselines and V5+CCE.
- `artifacts/paper_training_scales.csv`: paper-only normalization scales.
- `artifacts/paper_to_current_axis_crosswalk.csv`: accepted/rejected mappings.
- `artifacts/*_fit_support.csv` and `artifacts/*_axis_metrics.csv`: per-axis
  support and aggregate error evidence.

No model checkpoint, raw dataset, profile-level prediction, or food identifier
table is committed.
