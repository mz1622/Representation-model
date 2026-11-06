# Gjorshoska et al. DAE: paper-data-only transfer baseline

## Question

This experiment asks whether the denoising autoencoder (DAE) from Gjorshoska
et al. (2022), trained only on the dataset released with that paper, transfers
to the current dependency-masked food-composition completion task. A separate
name-enabled variant is allowed to receive food descriptions. Neither variant
is trained, fine-tuned, or checkpoint-selected on the current corpus.

Sources:

- Paper: [Missing value imputation in food composition data with denoising
  autoencoders](https://doi.org/10.1016/j.jfca.2022.104638)
- Authors' implementation:
  [gjorshoskaivana/MIDA-in-FCDBs](https://github.com/gjorshoskaivana/MIDA-in-FCDBs)

## Paper-native reproduction

The reproduction uses the authors' 13,105 by 198 average-value matrix and the
public notebook's architecture and optimization settings:

- 80/20 row split; 10,484 training foods and 2,621 held-out foods;
- per-column MinMax scaling to `[0,1]`, fitted on paper training rows only;
- layer widths `1.0 -> 0.7 -> 0.5 -> 0.2 -> 0.5 -> 0.7 -> 1.0`;
- Tanh after the 0.7 and 0.5 layers in the encoder and decoder;
- input dropout 0.2, full-vector MSE, SGD with learning rate 0.01;
- batch size 1, 200 epochs, fixed final checkpoint, seed 20261005;
- 20% MCAR cells on the paper holdout and paper-training means as fill values.

The paper metric sums the masked-cell RMSE computed separately for all 198
MinMax-scaled nutrients. The one-seed reproduction obtains `RMSE_sum = 5.157`.
The paper reports a ten-run mean of `5.26`, a difference of -1.97%. This is
close enough to verify the main architecture, preprocessing, and evaluation
path; it is not presented as an exact reproduction of the paper's ten random
runs.

## Food-name variant

The original paper also used frozen InferSent/GloVe sentence embeddings reduced
to 50 PCA components. The public embedding-generation notebook inserts
`fdc_id` before PCA, and the released 50-dimensional files consequently have
their leading components dominated by that identifier. The fitted PCA object
needed to transform names from a new dataset was not released.

For a usable external name-enabled comparison, the adapted variant uses frozen
`sentence-transformers/all-MiniLM-L6-v2` embeddings, L2 normalization, and 50
PCA components fitted only on the paper food names. The DAE is then trained only
on the paper rows. At evaluation, current food names are transformed by the
same frozen encoder and paper-fitted PCA. This is labelled **Paper-data DAE +
FoodText** and is not claimed to be an exact InferSent reproduction. Its
paper-native `RMSE_sum` is 5.373. For context, the paper reports 5.57 for its
name-enabled 198-nutrient experiment.

## Transfer protocol

The paper's 198 nutrients are mapped to the current axis vocabulary only when
the nutrient identity and unit are strictly compatible. No fuzzy chemical
merging is allowed. The resulting common panel contains 130 axes, of which 127
are supervised target axes. All reported current-task methods are rescored on
exactly these 127 axes and the same profile-axis jobs.

For each target job, the current dependency-closure mask is applied before the
paper DAE input is built. Visible current values are converted from grams into
the paper nutrient's `G`, `MG`, or `UG` unit. Hidden, unavailable, and unmapped
paper inputs are filled with paper-training means. The DAE reconstructs its
198 nutrient outputs, and the mapped target output is converted back to grams.
The name-enabled variant additionally receives the current food description.

Current validation and complete-test data are used only for forward evaluation.
They do not fit the scaler, text PCA, model parameters, epoch count, or
checkpoint selection. Existing KNN, RF, XGBoost, and V5+CCE predictions are
filtered and rescored on the identical common-axis jobs.

## Results

### Paper-native check

| Model | Food name | Paper holdout RMSE sum | Paper reference |
|---|---:|---:|---:|
| Paper DAE | No | **5.156620** | 5.26 |
| Paper-data DAE + FoodText | Yes | 5.372681 | 5.57 (InferSent variant) |

### Current validation, 127 common axes

| Method | Training corpus | Scaled-log MAE | Scaled-log RMSE | Raw MAE |
|---|---|---:|---:|---:|
| RF | Current | **0.128419** | **0.250992** | **0.206300** |
| XGBoost | Current | 0.136514 | 0.253638 | 0.218365 |
| V5+CCE | Current | 0.137452 | 0.293150 | 0.270260 |
| KNN | Current | 0.193570 | 0.348057 | 0.375417 |
| Paper DAE | Paper only | 0.546920 | 0.794806 | 1.059076 |
| Paper-data DAE + FoodText | Paper only | 0.552384 | 0.810123 | 1.090366 |

### Current complete test, 127 common axes

| Method | Training corpus | Scaled-log MAE | Scaled-log RMSE | Raw MAE |
|---|---|---:|---:|---:|
| V5+CCE | Current | **0.196371** | 0.353243 | 0.372027 |
| XGBoost | Current | 0.214957 | **0.342926** | **0.312990** |
| RF | Current | 0.217158 | 0.346190 | 0.331422 |
| KNN | Current | 0.319885 | 0.490872 | 0.581787 |
| Paper DAE | Paper only | 0.616934 | 0.877443 | 1.290308 |
| Paper-data DAE + FoodText | Paper only | 0.619005 | 0.901362 | 1.314242 |

The no-name paper DAE reproduces its source-domain result but does not transfer
well to the current task. V5+CCE's complete-test scaled-log MAE is 68.17% lower
than the source-only paper DAE's. Adding food text changes the paper DAE's
complete-test scaled-log MAE from 0.616934 to 0.619005, a 0.34% degradation.
Food names therefore do not repair the shift in nutrient vocabulary, data
coverage, source conventions, masking pattern, and target distribution.

This experiment is a **source-only transfer baseline**, not a controlled test
of DAE versus Transformer architecture. The existing methods were trained on
the current corpus, while both paper DAE variants were deliberately restricted
to the paper corpus. The appropriate conclusion is that the published DAE is
reproducible on its own problem but is not a competitive zero-shot foundation
model for the current completion problem.

## Versioned artifacts

- `artifacts/manifest.json`: hashes, frozen settings, paper-native checks, and
  current-task summary metrics.
- `artifacts/common_127_axis_metrics.csv`: the matched common-axis comparison.
- `artifacts/paper_to_current_axis_crosswalk.csv`: all accepted and rejected
  nutrient mappings with units and mapping reasons.
- `artifacts/history_*.csv`: aggregate epoch-level training reconstruction MSE.
- `artifacts/paper_holdout_axis_rmse_*.csv`: paper-native per-nutrient RMSE.
- `artifacts/paper_dae_*_axis_metrics.csv`: current-task per-axis aggregate
  metrics for both DAE variants.

No checkpoint, raw paper/current data, food-name embedding, profile-level
prediction, or profile identifier table is committed.
