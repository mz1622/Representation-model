# V9-R9 Transformer: first-group methods and parameter evidence

Status: a description of the current first-group methods, not a final model selection or complete results report. It documents the frozen configuration and executed code; both 60-epoch runs and their analyses have not yet finished. Historical MLP or PDF results are not results of this Transformer experiment. The corresponding Chinese file is [METHODS_ZH.md](METHODS_ZH.md).

## Data, targets and inputs

The study reuses the R0 `foodnutrigpt_v9_r0_v1 / quarantined` view: 64,700 source-native training profiles in 42,282 name candidate groups, and 11,175 validation profiles in 7,409 groups. Each partition covers 24 sources. Data, splits, quarantine rules, scales, tasks and name caches remain unchanged; RF/XGBoost are not refitted, and the complete test stays closed. Unresolved unit provenance and unconfirmed aliases remain limitations; freezing does not resolve them.

Of 252 axes, 142 nutrition and 45 metabolome axes receive supervision; 65 axes provide observed context only. Repeated measurements within a profile and axis are aggregated by the raw-unit median, without pooling across sources. Missing cells do not contribute supervision; explicit zeros remain observations. Targets use `t_a(y)=log(1+y/s_a)`, where `s_a` is the weighted median of training positives, balanced by name candidate group, source and profile. The 187 supervised axes contain 1,828,536 observed training targets; the 142 nutrition axes contain 1,795,133 targets, including 388,743 explicit zeros.

The name branch uses only `original_name`. Frozen MiniLM supplies 384-dimensional vectors, followed by the existing training-fitted PCA. The first 32 directions occupy 128 slots, with the remaining 96 set to zero. This matches the strongest completed common-protocol RF32 reference; it does not establish that 32 dimensions are optimal. Numeric inputs comprise transformed values and visibility for 252 axes. Source, food-group and processing metadata are not encoder inputs.

Each training task specifies a profile and chemical family, hides the entire family including related forms, and supervises only its observed eligible targets. Every epoch traverses 337,048 tasks and covers each observed training target exactly once. No additional name-only training mixture is used. Name-only is a required inference evaluation scenario in this group, not a capability explicitly optimized by a separate training task.

## Architecture and numeric output

The sequence contains one learned CLS token, one name token and a fixed grid of 252 axis tokens: 254 tokens without truncation. Names pass through LayerNorm and a two-layer GELU projection. Each visible value passes through a `1→192→192` GELU MLP and is added to its axis embedding. A learned mask vector replaces numeric encodings at missing or hidden positions. All axes are always present; query axes are not determined by whether their labels were observed. There are no positional, type or source embeddings in the sequence.

The encoder uses pre-norm Transformer layers and a final LayerNorm. Each axis state passes through a shared nonlinear scalar head, plus a rank-16 axis residual and an axis bias. It directly predicts the transformed value `z_hat`; inference returns `y_hat=s_a*expm1(max(z_hat,0))`. Training loss uses `z_hat` before this nonnegative clipping. Predictions are not multiplied by a presence probability. The inherited presence branch is still constructed and called to preserve initialization and dropout RNG consumption, but its parameters are frozen and its output contributes to neither loss nor prediction.

|Configuration field|Value|
|---|---|
|`d_model`|192|
|`n_layers`|3|
|`n_heads`|6|
|`feedforward_dim`|768|
|`dropout`|0.15|
|`axis_residual_rank`|16|
|`parameter_count`|1548594|
|`trainable_parameter_count`|1515929|
|`active_name_dimensions`|32|
|`name_input_slots`|128|
|`batch_size`|64|
|`epochs`|60|
|`schedule_epochs`|60|
|`learning_rates`|0.0001, 0.0003|
|`weight_decay`|0.0001|
|`gradient_clip`|1.0|
|`source_weight`|1.0|
|`source_residual_l2`|0.0001|
|`eta_min_fraction`|0.01|
|`seed`|20260922|
|`confirmation_seeds`|20260922, 20260923, 20260924|

## Loss, source calibration and optimization

Let `w_ia=1/(n_sources(g,a)*n_profiles(g,a,s))`: each source has equal total weight within a name candidate group and axis, divided equally among its profiles. Let `W_a` be the total training weight for axis `a`. The full-data direct regression objective is:

`L_base = (1/187) * sum_a [sum_i w_ia*abs(z_hat_ia-t_a(y_ia)) / W_a]`.

Minibatches sum only observed targets in their tasks and multiply by `N_tasks/(batch_tasks*187)`, using global training-axis denominators rather than re-averaging the axes present in each minibatch. Training and scoring aggregation are not identical: scoring first takes separate medians of profile labels and predictions within each candidate group, source and axis, then computes errors; training weights profile-level errors. Equivalence between training loss and the validation primary metric is not assumed.

Source is used only for training calibration: `z_calibrated=z_hat+b_sa`. Offsets are centered by the arithmetic mean across training sources for each axis; unseen-source offsets are zero. Total loss is `(L_base + lambda*L_calibrated)/(1+lambda) + rho*mean(R_train^2)`, with `lambda=1` and `rho=0.0001`. The penalty acts on the training-source parameter table before centering. Offsets are absent at inference. Division by `1+lambda` controls the overall scale of the two losses; any benefit of source calibration still requires a separate ablation.

The optimizer is AdamW; the two candidates differ only in initial learning rate. Batch size is 64, weight decay is 0.0001, and gradient-norm clipping is 1. Training is single-stage for 60 epochs with cosine `T_max=60` and a minimum learning rate of 0.01 times its initial value. Task order is determined by `seed+epoch`. Computation uses float32 without mixed precision. Every epoch verifies complete task coverage, target count, order fingerprints and frozen-file fingerprints, and records clipping frequency and pre-clipping gradient norms. NaN/Inf causes failure.

Before formal optimization, 50 overfitting steps are run on 32 training tasks. Model parameters and CPU/CUDA RNG state are then restored before creating the formal optimizer. This check demonstrates the ability to reduce small-batch training loss, not generalization.

## Selection, three capabilities and comparisons

Each epoch evaluates the complete fixed validation panel. The earliest strict minimum of 142-axis macro scaled-log MAE determines the checkpoint. Best-through-20 and best-through-60 checkpoints are retained. The 20/60 comparison uses one 60-epoch cosine trajectory and combines additional training with more validation selection opportunities; it is not an independent, pure-duration causal experiment.

Completion hides the entire target family; name-only hides all numeric context. Both score only genuinely observed labels and report 142/45/187-axis results, legacy log error, raw-unit error, and positive/zero errors. Sources are equally weighted within candidate groups, followed by averaging over food candidate groups and axes.

Nutrition-to-name retrieval uses the same checkpoint's name-predicted nutrition profiles followed by matching; no contrastive retrieval head is trained in this group. The fixed 49,913 candidate names generate 142-axis predictions from names alone. Queries contain observed nutrition only, and use mean squared distance in transformed space over visible axes. Evaluation covers fully visible and fixed 30%-visible scenarios with at least three visible nutrition axes. Candidate order resolves exact distance ties. Recall@1/5/10 and MRR use exact original names as correct answers; no confirmed alias map is available. This is not semantic-identity accuracy, and distances are not probabilities.

The screening seed is 20260922. A promising fixed recipe would subsequently be confirmed with 20260922, 20260923 and 20260924. Food-group paired resampling intervals do not remove repeated-validation selection bias or include the frozen RF's seed uncertainty. Acceptance requires mean Transformer primary error below the fixed RF with a paired interval supporting improvement, while relative regression in legacy nutrition log-MAE is at most 2%; superiority to XGBoost is not required.

## Parameter evidence and unresolved questions

|Choice|Evidence|Interpretation currently allowed|
|---|---|---|
|Width192, 3 layers, 6 heads, FF768|Inherited direct Transformer; exact functional equivalence to its parent|A matched-input control, not evidence of optimal capacity|
|Learning rates0.0001/0.0003|Preregistered single-factor contrast|Optimization differences can be assessed after both runs finish|
|60 epochs and a 20-epoch snapshot|Preregistered budget diagnostic|No complete budget comparison is available yet|
|Direct MAE|Motivated by earlier R2 direct-regression experiments; control rebuilt here|Historical gains cannot be transferred causally across different name-input protocols|
|Source weight1, residual penalty0.0001|Inherited source-calibration control|Not yet shown better than disabling calibration under the current inputs, objective and budget|
|Batch64, clipping1, weight decay0.0001|Fixed engineering choices and inherited training settings|Reproducible choices, not claimed tuned optima|
|Dropout0.15, rank16|Motivated by the parent architecture and PDF|No independent current-protocol ablation yet|
|PCA32, data and scoring|Frozen common-input comparison protocol|Not adjusted in this round or treated as hyperparameters for achieving a win|

The PDF specifies width256, 3 layers, 8 heads, FF1024 and macro-axis MSE, but uses different data merging, supervision, normalization and evaluation, and does not specify the initial learning rate. This work borrows methods; it neither reproduces the PDF scores nor restores two-stage training. Any later capacity-group or MSE intervention must be registered separately after inspecting complete first-group evidence. A new branch is not automatically the final recipe.

## Reproduction and evidence limits

The original run implementation commit is `b4fd4c1`. Later documentation commits are distinct from changes to the training implementation; each run's source snapshot and hashes establish exact code identity. The environment is Python3.10.19, PyTorch2.7.1+cu128 and an RTX5070Ti16GB, with four CPU training threads. Actual costs come from completed run manifests and histories, not planned timing estimates.

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr1e4_60 --output-dir output/v9_r9/tf192_mae_lr1e4_60
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr3e4_60 --output-dir output/v9_r9/tf192_mae_lr3e4_60
```

These record the registered commands; existing output directories cannot be overwritten and the commands are not instructions to launch duplicate jobs. Completed runs undergo independent score replay, budget and learning-rate comparisons, and full-training fit diagnostics. Functional verification is complete but does not replace performance evidence. The full study still requires eight-section iteration reports, all three tasks, replicated confirmation and final bilingual attribution. This appendix does not predeclare the final recipe or establish foundation-model transfer capabilities.

Evidence index: [configuration](../../experiments/foodnutrigpt_v9_research/r9/config.json), [preregistration](../../experiments/foodnutrigpt_v9_research/r9/PLAN.md), [freeze receipt](../v9_r9_freeze_v1/manifest.json), [functional verification](../v9_r9_functional_v1/verification.json), [data summary](../v9_final_data_evidence_v1/summary.json), and [baseline methods](../v9_r9_frozen_baseline_appendix_v2/BASELINES_EN.md). Numeric predictions and checkpoints remain in local ignored directories; this appendix contains no raw food-composition values.
