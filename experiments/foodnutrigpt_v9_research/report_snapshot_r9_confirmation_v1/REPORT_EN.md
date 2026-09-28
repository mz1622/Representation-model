# Nutrition representation model: integrated R0–R9 research report

Chinese counterpart: [REPORT_ZH.md](REPORT_ZH.md). This report consolidates completed experiments and verification; internal validation does not establish external transfer or label validity.

Version scope: a completed stage report through the R9 fixed-MAE three-seed confirmation. Subsequent method experiments are registered separately and are not reported as completed results here. A later report may supersede this snapshot.

## 1. Research objective and principal findings

The current objective is to improve nutrition completion through validated single-stage Transformer methods while keeping data and previously trained RF/XGBoost results frozen. Name-only prediction and nutrition-to-name retrieval remain tracked tasks. Historical MLP findings are retained as method exploration, not substituted for Transformer results.

The fixed lr=3e-4, 60-epoch recipe has three-seed primary completion error **0.189184 ± 0.006925**. Mean improvement over frozen RF is **-0.081%**, with a conditional food-group 95% interval of **[-2.173%, 2.074%]**. Mean relative improvement in legacy log-MAE is **-2.103%**. The recipe **does not pass the registered RF confirmation gates**. Sections 5–6 give all references, individual seeds and aggregate results; the interval is not a guarantee for arbitrary new seeds or datasets.

The registered gates require lower mean primary error than fixed RF, a food-group paired interval supporting improvement, and at most 2% relative regression in mean legacy nutrition log-MAE. There is no 5% minimum improvement or requirement to beat XGBoost. Each seed uses its completion-selected checkpoint for all three tasks, without selecting different task-specific winners.

## 2. Data sources, actual training data and label validity

### 2.1 Direct input package versus upstream databases

The code originated from [Representation-model](https://github.com/mz1622/Representation-model), parent commit `1289d38129dffb7d3490239fb516328fa5c905e3`. The direct data input was the user-provided `foodnutrigpt_v8_source_native_baseline.tar.gz`. The archive and V8 artifacts remain unchanged. The package records 25 sources, 76,915 source-native profiles and 2,259,178 observations; these are not the number of training samples in this study.

The research view loads 64,700 training and 11,175 validation profiles, covering 42,282 and 7,409 exact-name candidate groups and 24 sources in each partition. The package reserves 395 USDA Foundation profiles as a source holdout. Its complete test panel contains 659 profiles; another 381 profiles overlapping Foundation exact-name groups are excluded. These test counts come only from existing partition metadata; this iteration did not load test labels for model evaluation. Upstream historical test results already exist, so that panel would not become wholly new external evidence if evaluated later.

Counts below were recomputed from the quarantined research view. Source links identify database families. The later acquisition inventory may refer to newer releases: **a current source URL does not establish which exact upstream release produced the V8 records or certify individual values.**

|Source|Training profiles|Validation profiles|Observed training nutrition cells|
|---|---:|---:|---:|
|[Australian Food Composition Database](https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/data-files)|1314|247|35245|
|[FAO/INFOODS AnFooD 2.0](https://www.fao.org/food-composition/tables-and-databases/detail/%28global--2017%29-fao-infoods-analytical-food-composition-database---version-2.0-%28anfood2.0%29/en)|601|72|3979|
|[Bangladesh Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|437|57|7122|
|[FAO/INFOODS BioFoodComp 4.0](https://www.fao.org/infoods/infoods/food-biodiversity/en/)|959|140|964|
|[German Nutrient Database BLS 4.0](https://blsdb.de/download)|6051|1089|94523|
|[ANSES-Ciqual](https://ciqual.anses.fr/)|2945|521|92130|
|[Canadian Nutrient File](https://food-nutrition.canada.ca/cnf-fce/)|5044|922|298207|
|[UK Composition of Foods Integrated Dataset](https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid)|2414|423|34126|
|[EFSA EU Food Composition Database (package label: 2013)](https://www.efsa.europa.eu/en/data-report/food-composition)|14621|2324|147914|
|[USDA Food and Nutrient Database for Dietary Studies](https://fdc.nal.usda.gov/data-documentation.html)|4612|790|212152|
|[FooDB](https://foodb.ca/about)|9467|1780|239188|
|[Frida FoodData, Denmark](https://frida.fooddata.dk/)|1167|188|50658|
|[Lesotho Food Composition Table 2006](https://www.fao.org/infoods/infoods/tables-and-databases/africa/en/)|250|40|6153|
|[Japan MEXT Food Composition Tables (package label: 2023)](https://www.mext.go.jp/a_menu/syokuhinseibun/index.htm)|2148|390|80926|
|[Norwegian Food Composition Database](https://www.matvaretabellen.no/en/api/)|1808|287|61542|
|[FAO/INFOODS/IZiNCG PhyFoodComp 1.0](https://www.fao.org/food-composition/tables-and-databases/detail/%28global--2018%29-fao-infoods-izincg-global-food-composition-database-for-phytate---version-1.0-%28phyfoodcomp1.0%29/en)|1879|307|4956|
|[SMILING Cambodia Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|78|11|1631|
|[SMILING Indonesia Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|91|17|1182|
|[SMILING Laos Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|128|12|1790|
|[SMILING Thailand Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|122|17|1921|
|[SMILING Vietnam Food Composition Table 2013](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/)|130|32|1562|
|[Swiss Food Composition Database 7.1](https://valeursnutritives.ch/en/downloads/)|1078|168|25800|
|[USDA Standard Reference Legacy](https://fdc.nal.usda.gov/data-documentation.html)|6481|1191|363116|
|[FAO/INFOODS Western Africa Food Composition Table 2019](https://www.fao.org/food-composition/tables-and-databases/detail/food-composition-tables/en)|875|150|28346|
|USDA Foundation Foods (held out)|0|0|0|

Sources may share borrowed records, recipe calculations or compilation histories, and may use different analytical definitions. A source identifier is not proof of an independent laboratory measurement; 24 sources are not necessarily 24 statistically independent datasets. Comprehensive alias, translation and cross-database lineage review remains incomplete.

Archive identity: the user-supplied archive contains 135,711,396 bytes, SHA256 `83ee2b50f04963943797aa1818a91f66d0ffaeb7a5bc3c5b2db7a11be6dff792`. All10 native payload files match the embedded manifest, local V8 files and frozen R0 input fingerprints. This verifies file identity, not publisher authenticity, licensing or nutritional correctness.

### 2.2 Units, aggregation and quarantine

The package expresses normalized values in g/100g. Suspicious values were not corrected by intuition. Within each source-native profile and axis, point labels use the median in original units. Original observations are retained, and numerical values are not automatically pooled across sources. Food-name groups are partitioning and weighting units, not confirmed food-identity merges.

FooDB contains within-profile, within-axis positive values separated by factors of 1,000 or 1,000,000. The Goose fat/Cholesterol example remains unresolved because original Content.csv/staging evidence is unavailable. A total of 64 cells were quarantined—53 training and 11 validation—covering 145 original observations. The inclusive view was retained for the existing sensitivity analysis. The current parser treats “mg/100 g” and “mg/100g” identically; this does not establish that the historical processing chain was correct. Fresh/dry weight, edible portion and chemical-form definitions remain incompletely traced.

Before quarantine, the train/validation research view contains 2,221,276 observations and 2,153,204 profile-axis cells. After quarantine, training has 1,829,199 observed cells over all 252 axes: 1,828,536 supervised cells over 187 axes and 1,795,133 nutrition cells over 142 axes. There are 388,743 explicit-zero training nutrition cells. Validation contains 323,941 observed cells, 323,809 supervised targets, 317,616 nutrition cells and 69,271 explicit-zero nutrition cells.

Missing labels do not enter the loss; explicit zeros remain observed labels. Hidden input slots use a numerical zero together with a separate visibility indicator. This is an input representation, not imputation of unknown labels as true zeros. Quarantine neither certifies every retained label nor detects all systematic unit errors.

### 2.3 Transforms, partitioning and common evaluation

Each axis uses `t_a(y)=log(1+y/s_a)`, where `s_a` is a source-balanced weighted median of positive training values within food candidate groups. Scales, embedding projections and fitted normalization never use validation or test labels. The primary metric was fixed before model comparison: macro-average MAE over the 142 nutrition axes in scaled-log space. It was not selected to favor a winning model.

For evaluation, labels and predictions are separately aggregated by median within food candidate group × axis × source. Errors are then averaged equally over sources within each group, over food groups, and finally over axes. Secondary metrics include legacy log1p(g/100g) MAE, raw-unit MAE, conditional positive/zero errors, the 45-axis subset and all 187 targets. Conditional positive and zero MAEs have different denominators and are not additive; additive diagnostic contributions retain the original primary-metric denominator.

All models share validation labels, requested axes, family masks, visible context and evaluation weights. Exact-name candidate groups are disjoint across training and validation. Similar names are review candidates, not automatic merges; this partition check does not rule out every semantic near-duplicate. Primary paired intervals use 7,344 food groups with nutrition evaluation support, not independent nutrient cells.

## 3. Current Transformer methods and parameter evidence

### Data, targets and inputs

The study reuses the R0 `foodnutrigpt_v9_r0_v1 / quarantined` view: 64,700 source-native training profiles in 42,282 name candidate groups, and 11,175 validation profiles in 7,409 groups. Each partition covers 24 sources. Data, splits, quarantine rules, scales, tasks and name caches remain unchanged; RF/XGBoost are not refitted, and the complete test stays closed. Unresolved unit provenance and unconfirmed aliases remain limitations; freezing does not resolve them.

Of 252 axes, 142 nutrition and 45 metabolome axes receive supervision; 65 axes provide observed context only. Repeated measurements within a profile and axis are aggregated by the raw-unit median, without pooling across sources. Missing cells do not contribute supervision; explicit zeros remain observations. Targets use `t_a(y)=log(1+y/s_a)`, where `s_a` is the weighted median of training positives, balanced by name candidate group, source and profile. The 187 supervised axes contain 1,828,536 observed training targets; the 142 nutrition axes contain 1,795,133 targets, including 388,743 explicit zeros.

The name branch uses only `original_name`. Frozen MiniLM supplies 384-dimensional vectors, followed by the existing training-fitted PCA. The first 32 directions occupy 128 slots, with the remaining 96 set to zero. This matches the strongest completed common-protocol RF32 reference; it does not establish that 32 dimensions are optimal. Numeric inputs comprise transformed values and visibility for 252 axes. Source, food-group and processing metadata are not encoder inputs.

Each training task specifies a profile and chemical family, hides the entire family including related forms, and supervises only its observed eligible targets. Every epoch traverses 337,048 tasks and covers each observed training target exactly once. No additional name-only training mixture is used. Name-only is a required inference evaluation scenario in this group, not a capability explicitly optimized by a separate training task.

### Architecture and numeric output

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
|`learning_rate_screening / selected`|0.0001, 0.0003 / 0.0003|
|`weight_decay`|0.0001|
|`gradient_clip`|1.0|
|`source_weight`|1.0|
|`source_residual_l2`|0.0001|
|`eta_min_fraction`|0.01|
|`screening_seed`|20260922|
|`confirmation_seeds`|20260922, 20260923, 20260924|

### Loss, source calibration and optimization

Let `w_ia=1/(n_sources(g,a)*n_profiles(g,a,s))`: each source has equal total weight within a name candidate group and axis, divided equally among its profiles. Let `W_a` be the total training weight for axis `a`. The full-data direct regression objective is:

`L_base = (1/187) * sum_a [sum_i w_ia*abs(z_hat_ia-t_a(y_ia)) / W_a]`.

Minibatches sum only observed targets in their tasks and multiply by `N_tasks/(batch_tasks*187)`, using global training-axis denominators rather than re-averaging the axes present in each minibatch. Training and scoring aggregation are not identical: scoring first takes separate medians of profile labels and predictions within each candidate group, source and axis, then computes errors; training weights profile-level errors. Equivalence between training loss and the validation primary metric is not assumed.

Source is used only for training calibration: `z_calibrated=z_hat+b_sa`. Offsets are centered by the arithmetic mean across training sources for each axis; unseen-source offsets are zero. Total loss is `(L_base + lambda*L_calibrated)/(1+lambda) + rho*mean(R_train^2)`, with `lambda=1` and `rho=0.0001`. The penalty acts on the training-source parameter table before centering. Offsets are absent at inference. Division by `1+lambda` controls the overall scale of the two losses; any benefit of source calibration still requires a separate ablation.

The optimizer is AdamW; the two candidates differ only in initial learning rate. Batch size is 64, weight decay is 0.0001, and gradient-norm clipping is 1. Training is single-stage for 60 epochs with cosine `T_max=60` and a minimum learning rate of 0.01 times its initial value. Task order is determined by `seed+epoch`. Computation uses float32 without mixed precision. Every epoch verifies complete task coverage, target count, order fingerprints and frozen-file fingerprints, and records clipping frequency and pre-clipping gradient norms. NaN/Inf causes failure.

Before formal optimization, 50 overfitting steps are run on 32 training tasks. Model parameters and CPU/CUDA RNG state are then restored before creating the formal optimizer. This check demonstrates the ability to reduce small-batch training loss, not generalization.

### Selection, three capabilities and comparisons

Each epoch evaluates the complete fixed validation panel. The earliest strict minimum of 142-axis macro scaled-log MAE determines the checkpoint. Best-through-20 and best-through-60 checkpoints are retained. The 20/60 comparison uses one 60-epoch cosine trajectory and combines additional training with more validation selection opportunities; it is not an independent, pure-duration causal experiment.

Completion hides the entire target family; name-only hides all numeric context. Both score only genuinely observed labels and report 142/45/187-axis results, legacy log error, raw-unit error, and positive/zero errors. Sources are equally weighted within candidate groups, followed by averaging over food candidate groups and axes.

Nutrition-to-name retrieval uses the same checkpoint's name-predicted nutrition profiles followed by matching; no contrastive retrieval head is trained in this group. The fixed 49,913 candidate names generate 142-axis predictions from names alone. Queries contain observed nutrition only, and use mean squared distance in transformed space over visible axes. Evaluation covers fully visible and fixed 30%-visible scenarios with at least three visible nutrition axes. Candidate order resolves exact distance ties. Recall@1/5/10 and MRR use exact original names as correct answers; no confirmed alias map is available. This is not semantic-identity accuracy, and distances are not probabilities.

The screening seed is 20260922. The selected fixed recipe was subsequently repeated using 20260922, 20260923 and 20260924. Food-group paired resampling intervals do not remove repeated-validation selection bias or include the frozen RF's seed uncertainty. Acceptance requires mean Transformer primary error below the fixed RF with a paired interval supporting improvement, while relative regression in legacy nutrition log-MAE is at most 2%; superiority to XGBoost is not required.

### Parameter evidence and unresolved questions

|Choice|Evidence|Interpretation currently allowed|
|---|---|---|
|Width192, 3 layers, 6 heads, FF768|Inherited direct Transformer; exact functional equivalence to its parent|A matched-input control, not evidence of optimal capacity|
|Learning rates0.0001/0.0003|Preregistered single-factor contrast|The completed same-seed, same-initialization, same-order contrast supports0.0003; see Section7|
|60 epochs and a 20-epoch snapshot|Preregistered budget diagnostic|Completed nested-window contrasts support the60-epoch window, without separating training duration from extra selection opportunities|
|Direct MAE|Motivated by earlier R2 direct-regression experiments; control rebuilt here|Historical gains cannot be transferred causally across different name-input protocols|
|Source weight1, residual penalty0.0001|Inherited source-calibration control|Not yet shown better than disabling calibration under the current inputs, objective and budget|
|Batch64, clipping1, weight decay0.0001|Fixed engineering choices and inherited training settings|Reproducible choices, not claimed tuned optima|
|Dropout0.15, rank16|Motivated by the parent architecture and PDF|No independent current-protocol ablation yet|
|PCA32, data and scoring|Frozen common-input comparison protocol|Not adjusted in this round or treated as hyperparameters for achieving a win|

The PDF specifies width256, 3 layers, 8 heads, FF1024 and macro-axis MSE, but uses different data merging, supervision, normalization and evaluation, and does not specify the initial learning rate. This work borrows methods; it neither reproduces the PDF scores nor restores two-stage training. Any later capacity-group or MSE intervention must be registered separately after inspecting complete first-group evidence. A new branch is not automatically the final recipe.

### Reproduction and evidence limits

The original run implementation commit is `b4fd4c1`. Later documentation commits are distinct from changes to the training implementation; each run's source snapshot and hashes establish exact code identity. The environment is Python3.10.19, PyTorch2.7.1+cu128 and an RTX5070Ti16GB, with four CPU training threads. Actual costs come from completed run manifests and histories, not planned timing estimates.

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr1e4_60 --output-dir output/v9_r9/tf192_mae_lr1e4_60
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr3e4_60 --output-dir output/v9_r9/tf192_mae_lr3e4_60
```

These record the registered commands; existing output directories cannot be overwritten and the commands are not instructions to launch duplicate jobs. Completed runs undergo independent score replay, budget and learning-rate comparisons, and full-training fit diagnostics. Functional verification is complete but does not replace performance evidence. Iteration records, all three tasks, fixed-recipe repetitions and selection reasoning are consolidated below. They do not establish foundation-model transfer capabilities.

Evidence index: [configuration](../r9/config.json), [preregistration](../r9/PLAN.md), [freeze receipt](../../../reports/v9_r9_freeze_v1/manifest.json), [functional verification](../../../reports/v9_r9_functional_v1/verification.json), [data summary](../../../reports/v9_final_data_evidence_v1/summary.json), and [baseline methods](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_EN.md). Numeric predictions and checkpoints remain in local ignored directories; this appendix contains no raw food-composition values.

### Three public inference interfaces

`TransformerNutritionModel` provides `predict(food_name, observed_profile, target_axes)`, `encode(food_name=None, observed_profile=None, modality="fused")`, and `retrieve_names(observed_profile, candidate_names, top_k=10)`. Callers specify supervised output axes withheld from observed context; empty nutrition context gives name-only prediction. The interface rejects the 65 context-only axes as unsupported outputs.

`encode` supports name/nutrition/fused modes and currently returns a 192-dimensional mean of axis-token hidden states. Name mode hides numeric context; nutrition mode does not receive name information. This is a probe interface, not a contrastive representation validated for transfer. Retrieval receives a nutrition query, builds candidate profiles from names, and returns negative transformed-space MSE explicitly marked as non-probabilistic. The interface needs at least one nutrition observation; the formal retrieval panel separately requires at least three.

The completed seed 20260923 checkpoint passed 12 interface checks using three cached training names and one training profile: modality isolation, unobserved outputs, candidate-order/duplicate invariance, explicit-zero queries and score reconstruction. This is functional evidence for that checkpoint and these inputs, not a new performance or transfer experiment. Full record: [Public interface audit](../../../reports/v9_r9_public_interfaces_seed23_v1/README.md).

## 4. Iteration history and attribution

Each version retains all configurations, failures, per-axis/source analyses and three-task results. The table is not a causal curve made from the best score in each round. R0→R1 changes the training protocol, and R7/R8 change the name-input protocol.

|Version|Question, intervention and principal evidence|Interpretation and decision|
|---|---|---|
|[R0](../r0/README.md)|Common raw-space aggregation, training scales, name-only text and query/mask rules; 12-configuration baseline and quarantine|Comparable evaluation established; provenance unresolved and training-mask distributions still differed. No pure architecture attribution|
|[R1](../r1/README.md)|Exhaustive family tasks and global axis weights; fixed-schedule 8→20 epochs gave MLP .246757→.222459; V9 amount weights 1/2/3 and source ablation|Budget control supports longer training; larger amount weight helped that V9 branch; source removal had no clear completion gain. Joint protocol fixes are not single-factor evidence|
|[R2](../r2/README.md)|12 budget entries: SmoothL1→MAE .222459→.209484; width 256→512 .209484→.202428; 20→60 within one 60-epoch schedule .209506→.184491; also width 1024, normalization, direct V9, selection and name standardization|Retain MAE/512/60 and LayerNorm. MAE mainly helped zeros. Width1024 gain was uncertain and name-only worsened. Hurdle→direct changed multiple mechanisms|
|[R3](../r3/README.md)|Four configurations: 10%/20% name-only assignment, separate task heads and diagnostics|Some name gains cost completion performance; not adopted in the historical completion MLP. Not every trade-off establishes gradient conflict|
|[R4](../r4/README.md)|Eight entries: five complete including reused control, three numerical failures; query residuals, view consistency and metabolome coefficient .5; common inverse/query contract fixes|No sufficient completion benefit; failures preserved, no NaN omission or loose tolerance to manufacture success. Engineering fixes separated from performance claims|
|[R5](../r5/README.md)|Twelve configurations covering name prediction, nutrition–name alignment and partial views; full-input specialist three-seed R@10 .298199±.003720 versus then-current KNN .192192|Limited full-input retrieval benefit; 30% R@10 only .002252. A separate specialist, not the completion model; superiority over later PCA128 KNN was not demonstrated|
|[R6](../r6/README.md)|Additional fixed 30% and mixed30/60/90 context deletion: completion .197232/.210462 versus parent .184491|Both worsen the fixed completion panel; not adopted. Name gains do not replace the completion result|
|[R7](../r7/README.md)|Six entries; common PCA32→128 improved name-MLP/KNN primary errors by 8.84%/6.87%|Retain richer name input; MLP still 5.11% worse than matched KNN. Old-input trees are not matched baselines|
|[R8](../r8/SCOPE_CHANGE_CLOSURE.md)|10 of 12 entries complete, one partial and one unrun; six tree configurations complete. MLP32→128 completion gain .931% with interval crossing zero; name-only gain18.909%|Frozen by user scope change. No further tree fitting, no complete RF search or tree seed-confirmation claim|
|[Historical MLP repetition](../final_report_v1/NEURAL_REPLICATION.md)|Three fixed-recipe MLP seeds completed with exact replay; completion0.183435 ± 0.001222|Historical PCA128 evidence, not completion of the Transformer goal|
|[R9](../r9/README.md)|Common PCA32, width192 direct Transformer, MAE, source1; both learning rates completed60epochs and3e-4 received two additional seeds|1e-4 does not beat RF;3e-4 screening seed gives0.183553. Three-seed results and decision appear in Sections6–8|

The historical MLP/PCA128 recipe achieved three-seed completion error 0.183435 ± 0.001222 and name-only error 0.484042 ± 0.010458. These are not results of the current PCA32 Transformer. Its name-removal/permutation findings and common-axis underprediction diagnosis cannot automatically be transferred to the Transformer. Full parameters, seed results and failures remain in the [earlier snapshot](../final_report_v1/REPORT_EN.md) and individual iteration records.

Numeric-basis components underwent train-only feasibility checks but were not used in the current R9 training. Only two learning-rate candidates were screened in the first R9 group; additional seeds repeat the fixed recipe; feasibility checks are not model-performance evidence.

## 5. Frozen RF/XGBoost methods and complete results

### Shared inputs and supervision

Each configuration fits 187 separate axis regressors: 142 nutrition and 45 metabolome axes. The other 65 axes provide observed context only. Every eligible training profile is used for each target axis, with no 5,000-row cap. There are 1,828,536 training targets; the largest axis has 58,958 rows. Explicit zeros are labels, missing cells are excluded, and values are not pooled across sources.

The 632 input slots comprise 128 name slots, 252 transformed values and 252 visibility indicators. The 32-dimensional setting uses the first 32 directions of the same training-fitted PCA basis and zeros the other 96 slots; the 128-dimensional setting uses all directions. Text contains only the food name. Completion hides the entire target family; name-only hides all numeric values and visibility indicators. The scale s in t=log(1+y/s) is fitted on training data only. Per-axis source-balanced weights are divided by their mean and supplied as sample_weight.

### Fitting methods and realized search

RF uses scikit-learn 1.5.2: 400 bootstrap trees, squared error, unlimited depth, min_samples_leaf=1 and max_features=0.5, with completed 32- and 128-dimensional runs. XGBoost 2.1.3 uses 800 trees, hist, learning_rate=0.03, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1 and reg:squarederror, without early stopping; depths and input dimensions appear below. Both methods use random_state=20260922+axis_index and four fitting threads. RF inference accumulates trees serially in a fixed order. Full get_params records are preserved in summary.json; library defaults are not presented as tuned choices.

Six of eight registered tree configurations completed. RF leaf3/feature0.5/name128 stopped at 123/187 axes and RF leaf1/feature1.0/name128 did not run because the user froze tree experiments. Neither has a complete score or enters selection; the three-configuration RF search was not completed. No additional tree seeds were run: all rows below are fixed seed20260922 references, without an invented seed standard deviation.

|Run|Name dim|Trees|Depth|Min leaf|Feature fraction|Preparation + fit sum (s)|Run elapsed (s)|
|---|---|---|---|---|---|---|---|
|rf400leaf1half_name32|32|400|unlimited|1|0.5|5457.2|5807.5|
|rf400leaf1half_name128|128|400|unlimited|1|0.5|16821.1|17318.3|
|xgb800d10_name32|32|800|10|N/A|N/A|1607.6|1686.6|
|xgb800d10_name128|128|800|10|N/A|N/A|2753.6|2833.4|
|xgb800d6_name128|128|800|6|N/A|N/A|1359.1|1437.2|
|xgb800d14_name128|128|800|14|N/A|N/A|5733.2|5861.3|

Costs are measured per run: the sum of per-axis preparation-and-fitting times and total elapsed time including prediction and checks. The recorded fit_seconds starts before feature construction and includes preparation and checks, so it is not isolated fit-call time. Earlier concurrent workloads also prevent treating these as a controlled algorithm-speed comparison.

### Nutrition prediction results

All results are internal validation. The primary metric is macro scaled-log MAE over 142 axes, with food-group and within-group source balancing. Lower is better. Name dimensions are identified separately; cross-dimension differences are not attributed solely to architecture.

|Run|Completion 142|Legacy log 142|Name-only 142|Completion 45|Completion 187|
|---|---|---|---|---|---|
|rf400leaf1half_name32|0.189031|0.056270|0.659518|0.674196|0.305782|
|rf400leaf1half_name128|0.199422|0.060131|0.619922|0.698854|0.319606|
|xgb800d10_name32|0.174487|0.052593|0.656148|0.674273|0.294756|
|xgb800d10_name128|0.181550|0.055256|0.612496|0.676969|0.300768|
|xgb800d6_name128|0.189261|0.057610|0.597474|0.675860|0.306357|
|xgb800d14_name128|0.182485|0.055811|0.621506|0.677074|0.301504|

|Run|Raw MAE (g/100g)|Positive scaled MAE|Zero scaled MAE|
|---|---|---|---|
|rf400leaf1half_name32|0.321219|0.248086|0.168645|
|rf400leaf1half_name128|0.340156|0.261008|0.182566|
|xgb800d10_name32|0.298575|0.230310|0.146588|
|xgb800d10_name128|0.306447|0.240916|0.160178|
|xgb800d6_name128|0.322449|0.249631|0.164759|
|xgb800d14_name128|0.309121|0.241528|0.161849|

Positive and zero conditional errors use different denominators and cannot be added to recover the primary metric. Full-precision metrics for every task and subset are in summary.json.

### Nutrition-to-name retrieval

The same fitted axis models generate candidate nutrition vectors using candidate names alone. Queries contain no name and candidates use no measured nutrition profiles. The library has 49,913 names. Visibility fractions 0.3/1.0 have 8,610/10,479 eligible queries, requiring at least three observed nutrition axes. Relevance uses the exact original name; no confirmed alias map exists. All numbers below are fractions in [0,1]. The visibility strata have different eligible query populations and are not a pure visibility intervention.

|Run|Visible fraction|MRR|R@1|R@5|R@10|
|---|---|---|---|---|---|
|rf400leaf1half_name32|0.3|0.001477|0.000310|0.001239|0.001987|
|rf400leaf1half_name32|1.0|0.002528|0.000176|0.001566|0.004457|
|rf400leaf1half_name128|0.3|0.001149|0.000000|0.000310|0.001510|
|rf400leaf1half_name128|1.0|0.002827|0.000000|0.001474|0.004721|
|xgb800d10_name32|0.3|0.001559|0.000310|0.001471|0.001819|
|xgb800d10_name32|1.0|0.003417|0.000423|0.002468|0.006275|
|xgb800d10_name128|0.3|0.001408|0.000000|0.000968|0.001936|
|xgb800d10_name128|1.0|0.004482|0.000306|0.003852|0.008533|
|xgb800d6_name128|0.3|0.001779|0.000155|0.000929|0.001768|
|xgb800d6_name128|1.0|0.005945|0.000796|0.006092|0.011207|
|xgb800d14_name128|0.3|0.001439|0.000000|0.001007|0.002671|
|xgb800d14_name128|1.0|0.004076|0.000494|0.003046|0.006381|

### Interpretation and reproducibility

The matched 32-dimensional R9 references are frozen RF leaf1/feature0.5 and XGB depth10. They are references from the completed set, not optima of the unfinished search or across all random seeds. Complete three-seed Transformer results appear in Section 6. This appendix establishes neither a neural win nor universal label validity or foundation-model capability.

Per-axis execution checked reversed order, subbatches and in-memory pickle replay; completed audits also verified data, features, weights and saved predictions/retrieval ranks. Fitted forests were discarded after prediction, so saved artifacts do not support arbitrary new-name online tree inference. Numeric predictions and candidate matrices remain in local ignored directories; this appendix contains aggregates only.

Data manifest SHA256: `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`

Freeze manifest SHA256: `1ec58002d76beb329004db77ef512622f57b40e142725799918d20027b5fddf3`

Code, environment, full parameters, run hashes and fit-record hashes are in the [machine-readable results](../../../reports/v9_r9_frozen_baseline_appendix_v2/summary.json). Every numeric table is identical across languages. A completed appendix is not the final study report.

## 6. All R9 candidates, fixed-recipe repetitions and comparisons

### 6.1 Screening and budget contrasts (one seed)

Every row uses one checkpoint for all three tasks. Lower error and higher R@10 are better. Conditional positive/zero errors cannot simply be added to recover the main score. These screening contrasts use one training seed, so they have no seed SD; Section 6.2 separately reports fixed-recipe repetitions.

| Recipe / window | Selected epoch | Completion142 | Legacy log | Positive | Zero | Metabolome45 | All187 | Name-only142 | R@10 full / 30% |
|---|---|---|---|---|---|---|---|---|---|
| 1e-4 / 20 | 18 | 0.219576 | 0.066269 | 0.358686 | 0.093135 | 0.564114 | 0.302486 | 0.407723 | 0.8061% / 0.3807% |
| 1e-4 / 60 | 60 | 0.194543 | 0.058632 | 0.324743 | 0.086138 | 0.577952 | 0.286807 | 0.414298 | 1.2212% / 0.5618% |
| 3e-4 / 20 | 18 | 0.213405 | 0.063450 | 0.348083 | 0.110594 | 0.587303 | 0.303380 | 0.418662 | 0.5665% / 0.4491% |
| 3e-4 / 60 | 60 | 0.183553 | 0.054362 | 0.302773 | 0.092039 | 0.598022 | 0.283292 | 0.426411 | 1.0603% / 0.6781% |

Frozen RF32: completion142=0.189031, legacy log=0.056270, name-only=0.659518, full/30% R@10=0.4457%/0.1987%. Frozen XGB32:0.174487,0.052593,0.656148,0.6275%/0.1819%. Fixed name KNN32: name-only0.262260 and R@10=19.3572%/9.6722%; it is not substituted for a family-completion baseline. See the [frozen baseline appendix](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_EN.md) for all six completed tree configurations and costs.

At3e-4, primary improvement over RF is2.898%, paired food-group95% interval[0.681%,5.061%]; legacy improvement3.390% [0.237%,6.528%]. Primary error is5.196% worse than XGB, with improvement interval[-8.081%,-2.340%]. At1e-4, primary error is2.916% worse than RF with an interval crossing zero; legacy regression4.198% fails the2% guardrail.

Direct learning-rate contrast:3e-4 improves primary by5.649% [0.407%,10.135%] and legacy by7.282% [2.255%,12.134%]. Name-only worsens2.924% at the point estimate, improvement interval[-6.356%,0.366%]; neither harm nor equivalence is established. Full-profile R@10 decreases0.1609 percentage points, difference interval[-0.5006,0.1476] points. Partial-profile R@10 increases0.1163 points, interval[-0.1465,0.3616] points.

Best-through20 to best-through60: primary improvement11.401% [10.060%,12.683%] at1e-4 and13.988% [12.260%,15.684%] at3e-4. These nested windows share a60-epoch schedule and the longer window has more validation-selection opportunities. They do not isolate duration from selection or compare independent20/60 schedules.

Machine outputs retain per-axis, source, context and positive/zero results. Intervals use1,000 paired whole-food-group resamples conditional on the selected models. Nutrient cells or seed-food repetitions are not independent foods. Training-seed, repeated-selection, label and external-generalization uncertainties are excluded.

Failure record: the original first audit failed when Windows CP936 decoded UTF-8 candidate names. Its records remain intact. A versioned audit changes only explicit encoding; the first model was not retrained. The recovery queue and all stage analyses completed successfully. No training NaN/Inf occurred.

### 6.2 Independent results from three fixed-recipe seeds

Only the seed changes between repetitions. Each model is scored first and errors are then averaged; predictions are not ensembled and the best seed is not selected. Mean ± sample SD describes these three seeds, not the entire training-seed population.

|Seed|Selected epoch|Completion142|Legacy log142|Name-only142|R@10 full|R@10 30%|
|---|---|---|---|---|---|---|
|20260922|60|0.183553|0.054362|0.426411|0.010603|0.006781|
|20260923|60|0.187083|0.057067|0.409478|0.009537|0.004091|
|20260924|57|0.196917|0.060931|0.451395|0.009037|0.003394|

|Model|Completion 142|Legacy log 142|Name-only 142|Completion 45|Completion 187|
|---|---|---|---|---|---|
|transformer|0.189184 ± 0.006925|0.057454 ± 0.003301|0.429095 ± 0.021087|0.591929 ± 0.008974|0.286102 ± 0.005996|
|rf|0.189031|0.056270|0.659518|0.674196|0.305782|
|xgb|0.174487|0.052593|0.656148|0.674273|0.294756|
|name_knn|—|—|0.262260|—|—|

Each frozen tree reference is one previously fitted realization, without an invented seed SD. KNN is a fixed name baseline and is not a matched numeric-context completion comparator here. Tree name-only evaluation masks all nutrition inputs of completion-trained models; these are not separately optimized name-only trees. KNN uses per-axis observed training-name pools and Euclidean KDTree 10-neighbor search in the same PCA32 space. Weights are source-balanced cell weights divided by distance plus 0.001; transformed labels are averaged before inversion. K was not retuned.

### 6.3 Subsets, paired intervals and retrieval

|Task|Axes|Scaled MAE|Legacy log MAE|Raw MAE|Positive MAE|Zero MAE|
|---|---|---|---|---|---|---|
|completion|142|0.189184 ± 0.006925|0.057454 ± 0.003301|0.348136 ± 0.021134|0.310514 ± 0.011019|0.092668 ± 0.002517|
|completion|45|0.591929 ± 0.008974|0.035336 ± 0.000486|0.064796 ± 0.001569|0.802609 ± 0.011679|0.224642 ± 0.008340|
|completion|187|0.286102 ± 0.005996|0.052131 ± 0.002397|0.279952 ± 0.016300|0.428933 ± 0.009461|0.116876 ± 0.002874|
|name_only|142|0.429095 ± 0.021087|0.151743 ± 0.005202|0.937501 ± 0.046174|0.552585 ± 0.018126|0.410400 ± 0.045874|
|name_only|45|0.824736 ± 0.085107|0.057004 ± 0.006606|0.104080 ± 0.022302|0.977394 ± 0.100290|0.519521 ± 0.042180|
|name_only|187|0.524302 ± 0.033943|0.128945 ± 0.004151|0.736945 ± 0.029735|0.654812 ± 0.036603|0.430416 ± 0.035985|

Positive and zero errors have different conditional denominators. Raw MAE is in g/100g. Positive gains below favor the Transformer. Intervals use 1000 paired food-group resamples conditional on these three trained models and fixed tree artifacts; they exclude selection bias, tree randomness and label validity.

|Reference|Task|Metric|Gain (%)|95% conditional interval (%)|
|---|---|---|---|---|
|rf|completion|scaled_log_mae|-0.081|[-2.173, 2.074]|
|rf|completion|log_mae|-2.103|[-6.178, 1.689]|
|rf|name_only|scaled_log_mae|34.938|[33.048, 36.845]|
|rf|name_only|log_mae|29.034|[26.112, 31.515]|
|xgb|completion|scaled_log_mae|-8.423|[-11.026, -5.622]|
|xgb|completion|log_mae|-9.242|[-14.649, -3.638]|
|xgb|name_only|scaled_log_mae|34.604|[32.594, 36.637]|
|xgb|name_only|log_mae|28.999|[25.978, 31.717]|
|name_knn|name_only|scaled_log_mae|-63.614|[-68.879, -58.393]|
|name_knn|name_only|log_mae|-70.329|[-81.946, -58.985]|

Retrieval values are proportions between 0 and 1. Full/30%-visible panels have 10,479/8,610 queries and 7,090/6,458 food groups, with 49,913 candidate names. Eligible foods differ across visibility settings; their score difference is not a pure visibility effect.

|Model|Visible fraction|MRR|R@1|R@5|R@10|
|---|---|---|---|---|---|
|transformer|0.3|0.002837 ± 0.000637|0.000413 ± 0.000089|0.002398 ± 0.001140|0.004755 ± 0.001789|
|transformer|1.0|0.005138 ± 0.000647|0.000889 ± 0.000289|0.005164 ± 0.000821|0.009726 ± 0.000800|
|rf|0.3|0.001477|0.000310|0.001239|0.001987|
|rf|1.0|0.002528|0.000176|0.001566|0.004457|
|xgb|0.3|0.001559|0.000310|0.001471|0.001819|
|xgb|1.0|0.003417|0.000423|0.002468|0.006275|
|name_knn|0.3|0.034512|0.005149|0.054796|0.096722|
|name_knn|1.0|0.068879|0.013005|0.117573|0.193572|

Per-axis/source results, sparse support counts and exploratory intervals: [Summary](../../../reports/v9_r9_three_seed_confirmation_v1/summary.json) / [Per-axis by seed](../../../reports/v9_r9_three_seed_confirmation_v1/axis_metrics_by_seed.csv) / [Per-source by seed](../../../reports/v9_r9_three_seed_confirmation_v1/source_metrics_by_seed.csv) / [Axis paired intervals](../../../reports/v9_r9_three_seed_confirmation_v1/axis_paired_intervals.csv). Axis intervals are not multiplicity-adjusted. Source coverage differs, so source means are not database-quality rankings. Raw predictions remain in local ignored directories.

### 6.4 Registered decision gates

|Registered gate|Result|
|---|---|
|three_registered_neural_seeds|PASS|
|primary_mean_improvement_over_fixed_rf|FAIL|
|food_group_interval_supports_improvement|FAIL|
|legacy_mean_regression_at_most_2_percent|FAIL|

## 7. Mechanisms, attribution and the recipe choice

Sections 7.1–7.2 retain the completed screening-group interpretation for seed 20260922; they are not three-seed mechanism confirmation. Subsequent runs change only the seed. Section 7.3 gives the aggregate decomposition.

### 7.1 Screening-group diagnostics

The [four-panel learning curve](../../../reports/v9_r9_first_group_v1/learning_curves.png) was visually inspected. Both models select epoch60; later improvement slows without sustained primary deterioration or numerical divergence. Last10-epoch mean clipping fractions are0.350883 at3e-4 and0.482988 at1e-4. This observation does not establish clipping as the causal mechanism.

Source-free evaluation of all1,828,536 training targets gives train/validation primary0.143092/0.194543 at1e-4 and0.134937/0.183553 at3e-4. Both improve at the larger LR. Foods and support differ between partitions, so this gap alone does not prove overfitting. The online187-axis calibrated loss is not directly comparable with validation142-axis MAE.

The3e-4 minus1e-4 primary difference is-0.010990: positive contribution-0.014577 plus zero contribution+0.003587. Against RF, positive extra error remains+0.017036, comprising underprediction+0.012476 and overprediction+0.004560; the zero benefit is-0.022514, yielding net-0.005478. Overall improvement does not mean better positive predictions or improvement on every nutrient.

At3e-4,76/142 axes have lower point error than RF. The7 axes with<100 training groups contribute-0.002453;20 axes with100–999 contribute-0.005975;115 with>=1000 contribute+0.002951. The sparse7 previously contributed+0.008669 at1e-4. Much of the learning-rate gain is therefore concentrated in low-support axes, making seed replication particularly valuable. Retain their per-axis intervals; do not change labels, scoring or sampling in response. Fatty acids contribute-0.006294 overall, while amino acids and minerals retain extra error.

Forty extreme success/failure numerical cases stay in ignored local data, not public documents, and are not a representative sample. Source/context strata have different coverage and do not constitute causal interventions.

### 7.2 Controlled evidence and attribution limits

Observed:3e-4 lowers training and validation completion errors; longer selection windows improve completion; name-only and retrieval do not improve uniformly. Supported interpretation is limited to this controlled single-seed setting: learning rate affects optimization outcomes under the fixed architecture, data, schedule and selection policy. The earlier RF gap cannot be attributed entirely to insufficient Transformer capacity.

Unresolved alternatives include sparse-axis and initialization sensitivity, validation-selection bias, source-calibration/MAE interactions, and capacity/loss effects on positive values. MSE, PDF256 capacity and source-weight ablations have not been run on this architecture. None is established as optimal or unnecessary. Neither the train/validation gap nor the single-seed interval proves seed-stable superiority.

### 7.3 Three-seed trajectories and fixed-denominator decomposition

![Three-seed learning curves](../../../reports/v9_r9_three_seed_curves_v1/learning_curves.png)

[Figure review](../../../reports/v9_r9_three_seed_curves_v1/visual_review.json) records the actual image inspection. The 187-axis calibrated training objective and 142-axis validation metric differ; subtracting them does not diagnose overfitting. Stars mark the same completion-selected checkpoint throughout, without reselecting positive/zero checkpoints.

The three terms below share the primary-metric denominator and sum to the overall Transformer-minus-RF error difference. Negative values favor the Transformer. They locate error contributions, rather than identifying causal interventions.

|Primary-error contribution: Transformer minus RF|Difference|
|---|---|
|positive_under|+0.018121239|
|positive_over|+0.004878008|
|explicit_zero|-0.022845780|

### 7.4 Extreme cases and label limitations

The additional three-seed decomposition is recorded in the [axis/seed diagnosis](../../../reports/v9_r9_seed_variation_v1/summary.json) and [partition contributions](../../../reports/v9_r9_seed_variation_v1/partition_contributions.csv). Mean error is below RF on 64/142 axes; 53 axes improve in every seed and 62 worsen in every seed. The 7/20/115 axes with fewer than 100, 100–999 and at least 1000 training candidate groups contribute −0.001095/−0.005428/+0.006676 to the primary difference, using the fixed 142-axis denominator. The gap is therefore not confined to sparse axes.

Seed 24 increases primary error by 0.013363 relative to seed 22. Axes with at least 1000 training groups contribute 0.008705; axes below 100 contribute 0.003346. Lignin alone contributes 0.002837, but has only 17 validation groups and cannot justify selecting a model or changing weights by itself. Nineteen of the 20 amino-acid axes worsen relative to RF in all three seeds. Nine nutrition axes have fewer than 30 validation groups; axis intervals are exploratory. This decomposition motivates an objective/optimization contrast but does not separate MAE, source calibration, capacity or initialization effects.

All 40 existing extreme profile-axis cases from the screening parent were checked. All 20 worse examples come from FooDB; 17 involve Cholesterol. In those 17 cases, Transformer predictions are at least 500-fold below the retained positive labels and RF is closer to the recorded values. This proves neither a unit error, RF leakage, nor superior nutritional truth of the Transformer. Original processing evidence is absent; label definitions, systematic calibration and positive underprediction remain alternative explanations. Extremes are not random food samples and counts are not macro-error contributions; see [Extreme-case review](../../../reports/v9_r9_parent_extreme_cases_review_v1/REPORT_EN.md). No labels, exclusions or metrics were changed in response.

### 7.5 Recipe decision and parameters not independently validated

The direct reason for choosing lr=3e-4 over 1e-4 is completion improvement under identical initialization, task order and budget. The 60-epoch window also improves over the first 20-epoch window. Repeating the fixed recipe with three seeds separates this optimization choice from seed variability. Final conditional decision: the recipe **does not pass the registered RF confirmation gates**.

Width 192, 3 layers, 6 heads, dropout 0.15, rank 16, source calibration and MAE have not each received a full current-protocol ablation. A sound implementation and reproducible settings do not establish that every parameter is optimal. The learning-rate contrast supports a local optimization choice, not a general causal advantage of attention over simple models. Unseen-source, category-held-out, few-label and frozen-representation transfer evidence remains incomplete; this is a nutrition representation research model.

## 8. Reproduction, failures, version decision and next questions

Data, split, text cache, scales and panel retain the frozen R0/R8 identities; test remains closed. Sections 2–5 give training and baseline details; [First-group decision and evidence](../r9/first_group_v1/REPORT_EN.md) retains preregistration and selection evidence. Run code and checkpoint identities follow. Documentation commits are distinct from numerical code changes; run manifests retain exact source hashes.

|Seed|Run code commit|Checkpoint SHA256|Elapsed (s)|
|---|---|---|---|
|20260922|ca403e61a1495924bbe8c1f13da3138fb5d91dc5|bf96ea0abf4f68ab77d24adf4e043ffc01472d85ff067e7d8b7b26a5e4fb8c44|12254.359|
|20260923|c46284ceecd6b544da791143c1df9b36a32efa18|09f728d3bb161c22d942d0d617895007a6a850df679d7e6ccb94aae796bd8092|10481.047|
|20260924|4c539761e85a74c69da2b66b022377ba187ffc5f|7c1fec8110ecf13f270d6f891af583cfa92b2563c9a3efa7cc4a18eef2e91684|10516.812|

|Method|Seed|Recorded elapsed (s)|
|---|---|---|
|transformer|20260922|12254.359|
|transformer|20260923|10481.047|
|transformer|20260924|10516.812|
|rf|fixed|5807.484|
|xgb|fixed|1686.562|
|name_knn|fixed|600.109|

Repetition, independent-audit and aggregation commands follow. Seed 22 reuses the audited screening model. Each additional seed receives an independent audit and fixed-reference comparisons before aggregation; manifests retain configurations and source hashes. These are records of existing runs; existing output directories reject overwrites and duplicate jobs should not be launched.

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_confirmation.py --plan experiments/foodnutrigpt_v9_research/r9/first_group_v1/confirmation_plan.json --seed 20260923 --output-dir output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260923
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_r9_completed_utf8.py --run output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260923 --output-dir reports/v9_r9_confirmation_seed20260923_audit_v1
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_confirmation.py --plan experiments/foodnutrigpt_v9_research/r9/first_group_v1/confirmation_plan.json --seed 20260924 --output-dir output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260924
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_r9_completed_utf8.py --run output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260924 --output-dir reports/v9_r9_confirmation_seed20260924_audit_v1
.venv/Scripts/python.exe scripts/confirm_foodnutrigpt_r9_fixed_references.py --output-dir reports/v9_r9_three_seed_confirmation_v1
.venv/Scripts/python.exe scripts/plot_foodnutrigpt_r9_confirmation.py --output-dir reports/v9_r9_three_seed_curves_v1
```

Recorded elapsed times include within-run evaluation/checks and are not isolated training-speed comparisons; historical concurrent loads differed. The 1e-4 screening run adds 10,551.985 seconds, and the full-training fit diagnostic 110.75 seconds. The repetition table already includes reused seed 22; it must not be counted again as new training. Earlier costs and failures remain in each version's README and machine manifests.

Independent audits verify all 60 epoch orders, exposures and learning rates, and exactly replay two 323,809-row prediction tables, 49,913 candidate vectors and 19,089 retrieval ranks per completed model. An earlier audit failed because Windows CP936 decoded a UTF-8 name file, stopping dependent queues. A separate corrected audit specifies UTF-8 reads; original failures remain, and training was not repeated to hide the I/O failure. Historical R4 numerical failures, the KNN batch-dependence correction and partial/unrun R8 trees remain documented rather than counted as successful configurations.

Version decision: the fixed MAE recipe does not pass the registered RF confirmation gates. It remains a validated training control, not an accepted final method superior to RF. An MSE contrast changing only the training objective will be registered separately; its results are absent from this report. Source-held-out evaluation and low-label transfer remain future research questions. Test remains closed.

Repeated validation use, sparse axes, few seeds, single fixed tree realizations, and unresolved units, aliases and source borrowing limit every claim. R0 label-validity restrictions remain in force. Model comparison and nutritional truth require different evidence. Machine-readable references and report hashes appear in [evidence.json](evidence.json).
