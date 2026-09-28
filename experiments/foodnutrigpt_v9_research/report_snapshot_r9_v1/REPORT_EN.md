# Nutrition Representation Model: R9 Research Snapshot

**Status: research in progress, not the final deliverable.** This version includes the first independently audited R9 Transformer, frozen RF/XGBoost references, name neighbors and the R0–R8 research history. The second learning-rate candidate, complete budget comparison, training-fit diagnostics and Transformer seed confirmation are unfinished. Interim epochs are excluded from the completed-results tables. The Chinese counterpart is [REPORT_ZH.md](REPORT_ZH.md).

## 1. Current question and interim conclusion

The active goal is to improve a Transformer beyond the existing RF through validated method changes, while keeping data and RF/XGBoost results frozen. The earlier reporting endpoint without a superiority requirement and the MLP results remain historical context; they do not replace this goal. Both languages must explain sound training, all iterations and attribution, parameter selection and the three tasks.

The first complete Transformer has primary completion error **0.194543**, versus **0.189031** for frozen RF32. Estimated relative improvement is −2.916%, with a food-group 95% interval of [−7.503%, +2.033%]; this does not establish superiority. Name-only prediction improves on trees but remains weaker than name neighbors, and absolute retrieval performance is low. No Transformer has been finally selected and no general nutrition foundation model has been established.

Acceptance of a fixed Transformer recipe requires seeds 20260922, 20260923 and 20260924, mean primary error below frozen RF with a paired food-group interval supporting improvement, and no more than 2% relative regression in legacy nutrition log-MAE. Beating XGBoost is not required. The historical minimum 5% gain is not retained; labels, tasks, metrics and frozen references are unchanged.

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
|`learning_rates`|0.0001, 0.0003|
|`weight_decay`|0.0001|
|`gradient_clip`|1.0|
|`source_weight`|1.0|
|`source_residual_l2`|0.0001|
|`eta_min_fraction`|0.01|
|`seed`|20260922|
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

The screening seed is 20260922. A promising fixed recipe would subsequently be confirmed with 20260922, 20260923 and 20260924. Food-group paired resampling intervals do not remove repeated-validation selection bias or include the frozen RF's seed uncertainty. Acceptance requires mean Transformer primary error below the fixed RF with a paired interval supporting improvement, while relative regression in legacy nutrition log-MAE is at most 2%; superiority to XGBoost is not required.

### Parameter evidence and unresolved questions

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

### Reproduction and evidence limits

The original run implementation commit is `b4fd4c1`. Later documentation commits are distinct from changes to the training implementation; each run's source snapshot and hashes establish exact code identity. The environment is Python3.10.19, PyTorch2.7.1+cu128 and an RTX5070Ti16GB, with four CPU training threads. Actual costs come from completed run manifests and histories, not planned timing estimates.

```powershell
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr1e4_60 --output-dir output/v9_r9/tf192_mae_lr1e4_60
.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate tf192_mae_lr3e4_60 --output-dir output/v9_r9/tf192_mae_lr3e4_60
```

These record the registered commands; existing output directories cannot be overwritten and the commands are not instructions to launch duplicate jobs. Completed runs undergo independent score replay, budget and learning-rate comparisons, and full-training fit diagnostics. Functional verification is complete but does not replace performance evidence. The full study still requires eight-section iteration reports, all three tasks, replicated confirmation and final bilingual attribution. This appendix does not predeclare the final recipe or establish foundation-model transfer capabilities.

Evidence index: [configuration](../r9/config.json), [preregistration](../r9/PLAN.md), [freeze receipt](../../../reports/v9_r9_freeze_v1/manifest.json), [functional verification](../../../reports/v9_r9_functional_v1/verification.json), [data summary](../../../reports/v9_final_data_evidence_v1/summary.json), and [baseline methods](../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_EN.md). Numeric predictions and checkpoints remain in local ignored directories; this appendix contains no raw food-composition values.

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
|[R9](../r9/README.md)|Common PCA32, width192 direct Transformer, MAE, source1, lr1e-4/3e-4; first60-epoch run complete, second awaiting full analysis|First primary0.194543 does not beat RF. Negative result retained; no final recipe|

The historical MLP/PCA128 recipe achieved three-seed completion error 0.183435 ± 0.001222 and name-only error 0.484042 ± 0.010458. These are not results of the current PCA32 Transformer. Its name-removal/permutation findings and common-axis underprediction diagnosis cannot automatically be transferred to the Transformer. Full parameters, seed results and failures remain in the [earlier snapshot](../final_report_v1/REPORT_EN.md) and individual iteration records.

Numeric-basis components underwent train-only feasibility checks but were not used in the current R9 training. Only two learning-rate candidates are registered in the first R9 group; feasibility checks are not model-performance evidence.


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

The matched 32-dimensional R9 references are frozen RF leaf1/feature0.5 and XGB depth10. They are references from the completed set, not optima of the unfinished search or across all random seeds. Only the first complete Transformer result is currently available; see Section 6. This appendix establishes neither a neural win nor universal label validity or foundation-model capability.

Per-axis execution checked reversed order, subbatches and in-memory pickle replay; completed audits also verified data, features, weights and saved predictions/retrieval ranks. Fitted forests were discarded after prediction, so saved artifacts do not support arbitrary new-name online tree inference. Numeric predictions and candidate matrices remain in local ignored directories; this appendix contains aggregates only.

Data manifest SHA256: `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`

Freeze manifest SHA256: `1ec58002d76beb329004db77ef512622f57b40e142725799918d20027b5fddf3`

Code, environment, full parameters, run hashes and fit-record hashes are in the [machine-readable results](../../../reports/v9_r9_frozen_baseline_appendix_v2/summary.json). Every numeric table is identical across languages. A completed appendix is not the final study report.

## 6. Completed results: common-PCA32 internal validation

All rows use the same frozen evaluation protocol. Lower errors are better; trees and the Transformer have only screening seed 20260922. KNN is deterministic and has no invented seed standard deviation. KNN was not established as a completion comparator in this group, hence the dash. All three Transformer tasks use the same selected checkpoint, without mixing the best specialist models.

|Method|Completion142|Legacy log142|Name-only142|Completion45|Completion187|
|---|---|---|---|---|---|
|Transformer lr1e-4|0.194543|0.058632|0.414298|0.577952|0.286807|
|RF32|0.189031|0.056270|0.659518|0.674196|0.305782|
|XGB32|0.174487|0.052593|0.656148|0.674273|0.294756|
|KNN32|—|—|0.262260|—|—|

Relative improvements below are percentages; positive values favor the Transformer. Intervals condition on the selected single-seed models.

|Reference|Task|Metric|Gain (%)|95% interval (%)|
|---|---|---|---|---|
|rf32|completion|scaled_log_mae|-2.916|[-7.503, 2.033]|
|rf32|completion|log_mae|-4.198|[-9.931, 0.724]|
|rf32|name-only|scaled_log_mae|37.182|[34.601, 39.705]|
|rf32|name-only|log_mae|31.636|[27.825, 34.985]|
|xgb32|completion|scaled_log_mae|-11.495|[-16.915, -6.036]|
|xgb32|completion|log_mae|-11.483|[-19.664, -3.924]|
|xgb32|name-only|scaled_log_mae|36.859|[34.135, 39.436]|
|xgb32|name-only|log_mae|31.602|[27.791, 34.955]|
|knn32|name-only|scaled_log_mae|-57.972|[-63.536, -52.804]|
|knn32|name-only|log_mae|-64.083|[-73.198, -55.274]|

Complete subset metrics for the same Transformer:

|Task|Axes|Scaled MAE|Legacy log MAE|Raw MAE|Positive MAE|Zero MAE|
|---|---|---|---|---|---|---|
|completion|142|0.194543|0.058632|0.357728|0.324743|0.086138|
|completion|45|0.577952|0.032940|0.059523|0.808772|0.176678|
|completion|187|0.286807|0.052449|0.285968|0.441221|0.102746|
|name_only|142|0.414298|0.146179|0.901488|0.544090|0.406688|
|name_only|45|0.760225|0.056509|0.113176|0.922643|0.440394|
|name_only|187|0.497543|0.124600|0.711787|0.635186|0.412871|

Retrieval values are fractions in [0,1], higher is better, with a fixed library of 49,913 names. Visibility strata have different eligible query populations, not a pure causal visibility contrast.

|Method|Visible fraction|MRR|R@1|R@5|R@10|
|---|---|---|---|---|---|
|Transformer lr1e-4|0.3|0.003381|0.000336|0.003141|0.005618|
|Transformer lr1e-4|1.0|0.006852|0.001622|0.006617|0.012212|
|RF32|0.3|0.001477|0.000310|0.001239|0.001987|
|RF32|1.0|0.002528|0.000176|0.001566|0.004457|
|XGB32|0.3|0.001559|0.000310|0.001471|0.001819|
|XGB32|1.0|0.003417|0.000423|0.002468|0.006275|
|KNN32|0.3|0.034512|0.005149|0.054796|0.096722|
|KNN32|1.0|0.068879|0.013005|0.117573|0.193572|

Full and 30% retrieval cover 7,090/6,458 food groups. Per-metric paired intervals remain in the eight comparison records referenced by evidence.json. Despite beating trees on name-only, the Transformer has 57.972% higher error than KNN. Full-input R@10 is about 1.22% versus 19.36% for KNN, so good name performance has not been established.


## 7. Mechanisms, attribution and current decision

The first candidate selected epoch 60 and took 10,551.985 seconds, about 2.93 hours, including in-run evaluation and checks. Its saved through-20 snapshot selected epoch 18; the formal three-task budget comparison still awaits both complete runs. A favorable final epoch does not prove that further training would help. Online 187-axis loss with source calibration cannot be subtracted from the 142-axis validation metric to diagnose overfitting.

The net primary gap against RF is +0.005512. With the original metric denominator retained, additional positive underprediction contributes +0.025472, additional overprediction +0.006141, and explicit-zero gains −0.026100. Eight partitions reconstruct the main score within 1e-12. Lower point errors on 72/142 axes are not confirmed individual-axis wins.

Seven axes supported by fewer than 100 training food groups contribute +0.008669; 20 axes with 100–999 groups offset −0.004691, while 115 axes with at least 1000 groups contribute +0.001534. Isomeric linolenic acids (18:3) has only 68 training and 17 validation groups, an axis error gap of +0.944753, and a food-group interval of [−0.184128, 1.997180]. All seven sparse-axis intervals cross zero. Thus the earlier MLP explanation emphasizing common axes does not transfer to this Transformer. This still does not prove that sparse-axis sampling, capacity or MAE caused the gap.

**Observed facts** are the positive/zero trade-off and support-stratified differences. **Controlled evidence** currently includes functional checks; complete learning-rate and budget contrasts are pending. **Alternative explanations** include optimization, shared representation, sparse-supervision variability, calibration and generalization. Both learning-rate runs and source-free full-training fit diagnostics precede registration of another single-method intervention; loss, capacity, sampling and metrics will not all change together.

The first candidate fails the primary acceptance condition and its 4.198% point regression in legacy log-MAE exceeds the guard. Its complete result is retained but it is not accepted as the final improvement. Final selection requires a recipe fixed after screening and then three-seed confirmation. Unseen-source/category transfer, few-label performance and frozen-representation probes remain unestablished.

## 8. Failures, reproducibility and remaining work

After successful first-candidate training, the original independent audit failed because Windows CP936 could not decode UTF-8 candidate_names.json; three dependent queues then stopped. The correction explicitly specifies UTF-8 in four reads, with AST checks showing no training or scoring logic change. The corrected full audit exactly replayed both 323,809-row prediction tables, all 49,913 candidate vectors and 19,089 rankings. The first model was not retrained, original failure records were preserved, and the second candidate started with its registered recipe. See the [encoding incident record](../r9/AUDIT_UTF8_NOTE.md). This was an audit I/O failure, not a training-method failure or performance improvement.

The first run's implementation commit is b4fd4c1; the later evidence commit is 186ea75. Checkpoint SHA256 is `b9ee82eda2edeec17c62ac6e9b67df5b2dded0b6bbf53ab976e4e1319527f2a0`. Complete data, cache, panel, source and result hashes are in [evidence.json](evidence.json) and linked run receipts. The environment is Windows, Python 3.10.19, PyTorch 2.7.1+cu128 and RTX5070Ti16GB. Measured durations are not controlled algorithm-speed comparisons.

Full axis/source and paired results remain in the [RF comparison](../../../reports/v9_r9_tf192_mae_lr1e4_60_vs_rf32_completion_v1/summary.json), [independent audit](../../../reports/v9_r9_tf192_mae_lr1e4_60_audit_v1/verification.json) and [fixed-denominator diagnosis](../../../reports/v9_r9_tf192_lr1e4_rf32_gap_v1/summary.json). Sources cover different axes, so their means do not directly rank database quality. Original-value examples, predictions, caches and models remain in local ignored directories.

This snapshot corrects obsolete plans for additional tree fitting, a final MLP winner and untrained R9 status. Final reports still require the second candidate and subsequent method iterations, complete curves and fit diagnostics, three-seed confirmation of a fixed Transformer, and the final selection rationale. All current results use repeatedly consulted internal validation. Food-group intervals exclude training seeds, selection bias, frozen-tree randomness and label validity. The test stays closed and unit provenance, aliases and copied records remain unresolved. Having both reports does not establish completion of the research goal.
