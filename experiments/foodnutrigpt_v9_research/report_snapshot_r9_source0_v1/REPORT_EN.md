# Nutrition representation model: integrated R0–R9 research report

Chinese counterpart: [REPORT_ZH.md](REPORT_ZH.md). This report consolidates completed experiments and verification; internal validation does not establish external transfer or label validity.

Version scope: completed R0–R9 stage report, including three-seed MAE confirmation and single-factor MSE, axis-by-value and source-calibration-removal screens. All three added recipes fail their acceptance conditions; MAE with source weight=1 remains the current control. Sections 2–11 preserve previously reviewed historical evidence, including next-step statements from those historical stages. Section 12 records the latest complete version. The overall Transformer-over-frozen-RF objective remains unachieved.

## 1. Research objective and principal findings

The current objective is to improve nutrition completion through validated single-stage Transformer methods while keeping data and previously trained RF/XGBoost results frozen. Name-only prediction and nutrition-to-name retrieval remain tracked tasks. Historical MLP findings are retained as method exploration, not substituted for Transformer results.

The fixed lr=3e-4, 60-epoch recipe has three-seed primary completion error **0.189184 ± 0.006925**. Mean improvement over frozen RF is **-0.081%**, with a conditional food-group 95% interval of **[-2.173%, 2.074%]**. Mean relative improvement in legacy log-MAE is **-2.103%**. The recipe **does not pass the registered RF confirmation gates**. Sections 5–6 give all references, individual seeds and aggregate results; the interval is not a guarantee for arbitrary new seeds or datasets.

The registered gates require lower mean primary error than fixed RF, a food-group paired interval supporting improvement, and at most 2% relative regression in mean legacy nutrition log-MAE. There is no 5% minimum improvement or requirement to beat XGBoost. Each seed uses its completion-selected checkpoint for all three tasks, without selecting different task-specific winners.

The preceding MSE contrast completed 60 epochs and independent replay; the registered rule selected epoch 56. Primary completion error is 0.193006, a 5.150% regression against same-seed MAE, with a food-group 95% regression interval of [2.316%, 9.257%]; legacy log-MAE regresses 10.622%. All three screening gates fail. This fixed MSE recipe is rejected as a completion upgrade and will not receive the other two seeds. It does not achieve the Transformer-over-frozen-RF objective. Section 9 gives all tasks, mechanisms and failures; Section 10 adds parameter diagnostics.

The preceding axis-by-value recipe completed 60 epochs, independent replay and 11 comparisons. Primary completion error is 0.187792: a 2.309% point regression against same-seed MAE, with an improvement interval of [−5.099%, +0.263%]; legacy log-MAE regresses by 3.063% at the point estimate. All three registered screening gates fail. This fixed recipe is rejected as a completion upgrade and will not receive seeds 23/24. Its 0.655% point gain over frozen RF has an interval of [−1.888%, +3.017%] and only one seed, so it does not establish stable superiority over RF. Section 11 gives the complete results and interpretation.

Removing source calibration completed 60 epochs in 10,516.218 seconds and selected epoch 58 by the fixed validation primary metric. Primary completion error is 0.187741, a 2.281% point regression against same-seed MAE, with an improvement interval of [−5.219%, +0.430%]; legacy log-MAE regresses by 4.314% at the point estimate. All three preregistered screening gates fail, so this fixed recipe is rejected without seeds 23/24. Its 0.683% point gain over frozen RF has an interval of [−2.544%, +3.791%], crosses zero and comes from one seed; stable superiority is not established. Name-only error also increases. Section 12 provides the complete evidence and interpretation.

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

The [four-panel learning curve](../../../reports/v9_r9_first_group_v1/learning_curves.png) was visually inspected. Both models select epoch 60; later improvement slows without sustained primary deterioration or numerical divergence. Last10-epoch mean clipping fractions are0.350883 at3e-4 and0.482988 at1e-4. This observation does not establish clipping as the causal mechanism.

Source-free evaluation of all 1,828,536 training targets gives train/validation primary0.143092/0.194543 at1e-4 and0.134937/0.183553 at3e-4. Both improve at the larger LR. Foods and support differ between partitions, so this gap alone does not prove overfitting. The online187-axis calibrated loss is not directly comparable with validation142-axis MAE.

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

Width192,3 layers,6 heads, dropout0.15, rank16 and source calibration have not each received a full current-protocol ablation. Section 9 supports MAE over the fixed same-seed MSE recipe, without exhausting objectives or their optimization settings. A sound implementation and reproducible settings do not establish that every parameter is optimal. The learning-rate contrast supports a local optimization choice, not a general causal advantage of attention over simple models. Unseen-source, category-held-out, few-label and frozen-representation transfer evidence remains incomplete; this is a nutrition representation research model.

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

Version decision: the fixed MAE recipe does not pass the registered RF confirmation gates. It remains a validated training control, not an accepted final method superior to RF. That decision subsequently produced the completed MSE contrast; Section 9 gives its results and rejection rationale. Source-held-out evaluation and low-label transfer remain future research questions. Test remains closed.

Repeated validation use, sparse axes, few seeds, single fixed tree realizations, and unresolved units, aliases and source borrowing limit every claim. R0 label-validity restrictions remain in force. Model comparison and nutritional truth require different evidence. Machine-readable references and report hashes appear in [evidence.json](evidence.json).

## 9. MSE objective contrast: completed, upgrade recipe rejected

### 9.1. Research question and hypothesis

The hypothesis is that squared-residual training reduces positive underprediction seen with the MAE control. Explicit zeros, legacy log-MAE and both auxiliary tasks track possible costs. The PDF motivates macro-axis MSE, but its different data and scoring prevent historical score comparisons.

### 9.2. Parent and controlled intervention

Candidate 3 of 12 changes only MAE to MSE. Seed 20260922, initialization, all 60 epoch orders, learning-rate schedule, data, source weights and scoring match the control. The model has width 192, 3 layers, 6 heads, feedforward width 768, residual rank 16 and dropout 0.15. AdamW uses a learning rate of 3e-4 and weight decay of 1e-4, with batch size 64 and gradient clipping at 1. Base and calibrated losses are averaged; source-residual L2 is 1e-4. No two-stage training or tree refitting.

### 9.3. Reproduction

|Item|Value|
|---|---|
|Code commit|aa909f767c877d16945e4acef8a0d9bb430f97ae|
|Checkpoint SHA256|578cce5bd43d7b214b0f1368f97f1b54ab47984089665811bc2cf86abd2f2d85|
|Initial-state SHA256|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Selected epoch|56|
|Training elapsed (s)|10389.016|
|Training-fit inference elapsed (s)|57.047|

The manifest retains environment and exact source identities. Elapsed time includes within-run checks and evaluation, rather than isolated training speed. MAE reuses the parent and must not be counted again as new training.

[Registration](../r9/mse_v1/PLAN.md) · [Configuration](../r9/mse_v1/config.json) · [Analysis](../../../reports/v9_r9_mse_analysis_v1/summary.json) · [Training fit](../../../reports/v9_r9_mse_fit_v1/summary.json)

### 9.4. Complete results

MAE and MSE rows both use seed 22 and are distinct from the three-seed MAE mean in Section 6. RF/XGB/KNN are frozen. Each model uses its completion-selected checkpoint for every task. Conditional positive/zero errors have different denominators and cannot be added to recover primary error.

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|MSE|0.193006|0.060136|0.365023|0.268409|0.122075|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|MSE|0.457182|0.157425|0.905488|0.517463|0.504336|
|RF32|0.659518|0.213824|1.134940|0.695565|0.730742|
|XGB32|0.656148|0.213719|1.173381|0.693838|0.718390|
|KNN32|0.262260|0.089088|0.502782|0.338107|0.246578|

**45/187 axes**

|Method|Task|Subset|Axes|Scaled-log MAE|
|---|---|---|---|---|
|MAE|completion|food_metabolome|45|0.598022|
|MAE|completion|all|187|0.283292|
|MAE|name_only|food_metabolome|45|0.749716|
|MAE|name_only|all|187|0.504212|
|MSE|completion|food_metabolome|45|0.590152|
|MSE|completion|all|187|0.288576|
|MSE|name_only|food_metabolome|45|0.815017|
|MSE|name_only|all|187|0.543292|
|RF32|completion|food_metabolome|45|0.674196|
|RF32|completion|all|187|0.305782|
|RF32|name_only|food_metabolome|45|0.889240|
|RF32|name_only|all|187|0.714798|
|XGB32|completion|food_metabolome|45|0.674273|
|XGB32|completion|all|187|0.294756|
|XGB32|name_only|food_metabolome|45|0.851545|
|XGB32|name_only|all|187|0.703168|
|KNN32|name_only|food_metabolome|45|0.702450|
|KNN32|name_only|all|187|0.368188|

**检索 / Retrieval**

|Method|Visible fraction|Recall@1|Recall@5|Recall@10|MRR|
|---|---|---|---|---|---|
|MAE|0.3|0.000465|0.003452|0.006781|0.003381|
|MAE|1.0|0.001023|0.006003|0.010603|0.005720|
|MSE|0.3|0.000728|0.002573|0.006702|0.003591|
|MSE|1.0|0.001760|0.006705|0.011745|0.006941|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

There are 49,913 fixed candidate names, with vectors generated only from name predictions. Queries contain no names. Correct answers use exact original-name matching; no confirmed alias map exists. Conditional food-group intervals exclude seed-population, label-validity and repeated-selection uncertainty.

|Reference|Task|Metric|MSE gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-5.149879|-9.256798|-2.316294|
|mae_parent|completion|log_mae|-10.621548|-15.172193|-6.586417|
|mae_parent|name_only|scaled_log_mae|-7.216188|-9.913146|-4.638114|
|mae_parent|name_only|log_mae|-2.098213|-6.474531|2.049716|
|rf32|completion|scaled_log_mae|-2.102890|-5.179971|0.569103|
|rf32|completion|log_mae|-6.871111|-11.032837|-3.098267|
|rf32|name_only|scaled_log_mae|30.679347|28.460938|32.780659|
|rf32|name_only|log_mae|26.376228|23.344967|29.121762|
|xgb32|completion|scaled_log_mae|-10.613556|-14.127177|-7.494001|
|xgb32|completion|log_mae|-14.343356|-19.397318|-9.133662|
|xgb32|name_only|scaled_log_mae|30.323296|27.940966|32.473146|
|xgb32|name_only|log_mae|26.340148|23.192573|28.969658|
|knn32|name_only|scaled_log_mae|-74.323978|-80.280923|-68.396460|
|knn32|name_only|log_mae|-76.707504|-89.422073|-64.245350|

|Reference|Visible fraction|Metric|MSE minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|0.000209|-0.000789|0.001170|
|mae_parent|0.3|recall_at_1|0.000263|-0.000511|0.001038|
|mae_parent|0.3|recall_at_5|-0.000879|-0.002726|0.000916|
|mae_parent|0.3|recall_at_10|-0.000079|-0.002615|0.002568|
|mae_parent|1.0|mrr|0.001221|-0.000224|0.002672|
|mae_parent|1.0|recall_at_1|0.000737|-0.000383|0.001966|
|mae_parent|1.0|recall_at_5|0.000702|-0.001817|0.003189|
|mae_parent|1.0|recall_at_10|0.001142|-0.001947|0.004206|
|rf32|0.3|mrr|0.002114|0.001237|0.003036|
|rf32|0.3|recall_at_1|0.000418|-0.000279|0.001177|
|rf32|0.3|recall_at_5|0.001334|-0.000039|0.002847|
|rf32|0.3|recall_at_10|0.004715|0.002706|0.006942|
|rf32|1.0|mrr|0.004413|0.003286|0.005708|
|rf32|1.0|recall_at_1|0.001583|0.000701|0.002689|
|rf32|1.0|recall_at_5|0.005139|0.003218|0.007283|
|rf32|1.0|recall_at_10|0.007288|0.004714|0.010051|
|xgb32|0.3|mrr|0.002032|0.001195|0.002969|
|xgb32|0.3|recall_at_1|0.000418|-0.000279|0.001223|
|xgb32|0.3|recall_at_5|0.001102|-0.000377|0.002539|
|xgb32|0.3|recall_at_10|0.004883|0.002722|0.007013|
|xgb32|1.0|mrr|0.003524|0.002268|0.004878|
|xgb32|1.0|recall_at_1|0.001337|0.000349|0.002492|
|xgb32|1.0|recall_at_5|0.004236|0.002177|0.006385|
|xgb32|1.0|recall_at_10|0.005470|0.002570|0.008425|
|knn32|0.3|mrr|-0.030922|-0.033227|-0.028571|
|knn32|0.3|recall_at_1|-0.004421|-0.006232|-0.002821|
|knn32|0.3|recall_at_5|-0.052223|-0.057609|-0.047063|
|knn32|0.3|recall_at_10|-0.090020|-0.096773|-0.083183|
|knn32|1.0|mrr|-0.061938|-0.065575|-0.058669|
|knn32|1.0|recall_at_1|-0.011245|-0.014051|-0.008741|
|knn32|1.0|recall_at_5|-0.110869|-0.118453|-0.103745|
|knn32|1.0|recall_at_10|-0.181826|-0.191254|-0.173341|

### 9.5. Mechanism diagnostics

|Fixed-denominator component|MSE minus MAE|
|---|---|
|positive_under|-0.005673|
|positive_over|0.003953|
|explicit_zero|0.011173|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|MSE|train|0.130030|0.185514|0.077244|
|MSE|validation|0.193006|0.268409|0.122075|

|Objective|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|MSE|0.394570|0.141807|56|10389.016|

![MAE and MSE learning curves](../../../reports/v9_r9_mse_analysis_v1/learning_curves.png)

The common-denominator contributions reconstruct the primary difference exactly, and source-free fit over all 1,828,536 training targets was verified. The six-panel figure was actually viewed: text, legends and the MAE epoch 60/MSE epoch 56 markers are legible. The generation receipt remains unchanged; a separate [visual-review receipt](../../../reports/v9_r9_mse_analysis_v1/visual_review.json) records inspection. All 1000 food-group draws in each current nutrition-prediction comparison retain all 142 axes; none are discarded for missing-axis support.

Under common source-free scoring, training primary error decreases from 0.134937 to 0.130030 while validation increases from 0.183553 to 0.193006. The training-fit gain does not transfer to this validation panel; this alone establishes neither overfitting nor insufficient capacity. Foods, supports and selection differ between partitions. Online MAE and MSE objective magnitudes are not compared across losses.

With a common denominator, positive underprediction decreases by 0.005673, but positive overprediction increases by 0.003953, leaving a positive-only net gain of 0.001720. An explicit-zero cost of 0.011173 raises total error by 0.009453. The clipping fraction in the last 10 epochs is 0.141807 for MSE versus 0.350883 for MAE, contradicting an explanation based on more frequent late clipping. A higher early mean gradient norm does not imply more clipping throughout training; no gradient-scale control experiment was performed.

Of 142 axes, 31 improve in point estimate; 7 unadjusted axis intervals support improvement and 63 support regression. Nineteen of 20 amino-acid axes worsen in point estimate. These exploratory intervals have no multiplicity correction; 9 axes have fewer than 30 validation food groups, so the counts are not independent confirmations. Seven axes below 100 training groups contribute +0.004495 and 115 axes with at least 1000 contribute +0.004529. Both sparse and common axes are involved. Source contributions include different coverage and are not database-quality rankings.

|Partition|Stratum|Supported axes|MSE minus MAE contribution|
|---|---|---|---|
|mask_family|amino_acid|20|0.001165|
|mask_family|fatty_acid|54|0.005487|
|mask_family|vitamin|30|-0.001197|
|source_key|bls_4_0|17|-0.000283|
|source_key|cnf|98|0.001132|
|source_key|foodb|89|0.004807|
|source_key|frida|72|0.000998|
|source_key|usda_sr_legacy|103|0.001233|
|support_bin|100_to_999|20|0.000428|
|support_bin|at_least_1000|115|0.004529|
|support_bin|below_100|7|0.004495|

The complete [142-axis changes and support](../../../reports/v9_r9_mse_axis_changes_v1/axis_changes.csv) and [all source/family/concentration partitions](../../../reports/v9_r9_mse_analysis_v1/mse_minus_mae_partitions.csv) are retained. The two largest axis contributions are Isomeric linolenic acids (18:3), +0.001882, and Lignin, +0.001455, each with only 17 validation groups; the former's unadjusted interval crosses zero. Neither axis justifies relabeling, excluding samples or changing the primary metric.

All 40 existing MSE-versus-RF extreme cases were reconstructed and checked: [verification](../../../reports/v9_r9_mse_cases_review_v2/summary.json). The 20 worse cases contain 13 explicit zeros and 16 FooDB rows; 10 involve Biotin, 7 Dodecanoic acid and 3 other axes. The 20 better cases include 17 positives and 3 zeros across 6 sources. These profile-axis tails include repeated foods, are not random samples, and do not represent macro-error contributions. No Cholesterol case appears in the worse MSE tail; this does not establish that earlier problems are resolved or labels are correct.

Numerical training, replay and formal analysis succeeded. A dependent job failed before launch because repeated UTC parsing introduced an 8-hour shift; correction preserved the original training. Case-review v1 emitted vacuous all-true flags for an empty Cholesterol set. V2 marks it not applicable/null and retains v1; all 40 selected cases, predictions and metrics are unchanged. Individual concentrations and predictions remain only in local data directories.

### 9.6. Limits of causal interpretation

The objective substitution is controlled but also changes gradient scale, clipping and optimization trajectory. Improvements cannot uniquely establish a mean-versus-median mechanism; failure rejects this recipe under its fixed optimization settings. Train/validation foods, supports and selection differ, so fit gaps alone do not establish overfitting. Axis/source decompositions are exploratory; data, weights and gates were not changed in response.

### 9.7. Screening and version decision

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

Does not meet the gates for separately registered seed confirmation. Only one MSE seed exists; this is not final model acceptance.

Version decision: reject this fixed MSE recipe as a completion upgrade, do not add seeds 23/24, and retain all negative results and the MAE control. MAE three-seed confirmation itself did not establish RF superiority; the objective choice is not completion of the overall goal. See the [eight-section version record](../r9/mse_v1/README.md). Name-only primary error regresses 7.216% against MAE, and all 8 retrieval contrast intervals cross zero; higher retrieval point estimates do not compensate for completion regression.

### 9.8. Next step and test status

Retain MAE as the validated control. The next question is whether a shared numeric encoder plus axis identity limits axis-value interactions. The proposed single intervention is a zero-initialized axis-by-value linear residual, keeping the current Transformer, objective, budget and every frozen data identity. It remains an unregistered, untrained hypothesis, not a demonstrated gain. Added capacity and inductive structure must also be distinguished. Feature-specific numeric vectors are motivated by the [FT-Transformer paper](https://arxiv.org/html/2106.11959v5); the proposed residual is not a reproduction of that model or its results. Test remains closed; source-held-out and low-label transfer remain unverified.

## 10. MAE source-offset stability: descriptive appendix

|MAE seed pair|Nutrition offset Pearson r|Difference RMS|
|---|---|---|
|20260922/20260923|0.973741|0.014104|
|20260922/20260924|0.971479|0.016519|
|20260923/20260924|0.969257|0.016309|

This diagnostic reads parameters from the three audited MAE checkpoints, with no forward pass or optimization. Nutrition offsets over 24 training sources are correlated, but this does not establish calibration benefit or prediction stability. Large-offset axes may dominate correlation, and selected epochs are 60/60/57. No calibration-on/off ablation was performed.

[Parameter evidence](../../../reports/v9_r9_mae_calibration_stability_v1/summary.json)

## 11. Axis-by-value residual: screen complete, fixed upgrade recipe rejected

### 11.1. Research question and preregistered hypothesis

The hypothesis is that an explicit axis-by-value interaction improves conditional numeric representations. The preceding MSE recipe was rejected. This experiment retains MAE and changes only numeric tokens. Amino-acid, fatty-acid and positive/zero errors are diagnostics, not substitutes for the 142-axis primary metric.

### 11.2. Parent and controlled intervention

Candidate 4 of 12 uses the same-seed MAE model as its control and starts from scratch. Tokens change from e_axis+g(t) to e_axis+g(t)+t*r_axis, adding 252×192 = 48,384 zero-initialized parameters. Hidden values contribute no residual. Shared initial weights, constructor RNG and forward predictions at zero residual match exactly. Width 192, 3 layers, 6 heads, feedforward width 768, dropout 0.15, head rank 16, MAE, source-loss weight 1, source L2 of 1e-4, batch size 64, AdamW learning rate 3e-4/weight decay 1e-4, clipping at 1 and the 60-epoch cosine schedule are unchanged. No tree refitting or data changes.

### 11.3. Reproduction

|Item|Value|
|---|---|
|Code commit|89698ddeeda3377286c180e028119f00f4f06852|
|Checkpoint SHA256|512d963a4053262d90accaeae98f1c8d267f8501e9f7e1b69c6dac157e38a70c|
|Full initial-state SHA256|225b0385dd3ef60215abcf3543c00f63406c56ba8cf9836ef1ac1026244e6bbd|
|Shared parent initial-state SHA256|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Parameters|1596978|
|Trainable parameters|1564313|
|Added parameters|48384|
|Selected epoch|60|
|Run elapsed (s)|10564.781|
|Fit inference elapsed (s)|57.672|

Independent replay checks all 60 epoch orders, target exposures and learning rates, two prediction tables of 323,809 rows each, 49,913 candidate vectors and 19,089 retrieval ranks. The version record and manifest retain the environment and commands. Elapsed run time includes evaluation; the reused MAE control is not counted again as new training.

[Registration](../r9/axisvalue_v1/PLAN.md) · [Configuration](../r9/axisvalue_v1/config.json) · [Version record](../r9/axisvalue_v1/README.md) · [Analysis](../../../reports/v9_r9_axisvalue_analysis_v1/summary.json) · [Training fit](../../../reports/v9_r9_axisvalue_fit_v1/summary.json)



```powershell
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_axisvalue.py --candidate tf192_mae_axisvalue_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_axisvalue.py --run output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60 --output-dir reports/v9_r9_axisvalue_lr3e4_60_audit_v1
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_axisvalue.py --output-dir reports/v9_r9_axisvalue_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_axisvalue_cases.py --output-dir reports/v9_r9_axisvalue_cases_v1
.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_axisvalue_axes.py
```

Environment: Windows, Python 3.10.19, PyTorch 2.7.1+cu128, RTX 5070 Ti 16GB. These commands have completed and existing directories reject overwrite; manifests retain complete commands and dependency hashes. Formal training, replay and comparisons had no failures. Case loading emitted a pandas mixed-type warning, but exact values and metadata passed the explicit checks.

### 11.4. Complete results

MAE and AXISVALUE both use seed 22; RF/XGB/KNN results are frozen. Each neural model uses its completion-selected checkpoint for all tasks. Conditional positive and zero errors have different denominators and cannot be added to recover primary error. The historical three-seed MAE mean in Section 6 is distinct from these single-seed results.

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|AXISVALUE|0.187792|0.056027|0.337126|0.306668|0.085240|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|AXISVALUE|0.419489|0.150305|0.909482|0.544610|0.375529|
|RF32|0.659518|0.213824|1.134940|0.695565|0.730742|
|XGB32|0.656148|0.213719|1.173381|0.693838|0.718390|
|KNN32|0.262260|0.089088|0.502782|0.338107|0.246578|

**45/187 axes**

|Method|Task|Subset|Axes|Scaled-log MAE|
|---|---|---|---|---|
|MAE|completion|food_metabolome|45|0.598022|
|MAE|completion|all|187|0.283292|
|MAE|name_only|food_metabolome|45|0.749716|
|MAE|name_only|all|187|0.504212|
|AXISVALUE|completion|food_metabolome|45|0.579883|
|AXISVALUE|completion|all|187|0.282145|
|AXISVALUE|name_only|food_metabolome|45|0.878140|
|AXISVALUE|name_only|all|187|0.529859|
|RF32|completion|food_metabolome|45|0.674196|
|RF32|completion|all|187|0.305782|
|RF32|name_only|food_metabolome|45|0.889240|
|RF32|name_only|all|187|0.714798|
|XGB32|completion|food_metabolome|45|0.674273|
|XGB32|completion|all|187|0.294756|
|XGB32|name_only|food_metabolome|45|0.851545|
|XGB32|name_only|all|187|0.703168|
|KNN32|name_only|food_metabolome|45|0.702450|
|KNN32|name_only|all|187|0.368188|

**检索 / Retrieval**

|Method|Visible fraction|Recall@1|Recall@5|Recall@10|MRR|
|---|---|---|---|---|---|
|MAE|0.3|0.000465|0.003452|0.006781|0.003381|
|MAE|1.0|0.001023|0.006003|0.010603|0.005720|
|AXISVALUE|0.3|0.000632|0.002013|0.004341|0.002922|
|AXISVALUE|1.0|0.001481|0.004060|0.008550|0.005392|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

Retrieval uses 49,913 fixed candidate names. Candidate vectors come only from name predictions, with no measured candidate nutrition; queries contain no names. Correct answers still use exact original-name matching, without a confirmed alias map. Conditional food-group intervals do not cover seed-population, label-validity or repeated-selection uncertainty.

|Reference|Task|Metric|AXISVALUE gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-2.309245|-5.099278|0.262962|
|mae_parent|completion|log_mae|-3.062976|-7.865762|0.976248|
|mae_parent|name_only|scaled_log_mae|1.623499|-0.920789|4.037533|
|mae_parent|name_only|log_mae|2.519862|-2.238321|7.260391|
|rf32|completion|scaled_log_mae|0.655429|-1.887559|3.016789|
|rf32|completion|log_mae|0.431201|-4.232860|4.237859|
|rf32|name_only|scaled_log_mae|36.394649|34.184365|38.709253|
|rf32|name_only|log_mae|29.706355|26.084824|33.035853|
|xgb32|completion|scaled_log_mae|-7.625321|-11.094921|-4.308238|
|xgb32|completion|log_mae|-6.530480|-13.040331|0.095379|
|xgb32|name_only|scaled_log_mae|36.067954|33.814848|38.440340|
|xgb32|name_only|log_mae|29.671907|25.932994|33.012904|
|knn32|name_only|scaled_log_mae|-59.951434|-65.062756|-54.786306|
|knn32|name_only|log_mae|-68.714726|-79.771306|-58.301770|

|Reference|Visible fraction|Metric|AXISVALUE minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|-0.000459|-0.001386|0.000453|
|mae_parent|0.3|recall_at_1|0.000168|-0.000581|0.000890|
|mae_parent|0.3|recall_at_5|-0.001439|-0.003195|0.000291|
|mae_parent|0.3|recall_at_10|-0.002440|-0.004658|-0.000073|
|mae_parent|1.0|mrr|-0.000329|-0.001599|0.000835|
|mae_parent|1.0|recall_at_1|0.000458|-0.000588|0.001528|
|mae_parent|1.0|recall_at_5|-0.001943|-0.004215|0.000165|
|mae_parent|1.0|recall_at_10|-0.002054|-0.004857|0.001027|
|rf32|0.3|mrr|0.001446|0.000635|0.002257|
|rf32|0.3|recall_at_1|0.000323|-0.000374|0.001032|
|rf32|0.3|recall_at_5|0.000774|-0.000607|0.002104|
|rf32|0.3|recall_at_10|0.002354|0.000521|0.004142|
|rf32|1.0|mrr|0.002864|0.001880|0.003965|
|rf32|1.0|recall_at_1|0.001305|0.000470|0.002233|
|rf32|1.0|recall_at_5|0.002494|0.000897|0.004173|
|rf32|1.0|recall_at_10|0.004093|0.001612|0.006595|
|xgb32|0.3|mrr|0.001363|0.000591|0.002164|
|xgb32|0.3|recall_at_1|0.000323|-0.000348|0.001071|
|xgb32|0.3|recall_at_5|0.000542|-0.000826|0.001910|
|xgb32|0.3|recall_at_10|0.002521|0.000877|0.004290|
|xgb32|1.0|mrr|0.001974|0.000900|0.003135|
|xgb32|1.0|recall_at_1|0.001058|0.000141|0.002045|
|xgb32|1.0|recall_at_5|0.001592|-0.000098|0.003409|
|xgb32|1.0|recall_at_10|0.002275|-0.000344|0.004973|
|knn32|0.3|mrr|-0.031590|-0.033910|-0.029264|
|knn32|0.3|recall_at_1|-0.004516|-0.006349|-0.002864|
|knn32|0.3|recall_at_5|-0.052783|-0.058309|-0.047599|
|knn32|0.3|recall_at_10|-0.092382|-0.099002|-0.085861|
|knn32|1.0|mrr|-0.063487|-0.067102|-0.060062|
|knn32|1.0|recall_at_1|-0.011524|-0.014273|-0.009174|
|knn32|1.0|recall_at_5|-0.113513|-0.121315|-0.106566|
|knn32|1.0|recall_at_10|-0.185022|-0.194445|-0.176533|

### 11.5. Mechanism diagnostics

|Fixed-denominator component|AXISVALUE minus MAE|
|---|---|
|positive_under|0.004091|
|positive_over|0.000922|
|explicit_zero|-0.000774|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|AXISVALUE|train|0.135435|0.247870|0.051219|
|AXISVALUE|validation|0.187792|0.306668|0.085240|

|Method|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|AXISVALUE|0.400266|0.344902|60|10564.781|

![MAE control and axis-value learning curves](../../../reports/v9_r9_axisvalue_analysis_v1/learning_curves.png)

The common-denominator decomposition reconstructs the primary increase of 0.004238696 exactly: positive underprediction adds 0.004090529, positive overprediction adds 0.000921943, and explicit-zero improvement contributes −0.000773776. All six curve panels were visually inspected; both models select epoch 60. The last-ten-epoch clipping fraction is 0.344902, below the control at 0.350883, which does not support increased clipping frequency as the explanation. Source-free inference over all 1,828,536 training targets also raises primary error slightly, from 0.134937 to 0.135435; this alone cannot distinguish optimization, capacity or generalization mechanisms. Conditional positive and zero macro averages have their own denominators, so lower values for both do not guarantee a lower overall macro average.

Against same-seed MAE, 60/142 axes improve at the point estimate, with 14 axis intervals supporting improvement and 20 supporting regression (without multiplicity correction); 9 axes have fewer than 30 validation food groups. The 7 axes with fewer than 100 training groups contribute +0.004304441, the 20 with 100–999 contribute +0.000573374, and the 115 with at least 1000 contribute −0.000639120. Lignin and Isomeric linolenic acids contribute large sparse-axis differences, each with only 17 validation groups; the latter interval crosses zero. This does not establish sparse-axis support as a causal bottleneck. Fatty acids contribute +0.002138810 and dietary fibre +0.001829756; the FooDB source partition contributes +0.004233540. Source, family and support partitions describe overlapping decompositions of the same error and must not be added together.

Forty extreme RF-comparison cases were reconstructed and checked against saved predictions. All 20 worsening cases come from FooDB; 19 are positive Cholesterol labels with predictions at least 500-fold below the retained label. This proves neither a unit error nor label correctness. The 20 improving cases contain 14 positive and 6 zero labels across 5 sources and 13 candidate food groups. These extremes are neither representative nor 40 independent samples, and do not justify relabeling or exclusion. Individual food names, concentrations and predictions remain in the local data directory.

For 142-axis completion, conditional positive error rises from 0.302773 to 0.306668 while zero error falls from 0.092039 to 0.085240. Name-only primary error improves by 1.623% at the point estimate, but its [−0.921%, +4.038%] interval crosses zero; its 45-axis and 187-axis errors worsen. Full/30%-visible R@10 are 0.008550/0.004341, both below MAE. The 30%-visible R@10 difference interval is [−0.004658, −0.000073]; the other seven same-seed retrieval intervals cross zero. These conditional intervals are not adjusted for multiple-metric selection, and an auxiliary metric cannot substitute for the completion screen.

[Visual review](../../../reports/v9_r9_axisvalue_analysis_v1/visual_review.json) · [Axis contrasts](../../../reports/v9_r9_axisvalue_axis_changes_v1/summary.json) · [Case review](../../../reports/v9_r9_axisvalue_cases_v1/summary.json)

Eleven functional tests passed. A 50-step check on 32 actual training tasks reduced loss from 0.217358 to 0.044068, demonstrating learnability only; formal training resets parameters and RNG. An initial pytest invocation failed during collection because PYTHONPATH was missing. It passed after the execution environment was corrected, without restarting a formal candidate.

### 11.6. Limits of causal interpretation

Observed facts are that this fixed intervention fails the single-seed improvement gates, positive-label costs outweigh zero-label gains, and training primary error also rises slightly. The control supports rejecting this complete recipe, not a claim that all axis-specific encodings are ineffective. The additional 48,384 parameters, interaction form and optimization trajectory change together, preventing unique attribution. Another learning rate, regularization setting or matched-capacity control was not evaluated. Conditional intervals exclude seed-population, repeated-selection, label-validity and external-transfer uncertainty. Data, scales, metrics and gates were not changed in response to results.

### 11.7. Screening and version decision

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

Does not meet the preregistered screening gates. Only seed 20260922 has been evaluated; this does not establish a stable improvement or superiority over RF.

Version decision: reject the current axis-by-value recipe as a completion upgrade, retain every artifact, do not add seeds 20260923/20260924, and continue using the original MAE model as the method control. Intervals crossing zero establish neither equivalence nor universal harm; this decision applies the preregistered acceptance rule. It does not complete the overall Transformer-over-frozen-RF objective. See the [machine-readable decision](../r9/axisvalue_v1/decision.json).

### 11.8. Next questions and test status

The next priority is to test training-only source calibration: compare source weight 1→0 on the original MAE control, retaining loss normalization by (1+w) and all other training settings, and first verify matching shared-parameter gradients at zero residual initialization. This question is not yet registered or trained. Stable source offsets do not establish a benefit, and current source-partition differences do not establish a calibration mechanism. Future long runs will use estimated-duration follow-ups based on the existing approximately three-hour runtime, ending per-epoch observation. Data and RF/XGB remain frozen, test remains closed, and source-held-out transfer, few-shot transfer and a reliable foundation-model claim remain unverified.

## 12. Removing source calibration: screen complete, fixed upgrade recipe rejected

### 12.1. Question and hypothesis

Training fits both base and known-source calibrated outputs, whereas inference evaluates the base output. This experiment tests whether removing the calibration objective improves base predictions. Stable source offsets and source-partition errors establish neither benefit nor harm.

### 12.2. Parent and intervention

Candidate 5 of 12 uses the same-seed MAE control and changes only source weight from 1 to 0. The loss remains (L_base+w*L_calibrated)/(1+w)+1e-4*mean(R_train²); normalization avoids simultaneously rescaling the base objective. Architecture, zero-initialized source table, L2, optimizer and source-balanced sample weights are retained. Source offsets stay zero at w=0. Width 192, 3 layers, 6 heads, FF768, dropout 0.15, head rank 16, MAE, batch 64, AdamW lr 3e-4/weight decay 1e-4, clipping 1 and the 60-epoch cosine schedule are unchanged.

### 12.3. Reproduction

|Item|Value|
|---|---|
|Code commit|92f80d90b798822d39af4644d65ea9441e52d042|
|Checkpoint SHA256|172d5af07d978672824daa3ba9e376d5c7fb94504e81d39b54ef3cd83f107c73|
|Initial-state SHA256 (same as parent)|eedddcd2b1ef26824401f65ad7fe72fa3dc003a2d12298bd7a596d92f6a0b3fe|
|Data SHA256|48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923|
|Panel SHA256|3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a|
|Name cache SHA256|db361b9bcd3aa4d9b37c31a01ee08140c29976989b0e78b3ae6e8be3dfa6c12a|
|Parameters|1548594|
|Parameters with requires_grad|1515929|
|Source weight|0.0|
|Selected epoch|58|
|Run elapsed (s)|10516.218|
|Fit inference elapsed (s)|56.656|

All 60 epoch orders, target exposures and learning rates match the parent. Independent replay checks two 323,809-row prediction tables, 49,913 candidate vectors and 19,089 ranks exactly. Selected and final source tables are zero; they remain counted as requires_grad parameters but have zero effective gradients. Environment: Windows, Python 3.10.19, PyTorch 2.7.1+cu128, RTX 5070 Ti 16GB. Formal training retains the parent CUDA backend; deterministic settings apply only to the functional reference process. The pre-run estimate was about 3 hours with a 190-minute follow-up; an estimate is not actual compute cost.

[Registration](../r9/source0_v1/PLAN.md) · [Configuration](../r9/source0_v1/config.json) · [Version](../r9/source0_v1/README.md) · [Analysis](../../../reports/v9_r9_source0_analysis_v1/summary.json) · [Training fit](../../../reports/v9_r9_source0_fit_v1/summary.json) · [Functional checks](../../../reports/v9_r9_source0_functional_v1/verification.json)



```powershell
.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_source0.py --candidate tf192_mae_source0_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_source0_lr3e4_60
.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_source0.py --run output/v9_r9_methods/tf192_mae_source0_lr3e4_60 --output-dir reports/v9_r9_source0_lr3e4_60_audit_v1
.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_source0.py --output-dir reports/v9_r9_source0_analysis_v1
.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_source0_fit.py --output-dir reports/v9_r9_source0_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_source0_fit_v1
```

These runs completed. Independent audit and all 11 comparisons passed without a formal training failure. A pandas mixed-type warning during case loading did not affect value and metadata checks. The functional-preflight failure and recovery are recorded below. Additional training-fit inference took 56.656 seconds without optimizer updates.

### 12.4. Complete results

MAE and SOURCE0 both use seed 22; RF/XGB/KNN results are frozen. Each neural model uses one completion-selected checkpoint for all tasks. The historical three-seed MAE mean in Section 6 is distinct from these single-seed comparisons.

**补全 / Completion**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.183553|0.054362|0.330240|0.302773|0.092039|
|SOURCE0|0.187741|0.056708|0.344686|0.304187|0.090369|
|RF32|0.189031|0.056270|0.321219|0.248086|0.168645|
|XGB32|0.174487|0.052593|0.298575|0.230310|0.146588|

**仅名称 / Name-only**

|Method|142-axis scaled-log MAE|Legacy log-MAE|Raw MAE (g/100g)|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|---|
|MAE|0.426411|0.154190|0.977888|0.552216|0.385183|
|SOURCE0|0.444413|0.156721|0.974482|0.554378|0.424196|
|RF32|0.659518|0.213824|1.134940|0.695565|0.730742|
|XGB32|0.656148|0.213719|1.173381|0.693838|0.718390|
|KNN32|0.262260|0.089088|0.502782|0.338107|0.246578|

**45/187 axes**

|Method|Task|Subset|Axes|Scaled-log MAE|
|---|---|---|---|---|
|MAE|completion|food_metabolome|45|0.598022|
|MAE|completion|all|187|0.283292|
|MAE|name_only|food_metabolome|45|0.749716|
|MAE|name_only|all|187|0.504212|
|SOURCE0|completion|food_metabolome|45|0.592878|
|SOURCE0|completion|all|187|0.285234|
|SOURCE0|name_only|food_metabolome|45|0.845787|
|SOURCE0|name_only|all|187|0.541000|
|RF32|completion|food_metabolome|45|0.674196|
|RF32|completion|all|187|0.305782|
|RF32|name_only|food_metabolome|45|0.889240|
|RF32|name_only|all|187|0.714798|
|XGB32|completion|food_metabolome|45|0.674273|
|XGB32|completion|all|187|0.294756|
|XGB32|name_only|food_metabolome|45|0.851545|
|XGB32|name_only|all|187|0.703168|
|KNN32|name_only|food_metabolome|45|0.702450|
|KNN32|name_only|all|187|0.368188|

**检索 / Retrieval**

|Method|Visible fraction|Recall@1|Recall@5|Recall@10|MRR|
|---|---|---|---|---|---|
|MAE|0.3|0.000465|0.003452|0.006781|0.003381|
|MAE|1.0|0.001023|0.006003|0.010603|0.005720|
|SOURCE0|0.3|0.000426|0.002839|0.004762|0.002786|
|SOURCE0|1.0|0.000404|0.005040|0.009677|0.004908|
|RF32|0.3|0.000310|0.001239|0.001987|0.001477|
|RF32|1.0|0.000176|0.001566|0.004457|0.002528|
|XGB32|0.3|0.000310|0.001471|0.001819|0.001559|
|XGB32|1.0|0.000423|0.002468|0.006275|0.003417|
|KNN32|0.3|0.005149|0.054796|0.096722|0.034512|
|KNN32|1.0|0.013005|0.117573|0.193572|0.068879|

Retrieval uses 49,913 fixed candidate names and name-predicted vectors without measured candidate nutrition; nutrient queries contain no names. Correct answers use exact original-name matching without confirmed aliases. Food-group conditional intervals exclude seed-population, label-validity and repeated-selection uncertainty.

|Reference|Task|Metric|SOURCE0 gain (%)|95% lower (%)|95% upper (%)|
|---|---|---|---|---|---|
|mae_parent|completion|scaled_log_mae|-2.281351|-5.218685|0.429649|
|mae_parent|completion|log_mae|-4.314022|-9.464234|0.242264|
|mae_parent|name_only|scaled_log_mae|-4.221660|-7.037420|-1.251824|
|mae_parent|name_only|log_mae|-1.641299|-7.268159|3.944402|
|rf32|completion|scaled_log_mae|0.682515|-2.544394|3.790670|
|rf32|completion|log_mae|-0.777430|-6.360213|4.283938|
|rf32|name_only|scaled_log_mae|32.615460|30.424656|34.933673|
|rf32|name_only|log_mae|26.705711|22.994630|30.025658|
|xgb32|completion|scaled_log_mae|-7.595977|-11.287282|-4.005461|
|xgb32|completion|log_mae|-7.823616|-15.191958|-1.165845|
|xgb32|name_only|scaled_log_mae|32.269354|29.886638|34.557613|
|xgb32|name_only|log_mae|26.669792|22.731187|30.075311|
|knn32|name_only|scaled_log_mae|-69.455141|-74.486058|-64.052129|
|knn32|name_only|log_mae|-75.916697|-85.459950|-66.514116|

|Reference|Visible fraction|Metric|SOURCE0 minus reference|95% lower|95% upper|
|---|---|---|---|---|---|
|mae_parent|0.3|mrr|-0.000596|-0.001522|0.000277|
|mae_parent|0.3|recall_at_1|-0.000039|-0.000698|0.000619|
|mae_parent|0.3|recall_at_5|-0.000613|-0.002459|0.001174|
|mae_parent|0.3|recall_at_10|-0.002019|-0.004355|0.000381|
|mae_parent|1.0|mrr|-0.000812|-0.001971|0.000283|
|mae_parent|1.0|recall_at_1|-0.000618|-0.001422|0.000150|
|mae_parent|1.0|recall_at_5|-0.000963|-0.003221|0.001481|
|mae_parent|1.0|recall_at_10|-0.000927|-0.003826|0.002342|
|rf32|0.3|mrr|0.001309|0.000516|0.002114|
|rf32|0.3|recall_at_1|0.000116|-0.000504|0.000774|
|rf32|0.3|recall_at_5|0.001600|0.000155|0.002994|
|rf32|0.3|recall_at_10|0.002774|0.000916|0.004633|
|rf32|1.0|mrr|0.002380|0.001600|0.003162|
|rf32|1.0|recall_at_1|0.000228|-0.000207|0.000776|
|rf32|1.0|recall_at_5|0.003474|0.001831|0.005313|
|rf32|1.0|recall_at_10|0.005220|0.002609|0.007660|
|xgb32|0.3|mrr|0.001227|0.000476|0.001969|
|xgb32|0.3|recall_at_1|0.000116|-0.000466|0.000774|
|xgb32|0.3|recall_at_5|0.001368|-0.000052|0.002840|
|xgb32|0.3|recall_at_10|0.002942|0.001329|0.004777|
|xgb32|1.0|mrr|0.001491|0.000640|0.002447|
|xgb32|1.0|recall_at_1|-0.000019|-0.000583|0.000569|
|xgb32|1.0|recall_at_5|0.002572|0.000780|0.004500|
|xgb32|1.0|recall_at_10|0.003402|0.000728|0.006095|
|knn32|0.3|mrr|-0.031727|-0.034128|-0.029425|
|knn32|0.3|recall_at_1|-0.004723|-0.006465|-0.003071|
|knn32|0.3|recall_at_5|-0.051957|-0.057276|-0.047090|
|knn32|0.3|recall_at_10|-0.091961|-0.098826|-0.085333|
|knn32|1.0|mrr|-0.063971|-0.067440|-0.060863|
|knn32|1.0|recall_at_1|-0.012601|-0.015250|-0.010235|
|knn32|1.0|recall_at_5|-0.112533|-0.120300|-0.105451|
|knn32|1.0|recall_at_10|-0.183895|-0.193454|-0.175241|

### 12.5. Mechanism diagnostics

|Fixed-denominator component|SOURCE0 minus MAE|
|---|---|
|positive_under|0.002142|
|positive_over|0.001669|
|explicit_zero|0.000377|

|Method|Partition|142-axis MAE|Positive MAE|Explicit-zero MAE|
|---|---|---|---|---|
|MAE|train|0.134937|0.250149|0.054250|
|MAE|validation|0.183553|0.302773|0.092039|
|SOURCE0|train|0.136431|0.245846|0.052563|
|SOURCE0|validation|0.187741|0.304187|0.090369|

|Method|First 10 clip fraction|Last 10 clip fraction|Selected epoch|Elapsed (s)|
|---|---|---|---|---|
|MAE|0.401310|0.350883|60|12254.359|
|SOURCE0|0.402164|0.347503|58|10516.218|

![Source-loss weight 1 versus 0 learning curves](../../../reports/v9_r9_source0_analysis_v1/learning_curves.png)

The common-denominator decomposition reconstructs the primary increase of 0.004187495 exactly: positive underprediction contributes +0.002141826, positive overprediction +0.001668563 and explicit zeros +0.000377106. Conditional zero macro error falls from 0.092039 to 0.090369 while its contribution to overall primary error increases, because the statistics use different within-axis denominators and effective weights. A lower conditional zero mean must not be called an overall benefit. Source-free inference on all 1,828,536 training targets increases primary error from 0.134937 to 0.136431, providing no evidence of improved base fitting. Lower conditional positive and zero training means likewise cannot be added to infer primary improvement.

All six curve panels were visually inspected. SOURCE0 selects epoch 58 and MAE selects epoch 60; both curves flatten late, without proving global optimality. Last-ten-epoch clipping is 0.347503 versus 0.350883 for the control, which does not support more frequent late clipping as the explanation. Mean gradient norms fluctuate without a divergent trend. Removing calibration changes the meaning of training loss; therefore mechanism comparisons use common source-free fit metrics rather than interpreting raw training-loss levels as a performance comparison.

58/142 axes improve at the point estimate; 11 axis intervals support improvement and 26 support regression, without multiplicity correction, while 9 axes have fewer than 30 validation food groups. The 7 axes below 100 training groups contribute +0.003714232, the 20 with 100–999 contribute +0.000491940, and the 115 with at least 1000 contribute −0.000018677. Dietary fibre contributes +0.002794349 and fatty acids +0.001034044; the FooDB source partition contributes +0.003903528. Lignin and Isomeric linolenic acids each have only 17 validation food groups and both axis intervals cross zero, so they are not established causal bottlenecks. Source, family and support partitions overlap and must not be added together.

Forty extreme RF-comparison cases were reconstructed and checked against saved predictions. All 20 worsening cases come from FooDB: 19 positive labels and 1 explicit zero across 16 food groups. Ten Cholesterol predictions are at least 500-fold below the retained labels, proving neither unit error nor label correctness. The 20 improving cases include 15 positive and 5 zero labels across 5 sources and 15 food groups. These tails are neither representative nor independent samples and do not justify data changes. Individual names, concentrations and predictions remain in the local data directory.

Name-only 142-axis error is 0.444413, a 4.222% regression against same-seed MAE; its improvement interval of [−7.037%, −1.252%] supports this conditional regression. It remains better than frozen RF/XGB on name-only prediction but substantially worse than name kNN at 0.262260, so that task is not solved. Full/30%-visible retrieval R@10 is 0.009677/0.004762. All eight same-seed retrieval metrics worsen at the point estimate and all difference intervals cross zero. Completion improves on 45 axes but worsens on 187 axes; auxiliary metrics do not replace the 142-axis screen. Every task uses the same epoch 58 checkpoint.

[Visual review](../../../reports/v9_r9_source0_analysis_v1/visual_review.json) · [Axis contrasts](../../../reports/v9_r9_source0_axis_changes_v1/summary.json) · [Case review](../../../reports/v9_r9_source0_cases_v1/summary.json)

|Functional exercise|Value|
|---|---|
|Training tasks|32|
|Steps|50|
|Loss before|0.217358|
|Loss after|0.058560|

Fourteen shared/synthetic functional tests passed. Ordinary-GPU preflight failed a bitwise shared-gradient comparison before optimizer steps; repeating the same objective also produced tiny differences. CPU and deterministic-GPU references matched exactly within each device without relaxing assertions. The full 50-step check passed, offsets remained zero and the learned public-loader reload was exact. Failure and recovery are recorded; these checks support implementation and learnability, not validation performance.

### 12.6. Causal limits

Observed facts are that removing calibration fails every gate for this fixed recipe and also raises source-free training primary error. The control supports retaining source weight=1 and rejecting this removal recipe, not a universal benefit of calibration or harm from all removal schemes. Parameter count and initialization match, but the intervention changes calibration gradients and subsequent optimization. Real source-bias correction, implicit regularization and optimization effects remain inseparable. Ordinary GPU execution has small nondeterministic variation, and single-seed conditional intervals exclude seed-population, repeated-selection, label-validity and external-transfer uncertainty. Data, metrics and gates were not changed to obtain a win.

### 12.7. Screen and decision

|Registered screening gate vs same-seed MAE|Pass|
|---|---|
|primary_point_improves_over_same_seed_mae|False|
|paired_interval_supports_improvement|False|
|legacy_regression_at_most_2percent_vs_mae|False|

Does not meet the registered screen; one seed cannot confirm stable improvement or superiority over RF.

Version decision: reject this SOURCE0 recipe as a completion upgrade, do not add seeds 20260923/20260924, retain all artifacts and failures, and continue using MAE with source weight=1 as the method control. An interval crossing zero does not establish equivalence; retaining the control follows the predeclared acceptance conditions rather than proving every parameter optimal. This version does not complete the overall Transformer-over-frozen-RF objective. See the [machine-readable decision](../r9/source0_v1/decision.json).

### 12.8. Next questions and test status

The next priority is the original PDF capacity combination: width 256, 8 heads and FF1024 versus width 192, 6 heads and FF768, retaining MAE, source weight 1 and the other training settings. Greater conditional representation capacity may improve fit but may also increase generalization costs. Width, head count and feedforward width change together, so this is not a pure width-effect experiment. The direction appears in the R9 master plan, but a concrete candidate has not yet been registered or trained. Data and RF/XGB remain frozen, test stays closed, and future runs continue to use runtime estimates and scheduled follow-ups. Source-held-out transfer, few-shot transfer and a reliable foundation-model claim remain unverified.