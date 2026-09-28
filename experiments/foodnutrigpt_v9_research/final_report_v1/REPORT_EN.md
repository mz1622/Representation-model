# Nutrition Representation Model: Research Report

**Status: working draft; experiments and comparisons are being finalized (28 September 2026).** Data, methods, R0–R7 and the neural three-seed results are documented. The last two R8 RF configurations, repetitions of the selected trees, final paired intervals and final version decision remain incomplete. This is not the final deliverable. The corresponding Chinese draft is [REPORT_ZH.md](REPORT_ZH.md).

## 1. Research questions and scope of conclusions

The project studies three tasks: missing-nutrient completion from a food name and partial profile; prediction from the original food name alone; and ranking previously unseen candidate names from observed nutrition. The primary target set contains 142 nutrition axes. Another 45 metabolome axes are reported separately and in the historical 187-axis aggregate. The 65 context-only axes can supply observed inputs but are not claimed to have reliable prediction performance.

On 28 September, the user revised the endpoint: **outperforming RF/XGBoost is no longer required; sound training, validated methodological and parameter choices, fair comparisons and complete documentation are required.** This amendment occurred after some R8 results were available; its timing is explicit in the [amended plan](PLAN.md). The original 5% primary-gain and 2% legacy-regression thresholds remain historical records, not prerequisites for repetition or delivery.

The retained completion network is a frozen name encoder followed by a two-layer MLP. It is neither a Transformer claimed to have won the comparison nor an established foundation model. Its three-seed completion primary error is **0.183435 ± 0.001222**, with exact output replay checks. Name-only error is 0.484042 ± 0.010458, still worse than the matched name-neighbor reference. The final tree comparison in Section 7 is pending; no final ranking is asserted.

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

## 3. Model and training methods

### 3.1 From the original Transformer to the retained MLP

The original V8 model used a source token. The V9 branch investigated a source-free Transformer with a source-by-axis residual used only during training. Its early hurdle head learned positive-value presence and positive amount. R0–R2 examined common inputs and supervision, calibration, amount weighting, hurdle versus direct regression, and checkpoint selection. Historical V8/RF scores under different protocols are not used to infer an architectural advantage.

Under the common research protocol, V9 receives CLS, the projected name, and 252 fixed axis slots combining axis embeddings with encoded values or masking markers. It has three Transformer layers, width 192, six attention heads, a 768-unit feed-forward block, dropout .15, GELU and pre-normalization; axis-calibrated output heads use rank-16 residuals. Name projection and scalar encoding each use two linear layers. Source-by-axis residuals are centered across training sources, and validation uses only the source-free prediction. R1 normalizes the source-calibration objective as `(L_base + w_source L_calibrated)/(1+w_source) + 1e-4 L_residual` to control the overall loss scale when removing calibration.

The representative R1 V9 setting uses batch size 64, AdamW learning rate .0001, 20 epochs and a fixed 20-epoch schedule. Amount weights 1/2/3 and calibration on/off were evaluated separately. In the R2 direct-regression branch, replacing SmoothL1 with MAE within the same 20-epoch budget reduced completion error from .225345 to .213960, while conditional positive error increased from .318757 to .354390 and zero error decreased from .152736 to .087285. This is an aggregate improvement with a positive/zero trade-off, not improvement on every prediction. These historical runs use the earlier 32-dimensional name cache. Input, optimization and budget also differ from the retained PCA128 MLP, so their comparison is not a pure architecture ablation.

The retained neural representative is an MLP. Existing controlled experiments support its loss, capacity and training-budget choices, and it performed better on the current internal completion benchmark than the tested Transformer branches. Its 512-dimensional hidden state is available for future representation studies. Frozen probes, few-label transfer and unseen-source generalization have not been established; “representation research model” is justified, whereas “validated general nutrition foundation model” is not.

### 3.2 Name, numeric and visibility inputs

Only `original_name` enters the text branch; source, food group, processing and scientific-name metadata are excluded. The frozen [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Attention-mask-aware mean pooling and L2 normalization yield a 384-dimensional vector. The local limit is 128 tokens; the recorded maximum name length is 54, so this limit did not truncate the audited names.

R7/R8 fit a common exact float64 PCA basis using only the 64,700 training profiles, without whitening. The first 32 and 128 components preserve 54.533% and 87.846% of variance. The final input uses 128 components. The 32-component control retains 128 slots, with the final 96 set to zero, so both MLP arms have identical architecture and parameter count. Retained embedding variance is not proof of nutritional utility.

The final input contains 632 features: 128 name features, 252 scaled nutrient values and 252 visibility indicators. The entire target family, including related forms, is hidden. Hidden labels and their true availability do not determine which query axes the model receives. The model has a fixed output grid and the caller specifies the returned axes.

### 3.3 Architecture, loss, optimization and checkpoint selection

The network is `Linear(632,512) → GELU → LayerNorm(512) → Linear(512,512) → GELU → Linear(512,252)`, with 717,052 trainable parameters, excluding the frozen text encoder. It has no extra name-only head, source residual, presence-probability multiplier or additional contrastive loss.

Each epoch visits 337,048 profile-family tasks across 19 families. Every observed training target among the 187 supervised axes is exposed exactly once. Let `w_ia` balance sources and source-specific profiles within a food group, and let `Z_a` be the complete training weight sum for axis a. The objective is:

`L=(1/187) Σ_a [Σ_i w_ia |f_a(x_i)-t_a(y_ia)| / Z_a]`.

Minibatches use full-dataset axis normalization and the task-count correction rather than redefining axis weights within each batch. The 45 metabolome axes retain coefficient 1, while checkpoint selection uses the 142-axis nutrition metric. MAE can improve zero errors while worsening positive-value predictions; that trade-off is retained in reporting. The direct head is not multiplied by a presence probability. Negative transformed predictions are clipped to zero before inversion; nonfinite outputs fail the experiment instead of being omitted.

Optimization uses AdamW, learning rate .001, weight decay .0001, batch size 256, gradient clipping at 1, and 60 epochs. CosineAnnealingLR uses T_max=60 and eta_min=.00001. The earliest strict minimum of the fixed full validation nutrition metric selects the checkpoint. There is no extra name-only assignment or additional context dropout. Complete curves and selected epochs are saved; neither 60 epochs nor the learning rate is claimed to be globally optimal.

### 3.4 Three inference tasks

`predict(food_name, observed_profile, target_axes)` supports caller-specified targets; an empty observed profile gives name-only prediction. `encode(...)` exports an explicitly identified modality representation. `retrieve_names(...)` returns rankings, not calibrated probabilities. Retrieval candidates use 142-axis nutrition predicted from candidate names by the same model; the query receives no name and candidates use no measured profiles. The fixed library contains 49,913 names. Full and 30%-retained panels contain 10,479 and 8,610 profiles with at least three observed nutrition axes. Their populations differ, so their score difference is not an isolated masking effect.

## 4. Iteration history and attribution

Each version retains all configurations, failures, per-axis/source analyses and three-task results. The table is not a causal curve made from the best score in each round. R0→R1 changes the training protocol, and R7/R8 change the name-input protocol.

|Version|Question, intervention and principal evidence|Interpretation and decision|
|---|---|---|
|[R0](../r0/README.md)|Common raw-space aggregation, training scales, name-only text and query/mask rules; 12-configuration baseline and quarantine|Comparable evaluation established; provenance unresolved and training-mask distributions still differed. No pure architecture attribution|
|[R1](../r1/README.md)|Exhaustive family tasks and global axis weights; fixed-schedule 8→20 epochs gave MLP .246757→.222459; V9 amount weights 1/2/3 and source ablation|Budget control supports longer training; larger amount weight helped that V9 branch; source removal had no clear completion gain. Joint protocol fixes are not single-factor evidence|
|[R2](../r2/README.md)|12 budget entries: SmoothL1→MAE .222459→.209484; width 256→512 .209484→.202428; 20→60 within one 60-epoch schedule .209506→.184491; also width 1024, normalization, direct V9, selection and name standardization|Retain MAE/512/60 and LayerNorm. MAE mainly helped zeros. Width1024 gain was uncertain and name-only worsened. Hurdle→direct changed multiple mechanisms|
|[R3](../r3/README.md)|Four configurations: 10%/20% name-only assignment, separate task heads and diagnostics|Some name gains cost completion performance; not adopted in the final completion model. Not every trade-off establishes gradient conflict|
|[R4](../r4/README.md)|Eight entries: five complete including reused control, three numerical failures; query residuals, view consistency and metabolome coefficient .5; common inverse/query contract fixes|No sufficient completion benefit; failures preserved, no NaN omission or loose tolerance to manufacture success. Engineering fixes separated from performance claims|
|[R5](../r5/README.md)|Twelve configurations covering name prediction, nutrition–name alignment and partial views; full-input specialist three-seed R@10 .298199±.003720 versus then-current KNN .192192|Limited full-input retrieval benefit; 30% R@10 only .002252. A separate specialist, not the completion model; superiority over later PCA128 KNN was not demonstrated|
|[R6](../r6/README.md)|Additional fixed 30% and mixed30/60/90 context deletion: completion .197232/.210462 versus parent .184491|Both worsen the fixed completion panel; not adopted. Name gains do not replace the completion result|
|[R7](../r7/README.md)|Six entries; common PCA32→128 improved name-MLP/KNN primary errors by 8.84%/6.87%|Retain richer name input; MLP still 5.11% worse than matched KNN. Old-input trees are not matched baselines|
|[R8](../r8/README.md)|Twelve configurations with common 632-feature inputs; MLP32→128 completion gain .931% with interval crossing zero, name-only gain 18.909%; fixed-depth XGB completion worsened 4.048% with added name dimensions|Full search is being finalized. Three128-input configurations per tree family plus32-input controls; final choice after complete evidence|
|[Fixed-seed repetition](NEURAL_REPLICATION.md)|No new hyperparameters; MLP seeds 22/23/24 complete with exact replay; primary 0.183435 ± 0.001222|Supports conditional reproducibility, not independent-test generalization. Tree repetitions and final comparisons pending|

Fixed-checkpoint interventions show that the network uses names: removing or permuting names harms completion. Error decompositions attribute much of the observed net gap to underprediction of positive values on common axes, partially offset by zero-value gains. This does not support explaining the entire gap as rare-axis undertraining. These decompositions are descriptive, not proof that one loss, architecture or scaling mechanism is the sole cause. Numeric-basis work has only train-only feasibility and component checks; R9 was not trained and has no performance result.

## 5. Evidence behind the retained parameters

|Choice|Validation performed|Retained setting and remaining uncertainty|
|---|---|---|
|Family tasks and weights|R1 exhaustive feature/row/target/weight alignment; one exposure per observed target|Retain to align the learning problem; objectives and optimizers still differ between methods|
|Regression loss|R2 matched SmoothL1→MAE and XGB objective comparison|MAE for the neural model, squared error for trees; retain positive/zero trade-offs, no universal-MAE claim|
|Hidden width|256→512 helped;512→1024 had uncertain completion gain and worse name-only|512; not an exhaustive capacity search|
|Training budget|20/60 windows within one 60-epoch schedule, followed by three-seed curves|60 epochs and primary-metric selection; longer budgets not independently validated|
|LayerNorm|Matched removal worsened completion by 3.69%, supported by paired interval|Retain; document the name-only trade-off|
|Name dimensions|R7 common-basis32/128; R8 matched slots and parameter count|128; completion-specific dimension gain is not established by its interval|
|Task mixture and heads|Independent R3 and R6 controls|0% extra name-only assignment, shared head, no extra dropout; not jointly optimal for all tasks|
|Auxiliary weighting/complex readout|R4 coefficient .5, consistency and query residuals|Coefficient1 and simple head; complex branches lacked supported benefit|
|Learning rate, batch, weight decay, clipping, GELU|Held fixed in controlled experiments; trajectories/gradient logs checked; three-seed stability|Not individually exhaustively tuned. Reasonable engineering settings do not imply optimal hyperparameters|
|Random seeds|Fixed 22/23/24; all independent runs retained|Report mean, sample SD and range, not the best seed|

## 6. RF, XGBoost and name-neighbor methods

Each tree method fits 187 separate regressors using every eligible training profile for its axis, without the historical 5,000-row cap. The largest axis has 58,958 training rows. Name/value/visibility features, complete target-family hiding, targets and transforms match the MLP. Per-axis source weights are divided by their mean before passing sample_weight; labels are not pooled across sources. Trees regress scaled-log values and use the same nonnegative inverse transform.

[Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) uses scikit-learn 1.5.2, 400 bootstrap trees, no depth limit and squared error. Leaf size and feature fraction are listed below; other defaults are recorded in per-axis get_params receipts. Fitting uses four threads. Prediction accumulates trees serially in fixed order to remove a few-ULP dependency on parallel reduction order.

[XGBoost](https://arxiv.org/abs/1603.02754) uses version 2.1.3, 800 trees, hist, learning_rate=.03, min_child_weight=5, subsample=.8, colsample_bytree=.8, reg_lambda=1 and squared error, with max_depth 6/10/14. Per-axis random_state is the run seed plus axis_index. Fitting uses four threads.

|Method|R8 configuration|Name dimensions|Training objective|
|---|---|---:|---|
|RF control|400 trees, leaf=1, max_features=0.5|32|Squared error|
|RF A|400 trees, leaf=1, max_features=0.5|128|Squared error|
|RF B|400 trees, leaf=3, max_features=0.5|128|Squared error|
|RF C|400 trees, leaf=1, max_features=1.0|128|Squared error|
|XGB control|800 trees, max_depth=10|32|reg:squarederror|
|XGB A|800 trees, max_depth=6|128|reg:squarederror|
|XGB B|800 trees, max_depth=10|128|reg:squarederror|
|XGB C|800 trees, max_depth=14|128|reg:squarederror|

Each method has three 128-input search configurations and one 32-input control. This is an explicit tuning budget, not exhaustive optimization. Each method's configuration is selected by the fixed 142-axis completion metric; auxiliary tasks do not select the tree. Other seeds then use that frozen configuration. Historical random-PCA32 trees cannot substitute for new-input repetitions.

The same fitted per-axis models produce completion, numeric-input-free name-only outputs and name-predicted candidate nutrition for retrieval. Each model undergoes reversed-order, subset and in-memory pickle reload checks before release to control storage. Predictions and receipts are retained; discarded forests are not advertised as an online service for arbitrary new names. The additional name-only reference uses all eligible training data, K=10, KDTree search, source weighting and inverse-distance weighting.

## 7. Completed evidence and pending final comparison

All three neural seeds below use the same completion recipe. Lower errors are better.

|Seed|Selected epoch|Completion142 MAE|Legacy log-MAE|Name-only142 MAE|
|---|---:|---:|---:|---:|
|20260922|60|0.184290|0.059214|0.493993|
|20260923|59|0.183980|0.059999|0.484992|
|20260924|57|0.182035|0.058565|0.473141|

Completion legacy log-MAE is 0.059259 ± 0.000718; raw-unit MAE is 0.368583 ± 0.004683. Positive and zero errors are 0.279095 ± 0.001755 and 0.111908 ± 0.001335. The45-axis primary error is 0.578353 ± 0.014242, and the 187-axis aggregate is 0.278469 ± 0.002567.

The same models achieve full-input retrieval R@10 0.003182 ± 0.000271 and 30%-input R@10 0.002232 ± 0.000454, expressed as 0–1 rates. Name-only and retrieval remain limitations. The stronger R5 specialist result is not assigned to the completion MLP.

![Three-seed training and validation curves](../../../reports/v9_final_neural_summary_v1/learning_curves.png)

**Final matched-protocol comparison: pending.** It must include all three seeds for each selected RF/XGBoost configuration, complete mean/SD/range summaries for both regression tasks, retrieval MRR/R@1/5/10, axis support and source breakdowns, paired food-group intervals and computational cost. Missing tree repetitions are not replaced by a single-seed score in a supposedly final table.

Paired intervals will first average errors from three independent models, then resample complete food groups 1,000 times. Predictions are not averaged into an ensemble. Seed SD and food-group intervals answer different questions; neither eliminates repeated-validation selection bias. These remain internal validation results, with the historical test closed.

## 8. Retained model, limitations and reproducibility

The fixed MLP512/PCA128 is a practical, reproducible configuration supported by recorded trade-offs. Its retention does not require proof of superiority over both trees. Final tree selection and the final version decision are incomplete, so this draft gives no final ranking.

Limitations include unresolved original units/definitions, aliases and copied records, absent unseen-source/category and few-label transfer studies, weak name-only performance, sparse retrieval, a limited hyperparameter search, only three seeds and repeated validation use. Internal performance does not establish broad biological or health-use validity.

Data-manifest SHA256: `48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923`. Frozen training commit: `6ff6f36d4f32c9c2dbf12eac6d32d0f3c21d6df4`. PCA128 cache: `4132b3623da37df2f02aedc6dde7d06b5c1e87a6b65e71d87f58ebbed5d2f768`. Task panel: `384b66c10142aefa801a5d0fbed2772b2e128150c9bb1575e2c2fe9bc513fbf8`. Environment: Windows, Python 3.10.19, PyTorch 2.7.1+cu128, NumPy 2.1.3 and RTX 5070 Ti 16 GB.

Data evidence is in `reports/v9_final_data_evidence_v1/summary.json`; neural replay and complete seed summaries are in `reports/v9_final_neural_audit_v1/` and `reports/v9_final_neural_summary_v1/`. Commands are recorded in the [neural repetition record](NEURAL_REPLICATION.md), [R8 execution record](../r8/README.md) and each version's plan. Final statistics commands, tree fingerprints, costs and bilingual table-consistency checks will be added at closure. Raw values, predictions, checkpoints and caches remain in local data/output directories, outside public experiment prose.

