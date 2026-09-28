# V9-R9 first group: learning rate, training budget and replication decision

Status: both screening runs and all stage diagnostics are complete. Select lr=3e-4 for 60 epochs for fixed-recipe replication. Three-seed confirmation and the overall research objective remain incomplete. This is a stage report, not the final integrated report.

## 1. Research question and prior hypothesis

Can adequate optimization improve completion with a single-stage, direct-regression Transformer under the frozen common protocol? The registered contrast changes only initial learning rate, 1e-4 versus 3e-4. Each trajectory also supplies a best-through-20 and best-through-60 window. If optimization is limiting, additional training should reduce primary and positive-value errors; benefits to every task are not assumed.

## 2. Parent and changes

Reuse the R0 quarantined research view, R8 PCA32 name inputs and frozen RF/XGBoost results. No label, split, scale, cache, family mask or validation-panel changes. The model returns to the PDF-inspired Transformer approach while retaining single-stage training, source-free inference and the common metric. PDF data, normalization, positive-only labels and pooled scoring differ, so its numbers are not comparable here.

Both candidates share seed20260922, initial state, numerical code, and all 60 epoch task orders/exposures. Only learning rate and candidate name differ. Raw food names use frozen MiniLM/PCA32 features in 128 slots, with the last96 zero; visible nutrient values and observation masks provide context. Source is excluded from the encoder. There are no additional name-only training examples.

## 3. Reproducibility

Transformer: width192, 3 layers, 6 heads, FF768, dropout0.15, axis residual rank16; 1,548,594 parameters, 1,515,929 trainable. Source-balanced MAE over187 supervised axes directly predicts z=log(1+y/s). Training combines base and source-calibrated losses with normalized weights, source_weight1 and residual L2=1e-4. Inference uses only the base prediction. AdamW, batch64, weight decay1e-4, gradient clip1, 60-epoch cosine schedule, minimum LR=1% of initial. Select the earliest strict minimum of the full validation142-axis primary metric.

Each epoch visits337,048 family tasks and1,828,536 observed targets once. Explicit zeros remain targets and the entire target family is hidden. Training:64,700 profiles/42,282 candidate-name groups; validation:11,175 profiles/7,409 groups. Each prediction task has323,809 profile-axis requests; primary paired resampling includes7,344 groups. Retrieval uses49,913 candidate names, name-text-only candidate vectors and name-free nutrition queries.

Run commits are b4fd4c1 and ca403e6; all numerical training source hashes are identical. Windows, Python3.10.19, Torch2.7.1+cu128, RTX5070Ti, 4 CPU threads, fp32. Complete run times are10,551.985 and12,254.359 seconds, including final evaluation. Wall-time differences cannot be attributed to learning rate. Full training-fit diagnostics take another110.75 seconds. Manifests and evidence.json retain commands, environment, checkpoints and data/code/cache/panel hashes.

Training command: `.venv/Scripts/python.exe scripts/train_foodnutrigpt_v9_r9_transformer.py --candidate <name> --output-dir output/v9_r9/<name>`. Parent registration notes document the budget, audit and fit commands. Missing labels, hidden-label invariance, caller-specified queries and reload checks have passed. Both323,809-row prediction tables and all19,089 retrieval ranks replay exactly through the independent audit for each run.

## 4. Complete results

Every row uses one checkpoint for all three tasks. Lower error and higher R@10 are better. Conditional positive/zero errors cannot simply be added to recover the main score. There is one training seed, so no seed SD is available yet.

| Recipe / window | Selected epoch | Completion142 | Legacy log | Positive | Zero | Metabolome45 | All187 | Name-only142 | R@10 full / 30% |
|---|---|---|---|---|---|---|---|---|---|
| 1e-4 / 20 | 18 | 0.219576 | 0.066269 | 0.358686 | 0.093135 | 0.564114 | 0.302486 | 0.407723 | 0.8061% / 0.3807% |
| 1e-4 / 60 | 60 | 0.194543 | 0.058632 | 0.324743 | 0.086138 | 0.577952 | 0.286807 | 0.414298 | 1.2212% / 0.5618% |
| 3e-4 / 20 | 18 | 0.213405 | 0.063450 | 0.348083 | 0.110594 | 0.587303 | 0.303380 | 0.418662 | 0.5665% / 0.4491% |
| 3e-4 / 60 | 60 | 0.183553 | 0.054362 | 0.302773 | 0.092039 | 0.598022 | 0.283292 | 0.426411 | 1.0603% / 0.6781% |

Frozen RF32: completion142=0.189031, legacy log=0.056270, name-only=0.659518, full/30% R@10=0.4457%/0.1987%. Frozen XGB32:0.174487,0.052593,0.656148,0.6275%/0.1819%. Fixed name KNN32: name-only0.262260 and R@10=19.3572%/9.6722%; it is not substituted for a family-completion baseline. See the [frozen baseline appendix](../../../../reports/v9_r9_frozen_baseline_appendix_v2/BASELINES_EN.md) for all six completed tree configurations and costs.

At3e-4, primary improvement over RF is2.898%, paired food-group95% interval[0.681%,5.061%]; legacy improvement3.390% [0.237%,6.528%]. Primary error is5.196% worse than XGB, with improvement interval[-8.081%,-2.340%]. At1e-4, primary error is2.916% worse than RF with an interval crossing zero; legacy regression4.198% fails the2% guardrail.

Direct learning-rate contrast:3e-4 improves primary by5.649% [0.407%,10.135%] and legacy by7.282% [2.255%,12.134%]. Name-only worsens2.924% at the point estimate, improvement interval[-6.356%,0.366%]; neither harm nor equivalence is established. Full-profile R@10 decreases0.1609 percentage points, difference interval[-0.5006,0.1476] points. Partial-profile R@10 increases0.1163 points, interval[-0.1465,0.3616] points.

Best-through20 to best-through60: primary improvement11.401% [10.060%,12.683%] at1e-4 and13.988% [12.260%,15.684%] at3e-4. These nested windows share a60-epoch schedule and the longer window has more validation-selection opportunities. They do not isolate duration from selection or compare independent20/60 schedules.

Machine outputs retain per-axis, source, context and positive/zero results. Intervals use1,000 paired whole-food-group resamples conditional on the selected models. Nutrient cells or seed-food repetitions are not independent foods. Training-seed, repeated-selection, label and external-generalization uncertainties are excluded.

Failure record: the original first audit failed when Windows CP936 decoded UTF-8 candidate names. Its records remain intact. A versioned audit changes only explicit encoding; the first model was not retrained. The recovery queue and all stage analyses completed successfully. No training NaN/Inf occurred.

## 5. Mechanism diagnostics

The [four-panel learning curve](../../../../reports/v9_r9_first_group_v1/learning_curves.png) was visually inspected. Both models select epoch60; later improvement slows without sustained primary deterioration or numerical divergence. Last10-epoch mean clipping fractions are0.350883 at3e-4 and0.482988 at1e-4. This observation does not establish clipping as the causal mechanism.

Source-free evaluation of all1,828,536 training targets gives train/validation primary0.143092/0.194543 at1e-4 and0.134937/0.183553 at3e-4. Both improve at the larger LR. Foods and support differ between partitions, so this gap alone does not prove overfitting. The online187-axis calibrated loss is not directly comparable with validation142-axis MAE.

The3e-4 minus1e-4 primary difference is-0.010990: positive contribution-0.014577 plus zero contribution+0.003587. Against RF, positive extra error remains+0.017036, comprising underprediction+0.012476 and overprediction+0.004560; the zero benefit is-0.022514, yielding net-0.005478. Overall improvement does not mean better positive predictions or improvement on every nutrient.

At3e-4,76/142 axes have lower point error than RF. The7 axes with<100 training groups contribute-0.002453;20 axes with100–999 contribute-0.005975;115 with>=1000 contribute+0.002951. The sparse7 previously contributed+0.008669 at1e-4. Much of the learning-rate gain is therefore concentrated in low-support axes, making seed replication particularly valuable. Retain their per-axis intervals; do not change labels, scoring or sampling in response. Fatty acids contribute-0.006294 overall, while amino acids and minerals retain extra error.

Forty extreme success/failure numerical cases stay in ignored local data, not public documents, and are not a representative sample. Source/context strata have different coverage and do not constitute causal interventions.

## 6. Causal interpretation

Observed:3e-4 lowers training and validation completion errors; longer selection windows improve completion; name-only and retrieval do not improve uniformly. Supported interpretation is limited to this controlled single-seed setting: learning rate affects optimization outcomes under the fixed architecture, data, schedule and selection policy. The earlier RF gap cannot be attributed entirely to insufficient Transformer capacity.

Unresolved alternatives include sparse-axis and initialization sensitivity, validation-selection bias, source-calibration/MAE interactions, and capacity/loss effects on positive values. MSE, PDF256 capacity and source-weight ablations have not been run on this architecture. None is established as optimal or unnecessary. Neither the train/validation gap nor the single-seed interval proves seed-stable superiority.

## 7. Version decision

Select3e-4 for60 epochs as the recipe awaiting confirmation, retaining1e-4 and both20-window results. The direct controlled contrast supports completion improvement, the RF interval supports a gain, the legacy guard passes, and fit/curves show no obvious failure. Two fixed additional seeds address the remaining stability question before simultaneously changing architecture or loss.

This is not a global optimum or final acceptance. Selection targets completion; other checkpoints' best name-only/retrieval scores are not spliced into its capabilities. Foundation-model transfer remains unproven. decision.json and confirmation_plan.json bind actual evidence. Reuse the audited seed22 parent and train seeds23/24 from scratch with only the seed changed.

## 8. Next question and test status

Next is fixed-recipe replication. Score models independently, average their errors, report seed SD and paired food-group intervals. Averaging raw predictions into an ensemble is prohibited for the single-model claim. Acceptance requires mean primary improvement>0 over frozen RF, a supporting interval and legacy regression<=2%. XGB remains a comparator, not a gate. No tree refit, data change or historical-test opening.

If confirmation fails, use seed trajectories and failing axes to register the next single-factor method. If it passes, complete the integrated bilingual reports with explicit positive-value, name-only, retrieval and transfer limitations. MSE/capacity remain possible later experiments; neither branch has started. The two replication seeds have not run at this report boundary.
