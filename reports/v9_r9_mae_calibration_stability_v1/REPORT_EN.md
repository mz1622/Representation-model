# R9 MAE source-calibration parameter stability: read-only diagnosis

[中文](README.md) · [Machine-readable evidence](summary.json). This is a parameter diagnosis after MAE confirmation, not a new training version or calibration ablation.

## 1. Question and hypothesis

Completion error differs across the three MAE seeds. Inspect whether the learned source offsets are broadly unstable, providing descriptive evidence for a possible later calibration ablation. Parameter stability is neither sufficient evidence of calibration benefit nor a substitute for prediction stability.

## 2. Parent models and controls

Only independently audited checkpoints for seeds20260922/23/24 are read, selected at epochs60/60/57 respectively. These are selected models, not identical-epoch states. The current MSE run and its files are not read, modified or restarted. Data and tree references remain unchanged.

## 3. Methods and reproduction

On one CPU thread, load source-offset parameters and24 training-source indices with `weights_only=True`. Reconstruct the original per-axis arithmetic centering over these24 sources. There is no model forward pass, optimization, raw-composition-value loading or validation-label reading. Only names, families, supervision flags and training support counts are read from frozen axis metadata. Check187 supervised axes,65 context-only axes and all input identities.

Offsets are in scaled-log space, not g/100g. Descriptors weight24 source parameters equally and do not measure prediction effects weighted by actual supervision coverage. Report per-axis RMS, mean absolute and maximum absolute offsets. Correlation over the complete table can also be dominated by a few large offsets.

```powershell
.venv/Scripts/python.exe scripts/diagnose_foodnutrigpt_r9_calibration_stability.py --output-dir reports/v9_r9_mae_calibration_stability_v1
```

Existing directories reject overwrites. The script, model, audit and axis-metadata hashes are in summary.json. No full source-by-axis parameter matrix is saved or published.

## 4. Results

|Seed|Nutrition142 offset RMS|45-axis offset RMS|Nutrition mean absolute offset|Current raw-table L2 term|
|---|---|---|---|---|
|20260922|0.058359|0.030999|0.015395|2.132259e-7|
|20260923|0.061561|0.030471|0.016358|2.340507e-7|
|20260924|0.065714|0.024936|0.017350|2.600217e-7|

|Seed pair|Nutrition offset Pearson r|Difference RMS|Difference RMS / mean model RMS|
|---|---|---|---|
|22 / 23|0.973741|0.014104|0.235229|
|22 / 24|0.971479|0.016519|0.266271|
|23 / 24|0.969257|0.016309|0.256283|

All context-only offsets are zero. Maximum residual mean after CPU float32 centering is about1.44e-8, consistent with rounding and below the1e-7 check threshold. No parameters are nonfinite. The regularization term uses the raw24×252 table, including zero parameters for unsupervised axes. Its small value alone does not establish negligible gradient effects.

## 5. Mechanism diagnosis

The broad nutrition-offset directions are similar, but pairwise difference RMS is approximately24%–27% of the offsets' own RMS; the tables are not identical. Averaging per-axis RMS over the three seeds, the largest three axes are Pentadecenoic acid (15:1; isomer unspecified),0.289481; Carotene beta,0.258942; and Folic acid,0.222293. The amino-acid family's mean per-axis RMS is0.015911. Parameter magnitudes are not primary-error contributions.

## 6. Causal limits and alternatives

The observation concerns correlated parameter structure, not the performance effect of switching calibration on or off. Stable offsets could still help, harm or have little effect. Source supervision coverage, joint adaptation of backbone and residuals, different selected epochs and validation selection are not controlled here. Rankings of three RMS values and three prediction errors cannot establish causality. These parameters are not database-quality rankings.

## 7. Decision

Keep the running MSE protocol unchanged. The diagnosis does not support a simple explanation based on broadly random reversals of source-offset directions, and it does not replace a calibration-off experiment. The earlier conclusion that the MAE recipe failed to confirm RF superiority remains unchanged.

## 8. Next question and test status

Complete the registered MSE comparison first. If a calibration switch is later chosen, retain the normalization `(L_base+lambda*L_cal)/(1+lambda)` and register lambda as the isolated change, without simultaneously changing objective, capacity or optimizer. No such experiment is registered or started here. Complete test remains closed.
