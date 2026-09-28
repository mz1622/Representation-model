# R9 supplementary diagnosis: extreme errors from the parent seed

This record adds case interpretation; it is neither a new model candidate nor a new scoring protocol. The Chinese report is [README.md](README.md), and the machine evidence is [summary.json](summary.json). Three-seed confirmation remains a separate requirement.

## 1. Research question and hypothesis

The selected parent recipe improves the overall primary metric over fixed RF but retains higher positive-value error. We reviewed the previously saved extreme successes and failures for patterns by axis, source and positive/zero status. This is a post-hoc descriptive check; observations are not relabeled as preregistered hypotheses.

## 2. Parent and controls

The parent is `tf192_mae_lr3e4_60`, seed 20260922, compared with frozen `rf400leaf1half_name32`. Validation labels, scales and family masking remain unchanged. There was no new training, forward inference, relabeling, record exclusion or model selection.

## 3. Reproducible verification

All 40 saved case rows were inspected. Selection was reconstructed from frozen validation tasks and the two saved prediction tables: the 20 smallest and 20 largest profile-axis differences in scaled-log absolute error, Transformer minus RF. Case keys, labels, predictions, errors and food metadata matched; numerical tolerance was `rtol=atol=1e-12`. File identities are recorded in the machine evidence.

The [verification script](../../scripts/review_foodnutrigpt_r9_extreme_cases.py) uses no GPU, reads no test labels and refits no transformations. It does not recompute the original visible-context counts or near-name flags, and no new attribution is made from those columns. pandas emitted a DtypeWarning about mixed metadata columns; the run completed, all used-field matches and numerical assertions passed, and no numerical anomaly was suppressed.

```powershell
.venv\Scripts\python.exe scripts/review_foodnutrigpt_r9_extreme_cases.py --output-dir reports/v9_r9_parent_extreme_cases_review_v2
```

The example uses a new directory; existing evidence must not be overwritten. Individual food names, labels and predictions remain in the ignored local data directory. This report exports aggregates only.

## 4. Complete case-set overview

|Relative-error tail|Profile-axis rows|Food candidate groups|Group-axis pairs|Positive rows|Explicit-zero rows|
|---|---:|---:|---:|---:|---:|
|Transformer better|20|15|15|14|6|
|Transformer worse|20|18|18|19|1|

The better tail contains 14 FooDB, 2 BLS, 2 CNF, 1 FNDDS and 1 USDA SR Legacy rows. All 20 worse-tail rows are from FooDB. Counts include repeated profiles within food candidate groups; these are not 40 independent food samples.

|Axis|Better-tail rows|Worse-tail rows|Axis contribution to the full-panel primary difference|
|---|---:|---:|---:|
|Biotin|11|2|+0.000080788|
|Cholesterol|1|17|+0.000313938|
|Dodecanoic acid (12:0)|5|1|−0.000171843|
|Retinol|2|0|−0.000014229|
|Docosapentaenoic acid (DPA; 22:5n-3)|1|0|+0.000071941|

The final column comes from the existing full-panel decomposition, retaining source/group balancing and the fixed 142-axis denominator. It is not a mean over these 40 cases. Positive means higher Transformer error. Extreme-case counts do not determine an axis's overall result: Biotin has more better-tail cases yet a positive full-panel contribution.

## 5. Specific patterns

All 17 worse-tail Cholesterol cases are positive FooDB records. Every Transformer prediction is below the retained label by at least a factor of 500, while RF is closer to the recorded value. Repeated profiles do not increase the independence of this evidence.

The remaining worse cases include two positive Biotin underpredictions and one Dodecanoic acid overprediction on an explicit zero. Better cases include both positive values and explicit zeros, so the relative benefits cannot all be described as improved zero prediction. A better case means smaller error than RF, not necessarily an accurate prediction in absolute terms.

## 6. Attribution limits

**Observed facts** are the fixed models' differences relative to retained labels. **Unresolved explanations** include source-associated label-scale differences, inadequate fitting of large positive values, and effects of source calibration or observation patterns. There is no intervention that separates these explanations, and the original FooDB records and processing chain remain unavailable.

The pattern is related to the earlier unresolved unit concern, but it does not establish which records are wrong or where an error occurred. It also does not establish RF leakage or greater nutritional truthfulness of the Transformer. Freezing data does not certify label validity. Subsequent claims must be conditional on the frozen records and internal validation protocol; benchmark ranking is not itself proof of nutritional measurement accuracy.

## 7. Decision

Retain this supplementary diagnosis and continue the registered fixed-recipe repetitions. Case review changes no training parameter, data, reference or acceptance threshold. It strengthens interpretation and limitation records but does not replace complete scoring, seed variability or food-group intervals.

## 8. Next question and test status

Once all three seeds finish, present this parent-seed pattern alongside complete axis and positive/zero results without assuming cross-seed stability. Original-unit and copying-chain verification remains work for a future separate data version; the user's current data-freeze constraint is respected. The complete test remains closed. All 40 cases come from extreme error tails, not representative sampling or independent authentication of the original measurements.
