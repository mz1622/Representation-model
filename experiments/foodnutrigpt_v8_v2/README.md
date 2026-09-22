# FoodNutriGPT V8/V2 Experiment Record

## Frozen reference

The following artifacts are the fixed references for this experiment series:

- Dataset and split: `global_foodnutrigpt_v8_single_stage_v2_complete_test`.
- Transformer reference: unified all-axis hurdle loss, eight-epoch cosine schedule.
- Tabular reference: the 50-tree per-axis Random Forest baseline.
- Complete test: 659 held-out profiles; 187 axes; leave-one-mask-family-out.

The complete test panel is not used for choosing new variants.  Each candidate
is trained with `--skip-test`, selected solely by its fixed source-unknown
validation metrics, and recorded in `experiment_ledger.csv`.  The held-out
panel is opened only for a prespecified final candidate.

The validation-selected `amount_loss_weight=3` checkpoint was the only new
candidate opened on the complete test. Its final profile-axis comparison is in
`output/global_foodnutrigpt_v8_single_stage_v2_final_amount3_comparison/`.

## Selection metric

Primary selection metric: validation macro-axis `log1p(g/100 g)` MAE.  The
validation hurdle loss, log-RMSE, raw MAE, nutrition metric, and food-metabolome
metric are retained as diagnostics.  Each row must change one causal factor
relative to its parent unless the ledger explicitly labels it a compound
follow-up.

## Interpretation rules

- A lower validation error with the same data, split, mask sampler, seed and
  architecture supports a causal interpretation for the changed training
  factor.
- A result from the frozen complete test is descriptive only; it cannot be used
  to select further settings.
- No missing composition value is represented as zero.  Explicit zeros remain
  observed targets.
