# V9 research protocol

Priority: nutrient completion, name-only prediction, then nutrition-to-unseen-name retrieval.
Budget: 48–72 local hours per iteration; at most 12 screening candidates.
Confirmation seeds: 20260922, 20260923, 20260924. Screening is not confirmation.

R0 establishes evidence, a versioned data view, identical targets/context/weights and strong baselines.
R1 investigates optimization duration, amount loss and source calibration one factor at a time.
R2 investigates output/loss alignment and conditional-zero versus positive errors.
R3 investigates masking distributions and name-only task competition.
R4 investigates context representations and axis-query heads with transfer probes.
R5 investigates cross-modal name/profile alignment and unseen-name retrieval.
R6 investigates held-source/category and low-label generalization.

Preserve frozen V8 files and results. Keep the historical test closed during development.
Never infer a corrected label from a suspicious multiplier. Record unresolved evidence and
use a versioned, explicitly exploratory quarantine sensitivity view.

Primary metric: macro nutrition-axis MAE of log1p(value / training positive typical scale),
with equal source influence inside candidate food-axis cells. Also report legacy log MAE,
raw MAE, conditional positive/zero errors, per-axis support and metabolome results.

Success milestone: >=5% improvement against the stronger tuned RF/XGBoost baseline,
paired uncertainty supporting improvement, <=2% deterioration in legacy nutrition log MAE.
No success claim from a one-seed screen, a small subset, or unresolved label provenance.

Each iteration must contain README.md with hypothesis, parent/change, controls, commands,
hashes/seeds/resources, every attempted result, diagnostics, causal versus speculative
interpretations, decision, next experiment and test-open status. Numerical artifacts remain
under ignored data/output/reports directories; only protocol/code/summary may enter Git.
