# FoodNutrition Representation Model

FoodNutrition develops a source-free representation model for food composition.
The model receives standardized food text and optionally observed composition
values, then reconstructs masked nutrition and food-metabolome axes. The goal is
not to reproduce a particular database's reporting convention.

## Status

The current completed result is a frozen V8 source-native baseline:

- 76,915 source-native food profiles from 25 food-composition sources;
- 2,259,178 direct-mass observations in `g/100 g`;
- 252 active axes, including 187 masked-loss targets;
- 142 nutrition and 45 food-metabolome targets;
- 659 held-out profiles covering all 187 target axes.

V8 preserves source-native records and trains with a source token. It is useful
as a baseline, but it is not the final source-free benchmark. V9 will use source
only in a training-only source-axis calibration branch and will emit a
source-free prediction at inference.

The first V9 checkpoint was selected on validation and evaluated once on the
complete 187-axis test panel without a source input. Its source-free Random
Forest baseline is currently stronger than the V9 Transformer on both panels.
The full protocol and
metrics are recorded in
[experiments/foodnutrigpt_v9_source_calibrated_validation/](experiments/foodnutrigpt_v9_source_calibrated_validation/).

Read [AGENTS.md](AGENTS.md) for data rules, audit criteria, the V9 plan,
expected outcomes, and fine-tuning tasks.

## V8 Audit

Run the read-only audit against a local V8 checkout:

```bash
pip install -r requirements-colab.txt
python scripts/audit_v8_source_aware_source_free.py
```

The audit reports corpus semantics, split isolation, cross-source exact-name
candidates, and the implementation gap between V8 and V9. It does not alter data
or model outputs. A human-readable summary is in
[docs/V8_SOURCE_AWARE_SOURCE_FREE_AUDIT.md](docs/V8_SOURCE_AWARE_SOURCE_FREE_AUDIT.md).

## Frozen V8 Training

```bash
python scripts/train_global_foodnutrigpt_v8_single_stage.py \
  --loss-mode all_axis \
  --skip-test
```

Use `--skip-test` for validation-only experiments. The complete test is opened
only once for a checkpoint selected using validation. The frozen result ledger is
in `experiments/foodnutrigpt_v8_v2/`.

## Data Distribution

Raw and processed data are excluded from GitHub. The planned working-data host
is [Hugging Face Datasets](https://huggingface.co/datasets/mz1622/foodnutrition-composition),
with a later frozen Zenodo release for a DOI. The Hugging Face dataset has not
yet been published from this checkout because no valid publishing credential is
available. See [data/README.md](data/README.md) for package and upload steps.

## Layout

```text
scripts/       Builders, audits, training, evaluation, and release tools
src/           Dataset construction and composition-audit modules
tests/         Data and audit unit tests
experiments/   Frozen experiment protocol, hashes, and result ledger
docs/          Human-readable audit and modeling documentation
data/README.md Data distribution policy and package instructions
```

## Reproducibility

All comparisons must use identical data versions, food-group split rules, visible
context, masks, normalization, and source-free inference inputs. Missing values
are unknown, not zero. Explicit zero is an observed value.

Run the data and audit unit tests with:

```bash
PYTHONPATH=src python -m pytest tests -q
```
