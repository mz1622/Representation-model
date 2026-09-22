# V8 Audit Summary: Source-Aware Training and Source-Free Inference

Generated from the read-only audit on 2026-09-22. The full local audit artifact
is intentionally not committed because it contains derived data tables.

## Corpus Findings

| Item | Result |
| --- | ---: |
| Source-native profiles | 76,915 |
| Direct-mass tokens | 2,259,178 |
| Sources | 25 |
| Active axes | 252 |
| Masked-loss axes | 187 |
| Complete-test profiles | 659 |
| Complete-test axes | 187 |
| Active train/validation/test exact-name overlap groups | 0 |
| Excluded Foundation-overlap exact-name groups | 122 |
| Exact-name candidate food-axis cells | 1,804,315 |
| Candidates with two or more sources | 183,874 (10.19%) |

Exact-name candidates are an audit device, not a confirmed cross-source food
merge. Cross-source values can differ because names hide food facets, sample
variation, region, processing, analytical definition, or source expression.

Among candidate cells with two or more source values, the median log-scale span
was zero for both broad axis groups, but the 90th-percentile nutritional span was
`0.00599` and the largest nutritional span was `4.47961`. This long tail makes
automatic cross-source numerical pooling inappropriate.

## Code Findings

The frozen V8 trainer:

- keeps missing values absent and explicit zero observed;
- normalizes numeric values to `g/100 g` and train-only `log1p` scales;
- prevents active exact-name-group overlap across train, validation, and test;
- forces `SOURCE_UNKNOWN` for validation and test inputs;
- nevertheless concatenates a source embedding into the Transformer encoder for
  most training examples;
- weights duplicate measurements only within a source-native profile-axis cell,
  not across source observations of an exact-name candidate;
- evaluates source-native profile labels, not a source-free cell-level target.

Therefore V8 is a valid frozen source-native baseline but not a source-aware,
source-free benchmark.

## V9 Requirements

1. Keep raw source-native observations in the build layer.
2. Keep source out of the Transformer encoder and food embedding.
3. Use source only in a training-only, regularized, zero-centred source-axis
   residual head.
4. Emit only the source-free base prediction for validation, complete testing,
   and deployment.
5. Give each independent source equal total loss per repeated candidate food-axis
   cell, without automatic cross-source numerical pooling.
6. Use a source-free test input; aggregate raw-label errors offline to one equal
   food-axis score without exposing a source ID to the model.
7. Treat text-only prediction as a distinct zero-shot objective. Removing
   text-only augmentation does not remove food text embeddings.

Run `python scripts/audit_v8_source_aware_source_free.py` to reproduce the
audit against a local data checkout.
