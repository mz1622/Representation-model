# Taiwan deduplicated zero/few-shot analysis v1

> **Superseded:** the official protocol now uses a fixed 500-food test set and
> the remaining 196 foods for adaptation.  See
> [`taiwan_deduplicated_analysis_v2_500_196`](../taiwan_deduplicated_analysis_v2_500_196/README.md).

## Scope

This analysis replaces the former exact-English-name filter with a conservative
external-overlap audit. A Taiwan food is excluded when any English alias either
matches a project food name exactly after normalization or has an informative
base-food/species token signature contained in a project name after removing
preparation/form words. Foods without an informative English alias are also
excluded because non-overlap cannot be established automatically.

The reference scope is all 75,875 project train+validation source-native
profiles, not only the training partition. The rule is intentionally biased
toward over-exclusion: a lexical match is an exclusion candidate, not a claim
that two analytical samples are identical.

- Original strict Taiwan panel: 2,212 foods.
- Old exact training-name overlaps: 423.
- Expanded overlaps with training: 940.
- Expanded overlaps with any project profile: 1,055.
- Foods without an informative English alias: 463.
- Final clean panel: 696 foods, 575 lexical identity groups, 41,927 jobs, 80 axes.
- Few-shot test: 400 foods and 24,115 jobs; adaptation pool: 296 foods.
- Lexical identity groups are kept intact across adaptation and test.

The row-level overlap ledger, Taiwan source table, profile split, predictions,
raw data, and model weights are not committed. This directory contains code
manifests and aggregate/model-axis results only.

## Zero-shot results

All models are frozen source-trained models. Lower is better.

| Panel | Method | Macro scaled-log MAE | Macro scaled-log RMSE | Macro raw MAE |
|---|---|---:|---:|---:|
| All 696 clean foods | V5+CCE | **0.190885** | **0.366633** | 0.305051 |
| All 696 clean foods | XGBoost | 0.232564 | 0.368227 | **0.285357** |
| All 696 clean foods | RF | 0.235997 | 0.366773 | 0.305826 |
| All 696 clean foods | KNN | 0.280266 | 0.441004 | 0.456624 |
| Fixed 400-food test | V5+CCE | **0.186881** | **0.354774** | 0.297497 |
| Fixed 400-food test | XGBoost | 0.231062 | 0.364864 | **0.273991** |
| Fixed 400-food test | RF | 0.235556 | 0.364133 | 0.294877 |
| Fixed 400-food test | KNN | 0.277146 | 0.432795 | 0.439696 |

On the fixed test, V5+CCE lowers scaled-log MAE by 19.1% versus XGBoost,
20.7% versus RF, and 32.6% versus KNN. In all three paired 2,000-draw profile
bootstraps, the full 95% interval for V5+CCE minus the comparator is below zero.

## Few-shot results

Values are macro scaled-log MAE, mean +/- sample SD across three adaptation
seeds. V5 fine-tuning updates all model parameters for a fixed 20 epochs without
CCE. Source-continued RF/XGBoost retain their source models and append 20
Taiwan-only trees/boosting rounds. Taiwan-only trees are fitted from scratch on
the selected Taiwan profiles.

| Method | 8 shot | 32 shot | 128 shot |
|---|---:|---:|---:|
| V5+CCE frozen | 0.186881 | 0.186881 | 0.186881 |
| V5+CCE fine-tuned | **0.185942 +/- 0.002385** | **0.180485 +/- 0.000531** | **0.167395 +/- 0.004089** |
| XGBoost source frozen | 0.231062 | 0.231062 | 0.231062 |
| XGBoost source continued | 0.231148 +/- 0.001363 | 0.221703 +/- 0.006365 | 0.207775 +/- 0.002721 |
| RF source frozen | 0.235556 | 0.235556 | 0.235556 |
| RF source continued | 0.242114 +/- 0.002539 | 0.235564 +/- 0.001293 | 0.231451 +/- 0.000765 |
| XGBoost Taiwan-only | 0.484292 +/- 0.027119 | 0.305578 +/- 0.019331 | 0.224743 +/- 0.010872 |
| RF Taiwan-only | 0.424441 +/- 0.029469 | 0.296743 +/- 0.014674 | 0.227400 +/- 0.011352 |

Fine-tuning V5 changes the frozen score by -0.5%, -3.4%, and -10.4% at
8/32/128 shots. The best tree result is source-continued XGBoost at 128 shots
(0.207775), still above both frozen V5+CCE (0.186881) and 128-shot fine-tuned
V5+CCE (0.167395).

## UMAP label plan

The current category UMAP labels only 13,901/75,875 profiles (18.3%); in the
fixed 4,000-point UMAP sample, 772 points (19.3%) are labelled. The restriction
comes from the plotting code, which accepts only a small exact-name crosswalk
for USDA SR Legacy, CNF, and FooDB. The corpus itself contains a nonblank native
food-group field for 63,132 profiles (83.2%).

The proposed replacement is:

1. Use the official EFSA FoodEx2 Exposure Hierarchy level 1 as the single target
   taxonomy. FoodEx2 is hierarchical and separates food identity from facets
   such as processing and physical state.
2. Build a versioned source-specific crosswalk from every database's official
   category code/name to FoodEx2 level 1. Recover official category tables for
   FNDDS, BLS, and other sources whose category was dropped by the current
   adapter rather than infer labels from composition.
3. Propagate a label through an exact-name group only when all authoritative
   source labels in that group agree. Exact-name transfer can currently cover
   only 444 of the 12,743 profiles with a blank stored group, so it is a
   secondary mechanism.
4. Keep unresolved profiles as `Unlabelled`; do not assign categories from UMAP
   coordinates, `[FOOD]` embeddings, nutrient values, or nearest clusters.
5. Plot all points in light grey, overlay high-confidence labelled points with a
   fixed FoodEx2 palette, and add a separate label-provenance panel. Use a
   category-balanced view in addition to the all-profile view so large sources
   and classes do not dominate the display.

Authoritative starting points:

- EFSA FoodEx2 overview: https://www.efsa.europa.eu/en/press/news/150504
- EFSA DietEx guide showing FoodEx2 level-1 categories:
  https://www.efsa.europa.eu/sites/default/files/2024-12/dietex-userguide.pdf

## Reproducibility

All neural and tree comparisons use the same 400-food test, the same 296-food
adaptation pool, the same nested 8/32/128 profile subsets, three seeds, the same
80 axes, the frozen source normalizers, and the dependency-mask v1 policy.
Source RF/XGBoost refits reproduce every stored zero-shot prediction with
maximum absolute difference zero before Taiwan continuation is applied.
