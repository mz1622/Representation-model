# First- versus second-forward `[FOOD]` UMAP

## Which representation the earlier UMAP used

The earlier V5+CCE UMAP used the **first-forward** `[FOOD]` state. The extraction
path was `model.forward_details(...)[1]`. It supplied all observed numerical
axis/value tokens and an empty target-query grid, so the representation did not
depend on a selected completion target.

This experiment keeps the same 4,000 validation profiles, input tokens, empty
target-query grid, FoodEx2 labels, eight macro categories, and UMAP
hyperparameters. The intervention is only the representation:

```text
first forward:  visible axis/value tokens + learned [FOOD]
                -> h_food_1

second forward: visible axis/value tokens + detach(h_food_1)
                -> h_food_2
```

The second representation is `model.forward_prompt_details(...)[3]`. Both
passes use the same frozen V5+CCE checkpoint. No training or checkpoint
selection is performed.

## Result

| Representation | Label | Space | 15-neighbour purity | Silhouette |
|---|---|---|---:|---:|
| First `[FOOD]` | 8 macro food classes | Original embedding | 0.552 | -0.146 |
| Second `[FOOD]` | 8 macro food classes | Original embedding | 0.540 | -0.132 |
| First `[FOOD]` | 8 macro food classes | UMAP 2D | 0.532 | -0.212 |
| Second `[FOOD]` | 8 macro food classes | UMAP 2D | 0.512 | -0.191 |
| First `[FOOD]` | Database source | Original embedding | 0.784 | 0.122 |
| Second `[FOOD]` | Database source | Original embedding | 0.781 | 0.084 |

The second pass does not improve category separation. Its macro-category
neighbour purity falls from 0.552 to 0.540 in the original embedding and from
0.532 to 0.512 after UMAP. Its silhouette is slightly less negative, but remains
below zero; this reflects somewhat different global geometry rather than clean
class separation. Source structure remains almost unchanged and stronger than
food-category structure.

The two states are related but not identical. Across all 4,000 profiles their
mean cosine similarity is 0.949 (median 0.965; 5th--95th percentile
0.837--0.991). The second pass therefore performs a moderate prompt-conditioned
update, not a completely new representation.

## Interpretation

The two-pass mechanism was trained to improve target-axis completion. The
second `[FOOD]` state is an internal result of that refinement, but the official
prediction is still decoded from the second-pass target-axis tokens. There is no
training objective requiring `h_food_2` to become a better food-category
embedding. These results are consistent with that design: the second pass can
help completion without improving unsupervised category clustering.

## Versioned artifacts

- `artifacts/umap_first_vs_second_food.png`: fixed-sample visual comparison.
- `artifacts/first_second_structure_metrics.csv`: category and source metrics in
  original embedding and UMAP spaces.
- `artifacts/embedding_shift.json`: paired first/second representation shift.
- `artifacts/manifest.json`: input hashes, exact extraction slots, checkpoint,
  and UMAP configuration.

The sampled embeddings and profile-level UMAP coordinates are generated locally
for audit but excluded from Git. No model parameters or raw composition data are
committed.
