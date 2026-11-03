# V5+CCE UMAP with eight macro food categories

This view keeps the audited FoodEx2 level-1 labels from
`foodex2_umap_v1` but merges related branches into eight broad classes before
looking at the UMAP. The frozen V5+CCE embedding, the 4,000 validation profiles,
and the two-dimensional coordinates are unchanged. The merger does not use
nutrient values, model embeddings, database source, UMAP coordinates, or cluster
position.

## Fixed merger

| Macro category | FoodEx2 level-1 members |
|---|---|
| Cereal & starchy staples | Grains; starchy roots/tubers |
| Plant produce, legumes & nuts | Vegetables; legumes/nuts/oilseeds/spices |
| Fruit & juices | Fruit; fruit/vegetable juices |
| Meat, fish & eggs | Meat; fish/seafood; eggs |
| Dairy & milk | Milk/dairy; human milk |
| Sugars & fats | Sugar/confectionery; animal/vegetable fats and oils |
| Beverages | Coffee/cocoa/tea; water-based beverages; alcoholic beverages |
| Prepared & special-purpose foods | Composite dishes; seasonings/sauces; young-population foods; special-diet products; additives/processing aids |

These are display-level macro classes, not a proposed replacement for the
FoodEx2 hierarchy. The detailed FoodEx2 label remains available in the v1 audit.

## Result

Of the 4,000 UMAP profiles, 2,515 have an auditable FoodEx2 label and therefore
one of the eight macro labels. The other 1,485 profiles remain grey.

On the same 2,515 labelled profiles:

| Label being tested | Space | Classes | 15-neighbour purity | Random-frequency baseline | Silhouette |
|---|---|---:|---:|---:|---:|
| Eight macro food classes | Original V5+CCE embedding | 8 | 0.552 | 0.172 | -0.146 |
| Eight macro food classes | UMAP 2D | 8 | 0.532 | 0.172 | -0.212 |
| Detailed FoodEx2 classes | Original V5+CCE embedding | 19 | 0.470 | 0.100 | -0.312 |
| Database source | Original V5+CCE embedding | 18 | 0.784 | 0.188 | 0.122 |

Merging related categories makes the food signal clearer: macro-category
neighbour purity rises from 0.470 to 0.552, and it is well above its 0.172
class-frequency baseline. However, the macro silhouette remains negative, so
the classes overlap globally rather than forming eight clean islands. Database
source remains substantially more separable than food category.

Therefore the answer is nuanced: the representation distinguishes broad food
types locally, especially visible in several meat/fish/egg, staple, and plant
segments, but it has not learned a fully source-invariant taxonomy. This UMAP is
evidence of partial food-category structure, not complete category separation.

## Files

- `artifacts/umap_macro_categories_all_balanced.png`: primary all-profile and
  balanced views.
- `artifacts/macro_category_mapping.csv`: exact FoodEx2-to-macro mapping.
- `artifacts/macro_category_distribution.csv`: plotted category counts.
- `artifacts/macro_structure_metrics.csv`: detailed and macro metrics compared
  with database-source structure.
- `artifacts/manifest.json`: immutable input hashes and analysis policy.
