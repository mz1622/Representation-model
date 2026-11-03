# FoodEx2-labelled V5+CCE UMAP (v1)

This audit relabels the frozen 4,000-profile validation UMAP of the
dependency-mask V5+CCE `[FOOD]` representation. It does **not** retrain the
model and does **not** refit or move the UMAP coordinates. The only change is a
more complete, auditable category layer.

## Label policy

The target taxonomy is the EFSA FoodEx2 Exposure Hierarchy level 1. A label is
accepted only from:

1. a source-native FoodEx2 level-1 label;
2. a source-published category code or category name that maps unambiguously to
   one FoodEx2 level-1 branch; or
3. for a profile whose native group is blank, a unanimous label among the
   already-labelled members of its exact-name candidate group.

Present but ambiguous groups such as `Dairy and Egg Products`, `Meat, egg and
fish`, `Beverages`, and `Animal foods` remain `Unlabelled`. Food names,
composition values, model embeddings, UMAP coordinates, and cluster positions
are never used to create a label. Exact-name groups are treated only as a
limited recovery mechanism, not as confirmed food identity: they add 294 labels
in the full corpus and 13 labels in the plotted sample.

References:

- [EFSA FoodEx2 overview](https://www.efsa.europa.eu/en/press/news/150504)
- [EFSA DietEx guide with FoodEx2 level-1 categories](https://www.efsa.europa.eu/sites/default/files/2024-12/dietex-userguide.pdf)

## Coverage

| View | Labelled | Total | Coverage |
|---|---:|---:|---:|
| Full source-native corpus | 47,752 | 75,875 | 62.94% |
| Frozen UMAP sample | 2,515 | 4,000 | 62.88% |

The previous plot labelled 772/4,000 points (19.3%). The new crosswalk therefore
raises plotted category coverage by 43.6 percentage points while retaining
ambiguous records as grey points.

## What the UMAP shows

The two category panels are in `artifacts/umap_foodex2_all_balanced.png`. Panel A
shows every profile; panel B caps the coloured overlay at 140 profiles per
category, while preserving all points as grey context. The separate source plot
is `artifacts/umap_database_source.png`, and every database code is expanded in
`artifacts/database_sources.csv`. Label evidence is visualised in
`artifacts/umap_label_provenance.png`.

The representation is substantially more organised by database source than by
FoodEx2 food category. On the same 2,515 labelled profiles, 15-neighbour purity
in the original V5+CCE embedding is 0.784 for database source and 0.470 for
FoodEx2 category. The corresponding cosine silhouette scores are 0.122 and
-0.312. In the two-dimensional UMAP, neighbour purity is 0.771 for source and
0.448 for category. The negative category silhouette means that the broad food
classes overlap rather than forming clean, globally separated islands.

This does not mean the representation contains no food semantics: the observed
category neighbour purity (0.470) is above the class-frequency random baseline
(0.100). It does mean the
visible geometry is dominated by database-specific structure, plausibly
including source-specific axis coverage and measurement conventions. Therefore
the UMAP should be presented as a representation audit, not as evidence that the
model learned a source-invariant food taxonomy.

## Reproduction

```powershell
python scripts/build_foodex2_umap_v1.py `
  --coordinates <representation_r2/umap_v5_cce.csv> `
  --profiles <foodnutrigpt_v9_r0_v1/profiles.csv.gz> `
  --embedding <representation_r2/embedding_V5_CCE.npy> `
  --output-dir <new-output-directory>
```

The profile-level point file is intentionally excluded from Git. Versioned
artifacts contain only aggregate audits, rendered figures, source names, and the
reproduction manifest; no raw composition data or model parameters are stored.
