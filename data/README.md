# Data Access

This repository intentionally does not redistribute raw food-composition
databases, derived numeric matrices, text-embedding caches, or model checkpoints.
Several upstream sources have distinct redistribution terms, and the active V8
corpus alone contains a 125 MB compressed token table.

## Distribution plan

- **Working dataset release:** Hugging Face Datasets, under the planned address
  [`mz1622/foodnutrition-composition`](https://huggingface.co/datasets/mz1622/foodnutrition-composition).
  It should remain private or gated until every upstream licence is reviewed.
- **Citable frozen release:** Zenodo, created only after the V9 benchmark,
  source licences, and release manifest have been frozen. Zenodo provides the
  archival DOI; Hugging Face provides convenient programmatic download.
- **GitHub:** code, data schema, source manifest, checksums, build scripts, and
  release instructions only.

The Hugging Face repository has not been published from this checkout because
there is no valid authenticated Hugging Face token. Do not treat the planned URL
as a currently available download.

## Package a local release

From a checkout that contains the active V8 data:

```bash
python scripts/package_foodnutrigpt_dataset_release.py \
  --output-dir release/foodnutrigpt_v8_source_native_baseline
```

The command creates a versioned package and SHA-256 manifest without modifying
the original data. After authenticating with Hugging Face, upload the package:

```bash
huggingface-cli login
huggingface-cli upload mz1622/foodnutrition-composition \
  release/foodnutrigpt_v8_source_native_baseline . \
  --repo-type dataset
```

The package is explicitly labelled as a **source-native V8 baseline**, not the
planned source-aware/source-free V9 benchmark. A future V9 release must have its
own dataset version, split manifest, provenance statement, and checksum file.
