You are working in a repository that already contains an older Colab notebook named:

structured_transformer_pretraining_colab.ipynb

Your task is to create a new Colab-ready notebook that implements a FoodDB nutrient-focused transformer pretraining model.

Create a new notebook named:

fooddb_nutrient_transformer_pretraining_colab.ipynb

Before writing the new notebook, inspect structured_transformer_pretraining_colab.ipynb and reuse its existing logic for:
1. Google Drive / Colab connection.
2. Project root detection.
3. FoodDB data loading paths.
4. Existing data file names and directory structure.
5. Saving direction / output directory conventions.
6. Any existing helper functions that are still useful.

Do not blindly copy the old model. Reuse only connection, loading, path, and saving infrastructure. The model design must be replaced with the new nutrient-focused transformer described below.

Goal:
Build a transformer-like pretraining representation model for FoodDB foods using FoodDB nutrient values as the target. Do not encode too many heterogeneous features into one token. For now, do not use bins. Use the original continuous log-transformed nutrient values.

Core modeling principle:
Text is only a semantic prior. Nutrition values are the target and the main supervision.

Inputs:
- Food name.
- Optional food description, if available.
- Observed FoodDB nutrient values.

Targets:
- Continuous log-transformed FoodDB nutrient values.
- No binning.
- No classification of nutrient bins.
- No compound/pathway/health-effect/taxonomy tokens.

High-level architecture:
FoodNutriFormer-v1

Each food item is represented as an unordered set of nutrient tokens plus optional text and CLS tokens:

[CLS]
[TEXT]
[NUTRIENT_ID + LOG_VALUE]
[NUTRIENT_ID + LOG_VALUE]
...

The model randomly masks observed nutrient values and reconstructs the masked continuous log values.

Important:
- Missing nutrient value is not zero.
- True zero, if present in the data, is a valid observed value.
- Missing values must be represented by an observation mask and must not contribute to loss.
- Do not impute missing values for training unless explicitly needed for a baseline. Main transformer loss only uses observed values.

Notebook requirements:
The notebook must run on Google Colab with GPU if available.
Use PyTorch.
Use pandas/numpy/sklearn/scipy/matplotlib as needed.
Keep the model small because FoodDB is small.

Suggested notebook structure:

1. Title / project description markdown
Explain:
- We are replacing the previous overpacked-token structured transformer.
- The new target is FoodDB nutrient reconstruction.
- We use continuous log-transformed nutrient values, not bins.
- Food names/descriptions are used only as optional text context.
- Nutrient identity and nutrient value are separated into different embedding components and then added.

2. Environment setup
Reuse the old notebook’s Colab/Drive connection code.
Install only packages that are actually missing.
Use deterministic seeds.

3. Configuration cell
Create a Config dataclass or dictionary with:
- old_notebook_path = "structured_transformer_pretraining_colab.ipynb"
- output_notebook_name = "fooddb_nutrient_transformer_pretraining_colab.ipynb"
- d_model = 128
- n_layers = 3
- n_heads = 4
- ffn_dim = 512
- dropout = 0.15
- batch_size = 64
- max_epochs = 500
- early_stopping_patience = 40
- lr = 3e-4
- weight_decay = 1e-4
- mask_ratio = 0.30
- high_mask_ratio = 0.80
- high_mask_batch_prob = 0.20
- text_only_batch_prob = 0.10
- text_dropout_prob = 0.30
- consistency_loss_weight = 0.05
- use_description = True
- use_text_token = True
- min_nutrients_per_food = 3
- min_foods_per_nutrient = 10
- validation_fraction = 0.15
- test_fraction = 0.15
- random_seed = 42

4. Load FoodDB data
Inspect and reuse the old notebook’s data loading logic.

The notebook should be robust to likely FoodDB table names:
- foods
- nutrients
- contents

If the old notebook already loads a cleaned dataframe or matrix, detect it and use it if appropriate.

Expected raw tables:
foods:
- id or food_id
- name
- description, optional

nutrients:
- id or nutrient_id
- public_id, optional
- name
- description, optional

contents:
- food_id
- source_id
- source_type
- orig_content
- orig_min
- orig_max
- orig_unit

Filter contents to:
source_type == "Nutrient"

Build a food × nutrient matrix.

5. FoodDB nutrient value processing
Implement a robust preprocessing function.

For each food-nutrient pair:
- Prefer orig_content if available.
- Else if orig_min and orig_max exist, use midpoint.
- Else if only one of orig_min/orig_max exists, use that endpoint with lower quality.
- Else mark missing.

Handle duplicates by robust median in log-value space.

Use the existing log-transformed value if the old notebook already computed one. Otherwise compute:
log_value = log1p(value)

Do not bin values.

Also build:
Y_log: float32 matrix [num_foods, num_nutrients]
M_obs: bool matrix [num_foods, num_nutrients]
W_quality: float32 matrix [num_foods, num_nutrients]

Quality weights:
- 1.0 for direct orig_content
- 0.7 for midpoint from min/max
- 0.4 for one-sided endpoint
- 0.2 or drop for ambiguous values

If range width is available, downweight wide ranges.

After matrix construction:
- Drop foods with fewer than min_nutrients_per_food observed nutrients.
- Drop nutrients with fewer than min_foods_per_nutrient observed foods.
- Recompute matrix after filtering.

6. Nutrient scaling
Do robust per-nutrient scaling on observed log values only.

For each nutrient j:
median_j = median(observed Y_log[:, j])
iqr_j = q75 - q25
if iqr_j is too small, fallback to std
if still too small, set scale to 1.0

Y_norm = (Y_log - median_j) / scale_j

Keep:
- nutrient_medians
- nutrient_scales
- nutrient metadata
- food metadata

Save these later.

7. Text features
Use food name and optionally description.

Text string:
if use_description and description exists:
    text = name + ". " + description
else:
    text = name

Use a simple, Colab-safe text embedding approach.

Preferred:
- If sentence-transformers can be installed/imported, use a small frozen model such as "sentence-transformers/all-MiniLM-L6-v2".
- If unavailable, fallback to sklearn TfidfVectorizer + TruncatedSVD to produce a dense text vector.

Do not fine-tune a language model.
The transformer receives only a projected frozen text vector.

Implement:
text_embeddings: float32 matrix [num_foods, text_dim]

8. Train/validation/test split
Use food-level split, not row-level split.

Use sklearn train_test_split with fixed seed:
- train
- val
- test

No leakage across foods.

9. Dataset and collate function
Create a PyTorch Dataset where each item returns:
- food_idx
- observed nutrient ids
- normalized observed nutrient values
- observed quality weights
- text embedding
- full observed mask for evaluation if needed

Because the number of nutrients is small, the collate function can pad to max sequence length in the batch.

For each food in a batch:
- Randomly select observed nutrients to mask.
- Normal batches: mask about 30% observed nutrients.
- High-mask batches: with probability high_mask_batch_prob, mask about 80%.
- Text-only batches: with probability text_only_batch_prob, mask all observed nutrient values.
- Ensure at least one masked nutrient.
- Ensure at least one unmasked nutrient when not text-only and possible.

Return:
- nutrient_ids: [batch, max_nutrients]
- values_in: [batch, max_nutrients], where masked values can be 0 placeholder
- is_masked: [batch, max_nutrients]
- is_observed_token: [batch, max_nutrients]
- target_values: [batch, max_nutrients]
- target_weights: [batch, max_nutrients]
- attention_padding_mask
- text_embedding
- food_idx

Only masked observed nutrients contribute to reconstruction loss.

10. Model: FoodNutriFormer
Implement a PyTorch model.

Components:
- cls_token: learned parameter [1, 1, d_model]
- nutrient_id_embedding: nn.Embedding(num_nutrients, d_model)
- value_mlp: MLP from scalar value to d_model
- mask_value_embedding: learned parameter [d_model]
- text_projection: MLP text_dim -> d_model
- type embeddings for CLS, TEXT, OBSERVED_NUTRIENT, MASKED_NUTRIENT
- transformer encoder using nn.TransformerEncoderLayer
- no positional embeddings
- regression head for continuous value prediction

Token construction:
For observed nutrient token:
token = LayerNorm(nutrient_id_embedding[nid]) + LayerNorm(value_mlp(value)) + type_embedding_observed

For masked nutrient token:
token = LayerNorm(nutrient_id_embedding[nid]) + LayerNorm(mask_value_embedding) + type_embedding_masked

Text token:
text_token = LayerNorm(text_projection(text_embedding)) + type_embedding_text

CLS token:
cls_token + type_embedding_cls

Transformer:
- batch_first=True
- norm_first=True if available
- activation='gelu'
- dropout=config.dropout

Output:
For each nutrient token position, predict scalar normalized log value.
Use either:
A. one shared MLP head that takes token hidden state plus nutrient id embedding, or
B. one linear head shared across all nutrients.

Prefer shared MLP head for simplicity:
pred = pred_mlp(hidden_token).squeeze(-1)

CLS embedding:
Return final hidden state at CLS position as food representation.

11. Optional attention mask
Implement padding mask at minimum.

Do not implement complex custom masked-attention first unless easy. Full attention with padding mask is acceptable for v1, as long as the masked tokens do not contain the true value.

12. Loss
Main loss:
SmoothL1 / Huber loss on masked observed nutrient positions only.

loss = weighted SmoothL1(predicted_value, target_value)
weights = quality_weight * nutrient_coverage_weight

Nutrient coverage weight:
coverage_j = number of foods with nutrient j observed
nutrient_weight_j = 1 / sqrt(coverage_j)
clip to [0.5, 3.0]

Total reconstruction loss:
sum(loss * weights * mask_target) / sum(weights * mask_target)

Representation consistency loss:
For each batch, optionally create two masked views of same foods and compute:
1 - cosine_similarity(cls_view1, cls_view2)
Start with consistency_loss_weight = 0.05.
If this complicates training too much, implement it but allow turning off in config.

Final loss:
loss_total = loss_reconstruction + consistency_loss_weight * loss_consistency

13. Training loop
Implement:
- train_one_epoch
- evaluate
- early stopping on validation masked reconstruction loss
- gradient clipping
- AdamW
- learning rate scheduler optional

Log:
- train loss
- validation loss
- validation MAE on normalized log scale
- validation RMSE on normalized log scale

Save the best checkpoint.

14. Evaluation
On validation and test sets:
- Perform repeated stochastic masking, e.g. 10 repeats.
- Compute metrics only on masked observed values:
  - MAE
  - RMSE
  - Pearson r
  - Spearman rho
  - R2
- Compute macro average per nutrient.
- Compute weighted average by nutrient coverage.
- Create a per-nutrient metrics dataframe.

Also evaluate baselines:
Baseline 1: predict train median per nutrient.
Baseline 2: text-only MLP from text embedding to nutrient vector, trained with masked observed loss or multitarget loss.
Baseline 3: nutrient-only transformer without text token, if easy.

If baselines are too much for first pass, at least implement train-median baseline.

15. Embedding extraction
After training, extract three embeddings for every food:
- z_full: text + all observed nutrients, no masking
- z_nutrient_only: all observed nutrients, no text
- z_text_only: text token plus masked nutrient ID tokens, or text-only prediction mode

If implementing all three is too much, implement at least z_full.

Save:
- food_embeddings_full.npy
- food_embeddings_text_only.npy if available
- food_embeddings_nutrient_only.npy if available
- food_metadata.csv
- nutrient_metadata.csv
- nutrient_scaling.json
- model_checkpoint.pt
- config.json
- train_val_test_split.json
- per_nutrient_metrics.csv
- overall_metrics.json
- training_history.csv

Use the same output directory convention as the old notebook.

16. Visualization
Add simple diagnostic plots:
- training and validation loss curves
- per-nutrient MAE bar chart
- observed nutrient coverage bar chart
- UMAP or PCA of z_full embeddings colored by food group if food group exists
- nearest-neighbor examples using cosine similarity on z_full

For nearest neighbors, print examples for several foods:
- apple-like foods
- milk-like foods
- beef/meat-like foods
- oil/fat-like foods
- leafy vegetables if present

17. Important exclusions
Do not use:
- nutrient bins
- compound tokens
- pathway tokens
- food taxonomy tokens
- health-effect labels
- disease labels
- ontology labels as model inputs

Food group may be used only for visualization and evaluation, not as training input.

18. Robustness requirements
The notebook must include checks and clear printed diagnostics:
- number of foods loaded
- number of nutrient rows loaded
- number of unique nutrients
- matrix shape before and after filtering
- missingness rate
- top nutrients by coverage
- foods dropped due to low nutrient count
- nutrients dropped due to low coverage
- examples of final food text
- examples of final observed nutrient vectors

Make the code robust to column name variants:
- id vs food_id
- source_id vs nutrient_id
- name vs orig_food_common_name, etc., if relevant from the old notebook

If a required column is missing, print available columns and raise a clear error.

19. Colab execution
The notebook should be runnable top-to-bottom in Colab.
Use GPU if available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Avoid dependencies that require system-level installation unless necessary.
Do not require internet except optional sentence-transformers installation; include fallback TF-IDF/SVD.

20. Deliverable
Write the new notebook file:
fooddb_nutrient_transformer_pretraining_colab.ipynb

Also, if working in a repo, add or update a short README section explaining:
- what this notebook does
- how it differs from structured_transformer_pretraining_colab.ipynb
- where outputs are saved
- what embeddings are produced

Implementation priorities:
First make a complete runnable notebook.
Then optimize architecture.
Do not over-engineer.
The key correctness requirements are:
- no bins
- continuous log-transformed targets
- missing values masked out of loss
- nutrient identity separate from nutrient value
- no positional embeddings
- small transformer
- FoodDB nutrients only as training target
- text name/description only as semantic context
- reuse old notebook’s connection/loading/saving infrastructure