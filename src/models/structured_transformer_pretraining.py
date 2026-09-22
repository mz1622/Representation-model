from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class TransformerPretrainingConfig:
    data_dir: str = "/content/drive/MyDrive/food_model/data"
    output_dir: str = "/content/drive/MyDrive/food_model/outputs/structured_transformer"
    raw_foodb_dir: str = "/content/drive/MyDrive/food_model/data/raw/foodb_2020_04_07_csv"
    alternate_raw_foodb_dir: str = "/content/drive/MyDrive/FoodNutrition/foodb_2020_04_07_csv"
    processed_filename: str = "food_model_table_transformer.csv"
    force_rebuild_processed_data: bool = False
    min_compound_observations: int = 20
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_batch_size: int = 64
    seed: int = 42
    batch_size: int = 16
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    numeric_mask_prob: float = 0.15
    group_subgroup_mask_prob: float = 0.30
    lambda_nutrient: float = 1.0
    lambda_compound: float = 1.0
    lambda_group: float = 0.1
    lambda_subgroup: float = 0.1


TEXT_FIELD_COLUMNS = ["food_name", "food_description", "food_scientific_name"]
BASE_METADATA_COLUMNS = [
    "food_id",
    "food_name",
    "food_description",
    "food_group",
    "food_scientific_name",
    "food_subgroup",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify_feature_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError(f"Could not create a valid feature name from: {name}")
    return text


def validate_required_columns(frame: pd.DataFrame, required_columns: list[str], frame_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")


def find_raw_foodb_dir(config: TransformerPretrainingConfig) -> Path:
    candidate_paths = [
        Path(config.raw_foodb_dir),
        Path(config.alternate_raw_foodb_dir),
        Path("foodb_2020_04_07_csv"),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_dir():
            print(f"Using raw FoodDB directory: {candidate_path}")
            return candidate_path
    raise FileNotFoundError(
        "Could not find the raw FoodDB directory. Expected one of: "
        f"{candidate_paths}. In Colab, mount Drive and put the FooDB CSV folder under data/raw."
    )


def build_food_metadata(food_df: pd.DataFrame) -> pd.DataFrame:
    metadata = food_df[
        [
            "id",
            "public_id",
            "name",
            "description",
            "food_group",
            "food_subgroup",
            "name_scientific",
        ]
    ].copy()
    metadata = metadata.rename(
        columns={
            "id": "_food_internal_id",
            "public_id": "food_id",
            "name": "food_name",
            "description": "food_description",
            "name_scientific": "food_scientific_name",
        }
    )
    metadata["food_id"] = metadata["food_id"].fillna(metadata["_food_internal_id"].astype(str))
    return metadata[
        [
            "_food_internal_id",
            "food_id",
            "food_name",
            "food_description",
            "food_scientific_name",
            "food_group",
            "food_subgroup",
        ]
    ].drop_duplicates(subset=["_food_internal_id"])


def build_feature_table(
    content_df: pd.DataFrame,
    source_lookup_df: pd.DataFrame,
    source_type: str,
    prefix: str,
    min_food_observations: int = 1,
    allow_missing_source_names: bool = False,
) -> pd.DataFrame:
    content_columns = ["food_id", "source_id", "standard_content"]
    if "orig_source_name" in content_df.columns:
        content_columns.append("orig_source_name")

    subset = content_df.loc[
        (content_df["source_type"] == source_type) & content_df["standard_content"].notna(),
        content_columns,
    ].copy()
    if subset.empty:
        print(f"No usable {source_type.lower()} rows with standard_content were found.")
        return pd.DataFrame({"food_id": []})

    subset["food_id"] = pd.to_numeric(subset["food_id"], errors="raise").astype(int)
    subset["source_id"] = pd.to_numeric(subset["source_id"], errors="raise").astype(int)
    subset["standard_content"] = pd.to_numeric(subset["standard_content"], errors="coerce")
    if "orig_source_name" in subset.columns:
        subset["orig_source_name"] = subset["orig_source_name"].astype("string").str.strip()
        subset.loc[subset["orig_source_name"] == "", "orig_source_name"] = pd.NA
    subset = subset.dropna(subset=["standard_content"])

    merged = subset.merge(
        source_lookup_df[["id", "name"]].rename(columns={"id": "source_id", "name": "source_name"}),
        on="source_id",
        how="left",
        validate="many_to_one",
    )

    missing_source_names = int(merged["source_name"].isna().sum())
    if missing_source_names > 0:
        if not allow_missing_source_names:
            raise ValueError(
                f"{missing_source_names} {source_type.lower()} content rows could not be matched to source names."
            )

        missing_mask = merged["source_name"].isna()
        missing_source_ids = merged.loc[missing_mask, "source_id"].nunique()
        print(
            f"Warning: {missing_source_names} {source_type.lower()} content rows across "
            f"{missing_source_ids} source ids could not be matched to {source_type}.csv names."
        )

        if "orig_source_name" in merged.columns:
            fallback_mask = missing_mask & merged["orig_source_name"].notna()
            fallback_rows = int(fallback_mask.sum())
            fallback_source_ids = merged.loc[fallback_mask, "source_id"].nunique()
            if fallback_rows > 0:
                merged.loc[fallback_mask, "source_name"] = merged.loc[fallback_mask, "orig_source_name"]
                print(
                    f"Recovered {fallback_rows} {source_type.lower()} content row names across "
                    f"{fallback_source_ids} source ids from Content.orig_source_name."
                )

        unresolved_mask = merged["source_name"].isna()
        unresolved_rows = int(unresolved_mask.sum())
        if unresolved_rows > 0:
            unresolved_source_ids = merged.loc[unresolved_mask, "source_id"].nunique()
            print(
                f"Dropping {unresolved_rows} {source_type.lower()} content rows across "
                f"{unresolved_source_ids} source ids because no source name is available."
            )
            merged = merged.loc[~unresolved_mask].copy()

    if merged.empty:
        print(f"No named {source_type.lower()} rows remain after source-name recovery.")
        return pd.DataFrame({"food_id": []})

    feature_counts = merged.groupby("source_name")["food_id"].nunique().sort_values(ascending=False)
    selected_feature_names = feature_counts[feature_counts >= min_food_observations].index.tolist()
    merged = merged[merged["source_name"].isin(selected_feature_names)].copy()
    merged["feature_name"] = prefix + merged["source_name"].map(slugify_feature_name)

    duplicate_feature_names = (
        merged[["source_id", "feature_name"]].drop_duplicates()["feature_name"].duplicated(keep=False)
    )
    if duplicate_feature_names.any():
        duplicate_names = (
            merged[["source_id", "feature_name"]]
            .drop_duplicates()
            .loc[duplicate_feature_names, "feature_name"]
        )
        duplicate_mask = merged["feature_name"].isin(duplicate_names)
        merged.loc[duplicate_mask, "feature_name"] = (
            merged.loc[duplicate_mask, "feature_name"]
            + "__id"
            + merged.loc[duplicate_mask, "source_id"].astype(str)
        )

    print(
        f"{source_type}: kept {len(selected_feature_names)} features with at least "
        f"{min_food_observations} foods and {len(merged)} observed rows."
    )

    aggregated = (
        merged.groupby(["food_id", "feature_name"], as_index=False)["standard_content"]
        .mean()
        .rename(columns={"standard_content": "feature_value"})
    )
    wide = aggregated.pivot(index="food_id", columns="feature_name", values="feature_value").reset_index()
    wide.columns.name = None
    return wide


def build_processed_food_model_table(raw_foodb_dir: Path, config: TransformerPretrainingConfig) -> pd.DataFrame:
    food_path = raw_foodb_dir / "Food.csv"
    content_path = raw_foodb_dir / "Content.csv"
    nutrient_path = raw_foodb_dir / "Nutrient.csv"
    compound_path = raw_foodb_dir / "Compound.csv"
    for path in [food_path, content_path, nutrient_path, compound_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required raw FoodDB file is missing: {path}")

    print("Loading raw FoodDB tables...")
    food_df = pd.read_csv(
        food_path,
        usecols=["id", "public_id", "name", "description", "food_group", "food_subgroup", "name_scientific"],
    )
    content_df = pd.read_csv(
        content_path,
        usecols=["food_id", "source_id", "source_type", "orig_source_name", "standard_content"],
        dtype={"orig_source_name": "string"},
    )
    nutrient_df = pd.read_csv(nutrient_path, usecols=["id", "name"])
    compound_df = pd.read_csv(compound_path, usecols=["id", "name"])

    validate_required_columns(food_df, ["id", "public_id", "name"], "Food.csv")
    validate_required_columns(content_df, ["food_id", "source_id", "source_type", "standard_content"], "Content.csv")
    validate_required_columns(nutrient_df, ["id", "name"], "Nutrient.csv")
    validate_required_columns(compound_df, ["id", "name"], "Compound.csv")

    metadata = build_food_metadata(food_df)
    nutrient_wide = build_feature_table(
        content_df=content_df,
        source_lookup_df=nutrient_df,
        source_type="Nutrient",
        prefix="nutrient__",
        min_food_observations=1,
    )
    compound_wide = build_feature_table(
        content_df=content_df,
        source_lookup_df=compound_df,
        source_type="Compound",
        prefix="compound__",
        min_food_observations=config.min_compound_observations,
        allow_missing_source_names=True,
    )

    processed_df = metadata.merge(
        nutrient_wide.rename(columns={"food_id": "_food_internal_id"}),
        on="_food_internal_id",
        how="left",
        validate="one_to_one",
    )
    processed_df = processed_df.merge(
        compound_wide.rename(columns={"food_id": "_food_internal_id"}),
        on="_food_internal_id",
        how="left",
        validate="one_to_one",
    )
    processed_df = processed_df.drop(columns=["_food_internal_id"])

    nutrient_columns = sorted([column for column in processed_df.columns if column.startswith("nutrient__")])
    compound_columns = sorted([column for column in processed_df.columns if column.startswith("compound__")])
    if not nutrient_columns and not compound_columns:
        raise ValueError("Processed table contains no nutrient__ or compound__ columns.")

    print(f"Processed dataframe shape: {processed_df.shape}")
    print(f"Processed nutrients: {len(nutrient_columns)}")
    print(f"Processed compounds: {len(compound_columns)}")
    return processed_df


def load_or_build_processed_table(config: TransformerPretrainingConfig) -> pd.DataFrame:
    data_dir = Path(config.data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_path = processed_dir / config.processed_filename

    if data_path.exists() and not config.force_rebuild_processed_data:
        df = pd.read_csv(data_path)
        missing_columns = [column for column in BASE_METADATA_COLUMNS if column not in df.columns]
        stale_fallback_compound_columns = [
            column for column in df.columns if column.startswith("compound__source_id_")
        ]
        if not missing_columns and not stale_fallback_compound_columns:
            print(f"Loaded processed dataset from: {data_path}")
            return df
        print(
            "Processed dataset is stale. "
            f"Missing metadata columns: {missing_columns}. "
            f"Fallback compound columns: {len(stale_fallback_compound_columns)}. Rebuilding."
        )

    raw_foodb_dir = find_raw_foodb_dir(config)
    df = build_processed_food_model_table(raw_foodb_dir, config)
    df.to_csv(data_path, index=False)
    print(f"Saved processed dataset to: {data_path}")
    return df


def normalize_text_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    value_str = str(value).strip()
    return value_str if value_str else "unknown"


def is_percentage_column(column_name: str) -> bool:
    return column_name.endswith("_pct") or column_name.endswith("_percent")


def prepare_target_bundle(frame: pd.DataFrame, columns: list[str], bundle_name: str) -> dict[str, object] | None:
    if not columns:
        print(f"No {bundle_name} columns found. Skipping {bundle_name} targets.")
        return None

    raw_values = frame[columns].astype(float)
    mask = (~raw_values.isna()).astype(np.float32)

    observed_values = raw_values.where(~raw_values.isna())
    min_value = observed_values.min().min(skipna=True)
    if pd.notna(min_value) and min_value < 0:
        negative_columns = [
            column for column in columns if (observed_values[column] < 0).fillna(False).any()
        ]
        raise ValueError(
            f"Negative values found in {bundle_name} columns: {negative_columns}. "
            "This transformer pretraining model expects non-negative values."
        )

    adjusted = raw_values.copy()
    percentage_columns = [column for column in columns if is_percentage_column(column)]
    if percentage_columns:
        adjusted.loc[:, percentage_columns] = adjusted.loc[:, percentage_columns] / 100.0

    transformed = np.log1p(adjusted.fillna(0.0))
    bundle = {
        "columns": columns,
        "raw": raw_values.to_numpy(dtype=np.float32),
        "mask": mask.to_numpy(dtype=np.float32),
        "transformed": transformed.to_numpy(dtype=np.float32),
        "percentage_columns": percentage_columns,
    }
    print(
        f"Prepared {bundle_name} bundle with shape raw={bundle['raw'].shape}, "
        f"mask={bundle['mask'].shape}, transformed={bundle['transformed'].shape}"
    )
    return bundle


def encode_text_fields(
    frame: pd.DataFrame,
    config: TransformerPretrainingConfig,
    device: torch.device,
    cache_path: Path,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "sentence-transformers is required for text-field encoding. "
            "Install it in Colab with: !pip install -q sentence-transformers"
        ) from error

    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        if payload.shape[0] == len(frame) and payload.shape[1] == len(TEXT_FIELD_COLUMNS):
            print(f"Loaded text-field embedding cache from: {cache_path}")
            return payload.astype(np.float32)
        print(
            f"Ignoring stale text-field embedding cache at {cache_path}. "
            f"Expected first dims ({len(frame)}, {len(TEXT_FIELD_COLUMNS)}), found {payload.shape}."
        )

    print(f"Encoding text fields with {config.sentence_model_name}...")
    sentence_model = SentenceTransformer(config.sentence_model_name, device=str(device))
    field_embeddings = []
    for field in TEXT_FIELD_COLUMNS:
        texts = frame[field].map(normalize_text_value).tolist()
        embeddings = sentence_model.encode(
            texts,
            batch_size=config.text_embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        field_embeddings.append(embeddings.astype(np.float32))

    stacked = np.stack(field_embeddings, axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, stacked)
    print(f"Saved text-field embedding cache to: {cache_path}")
    return stacked


class StructuredFoodPretrainingDataset(Dataset):
    def __init__(
        self,
        text_field_embeddings: np.ndarray,
        nutrient_targets: np.ndarray,
        nutrient_observed_mask: np.ndarray,
        compound_targets: np.ndarray,
        compound_observed_mask: np.ndarray,
        group_labels: np.ndarray,
        subgroup_labels: np.ndarray,
        numeric_mask_prob: float,
        group_subgroup_mask_prob: float,
    ) -> None:
        self.text_field_embeddings = torch.tensor(text_field_embeddings, dtype=torch.float32)
        self.nutrient_targets = torch.tensor(nutrient_targets, dtype=torch.float32)
        self.nutrient_observed_mask = torch.tensor(nutrient_observed_mask, dtype=torch.float32)
        self.compound_targets = torch.tensor(compound_targets, dtype=torch.float32)
        self.compound_observed_mask = torch.tensor(compound_observed_mask, dtype=torch.float32)
        self.group_labels = torch.tensor(group_labels, dtype=torch.long)
        self.subgroup_labels = torch.tensor(subgroup_labels, dtype=torch.long)
        self.numeric_mask_prob = numeric_mask_prob
        self.group_subgroup_mask_prob = group_subgroup_mask_prob

        num_rows = self.text_field_embeddings.shape[0]
        assert self.nutrient_targets.shape[0] == num_rows
        assert self.nutrient_observed_mask.shape == self.nutrient_targets.shape
        assert self.compound_targets.shape[0] == num_rows
        assert self.compound_observed_mask.shape == self.compound_targets.shape
        assert self.group_labels.shape[0] == num_rows
        assert self.subgroup_labels.shape[0] == num_rows

    def __len__(self) -> int:
        return self.text_field_embeddings.shape[0]

    def _mask_numeric_values(self, target: torch.Tensor, observed_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        random_values = torch.rand_like(target)
        prediction_mask = ((random_values < self.numeric_mask_prob) & (observed_mask > 0)).float()
        visible_observed_mask = ((observed_mask > 0) & (prediction_mask == 0)).float()

        input_values = target * visible_observed_mask
        value_state = torch.full_like(target, fill_value=2, dtype=torch.long)
        value_state[visible_observed_mask.bool()] = 0
        value_state[prediction_mask.bool()] = 1
        return input_values, value_state, prediction_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        nutrient_input, nutrient_state, nutrient_prediction_mask = self._mask_numeric_values(
            self.nutrient_targets[index],
            self.nutrient_observed_mask[index],
        )
        compound_input, compound_state, compound_prediction_mask = self._mask_numeric_values(
            self.compound_targets[index],
            self.compound_observed_mask[index],
        )

        group_subgroup_masked = torch.rand(()) < self.group_subgroup_mask_prob
        group_prediction_mask = torch.tensor(float(group_subgroup_masked), dtype=torch.float32)
        subgroup_prediction_mask = torch.tensor(float(group_subgroup_masked), dtype=torch.float32)

        return {
            "text_field_embeddings": self.text_field_embeddings[index],
            "nutrient_input": nutrient_input,
            "nutrient_state": nutrient_state,
            "nutrient_target": self.nutrient_targets[index],
            "nutrient_prediction_mask": nutrient_prediction_mask,
            "compound_input": compound_input,
            "compound_state": compound_state,
            "compound_target": self.compound_targets[index],
            "compound_prediction_mask": compound_prediction_mask,
            "group_label": self.group_labels[index],
            "subgroup_label": self.subgroup_labels[index],
            "group_prediction_mask": group_prediction_mask,
            "subgroup_prediction_mask": subgroup_prediction_mask,
        }


class StructuredFoodTransformer(nn.Module):
    VALUE_STATE_OBSERVED = 0
    VALUE_STATE_MASKED = 1
    VALUE_STATE_MISSING = 2

    def __init__(
        self,
        text_dim: int,
        num_text_fields: int,
        num_nutrients: int,
        num_compounds: int,
        num_groups: int,
        num_subgroups: int,
        config: TransformerPretrainingConfig,
    ) -> None:
        super().__init__()
        self.num_text_fields = num_text_fields
        self.num_nutrients = num_nutrients
        self.num_compounds = num_compounds
        self.num_groups = num_groups
        self.num_subgroups = num_subgroups
        self.group_mask_id = num_groups
        self.subgroup_mask_id = num_subgroups

        d_model = config.d_model
        sequence_length = 1 + num_text_fields + 2 + num_nutrients + num_compounds
        self.sequence_length = sequence_length

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.text_projection = nn.Linear(text_dim, d_model)
        self.token_type_embedding = nn.Embedding(8, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.group_embedding = nn.Embedding(num_groups + 1, d_model)
        self.subgroup_embedding = nn.Embedding(num_subgroups + 1, d_model)
        self.nutrient_feature_embedding = nn.Embedding(num_nutrients, d_model)
        self.compound_feature_embedding = nn.Embedding(num_compounds, d_model)
        self.value_state_embedding = nn.Embedding(3, d_model)
        self.value_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.numeric_head = nn.Linear(d_model, 1)
        self.group_head = nn.Linear(d_model, num_groups)
        self.subgroup_head = nn.Linear(d_model, num_subgroups)

        nn.init.normal_(self.cls_token, std=0.02)

    def _type_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ids = [0]
        ids.extend([1, 2, 3][: self.num_text_fields])
        ids.extend([4, 5])
        ids.extend([6] * self.num_nutrients)
        ids.extend([7] * self.num_compounds)
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_fields = batch["text_field_embeddings"]
        batch_size = text_fields.shape[0]
        device = text_fields.device

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        text_tokens = self.text_projection(text_fields)

        group_input_ids = batch["group_label"].clone()
        subgroup_input_ids = batch["subgroup_label"].clone()
        group_mask = batch["group_prediction_mask"].bool()
        subgroup_mask = batch["subgroup_prediction_mask"].bool()
        group_input_ids[group_mask] = self.group_mask_id
        subgroup_input_ids[subgroup_mask] = self.subgroup_mask_id

        group_token = self.group_embedding(group_input_ids).unsqueeze(1)
        subgroup_token = self.subgroup_embedding(subgroup_input_ids).unsqueeze(1)

        nutrient_ids = torch.arange(self.num_nutrients, device=device).unsqueeze(0).expand(batch_size, -1)
        compound_ids = torch.arange(self.num_compounds, device=device).unsqueeze(0).expand(batch_size, -1)

        nutrient_value_tokens = self.value_projection(batch["nutrient_input"].unsqueeze(-1))
        nutrient_tokens = (
            self.nutrient_feature_embedding(nutrient_ids)
            + nutrient_value_tokens
            + self.value_state_embedding(batch["nutrient_state"])
        )

        compound_value_tokens = self.value_projection(batch["compound_input"].unsqueeze(-1))
        compound_tokens = (
            self.compound_feature_embedding(compound_ids)
            + compound_value_tokens
            + self.value_state_embedding(batch["compound_state"])
        )

        tokens = torch.cat(
            [cls_token, text_tokens, group_token, subgroup_token, nutrient_tokens, compound_tokens],
            dim=1,
        )
        if tokens.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {tokens.shape[1]}.")

        position_ids = torch.arange(self.sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
        type_ids = self._type_ids(batch_size, device)
        tokens = tokens + self.position_embedding(position_ids) + self.token_type_embedding(type_ids)

        # Every sequence has the same real tokens, so no padding attention mask is needed.
        encoded = self.encoder(tokens)
        encoded = self.final_norm(encoded)

        group_position = 1 + self.num_text_fields
        subgroup_position = group_position + 1
        nutrient_start = subgroup_position + 1
        compound_start = nutrient_start + self.num_nutrients

        nutrient_encoded = encoded[:, nutrient_start:compound_start]
        compound_encoded = encoded[:, compound_start : compound_start + self.num_compounds]

        return {
            "food_embedding": encoded[:, 0],
            "group_logits": self.group_head(encoded[:, group_position]),
            "subgroup_logits": self.subgroup_head(encoded[:, subgroup_position]),
            "nutrient_pred": self.numeric_head(nutrient_encoded).squeeze(-1),
            "compound_pred": self.numeric_head(compound_encoded).squeeze(-1),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert pred.shape == target.shape
    assert mask.shape == target.shape
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / (mask.sum() + eps)


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active_mask = mask.bool()
    if not active_mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[active_mask], target[active_mask])


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: TransformerPretrainingConfig,
) -> dict[str, torch.Tensor]:
    nutrient_loss = masked_mse(
        outputs["nutrient_pred"],
        batch["nutrient_target"],
        batch["nutrient_prediction_mask"],
    )
    compound_loss = masked_mse(
        outputs["compound_pred"],
        batch["compound_target"],
        batch["compound_prediction_mask"],
    )
    group_loss = masked_cross_entropy(
        outputs["group_logits"],
        batch["group_label"],
        batch["group_prediction_mask"],
    )
    subgroup_loss = masked_cross_entropy(
        outputs["subgroup_logits"],
        batch["subgroup_label"],
        batch["subgroup_prediction_mask"],
    )
    total_loss = (
        config.lambda_nutrient * nutrient_loss
        + config.lambda_compound * compound_loss
        + config.lambda_group * group_loss
        + config.lambda_subgroup * subgroup_loss
    )
    return {
        "total": total_loss,
        "nutrient": nutrient_loss,
        "compound": compound_loss,
        "group": group_loss,
        "subgroup": subgroup_loss,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: TransformerPretrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    metric_sums = {"total": 0.0, "nutrient": 0.0, "compound": 0.0, "group": 0.0, "subgroup": 0.0}
    row_count = 0

    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(batch)
            losses = compute_losses(outputs, batch, config)
            if is_training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = int(batch["text_field_embeddings"].shape[0])
        row_count += batch_size
        for key in metric_sums:
            metric_sums[key] += float(losses[key].detach().cpu()) * batch_size

    return {key: value / max(row_count, 1) for key, value in metric_sums.items()}


def build_datasets(
    df: pd.DataFrame,
    text_field_embeddings: np.ndarray,
    nutrient_bundle: dict[str, object],
    compound_bundle: dict[str, object],
    config: TransformerPretrainingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, StructuredFoodPretrainingDataset], LabelEncoder, LabelEncoder]:
    group_encoder = LabelEncoder()
    subgroup_encoder = LabelEncoder()
    group_labels = group_encoder.fit_transform(df["food_group"].fillna("unknown").astype(str))
    subgroup_labels = subgroup_encoder.fit_transform(df["food_subgroup"].fillna("unknown").astype(str))

    unique_food_ids = df["food_id"].astype(str).drop_duplicates().to_numpy()
    train_ids, temp_ids = train_test_split(unique_food_ids, test_size=0.30, random_state=config.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=config.seed)

    split_map = {food_id: "train" for food_id in train_ids}
    split_map.update({food_id: "val" for food_id in val_ids})
    split_map.update({food_id: "test" for food_id in test_ids})
    split_series = df["food_id"].astype(str).map(split_map)
    if split_series.isna().any():
        missing = df.loc[split_series.isna(), "food_id"].astype(str).unique().tolist()
        raise ValueError(f"Some food_ids were not assigned to a split: {missing}")

    split_indices = {
        split_name: df.index[split_series == split_name].to_numpy()
        for split_name in ["train", "val", "test"]
    }
    for split_name, indices in split_indices.items():
        print(f"{split_name}: rows={len(indices)}, unique_foods={df.loc[indices, 'food_id'].nunique()}")

    datasets = {}
    for split_name, indices in split_indices.items():
        datasets[split_name] = StructuredFoodPretrainingDataset(
            text_field_embeddings=text_field_embeddings[indices],
            nutrient_targets=nutrient_bundle["transformed"][indices],
            nutrient_observed_mask=nutrient_bundle["mask"][indices],
            compound_targets=compound_bundle["transformed"][indices],
            compound_observed_mask=compound_bundle["mask"][indices],
            group_labels=group_labels[indices],
            subgroup_labels=subgroup_labels[indices],
            numeric_mask_prob=config.numeric_mask_prob,
            group_subgroup_mask_prob=config.group_subgroup_mask_prob,
        )
    return split_indices, datasets, group_encoder, subgroup_encoder


def save_label_mapping(path: Path, encoder: LabelEncoder) -> None:
    mapping = {str(index): label for index, label in enumerate(encoder.classes_.tolist())}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)


def main() -> None:
    config = TransformerPretrainingConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "structured_food_transformer_pretraining.pt"
    config_path = output_dir / "structured_food_transformer_config.json"
    metrics_path = output_dir / "structured_food_transformer_history.csv"
    text_cache_path = Path(config.data_dir) / "processed" / "food_text_field_embeddings_transformer.npy"

    df = load_or_build_processed_table(config)
    for column in BASE_METADATA_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Processed table is missing required metadata column: {column}")

    nutrient_columns = sorted([column for column in df.columns if column.startswith("nutrient__")])
    compound_columns = sorted([column for column in df.columns if column.startswith("compound__")])
    if not nutrient_columns or not compound_columns:
        raise ValueError("Expected both nutrient__ and compound__ columns for transformer pretraining.")

    nutrient_bundle = prepare_target_bundle(df, nutrient_columns, "nutrient")
    compound_bundle = prepare_target_bundle(df, compound_columns, "compound")
    assert nutrient_bundle is not None
    assert compound_bundle is not None

    text_field_embeddings = encode_text_fields(df, config, device, text_cache_path)
    split_indices, datasets, group_encoder, subgroup_encoder = build_datasets(
        df=df,
        text_field_embeddings=text_field_embeddings,
        nutrient_bundle=nutrient_bundle,
        compound_bundle=compound_bundle,
        config=config,
    )

    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=config.batch_size, shuffle=False)

    model = StructuredFoodTransformer(
        text_dim=text_field_embeddings.shape[-1],
        num_text_fields=text_field_embeddings.shape[1],
        num_nutrients=len(nutrient_columns),
        num_compounds=len(compound_columns),
        num_groups=len(group_encoder.classes_),
        num_subgroups=len(subgroup_encoder.classes_),
        config=config,
    ).to(device)

    print(f"Transformer sequence length: {model.sequence_length}")
    print(f"Text dim: {text_field_embeddings.shape[-1]}")
    print(f"Groups: {len(group_encoder.classes_)}")
    print(f"Subgroups: {len(subgroup_encoder.classes_)}")
    print(f"Nutrients: {len(nutrient_columns)}")
    print(f"Compounds: {len(compound_columns)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, config, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, config, device, optimizer=None)
        history_row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(history_row)
        print(
            f"Epoch {epoch:02d} | "
            f"train total={train_metrics['total']:.4f} nutrient={train_metrics['nutrient']:.4f} "
            f"compound={train_metrics['compound']:.4f} group={train_metrics['group']:.4f} "
            f"subgroup={train_metrics['subgroup']:.4f} | "
            f"val total={val_metrics['total']:.4f} nutrient={val_metrics['nutrient']:.4f} "
            f"compound={val_metrics['compound']:.4f} group={val_metrics['group']:.4f} "
            f"subgroup={val_metrics['subgroup']:.4f}"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "nutrient_columns": nutrient_columns,
                    "compound_columns": compound_columns,
                    "text_field_columns": TEXT_FIELD_COLUMNS,
                    "group_classes": group_encoder.classes_.tolist(),
                    "subgroup_classes": subgroup_encoder.classes_.tolist(),
                    "split_indices": {key: value.tolist() for key, value in split_indices.items()},
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping after {config.patience} epochs without validation improvement.")
                break

    pd.DataFrame(history).to_csv(metrics_path, index=False)
    save_label_mapping(output_dir / "group_label_mapping.json", group_encoder)
    save_label_mapping(output_dir / "subgroup_label_mapping.json", subgroup_encoder)

    config_payload = asdict(config)
    config_payload.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_foods": int(len(df)),
            "num_nutrients": int(len(nutrient_columns)),
            "num_compounds": int(len(compound_columns)),
            "num_groups": int(len(group_encoder.classes_)),
            "num_subgroups": int(len(subgroup_encoder.classes_)),
            "sequence_length": int(model.sequence_length),
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
        }
    )
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    print(f"Saved training history to: {metrics_path}")
    print(f"Saved config to: {config_path}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
