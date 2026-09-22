#!/usr/bin/env python3
"""Build provenance-level FooDB food observations from Content.csv.

FooDB's Food.id is an umbrella concept. Content rows underneath it can represent
different source foods, plant parts, or preparations. This builder creates one
processed food per ``citation + orig_food_id + orig_food_part`` instead.

Only records with a numeric ``standard_content`` are emitted. An absent numeric
record remains missing in the resulting observation matrix; this builder never
creates zero-filled cells. The raw FooDB export is read-only.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "foodb_2020_04_07_csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "fooddb_observations_v1"
CHUNK_SIZE = 200_000

CONTENT_COLUMNS = [
    "id",
    "source_id",
    "source_type",
    "food_id",
    "orig_food_id",
    "orig_food_common_name",
    "orig_food_part",
    "orig_content",
    "orig_min",
    "orig_max",
    "orig_unit",
    "orig_citation",
    "citation",
    "citation_type",
    "standard_content",
    "preparation_type",
]


def normalize_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def display_value(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def entity_key(citation: pd.Series, orig_food_id: pd.Series, part: pd.Series) -> pd.Series:
    return citation.map(normalize_key) + "|" + orig_food_id.map(normalize_key) + "|" + part.map(normalize_key)


@dataclass
class EntityInfo:
    citation: str
    orig_food_id: str
    orig_food_part: str
    original_names: Counter[str] = field(default_factory=Counter)
    preparations: Counter[str] = field(default_factory=Counter)
    upper_food_ids: set[int] = field(default_factory=set)
    numeric_rows: int = 0

    def add(self, frame: pd.DataFrame) -> None:
        self.numeric_rows += len(frame)
        self.upper_food_ids.update(pd.to_numeric(frame["food_id"], errors="coerce").dropna().astype(int).tolist())
        self.original_names.update(name for name in frame["orig_food_common_name"].map(display_value) if name)
        self.preparations.update(value for value in frame["preparation_type"].map(display_value) if value)

    def preferred_name(self) -> str:
        if not self.original_names:
            return f"Unknown food ({self.orig_food_id})"
        return sorted(self.original_names.items(), key=lambda item: (-item[1], item[0].lower()))[0][0]

    def preparation_summary(self) -> str:
        return " | ".join(sorted(self.preparations)) if self.preparations else "unknown"


def selected_numeric_rows(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk[chunk["source_type"].astype("string").str.lower().isin(["nutrient", "compound"])].copy()
    chunk["standard_content"] = pd.to_numeric(chunk["standard_content"], errors="coerce")
    valid_key = chunk["citation"].map(normalize_key).ne("") & chunk["orig_food_id"].map(normalize_key).ne("")
    return chunk[chunk["standard_content"].notna() & valid_key].copy()


def first_pass(content_path: Path) -> tuple[dict[str, EntityInfo], dict[str, int]]:
    entities: dict[str, EntityInfo] = {}
    counts = defaultdict(int)
    for chunk in pd.read_csv(content_path, usecols=CONTENT_COLUMNS, chunksize=CHUNK_SIZE, low_memory=False):
        counts["raw_rows"] += len(chunk)
        selected = selected_numeric_rows(chunk)
        counts["numeric_rows_with_stable_origin"] += len(selected)
        if selected.empty:
            continue
        selected["_entity_key"] = entity_key(selected["citation"], selected["orig_food_id"], selected["orig_food_part"])
        for key, group in selected.groupby("_entity_key", sort=False):
            if key not in entities:
                first = group.iloc[0]
                entities[key] = EntityInfo(
                    citation=display_value(first["citation"]),
                    orig_food_id=display_value(first["orig_food_id"]),
                    orig_food_part=display_value(first["orig_food_part"]),
                )
            entities[key].add(group)
    return entities, dict(counts)


def build_food_table(entities: dict[str, EntityInfo]) -> tuple[pd.DataFrame, dict[str, int]]:
    records = []
    for observation_id, key in enumerate(sorted(entities), start=1):
        entity = entities[key]
        name = entity.preferred_name()
        part = entity.orig_food_part or "unknown"
        description = (
            f"source: {entity.citation}. source food id: {entity.orig_food_id}. "
            f"part: {part}. preparation: {entity.preparation_summary()}."
        )
        records.append(
            {
                "id": observation_id,
                "public_id": f"FOODOBS{observation_id:06d}",
                "name": name,
                "description": description,
                "food_group": "unknown",
                "source": "fooddb_observation",
                "origin_group_id": f"{normalize_key(entity.citation)}|{normalize_key(entity.orig_food_id)}",
                "origin_entity_key": key,
                "citation": entity.citation,
                "orig_food_id": entity.orig_food_id,
                "orig_food_part": entity.orig_food_part,
                "preparation_types": entity.preparation_summary(),
                "upper_food_id_count": len(entity.upper_food_ids),
                "upper_food_ids": " | ".join(str(value) for value in sorted(entity.upper_food_ids)),
                "original_name_count": len(entity.original_names),
                "original_names": " | ".join(sorted(entity.original_names)),
                "numeric_content_rows": entity.numeric_rows,
                "review_status": "needs_review" if len(entity.original_names) > 1 else "ready",
            }
        )
    manifest = pd.DataFrame(records)
    food_columns = ["id", "public_id", "name", "description", "food_group", "source", "origin_group_id", "origin_entity_key"]
    return manifest, {key: int(value) for key, value in zip(manifest.origin_entity_key, manifest.id)}, manifest[food_columns]


def second_pass(content_path: Path, output_path: Path, id_by_key: dict[str, int]) -> dict[str, int]:
    rows_written = 0
    first_chunk = True
    output_columns = [
        "source_content_id",
        "source_id",
        "source_type",
        "food_id",
        "orig_content",
        "orig_min",
        "orig_max",
        "orig_unit",
        "standard_content",
        "orig_citation",
        "citation",
        "citation_type",
        "orig_food_id",
        "orig_food_common_name",
        "orig_food_part",
        "preparation_type",
        "origin_entity_key",
    ]
    for chunk in pd.read_csv(content_path, usecols=CONTENT_COLUMNS, chunksize=CHUNK_SIZE, low_memory=False):
        selected = selected_numeric_rows(chunk)
        if selected.empty:
            continue
        selected["origin_entity_key"] = entity_key(selected["citation"], selected["orig_food_id"], selected["orig_food_part"])
        selected["food_id"] = selected["origin_entity_key"].map(id_by_key)
        if selected["food_id"].isna().any():
            raise ValueError("An eligible Content row did not resolve to a generated observation ID.")
        selected["food_id"] = selected["food_id"].astype(int)
        selected = selected.rename(columns={"id": "source_content_id"})
        selected.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False, columns=output_columns, quoting=csv.QUOTE_MINIMAL)
        rows_written += len(selected)
        first_chunk = False
    if first_chunk:
        raise ValueError("No observation-level numeric Content rows were written.")
    return {"content_rows_written": rows_written}


def write_readme(output_dir: Path, manifest: pd.DataFrame, counts: dict[str, int]) -> None:
    ready = int(manifest.review_status.eq("ready").sum())
    review = int(manifest.review_status.eq("needs_review").sum())
    text = f"""# FooDB Observation-Level Dataset v1

This processed dataset replaces FooDB's umbrella `Food.id` unit with a
source-level observation key:

```text
citation + orig_food_id + orig_food_part
```

`preparation_type` remains metadata and text context. It is not part of the
key because source-native food IDs and original names generally already encode
preparation, while using it as a key would create unnecessary splits from
inconsistent metadata.

## Contents

- `Food.csv`: {len(manifest):,} observation-level foods.
- `Content.csv`: {counts['content_rows_written']:,} numeric nutrient/compound
  records. Absence of a row remains missing.
- `entity_manifest.csv`: provenance, original names, preparation summaries,
  and the old FooDB upper food IDs used only for traceability.

## Important constraints

- The source raw export is unchanged at `data/foodb_2020_04_07_csv/`.
- {counts['raw_rows'] - counts['numeric_rows_with_stable_origin']:,} raw
  Content rows are intentionally not included because they lack either a
  numeric standard value or a stable source-level food key.
- {review:,} observations have multiple original names under the same key and
  are marked `needs_review`; {ready:,} have a single name.
- Values still require axis-specific unit/basis harmonization before cross-
  database numerical pooling. This dataset solves food-entity granularity, not
  chemical-axis equivalence or dry/fresh-weight conversion.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    content_path = raw_dir / "Content.csv"
    if not content_path.exists():
        raise FileNotFoundError(f"Raw FooDB Content.csv not found: {content_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    entities, counts = first_pass(content_path)
    manifest, id_by_key, foods = build_food_table(entities)
    # This exact raw export has 11,449 provenance-level observations under
    # citation + orig_food_id + orig_food_part. Do not discard valid source
    # records merely to match an earlier, non-reproducible count of 9,458.
    expected_count = 11_449
    if len(manifest) != expected_count:
        raise ValueError(f"Expected {expected_count:,} observation-level foods for this raw export, found {len(manifest):,}.")
    foods.to_csv(output_dir / "Food.csv", index=False)
    manifest.to_csv(output_dir / "entity_manifest.csv", index=False)
    counts.update(second_pass(content_path, output_dir / "Content.csv", id_by_key))
    output_foods = pd.read_csv(output_dir / "Food.csv", usecols=["id"])
    output_content_foods = pd.read_csv(output_dir / "Content.csv", usecols=["food_id"])["food_id"].nunique()
    if output_foods["id"].nunique() != expected_count or output_content_foods != expected_count:
        raise ValueError("Observation output validation failed: food IDs are not a complete one-to-one set.")
    write_readme(output_dir, manifest, counts)
    print(f"Built {len(manifest):,} FooDB observation-level foods at {output_dir}")
    print(f"Numeric Content rows retained: {counts['content_rows_written']:,}")
    print(f"Rows with stable origin: {counts['numeric_rows_with_stable_origin']:,} / {counts['raw_rows']:,}")


if __name__ == "__main__":
    main()
