#!/usr/bin/env python3
"""Build a conservative Taiwan-versus-source-corpus food-overlap ledger.

The old Taiwan ``name_novel`` flag only detected an exact normalized English
name.  This audit additionally removes preparation/form words and detects an
informative ingredient/species phrase inside a longer source name.  Matches are
exclusion candidates, not claims that two analytical samples are identical.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


MODIFIERS = {
    "added", "air", "baby", "baked", "barbecued", "blanched", "boiled",
    "boneless", "bottled", "braised", "brand", "canned", "chilled",
    "commercial", "concentrated", "cooked", "cooking", "crushed", "cut",
    "dehydrated", "diced", "dried", "drained", "dry", "fat", "fatfree",
    "flaked", "flavored", "flavoured", "fresh", "fried", "frozen", "grilled",
    "ground", "homemade", "instant", "lowfat", "mashed", "minced", "no",
    "packaged", "peeled", "powder", "powdered", "prepared", "processed",
    "raw", "reconstituted", "reduced", "roasted", "salt", "salted", "sliced",
    "smoked", "steamed", "stewed", "sun", "toasted", "unsalted", "with",
    "without", "and", "or", "in", "of", "the", "from", "as", "style",
}

# A one-token match is too broad for these words.  Multi-token signatures can
# still contain them, e.g. ``salmon oil`` or ``soy milk``.
GENERIC_SINGLETONS = {
    "bean", "beef", "bread", "cake", "cereal", "cheese", "chicken", "coffee",
    "cream", "dish", "drink", "egg", "fish", "flour", "food", "fruit", "juice",
    "meat", "milk", "noodle", "nut", "oil", "pork", "product", "rice", "sauce",
    "seed", "soup", "tea", "vegetable", "water",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize(token: str) -> str:
    irregular = {
        "berries": "berry", "fishes": "fish", "leaves": "leaf",
        "potatoes": "potato", "tomatoes": "tomato", "loaves": "loaf",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def core_tokens(value: object) -> tuple[str, ...]:
    tokens = []
    for token in normalize_name(value).split():
        if token.isdigit() or token in MODIFIERS:
            continue
        token = singularize(token)
        if len(token) > 1:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def informative(tokens: tuple[str, ...]) -> bool:
    return bool(tokens) and (len(tokens) >= 2 or tokens[0] not in GENERIC_SINGLETONS)


def aliases(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return sorted({
        normalize_name(part)
        for part in str(value).split(";")
        if normalize_name(part)
    })


def match_one(alias: str, source_records, exact_index, token_index):
    exact = exact_index.get(alias, [])
    if exact:
        return "exact_normalized_name", exact[0]
    query = core_tokens(alias)
    if not informative(query):
        return "", None
    candidates = None
    for token in query:
        rows = token_index.get(token, set())
        candidates = set(rows) if candidates is None else candidates & rows
    if not candidates:
        return "", None
    query_set = set(query)
    ranked = []
    for index in candidates:
        source = source_records[index]
        source_set = source["core_set"]
        if query_set <= source_set or source_set <= query_set:
            smaller = query if len(query) <= len(source_set) else tuple(source_set)
            if informative(tuple(smaller)):
                union = len(query_set | source_set)
                ranked.append((len(query_set & source_set) / union, -union, index))
    if not ranked:
        return "", None
    _, _, index = max(ranked)
    return "informative_core_containment", source_records[index]


def identity_key(english_names: object, food_id: str) -> str:
    candidates = [core_tokens(alias) for alias in aliases(english_names)]
    candidates = [tokens for tokens in candidates if informative(tokens)]
    if not candidates:
        return f"unresolved:{food_id}"
    # Prefer the most specific alias so generic synonyms do not collapse unlike foods.
    best = max(candidates, key=lambda value: (len(value), value))
    return "core:" + " ".join(sorted(best))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    profiles = pd.read_csv(args.data_dir / "profiles.csv.gz", low_memory=False)
    foods = pd.read_parquet(args.audit_dir / "foods.parquet")
    mapped = pd.read_parquet(args.audit_dir / "mapped_values.parquet")
    coverage = pd.read_csv(args.audit_dir / "axis_coverage.csv")
    axes = set(coverage.loc[coverage.strict_sample_backed_foods.ge(20), "axis_index"])
    strict_ids = set(mapped.loc[
        mapped.strict_sample_backed_target.astype(bool)
        & mapped.axis_index.isin(axes), "food_id"
    ])
    source_records = []
    exact_index: dict[str, list] = defaultdict(list)
    token_index: dict[str, set] = defaultdict(set)
    for row in profiles[[
        "profile_index", "partition", "source_key", "original_name"
    ]].itertuples(index=False):
        normalized = normalize_name(row.original_name)
        core = core_tokens(row.original_name)
        record = {
            "profile_index": int(row.profile_index),
            "partition": str(row.partition),
            "source_key": str(row.source_key),
            "original_name": str(row.original_name),
            "normalized": normalized,
            "core_set": set(core),
        }
        index = len(source_records)
        source_records.append(record)
        if normalized:
            exact_index[normalized].append(record)
        if informative(core):
            for token in core:
                token_index[token].add(index)

    ledger = []
    for food in foods.itertuples(index=False):
        food_aliases = aliases(food.food_name_en)
        matches = []
        for alias in food_aliases:
            basis, match = match_one(alias, source_records, exact_index, token_index)
            if match is not None:
                matches.append((basis, alias, match))
        train = [item for item in matches if item[2]["partition"] == "train"]
        dataset = matches
        chosen = (train or dataset or [None])[0]
        has_informative = any(informative(core_tokens(alias)) for alias in food_aliases)
        in_panel = bool(food.food_id in strict_ids and food.mapped_eligible_axes >= 10)
        external_clean = in_panel and has_informative and not dataset
        ledger.append({
            "food_id": food.food_id,
            "food_name_en": food.food_name_en,
            "food_category": food.food_category,
            "in_strict_panel": in_panel,
            "has_informative_english_alias": has_informative,
            "old_exact_train_overlap": bool(food.english_name_seen_in_pretraining),
            "overlap_with_train": bool(train),
            "overlap_with_any_project_profile": bool(dataset),
            "external_clean_eligible": external_clean,
            "identity_group_key": identity_key(food.food_name_en, str(food.food_id)),
            "match_basis": "" if chosen is None else chosen[0],
            "matched_alias": "" if chosen is None else chosen[1],
            "matched_partition": "" if chosen is None else chosen[2]["partition"],
            "matched_source_key": "" if chosen is None else chosen[2]["source_key"],
            "matched_source_name": "" if chosen is None else chosen[2]["original_name"],
        })
    ledger = pd.DataFrame(ledger).sort_values("food_id").reset_index(drop=True)
    ledger.to_csv(args.output_dir / "food_overlap_ledger.csv", index=False, encoding="utf-8-sig")

    clean = ledger[ledger.external_clean_eligible]
    summary = {
        "status": "complete",
        "version": "taiwan_corpus_overlap_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strict_panel_foods": int(ledger.in_strict_panel.sum()),
        "old_exact_train_overlap_foods": int((ledger.in_strict_panel & ledger.old_exact_train_overlap).sum()),
        "expanded_train_overlap_foods": int((ledger.in_strict_panel & ledger.overlap_with_train).sum()),
        "expanded_any_project_overlap_foods": int((ledger.in_strict_panel & ledger.overlap_with_any_project_profile).sum()),
        "unresolved_no_informative_english_alias_foods": int((ledger.in_strict_panel & ~ledger.has_informative_english_alias).sum()),
        "external_clean_foods": int(len(clean)),
        "external_clean_identity_groups": int(clean.identity_group_key.nunique()),
        "policy": {
            "reference_scope": "all 75,875 project train+validation source-native profiles",
            "exclude": "any exact normalized alias or informative core-token containment match; also exclude foods lacking an informative English alias",
            "interpretation": "conservative lexical exclusion candidate, not proof of biological sample identity",
            "fewshot_split": "keep identity_group_key intact across adaptation and test",
        },
        "input_hashes": {
            "profiles.csv.gz": sha256(args.data_dir / "profiles.csv.gz"),
            "foods.parquet": sha256(args.audit_dir / "foods.parquet"),
            "mapped_values.parquet": sha256(args.audit_dir / "mapped_values.parquet"),
            "axis_coverage.csv": sha256(args.audit_dir / "axis_coverage.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
