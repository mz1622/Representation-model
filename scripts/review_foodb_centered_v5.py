"""Inventory FooDB content identities without changing immutable source files.

Run from the repository root: python scripts/review_foodb_centered_v5.py
Dependencies: pandas, numpy (requirements-colab.txt).
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.ontology import parse_obo
from foodcomp.util import stable_id, write_csv


def main():
    output = ROOT / "data/audits/scientific_food_composition_v5/review"
    raw = ROOT / "foodb_2020_04_07_csv"
    columns = ["id", "source_id", "source_type", "orig_source_id", "orig_source_name",
               "food_id", "orig_food_id", "orig_food_common_name", "citation",
               "orig_method", "orig_unit_expression", "orig_content", "orig_min", "orig_max", "orig_unit"]
    parts = []
    for chunk in pd.read_csv(raw / "Content.csv", usecols=columns, dtype=str, chunksize=200000):
        quantitative = chunk[["orig_content", "orig_min", "orig_max"]].notna().any(axis=1)
        parts.append(chunk[quantitative].fillna(""))
    content = pd.concat(parts, ignore_index=True)
    content["measurement_id"] = content.id.map(lambda x: stable_id("measure", "foodb", x))
    content["component_observation_id"] = [stable_id("compobs", "foodb", t, i)
                                            for t, i in zip(content.source_type, content.source_id)]
    write_csv(content, output / "content_identity_metadata.csv.gz")
    known = pd.read_csv(raw / "Compound.csv", usecols=["id"], dtype=str)
    missing = content[content.source_type.eq("Compound") & ~content.source_id.isin(known.id)]
    external = pd.read_csv(raw / "CompoundExternalDescriptor.csv", dtype=str).fillna("")
    synonyms = pd.read_csv(raw / "CompoundSynonym.csv", dtype=str).fillna("")
    chebi = parse_obo(ROOT / "data/raw/ontology/chebi_lite.obo.gz")
    rows = []
    for source_id, group in missing.groupby("source_id", sort=True):
        ext = external[external.compound_id.eq(source_id)]
        syn = synonyms[synonyms.source_type.eq("Compound") & synonyms.source_id.eq(source_id)]
        names = group.loc[group.orig_source_name.ne(""), "orig_source_name"].value_counts()
        ids = sorted(set(ext.loc[ext.external_id.str.startswith("CHEBI:"), "external_id"]))
        preferred = [chebi[x]["name"] for x in ids if x in chebi]
        usda = group[group.citation.eq("USDA")]
        rows.append({
            "source_id": source_id, "component_observation_id": group.component_observation_id.iloc[0],
            "quantitative_content_records": len(group),
            "source_food_ids": group[["citation", "orig_food_id", "food_id"]].drop_duplicates().shape[0],
            "original_names": " | ".join(names.index), "most_frequent_original_name": names.index[0] if len(names) else "",
            "chebi_ids": ";".join(ids), "chebi_preferred_names": " | ".join(preferred),
            "external_ids": ";".join(sorted(set(ext.external_id))),
            "curated_synonyms": " | ".join(sorted(set(syn.loc[syn.synonym_source.isin(["ChEBI", "HMDB", "manual"]), "synonym"]))),
            "usda_original_component_ids": ";".join(sorted(set(usda.orig_source_id) - {""})),
            "citations": " | ".join(group.citation.value_counts().index),
            "example_content_id": group.id.iloc[0],
            "example_evidence_url": "https://foodb.ca/contents/" + group.id.iloc[0],
        })
    result = pd.DataFrame(rows).sort_values("quantitative_content_records", ascending=False)
    write_csv(result, output / "orphan_compound_inventory.csv")
    print(result[["source_id", "quantitative_content_records", "most_frequent_original_name",
                  "chebi_preferred_names", "usda_original_component_ids"]].head(55).to_string(index=False))
    print(f"Quantitative content records: {len(content):,}; absent Compound IDs: {len(result):,}")


if __name__ == "__main__":
    main()
