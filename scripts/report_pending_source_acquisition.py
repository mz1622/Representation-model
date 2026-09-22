#!/usr/bin/env python3
"""Record lawful acquisition status for registered, non-integrated sources.

This script never scrapes a search interface, bypasses a login, or accepts a
licence on a user's behalf. It records whether an official artifact is already
available locally and the next defensible action for every remaining source.

Colab:
    python scripts/report_pending_source_acquisition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import sha256_file, write_csv  # noqa: E402


DEFAULT_AUDIT = ROOT / "data" / "processed" / "global_frozen_prediction_panel_food_audit_v3"
DEFAULT_REGISTRY = ROOT / "data" / "processed" / "global_food_metabolome_axis_atlas_v1" / "source_registry.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "global_frozen_prediction_panel_food_audit_v3" / "source_acquisition"

TAIWAN_ARTIFACT = ROOT / "data" / "raw" / "source_acquisition_2026_09_13" / "taiwan_tfda" / "tfda_nutrition_export_20.zip"
LOCAL_ARTIFACT_OVERRIDES = {
    "foodfiles_2024": ROOT / "data" / "raw" / "expansion_2026_09_06" / "foodfiles" / "00_foodfiles-2024-v1.msi",
    "ifct_2017": ROOT / "data" / "raw" / "expansion_2026_09_06" / "ifct" / "IFCT2017_official.pdf",
    "ethiopian_fct_2025": ROOT / "data" / "raw" / "global_fcdb_inventory_2026_09_10" / "downloads" / "ethiopian_fct_2025_user_guide.pdf",
    "kenya_fct_2018": ROOT / "data" / "raw" / "global_fcdb_inventory_2026_09_10" / "downloads" / "kenya_fct_2018.pdf",
    "malawi_fct_2019": ROOT / "data" / "raw" / "global_fcdb_inventory_2026_09_10" / "downloads" / "malawi_fct_2019.pdf",
}


def _action(access_status: str, local_exists: bool, source_key: str) -> str:
    if source_key == "taiwan_tfda":
        return "official CSV archive acquired; perform a source-native schema audit before numerical integration"
    if source_key == "ethiopian_fct_2025":
        return "only the official user guide is local; obtain the release's structured value tables before numerical integration"
    if local_exists and "pdf_only" in access_status:
        return "retain the official PDF; obtain a structured release or complete a licensed, auditable table extraction before integration"
    if local_exists and "terms_review_required" in access_status:
        return "review current licence and redistribution conditions before parsing or using values"
    if local_exists:
        return "complete a source-native adapter and schema audit; do not infer missing fields"
    if "automated_mirror_blocked" in access_status:
        return "download manually from the official open-data page; automated access returned HTTP 403 and must not be bypassed"
    if "requires_acceptance" in access_status or "permission_required" in access_status:
        return "obtain the required terms acceptance or permission outside this pipeline"
    if "restricted" in access_status or "paid" in access_status:
        return "obtain authorised access; do not automate collection"
    if "bulk" in access_status or "endpoint" in access_status or "searchable" in access_status:
        return "verify an official bulk-data route and its licence; do not reconstruct a dataset from search-result pages"
    return "confirm the official release route, version and terms before any numerical integration"


def build_report(audit_dir: Path, registry_path: Path, output_dir: Path) -> Path:
    ledger = pd.read_csv(audit_dir / "global_source_ingestion_ledger.csv", keep_default_na=False)
    registry = pd.read_csv(registry_path, keep_default_na=False)
    required = {"source_key", "ingestion_status", "access_status"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"Ingestion ledger is missing: {sorted(missing)}")
    rows = ledger.loc[~ledger.ingestion_status.str.startswith("integrated")].copy()
    rows = rows.merge(
        registry.loc[:, ["source_key", "official_url", "download_url", "license_or_terms", "resolved_local_raw_path"]],
        how="left",
        on="source_key",
        validate="one_to_one",
    )
    local_paths = []
    hashes = []
    for row in rows.itertuples(index=False):
        path = Path(str(row.resolved_local_raw_path)) if str(row.resolved_local_raw_path) else None
        if path is not None and not path.is_absolute():
            path = ROOT / path
        path = LOCAL_ARTIFACT_OVERRIDES.get(row.source_key, path)
        if row.source_key == "taiwan_tfda":
            path = TAIWAN_ARTIFACT
        local_paths.append(str(path.relative_to(ROOT)) if path and path.exists() else "")
        hashes.append(sha256_file(path) if path and path.is_file() and path.exists() else "")
    rows["available_local_artifact"] = pd.Series(local_paths, index=rows.index).ne("")
    rows["local_artifact_path"] = local_paths
    rows["local_artifact_sha256"] = hashes
    rows["next_lawful_action"] = [
        _action(row.access_status, exists, row.source_key)
        for row, exists in zip(rows.itertuples(index=False), rows.available_local_artifact, strict=True)
    ]
    rows["acquisition_note"] = ""
    taiwan = rows.source_key.eq("taiwan_tfda")
    rows.loc[taiwan, "acquisition_note"] = (
        "Downloaded from the TFDA official CSV export on 2026-09-13; 226,720 source rows; no numerical integration in this report."
    )
    columns = [
        "source_key", "source_name", "region", "source_version", "access_status", "ingestion_status",
        "ingestion_reason", "available_local_artifact", "local_artifact_path", "local_artifact_sha256",
        "official_url", "download_url", "license_or_terms", "acquisition_note", "next_lawful_action",
    ]
    result = rows.loc[:, columns].sort_values(["region", "source_key"], kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pending_source_acquisition_status.csv"
    write_csv(result, path)
    (output_dir / "README.md").write_text(
        "# Pending Source Acquisition Status\n\n"
        "This ledger lists every registered source not numerically integrated in the selected audit snapshot. "
        "A local file is not treated as permission to redistribute or train on it. The ledger records only lawful next actions; it never substitutes a browser scrape, a login bypass or unreviewed PDF extraction for an official data release.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_report(args.audit_dir, args.registry, args.output_dir)
    print(f"Wrote pending-source acquisition status: {path}")


if __name__ == "__main__":
    main()
