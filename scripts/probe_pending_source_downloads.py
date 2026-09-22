#!/usr/bin/env python3
"""Probe official download routes for sources not in the numerical audit.

The probe makes one small, anonymous HTTP request to each source's registered
official download URL. It neither follows an authentication flow nor accepts
terms, completes forms, solves challenges, or reconstructs a database from a
search interface. A successful probe is not numerical integration: a source
needs a locally saved artifact, licence review, schema audit, and source-native
adapter before it can enter a food-value dataset.

Colab:
    python scripts/probe_pending_source_downloads.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import sha256_file, write_csv  # noqa: E402


DEFAULT_REGISTRY = ROOT / "data/raw/global_fcdb_inventory_2026_09_10/source_registry.csv"
DEFAULT_AUDIT = ROOT / "data/processed/global_frozen_prediction_panel_food_audit_v5"
DEFAULT_OUTPUT = ROOT / "reports/global_frozen_prediction_panel_food_audit_v5/source_acquisition"
USER_AGENT = "FoodCompositionAtlas/1.0 (+research provenance audit; no automated form submission)"
DATA_CONTENT_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "application/json",
    "text/plain",
}
LOCAL_ARTIFACT_OVERRIDES = {
    "taiwan_tfda": ROOT / "data/raw/source_acquisition_2026_09_13/taiwan_tfda/tfda_nutrition_export_20.zip",
}


@dataclass(frozen=True)
class ProbeResult:
    source_key: str
    official_url: str
    requested_url: str
    final_url: str
    http_status: str
    content_type: str
    content_length: str
    response_kind: str
    result: str
    detail: str
    probed_at_utc: str


def _local_artifact(root: Path, source: pd.Series) -> tuple[bool, str, str]:
    override = LOCAL_ARTIFACT_OVERRIDES.get(str(source.source_key))
    if override is not None and override.exists():
        return True, str(override.relative_to(root)), sha256_file(override)
    raw = str(source.get("resolved_local_raw_path", "") or source.get("local_raw_path", "")).strip()
    if not raw:
        return False, "", ""
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return False, "", ""
    if path.is_file():
        return True, str(path.relative_to(root)), sha256_file(path)
    return True, str(path.relative_to(root)), "directory_artifact"


def _response_kind(content_type: str, final_url: str) -> str:
    normalized = content_type.split(";", 1)[0].strip().casefold()
    suffix = Path(final_url.split("?", 1)[0]).suffix.casefold()
    if normalized in DATA_CONTENT_TYPES or suffix in {".csv", ".tsv", ".json", ".xlsx", ".xls", ".zip"}:
        return "machine_readable_candidate"
    if normalized == "application/pdf" or suffix == ".pdf":
        return "pdf_only"
    if "html" in normalized:
        return "portal_or_landing_page"
    return "unknown_response"


def probe(source: pd.Series, timeout: int) -> ProbeResult:
    requested = str(source.download_url or source.official_url).strip()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if not requested:
        return ProbeResult(str(source.source_key), str(source.official_url), "", "", "", "", "", "", "no_official_download_url", "No download URL is registered.", now)
    request = Request(
        requested,
        headers={
            "User-Agent": USER_AGENT,
            # Reading a small initial range distinguishes a downloadable file
            # from a landing page without collecting an entire large artifact.
            "Range": "bytes=0-4095",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read(4096)
            content_type = response.headers.get_content_type()
            return ProbeResult(
                str(source.source_key), str(source.official_url), requested, response.geturl(),
                str(getattr(response, "status", 200)), content_type,
                str(response.headers.get("Content-Length", "")),
                _response_kind(content_type, response.geturl()), "reachable", "Official route responded without authentication.", now,
            )
    except HTTPError as error:
        return ProbeResult(
            str(source.source_key), str(source.official_url), requested, error.geturl() or requested,
            str(error.code), error.headers.get_content_type() if error.headers else "", "",
            "http_error", "not_downloaded", f"Official route returned HTTP {error.code}; no bypass attempted.", now,
        )
    except URLError as error:
        return ProbeResult(
            str(source.source_key), str(source.official_url), requested, "", "", "", "", "network_error",
            "not_downloaded", f"Official route could not be reached: {error.reason}", now,
        )
    except TimeoutError:
        return ProbeResult(
            str(source.source_key), str(source.official_url), requested, "", "", "", "", "timeout",
            "not_downloaded", f"Official route did not respond within {timeout} seconds.", now,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    registry = pd.read_csv(args.registry, keep_default_na=False)
    ledger = pd.read_csv(args.audit_dir / "global_source_ingestion_ledger.csv", keep_default_na=False)
    pending_keys = set(ledger.loc[~ledger.ingestion_status.str.startswith("integrated"), "source_key"])
    pending = registry[registry.source_key.isin(pending_keys)].copy()
    source_rows = [source for _, source in pending.iterrows()]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda source: probe(source, args.timeout), source_rows))
    result_frame = pd.DataFrame(asdict(item) for item in results)

    local_rows = []
    for _, source in pending.iterrows():
        exists, path, digest = _local_artifact(ROOT, source)
        local_rows.append({"source_key": source.source_key, "local_artifact_exists": exists, "local_artifact_path": path, "local_artifact_sha256": digest})
    result_frame = result_frame.merge(pd.DataFrame(local_rows), on="source_key", validate="one_to_one")
    result_frame = result_frame.merge(
        pending.loc[:, ["source_key", "name", "access_status", "license_or_terms", "reported_foods"]],
        on="source_key", validate="one_to_one",
    )
    result_frame["numerical_dataset_decision"] = "not_integrated"
    local_machine_readable = result_frame.local_artifact_path.str.lower().str.endswith(
        (".csv", ".tsv", ".json", ".xlsx", ".xls", ".zip")
    )
    result_frame.loc[
        result_frame.local_artifact_exists & local_machine_readable,
        "numerical_dataset_decision",
    ] = "artifact_present_pending_schema_and_licence_audit"
    local_pdf = result_frame.local_artifact_path.str.lower().str.endswith(".pdf")
    result_frame.loc[
        result_frame.local_artifact_exists & local_pdf,
        "numerical_dataset_decision",
    ] = "official_pdf_retained_not_numerically_integrated"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "official_download_probe.csv"
    write_csv(result_frame.sort_values("source_key", kind="stable"), output)
    (args.output_dir / "OFFICIAL_DOWNLOAD_PROBE.md").write_text(
        "# Official Download Probe\n\n"
        "This report records one small anonymous request to every pending source's official registered URL. "
        "It is not a scraper and performs no login, form submission, licence acceptance, CAPTCHA handling, or access-control bypass. "
        "Only a locally saved, legally usable, source-native machine-readable artifact may proceed to schema review. All other sources remain outside the numerical dataset.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(result_frame)} official source probes: {output}")


if __name__ == "__main__":
    main()
