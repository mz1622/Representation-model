#!/usr/bin/env python3
"""Snapshot primary evidence for the validation/MEXT audit, without ML admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from download_scientific_expansion_v2 import Links


DIRECT = {
    "mext/faq.html": "https://fooddb.mext.go.jp/help.html",
    "mext/usage.html": "https://www.mext.go.jp/a_menu/syokuhinseibun/",
    "mext/english_2015_main.xlsx": "https://www.mext.go.jp/component/english/__icsFiles/afieldfile/2017/12/25/1374049_1r12_1.xlsx",
    "mext/english_2015_species.xlsx": "https://www.mext.go.jp/component/english/__icsFiles/afieldfile/2017/08/02/1385123_Table18.xlsx",
    "mext/analytical_manual_2020.pdf": "https://www.mext.go.jp/content/20220222-mext_kagsei-index_100.pdf",
    "afcd/ST19036_final_report.pdf": "https://edge.sitecorecloud.io/hortinnovat2fab-hortinnovat9661-production6d5f-0e78/media/Sub-pages/Information-hub/Project-reports/S/st19036-final-report-complete.pdf",
    "afcd/mince_2014.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4073145/fullTextXML",
    "afcd/beef_2007_repository.html": "https://ro.uow.edu.au/hbspapers/46/",
}
PAGES = {
    "uk_fruit_2013": "https://www.gov.uk/government/publications/nutrient-analysis-of-fruit-and-vegetables",
    "uk_fruit_2015": "https://www.gov.uk/government/publications/nutrient-analysis-of-fruits-and-vegetables",
    "uk_fish_2013": "https://www.gov.uk/government/publications/nutrient-analysis-of-fish",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.root / "data/raw/validation_mext_evidence_2026_09_06"
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": [], "failures": []}

    def fetch(url, relative):
        path = base / relative
        known = next((item for item in manifest["files"] if item["path"] == relative), None)
        if path.exists():
            payload = path.read_bytes()
            if known is None or hashlib.sha256(payload).hexdigest() != known["sha256"]:
                raise ValueError(f"Immutable evidence changed: {path}")
            if not payload:
                raise ValueError(f"Empty response is not evidence: {path}")
            return payload
        request = urllib.request.Request(url, headers={"User-Agent": "FoodCompositionResearch/2.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            resolved_url = response.url
            content_type = response.headers.get("Content-Type", "")
        if not payload:
            raise ValueError(f"Empty response is not evidence: {url}")
        suffix = path.suffix
        if suffix == ".pdf" and not payload.startswith(b"%PDF"):
            raise ValueError(f"Expected PDF, received {content_type}: {url}")
        if suffix == ".xlsx" and not payload.startswith(b"PK"):
            raise ValueError(f"Expected XLSX, received {content_type}: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest["files"].append({"path": relative, "url": url, "resolved_url": resolved_url,
                                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                                  "content_type": content_type, "retrieved_utc": datetime.now(timezone.utc).isoformat()})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Downloaded {relative}: {len(payload):,} bytes", flush=True)
        return payload

    def attempt(url, relative):
        try:
            return fetch(url, relative)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failure = {"url": url, "path": relative, "error": str(exc), "attempted_utc": datetime.now(timezone.utc).isoformat()}
            manifest["failures"].append(failure)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"EVIDENCE NOT DOWNLOADED: {failure}", flush=True)
            return None

    for relative, url in DIRECT.items():
        attempt(url, relative)
    for source, url in PAGES.items():
        payload = attempt(url, f"{source}/official_page.html")
        if payload is None:
            continue
        links = Links()
        links.feed(payload.decode("utf-8"))
        for href, label in dict.fromkeys(links.links):
            absolute = urllib.parse.urljoin(url, href)
            parsed = urllib.parse.urlparse(absolute)
            suffix = Path(parsed.path).suffix.lower()
            if parsed.netloc != "assets.publishing.service.gov.uk" or suffix not in {".xlsx", ".xls", ".pdf"}:
                continue
            if "sampling" in (absolute + label).casefold():
                continue
            attempt(absolute, f"{source}/{urllib.parse.unquote(Path(parsed.path).name)}")
    print(f"Evidence manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
