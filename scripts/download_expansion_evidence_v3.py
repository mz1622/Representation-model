#!/usr/bin/env python3
"""Archive public BLS/Fineli expansion evidence; never admit it as ML data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from zipfile import ZipFile
from io import BytesIO

from download_scientific_expansion_v2 import Links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.root / "data/raw/expansion_evidence_2026_09_07"
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": [], "failures": []}

    def fetch(url, relative):
        path = base / relative
        known = next((item for item in manifest["files"] if item["path"] == relative), None)
        if path.exists():
            payload = path.read_bytes()
            if known is None or hashlib.sha256(payload).hexdigest() != known["sha256"]:
                raise ValueError(f"Changed or unregistered immutable snapshot: {path}")
            return payload
        request = urllib.request.Request(url, headers={"User-Agent": "FoodCompositionResearch/3.0 (official public data)"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
            resolved, content_type = response.url, response.headers.get("Content-Type", "")
        if not payload:
            raise ValueError(f"Empty response: {url}")
        if path.suffix == ".zip" and not payload.startswith(b"PK"):
            raise ValueError(f"Not a ZIP: {url} ({content_type})")
        if path.suffix == ".pdf" and not payload.startswith(b"%PDF"):
            raise ValueError(f"Not a PDF: {url} ({content_type})")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest["files"].append({"path": relative, "url": url, "resolved_url": resolved,
                                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                                  "content_type": content_type, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                  "status": "raw_evidence_only_no_training_admission"})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Downloaded {relative}: {len(payload):,} bytes", flush=True)
        return payload

    for source, page in (("bls", "https://blsdb.de/download"), ("fineli", "https://fineli.fi/fineli/fi/avoin-data")):
        try:
            payload = fetch(page, f"{source}/download_page.html")
            links = Links()
            links.feed(payload.decode("utf-8"))
            selected = []
            for href, label in dict.fromkeys(links.links):
                absolute = urllib.parse.urljoin(page, href)
                parsed = urllib.parse.urlparse(absolute)
                suffix = Path(parsed.path).suffix.lower()
                if parsed.netloc != urllib.parse.urlparse(page).netloc:
                    continue
                if source == "bls" and suffix == ".zip":
                    selected.append(absolute)
                if source == "fineli" and suffix == ".zip" and "74" in label:
                    selected.append(absolute)
            if not selected:
                raise ValueError(f"No expected official archive link found: {page}")
            for index, url in enumerate(selected):
                archive = fetch(url, f"{source}/archive_{index:02d}.zip")
                with ZipFile(BytesIO(archive)) as zipped:
                    inventory = [{"filename": item.filename, "bytes": item.file_size, "crc32": item.CRC}
                                 for item in zipped.infolist()]
                    (base / source / f"archive_{index:02d}_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
                    print(json.dumps(inventory), flush=True)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failure = {"source": source, "url": page, "error": str(exc), "attempted_utc": datetime.now(timezone.utc).isoformat()}
            manifest["failures"].append(failure)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"NOT DOWNLOADED: {failure}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
