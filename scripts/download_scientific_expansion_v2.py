#!/usr/bin/env python3
"""Download immutable official-source snapshots; this does not admit ML values."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, data):
        if self.href:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, " ".join(self.label)))
            self.href = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sources", nargs="+", default=["norway", "afcd", "fineli", "mext", "foodfiles", "ifct"])
    args = parser.parse_args()
    base = args.root / "data/raw/expansion_2026_09_06"
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": [], "failures": []}

    def fetch(url, relative):
        path = base / relative
        existing = next((x for x in manifest["files"] if x["path"] == relative), None)
        if path.exists():
            payload = path.read_bytes()
            if existing is None or existing["sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"Unregistered or changed immutable snapshot: {path}")
            return payload
        request = urllib.request.Request(url, headers={"User-Agent": "FoodCompositionResearch/2.0 (official public datasets)"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
            final_url = response.url
            content_type = response.headers.get("Content-Type", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest["files"].append({
            "path": relative, "url": url, "resolved_url": final_url, "content_type": content_type,
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "status": "raw_snapshot_only_requires_value_and_licence_review",
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"Downloaded {relative}: {len(payload):,} bytes", flush=True)
        return payload

    pages = {
        "afcd": "https://www.foodstandards.gov.au/science-data/food-nutrient-databases/afcd/data-files",
        "fineli": "https://fineli.fi/fineli/fi/avoin-data",
        "mext": "https://www.mext.go.jp/a_menu/syokuhinseibun/mext_00001.html",
        "foodfiles": "https://www.foodcomposition.co.nz/foodfiles",
    }
    for source in args.sources:
        try:
            if source == "norway":
                for endpoint in ["foods", "nutrients", "sources", "food-groups"]:
                    payload = fetch(f"https://www.matvaretabellen.no/api/en/{endpoint}.json", f"norway/{endpoint}.json")
                    json.loads(payload)
                fetch("https://www.matvaretabellen.no/api/langual.json", "norway/langual.json")
                fetch("https://www.matvaretabellen.no/en/api/", "norway/documentation.html")
                continue
            if source == "ifct":
                fetch("https://www.nin.res.in/ebooks/IFCT2017_16122024.pdf", "ifct/IFCT2017_official.pdf")
                continue
            url = pages[source]
            payload = fetch(url, f"{source}/download_page.html")
            links = Links()
            links.feed(payload.decode("utf-8"))
            selected = []
            for href, label in links.links:
                absolute = urllib.parse.urljoin(url, href)
                ext = Path(urllib.parse.urlparse(absolute).path).suffix.lower()
                if source in {"afcd", "mext"} and ext in {".xlsx", ".xls", ".pdf"}:
                    selected.append((absolute, label))
                elif source == "fineli" and (ext == ".pdf" or ("74" in label and ext == ".zip")):
                    selected.append((absolute, label))
                elif source == "foodfiles" and "2024" in (absolute + label) and ext in {".msi", ".zip", ".pdf", ".xlsx"}:
                    selected.append((absolute, label))
            if not selected:
                raise ValueError(f"No official downloads discovered for {source}; inspect the cached page")
            for index, (absolute, label) in enumerate(dict.fromkeys(selected)):
                basename = urllib.parse.unquote(Path(urllib.parse.urlparse(absolute).path).name)
                fetch(absolute, f"{source}/{index:02d}_{basename}")
            if source == "foodfiles":
                fetch("https://www.foodcomposition.co.nz/terms/", "foodfiles/terms_of_use.html")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failure = {"source": source, "error": str(exc), "attempted_utc": datetime.now(timezone.utc).isoformat()}
            manifest["failures"].append(failure)
            print(f"SOURCE NOT COMPLETED: {source}: {exc}", flush=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
