"""Small dependency-free readers for the ontology snapshots used in the build."""

from __future__ import annotations

import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, TextIO

from .util import normalize_text


def parse_infoods_tagnames(directory: Path) -> dict[str, dict[str, str]]:
    """Read the official FAO/INFOODS lists available from the 2022 web page.

    A field that merely resembles a Tagname is not accepted. The identifier
    must occur in one of the archived FAO lists downloaded with the build.
    """
    if not directory.exists():
        raise FileNotFoundError(f"INFOODS Tagname snapshot is missing: {directory}")
    records: dict[str, dict[str, str]] = {}
    for path in sorted(directory.glob("PART*.TXT")):
        text = path.read_text(encoding="latin-1", errors="replace")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            match = re.match(r"^<([^<>]+)>\s*(.*)$", line)
            if not match:
                continue
            tag = match.group(1).strip().upper()
            if not re.fullmatch(r"[A-Z0-9+\-]+", tag):
                continue
            fragments = [match.group(2).strip()] if match.group(2).strip() else []
            for continuation in lines[index + 1:]:
                cleaned = continuation.strip()
                if re.match(r"^<[^<>]+>", cleaned):
                    break
                if re.match(r"^(Unit|Units|Synonym|Synonyms|Comments?|Tables?|Keywords?|Notes?|Examples?):", cleaned, flags=re.I):
                    break
                if cleaned:
                    fragments.append(cleaned)
            description = " ".join(fragments)
            records.setdefault(tag, {
                "tagname": tag,
                "preferred_name": description,
                "definition": description,
                "recommended_unit": "",
                "authority_source_file": path.name,
            })

    for filename in ("TAGREV__1_.xls", "Tagname_new_April_2010-web__2_.xls"):
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Required INFOODS Tagname supplement is missing: {path}")
        try:
            import pandas as pd
            frame = pd.read_excel(path, sheet_name="TAGNAMES", keep_default_na=False, na_values=[""])
        except ImportError as exc:
            raise ImportError(
                "Reading the official INFOODS .xls supplements requires xlrd; "
                "install requirements-colab.txt before building."
            ) from exc
        required = {"TAGNAME", "Description", "Recommended units"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"INFOODS supplement {path} is missing columns: {sorted(missing)}")
        for _, row in frame.iterrows():
            tag = str(row.get("TAGNAME", "")).strip().upper()
            if not tag or tag == "NAN":
                continue
            short = str(row.get("Short description", "")).strip()
            description = str(row.get("Description", "")).strip()
            preferred = description if description and description != "nan" else short
            records[tag] = {
                "tagname": tag,
                "preferred_name": preferred,
                "definition": description if description != "nan" else "",
                "recommended_unit": str(row.get("Recommended units", "")).strip(),
                "authority_source_file": path.name,
            }
    if not records:
        raise ValueError(f"No INFOODS Tagnames could be parsed from {directory}")
    return records


def parse_obo(path: Path) -> dict[str, dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    terms: dict[str, dict[str, Any]] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        current: dict[str, Any] | None = None
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                if current and current.get("id"):
                    terms[current["id"]] = current
                current = {"parents": [], "relationships": [], "synonyms": [], "xrefs": []}
                continue
            if line.startswith("["):
                if current and current.get("id"):
                    terms[current["id"]] = current
                current = None
                continue
            if current is None or not line or line.startswith("!"):
                continue
            key, separator, value = line.partition(": ")
            if not separator:
                continue
            if key == "id":
                current["id"] = value
            elif key == "name":
                current["name"] = value
            elif key == "def":
                current["definition"] = value
            elif key == "is_a":
                current["parents"].append(value.split(" ! ", 1)[0])
            elif key == "relationship":
                current["relationships"].append(value)
            elif key == "synonym":
                match = re.match(r'^"(.*?)"', value)
                if match:
                    current["synonyms"].append(match.group(1))
            elif key == "xref":
                current["xrefs"].append(value)
        if current and current.get("id"):
            terms[current["id"]] = current
    return terms


def ontology_label_index(terms: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for term_id, term in terms.items():
        labels = [term.get("name", ""), *term.get("synonyms", [])]
        for label in labels:
            normalized = normalize_text(label)
            if normalized:
                index.setdefault(normalized, []).append(term_id)
    return index


def ancestor_names(term_id: str, terms: dict[str, dict[str, Any]], max_depth: int = 5) -> list[str]:
    names: list[str] = []
    frontier = [(term_id, 0)]
    seen = {term_id}
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for parent in terms.get(current, {}).get("parents", []):
            if parent in seen:
                continue
            seen.add(parent)
            parent_name = terms.get(parent, {}).get("name")
            if parent_name:
                names.append(parent_name)
            frontier.append((parent, depth + 1))
    return names


def parse_foodon(path: Path) -> dict[str, dict[str, Any]]:
    rdf_about = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
    rdf_resource = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
    owl_class = "{http://www.w3.org/2002/07/owl#}Class"
    rdfs_label = "{http://www.w3.org/2000/01/rdf-schema#}label"
    rdfs_subclass = "{http://www.w3.org/2000/01/rdf-schema#}subClassOf"
    terms: dict[str, dict[str, Any]] = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != owl_class:
            continue
        iri = element.attrib.get(rdf_about, "")
        if "FOODON_" not in iri:
            element.clear()
            continue
        term_id = "FOODON:" + iri.rsplit("FOODON_", 1)[1]
        labels = [child.text.strip() for child in element if child.tag == rdfs_label and child.text]
        parents = []
        for child in element:
            if child.tag == rdfs_subclass and rdf_resource in child.attrib and "FOODON_" in child.attrib[rdf_resource]:
                parents.append("FOODON:" + child.attrib[rdf_resource].rsplit("FOODON_", 1)[1])
        terms[term_id] = {"id": term_id, "name": labels[0] if labels else "", "parents": parents}
        element.clear()
    return terms


def normalize_ontology_id(value: Any, prefix: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    match = re.search(rf"{re.escape(prefix)}[_:](\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"{prefix.upper()}:{match.group(1)}"
    return text if text.upper().startswith(prefix.upper() + ":") else ""
