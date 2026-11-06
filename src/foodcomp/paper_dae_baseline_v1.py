"""Gjorshoska et al. DAE and a strict crosswalk to the current axis panel.

The neural architecture mirrors the authors' public notebook.  The crosswalk
is deliberately narrower than a fuzzy nutrient-name match: calculated
carbohydrates, activity equivalents, method-specific fibre expressions and
chemically different forms are not silently merged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from foodcomp.final_prediction_axes import DISPLAY_CANONICAL, EXACT_SYNONYM_CANONICAL
from foodcomp.prediction_axis_design import _normalise_axis_name


VERSION = "gjorshoska_2022_paper_dae_external_v1"


MANUAL_EXACT_ALIASES = {
    1003: "Protein, total",
    1004: "Fat, total",
    1011: "Glucose",
    1063: "Sugars, total",
    1079: "Dietary fibre, total",
    1084: "Insoluble dietary fibers",
    1087: "Calcium",
    1089: "Iron",
    1090: "Magnesium",
    1091: "Phosphorus",
    1092: "Potassium",
    1093: "Sodium",
    1094: "Sulfur",
    1095: "Zinc",
    1098: "Copper",
    1101: "Manganese",
    1102: "Molybdenum",
    1112: "Cholecalciferol (vitamin D3)",
    1162: "Ascorbic acid (vitamin C)",
    1165: "Thiamin (vitamin B1)",
    1166: "Riboflavin (vitamin B2)",
    1170: "Pantothenic acid (vitamin B5)",
    1175: "Vitamin B6 - Pyridoxin",
    1178: "Cobalamin (vitamin B12)",
    1183: "Menaquinone-4",
    1268: "Octadecenoic acid (18:1; cis/trans unspecified)",
    1269: "Octadecadienoic acid (18:2; isomer unspecified)",
    1270: "Octadecatrienoic acid (18:3; isomer unspecified)",
    1271: "Eicosatetraenoic acid (20:4; isomer unspecified)",
    1279: "Docosenoic acid (22:1; cis/trans unspecified)",
    1325: "Eicosatrienoic acid (20:3; isomer unspecified)",
}


UNIT_TO_PAPER = {"G": 1.0, "MG": 1_000.0, "UG": 1_000_000.0}


@dataclass(frozen=True)
class PaperAxisLink:
    paper_column: int
    paper_nutrient_id: int
    paper_name: str
    paper_unit: str
    canonical_name: str
    axis_index: int
    loss_eligible: bool
    grams_to_paper: float
    mapping_basis: str


class PaperDenoisingAutoencoder(nn.Module):
    """The exact layer proportions and activations from the public notebook."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        if dim < 5:
            raise ValueError("DAE input dimension is too small")
        d70, d50, d20 = int(dim * 0.7), int(dim * 0.5), int(dim * 0.2)
        self.dim = int(dim)
        self.drop_out = nn.Dropout(p=dropout)
        self.encoder = nn.Sequential(
            nn.Linear(dim, d70), nn.Tanh(),
            nn.Linear(d70, d50), nn.Tanh(),
            nn.Linear(d50, d20),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d20, d50), nn.Tanh(),
            nn.Linear(d50, d70), nn.Tanh(),
            nn.Linear(d70, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(-1, self.dim)
        return self.decoder(self.encoder(self.drop_out(value)))


def _candidate_name(nutrient_id: int, raw_name: str) -> tuple[str, str]:
    if nutrient_id in MANUAL_EXACT_ALIASES:
        return MANUAL_EXACT_ALIASES[nutrient_id], "reviewed_exact_alias"
    normalized = _normalise_axis_name(raw_name)
    if normalized in EXACT_SYNONYM_CANONICAL:
        return EXACT_SYNONYM_CANONICAL[normalized], "panel_exact_synonym"
    if normalized in DISPLAY_CANONICAL:
        return DISPLAY_CANONICAL[normalized], "panel_chain_notation"
    return raw_name, "exact_canonical_name"


def build_paper_axis_crosswalk(
    paper_columns: Iterable[str], nutrient: pd.DataFrame, axes: pd.DataFrame
) -> tuple[list[PaperAxisLink], pd.DataFrame]:
    """Map only chemically equivalent paper columns into the current panel."""
    required_nutrient = {"id", "name", "unit_name"}
    required_axes = {"axis_index", "canonical_name", "loss_eligible"}
    if not required_nutrient.issubset(nutrient.columns):
        raise ValueError("paper nutrient table lacks required columns")
    if not required_axes.issubset(axes.columns):
        raise ValueError("axis table lacks required columns")
    nutrient_by_id = nutrient.assign(id=nutrient.id.astype(int)).set_index("id")
    axis_by_name = axes.set_index("canonical_name")
    links: list[PaperAxisLink] = []
    audit_rows = []
    for paper_column, raw_id in enumerate(paper_columns):
        nutrient_id = int(raw_id)
        if nutrient_id not in nutrient_by_id.index:
            raise ValueError(f"paper nutrient id {nutrient_id} is absent")
        source = nutrient_by_id.loc[nutrient_id]
        canonical, basis = _candidate_name(nutrient_id, str(source["name"]))
        unit = str(source["unit_name"]).upper()
        accepted = canonical in axis_by_name.index and unit in UNIT_TO_PAPER
        reason = "accepted_exact_identity" if accepted else (
            "unsupported_unit" if canonical in axis_by_name.index else
            "no_strict_current_axis_equivalent"
        )
        row = {
            "paper_column": paper_column,
            "paper_nutrient_id": nutrient_id,
            "paper_name": str(source["name"]),
            "paper_unit": unit,
            "candidate_canonical_name": canonical,
            "mapping_basis": basis,
            "status": reason,
            "axis_index": np.nan,
            "loss_eligible": False,
            "grams_to_paper": np.nan,
        }
        if accepted:
            target = axis_by_name.loc[canonical]
            link = PaperAxisLink(
                paper_column=paper_column,
                paper_nutrient_id=nutrient_id,
                paper_name=str(source["name"]),
                paper_unit=unit,
                canonical_name=canonical,
                axis_index=int(target.axis_index),
                loss_eligible=bool(target.loss_eligible),
                grams_to_paper=UNIT_TO_PAPER[unit],
                mapping_basis=basis,
            )
            links.append(link)
            row.update(
                axis_index=link.axis_index,
                loss_eligible=link.loss_eligible,
                grams_to_paper=link.grams_to_paper,
            )
        audit_rows.append(row)
    axes_seen = [link.axis_index for link in links]
    if len(axes_seen) != len(set(axes_seen)):
        raise ValueError("paper crosswalk contains duplicate current axes")
    return links, pd.DataFrame(audit_rows)


def paper_external_features(
    data,
    rows: np.ndarray,
    target_axes: np.ndarray,
    policy,
    links: Iterable[PaperAxisLink],
    paper_means: np.ndarray,
) -> np.ndarray:
    """Construct paper-unit inputs under the current target-specific mask."""
    rows = np.asarray(rows, dtype=np.int64)
    target_axes = np.asarray(target_axes, dtype=np.int64)
    means = np.asarray(paper_means, dtype=np.float64)
    if rows.shape != target_axes.shape or rows.ndim != 1:
        raise ValueError("rows and target axes must be aligned vectors")
    result = np.broadcast_to(means, (len(rows), len(means))).copy()
    hidden = policy.mask_matrix(target_axes)
    for link in links:
        visible = data.observed[rows, link.axis_index] & ~hidden[:, link.axis_index]
        result[visible, link.paper_column] = (
            data.raw[rows[visible], link.axis_index] * link.grams_to_paper
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite paper DAE input")
    return result


def target_link_by_axis(links: Iterable[PaperAxisLink]) -> dict[int, PaperAxisLink]:
    return {link.axis_index: link for link in links if link.loss_eligible}


__all__ = [
    "VERSION", "MANUAL_EXACT_ALIASES", "PaperAxisLink",
    "PaperDenoisingAutoencoder", "build_paper_axis_crosswalk",
    "paper_external_features", "target_link_by_axis",
]
