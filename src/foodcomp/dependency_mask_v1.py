"""Frozen, target-specific measurement-dependency masks.

This module deliberately separates a numerical dependency graph from the
project's legacy semantic ``mask_family`` column.  Only relations whose
measurement definitions make one axis an explicit aggregate (or component)
of another axis are accepted.  Every unregistered axis is a singleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


VERSION = "infoods_explicit_dependency_mask_v1"

INFOODS_TAGNAMES = (
    "FAO/INFOODS component identifiers (tagnames), version 4.0: "
    "https://www.fao.org/infoods/infoods/standards-guidelines/"
    "food-component-identifiers-tagnames/en/"
)
INFOODS_PART2 = (
    "FAO/INFOODS tagnames, Part 2 (fatty acids): "
    "https://www.fao.org/fileadmin/templates/food_composition/documents/"
    "upload/PART2.TXT"
)
INFOODS_PART3 = (
    "FAO/INFOODS tagnames, Part 3 (fibre, phytosterols): "
    "https://www.fao.org/fileadmin/templates/food_composition/documents/"
    "upload/PART3.TXT"
)
INFOODS_PART4 = (
    "FAO/INFOODS tagnames, Part 4 (sugars, vitamin D): "
    "https://www.fao.org/fileadmin/templates/food_composition/documents/"
    "upload/PART4.TXT"
)
INFOODS_BIOFOODCOMP = (
    "FAO/INFOODS Guidelines for Food Matching version 1.2, BioFoodComp "
    "component identifiers (LUTNZEA): https://www.fao.org/4/i3560e/i3560e.pdf"
)


@dataclass(frozen=True)
class EdgeSpec:
    parent: str
    child: str
    relation: str
    authority: str
    rationale: str


def _edge(parent: str, child: str, relation: str, authority: str, rationale: str):
    return EdgeSpec(parent, child, relation, authority, rationale)


_SATURATED = (
    "Butyric acid (4:0)",
    "Hexanoic acid (6:0)",
    "Octanoic acid (8:0)",
    "Decanoic acid (10:0)",
    "Dodecanoic acid (12:0)",
    "Tridecanoic acid (13:0)",
    "Tetradecanoic acid (14:0)",
    "Pentadecanoic acid (15:0)",
    "Hexadecanoic acid (16:0)",
    "Heptadecanoic acid (17:0)",
    "Octadecanoic acid (18:0)",
    "Eicosanoic acid (20:0)",
    "Docosanoic acid (22:0)",
    "Tetracosanoic acid (24:0)",
)


def accepted_edge_specs() -> tuple[EdgeSpec, ...]:
    """Return the immutable, evidence-frozen relation registry."""
    specs = [
        _edge("Vitamin D (D2 + D3)", "Cholecalciferol (vitamin D3)",
              "explicit_sum_component", INFOODS_PART4,
              "VITD is defined as the sum of vitamin D2 and vitamin D3."),
        _edge("Vitamin D (D2 + D3)", "Vitamin D2 (ergocalciferol)",
              "explicit_sum_component", INFOODS_PART4,
              "VITD is defined as the sum of vitamin D2 and vitamin D3."),
        _edge("Sugars, total", "Fructose", "explicit_sum_component", INFOODS_PART4,
              "SUGAR is the sum of free monosaccharides and disaccharides."),
        _edge("Sugars, total", "Galactose", "explicit_sum_component", INFOODS_PART4,
              "SUGAR is the sum of free monosaccharides and disaccharides."),
        _edge("Sugars, total", "Glucose", "explicit_sum_component", INFOODS_PART4,
              "SUGAR is the sum of free monosaccharides and disaccharides."),
        _edge("Sugars, total", "Lactose", "explicit_sum_component", INFOODS_PART4,
              "SUGAR is the sum of free monosaccharides and disaccharides."),
        _edge("Sugars, total", "Maltose", "explicit_sum_component", INFOODS_PART4,
              "SUGAR is the sum of free monosaccharides and disaccharides."),
        _edge("Sugars, total", "Sucrose", "explicit_sum_component", INFOODS_PART4,
              "SUGAR is the sum of free monosaccharides and disaccharides."),
        _edge("Dietary fibre, total", "Insoluble dietary fibers",
              "explicit_inclusion", INFOODS_PART3,
              "Total dietary fibre includes insoluble dietary fibre."),
        _edge("Insoluble dietary fibers", "Cellulose", "explicit_inclusion",
              INFOODS_PART3,
              "Insoluble dietary fibre consists primarily of cellulose, "
              "hemicellulose and lignin."),
        _edge("Insoluble dietary fibers", "Lignin", "explicit_inclusion",
              INFOODS_PART3,
              "Insoluble dietary fibre consists primarily of cellulose, "
              "hemicellulose and lignin."),
        _edge("Phytosterols", "Beta-sitosterol", "explicit_sum_component",
              INFOODS_PART3, "PHYSTR includes beta-sitosterol."),
        _edge("Phytosterols", "Campesterol", "explicit_sum_component",
              INFOODS_PART3, "PHYSTR includes campesterol."),
        _edge("Phytosterols", "Stigmasterol", "explicit_sum_component",
              INFOODS_PART3, "PHYSTR includes stigmasterol."),
        _edge("Lutein + zeaxanthin", "Lutein", "explicit_sum_component",
              INFOODS_BIOFOODCOMP,
              "LUTNZEA is the combined lutein and zeaxanthin quantity."),
        _edge("Lutein + zeaxanthin", "Zeaxanthin", "explicit_sum_component",
              INFOODS_BIOFOODCOMP,
              "LUTNZEA is the combined lutein and zeaxanthin quantity."),
        _edge("Fatty acids, total trans", "Fatty acids, total trans-monoenoic",
              "explicit_subtotal", INFOODS_PART2,
              "FATRNM is the trans-monoenoic subtotal of total trans fatty acids."),
        _edge("Fatty acids, total trans", "Fatty acids, total trans-polyenoic",
              "explicit_subtotal", INFOODS_PART2,
              "FATRNP is the trans-polyenoic subtotal of total trans fatty acids."),
        _edge("Fatty acids, total trans-monoenoic", "trans-Docosenoic acid (22:1 trans)",
              "structural_member", INFOODS_PART2,
              "A 22:1 trans fatty acid is trans-monoenoic."),
        _edge("Fatty acids, total trans-monoenoic", "trans-Hexadecenoic acid (16:1 trans)",
              "structural_member", INFOODS_PART2,
              "A 16:1 trans fatty acid is trans-monoenoic."),
        _edge("Fatty acids, total trans-monoenoic", "trans-Octadecenoic acid (18:1 trans)",
              "structural_member", INFOODS_PART2,
              "An 18:1 trans fatty acid is trans-monoenoic."),
        _edge("Fatty acids, total trans-polyenoic", "trans,trans-Linoleic acid (18:2 trans,trans)",
              "structural_member", INFOODS_PART2,
              "An 18:2 trans fatty acid is trans-polyenoic."),
        _edge("Fatty acids, total trans-polyenoic", "trans-Linoleic acid (18:2; isomer unspecified)",
              "structural_member", INFOODS_PART2,
              "An 18:2 trans fatty acid is trans-polyenoic."),
    ]
    specs.extend(
        _edge("Fatty acids, total saturated", child, "structural_member",
              INFOODS_PART2,
              "A fatty acid with zero double bonds is a saturated fatty acid.")
        for child in _SATURATED
    )
    return tuple(specs)


def rejected_candidates() -> pd.DataFrame:
    """Document tempting relations intentionally excluded from the graph."""
    rows = [
        ("semantic nutrient classes", "vitamins/minerals/amino acids/bioactives",
         "A taxonomy class does not imply an algebraic or direct measurement dependency."),
        ("Fat, total", "fatty-acid totals or individual fatty acids",
         "Conversion from fatty acids to total fat depends on analytical basis and factors; source tags are not preserved in this research view."),
        ("Fatty acids, total mono/polyunsaturated", "individual unsaturated fatty acids",
         "The registry contains overlapping unspecified and isomer-specific axes; without source tagnames an exact inclusion relation cannot be guaranteed."),
        ("unspecified fatty-acid isomer", "cis/trans or named isomers",
         "INFOODS states that unspecified positional/geometric entries may be mixtures; they are not defined as exact sums of the listed isomers."),
        ("Folate, total", "Folate, food / Folic acid",
         "INFOODS distinguishes analytical folate expressions and dietary folate equivalents; the canonical name alone does not establish the equation used by each source."),
        ("Choline, total", "Choline",
         "The unqualified Choline axis lacks an analysis tag proving whether it is free choline or another total expression."),
        ("Niacin", "Nicotinic acid",
         "Niacin may include nicotinamide and/or equivalents; the available name does not establish an exact parent-child value relation."),
        ("Available carbohydrate", "Sugars / starch",
         "Available carbohydrate is method- and expression-dependent; the view does not retain the required INFOODS expression tag."),
        ("Protein, total", "individual amino acids",
         "Protein is commonly calculated from nitrogen and is not the direct sum of the reported free/total amino-acid axes."),
    ]
    return pd.DataFrame(rows, columns=["candidate_parent", "candidate_child_or_group", "rejection_reason"])


class DependencyMaskPolicy:
    """Validated directed graph and target-specific ancestor/descendant masks."""

    def __init__(self, axes: pd.DataFrame, specs: Iterable[EdgeSpec] | None = None):
        required = {"axis_index", "canonical_name"}
        if not required.issubset(axes.columns):
            raise ValueError(f"axes must contain {sorted(required)}")
        if axes.canonical_name.duplicated().any() or axes.axis_index.duplicated().any():
            raise ValueError("axis names and indices must be unique")
        self.version = VERSION
        self.axis_count = len(axes)
        self.name_by_axis = dict(zip(axes.axis_index.astype(int), axes.canonical_name.astype(str)))
        self.axis_by_name = {name: axis for axis, name in self.name_by_axis.items()}
        self.specs = tuple(accepted_edge_specs() if specs is None else specs)
        missing = sorted({x for edge in self.specs for x in (edge.parent, edge.child)} - set(self.axis_by_name))
        if missing:
            raise ValueError(f"dependency registry names absent from axis table: {missing}")
        pairs = [(self.axis_by_name[e.parent], self.axis_by_name[e.child]) for e in self.specs]
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate dependency edge")
        self.parents = {axis: set() for axis in self.name_by_axis}
        self.children = {axis: set() for axis in self.name_by_axis}
        for parent, child in pairs:
            if parent == child:
                raise ValueError("self dependency edge")
            self.children[parent].add(child)
            self.parents[child].add(parent)
        self._assert_acyclic()
        self._masks = {axis: self._closure_mask(axis) for axis in self.name_by_axis}

    def _assert_acyclic(self):
        state = {axis: 0 for axis in self.name_by_axis}
        def visit(axis):
            if state[axis] == 1:
                raise ValueError("dependency graph contains a directed cycle")
            if state[axis] == 2:
                return
            state[axis] = 1
            for child in self.children[axis]:
                visit(child)
            state[axis] = 2
        for axis in state:
            visit(axis)

    @staticmethod
    def _walk(start: int, adjacency: dict[int, set[int]]) -> set[int]:
        found, pending = set(), list(adjacency[start])
        while pending:
            axis = pending.pop()
            if axis not in found:
                found.add(axis)
                pending.extend(adjacency[axis])
        return found

    def _closure_mask(self, axis: int) -> np.ndarray:
        if axis not in self.name_by_axis:
            raise KeyError(axis)
        selected = {axis} | self._walk(axis, self.parents) | self._walk(axis, self.children)
        return np.asarray(sorted(selected), dtype=np.int64)

    def mask_indices(self, axis: int) -> np.ndarray:
        return self._masks[int(axis)].copy()

    def mask_matrix(self, target_axes: np.ndarray) -> np.ndarray:
        target_axes = np.asarray(target_axes, dtype=np.int64)
        result = np.zeros((len(target_axes), self.axis_count), dtype=bool)
        for i, axis in enumerate(target_axes):
            result[i, self._masks[int(axis)]] = True
        return result

    def edge_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "parent_axis": self.axis_by_name[e.parent],
                "parent_name": e.parent,
                "child_axis": self.axis_by_name[e.child],
                "child_name": e.child,
                "relation": e.relation,
                "authority": e.authority,
                "rationale": e.rationale,
            }
            for e in self.specs
        ])

    def target_policy_frame(self, targets: Iterable[int] | None = None) -> pd.DataFrame:
        targets = sorted(self.name_by_axis if targets is None else map(int, targets))
        rows = []
        for axis in targets:
            mask = self._masks[axis]
            rows.append({
                "target_axis": axis,
                "target_name": self.name_by_axis[axis],
                "mask_size": len(mask),
                "mask_axes": ";".join(map(str, mask)),
                "mask_names": "; ".join(self.name_by_axis[item] for item in mask),
                "singleton": len(mask) == 1,
            })
        return pd.DataFrame(rows)


__all__ = [
    "VERSION", "EdgeSpec", "DependencyMaskPolicy", "accepted_edge_specs",
    "rejected_candidates", "INFOODS_TAGNAMES",
]
