#!/usr/bin/env python3
"""Validate and freeze the explicit measurement-dependency mask registry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_v1 import (  # noqa: E402
    DependencyMaskPolicy, INFOODS_TAGNAMES, VERSION, rejected_candidates,
)
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data = ResearchData(args.data_dir, "quarantined")
    policy = DependencyMaskPolicy(data.axes)
    edges = policy.edge_frame()
    targets = policy.target_policy_frame(data.targets)
    rejected = rejected_candidates()
    edges.to_csv(args.output_dir / "accepted_edges.csv", index=False)
    targets.to_csv(args.output_dir / "target_mask_policy.csv", index=False)
    rejected.to_csv(args.output_dir / "rejected_candidates.csv", index=False)

    # Machine-check the intended asymmetry: a child's siblings must remain visible.
    sibling_checks = 0
    for parent, children in policy.children.items():
        for child in children:
            mask = set(policy.mask_indices(child))
            siblings = set(children) - {child}
            if mask & siblings:
                raise AssertionError("a sibling was hidden by a child target")
            sibling_checks += len(siblings)
    broad = data.axes.mask_family.astype(str).to_numpy()
    broad_sizes = {axis: int((broad == broad[axis]).sum()) for axis in data.targets}
    policy_sizes = {axis: len(policy.mask_indices(int(axis))) for axis in data.targets}
    # The graph may cross a legacy hand-written category boundary (for example
    # an aggregate and its explicitly named component), so compare scope only
    # in aggregate rather than treating the legacy categories as authority.
    if np.mean(list(policy_sizes.values())) >= np.mean(list(broad_sizes.values())):
        raise AssertionError("dependency masks were not narrower on average")
    summary = {
        "status": "complete",
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "authority": INFOODS_TAGNAMES,
        "accepted_directed_edges": int(len(edges)),
        "registered_axes": int(len(set(edges.parent_axis) | set(edges.child_axis))),
        "supervised_targets": int(len(targets)),
        "singleton_targets": int(targets.singleton.sum()),
        "non_singleton_targets": int((~targets.singleton).sum()),
        "minimum_mask_size": int(targets.mask_size.min()),
        "maximum_mask_size": int(targets.mask_size.max()),
        "mean_mask_size": float(targets.mask_size.mean()),
        "legacy_mean_family_size": float(np.mean(list(broad_sizes.values()))),
        "sibling_visibility_checks": sibling_checks,
        "rejected_candidate_groups": int(len(rejected)),
        "category_used_as_model_input": False,
        "legacy_mask_family_used_for_masking": False,
        "policy_rule": "target + all directed ancestors + all directed descendants; siblings remain visible; unregistered axes are singleton",
        "training_results_used_to_define_edges": False,
    }
    summary["artifact_sha256"] = {
        name: digest(args.output_dir / name)
        for name in ("accepted_edges.csv", "target_mask_policy.csv", "rejected_candidates.csv")
    }
    write_json(args.output_dir / "manifest.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
