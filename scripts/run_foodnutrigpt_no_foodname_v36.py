#!/usr/bin/env python3
"""Train V18 with factorized ordered-axis value relations."""
from pathlib import Path

from run_foodnutrigpt_no_foodname_attention_common import (
    AttentionExperiment, ROOT, run_experiment,
)
from foodcomp.research_no_foodname_v36 import RelationValueAxisPairTransformer


if __name__ == "__main__":
    run_experiment(AttentionExperiment(
        version="no_foodname_v36_relation_value",
        model_class=RelationValueAxisPairTransformer,
        architecture="RelationValueAxisPairTransformer",
        label="v36_relation_value",
        causal_change=(
            "zero-initialized factorized ordered-axis relation vectors added "
            "to attention value messages; all V18 attention scores, routes, "
            "full pair bias, model dimensions, training parameters, data, and "
            "loss fixed"
        ),
        attention_description=(
            "V18 full self-attention plus ordered-axis relation-aware values"
        ),
        literature_basis=(
            "Shaw et al., 2018, self-attention with relative relation representations"
        ),
        variant_source=ROOT / "src/foodcomp/research_no_foodname_v36.py",
        wrapper_source=Path(__file__).resolve(),
    ))
