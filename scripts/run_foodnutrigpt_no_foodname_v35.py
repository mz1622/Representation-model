#!/usr/bin/env python3
"""Train V18 with disentangled content-axis attention scores."""
from pathlib import Path

from run_foodnutrigpt_no_foodname_attention_common import (
    AttentionExperiment, ROOT, run_experiment,
)
from foodcomp.research_no_foodname_v35 import (
    DisentangledAxisContentTransformer,
)


if __name__ == "__main__":
    run_experiment(AttentionExperiment(
        version="no_foodname_v35_disentangled_axis_content",
        model_class=DisentangledAxisContentTransformer,
        architecture="DisentangledAxisContentTransformer",
        label="v35_disentangled_axis_content",
        causal_change=(
            "zero-initialized content-to-axis and axis-to-content QK score "
            "terms in every layer; all V18 attention routes, full pair bias, "
            "model dimensions, training parameters, data, and loss fixed"
        ),
        attention_description=(
            "V18 full self-attention plus disentangled content-axis QK terms"
        ),
        literature_basis="He et al., 2020, DeBERTa disentangled attention",
        variant_source=ROOT / "src/foodcomp/research_no_foodname_v35.py",
        wrapper_source=Path(__file__).resolve(),
    ))
