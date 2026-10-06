#!/usr/bin/env python3
"""Train V18 with a zero-initialized target-query residual."""
from pathlib import Path

from run_foodnutrigpt_no_foodname_attention_common import (
    AttentionExperiment, ROOT, run_experiment,
)
from foodcomp.research_no_foodname_v34 import TargetQueryResidualTransformer


if __name__ == "__main__":
    run_experiment(AttentionExperiment(
        version="no_foodname_v34_target_query_residual",
        model_class=TargetQueryResidualTransformer,
        architecture="TargetQueryResidualTransformer",
        label="v34_target_query",
        causal_change=(
            "zero-initialized target-token-only query residual in every layer; "
            "all V18 attention routes, pair bias, model dimensions, training "
            "parameters, data, and loss fixed"
        ),
        attention_description=(
            "V18 full self-attention plus a target-token-only query residual"
        ),
        literature_basis="role-specialized query projection",
        variant_source=ROOT / "src/foodcomp/research_no_foodname_v34.py",
        wrapper_source=Path(__file__).resolve(),
    ))
