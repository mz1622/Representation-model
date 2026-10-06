"""ReGLU masked-axis completion with depth-shared axis-pair bias."""
from __future__ import annotations

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class SharedAxisPairBiasTransformer(AxisPairBiasMaskedAxisTransformer):
    """Use one ordered axis-relation table in every Transformer block."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        shared_bias = self.blocks[0].attention.pair_bias
        for block in self.blocks[1:]:
            block.attention.pair_bias = shared_bias
