"""Same-trajectory comparison of primary versus fixed-panel hurdle selection."""
from train_foodnutrigpt_v9_r1 import main

if __name__ == "__main__":
    main(version="V9-R2", protocol_change="Replay fixed R1 V9 a1/s1/20 configuration with both primary142 and source-free hurdle187 checkpoint selection on one shared validation panel. Compare selected checkpoints within this replay only. Historical hurdle objective is evaluated using R0 labels/scales and full-panel source/axis weights, not historical stochastic masks/minibatch averaging. Training is unaffected by either selector; no early stopping or schedule adaptation.")
