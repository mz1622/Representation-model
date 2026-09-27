"""R2 first single-factor probe: direct scaled-value SmoothL1 versus MAE."""
from train_foodnutrigpt_v9_r1 import main

if __name__=="__main__":
    main(version="V9-R2",protocol_change="Compared with R1 mlp20_panel, change only the supervised direct-regression objective from SmoothL1(beta=1) to MAE in the same scaled-log space. Architecture, data/tasks/weights, text, seed, initialization, training order, 20-epoch learning-rate schedule and selection panel are unchanged. Metadata-only runner refactoring does not alter R1 training.")
