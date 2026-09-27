"""Registered direct-value V9 comparison on the shared R1 panel."""
from train_foodnutrigpt_v9_r1 import main

if __name__=="__main__":
    main(version="V9-R2",protocol_change="Registered direct-output V9 branch. SmoothL1 arm changes the hurdle output objective relative to R1 v9_20_a1_s1: supervise amounts including explicit zeros, omit presence BCE and omit presence multiplication in inference. These are a coupled output-design change, not three separately identified mechanisms. The following direct MAE arm changes only SmoothL1 to MAE. Backbone, amount-head initialization, dropout RNG call order, data/tasks/text/weights, source calibration, seed, batch64, initial lr1e-4 and fixed20-epoch schedule are held constant.")
