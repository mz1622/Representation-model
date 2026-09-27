"""Controlled name-only task fraction on the completed MAE/512/LayerNorm control."""
from train_foodnutrigpt_v9_r1 import main

if __name__=="__main__":
    main(version="V9-R3",protocol_change="Compared with the completed R2 MAE/512/LayerNorm/60 control, only change the fraction of existing family tasks that hide all numeric context. Preserve canonical labels, axes, source weights, per-epoch target exposure, task order, architecture, MAE objective, initialization, optimizer, 60-epoch schedule and fixed validation selection. This is a training input-distribution intervention, not an architecture comparison or additional supervision. R1/R2 independent V9 ablations remain unfinished.")
