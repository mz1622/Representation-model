"""Registered width512 ->1024 comparison at a fixed60-epoch budget."""
from train_foodnutrigpt_v9_r1 import main


if __name__ == "__main__":
    main(version="V9-R2", protocol_change="Compared with completed mlp60_mae_width512, change only hidden width512 to1024. Retain two layers, LayerNorm, shared head, zero additional name-only tasks, MAE, all187 targets, original-name PCA32, labels/scales/family tasks/source weights, seed20260922, batch256, lr0.001, AdamW/clip and fixed60-epoch cosine schedule. Parameter count rises667900 to1859836. Different tensor shapes change initialization and optimization geometry; this is not matched weights or a pure proof of representational capacity. All three tasks use one primary142-selected checkpoint.")
