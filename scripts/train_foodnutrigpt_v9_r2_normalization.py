"""Registered normalization ablation against the completed MAE/512/60 control."""
from train_foodnutrigpt_v9_r1 import main

if __name__=="__main__":
    main(version="V9-R2",protocol_change="Compared with completed mlp60_mae_width512, replace only the hidden LayerNorm with Identity. Keep all linear-layer initial weights, width512, depth, MAE, data/tasks/text/scales/weights, seed, task order, optimizer, batch256, initial lr0.001 and fixed60-epoch schedule. This tests normalization's effect on numeric representation/optimization; it does not assume that LayerNorm entirely removes input magnitude information.")
