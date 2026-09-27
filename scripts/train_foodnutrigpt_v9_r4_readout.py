"""R4 first controlled experiment: nonlinear axis-query residual readout."""
from train_foodnutrigpt_v9_r1 import main

if __name__ == "__main__":
    main(version="V9-R4", protocol_change="Compared with completed mlp60_mae_width512, only add a shared nonlinear axis-query residual readout: context512->128, learned axis32, joint hidden128/GELU/scalar, zero final layer. Train from the same parent parameter initialization, not a pretrained parent checkpoint. Initial function and parent preclip gradients match; added parameters, global clipping and AdamW decay may change the first parent update and full optimization trajectory. Same R0 quarantined labels/scales, PCA32 input/cache, exhaustive family tasks/weights, MAE187, width512/LayerNorm/shared head, zero added name-only tasks, optimizer, batch256, seed and fixed60-epoch schedule. No text standardization, task-mix or consistency intervention. This experiment alone is not evidence of transfer or unseen-axis semantics.")
