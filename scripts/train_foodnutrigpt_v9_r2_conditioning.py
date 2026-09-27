"""R2 final probe: train-only internal standardization of unchanged name inputs."""
from train_foodnutrigpt_v9_r1 import main

if __name__ == "__main__":
    main(version="V9-R2", protocol_change="Compared with registered mlp60_mae_width512, only add a frozen model-internal affine transform of existing cached PCA32 name vectors. Mean/population std fit once on distinct original TRAIN name strings only, saved as buffers; no labels/held-out vectors in fit. External inputs, caches, tasks, labels, weights, width512, LayerNorm/shared head, MAE, zero added name-only tasks, optimizer, seed, batch256 and fixed60-epoch schedule stay unchanged. Same learnable parameter initialization/RNG; effective function, gradients and regularization geometry change. This is not evidence of improved text use until controlled evaluation completes.")
