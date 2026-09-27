"""R2 MAE/512 duration screen with a fixed 60-epoch learning-rate horizon."""
from train_foodnutrigpt_v9_r1 import main

if __name__ == "__main__":
    main(version="V9-R2", protocol_change="Predeclared MAE/512 training-budget experiment: compare best through epoch 20 with best through epoch 60 of this SAME 60-epoch cosine trajectory. Architecture, loss, data/tasks/weights, text, seed, initialization, batch size and initial learning rate stay fixed. Comparing the final run to the earlier 20-epoch-horizon model mixes duration and schedule; only the internal 20/60 comparison isolates the registered budget difference.")
