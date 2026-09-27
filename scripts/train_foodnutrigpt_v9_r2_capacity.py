"""R2 controlled capacity probe after the completed MAE loss comparison."""
from train_foodnutrigpt_v9_r1 import main

if __name__ == "__main__":
    main(version="V9-R2", protocol_change="Compared with R2 mlp20_mae, change only MLP hidden width from 256 to 512. MAE loss, depth, data/tasks/weights, name features, 20-epoch schedule, optimizer, batch size, seed and task order remain fixed. Different tensor shapes necessarily change initialized parameters; this is a capacity experiment, not matched initial weights.")
