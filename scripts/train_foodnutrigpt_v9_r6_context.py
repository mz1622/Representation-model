"""Preregistered R6 extra-context deletion on the unchanged MAE512 family panel."""
from train_foodnutrigpt_v9_r1 import main

if __name__=='__main__':
    main(version='V9-R6',protocol_change='Only training context visibility changes after original full family hiding. Same labels,source weights,target exposure,MAE512/LayerNorm architecture,60schedule andfixed evaluation. Additional deletion is not an extra supervised target or representation objective.')
