"""Registered R8 completion control: exact name dimensions, otherwise frozen R2 recipe."""
from train_foodnutrigpt_v9_r1 import main

if __name__ == '__main__':
    main(version='V9-R8', protocol_change='R8 exact training-only common PCA32/128 in128 name slots; matched717052parameter MLP. Same exhaustive family targets, weights, masks,187axis MAE and60epoch schedule. Old32slot baseline is historical only.')
