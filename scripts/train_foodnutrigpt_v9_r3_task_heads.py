"""Single-factor shared/separate prediction heads on the fixed 10% task mixture."""
from train_foodnutrigpt_v9_r1 import main

if __name__=="__main__":
    main(version="V9-R3",protocol_change="Compared with the completed mlp60_name_mix10 control, separate linear completion/name-only output heads while sharing the unchanged encoder. Route only by visible numeric input. Clone initial head without new RNG draws. Keep10% task assignments, observed targets, source weights, axes, MAE, optimizer,60-epoch schedule and selection unchanged. Extra head capacity and gradient routing are coupled architectural effects; no isolated gradient-conflict causality claim. Other registered R1/R2/R3 runs remain independent.")
