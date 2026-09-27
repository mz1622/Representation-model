"""Registered R4 auxiliary-weight comparison using the shared controlled trainer."""
from train_foodnutrigpt_v9_r1 import main

if __name__=="__main__":
    main(version="V9-R4",protocol_change="Metabolome training coefficient1->0.5 on original MAE512/60 control; nutrition coefficient1 and187 denominator retained. No consistency loss or architecture/input/scoring change. Query policy caller_or_schema_axes_v1 is common and reference-replayed.")
