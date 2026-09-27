"""Require the full8-epoch name baseline trajectory and predictions to replay exactly."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    old=ROOT/'output/v9_r0/name_mlp8_quarantined';new=ROOT/'output/v9_r5/name_mlp8_replay_v1'
    receipt={'status':'incomplete','script_sha256':digest(Path(__file__)),'complete_test_opened':False}
    try:
        a=json.loads((old/'run_manifest.json').read_text());b=json.loads((new/'run_manifest.json').read_text())
        assert a['status']==b['status']=='complete' and a['best_epoch']==b['best_epoch']
        assert not a['complete_test_opened'] and not b['complete_test_opened']
        for key in ['data_hash','protocol_hash','name_cache_hash']:assert a[key]==b[key]
        for path,expected in b['code_hashes'].items():assert digest(new/'code_snapshot'/Path(path).name)==expected
        ha=pd.read_csv(old/'history.csv',float_precision='round_trip');hb=pd.read_csv(new/'history.csv',float_precision='round_trip')
        for key in ['epoch','train_loss','validation_primary','validation_nutrition_legacy_log_mae','learning_rate']:
            np.testing.assert_array_equal(ha[key],hb[key],err_msg=key)
        sa=torch.load(old/'best_model.pt',map_location='cpu',weights_only=True)['model_state']
        sb=torch.load(new/'best_model.pt',map_location='cpu',weights_only=True)['model_state']
        assert sa.keys()==sb.keys()
        for key in sa:assert torch.equal(sa[key],sb[key]),key
        for mode in ['completion','name_only']:
            pd.testing.assert_frame_equal(pd.read_parquet(old/f'{mode}_predictions.parquet'),pd.read_parquet(new/f'{mode}_predictions.parquet'),check_exact=True)
        assert json.loads((old/'metrics.json').read_text())==json.loads((new/'metrics.json').read_text())
        receipt.update(status='complete',all8_history_fields_model_parameters_both_predictions_and_metrics_exact=True,
            best_epoch=b['best_epoch'],replay_elapsed_seconds=b['elapsed_seconds'],code_sha256=b['code_hashes'],
            replay_manifest_sha256=digest(new/'run_manifest.json'),parent_checkpoint_sha256=a['checkpoint_sha256'],
            scope='Full GPU trajectory compatibility replay; same8-epoch horizon. The registered60-epoch candidate has a different cosine horizon from its first update and is not an identical8-epoch prefix.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt)


if __name__=='__main__':main()
