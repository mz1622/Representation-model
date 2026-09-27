"""Check final corrected nameKNN artifacts against all training pools and scored cells."""
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import sklearn
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json,score_predictions
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import NameSpace
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    run=ROOT/'output/v9_r5/retrieval_name_knn10_tree';receipt={'status':'incomplete','script_sha256':digest(Path(__file__)),'complete_test_opened':False}
    try:
        m=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'));assert m['status']=='complete' and m['backend']=='kd_tree' and not m['complete_test_opened']
        for key in ['all187_reverse_query_predictions_exact','all142_candidate_validation_neighbor_indices_distances_predictions_exact',
            'all142_candidate_fixed256_subbatch_predictions_exact','all19089_ranks_reload_exact']:assert m[key],key
        for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
        data=ResearchData(ROOT/'data/processed'/VERSION);ns=NameSpace(data,ROOT)
        assert m['data_hash']==digest(data.root/'manifest.json') and m['name_cache_hash']==digest(ns.cache/'manifest.json')
        axes=np.flatnonzero(data.axes.loss_group.eq('nutrition')&data.axes.loss_eligible)
        fitting=json.loads((run/'axis_fitting_manifest.json').read_text(encoding='utf-8'));assert {x['axis_index'] for x in fitting}==set(data.targets)
        train_groups=set(data.profiles.iloc[data.train].exact_name_group_id);val_groups=set(data.profiles.iloc[data.validation].exact_name_group_id)
        assert not train_groups&val_groups
        for row in fitting:
            a=row['axis_index'];train=data.train[data.observed[data.train,a]]
            assert row['train_profiles']==len(train) and row['train_rows_sha256']==fingerprint_array(train)
            assert row['training_values_sha256']==fingerprint_array(data.values[train,a])
            assert row['training_weights_sha256']==fingerprint_array(data.weights[train,a])
            assert row['reverse_query_prediction_exact']
            if a in axes:assert row['candidate_vs_validation_mismatch_count']==0
        names=json.loads((run/'candidate_names.json').read_text(encoding='utf-8'));assert names==ns.names.tolist()
        raw=np.load(run/'candidate_predicted_raw.npy');scaled=np.load(run/'candidate_scaled.npy')
        assert raw.shape==scaled.shape==(49913,142) and np.isfinite(raw).all() and (raw>=0).all()
        np.testing.assert_array_equal(scaled,np.log1p(raw/data.scale[axes]).astype(np.float32))
        assert digest(run/'candidate_scaled.npy')==m['candidate_vectors_sha256']
        pred=pd.read_parquet(run/'replayed_nutrition_predictions.parquet')
        assert score_predictions(data,pred)[0]==json.loads((run/'nutrition_metrics.json').read_text(encoding='utf-8'))
        subset=pred[pred.axis_index.isin(axes)];columns={int(a):i for i,a in enumerate(axes)}
        reproduced=raw[ns.profile_name_ids[subset.profile_index.to_numpy()],subset.axis_index.map(columns).to_numpy()]
        np.testing.assert_array_equal(reproduced,subset.prediction.to_numpy())
        ranks,metrics=evaluate_candidates(data,names,scaled,axes,'cuda' if torch.cuda.is_available() else 'cpu')
        pd.testing.assert_frame_equal(ranks,pd.read_parquet(run/'ranks.parquet'),check_exact=True)
        assert metrics==json.loads((run/'metrics.json').read_text(encoding='utf-8'))['metrics']
        receipt.update(status='complete',manifest_sha256=digest(run/'run_manifest.json'),all187_training_pools_values_weights_verified=True,
            training_validation_candidate_groups_disjoint=True,all187_nutrition_scores_reproduced=True,
            nutrition_scored_cells=len(subset),all_scored_nutrition_predictions_exact_from_name_candidate_vectors=True,
            all19089_ranks_replayed_exact=True,candidate_vectors_sha256=m['candidate_vectors_sha256'],
            environment={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},
            scope='All frozen observed validation nutrition cells agree with name-only candidate predictions. Training pools use only original training observations; no validation labels used for candidate fitting. Batch checks cover full validation, reversed validation and fixed256-name subsets, not a proof over all future hardware or inputs.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt)


if __name__=='__main__':main()
