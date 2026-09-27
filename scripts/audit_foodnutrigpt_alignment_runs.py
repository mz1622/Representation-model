"""Verify completed matched mapping arms and replay saved ranks via public loaders."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import AlignmentPanel,evaluate_mapping,selection_score
from foodcomp.research_alignment_inference import AlignmentModel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    receipt={'status':'incomplete','script_sha256':digest(Path(__file__)),'complete_test_opened':False}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);device='cuda' if torch.cuda.is_available() else 'cpu'
        panel=AlignmentPanel(data,ROOT,device)
        manifests=[];histories=[];records=[]
        for objective in ['mse','contrastive']:
            run=ROOT/f'output/v9_r5/mapper_{objective}60';m=json.loads((run/'run_manifest.json').read_text())
            assert m['status']=='complete' and m['epoch_completed']==60 and not m['test_opened']
            for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
            assert json.loads((run/'panel_manifest.json').read_text())==panel.manifest
            h=pd.read_csv(run/'history.csv',float_precision='round_trip');np.testing.assert_array_equal(h.epoch,np.arange(1,61))
            for row in h.itertuples():
                assert row.training_profiles==len(panel.rows)
                assert row.training_order_sha256==fingerprint_array(np.random.default_rng(m['seed']+row.epoch).permutation(len(panel.rows)))
            assert m['best_epoch']==int(h.loc[h.selection_mean_mrr.idxmax(),'epoch'])
            model=AlignmentModel(run,device=device);ranks,scores=evaluate_mapping(model.predict_features,panel)
            pd.testing.assert_frame_equal(ranks,pd.read_parquet(run/'ranks.parquet'),check_exact=True)
            saved=json.loads((run/'metrics.json').read_text());assert scores==saved['metrics']
            assert selection_score(scores)==m['best_selection_mean_mrr']
            assert saved['candidate_sha256']=='e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
            names=panel.namespace.names[:3].tolist();profile={int(panel.axes[0]):0.}
            np.testing.assert_array_equal(model.encode(names[0],profile),model.encode(names[1],profile))
            ranking=model.retrieve_names(profile,names,top_k=3);assert len(ranking)==3
            manifests.append(m);histories.append(h)
            records.append({'run':str(run.relative_to(ROOT)),'checkpoint_sha256':m['checkpoint_hash'],'manifest_sha256':digest(run/'run_manifest.json'),
                'best_epoch':m['best_epoch'],'all19089_ranks_and_metrics_exact':True,'trained_nutrition_api_independent_of_name':True})
        for field in ['initial_state_sha256','code_hashes','data_hash','panel_hash','name_cache_hash','parameter_count','seed']:
            assert manifests[0][field]==manifests[1][field],field
        for field in ['epoch','training_profiles','training_order_sha256','learning_rate']:
            np.testing.assert_array_equal(histories[0][field],histories[1][field])
        for alpha in ['01','10','100']:
            path=ROOT/f'output/v9_r5/ridge_alpha{alpha}'
            assert json.loads((path/'run_manifest.json').read_text())['status']=='complete'
            assert json.loads((path/'panel_manifest.json').read_text())==panel.manifest
        receipt.update(status='complete',matched_initialization_source_inputs_weights_panel_exposure_order_lr=True,
            parameters=manifests[0]['parameter_count'],records=records,all_registered_ridge_inputs_match=True,
            scope='Both complete60-epoch single-seed arms; saved ranks/metrics replay exactly with public loader. No seed confirmation and no completion capability claimed.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt['status'],receipt['parameters'])


if __name__=='__main__':main()
