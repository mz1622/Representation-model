"""Replay candidate8 ranks and all60 registered training-view masks against parent."""
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
from foodcomp.research_alignment_views import partial_view_masks
from foodcomp.research_alignment_inference import AlignmentModel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    receipt={'status':'incomplete','script_sha256':digest(Path(__file__)),'complete_test_opened':False}
    try:
        run=ROOT/'output/v9_r5/mapper_contrastive60_partial05';parent=ROOT/'output/v9_r5/mapper_contrastive60'
        m=json.loads((run/'run_manifest.json').read_text());old=json.loads((parent/'run_manifest.json').read_text())
        assert m['status']=='complete' and m['epoch_completed']==60 and not m['test_opened']
        assert m['args']['partial_view_probability']==.5
        for key in ['initial_state_sha256','data_hash','panel_hash','name_cache_hash','parameter_count','seed']:
            assert m[key]==old[key],key
        for key in ['objective','epochs','batch_size','width','learning_rate','temperature','seed']:
            assert m['args'][key]==old['args'][key],key
        for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
        for path,expected in old['code_hashes'].items():
            assert digest(parent/'code_snapshot'/Path(path).name)==expected
            if Path(path).name!='train_foodnutrigpt_v9_r5_mapping.py':assert m['code_hashes'][path]==expected
        data=ResearchData(ROOT/'data/processed'/VERSION);device='cuda' if torch.cuda.is_available() else 'cpu'
        panel=AlignmentPanel(data,ROOT,device)
        assert json.loads((run/'panel_manifest.json').read_text())==panel.manifest
        assert json.loads((parent/'panel_manifest.json').read_text())==panel.manifest
        h=pd.read_csv(run/'history.csv',float_precision='round_trip');hp=pd.read_csv(parent/'history.csv',float_precision='round_trip')
        np.testing.assert_array_equal(h.epoch,np.arange(1,61))
        for field in ['epoch','training_profiles','training_order_sha256','learning_rate']:
            np.testing.assert_array_equal(h[field],hp[field])
        original=panel.features[:,142:].astype(bool)
        for row in h.itertuples():
            assert row.training_profiles==len(panel.rows)
            assert row.training_order_sha256==fingerprint_array(np.random.default_rng(m['seed']+row.epoch).permutation(len(panel.rows)))
            visible,assigned=partial_view_masks(original,.5,m['seed'],row.epoch)
            assert row.training_view_mask_sha256==fingerprint_array(visible)
            assert row.partial_assignment_sha256==fingerprint_array(assigned)
            assert row.partial_assigned_profiles==assigned.sum()
            assert row.partial_changed_profiles==np.any(visible!=original,axis=1).sum()
            assert row.original_visible_cells==original.sum() and row.training_visible_cells==visible.sum()
            assert (visible.sum(1)>=3).all() and not (visible & ~original).any()
            np.testing.assert_array_equal(visible[~assigned],original[~assigned])
        assert np.isfinite(h.select_dtypes('number')).all().all()
        assert m['best_epoch']==int(h.loc[h.selection_mean_mrr.idxmax(),'epoch'])
        model=AlignmentModel(run,device=device);ranks,scores=evaluate_mapping(model.predict_features,panel)
        pd.testing.assert_frame_equal(ranks,pd.read_parquet(run/'ranks.parquet'),check_exact=True)
        saved=json.loads((run/'metrics.json').read_text());assert scores==saved['metrics']
        assert selection_score(scores)==m['best_selection_mean_mrr']
        assert saved['candidate_sha256']=='e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
        names=panel.namespace.names[:3].tolist();profile={int(panel.axes[0]):0.}
        np.testing.assert_array_equal(model.encode(names[0],profile),model.encode(names[1],profile))
        assert len(model.retrieve_names(profile,names,top_k=3))==3
        receipt.update(status='complete',run=str(run.relative_to(ROOT)),checkpoint_sha256=m['checkpoint_hash'],
            manifest_sha256=digest(run/'run_manifest.json'),parent_checkpoint_sha256=old['checkpoint_hash'],
            initialization_parameters_labels_weights_panel_order_lr_unchanged=True,all60_view_hashes_and_counts_exact=True,
            all19089_saved_ranks_and_metrics_replayed_exact=True,trained_nutrition_api_independent_of_name=True,
            best_epoch=m['best_epoch'],training_profiles=len(panel.rows),parameter_count=m['parameter_count'],
            assignment_count_range=[int(h.partial_assigned_profiles.min()),int(h.partial_assigned_profiles.max())],
            changed_count_range=[int(h.partial_changed_profiles.min()),int(h.partial_changed_profiles.max())],
            retained_cell_range=[int(h.training_visible_cells.min()),int(h.training_visible_cells.max())],
            scope='Single-seed completed candidate8. Same parent except registered training views; names/weights/negative groups/architecture and evaluation unchanged. No seed confirmation or completion head.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt)


if __name__=='__main__':main()
