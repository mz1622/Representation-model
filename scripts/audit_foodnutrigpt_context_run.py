"""Verify all60 R6 input interventions, supervision counts and saved predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json,score_predictions
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,fingerprint_array
from foodcomp.research_context_dropout import context_dropout_masks
from foodcomp.research_neural import evaluate
from foodcomp.research_inference import NutritionModel


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        run=args.run;m=read(run/'run_manifest.json');parent=ROOT/'output/v9_r2/mlp60_mae_width512';pm=read(parent/'run_manifest.json')
        assert m['status']=='complete' and m['epoch_completed']==60 and not m['test_opened']
        assert m['args']['context_dropout'] in ['mix_30_60_90','fixed_30']
        for key in ['data_hash','panel_hash','name_cache_hash','seed']:assert m[key]==pm[key],key
        for key in ['kind','objective','mlp_width','epochs','schedule_epochs','batch_size','learning_rate','name_only_probability']:
            assert m['args'].get(key,0)==pm['args'].get(key,0),key
        for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
        assert digest(run/'best_model.pt')==m['checkpoint_hash']
        model=NutritionModel(run/'best_model.pt');data=model.data
        panel=FamilyPanel(data,model._cached_text,ROOT/'data/processed'/PANEL_VERSION,'cpu')
        assert digest(panel.root/'manifest.json')==m['panel_hash']
        visible=data.observed[panel.rows]&~panel.masks.numpy()[panel.family_ids]
        target=data.observed[panel.rows]&panel.masks.numpy()[panel.family_ids]&panel.eligible.numpy()
        target_count=int(target.sum());assert target_count==1828536
        original_cells=int(visible.sum());original_empty=int((~visible.any(1)).sum())
        h=pd.read_csv(run/'history.csv',float_precision='round_trip');hp=pd.read_csv(parent/'history.csv',float_precision='round_trip')
        np.testing.assert_array_equal(h.epoch,np.arange(1,61));assert np.isfinite(h.select_dtypes('number')).all().all()
        for field in ['epoch','learning_rate','training_tasks']:np.testing.assert_array_equal(h[field],hp[field])
        for row in h.itertuples():
            extra,ids=context_dropout_masks(len(panel.rows),len(data.axes),m['args']['context_dropout'],m['seed'],int(row.epoch))
            assert row.context_mask_sha256==fingerprint_array(extra) and row.context_rate_ids_sha256==fingerprint_array(ids)
            assert row.training_order_sha256==fingerprint_array(np.random.default_rng(m['seed']+row.epoch).permutation(len(panel.rows)))
            remaining=visible&~extra;removed=visible&extra
            expected={'original_visible_cells':original_cells,'training_visible_cells':int(remaining.sum()),'removed_visible_cells':int(removed.sum()),
                'changed_context_tasks':int(removed.any(1).sum()),'original_no_context_tasks':original_empty,
                'training_no_context_tasks':int((~remaining.any(1)).sum()),'training_target_cells':target_count}
            expected.update({f'context_rate_{rate}_tasks':int((ids==i).sum()) for i,rate in enumerate([30,60,90])})
            for field,value in expected.items():assert getattr(row,field)==value,(row.epoch,field)
            assert not (remaining&target).any()
        assert m['best_epoch']==int(h.loc[h.validation_primary.idxmin(),'epoch'])
        saved=read(run/'metrics.json')
        for mode in ['completion','name_only']:
            predictions=evaluate(model.model,data,model._cached_text,model.device,mode)
            pd.testing.assert_frame_equal(predictions,pd.read_parquet(run/f'{mode}_predictions.parquet'),check_exact=True)
            assert score_predictions(data,predictions)[0]==saved[mode]
        replay=args.output_dir/'retrieval_replay'
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py'),'--checkpoint',str(run/'best_model.pt'),
            '--output-dir',str(replay.resolve())],cwd=ROOT,check=True,capture_output=True,text=True)
        pd.testing.assert_frame_equal(pd.read_parquet(replay/'ranks.parquet'),pd.read_parquet(args.retrieval/'ranks.parquet'),check_exact=True)
        rm=read(replay/'metrics.json');original=read(args.retrieval/'metrics.json')
        for key in ['metrics','candidate_count','candidate_sha256','query_profiles','checkpoint_sha256','data_sha256','scoring','correct_answers']:assert rm[key]==original[key]
        receipt.update(status='complete',run=str(run),manifest_sha256=digest(run/'run_manifest.json'),checkpoint_sha256=m['checkpoint_hash'],
            data_sha256=m['data_hash'],panel_sha256=m['panel_hash'],best_epoch=m['best_epoch'],
            all60_masks_orders_exposures_learning_rates_verified=True,all60_supervision_counts_unchanged=True,
            complete_family_hidden=True,both_full_prediction_tables_metrics_reloaded_exact=True,all19089_ranks_metrics_reloaded_exact=True,
            training_tasks=len(panel.rows),training_target_cells=target_count,original_visible_cells=original_cells,original_no_context_tasks=original_empty,
            retained_cells_range=[int(h.training_visible_cells.min()),int(h.training_visible_cells.max())],
            no_context_tasks_range=[int(h.training_no_context_tasks.min()),int(h.training_no_context_tasks.max())],
            actual_retained_fraction_range=[float(h.training_visible_cells.min()/original_cells),float(h.training_visible_cells.max()/original_cells)],
            frozen_parent_checkpoint_sha256=pm['checkpoint_hash'],
            scope='Single-seed input-distribution intervention. Supervision/validation fixed; training contexts intentionally differ from unaugmented tree/control inputs. No model-confirmation claim.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt['status'],receipt['run'],receipt['retained_cells_range'])

if __name__=='__main__':main()
