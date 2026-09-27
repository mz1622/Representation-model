"""Registered fixed-input Ridge sweep and strict replay of the existing R0 arm."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_alignment import AlignmentPanel,evaluate_mapping,save_evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alpha',type=float,required=True);p.add_argument('--replay-reference',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.alpha not in [.1,1.,10.,100.]:raise ValueError('Unregistered Ridge coefficient.')
    if args.alpha==1 and args.replay_reference is None:raise ValueError('Alpha1 is a replay, not a new candidate.')
    if args.replay_reference is not None and args.alpha!=1:raise ValueError('Replay requires alpha1.')
    args.output_dir.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    files=[Path(__file__),ROOT/'src/foodcomp/research_alignment.py',ROOT/'src/foodcomp/research_r0.py',ROOT/'src/foodcomp/research_text.py']
    snapshot=args.output_dir/'code_snapshot';snapshot.mkdir()
    for file in files:shutil.copyfile(file,snapshot/file.name)
    manifest={'status':'running','version':'V9-R5','kind':'ridge','args':vars(args),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(file.relative_to(ROOT)):digest(file) for file in files},'test_opened':False,'confirmation_allowed':False}
    write_json(args.output_dir/'run_manifest.json',manifest)
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION)
        panel=AlignmentPanel(data,ROOT,'cuda' if torch.cuda.is_available() else 'cpu')
        write_json(args.output_dir/'panel_manifest.json',panel.manifest)
        ridge=Ridge(alpha=args.alpha,solver='lsqr',tol=1e-6).fit(panel.features,panel.targets,sample_weight=panel.weights)
        if not np.isfinite(ridge.coef_).all() or not np.isfinite(ridge.intercept_).all():raise FloatingPointError('Nonfinite Ridge coefficients.')
        np.savez(args.output_dir/'ridge.npz',coef=ridge.coef_,intercept=ridge.intercept_)
        ranks,scores=evaluate_mapping(ridge.predict,panel)
        save_evaluation(args.output_dir,panel,ranks,scores,method='ridge_to_text',extra={'ridge_alpha':args.alpha,'elapsed_seconds':time.monotonic()-started,'checkpoint_sha256':digest(args.output_dir/'ridge.npz')})
        if args.replay_reference:
            reference=args.replay_reference
            with np.load(reference/'ridge.npz') as old:
                np.testing.assert_array_equal(ridge.coef_,old['coef']);np.testing.assert_array_equal(ridge.intercept_,old['intercept'])
            pd.testing.assert_frame_equal(ranks,pd.read_parquet(reference/'ranks.parquet'),check_exact=True)
            previous=json.loads((reference/'metrics.json').read_text())
            assert scores==previous['metrics']
            assert digest(args.output_dir/'candidate_names.json')==previous['candidate_sha256']
            write_json(args.output_dir/'replay_verification.json',{'status':'complete','all_coefficients_ranks_metrics_exact':True,
                'rank_rows':len(ranks),'reference':str(reference),'reference_metrics_sha256':digest(reference/'metrics.json'),'panel_manifest_sha256':digest(args.output_dir/'panel_manifest.json'),'complete_test_opened':False})
        manifest.update(status='complete',data_hash=panel.manifest['data_sha256'],name_cache_hash=panel.manifest['name_cache_sha256'],
            panel_hash=digest(args.output_dir/'panel_manifest.json'),elapsed_seconds=time.monotonic()-started,checkpoint_hash=digest(args.output_dir/'ridge.npz'))
        write_json(args.output_dir/'run_manifest.json',manifest);print(scores)
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'run_manifest.json',manifest);raise


if __name__=='__main__':main()
