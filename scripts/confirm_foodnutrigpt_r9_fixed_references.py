"""Confirm three fixed-recipe Transformers against frozen trees on unchanged data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from foodcomp.research_confirmation import SEEDS
from foodcomp.research_final_statistics import (RATES, CELL_KEYS, ERRORS, CONTRIBUTIONS,
    mean_seed_frames, number_summary, prediction_details, retrieval_groups, retrieval_identity_for_run)
from foodcomp.research_fixed_reference_confirmation import confirm_fixed_references
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_statistics import paired_interval, paired_axis_intervals, paired_group_rates
from foodcomp.research_transformer_r9 import frozen_inputs
from train_foodnutrigpt_v9_r9_confirmation import load_confirmation, verify_bindings
from compare_foodnutrigpt_final_models import Inputs, completed, closed, audit_match


def locations(plan, parent):
    runs = {SEEDS[0]: ROOT / plan['parent_run']}
    audits = {SEEDS[0]: ROOT / plan['parent_audit']}
    for seed in SEEDS[1:]:
        runs[seed] = ROOT / 'output/v9_r9_confirmation' / f"{parent['candidate']}_seed{seed}"
        audits[seed] = ROOT / 'reports' / f'v9_r9_confirmation_seed{seed}_audit_v1/verification.json'
    return runs, audits


def validate_history(manifest, history):
    spec = manifest['spec']
    if (not np.array_equal(history.epoch, np.arange(1, spec['epochs'] + 1))
            or not np.isfinite(history.select_dtypes('number')).all().all()
            or not history.training_tasks.eq(manifest['training_tasks']).all()
            or not history.observed_target_cells.eq(manifest['observed_target_cells']).all()):
        raise ValueError('Invalid full training history.')
    for row in history.itertuples():
        order = np.random.default_rng(spec['seed'] + row.epoch).permutation(manifest['training_tasks'])
        if fingerprint_array(order) != row.training_order_sha256:
            raise ValueError('Training task order changed.')
        lr = spec['learning_rate'] * (.01 + .99 * (1 + np.cos(np.pi * (row.epoch - 1) / spec['schedule_epochs'])) / 2)
        np.testing.assert_allclose(row.learning_rate, lr, rtol=1e-12, atol=1e-15)
    if (int(history.loc[history.validation_primary.idxmin(), 'epoch']) != manifest['best_epoch']
            or float(history.validation_primary.min()) != manifest['best_primary']):
        raise ValueError('Checkpoint selection changed.')


def load_models(args, inputs, frozen, freeze_path, plan, parent, paths, audits):
    models = {'transformer': {}}
    plan_hash = digest(args.plan)
    for seed in SEEDS:
        run = paths[seed]
        manifest = inputs.read(run / 'run_manifest.json')
        completed(manifest)
        audit = inputs.read(audits[seed]); completed(audit)
        if (manifest['kind'] != 'transformer_direct' or manifest['version'] != 'V9-R9'
                or manifest['seed'] != seed or manifest['spec']['seed'] != seed
                or manifest['epoch_completed'] != manifest['spec']['epochs']
                or manifest['baseline_refit'] or manifest['data_modified']):
            raise ValueError('Invalid Transformer repetition identity.')
        expected_spec = dict(parent['spec']); expected_spec['seed'] = seed
        if manifest['spec'] != expected_spec:
            raise ValueError('Recipe differs across seeds.')
        for key in ['data_hash', 'name_cache_hash', 'panel_hash']:
            if manifest[key] != frozen[key]: raise ValueError('Frozen input mismatch: ' + key)
        if (manifest['freeze_sha256'] != digest(freeze_path)
                or manifest['environment'] != parent['environment']
                or manifest['parameter_count'] != parent['parameter_count']
                or manifest['trainable_parameter_count'] != parent['trainable_parameter_count']):
            raise ValueError('Changed environment, architecture or freeze.')
        expected_code = dict(parent['code_hashes'])
        if seed != SEEDS[0]:
            _, _, _, bindings = load_confirmation(ROOT, args.plan, seed, frozen, freeze_path)
            verify_bindings(ROOT, bindings)
            for path, expected in bindings.items(): inputs.record(path, expected)
            expected_code['scripts/train_foodnutrigpt_v9_r9_confirmation.py'] = digest(ROOT/'scripts/train_foodnutrigpt_v9_r9_confirmation.py')
            if (manifest['config_sha256'] != plan_hash or manifest['recipe_changes'] != ['seed']
                    or manifest['confirmation_input_hashes'] != bindings
                    or manifest['seed22_manifest_sha256'] != plan['parent_manifest_sha256']
                    or manifest['decision_record_sha256'] != plan['decision_record_sha256']):
                raise ValueError('Repetition is not bound to the registered decision.')
        if manifest['code_hashes'] != expected_code:
            raise ValueError('Numerical implementation changed across seeds.')
        for path, expected in expected_code.items(): inputs.record(path, expected)
        if (audit['manifest_sha256'] != digest(run/'run_manifest.json')
                or audit['checkpoint_sha256'] != manifest['checkpoint_sha256']
                or audit['freeze_sha256'] != digest(freeze_path)
                or not audit['all_epoch_orders_exposures_and_schedule_verified']
                or not audit['both323809_predictions_public_loader_replayed_exact']
                or not audit['all_candidate_vectors_and19089_ranks_replayed_exact']):
            raise ValueError('Incomplete or stale independent replay.')
        inputs.record(run/'best_model.pt', manifest['checkpoint_sha256'])
        for task in ['completion', 'name_only']:
            inputs.record(run/f'{task}_predictions.parquet', audit['prediction_sha256'][task])
        scores = inputs.read(run/'metrics.json')
        if scores != audit['metrics']: raise ValueError('Audited metrics changed.')
        history = pd.read_csv(inputs.record(run/'history.csv'), float_precision='round_trip')
        validate_history(manifest, history)
        models['transformer'][seed] = {'run':run, 'retrieval':run/'retrieval', 'manifest':manifest, 'scores':scores}
    for role, selected in [('rf',frozen['primary_rf']), ('xgb',frozen['matched_xgb'])]:
        run = ROOT/'output/v9_r8'/selected
        manifest = inputs.read(run/'run_manifest.json', frozen['input_hashes'][(run/'run_manifest.json').relative_to(ROOT).as_posix()])
        completed(manifest)
        if manifest['kind'] != role or manifest['seed'] != SEEDS[0] or manifest['training_row_cap'] is not None:
            raise ValueError('Incorrect frozen tree reference.')
        models[role] = {None:{'run':run,'retrieval':run/'retrieval','manifest':manifest,'scores':inputs.read(run/'metrics.json')}}
    run = ROOT/'output/v9_r7/exact_name_knn32'
    manifest = inputs.read(run/'run_manifest.json'); completed(manifest)
    audit_match(inputs, ROOT/'reports/v9_r7_knn_completed_audit_v1/verification.json', run)
    inputs.record(run/'nutrition_predictions.parquet', manifest['nutrition_prediction_sha256'])
    models['name_knn'] = {None:{'run':run,'retrieval':run,'manifest':manifest,
        'scores':{'name_only':inputs.read(run/'nutrition_metrics.json')}}}
    for method, runs in models.items():
        for item in runs.values():
            m = item['manifest']
            if m['data_hash'] != frozen['data_hash'] or m['name_cache_hash'] != frozen['name_cache_hash']:
                raise ValueError('Different model input data or text cache: ' + method)
            closed(item['scores'])
    return models


def statistics(args, inputs, models):
    data = ResearchData(ROOT/'data/processed'/VERSION)
    axes = data.axes.loc[data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible,'axis_index'].to_numpy()
    if len(axes) != 142: raise ValueError('Primary protocol changed.')
    all_axes, all_sources, scalars, rates, costs = [], [], [], [], []
    grouped, rank_groups, summaries = {}, {}, {}
    query_reference, retrieval_identity = None, None
    for method, runs in models.items():
        grouped[method], rank_groups[method] = {}, {}
        summaries[method] = {'tasks':{},'retrieval':{},'runs':[]}
        for seed, item in runs.items():
            run, ret, manifest = item['run'], item['retrieval'], item['manifest']
            grouped[method][seed] = {}
            summaries[method]['runs'].append({'seed':seed,'run':run.relative_to(ROOT).as_posix(),
                'manifest_sha256':digest(run/'run_manifest.json'),'best_epoch':manifest.get('best_epoch')})
            costs.append({'method':method,'seed':seed,'elapsed_seconds':manifest['elapsed_seconds']})
            tasks = ['name_only'] if method=='name_knn' else ['completion','name_only']
            for task in tasks:
                filename = 'nutrition_predictions.parquet' if method=='name_knn' else f'{task}_predictions.parquet'
                score, axis_frame, frame, source_frame = prediction_details(data, pd.read_parquet(inputs.record(run/filename)))
                if score != item['scores'][task]: raise ValueError('Saved scores cannot be reproduced.')
                grouped[method][seed][task] = frame
                all_axes.append(axis_frame.assign(method=method,seed=seed,task=task))
                all_sources.append(source_frame.assign(method=method,seed=seed,task=task))
                for subset in ['nutrition','food_metabolome','all']:
                    scalars.append({'method':method,'seed':seed,'task':task,'subset':subset,**score[subset]})
            meta = inputs.read(ret/'metrics.json'); closed(meta)
            if method=='transformer' and meta['checkpoint_sha256'] != manifest['checkpoint_sha256']:
                raise ValueError('Retrieval checkpoint differs from completion.')
            inputs.record(ret/'candidate_names.json', meta['candidate_sha256'])
            identity = retrieval_identity_for_run(meta, manifest)
            if retrieval_identity is None: retrieval_identity = identity
            if identity != retrieval_identity or identity['candidate_count'] != 49913:
                raise ValueError('Retrieval protocols differ.')
            query_keys, groups = retrieval_groups(pd.read_parquet(inputs.record(ret/'ranks.parquet')),meta)
            if query_reference is None: query_reference = query_keys
            pd.testing.assert_frame_equal(query_reference, query_keys)
            rank_groups[method][seed] = groups
            expected = {str(float(row['visible_fraction'])):row for row in meta['metrics']}
            for fraction, frame in groups.items():
                values = frame[RATES].mean().to_dict()
                np.testing.assert_allclose([values[k] for k in RATES],[expected[fraction][k] for k in RATES],rtol=1e-12,atol=1e-15)
                rates.append({'method':method,'seed':seed,'visible_fraction':fraction,**values})
            print(f'Rescored {method} seed {seed}',flush=True)
    scalar_frame, rate_frame = pd.DataFrame(scalars), pd.DataFrame(rates)
    averages, average_rates = {}, {}
    for method, runs in models.items():
        averages[method], average_rates[method] = {}, {}
        for task in grouped[method][next(iter(runs))]:
            frames = {seed:grouped[method][seed][task] for seed in runs}
            averages[method][task] = mean_seed_frames(frames,CELL_KEYS,ERRORS+CONTRIBUTIONS) if method=='transformer' else frames[None]
            summaries[method]['tasks'][task] = {}
            for subset in ['nutrition','food_metabolome','all']:
                frame = scalar_frame[(scalar_frame.method==method)&(scalar_frame.task==task)&(scalar_frame.subset==subset)]
                columns = [k for k in frame.columns if k not in ['method','seed','task','subset','axes']]
                summaries[method]['tasks'][task][subset] = {k:number_summary(frame[k]) for k in columns}
        for fraction in ['0.3','1.0']:
            frames = {seed:rank_groups[method][seed][fraction] for seed in runs}
            average_rates[method][fraction] = mean_seed_frames(frames,['exact_name_group_id'],RATES) if method=='transformer' else frames[None]
            frame = rate_frame[(rate_frame.method==method)&(rate_frame.visible_fraction==fraction)]
            summaries[method]['retrieval'][fraction] = {k:number_summary(frame[k]) for k in RATES}
    confirmation, confirmation_axes = confirm_fixed_references(
        {seed:grouped['transformer'][seed]['completion'] for seed in SEEDS},
        {role:averages[role]['completion'] for role in ['rf','xgb']},axes,repeats=1000)
    comparisons, intervals = {}, []
    for role in ['rf','xgb','name_knn']:
        comparisons[role] = {'tasks':{},'retrieval':{}}
        for task, base in averages[role].items():
            candidate = averages['transformer'][task]
            if task=='completion':
                pair = confirmation['comparisons'][role]['paired_intervals']
                axis = confirmation_axes[role]
            else:
                pair = {m:paired_interval(base,candidate,axes,metric=m) for m in ERRORS[:2]}
                axis = paired_axis_intervals(base,candidate)
            pieces = [f[f.axis_index.isin(axes)].groupby('axis_index')[CONTRIBUTIONS].mean().mean() for f in [base,candidate]]
            delta = (pieces[1]-pieces[0]).to_dict()
            np.testing.assert_allclose(sum(delta.values()),pair['scaled_log_mae']['candidate']-pair['scaled_log_mae']['baseline'],rtol=1e-10,atol=1e-12)
            comparisons[role]['tasks'][task] = {'paired_intervals':pair,'additive_primary_difference':delta}
            intervals.append(axis.assign(baseline=role,task=task))
        for fraction in ['0.3','1.0']:
            comparisons[role]['retrieval'][fraction] = paired_group_rates(average_rates[role][fraction],average_rates['transformer'][fraction],RATES)
    inputs.verify()
    frozen_inputs(ROOT)
    scalar_frame.to_csv(args.output_dir/'metrics_by_seed.csv',index=False)
    rate_frame.to_csv(args.output_dir/'retrieval_by_seed.csv',index=False)
    pd.concat(all_axes,ignore_index=True).to_csv(args.output_dir/'axis_metrics_by_seed.csv',index=False)
    pd.concat(all_sources,ignore_index=True).to_csv(args.output_dir/'source_metrics_by_seed.csv',index=False)
    pd.concat(intervals,ignore_index=True).merge(data.axes[['axis_index','canonical_name','loss_group']],on='axis_index',validate='many_to_one').to_csv(args.output_dir/'axis_paired_intervals.csv',index=False)
    pd.DataFrame(costs).to_csv(args.output_dir/'recorded_compute_seconds.csv',index=False)
    result = {'status':'complete_fixed_reference_statistics','models':summaries,'comparisons':comparisons,
        'confirmation':confirmation,'retrieval_protocol':retrieval_identity,'input_hashes':inputs.hashes,
        'workspace_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'complete_test_opened':False,'data_modified':False,'baseline_refit':False,
        'final_report_complete':False,'goal_achieved':False,
        'scope':confirmation['scope']+' All three tasks use each seed\'s completion-selected checkpoint. KNN is name-only/retrieval only. Per-axis intervals are exploratory without multiplicity correction.'}
    write_json(args.output_dir/'summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'experiments/foodnutrigpt_v9_research/r9/first_group_v1/confirmation_plan.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    inputs = Inputs()
    frozen,freeze_path = frozen_inputs(ROOT)
    inputs.record(freeze_path)
    plan,parent,_,bindings = load_confirmation(ROOT,args.plan,SEEDS[1],frozen,freeze_path)
    for path,expected in bindings.items(): inputs.record(path,expected)
    paths,audits = locations(plan,parent)
    missing,unfinished = [],[]
    for seed in SEEDS:
        for path in [paths[seed]/'run_manifest.json',audits[seed]]:
            if not path.exists(): missing.append(str(path))
            elif inputs.read(path)['status']!='complete': unfinished.append(str(path))
    if missing or unfinished:
        if args.check_only:
            print(json.dumps({'ready':False,'missing':missing,'unfinished':unfinished,'outputs_written':False}))
            return
        raise ValueError('Wait for all three complete runs and independent audits.')
    for path in ['scripts/confirm_foodnutrigpt_r9_fixed_references.py',
        'src/foodcomp/research_fixed_reference_confirmation.py','src/foodcomp/research_final_statistics.py',
        'src/foodcomp/research_confirmation.py','src/foodcomp/research_statistics.py',
        'scripts/compare_foodnutrigpt_final_models.py']:
        inputs.record(path)
    for path,expected in frozen['input_hashes'].items(): inputs.record(path,expected)
    models = load_models(args,inputs,frozen,freeze_path,plan,parent,paths,audits)
    if args.check_only:
        inputs.verify()
        print(json.dumps({'ready':True,'outputs_written':False,'methods':list(models)}))
        return
    if args.output_dir.exists(): raise FileExistsError('Do not overwrite a previous analysis.')
    args.output_dir.mkdir(parents=True)
    started=time.monotonic()
    write_json(args.output_dir/'status.json',{'status':'running','complete_test_opened':False})
    try:
        result=statistics(args,inputs,models)
        write_json(args.output_dir/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,
            'summary_sha256':digest(args.output_dir/'summary.json'),'complete_test_opened':False,'final_report_complete':False})
        print(json.dumps({'status':result['status'],'gates':result['confirmation']['gates']}))
    except Exception as error:
        write_json(args.output_dir/'status.json',{'status':'failed','error_type':type(error).__name__,
            'error':str(error),'complete_test_opened':False,'elapsed_seconds':time.monotonic()-started})
        raise


if __name__=='__main__': main()
