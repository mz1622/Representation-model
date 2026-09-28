"""Plot all three complete Transformer trajectories after aggregate confirmation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from foodcomp.research_confirmation import SEEDS
from foodcomp.research_r0 import digest, write_json
from confirm_foodnutrigpt_r9_fixed_references import validate_history


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def within_repo(path):
    resolved = (ROOT/str(path).replace('\\','/')).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Evidence must stay within the repository.')
    return resolved


def load_evidence(directory):
    path = directory/'summary.json'
    result, status = read(path), read(directory/'status.json')
    if (result['status'] != 'complete_fixed_reference_statistics' or status['status'] != 'complete'
            or status['summary_sha256'] != digest(path)
            or any(result[key] for key in ['complete_test_opened','data_modified','baseline_refit'])):
        raise ValueError('Completed, unchanged aggregate statistics required.')
    for relative, expected in result['input_hashes'].items():
        if digest(within_repo(relative)) != expected:
            raise ValueError('Changed statistics input: '+relative)
    records = result['models']['transformer']['runs']
    if [record['seed'] for record in records] != list(SEEDS):
        raise ValueError('Exactly the three registered ordered seeds required.')
    histories, identities, recipe = {}, [], None
    for record in records:
        run = within_repo(record['run'])
        manifest = read(run/'run_manifest.json')
        if (manifest['status'] != 'complete' or manifest['seed'] != record['seed']
                or manifest['kind'] != 'transformer_direct'
                or digest(run/'run_manifest.json') != record['manifest_sha256']):
            raise ValueError('Changed trained-model identity.')
        current = {k:v for k,v in manifest['spec'].items() if k!='seed'}
        if recipe is None: recipe = current
        if current != recipe: raise ValueError('Different recipes across seeds.')
        history_path = run/'history.csv'
        key = history_path.relative_to(ROOT).as_posix()
        if digest(history_path) != result['input_hashes'][key]:
            raise ValueError('Changed trajectory after confirmation.')
        history = pd.read_csv(history_path, float_precision='round_trip')
        validate_history(manifest, history)
        histories[record['seed']] = history
        identities.append({'seed':record['seed'],'run':record['run'],
            'best_epoch':manifest['best_epoch'],'best_primary':manifest['best_primary'],
            'history_sha256':digest(history_path),'manifest_sha256':record['manifest_sha256']})
    expected = result['models']['transformer']['tasks']['completion']['nutrition']['scaled_log_mae']['mean']
    np.testing.assert_allclose(np.mean([row['best_primary'] for row in identities]),expected,rtol=1e-13,atol=1e-15)
    return result, histories, identities, recipe


def plot(result, histories, identities, recipe, output):
    fig, axes = plt.subplots(2,3,figsize=(14.4,8.2),layout='constrained')
    fields = ['train_loss','validation_primary','validation_primary',
              'validation_legacy_log_mae','validation_positive_mae','validation_zero_mae']
    titles = ['Training objective: 187 axes + calibration','Validation primary: 142 axes',
              'Primary: final 20 epochs','Legacy nutrition log-MAE','Conditional positive nutrition MAE',
              'Conditional explicit-zero nutrition MAE']
    metric_names = [None,'scaled_log_mae','scaled_log_mae','log_mae',
                    'positive_scaled_log_mae','zero_scaled_log_mae']
    for seed, identity in zip(SEEDS,identities):
        history=histories[seed]
        for i,(ax,field) in enumerate(zip(axes.flat,fields)):
            subset=history if i!=2 else history[history.epoch>recipe['epochs']-20]
            line,=ax.plot(subset.epoch,subset[field],label=str(seed),linewidth=1.4)
            chosen=history[history.epoch.eq(identity['best_epoch'])]
            if i!=2 or identity['best_epoch']>recipe['epochs']-20:
                ax.scatter(chosen.epoch,chosen[field],s=65,marker='*',color=line.get_color(),zorder=4)
    for i,(ax,title,metric) in enumerate(zip(axes.flat,titles,metric_names)):
        if metric is not None:
            for role,color,style in [('rf','#444444','--'),('xgb','#888888',':')]:
                value=result['models'][role]['tasks']['completion']['nutrition'][metric]['mean']
                ax.axhline(value,color=color,linestyle=style,linewidth=1.3,label=f'Fixed {role.upper()}')
            if metric=='log_mae':
                limit=result['models']['rf']['tasks']['completion']['nutrition'][metric]['mean']*1.02
                ax.axhline(limit,color='#ad6c28',linestyle='-.',linewidth=1.1,label='RF legacy +2% guard')
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error / objective')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7.5,frameon=False)
    fig.suptitle(f"Transformer width{recipe['d_model']} / MAE / lr{recipe['learning_rate']:g}: three fixed-recipe seeds",fontsize=14)
    fig.savefig(output/'learning_curves.png',dpi=170)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    return {'panels':6,'seed_count':3,'all_full_trajectories':True,
        'stars':'Each seed completion-selected epoch, reused on all panels; no per-panel reselection.',
        'training_validation_warning':'Training objective includes187 axes and calibration; do not subtract it from validation142-axis MAE.',
        'zero_positive_warning':'Conditional errors have different denominators and cannot be added directly.',
        'interval_warning':'No uncertainty band is invented from three trajectories; seed variability and conditional food-group intervals are reported in aggregate statistics.',
        'zoom_warning':'Final20 panel is a visual zoom on the same full trajectory, not another selection window.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics-dir',type=Path,default=ROOT/'reports/v9_r9_three_seed_confirmation_v1')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    required=[args.statistics_dir/'summary.json',args.statistics_dir/'status.json']
    missing=[str(path) for path in required if not path.exists()]
    if missing:
        if args.check_only:
            print(json.dumps({'ready':False,'missing':missing,'outputs_written':False}))
            return
        raise FileNotFoundError('Wait for complete aggregate statistics.')
    if read(args.statistics_dir/'status.json')['status']!='complete' and args.check_only:
        print(json.dumps({'ready':False,'reason':'aggregate statistics incomplete','outputs_written':False}))
        return
    result,histories,identities,recipe=load_evidence(args.statistics_dir)
    if args.check_only:
        print(json.dumps({'ready':True,'outputs_written':False,'seeds':list(histories)}))
        return
    if args.output_dir.exists(): raise FileExistsError('Do not overwrite a previous figure version.')
    args.output_dir.mkdir(parents=True)
    try:
        notes=plot(result,histories,identities,recipe,args.output_dir)
        for row in identities:
            if digest(within_repo(row['run'])/'history.csv')!=row['history_sha256']:
                raise ValueError('Trajectory changed while plotting.')
        write_json(args.output_dir/'verification.json',{'status':'generated_visual_review_pending',
            'statistics_sha256':digest(args.statistics_dir/'summary.json'),'script_sha256':digest(Path(__file__)),
            'records':identities,'interpretation_notes':notes,
            'figure_hashes':{name:digest(args.output_dir/name) for name in ['learning_curves.png','learning_curves.svg']},
            'conditional_fixed_rf_milestone_passed':result['confirmation']['conditional_fixed_rf_milestone_passed'],
            'visual_review_complete':False,'complete_test_opened':False,'training_performed':False,
            'data_modified':False,'baseline_refit':False,'final_report_complete':False,'goal_achieved':False})
        print(json.dumps({'status':'generated_visual_review_pending','output_dir':str(args.output_dir)}))
    except Exception as error:
        write_json(args.output_dir/'failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':main()
