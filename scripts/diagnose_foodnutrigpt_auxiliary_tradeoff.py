"""Post-result subgroup uncertainty; never changes selection or refits a model."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_statistics import paired_interval


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline-dir','candidate-dir','comparison-dir','output-dir']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    data=ResearchData(ROOT/'data/processed'/VERSION)
    frames=[];hashes=[]
    comparison=json.loads((args.comparison_dir/'summary.json').read_text())
    for role,run in [('baseline',args.baseline_dir),('candidate',args.candidate_dir)]:
        manifest=json.loads((run/'run_manifest.json').read_text())
        if manifest['status']!='complete' or manifest['test_opened'] or manifest['data_hash']!=digest(data.root/'manifest.json'):raise ValueError('Incompatible run.')
        if digest(run/'completion_predictions.parquet')!=comparison[role+'_sha256']:raise ValueError('Stale paired reference.')
        path=run/'completion_candidate_errors.parquet';frames.append(pd.read_parquet(path));hashes.append(digest(path))
    axes=data.axes.loc[data.axes.loss_eligible & data.axes.loss_group.eq('food_metabolome'),'axis_index'].to_numpy()
    if len(axes)!=45:raise ValueError('Expected45 axes.')
    intervals={m:paired_interval(*frames,axes,metric=m) for m in ['scaled_log_mae','log_mae']}
    for metric,record in intervals.items():
        for role in ['baseline','candidate']:
            expected=comparison['scores'][role]['food_metabolome'][metric]
            if abs(record[role]-expected)>1e-12:raise ValueError('Saved group errors differ from scored predictions.')
    a=pd.read_csv(args.comparison_dir/'axis_paired_intervals.csv')
    support={}
    for group,g in a.groupby('loss_group'):
        support[group]={'axes':len(g),'point_improved_axes':int(g.candidate_minus_baseline.lt(0).sum()),
            'improved_axis_intervals_excluding_zero':int(g.difference_95_high.lt(0).sum()),
            'worse_axis_intervals_excluding_zero':int(g.difference_95_low.gt(0).sum()),
            'sparse_axes_below30':int(g.sparse_support_below_30.sum()),
            'minimum_candidate_support':int(g.candidate_support.min())}
    s=pd.read_csv(args.comparison_dir/'source_metrics.csv')
    pairs=s[s.role.eq('baseline')].merge(s[s.role.eq('candidate')],on='source',suffixes=('_baseline','_candidate'),validate='one_to_one')
    for key in ['supported_nutrition_axes','candidate_groups']:
        if not pairs[key+'_baseline'].equals(pairs[key+'_candidate']):raise ValueError('Source coverage mismatch.')
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'summary.json',{'status':'complete','metabolome_paired_intervals':intervals,
        'axis_summary':support,'sources':len(pairs),'nutrition_point_improved_sources':int((pairs.scaled_log_mae_candidate<pairs.scaled_log_mae_baseline).sum()),
        'saved_group_error_sha256':hashes,'comparison_sha256':digest(args.comparison_dir/'summary.json'),
        'script_sha256':digest(Path(__file__)),'data_sha256':digest(data.root/'manifest.json'),'complete_test_opened':False,
        'scope':'Post-result auxiliary tradeoff diagnostic; full main metric and checkpoint selection remain fixed. Whole-food-group uncertainty excludes seeds/selection/labels. Axis intervals are unadjusted multiple comparisons. Source scores have different coverage and do not estimate causal source effects.'})
    print(intervals);print(support)


if __name__=='__main__':main()
