"""Source/observation strata and paired visibility on identical query profiles."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json
from foodcomp.research_statistics import paired_group_rates


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);summaries=[];source=[];observed=[]
    metrics=['mrr','recall_at_1','recall_at_5','recall_at_10']
    for run in args.run_dir:
        meta=json.loads((run/'metrics.json').read_text());assert not meta['complete_test_opened']
        ranks=pd.read_parquet(run/'ranks.parquet')
        if not np.isfinite(ranks['rank']).all() or not ranks['rank'].between(1,meta['candidate_count']).all():raise ValueError('Invalid ranks.')
        a=ranks[ranks.visible_fraction.eq(1.)];b=ranks[ranks.visible_fraction.eq(.3)]
        common=np.intersect1d(a.profile_index,b.profile_index);assert len(common)==len(b)
        groups={}
        for key,frame in [('full',a),('partial',b)]:
            x=frame[frame.profile_index.isin(common)]
            groups[key]=x.groupby(['exact_name_group_id','source_key'])[metrics].mean().groupby('exact_name_group_id').mean().reset_index()
        pairs=paired_group_rates(groups['full'],groups['partial'],metrics)
        for (fraction,source_key),g in ranks.groupby(['visible_fraction','source_key']):
            mean=g.groupby('exact_name_group_id')[metrics].mean().mean()
            source.append({'run':run.name,'visible_fraction':fraction,'source_key':source_key,'query_profiles':len(g),'candidate_groups':g.exact_name_group_id.nunique(),**mean.to_dict()})
        for fraction,g in ranks.groupby('visible_fraction'):
            g=g.copy();g['bin']=pd.cut(g.observed_axes,[2,4,14,39,np.inf],labels=['3_to_4','5_to_14','15_to_39','40plus'])
            for label,x in g.groupby('bin',observed=True):
                mean=x.groupby(['exact_name_group_id','source_key'])[metrics].mean().groupby('exact_name_group_id').mean().mean()
                observed.append({'run':run.name,'visible_fraction':fraction,'observed_axes_bin':str(label),'query_profiles':len(x),'candidate_groups':x.exact_name_group_id.nunique(),**mean.to_dict()})
        summaries.append({'run':str(run),'metrics_sha256':digest(run/'metrics.json'),'ranks_sha256':digest(run/'ranks.parquet'),
            'common_query_profiles':len(common),'common_candidate_groups':len(groups['full']),
            'partial_minus_full_same_query_comparisons':pairs,
            'scope':'Post-result controlled input-removal diagnostic on the same eligible query profiles/groups, fixed model and name bank. Lost values and changed observedness pattern both vary; cannot separate those mechanisms or extrapolate to excluded profiles.'})
    pd.DataFrame(source).to_csv(args.output_dir/'source_metrics.csv',index=False)
    pd.DataFrame(observed).to_csv(args.output_dir/'observed_axes_metrics.csv',index=False)
    write_json(args.output_dir/'summary.json',{'status':'complete','models':summaries,'complete_test_opened':False,'script_sha256':digest(Path(__file__)),
        'scope':'Source/observation strata have different coverage. Same-query full/partial contrast is separately restricted to identical profile IDs; does not modify primary evaluation or selection.'})
    print([(x['run'],x['partial_minus_full_same_query_comparisons']['mrr']) for x in summaries])


if __name__=='__main__':main()
