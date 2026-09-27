"""Private deterministic success/failure case ranks; no nutritional values exported."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import VERSION,digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    root=ROOT/'data/processed'/VERSION
    manifest=json.loads((root/'manifest.json').read_text())
    if digest(root/'profiles.csv.gz')!=manifest['artifact_hashes']['profiles.csv.gz']:raise ValueError('Profile metadata changed.')
    profiles=pd.read_csv(root/'profiles.csv.gz',low_memory=False)
    frames=[]
    for tag,path in [('contrastive','output/v9_r5/mapper_contrastive60'),('name_mlp','output/v9_r0/retrieval_name_mlp_v1')]:
        f=pd.read_parquet(ROOT/path/'ranks.parquet')
        for fraction,g in f.groupby('visible_fraction'):
            frames.append(g[['profile_index','rank','observed_axes']].rename(columns={'rank':f'{tag}_{fraction}_rank','observed_axes':f'{tag}_{fraction}_observed'}))
    result=frames[0]
    for f in frames[1:]:result=result.merge(f,on='profile_index',validate='one_to_one')
    result=result.merge(profiles[['profile_index','original_name','source_key','exact_name_group_id','partition']],on='profile_index',validate='one_to_one')
    assert result.partition.eq('validation').all()
    result['full_rank_gain_over_name_mlp']=result['name_mlp_1.0_rank']-result['contrastive_1.0_rank']
    result['partial_rank_degradation']=result['contrastive_0.3_rank']-result['contrastive_1.0_rank']
    groups=[('full_success_vs_reference',result[result['contrastive_1.0_rank']<=10].sort_values(['full_rank_gain_over_name_mlp','profile_index'],ascending=[False,True]).head(12)),
        ('partial_failure_after_full_success',result[result['contrastive_1.0_rank']<=10].sort_values(['partial_rank_degradation','profile_index'],ascending=[False,True]).head(12)),
        ('full_failure',result.sort_values(['contrastive_1.0_rank','profile_index'],ascending=[False,True]).head(12))]
    selected=pd.concat([g.assign(case_type=tag) for tag,g in groups],ignore_index=True)
    args.output_dir.mkdir(parents=True);selected.to_csv(args.output_dir/'private_case_ranks.csv',index=False)
    write_json(args.output_dir/'summary.json',{'status':'complete','case_rows':len(selected),'common_profiles':len(result),
        'selection':'Post-result descriptive extremes; not a representative sample, scientific error rate or confirmation of food identity. Whole-table performance is reported separately.',
        'no_original_nutrition_values_exported':True,'private_case_sha256':digest(args.output_dir/'private_case_ranks.csv'),
        'script_sha256':digest(Path(__file__)),'complete_test_opened':False})


if __name__=='__main__':main()
