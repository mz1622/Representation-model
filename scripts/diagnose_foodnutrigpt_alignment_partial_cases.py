"""Private deterministic cases for the training-view intervention, never representative."""
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
    root=ROOT/'data/processed'/VERSION;manifest=json.loads((root/'manifest.json').read_text())
    assert digest(root/'profiles.csv.gz')==manifest['artifact_hashes']['profiles.csv.gz']
    profiles=pd.read_csv(root/'profiles.csv.gz',low_memory=False);frames=[]
    for tag,name in [('parent','mapper_contrastive60'),('partial','mapper_contrastive60_partial05')]:
        f=pd.read_parquet(ROOT/'output/v9_r5'/name/'ranks.parquet')
        for fraction,g in f.groupby('visible_fraction'):
            frames.append(g[['profile_index','rank']].rename(columns={'rank':f'{tag}_{fraction}'}))
    result=frames[0]
    for f in frames[1:]:result=result.merge(f,on='profile_index',validate='one_to_one')
    result=result.merge(profiles[['profile_index','original_name','source_key','exact_name_group_id','partition']],on='profile_index',validate='one_to_one')
    assert result.partition.eq('validation').all()
    result['partial_gain']=result['parent_0.3']-result['partial_0.3']
    result['full_loss']=result['partial_1.0']-result['parent_1.0']
    cases=[('partial_success',result[result['partial_0.3']<=10].sort_values(['partial_gain','profile_index'],ascending=[False,True]).head(12)),
        ('full_regression',result[result['parent_1.0']<=10].sort_values(['full_loss','profile_index'],ascending=[False,True]).head(12)),
        ('persistent_partial_failure',result.sort_values(['partial_0.3','profile_index'],ascending=[False,True]).head(12))]
    selected=pd.concat([g.assign(case_type=tag) for tag,g in cases],ignore_index=True)
    args.output_dir.mkdir(parents=True);selected.to_csv(args.output_dir/'private_case_ranks.csv',index=False)
    write_json(args.output_dir/'summary.json',{'status':'complete','case_rows':len(selected),'common_profiles':len(result),
        'selection':'12 deterministic extreme cases per type, not representative. No identity or causal inference from names.',
        'private_case_sha256':digest(args.output_dir/'private_case_ranks.csv'),'script_sha256':digest(Path(__file__)),
        'complete_test_opened':False,'original_nutrition_values_exported':False})


if __name__=='__main__':main()
