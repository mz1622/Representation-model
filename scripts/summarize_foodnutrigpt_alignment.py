"""Aggregate R5 run status and retrieval metrics without original nutrition values."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json


def main():
    folder=ROOT/'experiments/foodnutrigpt_v9_research/r5';config=json.loads((folder/'config.json').read_text())
    rows=[]
    for entry in config['registered_candidates']:
        path=ROOT/entry.get('output_dir','output/v9_r5/'+entry['name'])
        row={'registered_configuration':entry,'local_output':str(path.relative_to(ROOT)),'status':'not_started'}
        if entry.get('reused_control'):
            if not all((path/f).exists() for f in ['metrics.json','ranks.parquet','candidate_names.json']):raise ValueError('Missing reused reference.')
            row.update(status='complete_reused_reference',metrics=json.loads((path/'metrics.json').read_text()),metrics_sha256=digest(path/'metrics.json'))
        elif (path/'run_manifest.json').exists():
            manifest=json.loads((path/'run_manifest.json').read_text())
            row.update(status=manifest['status'],manifest_sha256=digest(path/'run_manifest.json'),code_commit=manifest['code_commit'],
                elapsed_seconds=manifest.get('elapsed_seconds'),epoch_completed=manifest.get('epoch_completed'),best_epoch=manifest.get('best_epoch'),checkpoint_sha256=manifest.get('checkpoint_hash'))
            if manifest['status']=='complete':row.update(metrics=json.loads((path/'metrics.json').read_text()),metrics_sha256=digest(path/'metrics.json'))
            elif manifest['status']=='failed':row['failure']={key:manifest.get(key) for key in ['error_type','error']}
        rows.append(row)
    output={'version':'V9-R5','status':'exploration in progress; no accepted model; main completion goal still unmet',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':digest(folder/'config.json'),'runs':rows,
        'independent_model_completion_and_name_only':'N/A; separate specialist, no nutrition prediction head',
        'confirmation_completed':False,'complete_test_opened':False,'individual_predictions_included':False,
        'scope':'Manifest status only; inspect actual sessions/processes for liveness. Specialist ranking gains are not evidence of a completion gain.'}
    selection=ROOT/'output/v9_r5/ridge_selection_v1/selection.json'
    if selection.exists():output['ridge_selection']=json.loads(selection.read_text(encoding='utf-8-sig'))
    write_json(folder/'results_summary.json',output)
    print([(x['registered_configuration']['name'],x['status']) for x in rows])


if __name__=='__main__':main()
