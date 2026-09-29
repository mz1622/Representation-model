"""Post-hoc numerical bridge: same learned LayerNorm weights, explicit execution only.

No optimizer or new checkpoint selection. This diagnoses the evaluation-kernel
difference noted before training; it cannot establish RMSNorm seed stability.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json,score_predictions
from foodcomp.research_neural import evaluate
from foodcomp.research_transformer_r9 import frozen_inputs,TransformerNutritionModel
from foodcomp.research_transformer_r9_rmsnorm import replace_encoder_norms


def main():
    out=ROOT/'reports/v9_r9_rmsnorm_bridge_v1'
    local=ROOT/'data/local/research_diagnostics/v9_r9_rmsnorm_bridge_v1'
    if out.exists() or local.exists():raise FileExistsError('Preserve bridge outputs')
    torch.set_num_threads(4)
    frozen,freeze=frozen_inputs(ROOT)
    run=ROOT/'output/v9_r9/tf192_mae_lr3e4_60'
    files=[run/'best_model.pt',run/'completion_predictions.parquet',run/'metrics.json',freeze,Path(__file__)]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    model=TransformerNutritionModel(run/'best_model.pt',ROOT)
    initial={k:v.detach().clone() for k,v in model.model.state_dict().items()}
    fastpath=torch.backends.mha.get_fastpath_enabled()
    replace_encoder_norms(model.model,rms=False)
    model.model.eval()
    assert all(torch.equal(v,model.model.state_dict()[k]) for k,v in initial.items())
    started=time.monotonic()
    predictions=evaluate(model.model,model.data,model._cached_text,model.device)
    scores,axes,_=score_predictions(model.data,predictions)
    baseline=pd.read_parquet(run/'completion_predictions.parquet')
    original_scores,_,_=score_predictions(model.data,baseline)
    assert original_scores==json.loads((run/'metrics.json').read_text(encoding='utf-8'))['completion']
    joined=predictions.merge(baseline,on=['profile_index','axis_index'],suffixes=('_bridge','_saved'),validate='one_to_one')
    assert len(joined)==323809
    scales=model.data.scale[joined.axis_index.to_numpy()]
    differences=np.log1p(joined.prediction_bridge/scales)-np.log1p(joined.prediction_saved/scales)
    assert np.isfinite(differences).all()
    assert torch.backends.mha.get_fastpath_enabled()==fastpath
    assert all(torch.equal(v,model.model.state_dict()[k]) for k,v in initial.items())
    local.mkdir(parents=True);out.mkdir(parents=True)
    pp=local/'parent_explicit_layernorm_completion_predictions.parquet'
    predictions.to_parquet(pp,index=False)
    axes.to_csv(out/'axis_metrics.csv',index=False)
    delta=scores['nutrition']['scaled_log_mae']-original_scores['nutrition']['scaled_log_mae']
    result={'status':'complete_learned_layernorm_execution_bridge','primary_bridge_minus_original':delta,
       'maximum_cell_scaled_prediction_absolute_difference':float(np.abs(differences).max()),
       'mean_cell_scaled_prediction_absolute_difference':float(np.abs(differences).mean()),
       'baseline_scores':original_scores,'bridge_scores':scores,'input_hashes':hashes,
       'prediction_sha256':digest(pp),'prediction_path':pp.relative_to(ROOT).as_posix(),
       'learned_weights_unchanged':True,'formal_global_backend_unchanged':True,
       'elapsed_seconds':time.monotonic()-started,'training_performed':False,'model_selection_performed':False,
       'data_modified':False,'baseline_refit':False,'complete_test_opened':False,
       'scope':'Post-hoc evaluation-only bridge on the existing selected parent; isolates fused versus explicit LayerNorm execution with identical learned weights. Does not isolate individual RMSNorm ingredients, training trajectories, or seed variation.'}
    frozen_inputs(ROOT)
    for name,expected in hashes.items():assert digest(ROOT/name)==expected
    write_json(out/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['status','primary_bridge_minus_original','maximum_cell_scaled_prediction_absolute_difference','elapsed_seconds']}))

if __name__=='__main__':main()
