"""History/provenance checks for the new statistics adapter; no model training."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1]/'scripts/confirm_foodnutrigpt_r9_fixed_references.py'
spec = importlib.util.spec_from_file_location('r9_statistics',path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def history_fixture():
    spec = {'seed':20260923,'epochs':3,'schedule_epochs':3,'learning_rate':.0003}
    manifest = {'spec':spec,'training_tasks':10,'observed_target_cells':15,
                'best_epoch':2,'best_primary':.2}
    rows=[]
    for epoch,value in enumerate([.3,.2,.2],1):
        rows.append({'epoch':epoch,'validation_primary':value,'train_loss':value,
            'training_tasks':10,'observed_target_cells':15,
            'learning_rate':.0003*(.01+.99*(1+np.cos(np.pi*(epoch-1)/3))/2),
            'training_order_sha256':module.fingerprint_array(np.random.default_rng(20260923+epoch).permutation(10))})
    return manifest,pd.DataFrame(rows)


def test_complete_history_preserves_earliest_tie_rule():
    manifest,history=history_fixture()
    module.validate_history(manifest,history)
    manifest['best_epoch']=3
    with pytest.raises(ValueError,match='selection'): module.validate_history(manifest,history)


@pytest.mark.parametrize('change',['short','nonfinite','tasks','targets','order','schedule','score'])
def test_corrupt_history_rejected(change):
    manifest,history=history_fixture()
    if change=='short': history=history.iloc[:-1]
    elif change=='nonfinite': history.loc[0,'train_loss']=np.nan
    elif change=='tasks': history.loc[0,'training_tasks']=9
    elif change=='targets': history.loc[0,'observed_target_cells']=14
    elif change=='order': history.loc[0,'training_order_sha256']='wrong'
    elif change=='schedule': history.loc[0,'learning_rate']=.0001
    elif change=='score': manifest['best_primary']=.1
    with pytest.raises((ValueError,AssertionError)): module.validate_history(manifest,history)


def test_paths_reuse_seed22_and_match_registered_new_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    plan={'parent_run':'output/v9_r9/recipe','parent_audit':'reports/parent/verification.json'}
    runs,audits=module.locations(plan,{'candidate':'recipe'})
    assert runs[20260922]==tmp_path/'output/v9_r9/recipe'
    assert runs[20260923]==tmp_path/'output/v9_r9_confirmation/recipe_seed20260923'
    assert audits[20260924]==tmp_path/'reports/v9_r9_confirmation_seed20260924_audit_v1/verification.json'
