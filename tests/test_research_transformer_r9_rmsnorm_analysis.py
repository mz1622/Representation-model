"""Fail-closed structural controls, decision boundaries and error decomposition."""
import importlib.util
from pathlib import Path
import pytest
from test_research_transformer_r9_drop25_analysis import partitions

path=Path(__file__).resolve().parents[1]/'scripts/analyze_foodnutrigpt_r9_rmsnorm.py'
spec=importlib.util.spec_from_file_location('rmsnorm_analysis',path)
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    parent=dict(seed=20260922,training_tasks=337048,observed_target_cells=1828536,
        data_hash='d',panel_hash='p',name_cache_hash='n',parameter_count=1548594,
        trainable_parameter_count=1515929,spec=dict(name='control',dropout=.15,
        objective='mae',learning_rate=.0003,d_model=192))
    spec=dict(parent['spec'],name='rmsnorm',architecture='encoder_rmsnorm_v1')
    candidate=dict(parent,spec=spec,kind='transformer_rmsnorm',
        numerical_recipe_changes=['encoder_normalization'],shared_initial_tensors_match_parent=True,
        parameter_count=1547250,trainable_parameter_count=1514585)
    return candidate,parent,spec


def test_declared_structure_passes():
    analysis.verify_method_control(*fixture())


@pytest.mark.parametrize('key,value',[('seed',20260923),('parameter_count',1548594),
    ('trainable_parameter_count',1515929),('panel_hash','changed'),
    ('shared_initial_tensors_match_parent',False),('kind','transformer_direct')])
def test_undeclared_changes_fail(key,value):
    candidate,parent,spec=fixture(); candidate[key]=value
    with pytest.raises(ValueError):analysis.verify_method_control(candidate,parent,spec)


@pytest.mark.parametrize('key,value',[('dropout',.25),('learning_rate',.0002),
    ('objective','mse'),('d_model',256)])
def test_no_banned_hyperparameter_change_can_hide_in_matching_spec(key,value):
    candidate,parent,spec=fixture(); spec[key]=value
    with pytest.raises(ValueError):analysis.verify_method_control(candidate,parent,spec)


@pytest.mark.parametrize('gain,lower,legacy,pass_all',[(.01,.001,-.02,True),
    (.01,0.,0.,False),(0.,.001,0.,False),(.01,.001,-.021,False)])
def test_screening_requires_all_predeclared_conditions(gain,lower,legacy,pass_all):
    value={'paired_intervals':{'scaled_log_mae':{'relative_improvement':gain,
        'relative_improvement_95_interval':[lower,.1]},'log_mae':{'relative_improvement':legacy}}}
    assert all(analysis.screening_gates(value).values()) is pass_all


def test_partition_contributions_sum_to_actual_delta():
    frame,result=analysis.partition_change(*partitions(),-.02)
    assert frame.rmsnorm_minus_mae_mae.sum()==pytest.approx(-.02)
    assert sum(result.values())==pytest.approx(-.02)


def test_no_nonfinite_interval_is_silently_accepted():
    with pytest.raises(ValueError):
        analysis.screening_gates({'paired_intervals':{
            'scaled_log_mae':{'relative_improvement':float('nan'),'relative_improvement_95_interval':[0.,1.]},
            'log_mae':{'relative_improvement':0.}}})
