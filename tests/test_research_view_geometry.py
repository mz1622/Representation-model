import numpy as np
import pytest
from foodcomp.research_view_geometry import geometry_summary


def test_identical_constant_direction_has_zero_spread_and_distance_despite_amplitude():
    a=np.array([[1.,2.],[2.,4.],[3.,6.]])
    r=geometry_summary(a,2*a,np.ones(3),3,3)
    assert r["uniform_task_view_distance"]==pytest.approx(0.)
    assert r["A"]["unit_mean_squared_distance_from_mean"]==pytest.approx(0.)
    assert r["A"]["unit_mean_vector_norm"]==pytest.approx(1.)


def test_known_orthogonal_geometry_and_axis_weighting():
    a=np.eye(2);b=np.array([[0.,1.],[0.,1.]])
    r=geometry_summary(a,b,np.array([1.,3.]),4,2)
    assert r["uniform_task_view_distance"]==pytest.approx(.5)
    assert r["axis_weighted_mean_view_distance"]==pytest.approx(.25)
    assert r["full_panel_consistency_estimate"]==pytest.approx(1.)
    assert r["full_panel_weight_mass_estimate"]==pytest.approx(4.)
    assert r["A"]["mean_unit_feature_population_std"]==pytest.approx(.5)
    assert r["A"]["unit_mean_squared_distance_from_mean"]==pytest.approx(.5)


def test_zero_vectors_are_reported_and_nonfinite_fails():
    r=geometry_summary(np.zeros((3,2)),np.zeros((3,2)),np.ones(3),3,1)
    assert r["A"]["zero_norm_fraction"]==1.
    assert r["uniform_task_view_distance"]==0.
    with pytest.raises(FloatingPointError):geometry_summary(np.full((3,2),np.nan),np.zeros((3,2)),np.ones(3),3,1)
    with pytest.raises(ValueError):geometry_summary(np.ones((3,2)),np.ones((3,2)),np.zeros(3),3,1)
