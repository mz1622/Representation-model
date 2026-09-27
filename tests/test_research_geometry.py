import numpy as np
import pytest
from foodcomp.research_geometry import nearest_other_rows,neighbor_overlap


def test_knn_excludes_self_preserves_duplicate_vectors_and_stable_ties():
    x=np.array([[0.,0.],[0.,0.],[1.,0.],[-1.,0.]])
    indices,distance=nearest_other_rows(x,[0,2],3,batch_size=1)
    np.testing.assert_array_equal(indices,[[1,2,3],[0,1,3]])
    np.testing.assert_array_equal(distance,[[0.,1.,1.],[1.,1.,4.]])
    np.testing.assert_array_equal(neighbor_overlap(indices,indices,3),np.ones(2))


def test_full_orthogonal_projection_preserves_neighbor_distances():
    rng=np.random.default_rng(17);x=rng.normal(size=(37,6));rotation,_=np.linalg.qr(rng.normal(size=(6,6)))
    a,da=nearest_other_rows(x,[0,6,23],10)
    b,db=nearest_other_rows((x-x.mean(0))@rotation,[0,6,23],10)
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(da,db,rtol=1e-12,atol=1e-12)
    x[0,0]=float("nan")
    with pytest.raises(ValueError,match="Finite"):nearest_other_rows(x,[0],10)
