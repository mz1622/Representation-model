import numpy as np
import pytest
import torch
from foodcomp.research_context_dropout import context_dropout_masks,drop_context


def test_deterministic_nested_masks_and_default_no_rng():
    np.random.seed(11);state=np.random.get_state()
    assert context_dropout_masks(30,10,'none',7,1)==(None,None)
    mixed,ids=context_dropout_masks(10000,30,'mix_30_60_90',7,1)
    fixed,_=context_dropout_masks(10000,30,'fixed_30',7,1)
    assert not (fixed&~mixed).any()
    np.testing.assert_array_equal(mixed,context_dropout_masks(10000,30,'mix_30_60_90',7,1)[0])
    assert not np.array_equal(mixed,context_dropout_masks(10000,30,'mix_30_60_90',7,2)[0])
    for i,rate in enumerate([.3,.6,.9]):assert abs(mixed[ids==i].mean()-rate)<.02
    after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);assert state[2:]==after[2:]


def test_only_visibility_changes_missing_zero_family_distinction():
    batch={'masked':torch.tensor([[False,True,True,False]]),'value':torch.tensor([[0.,0.,5.,2.]]),
        'target':torch.tensor([[False,False,True,False]]),'cell_weight':torch.ones(1,4)}
    assert drop_context(batch,None) is batch
    result=drop_context(batch,np.array([[False,False,False,True]]))
    assert result['masked'].tolist()==[[False,True,True,True]]
    # Explicit zero retained, missing and target-family still hidden, removed context not a new label.
    for key in batch:
        if key!='masked':assert result[key] is batch[key]
    assert batch['masked'].tolist()==[[False,True,True,False]]
    with pytest.raises(ValueError):drop_context(batch,np.ones((1,4)))
    with pytest.raises(ValueError):context_dropout_masks(2,2,'unsupported',7,1)
