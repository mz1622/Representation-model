from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from foodcomp.research_conditioning import fit_training_name_statistics
from foodcomp.research_conditioning import FrozenNameStandardizer
from foodcomp.research_neural import make_model, batch_from_arrays
import torch


def fixture():
    data = SimpleNamespace(train=np.array([0, 1, 2]), profiles=pd.DataFrame({
        "partition": ["train", "train", "train", "validation"],
        "original_name": ["a", "a", "b", "held out"],
    }))
    return data, np.array([[1., 2.], [1., 2.], [3., 6.], [900., -999.]], np.float32)


def test_statistics_use_unique_train_names_and_ignore_held_out_vectors():
    data, text = fixture()
    mean, std, meta = fit_training_name_statistics(data, text)
    np.testing.assert_array_equal(mean, [2., 4.])
    np.testing.assert_array_equal(std, [1., 2.])
    assert meta["train_unique_names"] == 2
    text[-1] = np.nan
    after = fit_training_name_statistics(data, text)
    np.testing.assert_array_equal(mean, after[0])
    np.testing.assert_array_equal(std, after[1])
    assert meta == after[2]


def test_conditioning_rejects_held_out_fit_and_inconsistent_duplicates():
    data, text = fixture()
    data.train = np.array([0, 2, 3])
    with pytest.raises(ValueError, match="Held-out"):
        fit_training_name_statistics(data, text)
    data.train = np.array([0, 1, 2])
    text[1, 0] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        fit_training_name_statistics(data, text)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_features_fail(bad):
    data, text = fixture()
    text[0, 0] = bad
    with pytest.raises(FloatingPointError):
        fit_training_name_statistics(data, text)


def test_constant_coordinate_is_not_silently_normalized():
    data, text = fixture()
    text[:3, 0] = 1
    with pytest.raises(ValueError, match="Degenerate"):
        fit_training_name_statistics(data, text)


def test_standardizer_is_frozen_and_cannot_run_before_fitting():
    transform = FrozenNameStandardizer(2)
    with pytest.raises(ValueError, match="not been fitted"):
        transform(torch.ones(1, 2))
    transform.set_statistics(np.array([2., 4.]), np.array([1., 2.]))
    transform.validate()
    torch.testing.assert_close(transform(torch.tensor([[3., 6.]])), torch.ones(1, 2))
    assert len(list(transform.parameters())) == 0
    with pytest.raises(ValueError, match="already frozen"):
        transform.set_statistics(np.zeros(2), np.ones(2))


def test_conditioned_model_preserves_initial_parameters_rng_and_hidden_label_isolation(tmp_path):
    data, _ = fixture()
    data.axes = list(range(4))
    torch.manual_seed(20260922)
    parent, _ = make_model(data, 2, "mlp", mlp_width=32)
    rng = torch.get_rng_state()
    torch.manual_seed(20260922)
    candidate, _ = make_model(data, 2, "mlp", mlp_width=32, mlp_text_conditioning="train_unique_name")
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    assert set(candidate.state_dict()) - set(parent.state_dict()) == {
        "name_standardizer.mean", "name_standardizer.std", "name_standardizer.fitted"}
    candidate.name_standardizer.set_statistics([2., 4.], [1., 2.])
    batch = batch_from_arrays(np.array([[0., 2., 3., 4.]]), np.array([[True, False, False, True]]),
                              np.array([[3., 6.]]), "cpu")
    candidate.eval()
    reference = candidate(batch)["amount_normalized"]
    batch["value"][batch["masked"]] = 1e5
    batch["target"] = torch.ones((1, 4), dtype=torch.bool)
    batch["source"] = torch.tensor([99])
    torch.testing.assert_close(reference, candidate(batch)["amount_normalized"], rtol=0, atol=0)
    reference.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    assert candidate.name_standardizer.mean.grad is None
    torch.save(candidate.state_dict(), tmp_path/"model.pt")
    loaded, _ = make_model(data, 2, "mlp", mlp_width=32, mlp_text_conditioning="train_unique_name")
    loaded.load_state_dict(torch.load(tmp_path/"model.pt", weights_only=True))
    loaded.name_standardizer.validate()
    loaded.eval()
    torch.testing.assert_close(reference, loaded(batch)["amount_normalized"], rtol=0, atol=0)
