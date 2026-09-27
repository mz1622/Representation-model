from types import SimpleNamespace
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_neural import evaluate
from foodcomp.research_selection import evaluate_dual_selection


class FixedHurdle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, batch):
        known = torch.where(batch["masked"], 0., batch["value"]).sum(1, keepdim=True)
        amount = known/10 + self.offset + batch["axis"].float()/2
        return {"amount_normalized": amount, "positive_logit": amount - 1.}


def data_fixture():
    values = np.array([[0, 2, 7], [2, 0, 9], [4, 0, 11], [6, 6, 13]], np.float32)
    observed = np.ones_like(values, bool); observed[1, 1] = False
    rows, axes = np.where(observed[:, :2])
    return SimpleNamespace(values=values, observed=observed, validation=np.arange(4),
        axes=list(range(3)), targets=np.array([0, 1]), scale=np.ones(3),
        families=np.array(["a", "b", "a"]),
        profiles=pd.DataFrame({"profile_id": ["p0", "p1", "p2", "p3"],
            "source_key": ["x", "x", "y", "x"], "exact_name_group_id": ["g", "g", "g", "h"]}),
        jobs=pd.DataFrame({"profile_index": rows, "axis_index": axes, "mask_family": np.array(["a", "b"])[axes]}))


def test_same_predictions_rng_weights_and_batch_invariance():
    data = data_fixture(); text = np.ones((4, 2), np.float32); model = FixedHurdle()
    rng = torch.get_rng_state().clone(); before = copy.deepcopy(model.state_dict())
    pred, metric = evaluate_dual_selection(model, data, text, "cpu", amount_weight=2., batch_size=1)
    pred2, metric2 = evaluate_dual_selection(model, data, text, "cpu", amount_weight=2., batch_size=3)
    baseline = evaluate(model, data, text, "cpu")
    keys = ["profile_index", "axis_index"]
    for frame in [pred2, baseline]:
        pd.testing.assert_frame_equal(pred.sort_values(keys).reset_index(drop=True), frame.sort_values(keys).reset_index(drop=True))
    assert metric == pytest.approx(metric2)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key in before:torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    # Independent calculation: equal candidate totals, equal sources, equal
    # profiles within a source. Missing row1/axis1 never gets a target or weight.
    presence_means = []; amount_means = []
    for a, rows, weights in [(0, [0, 1, 2, 3], [.25, .25, .5, 1]), (1, [0, 2, 3], [.5, .5, 1])]:
        known = data.observed[rows] & (data.families != data.families[a])
        predicted = np.where(known, data.values[rows], 0).sum(1).astype(float)/10 + .25 + a/2
        y = data.values[rows, a].astype(float); positive = y > 0
        logit = predicted-1; error = np.abs(predicted-y)
        presence_means.append(np.average(np.logaddexp(0, logit)-positive*logit, weights=weights))
        amount_means.append(np.average(np.where(error < 1, .5*error**2, error-.5)*positive, weights=weights))
    assert metric["source_free_hurdle"] == pytest.approx(np.mean(presence_means)+2*np.mean(amount_means), rel=1e-7)


def test_unobserved_labels_never_enter_selector_or_prediction():
    data = data_fixture(); model = FixedHurdle(); text = np.ones((4, 2), np.float32)
    pred, metric = evaluate_dual_selection(model, data, text, "cpu")
    data.values[1, 1] = 99999  # absent value; no fabricated zero target
    changed, changed_metric = evaluate_dual_selection(model, data, text, "cpu")
    pd.testing.assert_frame_equal(pred, changed)
    assert metric == changed_metric


def test_selector_rejects_bad_jobs_and_nonfinite_outputs():
    data = data_fixture(); text = np.ones((4, 2), np.float32); model = FixedHurdle()
    invalid = copy.deepcopy(data); invalid.jobs = pd.concat([invalid.jobs, invalid.jobs.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):evaluate_dual_selection(model, invalid, text, "cpu")
    invalid = copy.deepcopy(data); invalid.validation = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="validation profiles"):evaluate_dual_selection(model, invalid, text, "cpu")
    invalid = copy.deepcopy(data); invalid.observed[0, 0] = False
    with pytest.raises(ValueError, match="Missing cells"):evaluate_dual_selection(model, invalid, text, "cpu")
    with torch.no_grad():model.offset.fill_(float("nan"))
    with pytest.raises(FloatingPointError):evaluate_dual_selection(model, data, text, "cpu")
