"""Audit an untrained optional query residual on real training tasks before registration."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_neural import make_model
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, model_loss
from foodcomp.research_inference import NutritionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    result = {"status":"incomplete", "complete_test_opened":False, "script_sha256":digest(Path(__file__))}
    try:
        data = ResearchData(ROOT/"data/processed"/VERSION)
        text, cache = prepare_names(data, ROOT)
        panel = FamilyPanel(data, text, ROOT/"data/processed"/PANEL_VERSION, "cpu")
        ids = np.random.default_rng(20260923).choice(len(panel.rows), 64, replace=False)
        if not np.isin(panel.rows[ids], data.train).all():
            raise ValueError("Non-training audit task.")
        batch = panel.batch(ids)
        torch.manual_seed(20260922)
        parent, config = make_model(data, text.shape[1], "mlp", mlp_width=512)
        rng = torch.get_rng_state()
        torch.manual_seed(20260922)
        candidate, _ = make_model(data, text.shape[1], "mlp", mlp_width=512, mlp_query_residual=True)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        parent.eval(); candidate.eval()
        with torch.no_grad():
            torch.testing.assert_close(parent(batch)["amount_normalized"], candidate(batch)["amount_normalized"], rtol=0, atol=0)
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
        for model in [parent, candidate]:
            model_loss(model, batch, "mlp", config, "mae").backward()
        for (name, parameter) in parent.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(candidate.named_parameters())[name].grad, rtol=0, atol=0)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in candidate.parameters()):
            raise FloatingPointError("Absent or nonfinite query-model gradient.")
        count = sum(p.numel() for p in candidate.parameters())
        if count != 762365:
            raise ValueError("Unexpected candidate parameter count.")
        # Nonzero fixture avoids a vacuous hidden-label/loader test at zero initialization.
        with torch.no_grad():
            candidate.query_residual.joint[-1].weight.fill_(.01)
            reference = candidate(batch)["amount_normalized"]
            changed = dict(batch)
            changed["value"] = torch.where(batch["masked"], batch["value"]+999, batch["value"])
            changed["target"] = ~batch["target"]
            changed["positive"] = ~batch["positive"]
            changed["source"] = batch["source"]+99
            torch.testing.assert_close(reference, candidate(changed)["amount_normalized"], rtol=0, atol=0)
        fixture = out/"untrained_nonzero_loader_fixture.pt"
        saved = {"model_state":candidate.state_dict(), "kind":"mlp", "config":asdict(config), "text_dim":text.shape[1],
                 "data_root":str(data.root), "view":data.view, "name_cache":str(cache),
                 "data_hash":digest(data.root/"manifest.json"), "name_cache_hash":digest(cache/"manifest.json"),
                 "args":{"mlp_width":512, "mlp_query_residual":True}}
        torch.save(saved, fixture)
        loaded = NutritionModel(fixture, device="cpu")
        with torch.no_grad():
            torch.testing.assert_close(reference, loaded.model(batch)["amount_normalized"], rtol=0, atol=0)
        axes = data.axes.loc[data.axes.loss_eligible & data.axes.loss_group.eq("nutrition"), "axis_index"].to_numpy()
        names = data.profiles.iloc[data.train].original_name.drop_duplicates().iloc[:3].tolist()
        context = {int(axes[2]):0.}
        for observed in [{}, context]:
            prediction = loaded.predict(names[0], observed, axes[:2])
            if len(prediction) != 2 or not np.isfinite(list(prediction.values())).all():
                raise ValueError("Arbitrary unobserved-query API failure.")
        np.testing.assert_array_equal(loaded.encode(names[0], context, "nutrition"), loaded.encode(names[1], context, "nutrition"))
        ranking = loaded.retrieve_names(context, names, top_k=2)
        if len(ranking) != 2 or not all(np.isfinite(row["score"]) for row in ranking):
            raise ValueError("Retrieval API failure.")
        result.update(status="complete", training_tasks=64, parameter_count=count, learnable_parent_initialization_equal=True,
                      global_cpu_rng_equal=True, initial_predictions_and_parent_preclip_gradients_bitwise_equal=True,
                      nonzero_head_hidden_label_and_source_isolation=True, save_reload_bitwise_equal=True,
                      arbitrary_unobserved_axis_zero_context_and_three_api_methods_checked=True,
                      fixture_sha256=digest(fixture), data_sha256=saved["data_hash"], name_cache_sha256=saved["name_cache_hash"],
                      code_sha256={file:digest(ROOT/file) for file in ["src/foodcomp/research_neural.py", "src/foodcomp/research_query.py", "src/foodcomp/research_inference.py", "scripts/train_foodnutrigpt_v9_r1.py"]},
                      scope="Untrained real-input functional audit, not candidate training or quality evidence. No optimizer update. Final layer manually made nonzero only for isolation/loader/API checks. Global clipping and optimizer trajectories are not asserted equal.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out/"verification.json", result)
        raise
    write_json(out/"verification.json", result)
    print({k:result[k] for k in ["status", "parameter_count", "save_reload_bitwise_equal"]})


if __name__ == "__main__":
    main()
