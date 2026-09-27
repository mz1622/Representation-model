"""Check the registered name-conditioning model on actual training inputs before launch."""
from dataclasses import asdict
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, model_loss
from foodcomp.research_neural import make_model
from foodcomp.research_inference import NutritionModel
from foodcomp.research_conditioning import fit_training_name_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    out = parser.parse_args().output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    result = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    try:
        data = ResearchData(ROOT / "data/processed" / VERSION)
        text, cache = prepare_names(data, ROOT)
        panel = FamilyPanel(data, text, ROOT / "data/processed" / PANEL_VERSION, "cpu")
        indices = np.random.default_rng(20260923).choice(len(panel.rows), 64, replace=False)
        if not np.isin(panel.rows[indices], data.train).all():
            raise ValueError("Non-training audit task.")
        batch = panel.batch(indices)
        torch.manual_seed(20260922)
        parent, _ = make_model(data, text.shape[1], "mlp", mlp_width=512)
        rng = torch.get_rng_state()
        torch.manual_seed(20260922)
        model, config = make_model(data, text.shape[1], "mlp", mlp_width=512, mlp_text_conditioning="train_unique_name")
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        mean, std, statistics = fit_training_name_statistics(data, text)
        if statistics["statistics_sha256"] != "523a5eaf9f19291fa7fd22e0661b2489b1ffd0a35ba35755a7b98d5fed0afb4f":
            raise ValueError("Train statistics differ from preregistration.")
        model.name_standardizer.set_statistics(mean, std)
        if sum(p.numel() for p in model.parameters()) != 667900:
            raise ValueError("Unexpected learnable parameter count.")
        model.eval()
        with torch.no_grad():
            reference = model(batch)["amount_normalized"]
            changed = dict(batch)
            changed["value"] = torch.where(batch["masked"], batch["value"] + 999, batch["value"])
            changed["target"] = ~batch["target"]
            changed["positive"] = ~batch["positive"]
            changed["source"] = batch["source"] + 99
            torch.testing.assert_close(reference, model(changed)["amount_normalized"], rtol=0, atol=0)
        loss = model_loss(model, batch, "mlp", config, "mae")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing or nonfinite gradient.")
        if any(buffer.grad is not None for buffer in model.name_standardizer.buffers()):
            raise ValueError("Conditioning buffers must not receive gradients.")
        fixture = out / "untrained_loader_fixture.pt"
        saved = {"model_state": model.state_dict(), "kind": "mlp", "config": asdict(config), "text_dim": text.shape[1],
                 "data_root": str(data.root), "view": data.view, "name_cache": str(cache),
                 "data_hash": digest(data.root/"manifest.json"), "name_cache_hash": digest(cache/"manifest.json"),
                 "args": {"mlp_width": 512, "mlp_text_conditioning": "train_unique_name"}}
        torch.save(saved, fixture)
        loaded = NutritionModel(fixture, device="cpu")
        with torch.no_grad():
            torch.testing.assert_close(reference, loaded.model(batch)["amount_normalized"], rtol=0, atol=0)
        names = data.profiles.iloc[data.train].original_name.drop_duplicates().iloc[:3].tolist()
        nutrition_axes = data.axes.loc[data.axes.loss_eligible & data.axes.loss_group.eq("nutrition"), "axis_index"].to_numpy()
        zero_context = {int(nutrition_axes[2]): 0.}
        for observed in [{}, zero_context]:
            prediction = loaded.predict(names[0], observed, nutrition_axes[:2])
            if len(prediction) != 2 or not np.isfinite(list(prediction.values())).all():
                raise ValueError("Arbitrary unobserved-query API failed.")
        _, missing = loaded.profile_arrays({})
        _, zero = loaded.profile_arrays(zero_context)
        if missing.any() or not zero[0, nutrition_axes[2]]:
            raise ValueError("Explicit zero confused with missing context.")
        np.testing.assert_array_equal(loaded.encode(names[0], zero_context, "nutrition"),
                                      loaded.encode(names[1], zero_context, "nutrition"))
        if loaded.encode(names[0], {}, "name").shape != (512,):
            raise ValueError("Unexpected representation shape.")
        ranking = loaded.retrieve_names(zero_context, names, top_k=2)
        if len(ranking) != 2 or not all(np.isfinite(item["score"]) for item in ranking):
            raise ValueError("Names-only candidate retrieval API failed.")
        for name, value in model.name_standardizer.state_dict().items():
            torch.testing.assert_close(value, loaded.model.name_standardizer.state_dict()[name], rtol=0, atol=0)
        result.update(status="complete", parameter_count=667900, training_tasks=64, statistics=statistics,
                      learnable_initialization_and_rng_equal=True, hidden_values_labels_source_isolated=True,
                      finite_gradients_and_frozen_statistics=True, save_reload_bitwise_equal=True,
                      arbitrary_query_zero_missing_and_three_api_methods_checked=True,
                      nutrition_encoding_independent_of_name=True, initial_loss=float(loss.detach()),
                      fixture_sha256=digest(fixture), data_sha256=saved["data_hash"], name_cache_sha256=saved["name_cache_hash"],
                      code_sha256={f:digest(ROOT/f) for f in ["src/foodcomp/research_conditioning.py", "src/foodcomp/research_neural.py", "src/foodcomp/research_inference.py", "scripts/train_foodnutrigpt_v9_r1.py"]},
                      scope="Untrained functional fixture on actual training inputs. No optimizer update or held-out model-selection evaluation. Save/reload uses serialized statistics, never refitting.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out/"verification.json", result)
        raise
    write_json(out/"verification.json", result)
    print({k:result[k] for k in ["status", "parameter_count", "save_reload_bitwise_equal", "initial_loss"]})


if __name__ == "__main__":
    main()
