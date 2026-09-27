"""Invalid seed evidence must fail before any stability claim is produced."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("tree_seeds", Path(__file__).resolve().parents[1] / "scripts/summarize_foodnutrigpt_tree_seeds.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    fingerprint = {"data_manifest_sha256": "data", "protocol_sha256": "protocol",
                   "jobs_sha256": "jobs", "name_cache_manifest_sha256": "name"}
    registry = [{"name": f"xgb_{s}", "method": "xgb", "seed": s, "trees": 800} for s in module.SEEDS]
    manifests = [{"status": "complete", "complete_test_opened": False, "training_row_cap": None,
                  **fingerprint, "configuration": {"method": "xgb", "mode": "completion", "seed": s,
                    "trees": 800, "max_depth": 10, "output_dir": str(s)}, "code_hashes": {"train": "same"}}
                 for s in module.SEEDS]
    return manifests, registry, fingerprint


def test_independent_order_same_configuration_and_only_seed_changes():
    manifests, registry, fingerprint = fixture()
    settings, code = module.validate_manifests(manifests[::-1], "xgb", registry, fingerprint)
    assert "seed" not in settings and settings["trees"] == 800 and code == {"train": "same"}


@pytest.mark.parametrize("change", ["duplicate", "partial", "code", "config", "registry", "inputs", "cap", "test"])
def test_invalid_evidence_is_rejected(change):
    manifests, registry, fingerprint = fixture()
    if change == "duplicate": manifests[-1] = copy.deepcopy(manifests[0])
    elif change == "partial": manifests[-1]["status"] = "running"
    elif change == "code": manifests[-1]["code_hashes"]["train"] = "changed"
    elif change == "config": manifests[-1]["configuration"]["max_depth"] = 12
    elif change == "registry":
        for manifest in manifests: manifest["configuration"]["trees"] = 100
    elif change == "inputs": manifests[-1]["name_cache_manifest_sha256"] = "enriched"
    elif change == "cap": manifests[-1]["training_row_cap"] = 5000
    elif change == "test": manifests[-1]["complete_test_opened"] = True
    with pytest.raises(ValueError):
        module.validate_manifests(manifests, "xgb", registry, fingerprint)
