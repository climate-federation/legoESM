"""Controls for the faithful-card catalog reachability gate."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROBE = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226" /
         "recipe_transfer_identity.py")
SPEC = importlib.util.spec_from_file_location("recipe_transfer_identity", PROBE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_faithful_catalog_path_is_exact_and_controls_fire():
    result = MODULE.run_identity()

    assert result["verdict"] == "PASS"
    assert result["identity_differences"] == []
    assert result["ownership"]["collisions"] == []
    assert result["ownership"]["missing"] == []
    assert result["ownership"]["extra"] == []
    assert result["negative_control"]["difference_count"] > 0
    assert result["plant"] == {
        "field": "asselin_gamma",
        "expected_path": "config.asselin_gamma",
        "difference_paths": ["config.asselin_gamma"],
        "fired": True,
    }
    assert result["ownership_collision_plant"]["fired"] is True
    assert "outer_integrator" in result["ownership_collision_plant"]["message"]


def _behavior_artifact(path, source, catalog_recipe, value=1.0):
    config = {
        "recipe": "nemo_dino_kamm_mlf",
        "config_source": source,
        "catalog_recipe": catalog_recipe,
        "n_days": 5,
    }
    np.savez(
        path,
        config_source=np.str_(source),
        catalog_recipe=np.str_(catalog_recipe),
        run_config=np.str_(json.dumps(config)),
        prognostic_initial_state_sha256=np.str_("initial"),
        prognostic_final_state_sha256=np.str_("final"),
        T3d_day5=np.asarray([value], dtype=np.float64),
    )


def test_behavior_comparison_accepts_only_declared_source_stamp_difference(tmp_path):
    oracle = tmp_path / "oracle.npz"
    catalog = tmp_path / "catalog.npz"
    _behavior_artifact(oracle, "oracle", "")
    _behavior_artifact(catalog, "catalog", MODULE.CATALOG_RECIPE)

    result = MODULE.compare_behavior_artifacts(oracle, catalog)
    assert result["verdict"] == "PASS"
    assert result["bit_differences"] == []
    assert result["normalized_run_config_equal"] is True


def test_behavior_comparison_planted_mismatch_turns_gate_red(tmp_path):
    oracle = tmp_path / "oracle.npz"
    catalog = tmp_path / "catalog.npz"
    _behavior_artifact(oracle, "oracle", "")
    _behavior_artifact(catalog, "catalog", MODULE.CATALOG_RECIPE,
                       value=np.nextafter(1.0, np.inf))

    result = MODULE.compare_behavior_artifacts(oracle, catalog)
    assert result["verdict"] == "FAIL"
    assert [row["key"] for row in result["bit_differences"]] == ["T3d_day5"]
