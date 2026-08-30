"""Controls for the faithful-card catalog reachability gate."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


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
