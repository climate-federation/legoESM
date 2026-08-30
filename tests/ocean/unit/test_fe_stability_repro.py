"""Direct controls for the committed DINO FE stability reproducer."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[3]
PROBE = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226" /
         "fe_stability_repro.py")
sys.path.insert(0, str(PROBE.parent))
SPEC = importlib.util.spec_from_file_location("fe_stability_repro", PROBE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recipe_override_parser_accepts_typed_values():
    assert MODULE.parse_recipe_overrides([
        'barotropic_seed_evaluation="generic"',
        "barotropic_diffusion_alpha=0.01",
    ]) == {
        "barotropic_seed_evaluation": "generic",
        "barotropic_diffusion_alpha": 0.01,
    }


def test_column_mean_deposit_plant_fires_at_planted_cell():
    result = MODULE.column_mean_deposit_plant()
    assert result["fired"] is True
    assert result["metrics"]["u"]["max_abs"] > 0.0
    assert result["metrics"]["u"]["index"] == [1, 2]


def test_fast_term_trace_plant_fires_and_keeps_largest_call():
    result = MODULE.fast_term_trace_plant()
    assert result["fired"] is True
    assert result["metrics"]["calls"] == 2
    assert result["metrics"]["u"]["max_abs"] == 3.0
    assert result["metrics"]["u"]["index"] == [1, 0]


@pytest.mark.parametrize(
    "raw, message",
    [
        (["not-an-assignment"], "expected FIELD=JSON"),
        (["not_a_field=true"], "unknown DINOConfig override"),
        (["barotropic_diffusion_alpha=not-json"], "invalid JSON"),
        (["barotropic_diffusion_alpha=0.0",
          "barotropic_diffusion_alpha=0.01"], "duplicate DINOConfig override"),
    ],
)
def test_recipe_override_parser_fails_closed(raw, message):
    with pytest.raises(ValueError, match=message):
        MODULE.parse_recipe_overrides(raw)
