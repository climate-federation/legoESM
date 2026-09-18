"""Round-108 controls for the one-output compiled ``bn2`` walk."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round46_kt2_stage_gate",
    TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _small_bn2(selector: str = ""):
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

    T = jnp.asarray([[[12.0, 11.0, 10.0]]], dtype=jnp.float64)
    S = jnp.asarray([[[35.0, 35.1, 35.2]]], dtype=jnp.float64)
    gdept_0 = jnp.asarray([[[1.0, 3.0, 7.0]]], dtype=jnp.float64)
    gdepw_0 = jnp.asarray([[[2.0, 5.0]]], dtype=jnp.float64)
    stretch = jnp.asarray([[1.125]], dtype=jnp.float64)
    gdept = gdept_0 * stretch[..., None]
    gdepw = gdepw_0 * stretch[..., None]
    e3w = jnp.asarray([[[2.25, 4.5]]], dtype=jnp.float64)
    alpha = jnp.asarray([[[1.0, 2.0, 3.0]]], dtype=jnp.float64)
    beta = jnp.asarray([[[0.5, 0.75, 1.0]]], dtype=jnp.float64)
    return compute_buoyancy_frequency_nemo_bn2(
        T, S, gdept, gdepw, g=jnp.float64(9.0), eos_form="seos",
        e3w_int=e3w, e3w_source="mesh_reference",
        zrw_evaluation="nemo_literal", zrw_gdept_0=gdept_0,
        zrw_gdepw_0=gdepw_0, zrw_stretch=stretch,
        _alpha_beta_override=(alpha, beta),
        _return_intermediate=selector,
    )


@pytest.mark.parametrize("selector", gate.BN2_INTERMEDIATES)
def test_bn2_returns_only_the_selected_intermediate(selector):
    default = np.asarray(_small_bn2())
    output, selected = _small_bn2(selector)
    np.testing.assert_array_equal(
        np.asarray(output).view(np.uint64), default.view(np.uint64))
    assert np.asarray(selected).shape == default.shape


def test_bn2_private_override_and_selector_fail_loudly():
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

    with pytest.raises(ValueError, match="unknown private bn2 intermediate"):
        _small_bn2("not-a-statement")
    T = jnp.ones((1, 1, 3), dtype=jnp.float64)
    with pytest.raises(ValueError, match="must match T/S shape"):
        compute_buoyancy_frequency_nemo_bn2(
            T, T, T, T[..., :-1], e3w_int=T[..., :-1],
            _alpha_beta_override=(T[..., :-1], T[..., :-1]))


def test_bn2_scorer_is_bit_strict_and_ulp_plant_is_nonzero():
    reference = np.asarray([[[1.0, 2.0]]], dtype=np.float64)
    replay = {
        "references": {name: reference for name in gate.BN2_INTERMEDIATES},
        "recorded_rn2": reference,
        "reference_replay": {"classification": "BIT"},
        "input_rows": [{"classification": "BIT"}],
    }
    trace = SimpleNamespace(tke_statement_trace=SimpleNamespace(
        bn2_intermediate=reference,
        bn2_output=reference,
    ))
    clean = gate._bn2_intermediate_rows(
        trace, replay, "TEST", "zrw")
    assert clean["row"]["classification"] == "BIT"
    assert clean["output_row"]["classification"] == "BIT"
    assert clean["row"]["n_unequal"] == 0

    planted = gate._bn2_intermediate_rows(
        trace, replay, "TEST", "zrw",
        plant="stage-bn2-intermediate-ulp")
    assert planted["row"]["n_unequal"] == 1
    assert planted["row"]["plant_baseline"] != 0.0
    assert planted["plant_target"] == planted["row"]["name"]


def test_gate_enumerates_bn2_statements_in_compiled_order():
    assert gate.BN2_INTERMEDIATES == (
        "zrw",
        "zaw",
        "zbw",
        "temperature_contribution",
        "salinity_contribution",
        "contribution_difference",
        "gravity_product",
        "thickness_division",
        "masked_rn2",
    )


def test_private_bn2_hooks_default_to_no_measurement():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    hooks = _NEMOWSRK3TestHooks()
    assert hooks.bn2_intermediate == ""
    assert hooks.bn2_alpha_beta_override is None
