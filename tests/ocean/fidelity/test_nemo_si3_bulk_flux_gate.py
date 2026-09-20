"""Controls for the C1D SI3 ice-side bulk-flux fidelity gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

GATE_PATH = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo_si3_bulk_flux_gate.py"
SPEC = importlib.util.spec_from_file_location("nemo_si3_bulk_flux_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_frame_registry_matches_config_local_writer() -> None:
    writer = (Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
              "nemo502_si3bulk_MY_SRC/icesbc.F90").read_text()
    assert gate.STAGE_COUNTS == {0: 17, 1: 50, 2: 39}
    assert "CALL l3bulk_dump_tau( kt, utau_ice, vtau_ice )" in writer
    assert "CALL l3bulk_dump_flux( kt, 1 )" in writer
    assert "CALL l3bulk_dump_flux( kt, 2 )" in writer


def test_exchange_coverage_is_exactly_complete() -> None:
    drift_path = GATE_PATH.with_name("nemo_si3_exchange_drift_gate.py")
    spec = importlib.util.spec_from_file_location("exchange_gate", drift_path)
    assert spec and spec.loader
    drift = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drift)
    assert set(gate.COVERAGE) == {field.name for field in drift.FIELD_SPECS}
    gate.validate_coverage()
    bad = dict(gate.COVERAGE)
    bad.pop("fr_i")
    with pytest.raises(gate.GateError, match="incomplete"):
        gate.validate_coverage(bad)


def test_score_control_detects_pointwise_plant() -> None:
    row = gate._score(np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), "X", "plant")
    assert row["over_bar_count"] == 1
    assert row["first_over_step"] == 2


def test_score_distinguishes_bar_from_bit_identity() -> None:
    oracle = np.asarray([1.0], dtype=np.float64)
    predicted = np.nextafter(oracle, np.asarray([2.0], dtype=np.float64))
    row = gate._score(predicted, oracle, "X", "one_ulp")
    assert row["over_bar_count"] == 0
    assert row["non_bit_identical_count"] == 1
    assert row["bit_identical_count"] == 0
    assert row["max_relative_error_step"] == 1


def test_bitwise_evidence_runtime_is_stamped_and_fails_closed() -> None:
    assert gate.runtime_versions() == gate.ACCEPTED_RUNTIME
    bad = {**gate.ACCEPTED_RUNTIME, "jax": "0.11.1"}
    with pytest.raises(gate.GateError, match="unregistered numeric runtime"):
        gate.validate_runtime(bad)


def test_selector_is_single_orca1_identity() -> None:
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
    from legoesm.ice.config import validate_si3_bulk_config

    card = build_c1d_omip_l3_card()
    validate_si3_bulk_config(card.config)
    with pytest.raises(ValueError, match="Cd_ice=Ch_ice=Ce_ice"):
        validate_si3_bulk_config(card.config._replace(Cd_ice=card.config.Cd_ice * 2.0))


def test_positive_subnormal_snow_uses_nemo_nonzero_branch() -> None:
    from legoesm import constants
    from legoesm.ice.sea_ice import _nemo_si3_ice_albedo

    tiny = jnp.asarray(np.nextafter(np.float64(0.0), np.float64(1.0)))
    args = (jnp.asarray(constants.T_freeze), jnp.asarray(0.586), tiny, jnp.asarray(0.81))
    faithful = _nemo_si3_ice_albedo(*args)
    ablated = _nemo_si3_ice_albedo(*args, _preserve_subnormal_snow=False)
    assert np.asarray(faithful) != np.asarray(ablated)


def test_selected_dispatch_is_jittable_and_differentiable() -> None:
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
    from legoesm.ice.sea_ice import _bulk_flux_dispatch

    one = jnp.ones((1,), dtype=jnp.float64)
    forcing = AtmToSurface(
        sw_down=one, lw_down=one, precip_total=one, precip_snow=one,
        T_lowest=one * 250.0, q_lowest=one * 1.0e-3,
        u_lowest=one * 5.0, v_lowest=one * 2.0,
        p_lowest=one * 9.9e4, p_surface=one * 1.0e5,
        rho_lowest=one * 1.3, cos_zenith=one, co2_ppmv=one,
        has_radiation=one, has_precipitation=one,
    )
    config = build_c1d_omip_l3_card().config

    def run(T):
        return _bulk_flux_dispatch(T, forcing, config, 0.0)

    eager = run(one * 270.0)
    compiled = jax.jit(run)(one * 270.0)
    for left, right in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(left, right, rtol=1.0e-15, atol=1.0e-15)
    gradient = jax.grad(lambda T: sum(jnp.sum(x) for x in run(T)))(one * 270.0)
    assert np.all(np.isfinite(gradient))


@pytest.mark.skipif(
    not (gate.DEFAULT_ROOT / "oracle_si3_bulk_operands.bin").exists(),
    reason="retained C1D oracle is not mounted",
)
def test_full_year_gate_and_all_plants() -> None:
    result = gate.evaluate()
    assert result["verdict"] == "AT-BAR"
    assert result["over_bar_rows"] == 0
    assert result["steps"] == 8760
    assert result["oracle_version"] == "V2_SCALAR_MATH"
    assert result["bit_comparisons"] == result["comparisons"]
    assert result["non_bit_identical_rows"] > 0
    assert set(result["bit_owner_groups"]) == {
        "jax_exp_ice_alb", "binary64_operation_order",
    }
    assert result["bit_owner_groups"]["jax_exp_ice_alb"]["status"] == (
        "AWAITING_LIBM_POLICY"
    )
    assert all(
        row["non_bit_identical_count"] == 0
        for row in result["scalar_glibc_owner_probe"]["rows"]
    )
    for plant in (
        "blk_ice_1", "ice_alb", "blk_ice_2", "ice_flx_other",
        "stream_hash", "coverage", "selector", "bit_owner",
        "runtime",
    ):
        with pytest.raises(gate.GateError):
            gate.evaluate(plant=plant)
