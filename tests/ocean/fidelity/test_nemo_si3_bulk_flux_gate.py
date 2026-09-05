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
    assert row["row_scale_ulp"] == np.spacing(1.0)
    assert row["max_row_scale_ulp_error"] == 1.0


def test_nemo_source_round_is_identity_jittable_and_differentiable() -> None:
    from legoesm.core.source_rounding import nemo_source_round

    values = jnp.asarray([-2.0, -0.0, 3.0], dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(nemo_source_round(values)), values)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(nemo_source_round)(values)), values,
    )
    gradient = jax.grad(lambda x: jnp.sum(nemo_source_round(x)))(values)
    np.testing.assert_array_equal(np.asarray(gradient), np.ones(3))


def test_runtime_is_stamped_and_only_explicit_bit_claim_fails_closed() -> None:
    assert gate.runtime_versions() == gate.ACCEPTED_RUNTIME
    bad = {**gate.ACCEPTED_RUNTIME, "jax": "0.11.1"}
    stamp = gate.validate_runtime(bad)
    assert stamp["bit_identity_claim_valid"] is False
    assert "bit-exactness claims valid only under" in stamp["statement"]
    with pytest.raises(gate.GateError, match="unregistered numeric runtime"):
        gate.validate_runtime(bad, require_bit_identity=True)


@pytest.mark.skipif(
    not (gate.DEFAULT_ROOT / "oracle_si3_bulk_operands.bin").exists(),
    reason="retained C1D oracle is not mounted",
)
def test_unregistered_stack_runs_science_gate_with_bit_claim_withheld(monkeypatch) -> None:
    bad = {**gate.ACCEPTED_RUNTIME, "jax": "0.11.1"}
    monkeypatch.setattr(gate, "runtime_versions", lambda: bad)
    result = gate.evaluate()
    assert result["normalized_verdict"] == "AT-BAR"
    assert result["bit_verdict"] == "WITHHELD_RUNTIME"
    assert result["numeric_runtime"]["observed"] == bad
    assert result["nemo_order_numpy_probe"]["bit_identity_status"] == (
        "WITHHELD_RUNTIME"
    )
    assert result["nemo_order_numpy_probe"]["nemo_order_numpy"][
        "bit_identical"
    ] is None
    assert "withheld" in result["scalar_glibc_owner_probe"][
        "interpretation"
    ].lower()
    assert all(
        group["status"] == "WITHHELD_RUNTIME"
        for group in result["bit_owner_groups"].values()
    )
    assert result["friction_association_probe"]["bit_identical"] is None
    assert result["friction_association_probe"]["bit_identity_status"] == (
        "WITHHELD_RUNTIME"
    )
    with pytest.raises(gate.GateError, match="unregistered numeric runtime"):
        gate.evaluate(require_bit_identity=True)


def test_selector_is_single_orca1_identity() -> None:
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
    from legoesm.ice.config import validate_si3_bulk_config

    card = build_c1d_omip_l3_card()
    assert card.precision_policy.transcendentals == "libm"
    assert (
        card.config.Cd_ice, card.config.Ce_ice, card.config.Ch_ice,
        card.config.drag_ocean, card.config.snow_blow_exponent,
        card.config.albedo_snow_dry, card.config.albedo_snow_melt,
        card.config.albedo_ice_dry, card.config.albedo_ice_melt,
        card.config.albedo_ice_pivot,
    ) == (1.0e-3, 1.0e-3, 1.0e-3, 5.0e-3, 0.66, 0.85, 0.75, 0.64, 0.53, 1.0)
    validate_si3_bulk_config(card.config)
    with pytest.raises(ValueError, match="Cd_ice=Ch_ice=Ce_ice"):
        validate_si3_bulk_config(card.config._replace(Cd_ice=card.config.Cd_ice * 2.0))


def test_positive_subnormal_snow_uses_nemo_nonzero_branch() -> None:
    from legoesm import constants
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
    from legoesm.ice.sea_ice import _nemo_si3_ice_albedo

    tiny = jnp.asarray(np.nextafter(np.float64(0.0), np.float64(1.0)))
    args = (jnp.asarray(constants.T_freeze), jnp.asarray(0.586), tiny, jnp.asarray(0.81))
    config = build_c1d_omip_l3_card().config
    faithful = _nemo_si3_ice_albedo(*args, bulk_config=config)
    ablated = _nemo_si3_ice_albedo(
        *args, bulk_config=config, _preserve_subnormal_snow=False,
    )
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
def test_compiler_folded_constants_are_bit_pinned() -> None:
    result = gate.validate_folded_constants()
    assert result["bytes"] == 32
    assert [row["ieee754_hex"] for row in result["rows"]] == [
        "3fe3e76d0b3af3e8",
        "3fd247bcd3cd320c",
        "3fe9258f81f79246",
        "40026bb1bbb55516",
    ]


def test_planted_folded_constant_violation_fires() -> None:
    with pytest.raises(gate.GateError):
        gate.validate_folded_constants(plant=True)


@pytest.mark.skipif(
    not (gate.DEFAULT_ROOT / "oracle_si3_bulk_operands.bin").exists(),
    reason="retained C1D oracle is not mounted",
)
def test_full_year_gate_and_all_plants() -> None:
    result = gate.evaluate()
    assert result["verdict"] == "AT-BIT"
    assert result["normalized_verdict"] == "AT-BAR"
    assert result["over_bar_rows"] == 0
    assert result["steps"] == 8760
    assert result["oracle_version"] == "V2_SCALAR_MATH"
    assert result["numeric_runtime"]["bit_identity_claim_valid"] is True
    assert result["bit_comparisons"] == result["comparisons"]
    assert result["comparisons"] == 271_560
    assert result["non_bit_identical_rows"] == 0
    assert result["bit_identical_rows"] == 271_560
    assert result["friction_association_probe"]["bit_identical"] is True
    assert result["friction_association_probe"]["bit_identity_status"] == "VALID"
    assert result["friction_association_probe"]["inputs"]["mask"] == 0.25
    assert result["bit_owner_groups"] == {}
    assert all("max_relative_error_nonzero_oracle" in row for row in result["rows"])
    assert all("row_scale_ulp" in row for row in result["rows"])
    assert all("max_row_scale_ulp_error" in row for row in result["rows"])
    assert result["dtypes"]["transcendentals"] == "libm"
    assert all(
        row["non_bit_identical_count"] == 0
        for row in result["scalar_glibc_owner_probe"]["rows"]
    )
    assert result["nemo_order_numpy_probe"]["nemo_order_numpy"][
        "bit_identical"
    ] is True
    arm = result["private_arms"]["disable_source_statement_rounding"]
    assert arm["non_bit_rows"] == 16_494
    assert arm["per_formula_non_bit_rows"] == {
        "POST_BLK_ICE_1.utau_ice": 2821,
        "POST_BLK_ICE_1.vtau_ice": 2797,
        "POST_BLK_ICE_2.evap_ice": 3879,
        "POST_BLK_ICE_2.devap_ice": 3852,
        "POST_BLK_ICE_2.emp_ice": 1900,
        "POST_BLK_ICE_2.emp_tot": 1232,
        "POST_ICE_FLX_OTHER.fhld": 13,
    }
    for plant in (
        "blk_ice_1", "ice_alb", "blk_ice_2", "ice_flx_other",
        "stream_hash", "coverage", "selector", "bit_owner",
        "runtime",
        "source_round",
        "folded_constant",
        "friction_association",
        "libm_return",
        "libm_si3_albedo_log",
        "libm_si3_exner",
        "libm_si3_saturation_log10",
        "libm_si3_saturation_pow",
        "libm_si3_snowfall_pow",
    ):
        with pytest.raises(gate.GateError):
            gate.evaluate(plant=plant)
