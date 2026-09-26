"""Unit checks for the SI3 Phase-2b arithmetic and year gate."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card

set_policy(PrecisionPolicy.fp64())


@pytest.fixture(scope="module")
def gate():
    directory = str(Path("scripts/validate/ocean_fidelity/testcases").resolve())
    sys.path.insert(0, directory)
    try:
        yield importlib.import_module("nemo_si3thd_phase2b_year_gate")
    finally:
        sys.path.remove(directory)


def test_nemo_operation_order_replay_is_exact_and_normalised_order_differs(gate) -> None:
    card = build_c1d_omip_l3_card(oracle_root=gate.REPLAY_ROOT)
    result = gate._arithmetic_replay(
        gate.REPLAY_ROOT, card.config.ice_constants
    )
    assert result["classification"] == "FLOAT RE-ASSOCIATION"
    assert result["nemo_written_order_max_ulp"] == 0
    assert result["iteration2_qns"]["executing_legoesm_ulp"] > 2
    assert result["kt3_enthalpy"]["executing_legoesm_max_ulp"] == 0
    assert all(row["normalised_solution_max_ulp"] == 1
               for row in result["thomas_rows"])


def test_round9_year_artifact_and_default_root_are_scalar_math_v2(gate) -> None:
    from legoesm.ice.c1d_omip_l3 import ORACLE_V2_ROOT, ORACLE_VERSION

    path = gate.evidence_path("nemo_testcases_l3thd_scalarmath_v2_year_gate.json")
    if not path.exists():
        pytest.skip(f"evidence file not present: {path}")
    artifact = json.loads(path.read_text())
    assert gate.REPLAY_ROOT == ORACLE_V2_ROOT
    assert artifact["card"]["oracle_root"] == str(ORACLE_V2_ROOT)
    assert artifact["card"]["oracle_version"] == ORACLE_VERSION == "V2_SCALAR_MATH"


def test_arithmetic_plant_exits_red(gate) -> None:
    card = build_c1d_omip_l3_card(oracle_root=gate.REPLAY_ROOT)
    with pytest.raises(gate.phase2.GateError, match="exceeds 2 ULP"):
        gate._arithmetic_replay(
            gate.REPLAY_ROOT, card.config.ice_constants, plant=True
        )


def test_snow_free_entry_uses_nemo_zero_volume_convention(gate) -> None:
    path = gate.REPLAY_ROOT / "oracle_si3_thd_frames.bin"
    found = False
    with path.open("rb") as stream:
        for step in range(1, 8761):
            frames = gate.phase2._read_thd_step(stream, step)
            center = gate.phase2._center_global(frames[0])
            if center["v_s"] == 0.0:
                state = gate.phase2._entry_arrays(frames[0])
                np.testing.assert_array_equal(state.e_snow, np.zeros((1, 3)))
                assert np.all(np.isfinite(state.e_snow))
                found = True
                break
    assert found, "seasonal C1D stream never exercised its snow-free state"


def test_phenomenology_floor_classification_is_literal(gate) -> None:
    base = {
        "minimum_m": 1.0, "maximum_m": 2.0,
        "melt_onset_day": 4, "growth_onset_day": 8,
        "minimum_utc": "2018-01-01T00:00:00Z",
        "maximum_utc": "2018-01-02T00:00:00Z",
        "melt_onset": "2018-01-05", "growth_onset": "2018-01-09",
    }
    fp64 = {**base, "minimum_m": 1.2, "maximum_m": 2.1,
            "minimum_utc": "2018-01-02T00:00:00Z",
            "growth_onset": "2018-01-10"}
    fp32 = {**base, "minimum_m": 1.5, "maximum_m": 2.11,
            "minimum_utc": "2018-01-04T00:00:00Z",
            "growth_onset": None}
    rows = {row["quantity"]: row for row in gate._phenomenology_rows(base, fp64, fp32)}
    assert rows["minimum_m"]["status"] == "AT-FLOOR"
    assert rows["maximum_m"]["status"] == "ABOVE-FLOOR"
    assert rows["minimum_utc"]["status"] == "AT-FLOOR"
    assert rows["melt_onset"]["status"] == "AT-FLOOR"
    assert rows["growth_onset"]["status"] == "UNMEASURED"
    assert len(rows) == 6


def test_owner_arms_confirm_and_plants_exit_red(gate) -> None:
    card = build_c1d_omip_l3_card(oracle_root=gate.REPLAY_ROOT)
    result = gate._owner_arms(gate.REPLAY_ROOT, card)
    assert result["step74_negative_evaporation_deposition"]["verdict"] == "CONFIRMED"
    assert result["surface_melt"]["verdict"] == "CONFIRMED"
    assert result["snow_temperature_bounds"]["verdict"] == "CONFIRMED"
    assert all(step["source_replay_bit_exact"]
               for step in result["snow_temperature_bounds"]["steps"])
    split = result["snow_temperature_bounds"]["steps"][1]
    assert split["step"] == 5285
    assert split["nemo_cold_surface_condition"] is True
    assert split["pre_fix_cold_surface_condition"] is False
    assert split["post_fix_cold_surface_condition"] is True
    with pytest.raises(gate.phase2.GateError, match="deposition arm"):
        gate._owner_arms(gate.REPLAY_ROOT, card, plant_deposition=True)
    with pytest.raises(gate.phase2.GateError, match="surface-melt arm"):
        gate._owner_arms(gate.REPLAY_ROOT, card, plant_surface=True)
    with pytest.raises(gate.phase2.GateError, match="snow-temperature bounds arm"):
        gate._owner_arms(gate.REPLAY_ROOT, card, plant_snow_temperature=True)


def test_round4_zdf_operand_stream_registers_tiny_snow_temperature(gate) -> None:
    path = gate.REPLAY_ROOT / "oracle_si3_zdf_inputs.bin"
    wanted = {}
    with path.open("rb") as stream:
        for step in range(1, 5286):
            exact = gate.phase2._read_zdf_input_step(stream, step)
            if step in (4239, 5285):
                wanted[step] = exact
    assert set(wanted) == {4239, 5285}
    freezing = build_c1d_omip_l3_card().config.ice_constants.T0
    for exact in wanted.values():
        assert int(exact["_version"]) == 2
        np.testing.assert_array_equal(exact["t_s"], np.full((1, 3), freezing))
    registry = gate.phase2.ZDF_STATE_OPERAND_REGISTRY["t_s"]
    assert "immediately before ice_thd_zdf" in registry["time_level"]
    assert "icethd.F90:343-355" in registry["source"]


def test_round6_dh_operand_streams_replay_and_fail_closed(gate, tmp_path) -> None:
    from legoesm.ice.bitz_lipscomb import _nemo_snow_enthalpy_remap

    operands = gate._read_dh_operands(
        gate.REPLAY_ROOT / "oracle_si3_dh_operands.bin"
    )
    remaps = gate._read_dh_remap_operands(
        gate.REPLAY_ROOT / "oracle_si3_dh_remap_operands.bin"
    )
    assert set(operands) == {4242, 5734}
    assert remaps[5734]["zh_s"][-1] == 2.117582368135751e-22
    for step in (4242, 5734):
        replay = gate._snow_enthalpy_remap_replay(
            remaps[step]["zh_s"], remaps[step]["ze_s"]
        )
        np.testing.assert_array_equal(replay["e_s"], remaps[step]["e_s"])
        np.testing.assert_array_equal(
            np.asarray(_nemo_snow_enthalpy_remap(
                remaps[step]["zh_s"], remaps[step]["ze_s"]
            )),
            remaps[step]["e_s"],
        )

    planted = tmp_path / "planted_dh_operands.bin"
    planted.write_bytes(
        b"PLANTED_BAD_MAGIC" + (
            gate.REPLAY_ROOT / "oracle_si3_dh_operands.bin"
        ).read_bytes()[16:]
    )
    with pytest.raises(gate.phase2.GateError, match="DH-operand magic"):
        gate._read_dh_operands(planted)

    remap_source = (
        gate.REPLAY_ROOT / "oracle_si3_dh_remap_operands.bin"
    ).read_bytes()
    duplicate = tmp_path / "planted_duplicate_dh_remap.bin"
    duplicate.write_bytes(remap_source + remap_source[:276])
    with pytest.raises(gate.phase2.GateError, match="duplicate DH-remap step"):
        gate._read_dh_remap_operands(duplicate)

    reversed_order = tmp_path / "planted_reversed_dh_remap.bin"
    reversed_order.write_bytes(remap_source[276:] + remap_source[:276])
    with pytest.raises(gate.phase2.GateError, match="DH-remap record order"):
        gate._read_dh_remap_operands(reversed_order)

    truncated = tmp_path / "planted_truncated_dh_remap.bin"
    truncated.write_bytes(remap_source[:-1])
    with pytest.raises(gate.phase2.GateError, match="bad DH-remap payload"):
        gate._read_dh_remap_operands(truncated)


def test_round6_snow_remap_jit_and_gradients_are_finite() -> None:
    from legoesm.ice.bitz_lipscomb import _nemo_snow_enthalpy_remap

    enthalpy = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    remap = jax.jit(_nemo_snow_enthalpy_remap)
    for thickness in (
        jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64),
        jnp.asarray([0.0, 0.0, 0.0, 2.0e-22], dtype=jnp.float64),
        jnp.zeros(4, dtype=jnp.float64),
    ):
        value = remap(thickness, enthalpy)
        gradient = jax.grad(
            lambda energy: jnp.sum(remap(thickness, energy))
        )(enthalpy)
        thickness_gradient = jax.grad(
            lambda depth: jnp.sum(remap(depth, enthalpy))
        )(thickness)
        assert bool(jnp.all(jnp.isfinite(value)))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.all(jnp.isfinite(thickness_gradient)))


def test_round6_combined_dh_owner_and_plant(gate) -> None:
    card = build_c1d_omip_l3_card(oracle_root=gate.REPLAY_ROOT)
    result = gate._dh_owner_evidence(gate.REPLAY_ROOT, card)
    assert result["verdict"] == "CONFIRMED two-operation interaction"
    assert result["replays"]["5734"]["sublimation_max_ulp"] == 0
    assert result["replays"]["5734"]["remap_max_ulp"] == 0
    assert result["private_arms"]["kt4242_h_i_bit_identical"] is True
    assert result["private_arms"]["combined"]["improvement_factor"] > 100.0
    with pytest.raises(
        gate.phase2.GateError,
        match="DH sublimation/remap arm failed 100-fold discriminator",
    ):
        gate._dh_owner_evidence(gate.REPLAY_ROOT, card, plant=True)
    with pytest.raises(
        gate.phase2.GateError, match="exact-entry 1-D thickness bridge"
    ):
        gate._dh_owner_evidence(
            gate.REPLAY_ROOT, card, plant_bridge=True
        )


def test_operator_per_step_and_outlier_controls_exit_red(gate) -> None:
    rows = [
        {"step": step, "maximum_normalised_error": float(step),
         "over_bar_field_rows": 0, "owner": None}
        for step in range(1, 8761)
    ]
    operator = {
        "per_step": rows,
        "largest_outlier": {
            "step": 8760, "absolute_numerator": 8760.0,
            "normalisation_denominator": 1.0,
            "normalised_quotient": 8760.0,
        },
    }
    gate._validate_operator_trajectory(operator)
    with pytest.raises(gate.phase2.GateError, match="per-step count"):
        gate._validate_operator_trajectory(operator, plant=True)
    with pytest.raises(gate.phase2.GateError, match="largest-outlier"):
        gate._validate_operator_trajectory(operator, plant_outlier=True)


def test_scope_ablation_accounting_plant_exits_red(gate) -> None:
    scope = {
        name: {
            "field_rows_compared": 7,
            "changed_field_rows": 2,
            "unchanged_field_rows": 5,
        }
        for name in gate.SCOPE_ARM_KWARGS
    }
    gate._validate_scope_ablation_accounting(scope)
    with pytest.raises(gate.phase2.GateError, match="scope-arm row accounting"):
        gate._validate_scope_ablation_accounting(scope, plant=True)


def test_phase3_branch_census_plant_exits_red(gate) -> None:
    operator = {
        "full_state_snapshots": {
            "74": {"frames": [{}, {}, {
                "branch": "DH_NEGATIVE_EVAPORATION_SNOW_DEPOSITION"
            }]}
        }
    }
    gate._validate_step74_branch_snapshot(operator)
    with pytest.raises(gate.phase2.GateError, match="branch census plant"):
        gate._validate_step74_branch_snapshot(operator, plant=True)


def test_phase3_committed_censuses_retain_rows_and_branch_predicates(gate) -> None:
    before_path = gate.evidence_path(
        "nemo_testcases_l3thd_phase3_baseline_operator.json"
    )
    after_path = gate.evidence_path("nemo_testcases_l3thd_phase3_year_gate.json")
    if not before_path.exists() or not after_path.exists():
        pytest.skip(f"evidence file not present: {before_path if not before_path.exists() else after_path}")
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())["oracle_entry_operator_sweep"]
    assert before["over_bar_field_rows"] == 116_274
    assert before["first_above_1e-12"]["step"] == 74
    assert before["first_above_1e-3"]["step"] == 3837
    assert after["first_over_bar_frame"]["step"] > before["first_over_bar_frame"]["step"]
    assert after["first_above_1e-12"]["step"] == 4239
    assert after["first_above_1e-3"]["step"] == 5285
    assert after["first_above_1e-3"]["surface_melting_condition_differs"] is True
    assert sum(len(rows) for rows in after["over_bar_by_step"].values()) == 37_659
    assert all(len(after["full_state_snapshots"][str(step)]["frames"]) == 8
               for step in (73, 74, 75, 76))


def test_phase4_artifact_retains_per_step_operator_and_outlier_attribution(gate) -> None:
    path = gate.evidence_path("nemo_testcases_l3thd_phase4_year_gate.json")
    if not path.exists():
        pytest.skip(f"evidence file not present: {path}")
    artifact = json.loads(path.read_text())
    operator = artifact["oracle_entry_operator_sweep"]
    assert len(operator["per_step"]) == 8760
    assert [row["step"] for row in operator["per_step"]] == list(range(1, 8761))
    assert sum(row["over_bar_field_rows"] for row in operator["per_step"]) == 36_852
    largest = operator["largest_outlier"]
    assert (largest["step"], largest["sub_call"], largest["variable"]) == (
        5734, "POST_DH", "e_s",
    )
    assert (largest["absolute_numerator"]
            / largest["normalisation_denominator"]
            == largest["normalised_quotient"])
    historical = artifact["historical_1p1e8_outlier"]
    assert (historical["step"], historical["sub_call"], historical["variable"]) == (
        5406, "POST_DH", "e_s",
    )
    assert historical["normalisation_denominator"] == 1.0
    assert historical["absolute_numerator"] == historical["normalised_quotient"]


def test_phase5_artifact_dispositions_all_bundled_changes(gate) -> None:
    path = gate.evidence_path("nemo_testcases_l3thd_phase5_year_gate.json")
    if not path.exists():
        pytest.skip(f"evidence file not present: {path}")
    artifact = json.loads(path.read_text())
    arms = artifact["round4_bundled_change_ablations"]
    exact = arms["exact_entry"]
    assert set(exact) == {
        "zdf_no_snow_melting_row_ranges", "basal_layer_loop",
        "snow_ice_salinity", "eos_operation_order",
    }
    assert exact["zdf_no_snow_melting_row_ranges"]["changed_field_rows"] == 39_505
    assert exact["basal_layer_loop"]["changed_field_rows"] == 11_190
    assert exact["snow_ice_salinity"]["changed_field_rows"] == 0
    assert exact["eos_operation_order"]["zdf_surface_branch_splits"] == 0
    assert arms["continuous"]["snow_ice_salinity"]["changed_field_steps"] == 0

    histogram = artifact["oracle_entry_over_bar_histogram"]
    assert histogram["all_over_bar_rows"] == 36_852
    assert histogram["denominator_one_oracle_exactly_zero_rows"] == 0
    assert histogram["genuine_relative_rows"] == 36_852
    largest = histogram["largest_genuine_relative_row"]
    assert (largest["step"], largest["sub_call"], largest["variable"]) == (
        5734, "POST_DH", "e_s",
    )
    assert artifact["continuous_jump_attribution"][
        "initial_threshold_amplification_from_2e-15_class_steps"
    ] is True


def test_truncated_thermodynamics_frame_fails_closed(gate, tmp_path: Path) -> None:
    source = gate.REPLAY_ROOT / "oracle_si3_thd_frames.bin"
    planted = tmp_path / "truncated.bin"
    with source.open("rb") as stream:
        planted.write_bytes(stream.read(100))
    with planted.open("rb") as stream:
        with pytest.raises(gate.phase2.GateError):
            gate.phase2._read_thd_step(stream, 1)
