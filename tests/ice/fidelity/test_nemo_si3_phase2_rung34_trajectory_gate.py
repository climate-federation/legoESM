"""Direct controls for the SI3 rung-3.4 trajectory and regime gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung34_trajectory_gate.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_phase2_rung34_trajectory_gate_direct", _GATE_PATH
)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)
_CASE_DT_S = 30.0  # tests/ICE_RHEO/EXPREF/namelist_cfg:35
_EXPECTED_RESTART_MOMENTS = 160  # 32 tracers times five Prather moments
_SUBUNIT_ORACLE_VALUE = 0.29
_SUBUNIT_ABSOLUTE_ERROR = 6.8e-16


def test_closing_row_replays_si3_evp_equations_and_binds() -> None:
    divergence = np.asarray([[-1.0e-12, 2.0e-12]], dtype=np.float64)
    deformation = np.asarray([[3.0e-12, 6.0e-12]], dtype=np.float64)
    row = gate._closing_row(3, divergence, deformation, _CASE_DT_S)
    assert row["max_abs_delta_s_inv"] == 6.0e-12
    assert row["max_closing_net_s_inv"] == 1.5e-12
    assert row["max_opening_s_inv"] == 3.0e-12
    assert row["max_closing_net_area_per_step"] == 4.5e-11
    assert row["source_significant"] is False

    planted = gate._closing_row(3, divergence, deformation * 10.0, _CASE_DT_S)
    assert planted["source_significant"] is True


def test_oracle_zero_fields_are_uninformative_not_at_bar() -> None:
    zero = np.zeros((2, 2), dtype=np.float64)
    row = gate.gate._score("kt1.oa_i", zero, zero.copy(), uninformative_zero=True)
    assert row["status"] == "UNINFORMATIVE"
    assert row["bitwise_nonzero_over_n"] == "0 / 4"

    with pytest.raises(gate.gate.Rung34GateError, match="not oracle-zero"):
        gate.gate._score(
            "kt1.oa_i",
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            uninformative_zero=True,
        )


def test_active_window_registers_exactly_160_unique_prather_moments() -> None:
    names = {
        gate._moment_restart_name(moment, tracer)
        for moment in gate.SI3_PRATHER_MOMENT_NAMES
        for tracer in gate.gate.ICE_RHEO_TRACERS
    }
    assert len(names) == _EXPECTED_RESTART_MOMENTS
    assert {"sxice", "sxyice", "sxc0_l05", "sxye_l10", "sxysi_l10", "sxyvl"} <= names


def test_normalized_bar_and_field_relative_diagnostic_are_distinct() -> None:
    oracle = np.asarray([_SUBUNIT_ORACLE_VALUE], dtype=np.float64)
    candidate = oracle + _SUBUNIT_ABSOLUTE_ERROR
    row = gate.gate._score("stress", oracle, candidate)
    assert row["status"] == "AT-BAR"
    assert row["normalized_max_abs"] <= gate.gate.POINTWISE_BAR
    assert row["relative_max_abs"] > gate.gate.POINTWISE_BAR
    assert row["classification_metric"] == "normalized_max_abs"
    assert row["relative_is_diagnostic"] is True


def test_plant_self_check_requires_the_scored_row_to_move() -> None:
    clean = {
        "name": "active.step8.v_s",
        "status": "AT-BAR",
        "normalized_max_abs": 2.8e-17,
    }
    planted = {
        "name": "active.step8.v_s",
        "status": "DEBT",
        "normalized_max_abs": 1.0e-10,
    }
    evidence = gate._plant_row_evidence(clean, planted)
    assert evidence["clean_status"] == "AT-BAR"
    assert evidence["planted_status"] == "DEBT"

    with pytest.raises(gate.Rung34TrajectoryError, match="did not make its row DEBT"):
        gate._plant_row_evidence(clean, clean)


def test_full_walk_records_each_fields_first_debt_only() -> None:
    register = {}
    gate._update_first_over_bar(
        register,
        9,
        [
            {"name": "step9.u_ice", "status": "AT-BAR"},
            {"name": "step9.stress1_i", "status": "DEBT"},
        ],
    )
    gate._update_first_over_bar(
        register,
        10,
        [
            {"name": "step10.u_ice", "status": "DEBT"},
            {"name": "step10.stress1_i", "status": "DEBT"},
        ],
    )
    assert register["stress1_i"]["completed_step"] == 9
    assert register["u_ice"]["completed_step"] == 10


def test_full_walk_records_first_bit_departure_independently_of_bar() -> None:
    register = {}
    gate._update_first_non_bit_exact(
        register,
        9,
        [
            {
                "name": "step9.u_ice",
                "status": "AT-BAR",
                "bitwise_nonzero_over_n": "1 / 1000000",
            },
            {
                "name": "step9.stress1_i",
                "status": "AT-BAR",
                "bitwise_nonzero_over_n": "0 / 1000000",
            },
        ],
    )
    gate._update_first_non_bit_exact(
        register,
        10,
        [
            {
                "name": "step10.u_ice",
                "status": "DEBT",
                "bitwise_nonzero_over_n": "2 / 1000000",
            },
            {
                "name": "step10.stress1_i",
                "status": "AT-BAR",
                "bitwise_nonzero_over_n": "3 / 1000000",
            },
        ],
    )
    assert register["u_ice"]["completed_step"] == 9
    assert register["stress1_i"]["completed_step"] == 10


def test_full_walk_reports_jit_eager_state_leaf_difference() -> None:
    state = (jnp.asarray([1.0], dtype=jnp.float64),)
    clean = gate._jit_eager_rows(state, state)
    planted = gate._jit_eager_rows(
        state, (jnp.asarray([2.0], dtype=jnp.float64),)
    )
    assert clean[0]["status"] == "AT-BAR"
    assert planted[0]["status"] == "DEBT"


@pytest.mark.parametrize(
    ("divergence", "deformation", "message"),
    (
        (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            "shape mismatch",
        ),
        (
            np.asarray([[np.inf]], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            "non-finite",
        ),
    ),
)
def test_closing_row_fails_closed(divergence, deformation, message: str) -> None:
    with pytest.raises(gate.Rung34TrajectoryError, match=message):
        gate._closing_row(1, divergence, deformation, _CASE_DT_S)
