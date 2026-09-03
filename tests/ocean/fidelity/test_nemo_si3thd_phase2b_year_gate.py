"""Unit checks for the SI3 Phase-2b arithmetic and year gate."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

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


def test_phase3_owner_arms_confirm_and_plants_exit_red(gate) -> None:
    card = build_c1d_omip_l3_card(oracle_root=gate.REPLAY_ROOT)
    result = gate._owner_arms(gate.REPLAY_ROOT, card)
    assert result["step74_negative_evaporation_deposition"]["verdict"] == "CONFIRMED"
    assert result["surface_melt"]["verdict"] == "CONFIRMED"
    with pytest.raises(gate.phase2.GateError, match="deposition arm"):
        gate._owner_arms(gate.REPLAY_ROOT, card, plant_deposition=True)
    with pytest.raises(gate.phase2.GateError, match="surface-melt arm"):
        gate._owner_arms(gate.REPLAY_ROOT, card, plant_surface=True)


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


def test_truncated_thermodynamics_frame_fails_closed(gate, tmp_path: Path) -> None:
    source = gate.REPLAY_ROOT / "oracle_si3_thd_frames.bin"
    planted = tmp_path / "truncated.bin"
    with source.open("rb") as stream:
        planted.write_bytes(stream.read(100))
    with planted.open("rb") as stream:
        with pytest.raises(gate.phase2.GateError):
            gate.phase2._read_thd_step(stream, 1)
