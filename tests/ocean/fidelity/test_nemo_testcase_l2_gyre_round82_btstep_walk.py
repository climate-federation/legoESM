from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round82_btstep_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round82_btstep_walk", SCRIPT)
assert SPEC and SPEC.loader
WALK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WALK)


def _exact() -> dict:
    return {
        "bit_exact": True,
        "differing_cells": 0,
        "wet_cells": 1,
        "absolute_max": 0.0,
        "reference_max_abs": 1.0,
    }


def test_first_non_bit_uses_compiled_order() -> None:
    row = {"substep": 1, **{name: _exact() for name in WALK.SOURCE_ORDER}}
    row["slow_v"] = {**_exact(), "bit_exact": False, "differing_cells": 1}
    row["u_exit"] = {**_exact(), "bit_exact": False, "differing_cells": 1}
    assert WALK.first_non_bit([row])["boundary"] == "slow_v"


def test_trace_mapping_covers_every_record_array() -> None:
    assert set(WALK.TRACE_KEYS).issubset(set(WALK.round81.ARRAY_FIELDS))
    assert set(WALK.round81.ARRAY_FIELDS).issubset(set(WALK.SOURCE_ORDER))
    assert len(WALK.SOURCE_ORDER) == len(WALK.round81.ARRAY_FIELDS) + 7
    assert WALK._stagger("depth_u_mid") == "u"
    assert WALK._stagger("depth_v_mid") == "v"


def test_face_replay_uses_live_cell_coefficient_without_reassociation() -> None:
    cell = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
    got_u, got_v = WALK._face_replay_from_live_cells(
        {"drag_coefficient_t": cell})
    np.testing.assert_array_equal(
        got_u, np.array([[1.5, 3.0, 2.5], [12.0, 24.0, 20.0]]))
    np.testing.assert_array_equal(
        got_v, np.array([[4.5, 9.0, 18.0], [4.5, 9.0, 18.0]]))


def test_isolated_jit_replay_keeps_written_boundaries_distinct() -> None:
    shape = (2, 1, 1)
    fields = {
        name: np.zeros(shape, dtype=np.float64)
        for name in WALK.round81.ARRAY_FIELDS
    }
    fields.update({
        "mid_coefficients": np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        "back_coefficients": np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float64),
        "dt_s": np.float64(1.0),
        "t_mask": np.ones((1, 1), dtype=np.float64),
        "u_mask": np.ones((1, 1), dtype=np.float64),
        "v_mask": np.ones((1, 1), dtype=np.float64),
    })
    fields["u_entry"].fill(1.0)
    fields["v_entry"].fill(2.0)
    fields["eta_entry"].fill(5.0)
    fields["ssh_forcing"].fill(1.0)
    fields["continuity_div"].fill(1.0)
    fields["pgf_u"].fill(1.0)
    fields["pgf_v"].fill(1.0)
    fields["cor_u"].fill(1.0)
    fields["cor_v"].fill(1.0)
    fields["drag_coefficient_u"].fill(2.0)
    fields["drag_coefficient_v"].fill(2.0)
    fields["inverse_depth_u"].fill(1.0)
    fields["inverse_depth_v"].fill(1.0)
    fields["slow_u"].fill(1.0)
    fields["slow_v"].fill(1.0)

    got = WALK._isolated_jit_replay(fields)

    assert set(got) == {
        "u_mid", "v_mid", "eta_mid", "eta_continuity", "eta_pgf",
        "trd_u", "trd_v", "u_exit", "v_exit",
        "swap_u", "swap_v", "swap_eta",
    }
    np.testing.assert_array_equal(got["u_mid"], np.full(shape, 1.0))
    np.testing.assert_array_equal(got["v_mid"], np.full(shape, 2.0))
    np.testing.assert_array_equal(got["eta_continuity"], np.full(shape, 3.0))
    np.testing.assert_array_equal(got["trd_u"], np.full(shape, 3.0))
    np.testing.assert_array_equal(got["u_exit"], np.full(shape, 6.0))
    np.testing.assert_array_equal(got["swap_eta"], np.full(shape, 3.0))


def test_handoff_ulp_control_reaches_shared_qco_ratio() -> None:
    stage = WALK.round46.read_stage(
        WALK.STAGE_ROOT / "oracle_momstage_kt00000002_s3.bin",
        expected_kt=2, expected_stage=3)
    entry = WALK.gate.read_entry(
        WALK.ENTRY_ROOT / "oracle_step_entry_kt00000003.bin")
    active = WALK.round46._owned3(stage["arrays"]["tmask"])[..., 0] > 0.5

    control = WALK._handoff_plant_control(
        SimpleNamespace(stage_root=WALK.STAGE_ROOT),
        np.asarray(entry["ssh"], dtype=np.float64), active)

    assert control["after_bits"] == control["before_bits"] + 1
    assert control["ssh_delta"]["differing_cells"] == 1
    assert control["derived_r3u_delta"]["differing_cells"] > 0
    assert control["fires"] is True


def test_pytree_identity_detects_one_ulp_without_changing_structure() -> None:
    baseline = {
        "field": np.array([0.0, 1.0], dtype=np.float64),
        "count": np.array(2, dtype=np.int64),
    }
    exact = WALK._pytree_identity(baseline, {
        "field": np.array([0.0, 1.0], dtype=np.float64),
        "count": np.array(2, dtype=np.int64),
    })
    changed_field = np.array([0.0, 1.0], dtype=np.float64)
    changed_field[1] = np.nextafter(changed_field[1], np.inf)
    changed = WALK._pytree_identity(baseline, {
        "field": changed_field,
        "count": np.array(2, dtype=np.int64),
    })

    assert exact["bit_exact"] is True
    assert exact["differing_cells"] == 0
    assert changed["bit_exact"] is False
    assert changed["differing_cells"] == 1
    assert changed["absolute_max"] == np.spacing(np.float64(1.0))


def test_developed_registry_refuses_a_missing_compiled_boundary() -> None:
    WALK._validate_developed_registry()
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_developed_registry(WALK.SOURCE_ORDER[:-1])


def test_developed_comparison_counts_signed_zero_and_first_index() -> None:
    actual = np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    expected = np.array([[-0.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mask = np.ones((2, 2), dtype=bool)

    row = WALK._developed_comparison(actual, expected, mask)

    assert row["bit_exact"] is False
    assert row["differing_cells"] == 1
    assert row["absolute_max"] == 0.0
    assert row["rms"] == 0.0
    assert row["first_unequal_index"] == [0, 0]

    scalar = WALK._developed_comparison(
        np.float64(1.0), np.float64(1.0), np.ones((), dtype=bool))
    assert scalar["bit_exact"] is True
    assert scalar["cells_scored"] == 1
