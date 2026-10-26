from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity.time_levels import time_level_for_dump

SCRIPT = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round76_uamid_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round76_uamid_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _fields() -> dict:
    shape = (GATE.N_CYCLE, 2, 3)
    coefficients = np.zeros((GATE.N_CYCLE, 3), dtype=np.float64)
    coefficients[:, 0] = 1.5
    coefficients[:, 1] = -0.5
    coefficients[:, 2] = 0.25
    un_e = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 8.0
    ub_e = un_e / 2.0
    ubb_e = un_e / 4.0
    ua_e = (
        (coefficients[:, 0, None, None] * un_e) + coefficients[:, 1, None, None] * ub_e
    ) + coefficients[:, 2, None, None] * ubb_e
    return {
        "coefficients": coefficients,
        "un_e": un_e,
        "ub_e": ub_e,
        "ubb_e": ubb_e,
        "ua_e": ua_e,
    }


def test_replay_is_exact_and_nonzero_ulp_plant_fires() -> None:
    rows = GATE.validate_fields(_fields())
    assert len(rows) == GATE.N_CYCLE
    assert all(row["bit_exact"] for row in rows)
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate_fields(_fields(), replay_ulp=True)


def test_expected_record_size_and_reduced_bounds_are_exact() -> None:
    assert (GATE.NX, GATE.NY) == (32, 22)
    assert len(GATE.FIELDS) == 4
    assert GATE.EXPECTED_SIZE == 1_127_856


def test_record_time_level_is_registered_as_now() -> None:
    assert time_level_for_dump(GATE.RECORD) == "now"


def test_replay_distinguishes_signed_zero() -> None:
    fields = _fields()
    fields["ua_e"][0, 0, 0] = -0.0
    fields["un_e"][0, 0, 0] = 0.0
    fields["ub_e"][0, 0, 0] = 0.0
    fields["ubb_e"][0, 0, 0] = 0.0
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate_fields(fields)
