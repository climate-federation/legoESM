from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round73_advmean_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round73_advmean_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _fields() -> dict:
    shape = (50, 2, 2)
    result = {
        "weight": np.arange(1, 51, dtype=np.float64),
        "r1_e2u": np.full((2, 2), 0.25, dtype=np.float64),
        "r1_e1v": np.full((2, 2), 0.5, dtype=np.float64),
        "divisor": np.float64(2.0),
    }
    for face in ("u", "v"):
        entry = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        metric = np.full(shape, 4.0 if face == "u" else 6.0)
        reciprocal = result["r1_e2u" if face == "u" else "r1_e1v"]
        exit_value = entry + (
            result["weight"][:, None, None] * metric) * reciprocal
        result[f"sum_{face}_entry"] = entry
        result[f"metric_{face}"] = metric
        result[f"sum_{face}_exit"] = exit_value
        result[f"pre_lbc_{face}"] = exit_value[-1] / result["divisor"]
    return result


def test_replay_is_exact_and_nonzero_ulp_plant_fires() -> None:
    rows = GATE.validate_fields(_fields())
    assert rows["u_normalized"]["bit_exact"]
    assert rows["v_normalized"]["bit_exact"]
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate_fields(_fields(), replay_ulp=True)


def test_expected_record_size_matches_layout() -> None:
    assert GATE.N_SUBSTEP_FIELDS == 10
    assert GATE.EXPECTED_SIZE == 3_789_976
    missing_two_fields = GATE.EXPECTED_SIZE - 2 * GATE.N_CYCLE * GATE.N2 * 8
    assert missing_two_fields == 3_041_176
