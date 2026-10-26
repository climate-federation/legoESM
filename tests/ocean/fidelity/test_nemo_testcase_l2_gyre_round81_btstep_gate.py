from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity.time_levels import time_level_for_dump

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round81_btstep_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round81_btstep_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _fields() -> dict:
    shape = (GATE.N_CYCLE, 2, 3)
    entry = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 8.0
    zeros = np.zeros(shape, dtype=np.float64)
    ones = np.ones(shape[1:], dtype=np.float64)
    mid = np.zeros((GATE.N_CYCLE, 3), dtype=np.float64)
    mid[:, 0] = 1.0
    back = np.zeros((GATE.N_CYCLE, 4), dtype=np.float64)
    back[:, 1] = 1.0
    fields = {
        "dt_s": np.float64(288.0),
        "t_mask": ones,
        "u_mask": ones,
        "v_mask": ones,
        "mid_coefficients": mid,
        "back_coefficients": back,
        **{name: zeros.copy() for name in GATE.ARRAY_FIELDS},
    }
    for prefix in ("u", "v", "eta"):
        fields[f"{prefix}_entry"] = entry.copy()
        fields[f"{prefix}_b"] = zeros.copy()
        fields[f"{prefix}_bb"] = zeros.copy()
        fields[f"{prefix}_mid"] = entry.copy()
    fields["eta_continuity"] = entry.copy()
    fields["eta_pgf"] = entry.copy()
    fields["u_exit"] = entry.copy()
    fields["v_exit"] = entry.copy()
    fields["swap_u"] = entry.copy()
    fields["swap_v"] = entry.copy()
    fields["swap_eta"] = entry.copy()
    return fields


def test_replays_are_exact_and_plants_fire() -> None:
    rows = GATE.validate_fields(_fields())
    assert len(rows) == GATE.N_CYCLE
    assert all(result["bit_exact"] for row in rows
               for name, result in row.items() if name != "substep")
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate_fields(_fields(), replay_ulp=True)
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate_fields(_fields(), swap_ulp=True)


def test_swap_replay_distinguishes_signed_zero() -> None:
    fields = _fields()
    fields["u_exit"][0, 0, 0] = 0.0
    fields["swap_u"][0, 0, 0] = -0.0
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate_fields(fields)


def test_layout_and_record_level_are_exact() -> None:
    assert (GATE.NX, GATE.NY, GATE.N_ARRAYS) == (32, 22, 37)
    assert GATE.EXPECTED_SIZE == 10_439_164
    assert GATE.EXPECTED_DEVELOPED_SIZE == 10_444_796
    assert GATE.EXPECTED_QCO_SIZE == 22_508
    assert time_level_for_dump(GATE.RECORD) == "now"
    assert time_level_for_dump(GATE.DEVELOPED_RECORD) == "now"
    assert time_level_for_dump(GATE.DEVELOPED_QCO_RECORD) == "after"


def test_developed_final_boundary_and_qco_plants_fire() -> None:
    final_pssh = np.arange(GATE.COUNT, dtype=np.float64).reshape(
        (GATE.NY, GATE.NX)) / 32.0
    ssha = np.zeros((GATE.JPJ, GATE.JPI), dtype=np.float64)
    ssha[GATE.NTSJ - 1:GATE.NTEJ,
         GATE.NTSI - 1:GATE.NTEI] = final_pssh
    r1_ht_0 = np.full_like(ssha, 0.125)
    fields = {
        "final_pssh": final_pssh,
        "swap_eta": np.repeat(final_pssh[None, ...], GATE.N_CYCLE, axis=0),
    }
    qco = {"ssha": ssha, "r1_ht_0": r1_ht_0,
           "r3ta": ssha * r1_ht_0}
    result = GATE.validate_developed_boundary(fields, qco)
    assert all(row["bit_exact"] for row in result.values())
    with pytest.raises(SystemExit, match="stage-1 ssha"):
        GATE.validate_developed_boundary(fields, qco, final_pssh_ulp=True)
    with pytest.raises(SystemExit, match="QCO multiplication"):
        GATE.validate_developed_boundary(fields, qco, qco_ulp=True)
