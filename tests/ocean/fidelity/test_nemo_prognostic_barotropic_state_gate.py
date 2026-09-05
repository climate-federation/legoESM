"""Fail-closed unit guards for the Decision-8 cross-card state gate."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np


PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_prognostic_barotropic_state_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_barotropic_state_gate", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_reader_names_kaa_pair_and_ignores_transport_tail(tmp_path):
    nx, ny = 8, 7
    count = nx * ny
    path = tmp_path / "oracle_bt_frames_kt00000001.bin"
    values = np.arange(4 * count, dtype=np.float64)
    path.write_bytes(
        b"NEMO_L1_BTFRM_1 "
        + struct.pack("=6i", 1, 1, 3, nx, ny, 64)
        + values.tobytes()
    )
    got = gate.read_bt_pair(path)
    assert got["Kaa"] == 3
    assert got["uu_b"].shape == (ny - 4, nx - 4)
    assert got["vv_b"].shape == (ny - 4, nx - 4)
    assert not np.array_equal(got["uu_b"], got["vv_b"])


def test_exact_pair_passes_and_three_ulp_plant_fails_once():
    oracle = np.ones((2, 3), dtype=np.float64)
    mask = np.ones_like(oracle, dtype=bool)
    exact = gate.score("exact", oracle, oracle.copy(), mask, plant=False)
    planted = gate.score("plant", oracle, oracle.copy(), mask, plant=True)
    assert exact["status"] == "AT-BAR" and exact["unequal"] == 0
    assert planted["status"] == "DEBT" and planted["unequal"] == 1
    absent = gate.score(
        "absent", oracle, oracle.copy(), np.zeros_like(mask), plant=False,
        allow_empty_no_active_face=True)
    assert absent["status"] == "UNINFORMATIVE"
    assert absent["unequal"] == 0
