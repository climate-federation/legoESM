"""Direct fail-closed controls for the OVERFLOW 19-frame gate."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_overflow_barotropic_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_overflow_barotropic_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_zero_trace(path: Path) -> None:
    nx, ny, ncycle, nfields, bits = gate.EXPECTED
    full = np.zeros(nx * ny, dtype=np.float64).tobytes()
    a2d = np.zeros((nx - 4) * (ny - 4), dtype=np.float64).tobytes()
    with path.open("wb") as handle:
        handle.write(b"NEMO_L1_OVBT_1  ")
        handle.write(struct.pack("=11i", 1, 1, 1, 1, 1, 3, ncycle, nx, ny, nfields, bits))
        for jn in range(1, ncycle + 1):
            handle.write(struct.pack("=i", jn))
            for name in gate.FIELDS:
                handle.write(a2d if name in ("slow_u", "slow_v") else full)


def test_oracle_reader_inventory_is_exact_and_rejects_file_side_extra(tmp_path):
    path = tmp_path / "trace.bin"
    _write_zero_trace(path)
    report = gate.read_oracle_trace(path)
    assert report["header"]["nfields"] == 19
    assert len(report["substeps"]) == 4
    with path.open("ab") as handle:
        handle.write(b"unaccounted")
    with pytest.raises(gate.GateError, match="trailing unregistered bytes"):
        gate.read_oracle_trace(path)


def test_oracle_reader_rejects_truncated_payload(tmp_path):
    path = tmp_path / "trace.bin"
    _write_zero_trace(path)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(gate.GateError, match="truncated record"):
        gate.read_oracle_trace(path)


@pytest.mark.parametrize("name", ("eta_entry", "u_exit"))
def test_planted_frame_controls_make_the_gate_row_red(name):
    oracle = np.zeros((3, 4), dtype=np.float64)
    row = gate.score_frame(
        name, oracle, oracle.copy(), np.ones_like(oracle, dtype=bool), plant=True
    )
    assert row["status"] == "DEBT"
    assert row["normalized_max_abs"] == 1.0


def test_frame_registry_is_complete_and_nonduplicated():
    assert len(gate.FIELDS) == 19
    assert len(set(gate.FIELDS)) == 19
    assert set(gate.STAGGER.values()) == {"T", "U", "V"}
