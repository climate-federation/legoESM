"""Non-vacuous layout controls for the Round-154 transport acquisition."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
          / "nemo_testcase_l2_gyre_round154_developed_transport_gate.py")


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("round154_transport_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("round154_transport_gate", module)
    spec.loader.exec_module(module)
    return module


def _record(path: Path, gate) -> None:
    two = np.zeros((36, 26), dtype="=f8", order="F")
    three = np.zeros((36, 26, 31), dtype="=f8", order="F")
    arrays = {name: two.copy(order="F") for name in gate.FIELDS_2D}
    arrays.update({name: three.copy(order="F") for name in gate.FIELDS_3D})
    arrays["umask"].fill(1.0)
    arrays["vmask"].fill(1.0)
    arrays["e2u"].fill(1.0)
    arrays["e3u_0"].fill(1.0)
    arrays["uu_Kmm"].fill(1.0)
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.ljust(16).encode("ascii"))
        handle.write(struct.pack("=15i", *gate.HEADER))
        for name in gate.FIELDS_2D + gate.FIELDS_3D:
            handle.write(arrays[name].tobytes(order="F"))


def test_record_layout_and_truncation_are_fail_closed(tmp_path, gate):
    path = tmp_path / gate.RECORD
    _record(path, gate)
    assert path.stat().st_size == gate.RECORD_BYTES
    record = gate.read_record(path)
    assert tuple(record["fields"]) == gate.FIELDS_2D + gate.FIELDS_3D
    with pytest.raises(gate.GateError, match="bytes, expected"):
        gate.read_record(path, truncate=True)


def test_operand_ulp_control_moves_the_reconstructed_statement(tmp_path, gate):
    path = tmp_path / gate.RECORD
    _record(path, gate)
    record = gate.read_record(path)
    assert gate._effect_control(record)["reconstructed_zFu_cells_moved"] == 0
    with pytest.raises(gate.GateError, match="ULP moved reconstructed zFu"):
        gate._effect_control(record, plant=True)


def test_acquisition_uses_a_new_target_and_no_gnu_time():
    run = (SCRIPT.parent / "nemo_testcase_l2_gyre_round154_developed_transport"
           / "run.sh").read_text()
    assert "GYRE_OMIP_L2_P3_SM_R154TRPWALK" in run
    assert "./makenemo -r GYRE_PISCES" in run
    assert "/usr/bin/time" not in run
    assert "STATUS PLANT-FIRED" in run
    assert "ROUND154_DEVELOPED_TRANSPORT_READY" in run
