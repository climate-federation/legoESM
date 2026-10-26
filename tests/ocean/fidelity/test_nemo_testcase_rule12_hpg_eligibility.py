"""Fail-closed unit guards for the Rule-12 exact-input HPG eligibility gate.

The gate's own numeric row needs a full model step, so it is run as a script
with its planted control.  What is guarded here is the part that decides
whether a record may be BELIEVED at all: the two readers must refuse a wrong
magic, a wrong header shape and a wrong time level, because a reader that
accepts the wrong record would score the wrong frame silently.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


TESTCASES = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
)
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_rule12_hpg_eligibility",
    TESTCASES / "nemo_testcase_rule12_hpg_eligibility.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

CASE = "LOCK_EXCHANGE-zco"
NX, NY, NZ = gate.DIMS[CASE]


def _rhs_bytes(*, magic=b"NEMO_L1_RHS___1 ", version=1, kt=1, level=3,
               nx=NX, ny=NY, nz=NZ, bits=64, values=None):
    count = nx * ny * nz
    if values is None:
        values = np.arange(2 * count, dtype=np.float64)
    return (magic + struct.pack("=7i", version, kt, level, nx, ny, nz, bits)
            + np.asarray(values, dtype=np.float64).tobytes())


def test_rhs_reader_splits_u_and_v_on_the_owned_interior(tmp_path):
    path = tmp_path / "oracle_rhs_kt00000001.bin"
    path.write_bytes(_rhs_bytes())
    got = gate.read_rhs(path, CASE)
    assert got["u"].shape == (NY - 4, NX - 4, NZ)
    assert got["v"].shape == (NY - 4, NX - 4, NZ)
    # The two halves must not be aliased: v starts one full block later.
    assert not np.array_equal(got["u"], got["v"])


@pytest.mark.parametrize("kwargs, needle", [
    ({"magic": b"NEMO_L1_RHS___2 "}, "bad magic"),
    ({"nz": NZ + 1}, "bad header"),
    ({"kt": 2}, "expected kt=1"),
    ({"level": 1}, "expected kt=1"),
    ({"values": np.full(2 * NX * NY * NZ, np.nan)}, "non-finite"),
])
def test_rhs_reader_refuses_a_record_it_must_not_believe(tmp_path, kwargs, needle):
    path = tmp_path / "oracle_rhs_kt00000001.bin"
    path.write_bytes(_rhs_bytes(**kwargs))
    with pytest.raises(gate.GateError, match=needle):
        gate.read_rhs(path, CASE)


def test_entry_reader_returns_v_and_refuses_the_wrong_time_level(tmp_path):
    count = NX * NY * NZ
    payload = np.arange(4 * count + NX * NY, dtype=np.float64)
    good = (b"NEMO_L1_ENTRY_1 "
            + struct.pack("=8i", 1, 1, 1, NX, NY, NZ, 2, 64)
            + payload.tobytes())
    path = tmp_path / "oracle_step_entry_kt00000001.bin"
    path.write_bytes(good)
    got = gate.read_entry_full(path, CASE)
    # The sweep gate's own reader drops v; the rest-precondition needs it.
    assert set(got) == {"T", "S", "u", "v", "ssh"}
    assert not np.array_equal(got["u"], got["v"])
    assert got["ssh"].shape == (NY - 4, NX - 4)

    bad = (b"NEMO_L1_ENTRY_1 "
           + struct.pack("=8i", 1, 2, 3, NX, NY, NZ, 2, 64)
           + payload.tobytes())
    path.write_bytes(bad)
    with pytest.raises(gate.GateError, match="expected kt=1"):
        gate.read_entry_full(path, CASE)
