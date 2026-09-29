"""The VORTEX kt=2 walk's record readers refuse rather than slice silently.

Five acquisitions in this campaign were refused by checkers that predicted a
record's size by hand (operator note BD), so these readers parse the header and
check the payload against what that header itself asks for.  The tests below
plant each failure mode: a wrong magic, a wrong step, and a payload that does
not match the record's own declared shape.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "scripts/validate/ocean_fidelity/testcases"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    GateError, read_bt_frame, read_stage,
)

_NX, _NY, _NZ, _NTR = 9, 8, 3, 2


def _stage_bytes(tmp_path, magic=b"NEMO_L1_STAGE_1 ", extra=0, step=1):
    cell = _NX * _NY * _NZ
    payload = np.arange(_NTR * cell + 2 * cell + _NX * _NY + extra,
                        dtype=np.float64)
    path = tmp_path / "oracle_stage_kt00000001_s2.bin"
    path.write_bytes(
        magic + struct.pack("=9i", 1, step, 2, 3, _NX, _NY, _NZ, _NTR, 64)
        + payload.tobytes())
    return path


def _frame_bytes(tmp_path, extra=0):
    payload = np.arange(4 * _NX * _NY + extra, dtype=np.float64)
    path = tmp_path / "oracle_bt_frames_kt00000001.bin"
    path.write_bytes(
        b"NEMO_L1_BTFRM_1 " + struct.pack("=6i", 1, 1, 3, _NX, _NY, 64)
        + payload.tobytes())
    return path


def test_a_well_formed_stage_record_parses_to_its_own_declared_shape(tmp_path):
    record = read_stage(_stage_bytes(tmp_path), expect_step=1, expect_stage=2)
    interior = (_NY - 4, _NX - 4, _NZ)
    assert record["T"].shape == interior
    assert record["ssh"].shape == interior[:2]
    assert record["Kaa"] == 3


@pytest.mark.parametrize("kwargs, message", [
    ({"magic": b"NEMO_L1_ENTRY_1 "}, "bad magic"),
    ({"step": 7}, "header says step 7"),
    ({"extra": 5}, "asks for"),
])
def test_a_malformed_stage_record_is_refused(tmp_path, kwargs, message):
    with pytest.raises(GateError, match=message):
        read_stage(_stage_bytes(tmp_path, **kwargs),
                   expect_step=1, expect_stage=2)


def test_the_barotropic_frame_reader_parses_and_refuses(tmp_path):
    record = read_bt_frame(_frame_bytes(tmp_path), expect_step=1)
    for name in ("uu_b", "vv_b", "un_adv", "vn_adv"):
        assert record[name].shape == (_NY - 4, _NX - 4)
    with pytest.raises(GateError, match="asks for"):
        read_bt_frame(_frame_bytes(tmp_path, extra=3), expect_step=1)
