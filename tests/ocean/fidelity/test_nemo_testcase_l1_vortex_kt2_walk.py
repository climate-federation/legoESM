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
    _PLANTS, GateError, plant_moved, read_bt_frame, read_rhs, read_stage,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "scripts/validate/ocean_fidelity/testcases"
                       / "nemo_testcase_l1_vortex"))

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


# --------------------------------------------------------------------------
# The pre-stage momentum right-hand side reader (round 4).  Same rule: parse
# the header, check the payload against what that header itself asks for.
def _rhs_bytes(tmp_path, magic=b"NEMO_L1_RHS___1 ", extra=0, step=1):
    payload = np.arange(2 * _NX * _NY * _NZ + extra, dtype=np.float64)
    path = tmp_path / "oracle_rhs_kt00000001.bin"
    path.write_bytes(
        magic + struct.pack("=7i", 1, step, 3, _NX, _NY, _NZ, 64)
        + payload.tobytes())
    return path


def test_a_well_formed_rhs_record_parses_to_its_own_declared_shape(tmp_path):
    record = read_rhs(_rhs_bytes(tmp_path), expect_step=1)
    assert record["u"].shape == (_NY - 4, _NX - 4, _NZ)
    assert record["v"].shape == (_NY - 4, _NX - 4, _NZ)
    assert record["Krhs"] == 3


@pytest.mark.parametrize("kwargs, message", [
    ({"magic": b"NEMO_L1_STAGE_1 "}, "bad magic"),
    ({"step": 4}, "header says step 4"),
    ({"extra": 3}, "asks for"),
])
def test_a_malformed_rhs_record_is_refused(tmp_path, kwargs, message):
    with pytest.raises(GateError, match=message):
        read_rhs(_rhs_bytes(tmp_path, **kwargs), expect_step=1)


# --------------------------------------------------------------------------
# Round 4's acquisition checker for the per-term record.  The record is
# self-describing (note BD): the parser walks its groups and may predict
# nothing, so each malformation below has to be caught by the record's OWN
# declarations disagreeing with its bytes.
def _group(name, rank, n1, n2, n3, count=None):
    if count is None:
        count = n1 * n2 * (n3 if rank == 3 else 1)
    return (name.encode().ljust(16) + struct.pack("=4i", rank, n1, n2, n3)
            + np.arange(count, dtype=np.float64).tobytes())


def _rhsterm_bytes(tmp_path, *, magic=b"NEMO_L1_RHSTRM1 ", groups=None,
                   declared=None, trailing=b"", short=False, rank=3):
    names = groups if groups is not None else (
        "uu_rhs", "vv_rhs", "ww", "r3t_Kaa")
    body = b""
    for name in names:
        if name == "r3t_Kaa":
            body += _group(name, 2, _NX, _NY, 1)
        else:
            body += _group(name, rank, _NX, _NY, _NZ)
    # version, step, Kbb, Kmm, Kaa, Krhs, jpi, jpj, jpk, group count,
    # four spare, and the 64-bit word size LAST -- fifteen, the count the
    # committed writer emits.
    header = struct.pack(
        "=15i", 1, 1, 2, 3, 4, 5, _NX, _NY, _NZ,
        len(names) if declared is None else declared, 0, 0, 0, 0, 64)
    raw = magic + header + body + trailing
    if short:                      # truncate the LAST group's payload
        raw = raw[:-16]
    path = tmp_path / "oracle_rhsterm_kt00000001_vor.bin"
    path.write_bytes(raw)
    return path


def test_a_well_formed_per_term_record_parses_every_group(tmp_path):
    from check_records import parse_record

    record = parse_record(_rhsterm_bytes(tmp_path))
    assert sorted(record["groups"]) == ["r3t_Kaa", "uu_rhs", "vv_rhs", "ww"]
    assert record["groups"]["uu_rhs"]["shape"] == [_NX, _NY, _NZ]
    assert record["groups"]["r3t_Kaa"]["shape"] == [_NX, _NY]


@pytest.mark.parametrize("kwargs, message", [
    ({"magic": b"NEMO_L1_RHS___1 "}, "is not"),
    ({"groups": ("uu_rhs", "vv_rhs", "ww")}, "missing group"),
    ({"declared": 9}, "declares 9 groups"),
    ({"trailing": b"\x00" * 8}, "a group header is truncated"),
    ({"trailing": _group("spare", 3, 2, 2, 2)},
     "missing group|unexpected group|declares|truncated|remain"),
    ({"short": True}, "only"),
    ({"rank": 7}, "has rank 7"),
])
def test_a_malformed_per_term_record_is_refused(tmp_path, kwargs, message):
    from check_records import Refusal, parse_record

    with pytest.raises(Refusal, match=message):
        parse_record(_rhsterm_bytes(tmp_path, **kwargs))


# --------------------------------------------------------------------------
# Round 8 extends the SAME checker with one self-describing record per live
# momentum stage.  Stage 2 has an explicit-update boundary; stage 3 has LDF
# and the implicit ZDF output instead.  The parser, not a size formula, owns
# both layouts.
def _stage_term_bytes(tmp_path, stage=2, *, names=None, declared=None,
                      duplicate=False):
    from check_records import _STAGE_TERM_BY_STAGE

    group_names = list(names or _STAGE_TERM_BY_STAGE[stage])
    if duplicate:
        group_names[-1] = group_names[0]
    body = b""
    for name in group_names:
        if name == "ssh_kmm":
            body += _group(name, 2, _NX, _NY, 1)
        else:
            body += _group(name, 3, _NX, _NY, _NZ)
    header = struct.pack(
        "=15i", 1, 1, stage, 1, 2, 3, 4, _NX, _NY, _NZ,
        len(group_names) if declared is None else declared, 0, 0, 0, 64)
    path = tmp_path / f"oracle_stage_terms_kt00000001_s{stage}.bin"
    path.write_bytes(b"NEMO_L1_STGTRM1 " + header + body)
    return path


@pytest.mark.parametrize("stage, boundary", [(2, "update_u"), (3, "zdf_u")])
def test_a_well_formed_stage_term_record_parses_its_stage_layout(
        tmp_path, stage, boundary):
    from check_records import parse_record

    record = parse_record(_stage_term_bytes(tmp_path, stage))
    assert record["stage"] == stage
    assert boundary in record["groups"]
    assert record["groups"]["ssh_kmm"]["rank"] == 2
    assert record["groups"]["ww"]["shape"] == [_NX, _NY, _NZ]


@pytest.mark.parametrize("kwargs, message", [
    ({"names": ("kmm_u",)}, "missing group"),
    ({"declared": 99}, "declares 99 groups"),
    ({"duplicate": True}, "duplicate group name"),
])
def test_a_malformed_stage_term_record_is_refused(tmp_path, kwargs, message):
    from check_records import Refusal, parse_record

    with pytest.raises(Refusal, match=message):
        parse_record(_stage_term_bytes(tmp_path, 2, **kwargs))


def test_the_stage_term_extent_plant_is_fail_closed(tmp_path):
    from check_records import Refusal, parse_record

    with pytest.raises(Refusal):
        parse_record(_stage_term_bytes(tmp_path, 3), corrupt_header=True)


# --------------------------------------------------------------------------
# The plant DISCRIMINATION rule.  Round 3 shipped a walk whose plant perturbed
# arm 0's scoring only, so it passed even with both substitution hooks dead.
# The rule that replaced it -- each plant names the arm its substitution
# feeds, and the plant is VISIBLE only when THAT arm moves -- is a pure
# function, so it gets a test rather than only a shell log.
_PLANT_ARM = {
    "score": "0_card",
    "entry": "1_nemo_entry",
    "external": "2_nemo_external",
    "stage1": "3_nemo_stage1_out",
    "stage2": "4_nemo_stage2_out",
    "prestage_rhs": "5_nemo_prestage_rhs",
}


def _report(**velocities):
    return {"arms": [{"arm": name, "kt2": {"u": value, "v": value}}
                     for name, value in velocities.items()]}


def test_every_plant_names_an_arm_its_substitution_feeds():
    assert sorted(_PLANTS) == sorted(_PLANT_ARM)


@pytest.mark.parametrize("plant", sorted(_PLANT_ARM))
def test_a_plant_is_visible_only_when_its_own_arm_moves(plant):
    baseline = {name: 1.0e-7 for name in _PLANT_ARM.values()}
    baseline["6_prestage_rhs_only"] = 1.0e-7
    clean = _report(**baseline)

    moved = dict(baseline)
    moved[_PLANT_ARM[plant]] = 1.0
    assert plant_moved(plant, clean, _report(**moved)) is True

    # An INERT hook: every other arm moves, the named one does not.  This is
    # exactly round 3's failure mode, and it must read NOT VISIBLE.
    elsewhere = {name: (1.0 if name != _PLANT_ARM[plant] else 1.0e-7)
                 for name in baseline}
    assert plant_moved(plant, clean, _report(**elsewhere)) is False
