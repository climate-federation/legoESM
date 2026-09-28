"""Guards for the round-29 GYRE ``dyn_zdf`` matrix reader and gate.

The record these read does not exist yet -- it arrives when the user runs
``nemo_testcase_l2_gyre_round29_zdf/run.sh``.  What is guarded here is
everything that does not need NEMO: that the reader refuses a malformed or
short record instead of returning a plausible one, that the gate goes GREEN on
a self-consistent record, and that it goes RED when one operand moves by a
single ulp.

The green arm is deliberately CIRCULAR -- the fixture's matrix and solve are
filled by the same functions the gate scores with -- so it proves the
round-trip and that the gate CAN pass, and nothing about the transcription.
The non-circular calibration was taken once, outside CI, against the actual
Fortran writer compiled by gfortran: all six matrix rows and both solve rows
came back 0 bit-unequal.  It is recorded in the round-29 receipt with its
SHA-256, because reproducing it needs a Fortran compiler and CI has none.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round29_zdf_matrix",
    TESTCASES / "nemo_testcase_l2_gyre_round29_zdf_matrix.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

JPI, JPJ, JPK, JPKM1 = 6, 5, 4, 3
NTSI, NTEI, NTSJ, NTEJ = 2, 4, 2, 3


def _emit(handle, name: str, rank: int, extents, payload) -> None:
    handle.write(f"{name:<16}".encode("ascii"))
    handle.write(struct.pack("=4i", rank, *extents))
    handle.write(np.asarray(payload, dtype="<f8").tobytes(order="F"))


def _write_record(path: Path, arrays: dict) -> None:
    """Emit the writer's byte layout: magic, 16 header ints, then groups."""
    with path.open("wb") as handle:
        handle.write(f"{gate.MAGIC:<16}".encode("ascii"))
        handle.write(struct.pack(
            "=16i", 1, 1, 3, 1, 2, 3, 3, JPI, JPJ, JPK, JPKM1,
            NTSI, NTEI, NTSJ, NTEJ, 64))
        for name in gate.EXPECTED_ARRAYS:
            value = arrays[name]
            if np.isscalar(value):
                _emit(handle, name, 0, (1, 1, 1), [value])
            elif np.asarray(value).ndim == 2:
                _emit(handle, name, 2, (JPI, JPJ, 1), value)
            else:
                _emit(handle, name, 3, np.asarray(value).shape, value)


def _operands() -> dict:
    """A small, entirely made-up but PHYSICALLY SHAPED set of operands.

    Thicknesses and ``avm`` are positive and the drag rate is negative, as
    NEMO's are, so the tridiagonal system is diagonally dominant and the
    recurrences do not divide by anything near zero.
    """
    rng = np.random.default_rng(20260905)
    d3 = (JPI, JPJ, JPK)
    a = {
        "avm": rng.uniform(1e-4, 1e-2, d3),
        "e3u_Kaa": rng.uniform(10.0, 400.0, d3),
        "e3uw_Kmm": rng.uniform(10.0, 400.0, d3),
        "e3v_Kaa": rng.uniform(10.0, 400.0, d3),
        "e3vw_Kmm": rng.uniform(10.0, 400.0, d3),
        "wumask": np.ones(d3), "wvmask": np.ones(d3),
        "umask": np.ones(d3), "vmask": np.ones(d3),
        "rCdU_bot": -rng.uniform(1e-4, 1e-3, (JPI, JPJ)),
        "mbku": np.full((JPI, JPJ), float(JPKM1)),
        "mbkv": np.full((JPI, JPJ), float(JPKM1)),
        "utauU": rng.normal(0.0, 0.1, (JPI, JPJ)),
        "vtauV": rng.normal(0.0, 0.1, (JPI, JPJ)),
        "rDt": 1200.0, "rho0": 1026.0,
        "uu_Kaa_pre": rng.normal(0.0, 0.05, d3),
        "vv_Kaa_pre": rng.normal(0.0, 0.05, d3),
    }
    for name in ("uu_Krhs_in", "vv_Krhs_in", "uu_Kbb_in", "vv_Kbb_in"):
        a[name] = rng.normal(0.0, 1e-6, d3)
    for name in ("uu_b_Kaa", "vv_b_Kaa"):
        a[name] = rng.normal(0.0, 1e-3, (JPI, JPJ))
    for name in ("zwi_u", "zwd_u", "zws_u", "zwi_v", "zwd_v", "zws_v",
                 "uu_Kaa_out", "vv_Kaa_out"):
        a[name] = np.zeros(d3 if name.endswith("out") else (JPI, JPJ, JPKM1))
    return a


def _self_consistent_record(path: Path) -> None:
    """Write a record whose matrix and solve are what its operands imply."""
    arrays = _operands()
    _write_record(path, arrays)
    rec = gate.read_zdf_matrix(path)
    isl, jsl = gate._interior(rec["header"])
    for face in ("u", "v"):
        zwi, zwd, zws = gate.nemo_matrix(rec, face)
        for tag, built in (("zwi", zwi), ("zwd", zwd), ("zws", zws)):
            arrays[f"{tag}_{face}"][isl, jsl, :] = built
    _write_record(path, arrays)
    rec = gate.read_zdf_matrix(path)
    for face, out in (("u", "uu_Kaa_out"), ("v", "vv_Kaa_out")):
        arrays[out][isl, jsl, :JPKM1] = gate.nemo_solve(rec, face)
    _write_record(path, arrays)


def test_reader_reads_every_array_and_reaches_exactly_eof(tmp_path):
    path = tmp_path / "rec.bin"
    _write_record(path, _operands())
    rec = gate.read_zdf_matrix(path)
    assert rec["order"] == list(gate.EXPECTED_ARRAYS)
    assert rec["header"]["jpi"] == JPI and rec["header"]["jpkm1"] == JPKM1
    assert rec["arrays"]["rDt"] == 1200.0
    assert rec["arrays"]["avm"].shape == (JPI, JPJ, JPK)
    assert rec["arrays"]["rCdU_bot"].shape == (JPI, JPJ)


def test_reader_refuses_a_short_record_rather_than_reporting_on_what_it_got(tmp_path):
    """A missing array must be a hard failure, not a silently smaller arm."""
    path = tmp_path / "short.bin"
    arrays = _operands()
    with path.open("wb") as handle:
        handle.write(f"{gate.MAGIC:<16}".encode("ascii"))
        handle.write(struct.pack("=16i", 1, 1, 3, 1, 2, 3, 3, JPI, JPJ, JPK,
                                 JPKM1, NTSI, NTEI, NTSJ, NTEJ, 64))
        _emit(handle, "avm", 3, (JPI, JPJ, JPK), arrays["avm"])
    with pytest.raises(SystemExit):
        gate.read_zdf_matrix(path)


@pytest.mark.parametrize("corrupt", ["magic", "truncate", "extent"])
def test_reader_refuses_a_malformed_record(tmp_path, corrupt):
    path = tmp_path / "bad.bin"
    _write_record(path, _operands())
    raw = bytearray(path.read_bytes())
    if corrupt == "magic":
        raw[:16] = b"NEMO_L2_NOTMINE "
    elif corrupt == "truncate":
        del raw[-64:]
    else:                                     # a lying extent
        raw[80 + 16:80 + 20] = struct.pack("=i", 3)
        raw[80 + 20:80 + 24] = struct.pack("=i", JPI + 1)
    path.write_bytes(bytes(raw))
    with pytest.raises(SystemExit):
        gate.read_zdf_matrix(path)


def test_gate_is_green_on_a_self_consistent_record(tmp_path):
    path = tmp_path / "ok.bin"
    _self_consistent_record(path)
    report = gate.run(path)
    assert report["status"] == "AT-BAR"
    # ROUND 37: eight rows became ten.  The three rebuilt-matrix rows and the
    # NumPy calibration solve per face are joined by zdf_solve_lego, which
    # drives legoESM's OWN ordered sweep on the same dumped operands -- the
    # momentum half of the Rule-12 discharge for the shared solve, and the
    # only such measurement the two tank cards have.
    assert len(report["calibration_rows"]) == 10
    assert {r["name"].rsplit(".", 2)[-2] + "." + r["name"].rsplit(".", 1)[-1]
            for r in report["calibration_rows"]
            if "solve_lego" in r["name"]} == {
                "zdf_solve_lego.u", "zdf_solve_lego.v"}
    assert all(r["bit_unequal"] == 0 for r in report["calibration_rows"])
    assert report["legoesm_comparison"]["status"] == "UNMEASURED"


def test_one_ulp_on_one_operand_turns_the_gate_red(tmp_path):
    """Non-vacuity: without this the green arm above proves nothing."""
    path = tmp_path / "ok.bin"
    _self_consistent_record(path)
    planted = gate.run(path, plant=True)
    assert planted["status"] == "DEBT"
    moved = [r for r in planted["calibration_rows"] if r["bit_unequal"] > 0]
    assert moved, "the planted ulp moved no row"
    # ROUND 37: the two new solve rows are named ".u"/".v", not "_u"/"_v",
    # so the per-face assertion takes both spellings rather than silently
    # excusing a row it does not recognise.
    assert all(r["name"].split(".")[-1].endswith(("_u", "_v", "u", "v"))
               for r in moved)


def test_unknown_face_raises_rather_than_defaulting(tmp_path):
    path = tmp_path / "ok.bin"
    _self_consistent_record(path)
    rec = gate.read_zdf_matrix(path)
    with pytest.raises(SystemExit):
        gate.nemo_matrix(rec, "w")
    with pytest.raises(SystemExit):
        gate.nemo_solve(rec, "w")


def test_main_exit_code_carries_the_bit_bar(tmp_path, capsys):
    path = tmp_path / "ok.bin"
    _self_consistent_record(path)
    assert gate.main(["--record", str(path)]) == 0
    assert gate.main(["--record", str(path), "--plant"]) == 1
