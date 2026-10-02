"""Fail-closed unit controls for the round-196 barotropic substep walk.

The record this walk reads is produced by NEMO, so the controls here build a
synthetic one in the SAME self-describing format and prove three things: the
reader strips NEMO's halo and transposes to the model's interior, the
checker refuses a record with a required operand missing, and the walk's
boundary order is the compiled order it claims to be.
"""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
          "nemo_testcase_l1_vortex_round196_spgts_walk.py")
SPEC = importlib.util.spec_from_file_location("round196_spgts_walk", SCRIPT)
assert SPEC and SPEC.loader
walk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(walk)

CHECKER = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
           "nemo_testcase_l1_vortex/check_records.py")
_CSPEC = importlib.util.spec_from_file_location("round196_checker", CHECKER)
assert _CSPEC and _CSPEC.loader
checker = importlib.util.module_from_spec(_CSPEC)
_CSPEC.loader.exec_module(checker)

JPI = JPJ = 7          # 3 interior cells plus the two-cell halo on each side
JPK = 2
ICYCLE = 2
NI = JPI - 4


def _f16(text: str) -> bytes:
    """A Fortran CHARACTER(16): blank-padded, never NUL-padded."""
    assert len(text) <= 16
    return text.ljust(16).encode("ascii")


def _group(name: str, values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.float64)
    if flat.ndim == 1:
        header = struct.pack("=16s4i", _f16(name), 1, flat.size, 1, 1)
    else:
        nx, ny = flat.shape
        header = struct.pack("=16s4i", _f16(name), 2, nx, ny, 1)
    return header + flat.astype("<f8").tobytes(order="F")


def _write_record(path: Path, *, drop: str | None = None) -> None:
    body = struct.pack("=16s15i", _f16("NEMO_L1_SPGTS1"), 1, 1, 1, 2, 3, 4,
                       JPI, JPJ, JPK, ICYCLE, 3, 3, 5, 5, 64)
    counter = 0.0
    frames = [("i000", walk.ENTRY_NAMES_FOR_TEST)]
    frames += [(f"j{jn:03d}", walk.SUBSTEP_NAMES_FOR_TEST)
               for jn in range(1, ICYCLE + 1)]
    frames.append(("o000", walk.EXIT_NAMES_FOR_TEST))
    for prefix, names in frames:
        for name in names:
            full = f"{prefix}_{name}"
            if full == drop:
                continue
            if name in ("wgtbtp1", "wgtbtp2"):
                payload = np.arange(3 * 4, dtype=np.float64)
            elif name in ("entry_sc", "ext_coef", "bck_coef", "sum_coef"):
                payload = np.arange({"entry_sc": 6, "ext_coef": 3,
                                     "bck_coef": 4, "sum_coef": 2}[name],
                                    dtype=np.float64)
            elif name in ("zu_frc", "zv_frc"):
                payload = np.arange(NI * NI, dtype=np.float64).reshape(NI, NI)
            else:
                counter += 1.0
                payload = (counter * 100.0
                           + np.arange(JPI * JPJ, dtype=np.float64)
                           ).reshape(JPI, JPJ)
            body += _group(full, payload)
    path.write_bytes(body)


def test_reader_strips_the_halo_and_transposes(tmp_path):
    _write_record(tmp_path / "oracle_spgts_kt00000001.bin")
    meta, groups = walk.read_spgts(tmp_path, 1)
    assert meta["icycle"] == ICYCLE and meta["jpi"] == JPI
    full = groups["j001_zhU"]
    assert full.shape == (NI, NI)
    # The interior must be the halo-stripped, transposed Fortran plane.
    assert groups["i000_zu_frc"].shape == (NI, NI)
    assert groups["j001_ext_coef"].shape == (3,)


def test_checker_refuses_a_record_with_a_required_operand_missing(tmp_path):
    _write_record(tmp_path / "oracle_spgts_kt00000001.bin",
                  drop="j002_cor_u")
    with pytest.raises(checker.Refusal) as caught:
        checker.parse_record(tmp_path / "oracle_spgts_kt00000001.bin")
    assert "missing group" in str(caught.value)


def test_checker_refuses_a_truncated_record(tmp_path):
    path = tmp_path / "oracle_spgts_kt00000001.bin"
    _write_record(path)
    with pytest.raises(checker.Refusal):
        checker.parse_record(path, plant="truncated")


def test_lego_plane_drops_the_redundant_west_face_and_south_row():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert walk._lego_plane(values, "u").shape == (3, 3)
    assert walk._lego_plane(values, "v").shape == (2, 4)
    assert walk._lego_plane(values, "t").shape == (3, 4)


def test_substep_order_is_the_compiled_order():
    names = [row[0] for row in walk.SUBSTEP_ORDER]
    assert names[:4] == ["entry.eta", "entry.u", "entry.v", "mid.u"]
    # The velocity update must come after every trend it reads.
    assert names.index("new.u") > names.index("trd.u") > names.index("cor.u")
    assert names.index("cor.u") > names.index("spg.u") > names.index("bck.eta")
    assert names.index("ssha") > names.index("flux.u") > names.index("mid.u")
