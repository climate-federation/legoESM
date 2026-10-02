"""Fail-closed unit controls for round 200's flux-card stage-1 record.

The record itself is produced by NEMO, so these controls build a synthetic
one in the SAME self-describing format and prove: the checker admits a
well-formed flux stage record, it REFUSES one with a required operand
missing, it refuses a stage it does not know, the walk's reader strips
NEMO's halo and transposes to the model's interior, and the boundaries the
walk names are the compiled stage-1 order it claims.
"""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
          "nemo_testcase_l1_vortex_round200_flux_stage1.py")
_SPEC = importlib.util.spec_from_file_location("round200_flux_stage1", SCRIPT)
assert _SPEC and _SPEC.loader
walk = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(walk)

CHECKER = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
           "nemo_testcase_l1_vortex/check_records.py")
_CSPEC = importlib.util.spec_from_file_location("round200_checker", CHECKER)
assert _CSPEC and _CSPEC.loader
checker = importlib.util.module_from_spec(_CSPEC)
_CSPEC.loader.exec_module(checker)

# DELIBERATELY NOT SQUARE: a square plane cannot tell a transpose from an
# identity, so these controls could not otherwise fail for the reason their
# names give.
JPI, JPJ, JPK = 8, 7, 3
NI, NJ = JPI - 4, JPJ - 4


def _f16(text: str) -> bytes:
    assert len(text) <= 16
    return text.ljust(16).encode("ascii")


def _group(name: str, values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        nx, ny = arr.shape
        head = struct.pack("=16s4i", _f16(name), 2, nx, ny, 1)
    else:
        nx, ny, nz = arr.shape
        head = struct.pack("=16s4i", _f16(name), 3, nx, ny, nz)
    return head + arr.astype("<f8").tobytes(order="F")


def _write(path: Path, stage: int, *, drop: str | None = None,
           ngrps: int | None = None) -> Path:
    names = list(checker._STAGE_FLUX_BY_STAGE[stage])
    if drop:
        names.remove(drop)
    payload = b""
    counter = 1.0
    for name in names:
        if name == "ssh_kmm":
            values = np.arange(JPI * JPJ, dtype=np.float64).reshape(
                (JPI, JPJ), order="F") + counter
        else:
            values = np.arange(JPI * JPJ * JPK, dtype=np.float64).reshape(
                (JPI, JPJ, JPK), order="F") + counter
        payload += _group(name, values)
        counter += 1000.0
    declared = len(names) if ngrps is None else ngrps
    header = (_f16("NEMO_L1_STGFLX1")
              + struct.pack("=15i", 1, 1, stage, 1, 2, 3, 4,
                            JPI, JPJ, JPK, declared, 0, 0, 0, 64))
    path.write_bytes(header + payload)
    return path


def test_the_checker_admits_a_well_formed_flux_stage_record(tmp_path):
    for stage in (1, 2, 3):
        path = _write(tmp_path / f"oracle_stage_flux_terms_kt00000001_s{stage}.bin",
                      stage)
        parsed = checker.parse_record(path)
        assert parsed["stage"] == stage
        assert set(parsed["groups"]) == set(
            checker._STAGE_FLUX_BY_STAGE[stage])


def test_the_checker_refuses_a_missing_operand(tmp_path):
    path = _write(tmp_path / "oracle_stage_flux_terms_kt00000001_s1.bin", 1,
                  drop="zfw", ngrps=14)
    with pytest.raises(checker.Refusal, match="missing group"):
        checker.parse_record(path)


def test_the_checker_refuses_a_stage_it_does_not_know(tmp_path):
    path = _write(tmp_path / "oracle_stage_flux_terms_kt00000001_s1.bin", 1)
    raw = bytearray(path.read_bytes())
    raw[16 + 8:16 + 12] = struct.pack("=i", 4)      # header[2] = stage
    path.write_bytes(bytes(raw))
    with pytest.raises(checker.Refusal, match="unsupported stage"):
        checker.parse_record(path)


def test_the_checker_refuses_a_group_count_that_disagrees(tmp_path):
    path = _write(tmp_path / "oracle_stage_flux_terms_kt00000001_s1.bin", 1,
                  ngrps=99)
    with pytest.raises(checker.Refusal, match="declares 99 groups"):
        checker.parse_record(path)


def test_the_reader_strips_the_halo_and_transposes(tmp_path):
    _write(tmp_path / "oracle_stage_flux_terms_kt00000001_s1.bin", 1)
    groups = walk.read_flux_stage_terms(tmp_path, 1)
    assert set(groups) == set(checker._STAGE_FLUX_BY_STAGE[1])
    assert groups["base_u"].shape == (NJ, NI, JPK)
    assert groups["ssh_kmm"].shape == (NJ, NI)
    # The halo strip must take NEMO's interior, not the first NI x NJ block.
    raw = np.arange(JPI * JPJ * JPK, dtype=np.float64).reshape(
        (JPI, JPJ, JPK), order="F")
    offset = 1000.0 * list(checker._STAGE_FLUX_BY_STAGE[1]).index("base_u")
    expected = (raw + 1.0 + offset)[2:-2, 2:-2].transpose(1, 0, 2)
    assert np.array_equal(groups["base_u"], expected)


def test_the_walk_names_nemos_stage1_order():
    # the transports come before the continuity solve's ww, which comes
    # which comes before the stage output; a reordering here would report the
    # wrong statement as the first non-bit producer.
    assert walk.PLANTS == ("base.u", "base.v", "zfu", "zfv", "zfw", "ww",
                           "out.u", "out.v")
    assert walk.CASE == "VORTEX-zco"


def test_the_seam_control_refuses_an_inert_exposure():
    # The control the walk relies on to know its WRITE-only hooks are live.
    # Shown to FAIL on exactly the thing it guards against, and to stay
    # silent on a live seam, so it is not a tripwire that cannot trip.
    plain = np.arange(12.0).reshape(3, 4)
    with pytest.raises(Exception, match="the seam is inert"):
        walk.require_live("inert", "u", plain, plain)
    live = plain.copy()
    live[0, 0] += 1.0
    walk.require_live("live", "u", live, plain)
