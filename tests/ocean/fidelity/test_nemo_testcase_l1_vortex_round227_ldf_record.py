"""Fail-closed controls for round 227's internal SMT-3 LDF record."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
CHECKER = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
           "nemo_testcase_l1_vortex/check_records.py")
_SPEC = importlib.util.spec_from_file_location("round227_checker", CHECKER)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

NX, NY, NZ = 5, 4, 3


def _f16(text: str) -> bytes:
    assert len(text) <= 16
    return text.ljust(16).encode("ascii")


def _group(name: str, rank: int, *, offset: float = 0.0) -> bytes:
    shape = {1: (NZ,), 2: (NX, NY), 3: (NX, NY, NZ)}[rank]
    values = np.arange(np.prod(shape), dtype=np.float64) + offset
    n1 = shape[0]
    n2 = shape[1] if rank >= 2 else 1
    n3 = shape[2] if rank == 3 else 1
    return (struct.pack("=16s4i", _f16(name), rank, n1, n2, n3)
            + values.astype("<f8").tobytes())


def _write(path: Path, family: str, *, drop: str | None = None) -> Path:
    if family == "slope":
        magic = "NEMO_L1_LDFSLP1"
        names = checker._LDF_SLOPE_GROUPS
        rank = lambda _name: 3
        stage = 0
    else:
        magic = "NEMO_L1_LDFISO1"
        names = checker._LDF_ISO_GROUPS
        rank = lambda name: (1 if name == "e3w_1d" else
                             2 if name.startswith(("r3", "e2_", "e1")) else 3)
        stage = 3
    names = [name for name in names if name != drop]
    body = b"".join(_group(name, rank(name), offset=index)
                    for index, name in enumerate(names))
    header = struct.pack(
        "=15i", 1, 1, stage, 1, 2, 0, 3, NX, NY, NZ,
        len(names), 0, 0, 0, 64,
    )
    path.write_bytes(_f16(magic) + header + body)
    return path


@pytest.mark.parametrize(
    ("family", "filename", "required"),
    [("slope", "oracle_ldf_slope_kt00000001.bin", "uslp"),
     ("iso", "oracle_ldf_iso_kt00000001.bin", "fw_upper")],
)
def test_internal_records_are_self_describing(tmp_path, family, filename, required):
    record = checker.parse_record(_write(tmp_path / filename, family))
    assert required in record["groups"]
    assert record["doubles"] > 0


def test_internal_record_refuses_a_missing_named_seam(tmp_path):
    path = _write(tmp_path / "oracle_ldf_iso_kt00000001.bin", "iso",
                  drop="A31")
    with pytest.raises(checker.Refusal, match="missing group"):
        checker.parse_record(path)


@pytest.mark.parametrize("plant", ["header", "field-name", "truncated"])
def test_each_internal_record_plant_refuses(tmp_path, plant):
    path = _write(tmp_path / "oracle_ldf_iso_kt00000001.bin", "iso")
    with pytest.raises(checker.Refusal):
        checker.parse_record(path, corrupt_header=plant == "header",
                             plant=None if plant == "header" else plant)


def test_instruments_only_add_lines_and_cover_compiled_order():
    root = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
            "nemo_testcase_l1_vortex")
    names = ["ldfslp_r227_internal_record.patch",
             "traldf_iso_r227_internal_record.patch",
             "traldf_iso_scheme_r227_internal_record.patch"]
    patches = [root.joinpath(name).read_text() for name in names]
    for text in patches:
        removed = [line for line in text.splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        assert removed == []
    slope, iso, scheme = patches
    assert slope.index("r227_raw_u") < slope.index("r227_bound_u")
    assert iso.index("CALL traldf_iso_a33") < iso.index(
        "CALL r227_iso_begin( pt, Krhs )")
    assert scheme.index("r227_dit") < scheme.index("r227_A11")
    assert scheme.index("r227_A11") < scheme.index("r227_fu")
    assert scheme.index("r227_fu") < scheme.index("r227_A31")
    assert scheme.index("r227_A31") < scheme.index("r227_fw_upper")


def test_acquisition_stamps_both_internal_records_fail_closed():
    script = (ROOT / "scripts/validate/ocean_fidelity/testcases/"
              "nemo_testcase_l1_vortex/run.sh").read_text()
    assert "oracle_ldf_slope_kt00000001.bin.stamp" not in script
    assert "for record in oracle_ldf_slope_kt00000001.bin oracle_ldf_iso_kt00000001.bin" in script
    assert "fail-closed commit stamp disagrees" in script
