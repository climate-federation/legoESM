from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
          / "testcases" / "nemo_testcase_l2_gyre_round54_tke_operands.py")
SPEC = importlib.util.spec_from_file_location("round54_tke_operands", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(path: Path) -> None:
    parts = [MODULE.MAGIC, struct.pack("=13i", 2, 2, 1, 2, 32, 22, 31, 30,
                                      1, 32, 1, 22, 64)]
    shape = (32, 22, 31)
    for name in MODULE.FIELDS:
        parts.append(name.ljust(16).encode("ascii"))
        if name in {"rn_Dt", "rn_ediff", "rn_ediss", "rn_ebb", "rn_emin",
                    "rn_emin0", "rmxl_min", "rn_mxl0", "rn_bshear", "rn_lc",
                    "nn_pdl", "nn_mxl", "ln_mxl0", "nn_etau", "nn_htau",
                    "nn_eice", "ln_lc"}:
            values = {
                "rn_Dt": 14400.0, "rn_ediff": 0.1, "rn_ediss": 0.7,
                "rn_ebb": 67.83, "rn_emin": 1.0e-6, "rn_emin0": 1.0e-4,
                "rmxl_min": 0.01, "rn_mxl0": 0.01, "rn_bshear": 1.0e-20,
                "rn_lc": 0.15, "nn_pdl": 1.0, "nn_mxl": 3.0,
                "ln_mxl0": 1.0, "nn_etau": 0.0, "nn_htau": 1.0,
                "nn_eice": 0.0, "ln_lc": 1.0,
            }
            value = values[name]
            parts.extend((struct.pack("=4i", 0, 1, 1, 1),
                          struct.pack("=d", value)))
        elif name == "taum_entry":
            parts.extend((struct.pack("=4i", 2, 32, 22, 1),
                          np.zeros((32, 22), dtype="=f8").tobytes(order="F")))
        else:
            parts.append(struct.pack("=4i", 3, *shape))
            parts.append(np.zeros(shape, dtype="=f8").tobytes(order="F"))
    path.write_bytes(b"".join(parts))


def test_reader_accepts_complete_schema(tmp_path):
    path = tmp_path / "record.bin"
    _record(path)
    result = MODULE.read_record(path)
    assert result["header"]["kt"] == 2
    assert tuple(result["arrays"]["avt_pre_evd"].shape) == (32, 22, 31)
    assert tuple(result["arrays"]["taum_entry"].shape) == (32, 22)


@pytest.mark.parametrize(
    "plant", ["header", "truncation", "nan", "config", "copy", "shape"])
def test_reader_plants_fail(tmp_path, plant):
    path = tmp_path / "record.bin"
    _record(path)
    with pytest.raises(MODULE.GateError):
        MODULE.read_record(path, plant=plant)


def test_stamp_plant_exits_nonzero_before_report_emission(tmp_path):
    path = tmp_path / "record.bin"
    producer = tmp_path / "producer_commit.txt"
    _record(path)
    producer.write_text("1" * 40 + "\n")
    assert MODULE.main([
        "--record", str(path),
        "--expect-commit", "1" * 40,
        "--producer-commit", str(producer),
        "--plant", "stamp",
    ]) == 1
