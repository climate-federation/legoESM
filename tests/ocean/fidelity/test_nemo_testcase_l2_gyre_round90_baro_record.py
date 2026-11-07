"""Hermetic checks for the Round-90 correction-site acquisition package."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round90_baro_record",
    TESTCASES / "nemo_testcase_l2_gyre_round90_baro_record.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _payload(value: np.ndarray, rank: int) -> bytes:
    if rank == 2:
        return np.asarray(value).T.ravel(order="F").tobytes()
    return np.asarray(value).transpose(1, 0, 2).ravel(order="F").tobytes()


def _write_record(path: Path, *, wrong_owned_extent: bool = False) -> Path:
    nx, ny, nz = gate.DIMS
    owned_nx, owned_ny = nx - 4, ny - 4
    zeros3 = np.zeros((ny, nx, nz), dtype=np.float64)
    ones3 = np.ones((ny, nx, nz), dtype=np.float64)
    ones2 = np.ones((ny, nx), dtype=np.float64)
    owned2 = np.ones(
        (owned_ny - int(wrong_owned_extent), owned_nx), dtype=np.float64)
    values = {
        "baro_raw_u": zeros3, "baro_raw_v": zeros3,
        "baro_target_u": ones2, "baro_target_v": ones2,
        "baro_zub": owned2, "baro_zvb": owned2,
        "baro_e3u_0": ones3, "baro_e3v_0": ones3,
        "baro_r1_hu_0": ones2, "baro_r1_hv_0": ones2,
        "baro_umask": ones3, "baro_vmask": ones3,
        "baro_final_u": ones3, "baro_final_v": ones3,
    }
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.encode())
        handle.write(struct.pack(
            "=13i", 1, 2, 1, 1, nx, ny, nz, 30, 3, 34, 3, 24, 64))
        for name in sorted(values):
            value = values[name]
            rank = value.ndim
            n1, n2 = value.shape[1], value.shape[0]
            handle.write(name.ljust(16).encode())
            handle.write(struct.pack(
                "=4i", rank, n1, n2, 1 if rank == 2 else value.shape[2]))
            handle.write(_payload(value, rank))
    return path


def _args(tmp_path: Path, plant=None):
    record = _write_record(tmp_path / "record.bin")
    commit = "a" * 40
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    stamp = tmp_path / "record.bin.stamp"
    stamp.write_text(f"{digest} {commit} {record.name}\n")
    return SimpleNamespace(
        record=record, stamp=stamp, expect_commit=commit, plant=plant)


def test_record_reader_and_final_add_replay(tmp_path):
    report = gate.measure(_args(tmp_path))
    assert report["status"] == "READY"
    assert all(row["bit_exact"] for row in report["faces"].values())


def test_final_add_ulp_plant_fires(tmp_path):
    report = gate.measure(_args(tmp_path, plant="final-ulp"))
    assert report["status"] == "PLANT_FIRED"
    assert not report["faces"]["u"]["bit_exact"]


def test_record_reader_rejects_wrong_owned_correction_extent(tmp_path):
    record = _write_record(tmp_path / "record.bin", wrong_owned_extent=True)
    try:
        gate.read_record(record)
    except RuntimeError as error:
        assert "wrong shape for baro_zub" in str(error)
    else:
        raise AssertionError("wrong owned correction extent was accepted")


def test_source_card_is_additive_write_only_and_configuration_neutral():
    package = TESTCASES / "nemo_testcase_l2_gyre_round90_baro_correction"
    writer = (package / "l2_r90_baro.F90").read_text()
    patch = (package / "stprk3_stg_round90.patch").read_text()
    run = (package / "run.sh").read_text()
    assert "ACTION='WRITE'" in writer and "STATUS='REPLACE'" in writer
    assert "kt == nit000 + 1 .AND. kstg == 1" in writer
    assert not any(line.startswith("-") and not line.startswith("---")
                   for line in patch.splitlines())
    assert 'cmp "$SOURCE_ROOT/EXP00/namelist_cfg"' in run
    assert "--plant final-ulp" in run and "--plant-consumed" in run
