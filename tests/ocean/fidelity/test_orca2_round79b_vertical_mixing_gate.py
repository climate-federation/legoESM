"""Non-vacuity proof for the ORCA2 round-79b vertical-mixing admission gate.

A synthetic record in the writer's exact on-disk format is admitted, and every
plant the acquisition launcher runs is shown to make the gate refuse.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = (REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "orca2_l4"
             / "nemo_testcase_l4_orca2_round79b_vertical_mixing_gate.py")


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_l4_orca2_round79b_vertical_mixing_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()
JPI, JPJ, JPK = 6, 5, 6
COMMIT = "a" * 40


def _closure_scalars() -> np.ndarray:
    values = np.zeros((1, 1, JPK))
    values[0, 0, :5] = (0.1, 1.0e-3, 2.0, 1.0, 1.0)
    return values


def _field(stream, name: str, values: np.ndarray, rank: int) -> None:
    stream.write(name.ljust(16).encode("ascii"))
    n1, n2, n3 = values.shape
    stream.write(struct.pack("=7i", rank, n1, n2, n3, 1, 1, 1))
    stream.write(np.asfortranarray(values, dtype="=f8").tobytes(order="F"))


def _write_record(path: Path, kt: int, rank: int) -> None:
    rng = np.random.default_rng(1000 * rank + kt)
    shape = (JPI, JPJ, JPK)
    wmask = np.ones(shape)
    wmask[0, :, :] = 0.0
    tmask = wmask.copy()
    avt_k = rng.uniform(1e-6, 1e-4, shape)
    avm_k = rng.uniform(1e-6, 1e-4, shape)
    avt_tke = avt_k.copy()
    avt_tke[:, :, 0] = 0.0
    avt_tke[:, :, JPK - 1] = 0.0
    avt_rnf = avt_tke + 1e-7
    avt_evd = avt_rnf + 1e-8
    avm_evd = avm_k.copy()
    avt_ddm = avt_evd.copy()
    avs_ddm = avt_ddm * 0.9
    avm_ddm = avm_evd + 2e-7
    increment = rng.uniform(1.4e-7, 1e-4, shape) * wmask
    arrays = [
        ("avt_after_tke", avt_tke, 3),
        ("avm_after_tke", avm_k, 3),
        ("avt_after_rnf", avt_rnf, 3),
        ("avt_after_evd", avt_evd, 3),
        ("avm_after_evd", avm_evd, 3),
        ("avt_after_ddm", avt_ddm, 3),
        ("avs_after_ddm", avs_ddm, 3),
        ("avm_after_ddm", avm_ddm, 3),
        ("avt_after_iwm", avt_ddm + increment, 3),
        ("avm_after_iwm", avm_ddm + increment, 3),
        ("avs_after_iwm", avs_ddm + increment, 3),
        ("avt_k", avt_k, 3),
        ("avm_k", avm_k, 3),
        ("en", rng.uniform(1e-10, 1e-4, shape), 3),
        ("dissl", rng.uniform(1e-4, 1e-2, shape), 3),
        ("mxlm", rng.uniform(1e-3, 50.0, shape), 3),
        ("mxld", rng.uniform(1e-3, 50.0, shape), 3),
        ("wmask", wmask, 3),
        ("tmask", tmask, 3),
        ("rnfmsk", np.zeros((JPI, JPJ, 1)), 2),
        ("avtb_2d", np.ones((JPI, JPJ, 1)), 2),
        ("avtb", np.full((1, 1, JPK), 1.2e-5), 1),
        ("avmb", np.full((1, 1, JPK), 1.2e-4), 1),
        ("closure_scalars", _closure_scalars(), 1),
    ]
    assert tuple(name for name, _, _ in arrays) == GATE.EXPECTED_ORDER
    with path.open("wb") as stream:
        stream.write(GATE.MAGIC.ljust(16).encode("ascii"))
        stream.write(struct.pack(
            f"={GATE.HEADER_INTS}i", 1, kt, 1, 1, rank, 1, 1,
            JPI, JPJ, JPK, 64, len(arrays), 3, 1, 1))
        for name, values, rank_code in arrays:
            _field(stream, name, values, rank_code)


@pytest.fixture(scope="module")
def record_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("round79b")
    (root / "producer_commit.txt").write_text(COMMIT + "\n")
    for rank in GATE.RANKS:
        for step in range(1, GATE.N_STEPS + 1):
            _write_record(
                root / f"oracle_zdf_vmix_kt{step:08d}_r{rank:04d}.bin",
                step, rank)
    return root


def test_clean_record_is_admitted(record_root: Path) -> None:
    result = GATE.run_gate(record_root, expect_commit=COMMIT)
    assert result["status"] == "PASS"
    assert len(result["records"]) == len(GATE.RANKS) * GATE.N_STEPS
    first = result["records"]["oracle_zdf_vmix_kt00000001_r0000.bin"]
    assert first["ediff"] == pytest.approx(0.1)
    assert first["rmxl_min"] == pytest.approx(1.0e-3)
    assert first["iwm_avt_max"] > 0.0


@pytest.mark.parametrize(
    "plant", ["header", "field-order", "truncation", "stamp", "content"])
def test_every_plant_makes_the_gate_refuse(record_root: Path, plant: str) -> None:
    with pytest.raises(GATE.GateError):
        GATE.run_gate(record_root, expect_commit=COMMIT, plant=plant)


def test_a_missing_record_is_refused(record_root: Path, tmp_path: Path) -> None:
    thin = tmp_path / "thin"
    thin.mkdir()
    (thin / "producer_commit.txt").write_text(COMMIT + "\n")
    _write_record(thin / "oracle_zdf_vmix_kt00000001_r0000.bin", 1, 0)
    with pytest.raises(GATE.GateError):
        GATE.run_gate(thin, expect_commit=COMMIT)
