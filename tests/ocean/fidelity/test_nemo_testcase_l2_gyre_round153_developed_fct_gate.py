from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l2_gyre_round153_developed_fct_gate as gate  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.ljust(16).encode("ascii"))
        handle.write(struct.pack("=14i", *gate.HEADER))
        for name, rank, shape, origin in gate.EXPECTED_ROWS:
            values = np.zeros(shape, dtype="=f8", order="F")
            if name == "p2dt":
                values[...] = 14400.0
            elif name in ("tmask", "wmask"):
                values[...] = 1.0
            elif name.startswith("coef_"):
                values[...] = 1.0
                values.flat[0] = 0.5
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(struct.pack("=7i", rank, *shape, *origin))
            handle.write(values.tobytes(order="F"))


def _tree(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    root.mkdir()
    baseline.mkdir()
    commit = "a" * 40
    _write_record(root / gate.RECORD)
    (root / "producer_commit.txt").write_text(commit + "\n")
    (root / "nemo").write_bytes(b"binary")
    (root / "binary.sha256").write_text(f"{_sha256(root / 'nemo')}  nemo\n")
    for name, payload in (
        ("mesh_mask.nc", b"mesh"),
        ("GYRE_OMIP_L2_P3_00001081_restart.nc", b"restart"),
        ("oracle_process_budget_kt00001081.bin", b"process"),
    ):
        (baseline / name).write_bytes(payload)
        (root / name).write_bytes(payload)
    for name in gate.EXPECTED_CHANGED_DIAGNOSTICS:
        (baseline / name).write_bytes(b"before")
        (root / name).write_bytes(b"after")
    manifest = "".join(
        f"{_sha256(root / name)} {name}\n"
        for name in gate.inherited_names(baseline)
    )
    (root / "round153_inherited.sha256").write_text(manifest)
    (root / f"{gate.RECORD}.stamp").write_text(
        f"{_sha256(root / gate.RECORD)} {commit} {gate.RECORD}\n")
    return root, baseline, commit


def _args(root: Path, baseline: Path, commit: str, output: Path) -> list[str]:
    return [
        "--root", str(root), "--baseline", str(baseline),
        "--expect-commit", commit, "--output", str(output),
    ]


def test_round153_gate_accepts_complete_record(tmp_path: Path) -> None:
    root, baseline, commit = _tree(tmp_path)
    assert gate.main(_args(root, baseline, commit, tmp_path / "report.json")) == 0


@pytest.mark.parametrize(
    "plant",
    ("stamp", "truncation", "missing-field", "coefficients-one",
     "inherited-byte", "restart-byte"),
)
def test_round153_gate_plants_exit_nonzero(
    tmp_path: Path, plant: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root, baseline, commit = _tree(tmp_path)
    argv = _args(root, baseline, commit, tmp_path / f"{plant}.json")
    assert gate.main([*argv, "--plant", plant]) == 1
    assert f"STATUS PLANT-FIRED: {plant}" in capsys.readouterr().err


def test_round153_inherited_registry_is_source_derived(tmp_path: Path) -> None:
    root, baseline, commit = _tree(tmp_path)
    (baseline / "oracle_extra.bin").write_bytes(b"new")
    assert gate.main(_args(root, baseline, commit, tmp_path / "report.json")) == 1
