from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l1_overflow_round50_pair_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round50_pair_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _record_bytes(kind: str, stage: int) -> bytes:
    magic, schemas = gate.KINDS[kind]
    nx, ny, nz = gate.EXPECTED_SHAPE
    fields = schemas[stage]
    kbb, kmm, krhs, kaa = gate.SLOTS[stage]
    out = bytearray(magic.ljust(16).encode("ascii"))
    out.extend(gate.HEADER.pack(
        1, 3, stage, kbb, kmm, krhs, kaa, nx, ny, nz,
        *gate.EXPECTED_ORIGIN, 64, len(fields),
    ))
    for index, name in enumerate(fields):
        is_ssh = name.endswith("ssh")
        dims = (nx, ny, 1) if is_ssh else (nx, ny, nz)
        out.extend(name.ljust(16).encode("ascii"))
        out.extend(gate.FIELD_HEADER.pack(2 if is_ssh else 3, *dims))
        values = np.full(dims, float(index + 1), dtype="=f8", order="F")
        out.extend(values.tobytes(order="F"))
    return bytes(out)


@pytest.fixture(autouse=True)
def _small_shape(monkeypatch):
    monkeypatch.setattr(gate, "EXPECTED_SHAPE", (2, 2, 2))


@pytest.mark.parametrize("kind", ("momentum", "tracer"))
@pytest.mark.parametrize("stage", (1, 2, 3))
def test_round50_reader_accepts_exact_schema(tmp_path, kind, stage):
    path = tmp_path / "record.bin"
    path.write_bytes(_record_bytes(kind, stage))
    result = gate.read_record(path, kind, stage)
    assert tuple(result["fields"]) == gate.KINDS[kind][1][stage]
    assert result["header"]["bits"] == 64


def test_round50_reader_refuses_every_schema_corruption(tmp_path):
    clean = bytearray(_record_bytes("momentum", 1))
    corruptions = {}

    bad = clean.copy()
    bad[:16] = b"WRONG_MAGIC".ljust(16)
    corruptions["magic"] = bad
    for name, index, value in (
        ("version", 0, 2), ("kt", 1, 4), ("stage", 2, 2),
        ("slot", 3, 9), ("dimension", 7, 9), ("origin", 10, 9),
        ("bits", 12, 32), ("count", 13, 99),
    ):
        bad = clean.copy()
        struct.pack_into("=i", bad, 16 + 4 * index, value)
        corruptions[name] = bad
    bad = clean.copy()
    bad[72:88] = b"wrong_field".ljust(16)
    corruptions["field-order"] = bad
    bad = clean.copy()
    struct.pack_into("=i", bad, 88, 2)
    corruptions["field-rank"] = bad
    bad = clean.copy()
    struct.pack_into("=d", bad, 104, np.nan)
    corruptions["nonfinite"] = bad
    corruptions["truncated"] = clean[:-1]
    corruptions["trailing"] = clean + b"x"

    for name, raw in corruptions.items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(raw)
        with pytest.raises(gate.GateError):
            gate.read_record(path, "momentum", 1)


def test_round50_admission_and_both_plants(tmp_path):
    commit = "a" * 40
    for kind in ("momentum", "tracer"):
        for stage in (1, 2, 3):
            path = tmp_path / f"oracle_r50_{kind}_kt00000003_s{stage}.bin"
            path.write_bytes(_record_bytes(kind, stage))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            Path(f"{path}.stamp").write_text(f"{digest} {commit} {path.name}\n")

    report = gate.admit(tmp_path, commit, None)
    assert report["status"] == "AT_BAR"
    with pytest.raises(gate.GateError, match="digest mismatch"):
        gate.admit(tmp_path, commit, "payload")
    with pytest.raises(gate.GateError, match="producer commit"):
        gate.admit(tmp_path, commit, "stamp")


def test_round50_preflight_is_additions_only():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["removed_source_lines"] == 0
    assert "worktree" in report
