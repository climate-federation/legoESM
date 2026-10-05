"""Controls for the round-53 OVERFLOW ENS operand acquisition gate."""

from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round53_ens_operands_gate as gate  # noqa: E402

SHAPE = (2, 2, 2)
ORIGIN = (3, 3)


def _ens_record_bytes() -> bytes:
    out = bytearray(gate.MAGIC.encode("ascii"))
    out.extend(gate.HEADER.pack(
        1, 3, 2, 3, gate._compiled_cme_value(), *SHAPE, *ORIGIN,
        64, len(gate.FIELDS),
    ))
    for index, name in enumerate(gate.FIELDS):
        out.extend(name.ljust(16).encode("ascii"))
        out.extend(gate.FIELD_HEADER.pack(3, *SHAPE))
        values = np.full(SHAPE, float(index + 1), dtype="=f8", order="F")
        out.extend(values.tobytes(order="F"))
    return bytes(out)


def _parent_record_bytes() -> bytes:
    fields = gate.r50_gate.MOMENTUM_FIELDS[2]
    kbb, kmm, krhs, kaa = gate.r50_gate.SLOTS[2]
    out = bytearray("NEMO_L1_R50MOM1".ljust(16).encode("ascii"))
    out.extend(gate.r50_gate.HEADER.pack(
        1, 3, 2, kbb, kmm, krhs, kaa, *SHAPE, *ORIGIN, 64, len(fields),
    ))
    for index, name in enumerate(fields):
        is_ssh = name.endswith("ssh")
        dims = (*SHAPE[:2], 1) if is_ssh else SHAPE
        out.extend(name.ljust(16).encode("ascii"))
        out.extend(gate.r50_gate.FIELD_HEADER.pack(2 if is_ssh else 3, *dims))
        values = np.full(dims, float(index + 1), dtype="=f8", order="F")
        out.extend(values.tobytes(order="F"))
    return bytes(out)


def test_reader_accepts_header_declared_schema(tmp_path):
    path = tmp_path / "record.bin"
    path.write_bytes(_ens_record_bytes())
    result = gate.read_record(path)
    assert result["header"]["shape"] == list(SHAPE)
    assert tuple(result["fields"]) == gate.FIELDS


def test_reader_refuses_schema_and_payload_corruption(tmp_path):
    clean = bytearray(_ens_record_bytes())
    corruptions = {}
    bad = clean.copy()
    bad[:16] = b"WRONG_MAGIC".ljust(16)
    corruptions["magic"] = bad
    for name, index, value in (
        ("version", 0, 2), ("kt", 1, 4), ("stage", 2, 3),
        ("kvor", 4, 4), ("shape", 5, 0), ("origin", 8, 0),
        ("bits", 10, 32), ("count", 11, 99),
    ):
        bad = clean.copy()
        struct.pack_into("=i", bad, 16 + 4 * index, value)
        corruptions[name] = bad
    bad = clean.copy()
    bad[64:80] = b"wrong_field".ljust(16)
    corruptions["name"] = bad
    bad = clean.copy()
    struct.pack_into("=i", bad, 80, 2)
    corruptions["rank"] = bad
    bad = clean.copy()
    struct.pack_into("=d", bad, 96, np.nan)
    corruptions["nonfinite"] = bad
    corruptions["truncated"] = clean[:-1]
    corruptions["trailing"] = clean + b"x"

    for name, raw in corruptions.items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(raw)
        with pytest.raises(gate.GateError):
            gate.read_record(path)


def test_admission_binds_parent_and_both_plants(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.r50_gate, "EXPECTED_SHAPE", SHAPE)
    record = tmp_path / "oracle_r53_ens_kt00000003_s2.bin"
    record.write_bytes(_ens_record_bytes())
    parent = tmp_path / "oracle_r50_momentum_kt00000003_s2.bin"
    parent.write_bytes(_parent_record_bytes())
    commit = "a" * 40
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    Path(f"{record}.stamp").write_text(f"{digest} {commit} {record.name}\n")

    report = gate.admit(record, parent, commit, None)
    assert report["status"] == "AT_BAR"
    with pytest.raises(gate.GateError, match="digest mismatch"):
        gate.admit(record, parent, commit, "payload")
    with pytest.raises(gate.GateError, match="producer commit"):
        gate.admit(record, parent, commit, "stamp")


def test_parent_shape_drift_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.r50_gate, "EXPECTED_SHAPE", SHAPE)
    record = tmp_path / "record.bin"
    record.write_bytes(_ens_record_bytes())
    parent = tmp_path / "parent.bin"
    raw = bytearray(_parent_record_bytes())
    struct.pack_into("=i", raw, 16 + 4 * 7, 3)
    parent.write_bytes(raw)
    with pytest.raises(gate.r50_gate.GateError):
        gate.admit(record, parent, "a" * 40, None)


def test_preflight_is_additions_only_and_compiled_branch_bound():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["removed_source_lines"] == 0
    assert report["compiled_kvor"] == 5


def test_launcher_creates_only_missing_target_parent_before_df(tmp_path):
    run = (gate.INSTRUMENT / "run.sh").read_text()
    target_guard = '[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]]'
    create_parent = 'mkdir -p "$(dirname "$TARGET_RUN")"'
    space_check = 'for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do'
    assert run.index(target_guard) < run.index(create_parent) < run.index(space_check)

    target = tmp_path / "absent" / "nested" / "run"
    result = subprocess.run(
        [
            "bash",
            "-ceu",
            '[[ ! -e "$1" ]]; mkdir -p "$(dirname "$1")"; df -Pk "$(dirname "$1")"',
            "round54-parent-check",
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert target.parent.is_dir()
    assert not target.exists()
