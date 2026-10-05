"""Controls for the round-62 self-describing dyn_zdf acquisition."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round62_dynzdf_gate as gate  # noqa: E402

SHAPE = (2, 2, 3)
ORIGIN = (3, 3)


def _parent_fields() -> dict[str, np.ndarray]:
    return {
        name: np.full((*SHAPE[:2], 1) if name.endswith("ssh") else SHAPE,
                      float(index + 1), dtype="=f8", order="F")
        for index, name in enumerate(gate.r50_gate.MOMENTUM_FIELDS[3])
    }


def _parent_record_bytes() -> bytes:
    fields = _parent_fields()
    out = bytearray("NEMO_L1_R50MOM1".ljust(16).encode("ascii"))
    out.extend(gate.r50_gate.HEADER.pack(
        1, 3, 3, 1, 2, 3, 3, *SHAPE, *ORIGIN, 64, len(fields),
    ))
    for name, values in fields.items():
        dims = values.shape
        out.extend(name.ljust(16).encode("ascii"))
        out.extend(gate.r50_gate.FIELD_HEADER.pack(
            2 if name.endswith("ssh") else 3, *dims
        ))
        out.extend(values.tobytes(order="F"))
    return bytes(out)


def _record_bytes() -> bytes:
    parent = _parent_fields()
    out = bytearray(gate.MAGIC.encode("ascii"))
    out.extend(gate.HEADER.pack(
        1, 3, 3, 1, 2, 3, 3, *SHAPE, *ORIGIN, 64, len(gate.FIELDS),
    ))
    for index, name in enumerate(gate.FIELDS):
        values = np.full(SHAPE, float(index + 1), dtype="=f8", order="F")
        if name == "implicit_solve_u":
            values = parent["raw_kaa_u"]
        elif name == "implicit_solve_v":
            values = parent["raw_kaa_v"]
        out.extend(name.ljust(16).encode("ascii"))
        out.extend(gate.FIELD_HEADER.pack(3, *SHAPE))
        out.extend(values.tobytes(order="F"))
    return bytes(out)


def test_reader_uses_header_schema_and_physical_eof(tmp_path):
    path = tmp_path / "record.bin"
    path.write_bytes(_record_bytes())
    result = gate.read_record(path)
    assert result["header"]["shape"] == list(SHAPE)
    assert tuple(result["fields"]) == gate.FIELDS


@pytest.mark.parametrize("mutation", ["magic", "slot", "field", "nan",
                                       "truncated", "trailing"])
def test_reader_refuses_corruption(tmp_path, mutation):
    raw = bytearray(_record_bytes())
    if mutation == "magic":
        raw[:16] = b"WRONG_MAGIC".ljust(16)
    elif mutation == "slot":
        struct.pack_into("=i", raw, 16 + 4 * 4, 3)
    elif mutation == "field":
        raw[16 + gate.HEADER.size:16 + gate.HEADER.size + 16] = (
            b"wrong_field".ljust(16)
        )
    elif mutation == "nan":
        first_payload = 16 + gate.HEADER.size + 16 + gate.FIELD_HEADER.size
        struct.pack_into("=d", raw, first_payload, np.nan)
    elif mutation == "truncated":
        raw = raw[:-1]
    else:
        raw.extend(b"x")
    path = tmp_path / f"{mutation}.bin"
    path.write_bytes(raw)
    with pytest.raises(gate.GateError):
        gate.read_record(path)


def test_admission_binds_parent_endpoint_and_plants(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.r50_gate, "EXPECTED_SHAPE", SHAPE)
    record = tmp_path / "oracle_r62_dynzdf_kt00000003_s3.bin"
    record.write_bytes(_record_bytes())
    parent = tmp_path / "oracle_r50_momentum_kt00000003_s3.bin"
    parent.write_bytes(_parent_record_bytes())
    commit = "a" * 40
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    Path(f"{record}.stamp").write_text(f"{digest} {commit} {record.name}\n")

    monkeypatch.setattr(gate, "preflight", lambda: {"status": "PREFLIGHT_PASS"})
    assert gate.admit(record, parent, commit, None)["status"] == "AT_BAR"
    with pytest.raises(gate.GateError, match="digest mismatch"):
        gate.admit(record, parent, commit, "payload")
    with pytest.raises(gate.GateError, match="producer commit"):
        gate.admit(record, parent, commit, "stamp")

    bad = bytearray(record.read_bytes())
    parsed = gate.read_record(record)
    offset = parsed["payload_offsets"]["implicit_solve_u"]
    struct.pack_into("=d", bad, offset, -999.0)
    record.write_bytes(bad)
    with pytest.raises(gate.GateError, match="implicit U endpoint"):
        gate.admit(record, parent, commit, None)


def test_preflight_proves_additions_only_and_inventory_gap():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["removed_source_lines"] == 0
    assert report["inventory"]["complete_carriers"] == []
    assert len(report["sentinels"]) == 7


def test_operator_script_is_new_target_fail_closed_and_committed_inputs():
    script = (gate.INSTRUMENT / "run.sh").read_text()
    assert "readonly TARGET_CFG=OVERFLOW_OMIP_L1_P3_R62ZDF" in script
    assert "round62/acquisition/oracle_overflow_dynzdf_internals" in script
    assert "readonly PATCH_REL=scripts/validate/" in script
    assert '"$PY" "$GATE" --preflight' in script
    assert script.index('mkdir -p "$(dirname "$TARGET_RUN")"') < script.index(
        'df -Pk "$mount"'
    )
    assert "/usr/bin/time" not in script
    assert '[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]]' in script
    assert "--plant-consumed" in script


def test_writer_refuses_partial_rows_and_overwrite():
    writer = (gate.INSTRUMENT / "l1_r62_dynzdf.F90").read_text()
    assert "ANY(r62_counts /= expected)" in writer
    assert "STATUS='NEW'" in writer
