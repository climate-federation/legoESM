"""Controls for the round-56 self-describing UP3 acquisition schema."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round56_up3_operands_gate as gate  # noqa: E402


def _write(path: Path, *, truncate: bool = False) -> None:
    shape = (2, 3, 4)
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.encode("ascii"))
        handle.write(gate.HEADER.pack(1, 3, 2, 3, 3, 2, *shape, 3, 3, 64, len(gate.FIELDS)))
        for index, name in enumerate(gate.FIELDS):
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(gate.FIELD_HEADER.pack(3, *shape))
            values = np.full(shape, float(index), dtype="=f8", order="F")
            handle.write(values.tobytes(order="F"))
    if truncate:
        raw = path.read_bytes()
        path.write_bytes(raw[:-8])


def test_parser_derives_all_shapes_and_payloads_from_header(tmp_path):
    path = tmp_path / "record.bin"
    _write(path)
    report = gate.read_up3_record(path)
    assert report["header"]["shape"] == [2, 3, 4]
    assert tuple(report["fields"]) == gate.FIELDS
    assert report["fields"]["v_rhs_after_v"].shape == (2, 3, 4)


def test_parser_refuses_truncated_payload_from_physical_eof(tmp_path):
    path = tmp_path / "record.bin"
    _write(path, truncate=True)
    with pytest.raises(gate.GateError, match="truncated v_rhs_after_v payload"):
        gate.read_up3_record(path)


def test_one_ulp_digest_plant_moves_real_payload(tmp_path):
    path = tmp_path / "record.bin"
    _write(path)
    report = gate.read_up3_record(path)
    planted = gate._one_ulp_digest(path, report["payload_offsets"]["transport_u"])
    assert planted != report["sha256"]


def test_preflight_proves_patch_is_additions_only_and_complete():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["removed_source_lines"] == 0
    assert len(report["sentinels"]) == 11


def test_operator_script_binds_committed_inputs_and_creates_parent_before_df():
    script = (gate.INSTRUMENT / "run.sh").read_text()
    assert "readonly TARGET_CFG=OVERFLOW_OMIP_L1_P3_R56UP3" in script
    assert "readonly PATCH_REL=scripts/validate/" in script
    assert '"$PY" "$GATE" --preflight' in script
    assert script.index('mkdir -p "$(dirname "$TARGET_RUN")"') < script.index('df -Pk "$mount"')
    assert "/usr/bin/time" not in script
    assert '[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]]' in script
