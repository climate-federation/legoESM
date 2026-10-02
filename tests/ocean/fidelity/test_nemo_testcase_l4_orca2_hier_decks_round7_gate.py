"""Controls for the hierarchy recorder ABSENT repair."""

from __future__ import annotations

import ast
import importlib.util
import struct
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round7_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round7_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round7_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _frame(path: Path) -> None:
    nx = ny = 2
    payload = bytearray(gate.MAGIC.encode().ljust(16, b" "))
    payload.extend(gate.HEADER.pack(2, 1, 1, 0, len(gate.FIELDS), nx, ny, 64))
    for name in gate.FIELDS:
        payload.extend(name.encode().ljust(16, b" "))
        if name in gate.RUNOFF_FIELDS:
            payload.extend(gate.FIELD_HEADER.pack(0, 0, 0, 0))
        else:
            payload.extend(gate.FIELD_HEADER.pack(2, nx, ny, 1))
            payload.extend(struct.pack("=4d", 1.0, 2.0, 3.0, 4.0))
    path.write_bytes(payload)


def test_real_preflight_is_additions_only_and_owner_complete():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG5_ABSENT_REPAIR"
    assert report["patch_additions_only"] is True
    assert report["owners"]["ln_rnf"] == ["rnf", "rnf_tsc"]
    assert report["owners"]["surface_core"] == [
        "emp",
        "fr_i",
        "qns",
        "qsr",
        "sfx",
        "taum",
        "utau",
        "vtau",
    ]


def test_self_describing_absent_frame_passes(tmp_path):
    path = tmp_path / "frame.bin"
    _frame(path)
    report = gate.read_surface(path, kt=1, rank=0, runoff_on=False)
    assert report["version"] == 2
    assert report["absent"] == ["rnf", "rnf_tsc"]
    assert report["fields"]["qsr"] == {"status": "PRESENT", "count": 4}


@pytest.mark.parametrize(
    "plant", ("field-name", "truncated", "frame-nonfinite", "absent-as-zero", "owner-on")
)
def test_surface_plants_fire(tmp_path, plant):
    path = tmp_path / "frame.bin"
    _frame(path)
    with pytest.raises(gate.GateError):
        gate.read_surface(path, kt=1, rank=0, runoff_on=False, plant=plant)


def test_active_owner_refuses_absent_payload(tmp_path):
    path = tmp_path / "frame.bin"
    _frame(path)
    with pytest.raises(gate.GateError, match="shape"):
        gate.read_surface(path, kt=1, rank=0, runoff_on=True)


def test_runner_calibrates_before_rung5_and_is_fail_closed():
    runner = RUNNER.read_text()
    assert "makenemo" in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert runner.index('run_one "$CALIBRATION"') < runner.index('run_one "$TARGET_RUN"')
    assert runner.index("check_calibration") < runner.index('stage_run "$TARGET_RUN"')
    assert "cmp -s" in runner
    assert "--fuzz=0" in runner


def test_calibration_uses_current_rung6_resolved_validator_interface():
    tree = ast.parse(SCRIPT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "validate_resolved"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "rung6"
    ]
    assert len(calls) == 1
    assert [keyword.arg for keyword in calls[0].keywords] == []
