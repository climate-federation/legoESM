from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_round45_ablation_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round45_ablation_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_kt2_uv_reads_registered_velocity_rows():
    report = {
        "steps": [
            {"rows": []},
            {"rows": [
                {"name": "GYRE-zco.kt2.before.u", "absolute_max": 1.0},
                {"name": "GYRE-zco.kt2.before.v", "absolute_max": 2.0},
            ]},
        ]
    }
    assert gate.kt2_uv(report) == (1.0, 2.0)


def test_missing_artifacts_fail_closed(tmp_path, capsys):
    assert gate.main(["--root", str(tmp_path), "--round44", str(tmp_path)]) == 2
    assert "missing" in capsys.readouterr().err
