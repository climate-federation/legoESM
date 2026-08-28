"""Controls and invariants for the DINO full-step coverage registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
    "dino_1226/full_step_coverage.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("full_step_coverage_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registry_enumerates_current_oracle_without_structural_errors():
    module = _module()
    rows, errors = module.build_rows()
    assert errors == []
    assert len(rows) == 208  # 148 direct CALLs plus concrete/inactive dispatch rows.
    assert len({row.parent_call for row in rows}) == 148
    assert {row.disposition for row in rows} <= module.ALLOWED
    assert sum(row.disposition == "VERIFIED" for row in rows) == 6
    assert sum(row.disposition == "UNMEASURED" for row in rows) == 36


def test_gate_fails_open_campaign_and_prints_ranked_list(capsys):
    module = _module()
    assert module.main([]) == 1
    output = capsys.readouterr().out
    assert "FULL-STEP COVERAGE: 6/42 = 14.3% VERIFIED" in output
    assert "RANKED UNMEASURED (36" in output
    assert "dyn_spg_ts" in output


def test_planted_controls_reject_fake_receipt_and_unregistered_call(capsys):
    module = _module()
    assert module.self_test() == 0
    output = capsys.readouterr().out
    assert "planted fake receipt rejected (sha mismatch)" in output
    assert "planted unregistered CALL rejected" in output
