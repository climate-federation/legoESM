from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
          / "testcases" / "nemo_testcase_l2_gyre_round85_bundle_gate.py")
SPEC = importlib.util.spec_from_file_location("round85_bundle_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(name: str, value: float, status: str) -> dict:
    return {
        "name": name,
        "normalized_max_abs": value,
        "absolute_max": value,
        "status": status,
        "n": 1,
        "bar": 1e-15,
        "oracle_dtype": "float64",
        "candidate_dtype": "float64",
    }


def _reports(target_after: float = 0.5):
    names = (*MODULE.TARGETS, "GYRE-zco.kt2.before.T", "GYRE-zco.kt3.before.T")
    before_values = (1.0, 1.0, 0.0, 0.2)
    after_values = (target_after, target_after, 0.0, 0.8)
    before = {
        "steps": [{"rows": [_row(n, v, "AT-BAR" if v == 0.0 else "DEBT")
                              for n, v in zip(names, before_values, strict=True)]}],
        "first_over_bar": {"kt": 2, "fields": ["u", "v"]},
    }
    after = {
        "steps": [{"rows": [_row(n, v, "AT-BAR" if v == 0.0 else "DEBT")
                              for n, v in zip(names, after_values, strict=True)]}],
        "first_over_bar": {"kt": 2, "fields": ["u", "v"]},
    }
    oracle = np.array([0.0])
    before_fields = {
        name: {"oracle": oracle, "candidate": np.array([value]),
               "residual": np.array([abs(value)])}
        for name, value in zip(names, before_values, strict=True)
    }
    after_fields = {
        name: {"oracle": oracle, "candidate": np.array([value]),
               "residual": np.array([abs(value)])}
        for name, value in zip(names, after_values, strict=True)
    }
    return before, after, before_fields, after_fields


def test_decision38_allows_registered_later_worsening_when_targets_improve():
    result = MODULE.evaluate(*_reports())
    assert result["status"] == "PASS"
    assert result["canonical_two_ulp_status"] == "FAIL"
    assert result["n_moved_rows"] == 3


def test_decision38_refuses_unchanged_or_worse_target():
    result = MODULE.evaluate(*_reports(target_after=1.0))
    assert result["status"] == "FAIL"
    assert any("must move at field level toward the bar" in item
               for item in result["violations"])


def test_decision38_refuses_at_bar_loss_and_earlier_boundary():
    before, after, before_fields, after_fields = _reports()
    after["steps"][0]["rows"][2]["status"] = "DEBT"
    after["first_over_bar"] = {"kt": 1, "fields": ["u"]}
    result = MODULE.evaluate(before, after, before_fields, after_fields)
    assert result["status"] == "FAIL"
    assert any("AT-BAR -> DEBT" in item for item in result["violations"])
    assert any("first_over_bar moved earlier" in item for item in result["violations"])
