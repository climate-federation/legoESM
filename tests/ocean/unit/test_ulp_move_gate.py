"""The ulp bar that admits a re-association must be able to refuse one.

Every test here plants a specific violation and asserts the comparison goes
red, because a comparison that cannot fail is decoration.  The real committed
phase-3 reference JSONs are exercised too, so the checks are not only against a
hand-written schema that happens to match the code.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest
from legoesm.ocean.fidelity.ulp_move_gate import (
    MAX_ULP_MOVE,
    ULP,
    certified_rows,
    compare_gate_reports,
    plant_ulp_move,
    row_move_tolerance,
)

REFERENCE_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs" / "ocean" / "fidelity" / "testcases"
)
REFERENCE_JSONS = sorted(REFERENCE_DIR.glob("nemo_testcases_l1_phase3_ref_*.json"))


def _report(**overrides) -> dict:
    report = {
        "status": "DEBT",
        "first_over_bar": {"kt": 2, "fields": ["T", "u"]},
        "steps": [
            {
                "kt": 2,
                "rows": [
                    {"name": "CASE.kt2.before.T", "status": "DEBT",
                     "exact": False, "n": 17000,
                     "normalized_max_abs": 1.1191048088221578e-14},
                    {"name": "CASE.kt2.before.S", "status": "UNINFORMATIVE",
                     "exact": False, "n": 17000,
                     "normalized_max_abs": 2.0301221021717148e-16},
                    {"name": "CASE.kt2.before.u", "status": "DEBT",
                     "exact": False, "n": 16900,
                     "normalized_max_abs": 2.598797930308122e-07},
                    {"name": "CASE.kt2.before.ssh", "status": "DEBT",
                     "exact": False, "n": 200,
                     "normalized_max_abs": 1.049160758270773e-14},
                ],
            },
        ],
    }
    report.update(overrides)
    return report


def _moved(row_name: str, n_ulps: float) -> dict:
    candidate = _report()
    for row in candidate["steps"][0]["rows"]:
        if row["name"] == row_name:
            row["normalized_max_abs"] += n_ulps * ULP
            return candidate
    raise AssertionError(f"no such row: {row_name}")


def test_identical_reports_pass():
    result = compare_gate_reports(_report(), _report())
    assert result["status"] == "PASS", result["violations"]
    assert result["n_certified_rows_compared"] == 4
    assert result["moves"] == []


@pytest.mark.parametrize("n_ulps", [0.5, 1.0, float(MAX_ULP_MOVE)])
def test_move_within_the_bar_passes(n_ulps):
    result = compare_gate_reports(_report(), _moved("CASE.kt2.before.u", n_ulps))
    assert result["status"] == "PASS", result["violations"]
    assert result["largest_move_ulps"] == pytest.approx(n_ulps, rel=1e-9)


@pytest.mark.parametrize("n_ulps", [MAX_ULP_MOVE + 1, 10, 1000])
def test_move_over_the_bar_fails(n_ulps):
    """The planted control: 3 ulps is one more than the bar and must refuse."""
    result = compare_gate_reports(_report(), _moved("CASE.kt2.before.u", n_ulps))
    assert result["status"] == "FAIL"
    assert any("CASE.kt2.before.u" in v and "moved" in v
               for v in result["violations"]), result["violations"]


@pytest.mark.parametrize("field", ["T", "S"])
def test_tracer_rows_must_be_bit_identical(field):
    """Half an ulp is inside the bar for u and still refused for T and S."""
    name = f"CASE.kt2.before.{field}"
    result = compare_gate_reports(_report(), _moved(name, 0.5))
    assert result["status"] == "FAIL"
    assert any(name in v and "BIT-IDENTICAL" in v
               for v in result["violations"]), result["violations"]


def test_first_over_bar_change_fails():
    candidate = _report(first_over_bar={"kt": 3, "fields": ["T", "u"]})
    result = compare_gate_reports(_report(), candidate)
    assert result["status"] == "FAIL"
    assert any("first_over_bar" in v for v in result["violations"])


def test_first_over_bar_field_set_change_fails():
    candidate = _report(first_over_bar={"kt": 2, "fields": ["T", "u", "ssh"]})
    assert compare_gate_reports(_report(), candidate)["status"] == "FAIL"


def test_report_status_change_fails():
    assert compare_gate_reports(_report(), _report(status="AT-BAR"))["status"] == "FAIL"


def test_row_status_change_fails():
    candidate = _report()
    candidate["steps"][0]["rows"][2]["status"] = "AT-BAR"
    result = compare_gate_reports(_report(), candidate)
    assert result["status"] == "FAIL"
    assert any("'status' changed" in v for v in result["violations"])


def test_missing_and_extra_rows_fail():
    shorter = _report()
    shorter["steps"][0]["rows"].pop()
    assert compare_gate_reports(_report(), shorter)["status"] == "FAIL"
    assert compare_gate_reports(shorter, _report())["status"] == "FAIL"


def test_empty_reference_cannot_silently_pass():
    result = compare_gate_reports({"status": "DEBT"}, {"status": "DEBT"})
    assert result["status"] == "FAIL"
    assert any("no certified rows" in v for v in result["violations"])


def test_duplicate_row_name_is_refused():
    report = _report()
    report["steps"][0]["rows"].append(dict(report["steps"][0]["rows"][0]))
    with pytest.raises(ValueError, match="duplicate certified row name"):
        certified_rows(report)


def test_row_filter_selects_only_matching_rows():
    rows = certified_rows(_report(), row_filter=lambda name: name.endswith(".u"))
    assert list(rows) == ["CASE.kt2.before.u"]


def test_absolute_max_tolerance_follows_the_row_scale():
    """A row in physical units gets the same RELATIVE bar, not the same absolute one."""
    row = {"name": "x.u", "normalized_max_abs": 0.0,
           "absolute_max": 0.0, "reference_max_abs": 1024.0}
    assert row_move_tolerance(row, "normalized_max_abs") == MAX_ULP_MOVE * ULP
    assert row_move_tolerance(row, "absolute_max") == MAX_ULP_MOVE * ULP * 1024.0


def test_plant_helper_targets_a_non_tracer_row_and_is_caught():
    planted = plant_ulp_move(_report(), MAX_ULP_MOVE + 1)
    result = compare_gate_reports(_report(), planted)
    assert result["status"] == "FAIL"
    assert not any(".T" in v or ".S" in v for v in result["violations"]), (
        "the plant landed on a tracer row, so it did not probe the ulp bar")


@pytest.mark.skipif(not REFERENCE_JSONS, reason="no committed phase-3 reference JSON")
@pytest.mark.parametrize("path", REFERENCE_JSONS, ids=lambda p: p.stem)
def test_committed_reference_against_itself_passes_and_a_plant_fails(path):
    """Same checks, on the real gate schema rather than a hand-written one."""
    reference = json.loads(path.read_text())
    assert compare_gate_reports(reference, copy.deepcopy(reference))["status"] == "PASS"
    planted = plant_ulp_move(copy.deepcopy(reference), MAX_ULP_MOVE + 1)
    result = compare_gate_reports(reference, planted)
    assert result["status"] == "FAIL", "a 3-ulp plant went unnoticed"
    assert result["n_certified_rows_compared"] > 0


def test_exact_flag_may_flip_when_the_residual_move_is_admitted():
    """LOCK's kt=2 SSH went 4.78e-28 -> 0.0: one admitted move, not two."""
    reference = _report()
    reference["steps"][0]["rows"][3]["normalized_max_abs"] = 4.782137916322191e-28
    candidate = copy.deepcopy(reference)
    candidate["steps"][0]["rows"][3]["normalized_max_abs"] = 0.0
    candidate["steps"][0]["rows"][3]["exact"] = True
    result = compare_gate_reports(reference, candidate)
    assert result["status"] == "PASS", result["violations"]
    assert result["derived_field_changes"] == [{
        "row": "CASE.kt2.before.ssh", "field": "exact",
        "reference": False, "candidate": True,
        "note": ("derived from the residual, which moved within "
                 "the bar; not counted a second time"),
    }]


def test_exact_flag_inconsistent_with_its_own_residual_fails():
    """If 'exact' ever stops meaning 'the residual is zero', fail closed."""
    candidate = _report()
    candidate["steps"][0]["rows"][2]["exact"] = True  # residual is 2.6e-07
    result = compare_gate_reports(_report(), candidate)
    assert result["status"] == "FAIL"
    assert any("does not agree with its own" in v for v in result["violations"])
