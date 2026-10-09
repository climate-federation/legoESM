"""The oracle-relative field gate and each of its non-vacuity controls."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from legoesm.ocean.fidelity.ulp_move_gate import (
    MAX_ULP_MOVE,
    ResidualFieldRecorder,
    compare_gate_reports,
    comparison_exit_code,
    load_residual_artifact,
    plant_at_bar_to_debt,
    plant_cellwise_comparison,
    write_residual_artifact,
)


def _report(*, status="DEBT", first_kt=2, row_status="AT-BAR") -> dict:
    return {
        "status": status,
        "first_over_bar": None if first_kt is None else {"kt": first_kt, "fields": ["u"]},
        "precision_policy": "fp64",
        "steps": [{"rows": [{
            "name": "CASE.kt2.before.u",
            "status": row_status,
            "exact": False,
            "normalized_max_abs": 1.0e-16,
            "bar": 1.0e-15,
            "n": 3,
        }]}],
    }


def _payload(oracle=(1.0, 1.5, 2.0), candidate=None) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    if candidate is None:
        candidate = oracle.copy()
    candidate = np.asarray(candidate, dtype=np.float64)
    return {"CASE.kt2.before.u": {
        "oracle": oracle,
        "candidate": candidate,
        "residual": np.abs(candidate - oracle),
    }}


def _step(value: float, count: int, direction=np.inf) -> float:
    for _ in range(count):
        value = np.nextafter(value, direction)
    return value


def _compare(before_report, after_report, before_fields, after_fields):
    return compare_gate_reports(
        before_report, after_report,
        reference_fields=before_fields,
        candidate_fields=after_fields)


def test_identical_field_reports_pass():
    result = _compare(_report(), _report(), _payload(), _payload())
    assert result["status"] == "PASS", result["violations"]
    assert result["largest_oracle_residual_worsening_ulps"] == 0.0


def test_three_row_scale_oracle_ulp_cell_worsening_fails():
    """Required control: one cell worsened by 3 ulp exits red."""
    before = _payload()
    after = plant_cellwise_comparison(before, "worsen-3ulp")
    result = _compare(_report(), _report(), before, after)
    assert result["status"] == "FAIL"
    assert result["largest_oracle_residual_worsening_ulps"] == 3.0
    assert any("cell 0 worsened" in item for item in result["violations"])


def test_two_row_scale_oracle_ulp_cell_worsening_passes():
    oracle = np.array([1.0, 1.5, 2.0])
    candidate = oracle.copy()
    row_ulp = np.spacing(np.float64(2.0))
    candidate[0] += MAX_ULP_MOVE * row_ulp
    result = _compare(_report(), _report(), _payload(), _payload(candidate=candidate))
    assert result["status"] == "PASS", result["violations"]


def test_one_cell_improvement_passes():
    """Required control: movement toward NEMO is free."""
    oracle = np.array([1.0, 1.5, 2.0])
    candidate = oracle.copy()
    candidate[0] = _step(candidate[0], 20)
    before = _payload(candidate=candidate)
    after = plant_cellwise_comparison(before, "improve")
    result = _compare(_report(), _report(), before, after)
    assert result["status"] == "PASS", result["violations"]
    assert result["field_moves"][0]["n_improved_cells"] == 1


def test_at_bar_to_debt_flip_fails_without_field_movement():
    """Required control: a row threshold crossing is an independent prong."""
    candidate = plant_at_bar_to_debt(copy.deepcopy(_report()))
    result = _compare(_report(), candidate, _payload(), _payload())
    assert result["status"] == "FAIL"
    assert any("AT-BAR -> DEBT" in item for item in result["violations"])


def test_planted_control_exit_contracts_are_fail_closed():
    assert comparison_exit_code({"plant": "worsen-3ulp", "status": "FAIL"}) == 1
    assert comparison_exit_code({"plant": "at-bar-to-debt", "status": "FAIL"}) == 1
    assert comparison_exit_code({"plant": "improve", "status": "PASS"}) == 0
    assert comparison_exit_code({"plant": "worsen-3ulp", "status": "PASS"}) == 2
    assert comparison_exit_code({"plant": "improve", "status": "FAIL"}) == 2


def test_debt_to_at_bar_is_allowed():
    before = _report(row_status="DEBT")
    after = _report(row_status="AT-BAR")
    result = _compare(before, after, _payload(), _payload())
    assert result["status"] == "PASS", result["violations"]


def test_first_over_bar_may_move_later_but_not_earlier():
    fields = _payload()
    assert _compare(_report(first_kt=2), _report(first_kt=3), fields, fields)["status"] == "PASS"
    result = _compare(_report(first_kt=3), _report(first_kt=2), fields, fields)
    assert result["status"] == "FAIL"
    assert any("moved earlier" in item for item in result["violations"])


def test_first_debt_from_none_fails():
    fields = _payload()
    result = _compare(_report(first_kt=None), _report(first_kt=2), fields, fields)
    assert result["status"] == "FAIL"


def test_cellwise_gate_sees_compensation_hidden_by_equal_max_reduction():
    oracle = np.array([1.0, 1.0])
    before_candidate = np.array([_step(1.0, 3), 1.0])
    after_candidate = np.array([1.0, _step(1.0, 3)])
    before = _payload(oracle=oracle, candidate=before_candidate)
    after = _payload(oracle=oracle, candidate=after_candidate)
    result = _compare(_report(), _report(), before, after)
    assert (
        before["CASE.kt2.before.u"]["residual"].max()
        == after["CASE.kt2.before.u"]["residual"].max()
    )
    assert result["status"] == "FAIL"
    assert result["field_moves"][0]["n_worsened_cells"] == 1


def test_previous_legoesm_movement_is_disclosed_but_does_not_fail_improvement():
    oracle = np.array([1.0, 1.5, 2.0])
    old = oracle.copy()
    old[0] = _step(old[0], 8)
    result = _compare(
        _report(), _report(), _payload(candidate=old), _payload(candidate=oracle))
    assert result["status"] == "PASS", result["violations"]
    assert result["largest_previous_legoesm_field_move_in_row_scale_oracle_ulps"] == 4.0


def test_cell_fields_not_derived_reductions_decide_admission():
    before = _report()
    after = copy.deepcopy(before)
    before_row = before["steps"][0]["rows"][0]
    after_row = after["steps"][0]["rows"][0]
    before_row.update(n_unequal=3, relative_max_abs=2.0e-16)
    after_row.update(n_unequal=1, relative_max_abs=1.0e-16)
    result = _compare(before, after, _payload(), _payload())
    assert result["status"] == "PASS", result["violations"]


def test_denormal_cell_uses_floored_row_scale_not_local_spacing():
    oracle = np.array([0.0, np.nextafter(0.0, np.inf), 0.5])
    candidate = oracle.copy()
    candidate[0] = np.nextafter(0.0, np.inf)
    result = _compare(
        _report(), _report(), _payload(oracle=oracle),
        _payload(oracle=oracle, candidate=candidate),
    )
    assert result["status"] == "PASS", result["violations"]
    assert result["field_moves"][0]["row_scale_ulp"] == np.spacing(1.0)


def test_changed_nemo_oracle_fails_closed():
    result = _compare(_report(), _report(), _payload(), _payload(oracle=(1.0, 1.5, 3.0)))
    assert result["status"] == "FAIL"
    assert any("NEMO oracle field changed" in item for item in result["violations"])


def test_missing_or_extra_rows_fail_closed():
    extra = copy.deepcopy(_report())
    extra["steps"][0]["rows"].append({
        **extra["steps"][0]["rows"][0], "name": "CASE.kt2.before.v"})
    result = _compare(_report(), extra, _payload(), _payload())
    assert result["status"] == "FAIL"


def test_hashed_compressed_sidecar_round_trip_and_tamper(tmp_path):
    report = _report()
    recorder = ResidualFieldRecorder()
    oracle = np.array([1.0, 1.5, 2.0])
    candidate = np.array([1.0, _step(1.5, 1), 2.0])
    recorder.record("CASE.kt2.before.u", oracle, candidate, np.ones(3, dtype=bool))
    report_path = tmp_path / "gate.json"
    sidecar = write_residual_artifact(report, report_path, recorder)
    loaded = load_residual_artifact(report, report_path)
    assert np.array_equal(loaded["CASE.kt2.before.u"]["candidate"], candidate)
    with sidecar.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_residual_artifact(report, report_path)


def test_persisted_residual_must_equal_candidate_minus_oracle(tmp_path):
    report = _report()
    recorder = ResidualFieldRecorder()
    oracle = np.array([1.0, 1.5, 2.0])
    recorder.record("CASE.kt2.before.u", oracle, oracle, np.ones(3, dtype=bool))
    report_path = tmp_path / "gate.json"
    sidecar = write_residual_artifact(report, report_path, recorder)
    meta = report["oracle_relative_residual_artifact"]
    with np.load(sidecar) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    arrays[meta["rows"]["CASE.kt2.before.u"]["residual"]][0] = 1.0
    np.savez_compressed(sidecar, **arrays)
    # Update the hash so this reaches the semantic consistency check.
    import hashlib
    meta["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="inconsistent"):
        load_residual_artifact(report, report_path)
