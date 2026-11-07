"""Controls for the Round-130 held-patch year ranking gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
          / "ocean_fidelity" / "testcases"
          / "nemo_testcase_l2_gyre_round130_year_patch_gate.py")
SPEC = importlib.util.spec_from_file_location("round130_year_patch_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


COMMIT = "1" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_reports():
    year = {
        "format": gate.DAY_GAP_FORMAT,
        "rows": [
            {"day": day, "rms_T": value}
            for day, value in gate.BASELINE_YEAR_T.items()
        ],
        "member_admission": {
            "record": {"worktree": {"clean": True, "commit": COMMIT}}
        },
    }
    month = {
        "format": gate.DAY_GAP_FORMAT,
        "rows": [
            {"day": day,
             "rms_T": (gate.BASELINE_YEAR_T[30] if day == 30 else day * 1e-9)}
            for day in gate.MONTH_DAYS
        ],
        "worktree": {"clean": True, "commit": COMMIT},
    }
    return year, month


def _registry():
    return {
        "format": gate.REGISTRY_FORMAT,
        "baseline_commit": COMMIT,
        "candidates": [{"id": name} for name in gate.CANDIDATES],
    }


def test_json_pointer_and_all_field_local_proof(tmp_path):
    artifact = tmp_path / "proof.json"
    artifact.write_text(json.dumps({
        "worktree": {"clean": True, "commit": COMMIT},
        "exact": True,
        "rows": [{"n_unequal": 0}, {"n_unequal": 0}],
        "status": "PLANT-FIRED",
    }))
    entry = {
        "id": "r62_coeff",
        "local_proof": {
            "artifacts": {
                "proof": {"path": str(artifact), "sha256": _sha(artifact)},
            },
            "checks": [
                {"artifact": "proof", "pointer": "/exact", "equals": True},
                {"artifact": "proof", "pointer": "/rows",
                 "kind": "all-field", "field": "n_unequal", "equals": 0},
            ],
            "plant": {"artifact": "proof", "pointer": "/status",
                      "accepted": ["PLANT-FIRED"]},
        },
    }
    report = gate._local_proof(entry, tmp_path, COMMIT)
    assert report["status"] == "PASS"
    assert sum(row["assertions"] for row in report["checks"]) == 3


def test_local_proof_refuses_empty_assertion_registry(tmp_path):
    artifact = tmp_path / "proof.json"
    artifact.write_text(json.dumps({
        "worktree": {"clean": True, "commit": COMMIT},
        "status": "PLANT-FIRED",
    }))
    entry = {
        "id": "r62_coeff",
        "local_proof": {
            "artifacts": {
                "proof": {"path": str(artifact), "sha256": _sha(artifact)},
            },
            "checks": [],
            "plant": {"artifact": "proof", "pointer": "/status",
                      "accepted": ["PLANT-FIRED"]},
        },
    }
    with pytest.raises(gate.GateError, match="no local assertions"):
        gate._local_proof(entry, tmp_path, COMMIT)


def test_filtered_local_proof_selects_named_stage_rows(tmp_path):
    artifact = tmp_path / "proof.json"
    artifact.write_text(json.dumps({
        "worktree": {"clean": True, "commit": COMMIT},
        "rows": [
            {"kt": 1, "stage": 1, "n_unequal": 0},
            {"kt": 1, "stage": 2, "n_unequal": 9},
        ],
        "status": "PLANT-FIRED",
    }))
    entry = {
        "id": "r99_wclock",
        "local_proof": {
            "artifacts": {
                "proof": {"path": str(artifact), "sha256": _sha(artifact)},
            },
            "checks": [{
                "artifact": "proof", "pointer": "/rows",
                "kind": "filtered-all-field", "where": {"kt": 1, "stage": 1},
                "field": "n_unequal", "equals": 0,
            }],
            "plant": {"artifact": "proof", "pointer": "/status",
                      "accepted": ["PLANT-FIRED"]},
        },
    }
    report = gate._local_proof(entry, tmp_path, COMMIT)
    assert report["checks"][0]["assertions"] == 1


def test_measured_card_requires_hashed_passing_comparison(tmp_path):
    comparison = tmp_path / "generic.json"
    comparison.write_text(json.dumps({
        "format": gate.CARD_COMPARISON_FORMAT,
        "status": "PASS",
        "after_commit": COMMIT,
        "certifications_unchanged": True,
        "rows": [{"row": f"row-{index}"} for index in range(15)],
        "moved_rows": [{"row": "row-1"}],
        "moved_row_count": 1,
    }))
    entry = {
        "id": "r109_handoff",
        "card_measurements": {
            "NEMO-GYRE-recipe": {
                "path": str(comparison), "sha256": _sha(comparison),
            },
        },
    }
    rows = gate._card_measurements(
        entry, tmp_path, COMMIT, ["NEMO-GYRE-recipe"])
    assert rows == [{
        "card": "NEMO-GYRE-recipe",
        "path": str(comparison),
        "sha256": _sha(comparison),
        "moved_row_count": 1,
    }]


def test_measured_card_claim_without_artifact_is_refused(tmp_path):
    with pytest.raises(gate.GateError, match="lack exact artifacts"):
        gate._card_measurements(
            {"id": "r109_handoff"}, tmp_path, COMMIT,
            ["NEMO-GYRE-recipe"])


def test_missing_candidate_plant_fires_before_ranking(tmp_path):
    year, month = _baseline_reports()
    with pytest.raises(gate.GateError, match="candidate registry must be exactly"):
        gate.evaluate(_registry(), tmp_path / "registry.json", year, month,
                      "missing-candidate")


def test_baseline_day240_plant_fires(tmp_path):
    year, month = _baseline_reports()
    with pytest.raises(gate.GateError, match="baseline day 240 moved"):
        gate.evaluate(_registry(), tmp_path / "registry.json", year, month,
                      "baseline-day240")


def test_top_two_improvers_require_the_registered_pair(tmp_path, monkeypatch):
    year, month = _baseline_reports()
    improvements = {
        gate.CANDIDATES[0]: 2e-3,
        gate.CANDIDATES[1]: 1e-3,
    }

    def synthetic(entry, base, baseline_year, baseline_month):
        improvement = improvements.get(entry["id"], -1e-3)
        return {
            "id": entry["id"],
            "day240_improvement": improvement,
            "day360_T_rms": gate.BASELINE_YEAR_T[360] + 1e-4,
            "day30_T_rms": gate.BASELINE_YEAR_T[30] + 1e-4,
            "core_trajectory_and_year_pass": False,
            "landing_ready": False,
        }

    monkeypatch.setattr(gate, "_measurement", synthetic)
    monkeypatch.setattr(gate, "worktree_stamp",
                        lambda: {"clean": True, "commit": "2" * 40})
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry()))
    report = gate.evaluate(_registry(), registry_path, year, month, None)
    assert report["status"] == "PAIR-REQUIRED"
    assert report["strict_day240_improvers"] == list(gate.CANDIDATES[:2])
    assert report["pair_required"] is True


def test_dirty_ranking_tree_is_refused(tmp_path, monkeypatch):
    year, month = _baseline_reports()
    monkeypatch.setattr(gate, "worktree_stamp",
                        lambda: {"clean": False, "commit": "2" * 40})
    with pytest.raises(gate.GateError, match="worktree is dirty"):
        gate.evaluate(_registry(), tmp_path / "registry.json", year, month, None)


def test_vector_handoff_route_requires_both_resolved_selectors():
    exact = SimpleNamespace(
        tracer_time_integrator="rk3_ws",
        momentum_advection="vector_invariant",
    )
    flux = SimpleNamespace(
        tracer_time_integrator="rk3_ws",
        momentum_advection="flux_form",
    )
    leapfrog = SimpleNamespace(
        tracer_time_integrator="leapfrog",
        momentum_advection="vector_invariant",
    )
    route = "rk3_ws_vector_stage1_handoff"
    assert gate._route_executes(exact, route) is True
    assert gate._route_executes(flux, route) is False
    assert gate._route_executes(leapfrog, route) is False


def test_unknown_execution_route_is_refused():
    with pytest.raises(gate.GateError, match="unknown Round-130 execution route"):
        gate._route_executes(SimpleNamespace(), "not-a-route")
