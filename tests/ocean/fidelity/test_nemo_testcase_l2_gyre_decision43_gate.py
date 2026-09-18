from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_testcase_l2_gyre_decision43_gate.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("decision43_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _day(value: float, commit: str = "c" * 40):
    return {
        "format": "gyre-year-owners-day-gap-v1",
        "rows": [{"day": 30, "rms_T": value}],
        "worktree": {"clean": True, "commit": commit},
    }


def _comparison(commit: str = "c" * 40):
    return {
        "format": "legoesm-ocean-oracle-relative-move-gate-v3",
        "n_certified_rows_compared": 70,
        "row_filter_applied": False,
        "plant": None,
        "worktree": {"clean": True, "commit": commit},
        "first_over_bar_reference": {"kt": 2, "fields": ["u", "v"]},
        "first_over_bar_candidate": {"kt": 2, "fields": ["u", "v"]},
        "row_status_changes": [],
        "field_moves": [{
            "row": "GYRE-zco.kt3.before.T",
            "max_previous_legoesm_field_move": 1.0,
            "n_improved_cells": 1,
            "n_worsened_cells": 1,
            "n_cells_worse_than_bar": 1,
        }],
    }


def _cards():
    return {
        "GYRE-zco": {"executes_route": True},
        "NEMO-GYRE-recipe": {"executes_route": False},
        "LOCK_EXCHANGE-zco": {"executes_route": False},
        "OVERFLOW-zps": {"executes_route": False},
        "DINO:nemo_dino_kamm": {"executes_route": False},
        "DINO:nemo_dino_kamm_mlf": {"executes_route": False},
    }


def test_decision43_passes_only_for_a_month_improvement(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    report = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        expected_candidate_commit="c" * 40)
    assert report["status"] == "PASS"
    assert report["moved_row_count"] == 1
    failed = module.evaluate(
        _comparison(), _day(1.0), _day(1.0),
        expected_candidate_commit="c" * 40)
    assert failed["status"] == "FAIL"


def test_shared_dino_statement_requires_a_separate_measured_gate(monkeypatch):
    module = _module()
    cards = _cards()
    cards["DINO:nemo_dino_kamm"]["executes_route"] = True
    monkeypatch.setattr(module, "_card_execution", lambda route: cards)
    report = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        expected_candidate_commit="c" * 40)
    assert report["status"] == "FAIL"
    assert report["criteria"]["dino_measurement_required"] is True
    assert report["criteria"]["dino_statement_not_executed"] is False


def test_all_three_plants_fail_the_gate(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    for plant in (
            "day30-no-improvement", "earlier-first-over-bar",
            "kt1-at-bar-loss"):
        report = module.evaluate(
            _comparison(), _day(1.0), _day(0.1),
            expected_candidate_commit="c" * 40, plant=plant)
        assert report["status"] == "FAIL", plant


def test_real_cards_resolve_the_source_condition():
    module = _module()
    cards = module._card_execution("ldf_stage3")
    assert cards["GYRE-zco"]["executes_route"] is True
    assert cards["LOCK_EXCHANGE-zco"]["executes_route"] is False
    assert cards["OVERFLOW-zps"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm"]["tracer_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm_mlf"]["tracer_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm_mlf"]["executes_route"] is False


def test_fct_metric_route_is_derived_from_every_recipe_and_fails_unmeasured():
    module = _module()
    cards = module._card_execution("fct_metric_upstream")
    assert cards["GYRE-zco"]["executes_route"] is True
    assert cards["NEMO-GYRE-recipe"]["executes_route"] is True
    assert cards["LOCK_EXCHANGE-zco"]["adaptive_implicit_vertadv"] is True
    assert cards["LOCK_EXCHANGE-zco"]["executes_route"] is False
    assert cards["OVERFLOW-zps"]["adaptive_implicit_vertadv"] is True
    assert cards["OVERFLOW-zps"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm"]["tracer_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm"]["executes_route"] is False

    report = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        expected_candidate_commit="c" * 40,
        route="fct_metric_upstream")
    assert report["status"] == "FAIL"
    assert report["unmeasured_executing_cards"] == ["NEMO-GYRE-recipe"]
    measured = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        expected_candidate_commit="c" * 40,
        route="fct_metric_upstream",
        measured_cards=("NEMO-GYRE-recipe",))
    assert measured["status"] == "PASS"
