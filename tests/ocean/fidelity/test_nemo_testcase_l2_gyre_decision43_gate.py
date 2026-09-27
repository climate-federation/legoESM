from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


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


def _year(value: float, commit: str):
    return {
        "format": "gyre-year-owners-day-gap-v1",
        "seed": 0,
        "plant": None,
        "rows": [
            {"day": day, "rms_T": value}
            for day in (30, 60, 90, 120, 180, 240, 300, 360)
        ],
        "member_admission": {
            "record": {"worktree": {"clean": True, "commit": commit}},
        },
        "worktree": {"clean": True, "commit": "c" * 40},
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


def _registry(comparison=None):
    comparison = _comparison() if comparison is None else comparison
    return tuple(
        row["row"] for row in comparison["field_moves"]
        if row["max_previous_legoesm_field_move"] != 0)


def test_decision43_passes_only_for_a_month_improvement(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    report = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=_registry())
    assert report["status"] == "PASS"
    assert report["moved_row_count"] == 1
    failed = module.evaluate(
        _comparison(), _day(1.0), _day(1.0),
        _year(1.0, "b" * 40), _year(1.0, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=_registry())
    assert failed["status"] == "FAIL"


def test_shared_dino_statement_requires_a_separate_measured_gate(monkeypatch):
    module = _module()
    cards = _cards()
    cards["DINO:nemo_dino_kamm"]["executes_route"] = True
    monkeypatch.setattr(module, "_card_execution", lambda route: cards)
    report = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=_registry())
    assert report["status"] == "FAIL"
    assert report["criteria"]["dino_measurement_required"] is True
    assert report["criteria"]["dino_statement_not_executed"] is False


def test_all_admission_plants_fail_the_gate(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    for plant in (
            "day30-no-improvement", "earlier-first-over-bar",
            "kt1-at-bar-loss", "year-day240-worse"):
        report = module.evaluate(
            _comparison(), _day(1.0), _day(0.1),
            _year(1.0, "b" * 40), _year(0.1, "c" * 40),
            expected_candidate_commit="c" * 40,
            expected_before_year_commit="b" * 40,
            registered_rows=_registry(), plant=plant)
        assert report["status"] == "FAIL", plant
        if plant == "year-day240-worse":
            assert report["criteria"]["year_day240_T_rms_not_worse"] is False


def test_moved_row_registry_is_exact_and_missing_entry_fails(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    comparison = _comparison()
    comparison["field_moves"].append({
        "row": "GYRE-zco.kt3.before.S",
        "max_previous_legoesm_field_move": 2.0,
        "n_improved_cells": 1,
        "n_worsened_cells": 0,
        "n_cells_worse_than_bar": 0,
    })
    complete = _registry(comparison)
    passed = module.evaluate(
        comparison, _day(1.0), _day(0.1),
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=complete)
    assert passed["criteria"]["all_moved_rows_registered"] is True
    missing = module.evaluate(
        comparison, _day(1.0), _day(0.1),
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=complete[:-1])
    assert missing["status"] == "FAIL"
    assert missing["missing_registered_rows"] == [complete[-1]]


def test_real_cards_resolve_the_source_condition():
    module = _module()
    cards = module._card_execution("ldf_stage3")
    assert cards["GYRE-zco"]["executes_route"] is True
    assert cards["NEMO-GYRE-recipe"]["executes_route"] is True
    assert cards["LOCK_EXCHANGE-zco"]["executes_route"] is False
    assert cards["OVERFLOW-zps"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm"]["tracer_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm_mlf"]["tracer_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm_mlf"]["executes_route"] is False


def test_stage_momentum_census_builds_real_orca2_card():
    module = _module()
    cards = module._card_execution("stage_momentum_wzv")
    orca2 = cards["ORCA2-zps"]
    assert orca2["recipe_source"] == "build_orca2_zps_card"
    assert orca2["executes_route"] is True
    assert orca2["executes_at_this_tip"] is True
    assert orca2["unmeasured_features"]


def test_stage1_r3t_ratio_execution_is_recipe_derived():
    module = _module()
    cards = module._card_execution("stage1_r3t_ratio")
    executing = {
        name for name, row in cards.items() if row["executes_route"]}

    assert executing == {
        "GYRE-zco", "LOCK_EXCHANGE-zco", "ORCA2-zps", "OVERFLOW-zps"}
    assert cards["NEMO-GYRE-recipe"]["linear_free_surface"] is True
    assert cards["NEMO-GYRE-recipe"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm"]["tracer_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm"]["executes_route"] is False


def test_zero_ladder_moves_are_vacuously_registered(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    comparison = _comparison()
    comparison["field_moves"] = []
    report = module.evaluate(
        comparison, _day(1.0), _day(0.1),
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=())
    assert report["status"] == "PASS"
    assert report["criteria"]["all_moved_rows_registered"] is True


def test_decision59_admits_only_strictly_sub_ten_floor_unit_year_moves(
        monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    before = _year(1.0, "b" * 40)
    within = _year(0.1, "c" * 40)
    for row in within["rows"]:
        if row["day"] in (240, 360):
            row["rms_T"] = 1.0 + 0.5 * module.DECISION59_MAX_ABS_K
    passed = module.evaluate(
        _comparison(), _day(1.0), _day(0.1), before, within,
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=_registry())
    assert passed["status"] == "PASS"
    assert passed["criteria"]["year_day240_T_rms_not_worse"] is False
    assert passed["criteria"]["year_day240_T_rms_admitted"] is True

    outside = _year(0.1, "c" * 40)
    for row in outside["rows"]:
        if row["day"] in (240, 360):
            row["rms_T"] = 1.0 + 2.0 * module.DECISION59_MAX_ABS_K
    failed = module.evaluate(
        _comparison(), _day(1.0), _day(0.1), before, outside,
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        registered_rows=_registry())
    assert failed["status"] == "FAIL"
    assert failed["criteria"]["year_day240_T_rms_admitted"] is False


def test_tke_shear_step_entry_eta_execution_is_recipe_derived():
    module = _module()
    cards = module._card_execution("tke_shear_step_entry_eta")
    executing = {
        name for name, row in cards.items() if row["executes_route"]}

    assert executing == {"GYRE-zco", "ORCA2-zps"}
    assert cards["GYRE-zco"]["tke_shear_production"] == (
        "nemo_face_native_now2")
    assert cards["ORCA2-zps"]["tke_shear_production"] == (
        "nemo_face_native_nbb2")
    assert not cards["NEMO-GYRE-recipe"]["executes_route"]
    assert not cards["DINO:nemo_dino_kamm"]["executes_route"]
    assert not cards["DINO:nemo_dino_kamm_mlf"]["executes_route"]


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
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        route="fct_metric_upstream", registered_rows=_registry())
    assert report["status"] == "FAIL"
    assert report["unmeasured_executing_cards"] == [
        "NEMO-GYRE-recipe", "ORCA2-zps"]
    measured = module.evaluate(
        _comparison(), _day(1.0), _day(0.1),
        _year(1.0, "b" * 40), _year(0.1, "c" * 40),
        expected_candidate_commit="c" * 40,
        expected_before_year_commit="b" * 40,
        route="fct_metric_upstream",
        measured_cards=("NEMO-GYRE-recipe", "ORCA2-zps"),
        registered_rows=_registry())
    assert measured["status"] == "PASS"


def test_wind_qco_route_is_derived_from_every_recipe():
    module = _module()
    cards = module._card_execution("wind_qco")
    assert cards["GYRE-zco"]["executes_route"] is True
    assert cards["NEMO-GYRE-recipe"]["executes_route"] is True
    assert cards["LOCK_EXCHANGE-zco"]["surface_stress_implicit"] is False
    assert cards["LOCK_EXCHANGE-zco"]["executes_route"] is False
    assert cards["OVERFLOW-zps"]["surface_stress_implicit"] is False
    assert cards["OVERFLOW-zps"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm"]["surface_stress_implicit"] is False
    assert cards["DINO:nemo_dino_kamm"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm_mlf"]["surface_stress_implicit"] is False
    assert cards["DINO:nemo_dino_kamm_mlf"]["executes_route"] is False


def test_momentum_ldf_live_geometry_route_is_derived_from_every_recipe():
    module = _module()
    cards = module._card_execution("momentum_ldf_live_geometry")
    assert cards["GYRE-zco"]["executes_route"] is True
    assert cards["NEMO-GYRE-recipe"]["lateral_viscosity_operator"] == (
        "vector_laplacian")
    assert cards["NEMO-GYRE-recipe"]["executes_route"] is False
    assert cards["LOCK_EXCHANGE-zco"]["lateral_viscosity_operator"] == (
        "vector_laplacian")
    assert cards["LOCK_EXCHANGE-zco"]["executes_route"] is False
    assert cards["OVERFLOW-zps"]["lateral_viscosity_operator"] == (
        "vector_laplacian")
    assert cards["OVERFLOW-zps"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm"]["momentum_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm"]["executes_route"] is False
    assert cards["DINO:nemo_dino_kamm_mlf"]["momentum_time_integrator"] == "euler"
    assert cards["DINO:nemo_dino_kamm_mlf"]["executes_route"] is False


def test_generic_comparison_registers_an_unchanged_card(tmp_path):
    module = _module()
    snapshot = tmp_path / "same.npz"
    np.savez(snapshot, step1_T=np.ones((2,), dtype=np.float64))
    report = {
        "format": "nemo-gyre-generic-card-three-step-v1",
        "status": "PASS",
        "worktree": {"commit": "a" * 40},
        "certifications": {"finite": True},
    }
    compared = module.compare_generic_nemo_gyre(
        report, snapshot, report, snapshot)
    assert compared["status"] == "PASS"
    assert compared["moved_row_count"] == 0
    assert compared["rows"][0]["cells_unequal"] == 0


def test_year_member_admission_requires_the_registered_harness_and_fp64(
        tmp_path):
    module = _module()
    root = tmp_path / "year"
    member = root / "lego_seed0_year"
    member.mkdir(parents=True)
    commit = "d" * 40
    manifest = {
        "format": "nemo-testcase-l2-gyre-year-fromrest-member-v1",
        "case": "GYRE-zco",
        "seed": 0,
        "tag": "year",
        "days": 360,
        "steps": 2160,
        "dt_s": 14400.0,
        "snapshot_step_interval": 6,
        "snapshot_days": list(range(1, 361)),
        "worktree": {"clean": True, "commit": commit},
    }
    (member / "manifest.json").write_text(json.dumps(manifest))
    fields = {
        name: np.ones((1,), dtype=np.float64)
        for name in ("T", "S", "u", "v", "ssh")
    }
    for day in module.YEAR_DAYS:
        np.savez(member / f"day{day:03d}.npz", **fields)
    admitted = module._admit_year_member(
        root, expected_commit=commit, label="test")
    assert set(admitted["snapshot_sha256"]) == {
        str(day) for day in module.YEAR_DAYS}

    fields["T"] = np.ones((1,), dtype=np.float32)
    np.savez(member / "day240.npz", **fields)
    with pytest.raises(module.GateError, match="expected float64"):
        module._admit_year_member(root, expected_commit=commit, label="test")


def test_year_day240_cli_plant_prints_and_returns_nonzero(
        monkeypatch, tmp_path, capsys):
    module = _module()
    candidate = "c" * 40
    before = "b" * 40
    comparison_path = tmp_path / "comparison.json"
    before_day_path = tmp_path / "before_day.json"
    after_day_path = tmp_path / "after_day.json"
    registry_path = tmp_path / "moved.tsv"
    output_path = tmp_path / "plant.json"
    values = {
        comparison_path: _comparison(candidate),
        before_day_path: _day(1.0, before),
        after_day_path: _day(0.1, candidate),
    }
    year_reports = iter((
        _year(1.0, before),
        _year(0.1, candidate),
    ))
    monkeypatch.setattr(module, "_read", lambda path: values[path])
    monkeypatch.setattr(
        module, "score_year_root", lambda *args, **kwargs: next(year_reports))
    monkeypatch.setattr(module, "_read_moved_row_registry",
                        lambda path: _registry())
    monkeypatch.setattr(module, "_card_execution", lambda route: _cards())
    monkeypatch.setattr(
        module, "worktree_stamp",
        lambda: {"clean": True, "commit": candidate})

    exit_code = module.main([
        "--comparison", str(comparison_path),
        "--before-day-gap", str(before_day_path),
        "--after-day-gap", str(after_day_path),
        "--expect-candidate-commit", candidate,
        "--before-year-root", str(tmp_path / "before_year"),
        "--after-year-root", str(tmp_path / "after_year"),
        "--year-nemo-root", str(tmp_path / "nemo_year"),
        "--expect-before-year-commit", before,
        "--moved-row-registry", str(registry_path),
        "--plant", "year-day240-worse",
        "--output", str(output_path),
    ])
    report = json.loads(output_path.read_text())
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["criteria"]["year_day240_T_rms_not_worse"] is False
    assert "STATUS PLANT-FIRED: year-day240-worse" in capsys.readouterr().out
