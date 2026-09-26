"""Non-vacuity and dispatch tests for the OVERFLOW stability instrument."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "scripts/validate/ocean_fidelity/testcases/nemo_overflow_stability_probe.py"
)
SPEC = importlib.util.spec_from_file_location("overflow_stability_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_argmax_row_reports_location_and_envelope_excursion():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card

    card = build_overflow_zps_card()
    oracle = np.full((3, 202, 100), 20.0, dtype=np.float64)
    candidate = oracle.copy()
    mask = np.asarray(card.recipe.z_coord.is_active) & (
        np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    )[..., None]
    location = tuple(np.argwhere(mask)[0])
    candidate[location] = 50.0
    row = PROBE._argmax_row("T", oracle, candidate, mask, card)
    assert row["argmax_index"] == list(location)
    assert row["gross_relative_excursion"]
    assert row["outside_roundoff_padded_oracle_range"]


def test_argmax_row_rejects_nonfinite_candidate():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card

    card = build_overflow_zps_card()
    oracle = np.zeros((3, 202), dtype=np.float64)
    candidate = oracle.copy()
    mask = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    candidate[tuple(np.argwhere(mask)[0])] = np.nan
    with pytest.raises(PROBE.ProbeError, match="candidate nonfinite"):
        PROBE._argmax_row("ssh", oracle, candidate, mask, card)


def test_private_vertical_transport_arm_changes_only_private_hook():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    baseline = _NEMOWSRK3TestHooks()
    arm = _NEMOWSRK3TestHooks(disable_tracer_vertical_transport=True)
    changed = [
        name for name, before, after in zip(baseline._fields, baseline, arm, strict=True)
        if before != after
    ]
    assert changed == ["disable_tracer_vertical_transport"]


def test_private_primary_transport_arm_changes_only_private_hook():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    baseline = _NEMOWSRK3TestHooks()
    arm = _NEMOWSRK3TestHooks(primary_transport_average=False)
    changed = [
        name for name, before, after in zip(baseline._fields, baseline, arm, strict=True)
        if before != after
    ]
    assert changed == ["primary_transport_average"]


def test_private_adaptive_momentum_arm_changes_only_private_hook():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    baseline = _NEMOWSRK3TestHooks()
    arm = _NEMOWSRK3TestHooks(disable_adaptive_implicit_momentum=True)
    changed = [
        name for name, before, after in zip(baseline._fields, baseline, arm, strict=True)
        if before != after
    ]
    assert changed == ["disable_adaptive_implicit_momentum"]


def test_private_bbl_arm_changes_only_private_hook():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    baseline = _NEMOWSRK3TestHooks()
    arm = _NEMOWSRK3TestHooks(disable_bbl=True)
    changed = [
        name for name, before, after in zip(baseline._fields, baseline, arm, strict=True)
        if before != after
    ]
    assert changed == ["disable_bbl"]


def test_shifted_step_control_fails_time_alignment(monkeypatch, tmp_path):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card

    card = build_overflow_zps_card()
    fields = PROBE._state_arrays(card.recipe.initial_state)
    oracle = {**fields, "step": 1}
    monkeypatch.setattr(PROBE, "read_entry", lambda *_: oracle)
    monkeypatch.setattr(PROBE, "_load_candidate", lambda *_: fields)
    monkeypatch.setattr(
        PROBE,
        "_certified_kt1_control",
        lambda *_: {"status": "VERIFIED_TEST_DOUBLE"},
    )
    with pytest.raises(PROBE.ProbeError, match="planted step-number mismatch"):
        PROBE.score(tmp_path, tmp_path, start_completed=0, end_completed=0,
                    plant="shift_step")


def test_resolved_coverage_is_file_driven_and_certifies_aimp(tmp_path):
    resolved = PROBE.DEFAULT_ORACLE / "output.namelist.dyn"
    report = PROBE.overflow_resolved_coverage(
        resolved, tmp_path / "coverage.json")
    assert report["status"] == "VERIFIED"
    assert report["unmeasured"] == []
    aimp = next(row for row in report["rows"]
                if row["key"] == "namzdf.ln_zad_aimp")
    assert aimp["status"] == "VERIFIED"
    assert sum(report["counts"].values()) == len(PROBE.OVERFLOW_RESOLVED_KEYS)
    assert report["barotropic_composition"]["status"] == "VERIFIED"
    assert report["barotropic_composition"]["oracle"]["runtime_nn_e"] == 3
    assert report["barotropic_composition"]["oracle"][
        "raw_secondary_weights"] == [3.0, 3.0, 2.0, 1.0]


def test_resolved_coverage_planted_file_side_key_goes_red():
    resolved = PROBE.DEFAULT_ORACLE / "output.namelist.dyn"
    with pytest.raises(PROBE.ProbeError, match="unaccounted resolved NEMO keys"):
        PROBE.overflow_resolved_coverage(
            resolved, plant_unaccounted=True)


def test_resolved_coverage_rejects_card_momentum_form_mutation():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card

    card = build_overflow_zps_card()
    bad_cfg = card.recipe.model_config._replace(momentum_advection="vector_invariant")
    bad_recipe = card.recipe._replace(model_config=bad_cfg)
    bad_card = card._replace(recipe=bad_recipe)
    with pytest.raises(PROBE.ProbeError, match="card momentum_advection"):
        PROBE.overflow_resolved_coverage(
            PROBE.DEFAULT_ORACLE / "output.namelist.dyn", card=bad_card)


def test_frozen_adaptive_arm_labels_are_non_vacuous():
    assert PROBE._arm_d_label(2877, 6121, 2.0, 2.0) == "CONFIRMED"
    assert PROBE._arm_d_label(2877, 3928, 159.0, 3.0e6) == "PLAUSIBLE"
    assert PROBE._arm_d_label(2877, 2877, 1.01, 1.01) == "REFUTED_PRIMARY"
    assert PROBE._arm_d_label(2877, 3000, 1.5, 3.0) == "UNMEASURED"


def test_completed_arm_can_reach_confirmation_and_short_null_run_fails():
    failure, completed = PROBE._arm_failure_for_label({
        "first_nonfinite_completed_step": None,
        "requested_end_step": 6120,
    })
    assert (failure, completed) == (6121, True)
    assert PROBE._arm_d_label(2877, failure, 2.0, 2.0) == "CONFIRMED"
    with pytest.raises(PROBE.ProbeError, match="not evidence"):
        PROBE._arm_failure_for_label({
            "first_nonfinite_completed_step": None,
            "requested_end_step": 3200,
        })
