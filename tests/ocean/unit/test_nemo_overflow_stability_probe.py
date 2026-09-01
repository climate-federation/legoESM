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
