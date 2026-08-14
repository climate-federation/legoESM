"""Direct tests for the sub-cloud SCM-RCE metrics and the objective selector.

The sub-cloud objective exists to make a layer visible that the full-column
score cannot see, so the decisive test here is not "the number is finite" but
the CONTRAST: perturb only the sub-cloud layer and show that the sub-cloud
score moves by orders of magnitude more than the column score does.  Without
that control, a new objective is just another number.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from legoesm.training.scm_rce_metrics import (
    DEFAULT_SUBCLOUD_TOP_M,
    score_profiles_precip_jax,
    score_subcloud_jax,
    subcloud_bulk_state,
    subcloud_mass_weights,
)

#: A 20-level idealized column, top -> surface like the CRM reference's grid.
_NLEV = 20


def _fake_reference():
    z = np.linspace(16000.0, 50.0, _NLEV)
    weights = np.full(_NLEV, 1.0 / _NLEV)
    T = 300.0 - 6.5e-3 * z
    qv = 0.018 * np.exp(-z / 3000.0)
    qcond = 1.0e-5 * np.exp(-((z - 9000.0) / 3000.0) ** 2)
    return SimpleNamespace(
        z_m=z, mass_weights=weights, T_ref=T, qv_ref=qv, qcond_ref=qcond,
        precip_ref_mm_day=2.4,
    )


def test_weights_select_only_the_subcloud_layer_and_renormalize():
    ref = _fake_reference()
    w = np.asarray(subcloud_mass_weights(ref.z_m, ref.mass_weights, top_m=1000.0))
    assert w.sum() == pytest.approx(1.0, rel=1e-12)
    selected = ref.z_m <= 1000.0
    assert selected.sum() >= 1
    assert np.all(w[~selected] == 0.0)
    assert np.all(w[selected] > 0.0)


def test_empty_selection_raises_rather_than_returning_zero_weights():
    """An all-zero weight vector would score every column identically — a
    plausible-looking number, which is worse than an exception."""
    ref = _fake_reference()
    with pytest.raises(ValueError, match="no level at or below"):
        subcloud_mass_weights(ref.z_m, ref.mass_weights, top_m=1.0)


def test_shape_mismatch_raises():
    ref = _fake_reference()
    with pytest.raises(ValueError, match="must share a shape"):
        subcloud_mass_weights(ref.z_m, ref.mass_weights[:-1], top_m=1000.0)


def test_perfect_profile_scores_zero():
    ref = _fake_reference()
    w = subcloud_mass_weights(ref.z_m, ref.mass_weights)
    T_sc, qv_sc, combined = score_subcloud_jax(
        ref, ref.T_ref, ref.qv_ref, subcloud_weights=w, profile_floor=1.0e-12)
    assert float(T_sc) == pytest.approx(0.0, abs=1e-12)
    assert float(qv_sc) == pytest.approx(0.0, abs=1e-12)
    assert float(combined) == pytest.approx(0.0, abs=1e-12)


def test_subcloud_objective_sees_what_the_column_score_cannot():
    """THE control that justifies the objective.

    Warm and moisten ONLY the levels below 1 km — the measured defect — and
    compare how far each score moves.  The column score is dominated by the
    condensate term over 20 levels; the sub-cloud score is the layer itself.
    """
    ref = _fake_reference()
    w = subcloud_mass_weights(ref.z_m, ref.mass_weights, top_m=DEFAULT_SUBCLOUD_TOP_M)
    below = np.asarray(ref.z_m) <= DEFAULT_SUBCLOUD_TOP_M
    T_bad = np.where(below, ref.T_ref + 2.5, ref.T_ref)
    qv_bad = np.where(below, ref.qv_ref * 1.20, ref.qv_ref)

    _t, _q, sc_good = score_subcloud_jax(
        ref, ref.T_ref, ref.qv_ref, subcloud_weights=w, profile_floor=1.0e-12)
    _t, _q, sc_bad = score_subcloud_jax(
        ref, T_bad, qv_bad, subcloud_weights=w, profile_floor=1.0e-12)

    col_good = float(score_profiles_precip_jax(
        ref, ref.T_ref, ref.qv_ref, ref.qcond_ref, 2.4,
        profile_floor=1.0e-12)[-1])
    col_bad = float(score_profiles_precip_jax(
        ref, T_bad, qv_bad, ref.qcond_ref, 2.4, profile_floor=1.0e-12)[-1])

    d_subcloud = float(sc_bad) - float(sc_good)
    d_column = col_bad - col_good
    assert d_subcloud > 0.0
    assert d_column >= 0.0
    # Not merely "larger": the whole point is that the column score barely
    # registers the same perturbation.
    assert d_subcloud > 5.0 * d_column


def test_bulk_state_matches_hand_values_and_is_subsaturated():
    rh, driver, dT = subcloud_bulk_state(
        T_air_K=296.94, r_air=0.014048, p_air_Pa=101056.0,
        sst_K=300.0, p_sfc_Pa=101480.0)
    # The CRM reference's own lowest level: sub-saturated, 3 K below the SST,
    # with a driver of roughly 8 g/kg.
    assert 0.70 < float(rh) < 0.80
    assert float(driver) == pytest.approx(0.0084, abs=5.0e-4)
    assert float(dT) == pytest.approx(3.06, abs=0.01)


def test_bulk_state_agrees_with_the_validate_probe():
    """One helper, two callers: the probe must not carry its own arithmetic."""
    from scripts.validate.check_scm_rce_surface_humidity import surface_humidity_row

    row = surface_humidity_row(
        "x", T_air_K=298.5, r_air=0.0190, p_air_Pa=101056.0,
        evap_mm_day=1.5, evap_source="test",
        sst_K=300.0, p_sfc_Pa=101480.0)
    rh, driver, dT = subcloud_bulk_state(
        T_air_K=298.5, r_air=0.0190, p_air_Pa=101056.0,
        sst_K=300.0, p_sfc_Pa=101480.0)
    assert row.relative_humidity == pytest.approx(float(rh), rel=1e-12)
    assert row.driver_kg_kg == pytest.approx(float(driver), rel=1e-12)
    assert row.delta_T_K == pytest.approx(float(dT), rel=1e-12)


def test_objective_value_dispatch_and_nan_mapping():
    from scripts.run.run_scm_rce_campaign import TUNE_OBJECTIVES, objective_value

    run = SimpleNamespace(score=1.25, subcloud_score=7.5)
    assert objective_value(run, "combined") == pytest.approx(1.25)
    assert objective_value(run, "subcloud") == pytest.approx(7.5)
    assert set(TUNE_OBJECTIVES) == {"combined", "subcloud"}
    with pytest.raises(ValueError, match="unknown objective"):
        objective_value(run, "condensate")
    # NaN must read as +inf: a NaN incumbent makes every ``trial < best``
    # comparison False and silently discards the whole search.
    nan_run = SimpleNamespace(score=float("nan"), subcloud_score=float("nan"))
    assert objective_value(nan_run, "combined") == float("inf")
    assert objective_value(nan_run, "subcloud") == float("inf")
