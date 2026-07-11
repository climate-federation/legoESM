"""Tests for the WB2 forecast-eval orchestrator (Stage 0, Task 9b).

The rollout is injected (identity) so the orchestration plumbing — loop over
cases x leads, diagnose+regrid, score, average — is tested without a GPU rollout.
Builds a T21 spectral state, so runs on a compute node.
"""
import numpy as np
import jax.numpy as jnp

from evaluations.wb_orchestrator import run_wb_forecast_eval, ForecastCase
from evaluations.wb_forecast import diagnose_and_regrid, HEADLINE_FIELD_KEYS


def _rest_state():
    from legoesm.atmosphere.dynamics.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(8)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)
    return state, grid, sigma


def test_orchestrator_identity_rollout_perfect_scores():
    state, grid, sigma = _rest_state()
    vf, vv, _, _ = diagnose_and_regrid(state, grid, sigma)          # verification == IC diagnosis
    clim = {k: np.zeros_like(vf[k]) for k in vf}                     # nonzero anomalies -> ACC defined
    verif_by_lead = {6: {"fields": vf, "valid": vv}, 24: {"fields": vf, "valid": vv}}
    case = ForecastCase(init_state=state, verif_by_lead=verif_by_lead)

    identity = lambda s, *a, **k: s                                  # no-op rollout
    scorecard = run_wb_forecast_eval(
        None, grid, sigma, None, 1800.0, [case], [6, 24], clim, rollout_fn=identity)

    expected_keys = {(k, lead) for k in HEADLINE_FIELD_KEYS for lead in (6, 24)}
    assert set(scorecard.keys()) == expected_keys
    for (key, lead), m in scorecard.items():
        assert m["rmse"] < 1e-6, (key, lead)                         # identity -> pred == verif
        assert abs(m["bias"]) < 1e-6, (key, lead)
        assert np.isfinite(m["acc"]), (key, lead)


def test_orchestrator_rejects_lead_not_multiple_of_dt():
    import pytest
    state, grid, sigma = _rest_state()
    vf, vv, _, _ = diagnose_and_regrid(state, grid, sigma)
    clim = {k: np.zeros_like(vf[k]) for k in vf}
    case = ForecastCase(state, {1: {"fields": vf, "valid": vv}})
    with pytest.raises(ValueError):
        # 1 h = 3600 s is not a multiple of dt = 1000 s -> would round -> wrong valid time
        run_wb_forecast_eval(None, grid, sigma, None, 1000.0, [case], [1], clim,
                             rollout_fn=lambda s, *a, **k: s)


def test_orchestrator_threads_forcing_to_rollout():
    """#919: a case carrying `forcing` passes it as `forcing_base=` to the
    rollout (the neural_gcm/sfno prescribed-SST pathway)."""
    state, grid, sigma = _rest_state()
    vf, vv, _, _ = diagnose_and_regrid(state, grid, sigma)
    clim = {k: np.zeros_like(vf[k]) for k in vf}
    forcing = {"T_sfc": jnp.zeros(4), "sic": jnp.zeros(4),
               "day_of_year": jnp.asarray(1.0), "seconds_of_day": jnp.asarray(0.0)}
    case = ForecastCase(state, {6: {"fields": vf, "valid": vv}}, forcing=forcing)

    recorded = {}

    def recording_rollout(s, *a, forcing_base="MISSING", **k):
        recorded["forcing_base"] = forcing_base
        return s

    run_wb_forecast_eval(None, grid, sigma, None, 1800.0, [case], [6], clim,
                         rollout_fn=recording_rollout)
    assert recorded["forcing_base"] is forcing        # threaded through by identity


def test_orchestrator_unforced_signature_when_no_forcing():
    """#919: with `forcing=None` (the default), the rollout is called WITHOUT a
    `forcing_base` kwarg — the classical-physics/default `spectral_rollout`
    signature is preserved (ABI-safe positional construction)."""
    state, grid, sigma = _rest_state()
    vf, vv, _, _ = diagnose_and_regrid(state, grid, sigma)
    clim = {k: np.zeros_like(vf[k]) for k in vf}
    case = ForecastCase(state, {6: {"fields": vf, "valid": vv}})   # forcing defaults None
    assert case.forcing is None

    seen = {}

    def rollout(s, *a, **k):
        seen["has_forcing_kw"] = "forcing_base" in k
        return s

    run_wb_forecast_eval(None, grid, sigma, None, 1800.0, [case], [6], clim,
                         rollout_fn=rollout)
    assert seen["has_forcing_kw"] is False


def test_orchestrator_averages_over_cases():
    # Two cases, identity rollout; a nonzero-error case + a perfect case -> mean RMSE > 0.
    state, grid, sigma = _rest_state()
    vf, vv, _, _ = diagnose_and_regrid(state, grid, sigma)
    clim = {k: np.zeros_like(vf[k]) for k in vf}
    perfect = ForecastCase(state, {6: {"fields": vf, "valid": vv}})
    # a case whose verification is offset so RMSE > 0
    vf_off = {k: vf[k] + (1.0 if k == "t850" else 0.0) for k in vf}
    offset = ForecastCase(state, {6: {"fields": vf_off, "valid": vv}})
    identity = lambda s, *a, **k: s
    sc = run_wb_forecast_eval(None, grid, sigma, None, 1800.0, [perfect, offset], [6], clim,
                              rollout_fn=identity)
    assert sc[("t850", 6)]["rmse"] > 0.0        # averaged: 0 (perfect) and ~1 (offset)
    assert sc[("z500", 6)]["rmse"] < 1e-6       # z500 identical in both cases
