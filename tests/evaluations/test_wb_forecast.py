"""Tests for the WB2 headline-field assembly (Stage 0, Task 5).

Runs on a compute node (builds T21 SH transforms):
    srun --account=glab --time=0:15:00 env JAX_ENABLE_X64=1 \
        <conda>/python -m pytest tests/evaluations/test_wb_forecast.py -v
"""
import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from evaluations.wb_forecast import (
    diagnose_headline_fields,
    HEADLINE_FIELD_KEYS,
    _PLEVEL_TARGET_PA,
)


class _FakeField:
    """Minimal duck-type of a grid-space Field (only ``.data`` is read)."""

    def __init__(self, data):
        self.data = data


def _rest_state(T_init=300.0, p_s_init=constants.p_ref):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(21)          # T21 — small but valid
    sigma = create_sigma_coordinate(8)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=T_init, p_s_init=p_s_init)
    return state, grid, sigma


def test_headline_fields_keys_shapes_finite():
    state, grid, sigma = _rest_state()
    diag = diagnose_headline_fields(state, grid, sigma)
    assert set(diag.fields) == set(HEADLINE_FIELD_KEYS)
    assert set(diag.valid) == set(HEADLINE_FIELD_KEYS)
    for k, arr in diag.fields.items():
        assert arr.shape == (grid.n_lat, grid.n_lon), k
        assert bool(jnp.all(jnp.isfinite(arr))), k
        assert diag.valid[k].shape == (grid.n_lat, grid.n_lon), k
        assert diag.valid[k].dtype == jnp.bool_, k


def test_headline_fields_isothermal_rest_physical():
    # Isothermal T=300, p_s=p_ref, flat (phis=0), at rest -> known headline values.
    state, grid, sigma = _rest_state()
    diag = diagnose_headline_fields(state, grid, sigma)
    out = diag.fields
    z500_analytic = float(constants.R_d * 300.0 / constants.g * np.log(2.0))  # ~6086 m
    assert abs(float(jnp.mean(out["z500"])) - z500_analytic) < 100.0
    assert abs(float(jnp.mean(out["t850"])) - 300.0) < 4.0
    assert abs(float(jnp.mean(out["mslp"])) - float(constants.p_ref)) < 100.0  # phis=0 -> mslp=p_s
    assert float(jnp.max(jnp.abs(out["u500"]))) < 1e-2       # at rest
    assert float(jnp.max(jnp.abs(out["v500"]))) < 1e-2
    assert float(jnp.max(out["wind_speed_10m"])) < 1e-2
    assert abs(float(jnp.mean(out["t2m"])) - 300.0) < 4.0
    # flat surface at p_ref -> every headline level is above ground
    for k in HEADLINE_FIELD_KEYS:
        assert bool(jnp.all(diag.valid[k])), k


def test_pressure_level_mask_flags_below_surface():
    # Uniform surface at 800 hPa: 850 hPa is BELOW ground (masked), 700/500 above.
    state, grid, sigma = _rest_state(T_init=280.0, p_s_init=80000.0)
    diag = diagnose_headline_fields(state, grid, sigma)
    assert not bool(jnp.any(diag.valid["t850"]))   # 850 hPa below an 800 hPa surface
    assert not bool(jnp.any(diag.valid["u850"]))
    assert bool(jnp.all(diag.valid["q700"]))       # 700 hPa above surface
    assert bool(jnp.all(diag.valid["z500"]))       # 500 hPa above surface
    assert bool(jnp.all(diag.valid["mslp"]))       # surface field always valid
    assert bool(jnp.all(diag.valid["t2m"]))


def test_qv_tracer_field_or_raw_array_and_shape_check():
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    state, grid, sigma = _rest_state()
    T = spectral_pe_to_grid(state, grid, sigma)["T"]
    q_arr = jnp.full(T.shape, 1e-3)

    # Field-like tracer (.data) and raw-array tracer must both work + agree.
    diag_field = diagnose_headline_fields(
        state._replace(tracers={"q_v": _FakeField(q_arr)}), grid, sigma)
    diag_raw = diagnose_headline_fields(
        state._replace(tracers={"q_v": q_arr}), grid, sigma)
    assert float(jnp.mean(diag_field.fields["q700"])) > 0.0
    assert float(jnp.mean(diag_raw.fields["q700"])) > 0.0
    assert float(jnp.max(jnp.abs(diag_field.fields["q700"] - diag_raw.fields["q700"]))) < 1e-9

    # Wrong tracer layout must fail loudly, not silently mis-broadcast.
    bad = jnp.zeros(T.shape[:-1] + (T.shape[-1] + 1,))
    with pytest.raises(ValueError):
        diagnose_headline_fields(state._replace(tracers={"q_v": bad}), grid, sigma)


def test_all_masks_consistent_with_targets():
    # Uniform 800 hPa surface: every pressure-level mask must match (target_pa<=p_s),
    # and every surface field must be all-valid.
    state, grid, sigma = _rest_state(T_init=280.0, p_s_init=80000.0)
    diag = diagnose_headline_fields(state, grid, sigma)
    for key, pa in _PLEVEL_TARGET_PA.items():
        if pa <= 80000.0:
            assert bool(jnp.all(diag.valid[key])), f"{key} pa={pa} should be all-valid"
        else:
            assert not bool(jnp.any(diag.valid[key])), f"{key} pa={pa} should be all-invalid"
    for key in ("mslp", "t2m", "u10", "v10", "wind_speed_10m"):
        assert bool(jnp.all(diag.valid[key])), key


def test_diagnose_and_regrid_isothermal():
    from evaluations.wb_forecast import diagnose_and_regrid
    state, grid, sigma = _rest_state()
    fields_wb2, valid_wb2, lat, lon = diagnose_and_regrid(state, grid, sigma)
    assert set(fields_wb2) == set(HEADLINE_FIELD_KEYS)
    assert lat.shape == (121,) and lon.shape == (240,)
    for k, arr in fields_wb2.items():
        assert arr.shape == (121, 240), k
        assert bool(np.all(np.isfinite(arr))), k
        assert valid_wb2[k].shape == (121, 240)
        assert valid_wb2[k].dtype == np.bool_
    # isothermal T=300 -> z500 ~ 6086 m everywhere; flat p_ref -> all valid
    z500_analytic = float(constants.R_d * 300.0 / constants.g * np.log(2.0))
    assert abs(float(np.mean(fields_wb2["z500"])) - z500_analytic) < 150.0
    assert bool(np.all(valid_wb2["z500"]))


def test_score_forecast_perfect_and_climatology():
    from evaluations.wb_forecast import score_forecast
    lat = np.linspace(-90.0, 90.0, 24)
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.normal(5500.0, 100.0, (24, 48)))
    clim = {"z500": jnp.full((24, 48), 5500.0)}
    verif = {"z500": v}

    # perfect forecast -> RMSE 0, bias 0, ACC 1
    s = score_forecast({"z500": v}, verif, clim, lat)
    assert s["z500"]["rmse"] < 1e-6
    assert abs(s["z500"]["bias"]) < 1e-6
    assert s["z500"]["acc"] > 0.999

    # climatology forecast -> ACC ~ 0 (zero anomaly), RMSE > 0
    s2 = score_forecast(clim, verif, clim, lat)
    assert abs(s2["z500"]["acc"]) < 1e-6
    assert s2["z500"]["rmse"] > 0.0


def test_score_forecast_mask_changes_rmse():
    from evaluations.wb_forecast import score_forecast
    lat = np.array([-45.0, 45.0])
    pred = {"t850": jnp.array([[10.0, 10.0], [10.0, 10.0]])}
    verif = {"t850": jnp.array([[10.0, 10.0], [10.0, 1000.0]])}  # one bad cell
    clim = {"t850": jnp.zeros((2, 2))}
    valid = {"t850": jnp.array([[1.0, 1.0], [1.0, 0.0]])}        # mask the bad cell
    s_masked = score_forecast(pred, verif, clim, lat, valid=valid)
    s_unmasked = score_forecast(pred, verif, clim, lat)
    assert s_masked["t850"]["rmse"] < 1e-6                       # bad cell excluded
    assert s_unmasked["t850"]["rmse"] > 1.0                      # bad cell dominates


def test_score_forecast_all_masked_records_nan_and_warns():
    # An all-masked field (undefined score) must NOT crash the whole scorecard —
    # a degenerate forecast is a legitimate comparison input. It records NaN and
    # warns loudly, and the OTHER fields still score.
    import math
    import warnings

    from evaluations.wb_forecast import score_forecast
    lat = np.array([-45.0, 45.0])
    pred = {"t850": jnp.ones((2, 2)), "z500": jnp.ones((2, 2))}
    verif = {"t850": jnp.zeros((2, 2)), "z500": jnp.zeros((2, 2))}
    clim = {"t850": jnp.zeros((2, 2)), "z500": jnp.zeros((2, 2))}
    valid = {
        "t850": jnp.zeros((2, 2), dtype=bool),   # entirely below ground -> NaN
        "z500": jnp.ones((2, 2), dtype=bool),    # fully valid -> real score
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = score_forecast(pred, verif, clim, lat, valid=valid)
    assert math.isnan(s["t850"]["rmse"])         # degenerate field -> NaN
    assert math.isnan(s["t850"]["acc"])
    assert s["z500"]["rmse"] >= 0.0              # other field still scored
    assert any("no valid" in str(x.message) for x in w)   # warned loudly


def test_headline_10m_wind_not_stronger_than_lowest_level():
    # The neutral reduction must weaken (never amplify) the lowest-level wind.
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    state, grid, sigma = _rest_state()
    diag = diagnose_headline_fields(state, grid, sigma)
    g = spectral_pe_to_grid(state, grid, sigma)
    speed_low = jnp.sqrt(g["u"][..., -1] ** 2 + g["v"][..., -1] ** 2)
    assert float(jnp.max(diag.fields["wind_speed_10m"] - speed_low)) < 1e-6
