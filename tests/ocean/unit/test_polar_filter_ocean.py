"""Mask-aware Fourier polar filter for the lat-lon C-grid ocean
(``config.use_polar_filter``).

A global lat-lon ocean has converging meridians (dx = R*dlon*cos(lat) -> 0 at the
poles), so explicit advection/metric terms violate CFL poleward and the cold-start
blows up regardless of integrator.  ``LatLonCGridOceanModel._apply_polar_filter``
truncates the zonal Fourier modes above the per-latitude CFL cap poleward of the
cutoff, reusing the shared ``grids.polar_filter`` (already used by the atmosphere
C-grid).  It is mask-aware (land filled with the ocean zonal mean before the FFT,
restored after) and exactly conserves the per-latitude ocean zonal mean.

Tests (direct on the leaf ``_apply_polar_filter`` + the ``step`` gate):
  * tracer (T,S) per-latitude ocean zonal mean is conserved to round-off;
  * land cells are bit-unchanged;
  * a high zonal-wavenumber mode poleward of the cutoff is damped;
  * a sub-cutoff (equatorward) latitude is left ~unchanged (full passband);
  * the periodic u-face wrap column is preserved (u[:, n_lon] == u[:, 0]);
  * all prognostic field shapes are preserved;
  * gating: default config has the filter OFF (bit-exact legacy); a poleward
    high-k mode survives a step with the filter off and is damped with it on.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _model(use_polar_filter=True, n_lat=24, n_lon=48, cutoff=60.0,
           land_lat_threshold=85.0, dt_for_mask=900.0, **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=land_lat_threshold)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
        enable_runtime_checks=False, use_polar_filter=use_polar_filter,
        polar_filter_cutoff_lat_deg=cutoff, **cfg_kw)
    return grid, state, LatLonCGridOceanModel(grid, z_coord, cfg)


def test_default_filter_off():
    """Off by default -> bit-exact for legacy/tripole configs."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    cfg = LatLonCGridOceanConfig.from_flat()
    assert cfg.use_polar_filter is False
    assert cfg.polar_filter_cutoff_lat_deg == pytest.approx(60.0)


def _set_T(state, arr):
    return state._replace(T=state.T.replace(data=jnp.asarray(arr)))


def test_tracer_zonal_mean_conserved():
    """Per-latitude, per-level OCEAN zonal mean of T,S preserved to round-off."""
    grid, state, model = _model()
    land = np.asarray(state.land_mask.data)  # (n_lat, n_lon)
    rng = np.random.default_rng(0)
    T = np.array(state.T.data)
    T = T + rng.standard_normal(T.shape) * 2.0  # rich zonal structure
    state = _set_T(state, T)

    out = model._apply_polar_filter(state, 900.0)
    To = np.asarray(out.T.data)

    w = land[:, :, None]
    cnt = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    mean_in = (T * w).sum(axis=1, keepdims=True) / cnt
    mean_out = (To * w).sum(axis=1, keepdims=True) / cnt
    assert np.allclose(mean_in, mean_out, atol=1e-9, rtol=0), \
        np.abs(mean_in - mean_out).max()


def test_land_cells_unchanged():
    """Land cells are restored bit-exactly to their input value."""
    grid, state, model = _model()
    land = np.asarray(state.land_mask.data)
    assert land.min() < 0.5, "test needs some land cells (raise land threshold?)"
    T = np.array(state.T.data)
    T[land < 0.5, :] = -999.0  # sentinel on land
    state = _set_T(state, T)
    out = np.asarray(model._apply_polar_filter(state, 900.0).T.data)
    assert np.all(out[land < 0.5, :] == -999.0)


def test_highk_poleward_damped():
    """A high zonal-wavenumber mode poleward of the cutoff is damped; a low-k
    mode at the same latitude is kept."""
    grid, state, model = _model(land_lat_threshold=89.0)  # full-ocean band at 71
    lat_deg = np.rad2deg(np.asarray(grid.lat))
    j = int(np.argmin(np.abs(lat_deg - 71.0)))  # poleward of 60 cutoff
    assert abs(lat_deg[j]) > 60.0
    n_lon = grid.n_lon
    lon = np.arange(n_lon)
    k_hi, k_lo = 20, 3  # k_max(71 deg, dt=900) ~ 12 -> hi cut, lo kept
    T = np.array(state.T.data)
    T[j, :, 0] = 10.0 + np.cos(2 * np.pi * k_hi * lon / n_lon) \
        + np.cos(2 * np.pi * k_lo * lon / n_lon)
    state = _set_T(state, T)
    out = np.asarray(model._apply_polar_filter(state, 900.0).T.data)

    amp_in = np.abs(np.fft.rfft(T[j, :, 0]))
    amp_out = np.abs(np.fft.rfft(out[j, :, 0]))
    assert amp_out[k_hi] < 0.1 * amp_in[k_hi]          # high-k strongly damped
    assert amp_out[k_lo] > 0.9 * amp_in[k_lo]          # low-k preserved
    # zonal mean (k=0) kept; tol is float32 (model compute precision) FFT round-trip
    assert amp_out[0] == pytest.approx(amp_in[0], rel=1e-4)


def test_equatorward_unchanged():
    """A latitude equatorward of the cutoff is left ~unchanged (full passband)."""
    grid, state, model = _model(land_lat_threshold=89.0)
    lat_deg = np.rad2deg(np.asarray(grid.lat))
    j = int(np.argmin(np.abs(lat_deg - 0.0)))  # equator, well within cutoff
    assert abs(lat_deg[j]) < 60.0
    n_lon = grid.n_lon
    lon = np.arange(n_lon)
    T = np.array(state.T.data)
    T[j, :, 0] = 20.0 + np.cos(2 * np.pi * 18 * lon / n_lon)  # high-k at equator
    state = _set_T(state, T)
    out = np.asarray(model._apply_polar_filter(state, 900.0).T.data)
    # full passband equatorward -> identity up to float32 FFT round-trip
    assert np.allclose(out[j, :, 0], T[j, :, 0], atol=1e-3)


def test_u_wrap_and_shapes_preserved():
    """u-face periodic wrap column preserved; all field shapes unchanged."""
    grid, state, model = _model()
    rng = np.random.default_rng(1)
    u = np.array(state.u.data) + rng.standard_normal(state.u.data.shape) * 0.1
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
    out = model._apply_polar_filter(state, 900.0)
    uo = np.asarray(out.u.data)
    n_lon = grid.n_lon
    assert uo.shape == u.shape
    assert np.allclose(uo[:, n_lon], uo[:, 0], atol=1e-12)  # wrap preserved
    for f_in, f_out in ((state.T, out.T), (state.S, out.S),
                        (state.v, out.v), (state.eta, out.eta)):
        assert f_out.data.shape == f_in.data.shape


def test_step_gating_off_vs_on():
    """A poleward high-k surface mode survives a step with the filter OFF and is
    damped with it ON."""
    n_lon = 48
    lon = np.arange(n_lon)
    k_hi = 20

    def _seed(grid, state):
        lat_deg = np.rad2deg(np.asarray(grid.lat))
        j = int(np.argmin(np.abs(lat_deg - 71.0)))
        T = np.array(state.T.data)
        T[j, :, 0] += np.cos(2 * np.pi * k_hi * lon / n_lon)
        return _set_T(state, T), j

    g0, s0, m0 = _model(use_polar_filter=False, land_lat_threshold=89.0,
                        n_lon=n_lon)
    s0, j = _seed(g0, s0)
    out0 = np.asarray(m0.step(s0, 900.0).T.data)
    amp_off = np.abs(np.fft.rfft(out0[j, :, 0]))[k_hi]

    g1, s1, m1 = _model(use_polar_filter=True, land_lat_threshold=89.0,
                        n_lon=n_lon)
    s1, j = _seed(g1, s1)
    out1 = np.asarray(m1.step(s1, 900.0).T.data)
    amp_on = np.abs(np.fft.rfft(out1[j, :, 0]))[k_hi]

    assert amp_on < 0.2 * amp_off  # filter ON strongly damps the poleward high-k
