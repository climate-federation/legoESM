"""Mechanical validation of era5_to_latlon_carry on a synthetic ERA5 slice.

No real ERA5 data is required: a fabricated ERA5Slice on a small lat-lon
grid exercises the regrid + vertical-interp + specific-humidity->mixing-
ratio conversion and asserts shape, finiteness, and physical bounds on
the model lat-lon grid.  (Full physical validation against real ERA5 is
deferred to when a zarr path is provided.)
"""

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.factory import create_grid
from legoesm.training.era5_to_state import ERA5Slice, era5_to_latlon_carry


def _synthetic_era5(n_lat=37, n_lon=72, n_plev=8):
    lat = np.linspace(-np.pi / 2 * 0.98, np.pi / 2 * 0.98, n_lat)
    lon = np.linspace(0.0, 2 * np.pi * (1 - 1.0 / n_lon), n_lon)
    plev = np.array([1000, 2000, 5000, 10000, 25000, 50000, 85000, 100000.0])
    latg = lat[:, None, None]
    # Earth-like: warm tropics, cold poles + a pressure-consistent lapse
    # (warm at high pressure / surface, cold at low pressure / top) +
    # a zonal jet in u.  Altitude proxy z(plev) = (1 - plev/p_s)*12 km
    # so the lapse decreases T with HEIGHT, not with array index.
    z = (1.0 - plev / 1.0e5) * 12000.0          # (n_plev,) m, 0 at surface
    T_sfc = 300.0 - 40.0 * np.sin(latg) ** 2    # (n_lat,1,1)
    T = T_sfc - 6.5e-3 * z[None, None, :]
    T = np.broadcast_to(T, (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    u = (30.0 * np.cos(latg) * np.sin(2 * latg)).astype(np.float32)
    u = np.broadcast_to(u, (n_lat, n_lon, n_plev)).astype(np.float32).copy()
    v = np.zeros((n_lat, n_lon, n_plev), np.float32)
    q = np.broadcast_to(
        (0.018 * np.cos(latg) ** 2), (n_lat, n_lon, n_plev)
    ).astype(np.float32).copy()
    p_s = np.full((n_lat, n_lon), 1.0e5, np.float32)
    sst = (300.0 - 40.0 * np.sin(lat[:, None]) ** 2) * np.ones((n_lat, n_lon))
    phis = np.zeros((n_lat, n_lon), np.float32)
    return ERA5Slice(
        T=T, u=u, v=v, q=q, p_s=p_s, sst=sst.astype(np.float32),
        phis=phis, lat=lat, lon=lon, plev_Pa=plev,
    )


def test_latlon_carry_shapes_and_physical():
    grid = create_grid("latlon", 24)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(30)
    carry = era5_to_latlon_carry(_synthetic_era5(), grid, sigma)
    # SegmentCarry exposes T/u/v/p_s and q_v via the packed layout.
    T = np.asarray(carry.T.data if hasattr(carry.T, "data") else carry.T)
    u = np.asarray(carry.u.data if hasattr(carry.u, "data") else carry.u)
    assert T.shape == (grid.n_lat, grid.n_lon, 30)
    assert u.shape == (grid.n_lat, grid.n_lon, 30)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(u))
    # Physical bounds: T in a sane atmospheric range, jet present.
    assert 180.0 < T.min() and T.max() < 320.0
    assert np.abs(u).max() > 5.0, "zonal jet did not survive the regrid"
    qv = np.asarray(carry.q_v.data if hasattr(carry, "q_v") and hasattr(carry.q_v, "data")
                    else getattr(carry, "q_v"))
    assert np.all(qv >= 0.0) and qv.max() < 0.05, "mixing ratio unphysical"


def test_latlon_carry_specific_to_mixing_ratio():
    # A column with q_specific=0.02 must map to r = q/(1-q) ≈ 0.0204.
    # Build a fresh slice with the desired q (no in-place mutation).
    base = _synthetic_era5()
    era5 = base._replace(q=np.full_like(base.q, 0.02))
    grid = create_grid("latlon", 12)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(20)
    carry = era5_to_latlon_carry(era5, grid, sigma)
    qv = np.asarray(getattr(carry, "q_v").data
                    if hasattr(getattr(carry, "q_v"), "data") else getattr(carry, "q_v"))
    # Near the surface (sigma~1) where interpolation is well-posed.
    assert 0.0200 < float(qv[..., -1].mean()) < 0.0210


def _mountain_era5(n_lat=37, n_lon=72, n_plev=8, peak_phis=5.6e4):
    """Synthetic ERA5 slice with a single steep orographic peak in phis."""
    base = _synthetic_era5(n_lat=n_lat, n_lon=n_lon, n_plev=n_plev)
    phis = np.zeros((n_lat, n_lon), np.float32)
    phis[n_lat // 3, n_lon // 2] = peak_phis      # ~5600 m ERA5-like spike
    phis[n_lat // 3, n_lon // 2 + 1] = peak_phis * 0.6
    return base._replace(phis=phis)


def _max_abs_grad(arr):
    di = np.abs(np.diff(arr, axis=0)).max()
    dj = np.abs(arr - np.roll(arr, 1, axis=1)).max()
    return float(max(di, dj))


# --- T2.2: shared hydrostatic-adjustment helper (cube + lat-lon) -------------

def test_phis_adjust_identity_when_no_smoothing_nonhybrid():
    # phis_raw == phis_smooth and non-hybrid ⇒ p_s and phis unchanged.
    from legoesm.training.era5_to_state import _apply_phis_hydrostatic_adjustment
    phis = jnp.asarray(np.full((4, 6), 2.0e4))
    p_s = jnp.asarray(np.full((4, 6), 9.0e4))
    T_sfc = jnp.asarray(np.full((4, 6), 288.0))
    phis_adj, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis, phis, p_s, T_sfc, sigma=None, is_hybrid=False)
    np.testing.assert_allclose(np.asarray(phis_adj), np.asarray(phis), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(p_s_adj), np.asarray(p_s), rtol=1e-6)


def test_phis_adjust_lowering_terrain_raises_ps_sign():
    # Smoothing LOWERS a peak (phis_smooth < phis_raw) ⇒ p_s must INCREASE
    # (descend from higher raw surface to lower smoothed surface).
    from legoesm.training.era5_to_state import _apply_phis_hydrostatic_adjustment
    phis_raw = jnp.asarray(np.array([[5.0e4]]))
    phis_smooth = jnp.asarray(np.array([[3.0e4]]))   # terrain cut by smoothing
    p_s = jnp.asarray(np.array([[6.0e4]]))
    T_sfc = jnp.asarray(np.array([[270.0]]))
    _, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis_raw, phis_smooth, p_s, T_sfc, sigma=None, is_hybrid=False)
    assert float(p_s_adj[0, 0]) > float(p_s[0, 0]), "lowering terrain must raise p_s"
    # Magnitude matches the barometric formula exactly.
    from legoesm import constants
    expect = 6.0e4 * np.exp((5.0e4 - 3.0e4) / (constants.R_d * 270.0))
    np.testing.assert_allclose(float(p_s_adj[0, 0]), expect, rtol=1e-6)


def test_phis_adjust_hybrid_floor_raises_ps_and_lowers_phis():
    # A p_s well below the degenerate-layer floor must be raised to the floor,
    # and phis lowered by the barometric equivalent (split-PGF consistency).
    from legoesm.training.era5_to_state import (
        _apply_phis_hydrostatic_adjustment, _hybrid_p_s_floor)
    from legoesm.grids.vertical import make_hybrid_levels
    sigma = make_hybrid_levels(20)
    floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
    phis_s = jnp.asarray(np.full((3, 3), 4.0e4))
    p_s = jnp.asarray(np.full((3, 3), floor * 0.5))   # below the floor
    T_sfc = jnp.asarray(np.full((3, 3), 260.0))
    phis_adj, p_s_adj = _apply_phis_hydrostatic_adjustment(
        phis_s, phis_s, p_s, T_sfc, sigma=sigma, is_hybrid=True)
    assert np.all(np.asarray(p_s_adj) >= floor - 1.0), "p_s not raised to floor"
    assert np.all(np.asarray(phis_adj) < np.asarray(phis_s)), "phis not lowered to match"


# --- T3.2: lat-lon carry with real (non-flat) orography ----------------------

def test_latlon_carry_smooths_orography_and_stays_finite():
    # With a steep ERA5 peak, the carry's phis must be SMOOTHER (smaller max
    # gradient) than the raw regridded orography — the fix for the blow-up.
    from legoesm.training.era5_to_state import era5_to_latlon_carry, regrid_2d_to_gaussian
    from legoesm.grids.vertical import create_sigma_coordinate
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = create_sigma_coordinate(20)
    carry = era5_to_latlon_carry(era5, grid, sigma)
    raw = np.asarray(regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid))
    phis = np.asarray(carry.phis)
    assert np.all(np.isfinite(phis)) and np.all(np.isfinite(np.asarray(carry.p_s)))
    assert _max_abs_grad(phis) < _max_abs_grad(raw), "carry did not smooth phis"
    assert phis.max() <= raw.max() + 1e-6, "smoothing must not amplify the peak"
    T = np.asarray(carry.T)
    assert np.all(np.isfinite(T)) and 150.0 < T.min() and T.max() < 340.0


def test_latlon_carry_hybrid_orography_respects_ps_floor():
    from legoesm.training.era5_to_state import era5_to_latlon_carry, _hybrid_p_s_floor
    from legoesm.grids.vertical import make_hybrid_levels
    era5 = _mountain_era5()
    grid = create_grid("latlon", 24)
    sigma = make_hybrid_levels(20)
    carry = era5_to_latlon_carry(era5, grid, sigma)
    floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
    p_s = np.asarray(carry.p_s)
    assert np.all(np.isfinite(np.asarray(carry.T)))
    assert np.all(p_s >= floor - 1.0), "hybrid p_s floor not honored on lat-lon"


def test_latlon_carry_analytic_temperature_value():
    # Value-based check (not just bounds): a latitude-only ERA5 T field
    # must interpolate to the model grid with the right equator-pole
    # structure — equatorial column warmer than polar at the surface.
    era5 = _synthetic_era5(n_lat=73, n_lon=144)
    grid = create_grid("latlon", 24)
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(30)
    carry = era5_to_latlon_carry(era5, grid, sigma)
    T = np.asarray(carry.T.data if hasattr(carry.T, "data") else carry.T)
    lat = np.asarray(grid.lat)
    i_eq = int(np.argmin(np.abs(lat)))
    i_pole = int(np.argmax(np.abs(lat)))
    T_eq_sfc = float(T[i_eq, :, -1].mean())
    T_pole_sfc = float(T[i_pole, :, -1].mean())
    assert T_eq_sfc > T_pole_sfc + 15.0, (
        f"equator-pole gradient lost: eq={T_eq_sfc:.1f} pole={T_pole_sfc:.1f}")
    # Equatorial surface T near the synthetic 300 K equator value.
    assert 285.0 < T_eq_sfc < 305.0
