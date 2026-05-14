"""Unit tests for the DINO analytical bathymetry (Phase 2B, Appendix A).

Cross-checks against:
- Paper Fig 1: deep interior at H_deep=4000m, shallow walls at
  H_shallow=2000m, channel re-entrant at 45-65°S.
- Zenodo source (vopikamm/DINO@v0.2.0 MY_SRC/usrdef_zgr.F90, the
  nn_botcase=1 "bowl_cosh" branch).
- Paper Sect 2.2 / namelist: NO Mid-Atlantic Ridge.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    _exp_bathy,
    _gauss_ring,
    _smooth_step,
    dino_bathymetry,
)


# -------------------- _smooth_step ----------------------------------

def test_smooth_step_clamps_below_a():
    x = jnp.array([-1.0, 0.0])
    out = _smooth_step(x, a=0.0, b=1.0)
    assert float(out[0]) == 0.0


def test_smooth_step_clamps_above_b():
    x = jnp.array([1.0, 2.0])
    out = _smooth_step(x, a=0.0, b=1.0)
    assert float(out[1]) == 1.0


def test_smooth_step_midpoint_is_half():
    x = jnp.array([0.5])
    out = _smooth_step(x, a=0.0, b=1.0)
    assert float(out[0]) == pytest.approx(0.5)


def test_smooth_step_C2_continuous_at_endpoints():
    """Quintic 6t⁵-15t⁴+10t³ has zero derivative at both endpoints."""
    eps = 1e-3
    x_lo = jnp.array([0.0, 0.0 + eps])
    x_hi = jnp.array([1.0 - eps, 1.0])
    s_lo = _smooth_step(x_lo, 0.0, 1.0)
    s_hi = _smooth_step(x_hi, 0.0, 1.0)
    # df/dx ~ 0 at endpoints (within O(eps²))
    assert abs(float(s_lo[1] - s_lo[0]) / eps) < 1e-3
    assert abs(float(s_hi[1] - s_hi[0]) / eps) < 1e-3


# -------------------- _exp_bathy ------------------------------------

def test_exp_bathy_zero_outside_domain():
    x = jnp.array([-10.0, 60.0])  # outside [0, 50]
    out = _exp_bathy(x, x_left=0.0, x_right=50.0, width=50.0,
                    dist_lam=3.0, dist_taper=10.0)
    assert float(out[0]) == 0.0
    assert float(out[1]) == 0.0


def test_exp_bathy_one_in_interior():
    """Far from both walls the function saturates to 1."""
    x = jnp.array([25.0])  # middle of [0, 50] basin
    out = _exp_bathy(x, x_left=0.0, x_right=50.0, width=50.0,
                    dist_lam=3.0, dist_taper=10.0)
    assert float(out[0]) == pytest.approx(1.0, abs=1e-6)


def test_exp_bathy_symmetric_about_center():
    """Symmetric basin → symmetric exp_bathy values."""
    x = jnp.array([5.0, 45.0])  # mirror points across midline 25
    out = _exp_bathy(x, x_left=0.0, x_right=50.0, width=50.0,
                    dist_lam=3.0, dist_taper=10.0)
    assert float(out[0]) == pytest.approx(float(out[1]), abs=1e-9)


# -------------------- _gauss_ring -----------------------------------

def test_gauss_ring_peaks_on_ring():
    """On the ring (r = ring_radius) the result equals depth_top."""
    # Point exactly on a ring of radius 10° at angle 0
    lon = jnp.array([0.0 + 10.0])  # lon0 + ring_radius
    lat = jnp.array([0.0])
    out = _gauss_ring(
        lon, lat, lon0=0.0, lat0=0.0,
        ring_radius=10.0, dist_lam=4.0,
        depth_top=2500.0, depth_bot=4000.0,
    )
    assert float(out[0]) == pytest.approx(2500.0, abs=1e-6)


def test_gauss_ring_decays_away_from_ring():
    """Far from the ring, depth → depth_bot."""
    lon = jnp.array([100.0])  # well outside dist_lam=4°
    lat = jnp.array([0.0])
    out = _gauss_ring(
        lon, lat, lon0=0.0, lat0=0.0,
        ring_radius=10.0, dist_lam=4.0,
        depth_top=2500.0, depth_bot=4000.0,
    )
    assert float(out[0]) == pytest.approx(4000.0, abs=1.0)


def test_gauss_ring_symmetric_about_center():
    """Two points equidistant from the ring give identical depths."""
    lon = jnp.array([0.0 + 10.0, 0.0 - 10.0])
    lat = jnp.array([0.0, 0.0])
    out = _gauss_ring(
        lon, lat, lon0=0.0, lat0=0.0,
        ring_radius=10.0, dist_lam=4.0,
        depth_top=2500.0, depth_bot=4000.0,
    )
    assert float(out[0]) == pytest.approx(float(out[1]), abs=1e-9)


# -------------------- dino_bathymetry — DINO-specific ----------------

def _basin_grid(n_lon=51, n_lat=141):
    """Helper: 1° lat-lon grid covering the DINO basin."""
    cfg = DINOConfig()
    lon = np.linspace(cfg.lon_west_deg, cfg.lon_east_deg, n_lon)
    lat = np.linspace(-cfg.lat_max_deg, cfg.lat_max_deg, n_lat)
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
    return cfg, jnp.asarray(lon2d), jnp.asarray(lat2d), lon, lat


def test_bathymetry_shape_matches_input():
    cfg, lon2d, lat2d, _, _ = _basin_grid()
    bathy = dino_bathymetry(lon2d, lat2d, cfg)
    assert bathy.shape == lon2d.shape


def test_bathymetry_is_within_shallow_to_deep_range():
    cfg, lon2d, lat2d, _, _ = _basin_grid()
    bathy = dino_bathymetry(lon2d, lat2d, cfg)
    # Bathymetry should be between H_shallow (coast) and H_deep (interior),
    # except inside the sill region where it can be as shallow as H_sill.
    # Allow small numerical slack.
    assert float(jnp.min(bathy)) >= cfg.H_shallow - 1e-6
    assert float(jnp.max(bathy)) <= cfg.H_deep + 1e-6


def test_bathymetry_is_deep_in_basin_interior():
    """Far from any wall and the sill, depth ≈ H_deep."""
    cfg = DINOConfig()
    # Pick a point in the deep interior: away from N/S walls and from
    # the channel/sill — say (lon=-25°, lat=20°N).
    lon = jnp.array([[-25.0]])
    lat = jnp.array([[20.0]])
    bathy = dino_bathymetry(lon, lat, cfg)
    assert float(bathy[0, 0]) == pytest.approx(cfg.H_deep, abs=10.0)


def test_bathymetry_is_shallow_at_north_boundary():
    """At the very northern wall, depth ≈ H_shallow."""
    cfg = DINOConfig()
    lon = jnp.array([[-25.0]])  # mid-basin lon
    lat = jnp.array([[cfg.lat_max_deg]])  # at the northern wall
    bathy = dino_bathymetry(lon, lat, cfg)
    assert float(bathy[0, 0]) == pytest.approx(cfg.H_shallow, abs=1.0)


def test_bathymetry_is_shallow_at_eastern_wall():
    """At the eastern wall (north of the channel), depth ≈ H_shallow."""
    cfg = DINOConfig()
    lon = jnp.array([[cfg.lon_east_deg]])
    lat = jnp.array([[20.0]])  # well north of the channel
    bathy = dino_bathymetry(lon, lat, cfg)
    assert float(bathy[0, 0]) == pytest.approx(cfg.H_shallow, abs=1.0)


def test_bathymetry_in_channel_re_entrant():
    """Inside the channel band, the eastern boundary is OPEN.

    The basin walls vanish in the channel: at the eastern lon, channel
    latitude, depth should be > H_shallow (closer to H_deep than to
    H_shallow because the only relevant taper is the meridional one
    across the channel).
    """
    cfg = DINOConfig()
    # Eastern lon, mid-channel lat
    lon = jnp.array([[cfg.lon_east_deg]])
    lat = jnp.array([[-55.0]])  # middle of channel
    bathy = dino_bathymetry(lon, lat, cfg)
    # Interior of the channel sees no zonal wall at lon_east. The
    # meridional taper still applies; in the channel mid-latitude
    # the meridional g_phi from the full (lat_min, lat_max) extent
    # should give a value far from the wall — so bathy is close to
    # H_deep, certainly > (H_shallow + H_deep)/2 = 3000m.
    assert float(bathy[0, 0]) > 3000.0


def test_bathymetry_sill_is_shallower_than_deep():
    """On the Drake sill ring, depth should be H_sill (=2500m) or shallower.

    Take a point on the ring centered at (sill_lon_m, sill_lat_m) with
    radius = channel_width/2. At angle 0° (east of center), the point
    is (sill_lon_m + ring_radius, sill_lat_m) = (-50+10, -55) = (-40, -55).
    """
    cfg = DINOConfig()
    lon = jnp.array([[-40.0]])  # exactly on the ring at angle 0
    lat = jnp.array([[-55.0]])
    bathy = dino_bathymetry(lon, lat, cfg)
    # Should be at most ~ H_sill + some tolerance for taper
    assert float(bathy[0, 0]) <= cfg.H_sill + 50.0
    assert float(bathy[0, 0]) > 0.0


def test_bathymetry_no_NaN_or_Inf():
    cfg, lon2d, lat2d, _, _ = _basin_grid()
    bathy = dino_bathymetry(lon2d, lat2d, cfg)
    assert bool(jnp.all(jnp.isfinite(bathy)))


def test_bathymetry_zonally_uniform_far_from_walls_and_sill():
    """Outside the channel and away from the sill, depth depends only on
    latitude (zonally uniform after zonal taper saturates).
    """
    cfg = DINOConfig()
    lat = jnp.array([20.0])  # mid-basin latitude
    # Three lons in the deep interior, all far from east/west walls
    # (basin is 50° wide; pick lons -30, -25, -20)
    lon_a = jnp.array([-30.0])
    lon_b = jnp.array([-25.0])
    lon_c = jnp.array([-20.0])
    b_a = float(dino_bathymetry(lon_a, lat, cfg)[0])
    b_b = float(dino_bathymetry(lon_b, lat, cfg)[0])
    b_c = float(dino_bathymetry(lon_c, lat, cfg)[0])
    assert b_a == pytest.approx(b_b, abs=1.0)
    assert b_b == pytest.approx(b_c, abs=1.0)


def test_bathymetry_overrideable():
    cfg = DINOConfig(H_deep=3500.0, H_shallow=1000.0)
    lon = jnp.array([[-25.0]])  # interior
    lat = jnp.array([[20.0]])
    bathy = dino_bathymetry(lon, lat, cfg)
    assert float(bathy[0, 0]) == pytest.approx(3500.0, abs=10.0)


def test_bathymetry_no_mid_atlantic_ridge():
    """Sanity: at lat range -40 to 55 (where MAR would be), the
    bathymetry has no signature of a ridge at mid-basin (lon = -25°).
    Should be smooth and equal to H_deep for all those latitudes.
    """
    cfg = DINOConfig()
    lats = jnp.linspace(-39.0, 54.0, 30)
    lon = jnp.full_like(lats, -25.0)  # mid-basin
    bathy = dino_bathymetry(lon, lats, cfg)
    # Should be ~constant (no ridge bump at mid-basin)
    bathy_np = np.asarray(bathy)
    assert bathy_np.std() < 50.0  # no big variations
    assert bathy_np.mean() == pytest.approx(cfg.H_deep, abs=50.0)
