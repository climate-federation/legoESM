"""Faithfulness pins for the idealized ocean wind-stress profiles.

Target: ``compute_wind_stress`` in
``legoesm.ocean.physics.surface_forcing.wind_profiles`` (the 9 selectable
``wind_profile`` analytic forms + the tropical Gaussian reduction + dispatch).

Most-trustful sources
---------------------
Canonical idealized-ocean wind forcings:
- single_gyre: Stommel (1948) / Munk (1950) subtropical-gyre cosine stress.
- double_gyre: Holland & Lin (1975) double-gyre cosine stress.
- double_gyre_sin2 / _tapered / channel_sine: sin^2 / sine jets with zero
  Ekman transport at the walls (Zhang et al. 2024 channel).
- global_wind: Nikurashin & Vallis (2012)-style 3-belt zonal stress.
- two_belt: equatorial-easterlies + mid-latitude-westerly-jet Gaussian sum.

These analytic forms (and their SIGN conventions -> gyre circulation) are the
spec; a wrong sign or coefficient silently changes the ocean circulation. The
module was never oracle-pinned.

Certification (test-only)
-------------------------
1. Exact closed form vs an independent reimplementation for every profile
   (gyre/cosine/sin/sin^2/tapered/channel), to round-off.
2. Design-spec physical targets for the two coefficient-heavy belts
   (global_wind, two_belt) -- pinned to their published target amplitudes/
   crossings, NOT by re-typing the module's private polynomial coefficients.
3. Sign conventions (easterlies at boundaries, westerly jet, gyre curl); tau_y=0
   for all latitude profiles; the tropical Gaussian reduction; dispatch hardening
   on an unknown profile; differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress

jax.config.update("jax_enable_x64", True)

_DEG = np.pi / 180.0


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _lat_rad(deg):
    return _a(np.asarray(deg, dtype=np.float64) * _DEG)


def _tx(deg, cfg):
    tau_x, _ = compute_wind_stress(_lat_rad(deg), cfg)
    return np.asarray(tau_x)


def _ty(deg, cfg):
    _, tau_y = compute_wind_stress(_lat_rad(deg), cfg)
    return np.asarray(tau_y)


# Latitudes strictly inside the default basin [15, 75] deg for the gyre profiles.
_BASIN_DEG = np.array([20.0, 35.0, 50.0, 65.0, 74.0])


# ---------------------------------------------------------------------------
# 1. Exact closed forms.
# ---------------------------------------------------------------------------
def test_constant_profile():
    cfg = PrescribedForcingConfig(wind_profile="constant", tau_x=0.03, tau_y=-0.01)
    lat = np.array([-40.0, 0.0, 60.0])
    np.testing.assert_allclose(_tx(lat, cfg), 0.03, rtol=1e-12)
    np.testing.assert_allclose(_ty(lat, cfg), -0.01, rtol=1e-12)


def test_cosine_latitude():
    cfg = PrescribedForcingConfig(wind_profile="cosine_latitude", tau_max=0.12)
    lat = np.array([-60.0, -20.0, 0.0, 30.0, 80.0])
    lat_r = lat * _DEG
    expect = -0.12 * np.cos(np.pi * lat_r / (np.pi / 2.0))     # -tau_max*cos(2*lat)
    np.testing.assert_allclose(_tx(lat, cfg), expect, rtol=1e-12)
    np.testing.assert_allclose(_ty(lat, cfg), 0.0, atol=1e-15)
    # Independent hand-computed anchors: peak easterly at equator, zero at 45 deg.
    np.testing.assert_allclose(_tx(np.array([0.0]), cfg)[0], -0.12, rtol=1e-12)
    np.testing.assert_allclose(_tx(np.array([45.0]), cfg)[0], 0.0, atol=1e-12)


def test_single_gyre_stommel_munk():
    cfg = PrescribedForcingConfig(wind_profile="single_gyre", tau_max=0.1,
                                  lat_south_deg=15.0, lat_north_deg=75.0)
    lat_s, lat_n = 15.0 * _DEG, 75.0 * _DEG
    lat_r = _BASIN_DEG * _DEG
    expect = -0.1 * np.cos(np.pi * (lat_r - lat_s) / (lat_n - lat_s))
    np.testing.assert_allclose(_tx(_BASIN_DEG, cfg), expect, rtol=1e-12)
    # Sign convention: easterly (tau_x<0) at south wall, westerly (tau_x>0) at north.
    assert _tx(np.array([15.0]), cfg)[0] < 0.0
    assert _tx(np.array([75.0]), cfg)[0] > 0.0
    # Independent hand anchors: -tau_max at south wall, 0 at mid-basin, +tau_max north.
    np.testing.assert_allclose(_tx(np.array([15.0, 45.0, 75.0]), cfg),
                               np.array([-0.1, 0.0, 0.1]), atol=1e-12)


def test_double_gyre_holland_lin():
    cfg = PrescribedForcingConfig(wind_profile="double_gyre", tau_max=0.1,
                                  lat_south_deg=15.0, lat_north_deg=75.0)
    lat_s, lat_n = 15.0 * _DEG, 75.0 * _DEG
    lat_r = _BASIN_DEG * _DEG
    expect = -0.1 * np.cos(2.0 * np.pi * (lat_r - lat_s) / (lat_n - lat_s))
    np.testing.assert_allclose(_tx(_BASIN_DEG, cfg), expect, rtol=1e-12)
    # Independent hand anchors: -tau_max at south wall, 0 at quarter-basin (lat=30,
    # y=0.25 -> cos(pi/2)=0), +tau_max westerly jet at mid-basin (lat=45).
    np.testing.assert_allclose(_tx(np.array([15.0, 30.0, 45.0]), cfg),
                               np.array([-0.1, 0.0, 0.1]), atol=1e-12)
    # cos vs cos(2x) canary: double gyre reverses sign across mid-basin (single
    # gyre would not) -- south half easterly-ish, mid-basin westerly.
    assert _tx(np.array([22.0]), cfg)[0] < 0.0 and _tx(np.array([45.0]), cfg)[0] > 0.0


def test_double_gyre_sin2_nonneg_zero_at_walls():
    cfg = PrescribedForcingConfig(wind_profile="double_gyre_sin2", tau_max=0.1,
                                  lat_south_deg=15.0, lat_north_deg=75.0,
                                  wind_buffer_deg=5.0)
    buf = 5.0 * _DEG
    lat_s = 15.0 * _DEG + buf
    lat_n = 75.0 * _DEG - buf
    lat_r = _BASIN_DEG * _DEG
    y = (lat_r - lat_s) / (lat_n - lat_s)
    expect = np.where((lat_r >= lat_s) & (lat_r <= lat_n), 0.1 * np.sin(np.pi * y) ** 2, 0.0)
    np.testing.assert_allclose(_tx(_BASIN_DEG, cfg), expect, rtol=1e-12)
    assert np.all(_tx(_BASIN_DEG, cfg) >= 0.0)                 # eastward (westerly jet)
    # Hand anchor: peak at mid inset-basin (lat=45 = midpoint of [20,70]) -> tau_max.
    np.testing.assert_allclose(_tx(np.array([45.0]), cfg)[0], 0.1, rtol=1e-12)


def test_double_gyre_tapered_requires_buffer_and_zeros_walls():
    cfg = PrescribedForcingConfig(wind_profile="double_gyre_tapered", tau_max=0.1,
                                  lat_south_deg=15.0, lat_north_deg=75.0,
                                  wind_buffer_deg=5.0)
    buf = 5.0 * _DEG
    lat_s, lat_n = 15.0 * _DEG, 75.0 * _DEG
    width = lat_n - lat_s
    lat_r = _BASIN_DEG * _DEG
    y = (lat_r - lat_s) / width
    base = -0.1 * np.cos(2.0 * np.pi * y)
    d_s = np.clip((lat_r - lat_s) / buf, 0, 1)
    d_n = np.clip((lat_n - lat_r) / buf, 0, 1)
    tap_s = np.where((lat_r - lat_s) / buf < 1.0, np.sin(0.5 * np.pi * d_s) ** 2, 1.0)
    tap_n = np.where((lat_n - lat_r) / buf < 1.0, np.sin(0.5 * np.pi * d_n) ** 2, 1.0)
    expect = base * tap_s * tap_n
    np.testing.assert_allclose(_tx(_BASIN_DEG, cfg), expect, rtol=1e-12)
    # Sample the SOUTH buffer interior (16, 18 deg) and the north buffer (74 deg) so a
    # one-sided taper-shape error is caught, plus smooth zero exactly at both walls.
    south = np.array([16.0, 18.0])
    lat_rs = south * _DEG
    ys = (lat_rs - lat_s) / width
    ds = np.clip((lat_rs - lat_s) / buf, 0, 1)
    ts = np.where((lat_rs - lat_s) / buf < 1.0, np.sin(0.5 * np.pi * ds) ** 2, 1.0)
    exp_s = (-0.1 * np.cos(2.0 * np.pi * ys)) * ts * 1.0      # north taper = 1 far from north wall
    np.testing.assert_allclose(_tx(south, cfg), exp_s, rtol=1e-12)
    # Taper magnitude grows moving inward from the south wall (0 at 15 -> larger at 18).
    assert abs(_tx(np.array([18.0]), cfg)[0]) > abs(_tx(np.array([16.0]), cfg)[0]) > 0.0
    np.testing.assert_allclose(_tx(np.array([15.0, 75.0]), cfg), 0.0, atol=1e-12)


def test_channel_sine():
    cfg = PrescribedForcingConfig(wind_profile="channel_sine", tau_max=0.2,
                                  lat_south_deg=-60.0, lat_north_deg=-40.0)
    lat_s, lat_n = -60.0 * _DEG, -40.0 * _DEG
    lat = np.array([-58.0, -50.0, -42.0])
    lat_r = lat * _DEG
    expect = 0.2 * np.sin(np.pi * (lat_r - lat_s) / (lat_n - lat_s))
    np.testing.assert_allclose(_tx(lat, cfg), expect, rtol=1e-12)
    assert np.all(_tx(lat, cfg) > 0.0)                        # eastward (ACC-like)
    # Hand anchor: peak at channel center (-50 deg) -> tau_max.
    np.testing.assert_allclose(_tx(np.array([-50.0]), cfg)[0], 0.2, rtol=1e-12)
    # Outside the channel the basin-window mask zeros the stress (canary for a
    # removed jnp.where mask).
    np.testing.assert_allclose(_tx(np.array([-62.0, -38.0]), cfg), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. Design-spec targets for the coefficient-heavy belts (non-circular).
# ---------------------------------------------------------------------------
def test_global_wind_design_targets():
    # Nikurashin-Vallis-style 3-belt: -0.08 Pa easterlies at eq, zero crossings at
    # 30 and 70 deg, westerly peak ~+0.10 Pa near 50 deg (for tau_max=0.1).
    cfg = PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1)
    np.testing.assert_allclose(_tx(np.array([0.0]), cfg)[0], -0.08, atol=2e-3)
    np.testing.assert_allclose(_tx(np.array([30.0]), cfg)[0], 0.0, atol=2e-3)
    np.testing.assert_allclose(_tx(np.array([70.0]), cfg)[0], 0.0, atol=2e-3)
    assert _tx(np.array([50.0]), cfg)[0] > 0.08               # westerly peak
    # The westerly maximum actually SITS near 50 deg (not merely a value there): the
    # argmax over the mid-latitude band is within a few degrees of 50.
    band = np.arange(35.0, 66.0, 1.0)
    assert 46.0 <= band[int(np.argmax(_tx(band, cfg)))] <= 54.0
    # global_wind is hemispherically symmetric (sin^2 and cos are both even), so the
    # Southern-Hemisphere westerly peak mirrors the Northern one exactly (a one-sided
    # regression that moved only the SH jet would break this).
    d = np.array([5.0, 30.0, 50.0, 70.0])
    np.testing.assert_allclose(_tx(d, cfg), _tx(-d, cfg), rtol=1e-12)
    # tau_max scales the whole profile linearly.
    cfg2 = PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.2)
    np.testing.assert_allclose(_tx(np.array([0.0]), cfg2)[0], 2.0 * -0.08, atol=4e-3)


def test_two_belt_design_targets_and_symmetry():
    cfg = PrescribedForcingConfig(wind_profile="two_belt", tau_max=0.1)
    # Equatorial easterlies at half the westerly peak; westerly peak ~tau_max at 50.
    np.testing.assert_allclose(_tx(np.array([0.0]), cfg)[0], -0.05, atol=2e-3)
    np.testing.assert_allclose(_tx(np.array([50.0]), cfg)[0], 0.1, atol=2e-3)
    np.testing.assert_allclose(_tx(np.array([-50.0]), cfg)[0], 0.1, atol=2e-3)
    # The jet MAXIMUM sits near +-50 deg (a shifted jet would fail this even though
    # the value at exactly 50 might still pass the tolerance).
    band = np.arange(35.0, 66.0, 1.0)
    assert 46.0 <= band[int(np.argmax(_tx(band, cfg)))] <= 54.0
    # Hemispheric symmetry (even in latitude).
    d = np.array([10.0, 30.0, 55.0, 70.0])
    np.testing.assert_allclose(_tx(d, cfg), _tx(-d, cfg), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Tropical reduction, dispatch, tau_y, differentiability.
# ---------------------------------------------------------------------------
def test_tropical_wind_reduction_gaussian():
    # scale_factor = 1 + (tropical_wind_scale - 1) * exp(-0.5*(lat/sigma)^2).
    base = PrescribedForcingConfig(wind_profile="cosine_latitude", tau_max=0.12)
    red = PrescribedForcingConfig(wind_profile="cosine_latitude", tau_max=0.12,
                                  tropical_wind_scale=0.5, tropical_wind_lat_deg=15.0)
    lat = np.array([0.0, 10.0, 60.0])
    sigma = 15.0 * _DEG
    gauss = np.exp(-0.5 * (lat * _DEG / sigma) ** 2)
    scale = 1.0 + (0.5 - 1.0) * gauss
    np.testing.assert_allclose(_tx(lat, red), _tx(lat, base) * scale, rtol=1e-12)
    # At the equator the stress is exactly halved; far from it, unchanged.
    np.testing.assert_allclose(_tx(np.array([0.0]), red)[0],
                               0.5 * _tx(np.array([0.0]), base)[0], rtol=1e-12)


def test_tropical_reduction_scales_nonzero_tau_y():
    # The Gaussian reduction multiplies BOTH components; a constant profile with a
    # nonzero tau_y catches a reduction that forgets the meridional stress.
    cfg = PrescribedForcingConfig(wind_profile="constant", tau_x=0.03, tau_y=-0.02,
                                  tropical_wind_scale=0.5, tropical_wind_lat_deg=15.0)
    lat = np.array([0.0, 10.0, 60.0])
    sigma = 15.0 * _DEG
    scale = 1.0 + (0.5 - 1.0) * np.exp(-0.5 * (lat * _DEG / sigma) ** 2)
    np.testing.assert_allclose(_tx(lat, cfg), 0.03 * scale, rtol=1e-12)
    np.testing.assert_allclose(_ty(lat, cfg), -0.02 * scale, rtol=1e-12)
    np.testing.assert_allclose(_ty(np.array([0.0]), cfg)[0], -0.01, rtol=1e-12)  # halved at eq


def test_tau_y_zero_for_all_latitude_profiles():
    lat = np.array([-30.0, 20.0, 55.0])
    for p in ("cosine_latitude", "single_gyre", "double_gyre", "double_gyre_sin2",
              "double_gyre_tapered", "channel_sine", "global_wind", "two_belt"):
        cfg = PrescribedForcingConfig(wind_profile=p, tau_max=0.1,
                                      lat_south_deg=-60.0, lat_north_deg=75.0,
                                      wind_buffer_deg=5.0)
        np.testing.assert_allclose(_ty(lat, cfg), 0.0, atol=1e-15)


def test_dispatch_hardening_unknown_profile_raises():
    cfg = PrescribedForcingConfig(wind_profile="not_a_profile")
    with pytest.raises(ValueError):
        compute_wind_stress(_lat_rad(np.array([0.0])), cfg)


def test_differentiable_in_latitude():
    cfg = PrescribedForcingConfig(wind_profile="cosine_latitude", tau_max=0.1)
    g = float(jax.grad(lambda phi: compute_wind_stress(jnp.reshape(phi, (1,)), cfg)[0][0])(0.3))
    assert np.isfinite(g) and abs(g) > 0.0
