"""Category 12: Surface Albedo -- All Tiles."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.surface_albedo import (
    land_vegetation_albedo, snow_albedo, snow_cover_fraction,
    land_albedo as compute_land_albedo,
    ice_albedo as compute_ice_albedo,
    ocean_albedo as compute_ocean_albedo,
    LandAlbedoConfig, IceAlbedoConfig, OceanAlbedoConfig,
)


LCFG = LandAlbedoConfig()
ICFG = IceAlbedoConfig()
OCFG = OceanAlbedoConfig()


class Test12a_LandVegetation:
    def test_tropics(self):
        lat = jnp.array([0.0])  # equator
        alpha = land_vegetation_albedo(lat, LCFG)
        assert jnp.allclose(alpha, LCFG.alpha_veg_tropics, atol=0.02)

    def test_highlat(self):
        lat = jnp.array([1.2])  # ~69 deg
        alpha = land_vegetation_albedo(lat, LCFG)
        assert jnp.allclose(alpha, LCFG.alpha_veg_highlat, atol=0.02)

    def test_smooth_transition(self):
        lats = jnp.linspace(0.0, jnp.pi / 2, 100)
        alpha = land_vegetation_albedo(lats, LCFG)
        dalpha = jnp.abs(jnp.diff(alpha))
        # No single jump > 0.05
        assert jnp.all(dalpha < 0.05)


class Test12b_SnowAlbedoDecay:
    def test_fresh_max(self):
        assert jnp.allclose(snow_albedo(jnp.array([0.0]), LCFG), LCFG.alpha_snow_max)

    def test_old_min(self):
        assert jnp.allclose(snow_albedo(jnp.array([1e8]), LCFG), LCFG.alpha_snow_min, atol=0.01)

    def test_monotone(self):
        ages = jnp.linspace(0, 5e6, 100)
        alpha = snow_albedo(ages, LCFG)
        assert jnp.all(jnp.diff(alpha) <= 0)

    def test_bounded(self):
        alpha = snow_albedo(jnp.linspace(0, 1e8, 100), LCFG)
        assert jnp.all(alpha >= LCFG.alpha_snow_min - 1e-10)
        assert jnp.all(alpha <= LCFG.alpha_snow_max + 1e-10)


class Test12c_LandAlbedoBlend:
    def test_no_snow(self):
        lat = jnp.array([0.8])
        alpha_veg = land_vegetation_albedo(lat, LCFG)
        alpha = compute_land_albedo(lat, jnp.array([0.0]), jnp.array([0.0]), LCFG)
        assert jnp.allclose(alpha, alpha_veg)

    def test_full_snow(self):
        lat = jnp.array([0.8])
        snow_depth = jnp.array([100.0])  # >> snow_depth_crit
        alpha = compute_land_albedo(lat, snow_depth, jnp.array([0.0]), LCFG)
        assert float(alpha[0]) > float(land_vegetation_albedo(lat, LCFG)[0])

    def test_base_albedo_override_snowfree(self):
        # snow-free: a per-cell base albedo (e.g. CLM PFT map) is used verbatim,
        # NOT the latitude-band default.
        lat = jnp.array([0.4, 0.4])
        base = jnp.array([0.30, 0.13])  # bright desert vs dark forest
        alpha = compute_land_albedo(lat, jnp.zeros(2), jnp.zeros(2), LCFG,
                                    base_albedo=base)
        assert jnp.allclose(alpha, base)
        assert not jnp.allclose(alpha, land_vegetation_albedo(lat, LCFG))

    def test_base_albedo_blends_with_snow(self):
        # snow albedo still blends ON TOP of the per-cell base (deserts AND ice
        # sheets both realistic): full snow -> brighter than the base everywhere.
        lat = jnp.array([0.4, 0.4])
        base = jnp.array([0.30, 0.13])
        snowy = compute_land_albedo(lat, jnp.full(2, 100.0), jnp.zeros(2), LCFG,
                                    base_albedo=base)
        assert bool(jnp.all(snowy > base))


class Test12d_IceAlbedo:
    def test_cold_high(self):
        T_cold = jnp.array([ICFG.T_freeze - 10.0])
        alpha = compute_ice_albedo(T_cold, ICFG)
        assert jnp.allclose(alpha, ICFG.alpha_ice_cold, atol=0.01)

    def test_warm_low(self):
        T_warm = jnp.array([ICFG.T_freeze])
        alpha = compute_ice_albedo(T_warm, ICFG)
        assert jnp.allclose(alpha, ICFG.alpha_ice_warm, atol=0.01)

    def test_cold_gt_warm(self):
        assert ICFG.alpha_ice_cold > ICFG.alpha_ice_warm


class Test12e_OceanAlbedo:
    def test_constant(self):
        alpha = compute_ocean_albedo(jnp.array([0.5]), OCFG)
        assert jnp.allclose(alpha, OCFG.alpha_ocean_const)

    def test_zenith_dependent(self):
        cfg = OceanAlbedoConfig(method="zenith")
        cos_z = jnp.linspace(0.05, 1.0, 50)
        alpha = compute_ocean_albedo(cos_z, cfg)
        assert jnp.all(alpha >= 0.0)
        assert jnp.all(alpha <= 0.5)

    def test_zenith_low_sun_high_albedo(self):
        cfg = OceanAlbedoConfig(method="zenith")
        alpha_low = compute_ocean_albedo(jnp.array([0.05]), cfg)
        alpha_high = compute_ocean_albedo(jnp.array([0.9]), cfg)
        assert float(alpha_low[0]) > float(alpha_high[0])


class Test12f_AllBounded:
    def test_land_bounded(self):
        lat = jnp.linspace(-jnp.pi/2, jnp.pi/2, 50)
        alpha = compute_land_albedo(lat, jnp.full(50, 25.0), jnp.zeros(50), LCFG)
        assert jnp.all(alpha >= 0.0)
        assert jnp.all(alpha <= 1.0)

    def test_ice_bounded(self):
        T = jnp.linspace(200.0, 280.0, 50)
        alpha = compute_ice_albedo(T, ICFG)
        assert jnp.all(alpha >= 0.0)
        assert jnp.all(alpha <= 1.0)


def test_snow_never_darkens_a_brighter_base():
    """Aged snow (0.52 asymptote) over an ice-sheet band (0.82) must not pull
    the cell below the base: on the plateau the snow IS the surface.  Over a
    dark base the overlay is untouched (the floor is a no-op there)."""
    from legoesm.surface_albedo import LandAlbedoConfig, land_albedo, snow_albedo
    cfg = LandAlbedoConfig()
    lat = jnp.zeros(3)
    deep = jnp.full(3, 500.0)                    # fully snow covered
    old = jnp.full(3, 200.0 * 86400.0)           # 200-day-old snow
    aged = float(snow_albedo(old[:1], cfg)[0])
    assert aged < 0.6, aged
    base = jnp.asarray([0.10, 0.82, 0.62])
    out = land_albedo(lat, deep, old, cfg, base_albedo=base)
    assert float(out[0]) == pytest.approx(aged, rel=1e-6)    # dark base: aged snow (fp32 age decay)
    assert float(out[1]) == pytest.approx(0.82, rel=1e-12)   # ice-sheet VIS: floored
    assert float(out[2]) == pytest.approx(0.62, rel=1e-12)   # ice-sheet NIR: floored
    fresh = land_albedo(lat, deep, jnp.zeros(3), cfg, base_albedo=base)
    assert bool(jnp.all(fresh >= out))                   # fresh snow still brighter
