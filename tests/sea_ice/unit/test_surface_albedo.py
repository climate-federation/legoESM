"""Unit tests for surface albedo parameterizations (Task 10).

Tests cover:
- Land vegetation albedo (latitude dependence)
- Snow albedo (aging decay)
- Snow cover fraction
- Land albedo (vegetation + snow blending)
- Sea ice albedo (temperature dependence)
- Ocean albedo (constant and zenith-angle dependent)
- Slab land snow budget integration
- Multi-layer land snow budget integration
- Sea ice temperature-dependent albedo integration
- JAX differentiability
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from legoesm.surface_albedo import (
    LandAlbedoConfig,
    IceAlbedoConfig,
    OceanAlbedoConfig,
    land_vegetation_albedo,
    snow_albedo,
    snow_cover_fraction,
    land_albedo,
    ice_albedo,
    ocean_albedo,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Tight tolerances require float64 precision."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


class TestLandVegetationAlbedo:
    """Tests for latitude-dependent vegetation albedo."""

    def test_tropical_albedo(self):
        """Equatorial albedo should be near alpha_veg_tropics."""
        config = LandAlbedoConfig()
        lat = jnp.array([0.0])  # equator, radians
        alpha = land_vegetation_albedo(lat, config)
        assert abs(float(alpha[0]) - config.alpha_veg_tropics) < 0.01

    def test_high_latitude_albedo(self):
        """Polar albedo should be near alpha_veg_highlat."""
        config = LandAlbedoConfig()
        lat = jnp.array([1.3])  # ~75 degrees
        alpha = land_vegetation_albedo(lat, config)
        assert abs(float(alpha[0]) - config.alpha_veg_highlat) < 0.02

    def test_monotonic_with_latitude(self):
        """Albedo should increase with absolute latitude."""
        config = LandAlbedoConfig()
        lats = jnp.linspace(0, jnp.pi / 2, 50)
        alpha = land_vegetation_albedo(lats, config)
        # Generally non-decreasing
        assert float(alpha[-1]) >= float(alpha[0])

    def test_symmetric(self):
        """North and south should give same albedo."""
        config = LandAlbedoConfig()
        lat_n = jnp.array([0.7])
        lat_s = jnp.array([-0.7])
        assert jnp.allclose(
            land_vegetation_albedo(lat_n, config),
            land_vegetation_albedo(lat_s, config),
        )


class TestSnowAlbedo:
    """Tests for snow albedo aging."""

    def test_fresh_snow(self):
        """Fresh snow (age=0) should have maximum albedo."""
        config = LandAlbedoConfig()
        age = jnp.array([0.0])
        alpha = snow_albedo(age, config)
        npt.assert_allclose(float(alpha[0]), config.alpha_snow_max, atol=1e-10)

    def test_old_snow(self):
        """Very old snow should approach minimum albedo."""
        config = LandAlbedoConfig()
        age = jnp.array([1e8])  # very old
        alpha = snow_albedo(age, config)
        npt.assert_allclose(float(alpha[0]), config.alpha_snow_min, atol=0.01)

    def test_decay_monotonic(self):
        """Snow albedo should decrease with age."""
        config = LandAlbedoConfig()
        ages = jnp.linspace(0, 1e7, 100)
        alpha = snow_albedo(ages, config)
        # Monotonically non-increasing
        assert jnp.all(jnp.diff(alpha) <= 1e-10)


class TestSnowCoverFraction:
    """Tests for snow cover fraction."""

    def test_no_snow(self):
        """Zero snow should give zero cover fraction."""
        config = LandAlbedoConfig()
        f = snow_cover_fraction(jnp.array([0.0]), config)
        assert float(f[0]) == 0.0

    def test_full_cover(self):
        """Deep snow (>=3x crit) saturates to ~1 (Niu-Yang tanh form)."""
        config = LandAlbedoConfig()
        f = snow_cover_fraction(jnp.array([config.snow_depth_crit * 3]), config)
        assert float(f[0]) > 0.99

    def test_partial_cover_at_crit(self):
        """SWE == crit gives tanh(1) ~ 0.76 (a thin pack already masks most surface)."""
        config = LandAlbedoConfig()
        f = snow_cover_fraction(jnp.array([config.snow_depth_crit]), config)
        npt.assert_allclose(float(f[0]), float(jnp.tanh(jnp.array(1.0))), atol=1e-10)


class TestLandAlbedo:
    """Tests for combined land albedo (vegetation + snow)."""

    def test_no_snow_equals_vegetation(self):
        """Without snow, land albedo should equal vegetation albedo."""
        config = LandAlbedoConfig()
        lat = jnp.array([0.5])
        alpha = land_albedo(lat, jnp.zeros(1), jnp.zeros(1), config)
        alpha_veg = land_vegetation_albedo(lat, config)
        npt.assert_allclose(float(alpha[0]), float(alpha_veg[0]), atol=1e-10)

    def test_full_fresh_snow(self):
        """Full fresh snow cover should give snow max albedo."""
        config = LandAlbedoConfig()
        lat = jnp.array([0.5])
        snow = jnp.array([config.snow_depth_crit * 10])
        age = jnp.zeros(1)
        alpha = land_albedo(lat, snow, age, config)
        npt.assert_allclose(float(alpha[0]), config.alpha_snow_max, atol=1e-10)

    def test_snow_increases_albedo(self):
        """Adding snow should increase albedo."""
        config = LandAlbedoConfig()
        lat = jnp.array([0.5])
        alpha_bare = land_albedo(lat, jnp.zeros(1), jnp.zeros(1), config)
        alpha_snow = land_albedo(lat, jnp.array([30.0]), jnp.zeros(1), config)
        assert float(alpha_snow[0]) > float(alpha_bare[0])

    def test_differentiable(self):
        """jax.grad should work through land_albedo."""
        config = LandAlbedoConfig()
        lat = jnp.array([0.5, 1.0])

        def loss(snow):
            return jnp.sum(land_albedo(lat, snow, jnp.zeros(2), config) ** 2)

        grad = jax.grad(loss)(jnp.array([20.0, 40.0]))
        assert jnp.all(jnp.isfinite(grad))


class TestIceAlbedo:
    """Tests for temperature-dependent sea ice albedo."""

    def test_cold_ice(self):
        """Very cold ice should have cold albedo."""
        config = IceAlbedoConfig()
        T = jnp.array([200.0])  # very cold
        alpha = ice_albedo(T, config)
        npt.assert_allclose(float(alpha[0]), config.alpha_ice_cold, atol=1e-10)

    def test_warm_ice(self):
        """Ice at freezing should have warm albedo."""
        config = IceAlbedoConfig()
        T = jnp.array([config.T_freeze])
        alpha = ice_albedo(T, config)
        npt.assert_allclose(float(alpha[0]), config.alpha_ice_warm, atol=1e-10)

    def test_transition(self):
        """Mid-transition should be between cold and warm."""
        config = IceAlbedoConfig()
        T_mid = config.T_freeze - config.T_transition_width / 2
        alpha = ice_albedo(jnp.array([T_mid]), config)
        mid_alpha = (config.alpha_ice_cold + config.alpha_ice_warm) / 2
        npt.assert_allclose(float(alpha[0]), mid_alpha, atol=1e-10)

    def test_monotonic(self):
        """Ice albedo should decrease with temperature."""
        config = IceAlbedoConfig()
        T = jnp.linspace(240, 275, 50)
        alpha = ice_albedo(T, config)
        # Non-increasing
        assert jnp.all(jnp.diff(alpha) <= 1e-10)

    def test_differentiable(self):
        """jax.grad should work through ice_albedo."""
        config = IceAlbedoConfig()

        def loss(T):
            return jnp.sum(ice_albedo(T, config) ** 2)

        grad = jax.grad(loss)(jnp.array([265.0, 270.0]))
        assert jnp.all(jnp.isfinite(grad))


class TestOceanAlbedo:
    """Tests for ocean albedo."""

    def test_constant_method(self):
        """Constant method should return fixed value."""
        config = OceanAlbedoConfig(method="constant")
        alpha = ocean_albedo(jnp.array([0.5]), config)
        npt.assert_allclose(float(alpha[0]), config.alpha_ocean_const, atol=1e-10)

    def test_zenith_high_sun(self):
        """High sun (cos_zenith ~ 1) should give low ocean albedo."""
        config = OceanAlbedoConfig(method="zenith")
        alpha = ocean_albedo(jnp.array([1.0]), config)
        assert float(alpha[0]) < 0.10

    def test_zenith_low_sun(self):
        """Low sun (cos_zenith ~ 0) should give higher ocean albedo."""
        config = OceanAlbedoConfig(method="zenith")
        alpha = ocean_albedo(jnp.array([0.05]), config)
        assert float(alpha[0]) > 0.10

    def test_zenith_bounded(self):
        """Zenith-dependent albedo should be in [0.03, 0.40]."""
        config = OceanAlbedoConfig(method="zenith")
        cos_z = jnp.linspace(0.01, 1.0, 100)
        alpha = ocean_albedo(cos_z, config)
        assert jnp.all(alpha >= 0.03)
        assert jnp.all(alpha <= 0.40)

    def test_differentiable(self):
        """jax.grad should work through zenith ocean albedo."""
        config = OceanAlbedoConfig(method="zenith")

        def loss(cos_z):
            return jnp.sum(ocean_albedo(cos_z, config) ** 2)

        grad = jax.grad(loss)(jnp.array([0.3, 0.7]))
        assert jnp.all(jnp.isfinite(grad))

    def test_none_cos_zenith_returns_constant(self):
        """If cos_zenith is None, should return constant."""
        config = OceanAlbedoConfig(method="zenith")
        alpha = ocean_albedo(None, config)
        assert float(alpha) == config.alpha_ocean_const
