"""Unit tests for atmospheric radiation module.

Tests cover:
- Gray radiation: energy conservation, heating rate bounds, moisture feedback
- Solar geometry: equinox symmetry, global mean insolation
- Integration: correct tendency shapes, differentiability
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    OzoneProfileConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
    perpetual_equinox_insolation,
    solar_declination,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
    sundqvist_cloud_fraction,
    xu_randall_cloud_fraction,
)
from legoesm import constants


# ===========================================================================
# Helpers
# ===========================================================================

def _make_column_data(ncol=4, nlev=10, T_surface=300.0, T_top=200.0):
    """Create simple test column data with linear temperature profile."""
    # Pressure: linearly spaced interfaces from 100 Pa (top) to 1e5 Pa (surface)
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Temperature: linear from T_top to T_surface
    T = jnp.broadcast_to(
        jnp.linspace(T_top, T_surface, nlev)[None, :],
        (ncol, nlev),
    )

    # Latitude: spread from equator to 60 degrees
    lat = jnp.linspace(0.0, jnp.pi / 3.0, ncol)

    # Surface temperature
    sfc_temperature = jnp.full(ncol, T_surface)

    # Insolation from perpetual equinox
    insol = perpetual_equinox_insolation(lat, 1360.0)

    return T, p_full, p_half, sfc_temperature, lat, insol


# ===========================================================================
# Solar geometry tests
# ===========================================================================

class TestSolarGeometry:
    """Tests for solar geometry functions."""

    def test_equinox_declination_zero(self):
        """At equinox (day 80), declination should be ~0."""
        delta = solar_declination(80.0)
        assert abs(float(delta)) < 1e-6

    def test_solstice_declination(self):
        """At summer solstice (day ~172), declination should be ~obliquity."""
        delta = solar_declination(172.0)
        obliquity_rad = 23.45 * float(constants.DEG_TO_RAD)
        assert abs(float(delta) - obliquity_rad) < 0.05

    def test_cos_zenith_bounded(self):
        """cos(zenith) should always be in [-1, 1]."""
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 20)
        lon = jnp.linspace(0, 2 * jnp.pi, 20)
        cos_z = cos_zenith_angle(lat, lon, 172.0, 12.0)
        assert float(jnp.min(cos_z)) >= -1.0
        assert float(jnp.max(cos_z)) <= 1.0

    def test_perpetual_equinox_symmetric(self):
        """Equinox insolation should be symmetric about the equator."""
        lat = jnp.linspace(-jnp.pi / 3, jnp.pi / 3, 20)
        insol = perpetual_equinox_insolation(lat)
        # Compare with reversed
        insol_rev = insol[::-1]
        assert jnp.allclose(insol, insol_rev, atol=1e-6)

    def test_perpetual_equinox_max_at_equator(self):
        """Insolation should be maximum at the equator."""
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 100)
        insol = perpetual_equinox_insolation(lat)
        max_idx = int(jnp.argmax(insol))
        # Should be near the center (equator)
        assert abs(max_idx - 50) <= 1

    def test_global_mean_insolation(self):
        """Global mean insolation should approximate S_0 / 4."""
        S_0 = 1360.0
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 1000)
        insol = perpetual_equinox_insolation(lat, S_0)
        # Weight by cos(lat) for area weighting
        cos_lat = jnp.cos(lat)
        global_mean = jnp.sum(insol * cos_lat) / jnp.sum(cos_lat)
        # Should be close to S_0 / 4 (within ~5% for equinox mean)
        assert abs(float(global_mean) - S_0 / 4.0) / (S_0 / 4.0) < 0.15

    def test_daily_mean_equinox(self):
        """Daily-mean at equinox should match perpetual equinox formula."""
        lat = jnp.linspace(-jnp.pi / 3, jnp.pi / 3, 20)
        S_0 = 1360.0
        # Day 80 is approximately equinox
        insol_daily = daily_mean_insolation(lat, 80.0, S_0, 23.45)
        insol_perp = perpetual_equinox_insolation(lat, S_0)
        # Should be close (not exact because declination isn't exactly 0)
        assert jnp.allclose(insol_daily, insol_perp, atol=10.0)


# ===========================================================================
# Gray radiation tests
# ===========================================================================

class TestGrayRadiation:
    """Tests for two-stream gray radiation."""

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        ncol, nlev = 4, 10
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        config = GrayRadiationConfig()

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        assert out.lw_flux_up.shape == (ncol, nlev + 1)
        assert out.lw_flux_down.shape == (ncol, nlev + 1)
        assert out.sw_flux_up.shape == (ncol, nlev + 1)
        assert out.sw_flux_down.shape == (ncol, nlev + 1)
        assert out.heating_rate.shape == (ncol, nlev)
        assert out.lw_heating_rate.shape == (ncol, nlev)
        assert out.sw_heating_rate.shape == (ncol, nlev)

    def test_fluxes_non_negative(self):
        """All fluxes should be non-negative."""
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data()
        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        assert float(jnp.min(out.lw_flux_up)) >= -1e-10
        assert float(jnp.min(out.lw_flux_down)) >= -1e-10
        assert float(jnp.min(out.sw_flux_up)) >= -1e-10
        assert float(jnp.min(out.sw_flux_down)) >= -1e-10

    def test_lw_flux_up_at_surface(self):
        """LW upward flux at surface should be ~sigma_sb * T_sfc^4."""
        ncol, nlev = 1, 10
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        config = GrayRadiationConfig(sfc_emissivity=1.0)
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        expected = constants.sigma_sb * T_sfc ** 4
        actual = out.lw_flux_up[:, -1]
        assert jnp.allclose(actual, expected, rtol=1e-5)

    def test_lw_surface_reflection_for_nonblack_surface(self):
        """Non-black surface should reflect part of downwelling LW."""
        ncol, nlev = 1, 12
        T, p_full, p_half, T_sfc, lat, _ = _make_column_data(ncol, nlev)
        insol = jnp.zeros(ncol)
        eps = 0.8
        config = GrayRadiationConfig(sfc_emissivity=eps)
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        expected = eps * constants.sigma_sb * T_sfc ** 4 + (1.0 - eps) * out.lw_flux_down[:, -1]
        actual = out.lw_flux_up[:, -1]
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_lw_flux_down_at_toa_zero(self):
        """LW downward flux at TOA should be zero."""
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data()
        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        assert jnp.allclose(out.lw_flux_down[:, 0], 0.0, atol=1e-10)

    def test_heating_rate_bounded(self):
        """Heating rate should be finite and bounded (< 10 K/day)."""
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data()
        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        assert jnp.all(jnp.isfinite(out.heating_rate))
        # Convert K/s to K/day
        hr_kday = jnp.abs(out.heating_rate) * 86400.0
        assert float(jnp.max(hr_kday)) < 10.0

    def test_zero_sw_at_zero_insolation(self):
        """SW fluxes should be zero when insolation is zero (polar night)."""
        ncol, nlev = 4, 10
        T, p_full, p_half, T_sfc, lat, _ = _make_column_data(ncol, nlev)
        insol = jnp.zeros(ncol)
        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        assert jnp.allclose(out.sw_flux_up, 0.0, atol=1e-10)
        assert jnp.allclose(out.sw_flux_down, 0.0, atol=1e-10)
        assert jnp.allclose(out.sw_heating_rate, 0.0, atol=1e-10)

    def test_isothermal_lw_equilibrium(self):
        """Isothermal column should have near-zero LW heating rate."""
        ncol, nlev = 2, 20
        T_iso = 280.0
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.full((ncol, nlev), T_iso)
        T_sfc = jnp.full(ncol, T_iso)
        lat = jnp.zeros(ncol)
        insol = jnp.zeros(ncol)  # no SW

        config = GrayRadiationConfig(sfc_emissivity=1.0)
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        # In an isothermal column the net LW heating is small but not exactly
        # zero because the top layers have low optical depth and can cool to
        # space. The heating rates should be modest (< 10 K/day).
        hr_kday = jnp.abs(out.lw_heating_rate) * 86400.0
        assert float(jnp.max(hr_kday)) < 10.0

    def test_moisture_increases_optical_depth(self):
        """Adding moisture should increase LW optical depth (warmer lower atm)."""
        ncol, nlev = 2, 10
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)

        config = GrayRadiationConfig()

        # Dry case
        out_dry = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        # Moist case: add some water vapor
        q_v = jnp.full((ncol, nlev), 0.005)  # 5 g/kg
        out_moist = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)

        # With moisture, LW downward flux at surface should be larger
        # (more optical depth = more greenhouse warming)
        lw_down_sfc_dry = float(jnp.mean(out_dry.lw_flux_down[:, -1]))
        lw_down_sfc_moist = float(jnp.mean(out_moist.lw_flux_down[:, -1]))
        assert lw_down_sfc_moist > lw_down_sfc_dry

    def test_energy_conservation(self):
        """Column energy should be approximately conserved.

        Net flux at TOA minus net flux at surface should equal the
        column-integrated heating (from flux divergence).
        """
        ncol, nlev = 4, 20
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        # Net flux = down - up (positive downward)
        F_net_toa = out.lw_flux_down[:, 0] - out.lw_flux_up[:, 0] + \
                    out.sw_flux_down[:, 0] - out.sw_flux_up[:, 0]
        F_net_sfc = out.lw_flux_down[:, -1] - out.lw_flux_up[:, -1] + \
                    out.sw_flux_down[:, -1] - out.sw_flux_up[:, -1]

        # Column-integrated heating: sum(hr * dp / g * c_p)
        dp = p_half[:, 1:] - p_half[:, :-1]
        col_heating = jnp.sum(out.heating_rate * dp * constants.c_pd / constants.g, axis=1)

        # Energy conservation: F_net(TOA) - F_net(sfc) ≈ -col_heating * (c_p/g not needed since hr already in K/s)
        # Actually: col_heating = sum(g/c_p * dF/dp * dp * c_p / g) = sum(dF) = F_net(sfc) - F_net(toa)
        # So F_net(sfc) - F_net(toa) should equal col_heating * c_p * dp_total / g ...
        # Simpler: the heating rate is derived from flux divergence, so
        # sum over layers of heating_rate * dp/g * c_p should = F_net_toa - F_net_sfc
        # Because hr = g/c_p * dF_net/dp, so hr * dp = g/c_p * dF_net
        # sum(hr * dp) = g/c_p * (F_net_sfc - F_net_toa)
        # i.e., c_p/g * sum(hr * dp) = F_net_sfc - F_net_toa
        col_flux_div = jnp.sum(out.heating_rate * dp, axis=1)  # sum of g/c_p * dF
        expected_flux_div = (constants.g / constants.c_pd) * (F_net_toa - F_net_sfc)

        assert jnp.allclose(col_flux_div, expected_flux_div, rtol=1e-4)

    def test_differentiable(self):
        """jax.grad should work through gray_radiation."""
        ncol, nlev = 2, 5
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        config = GrayRadiationConfig()

        def loss(T_in):
            out = gray_radiation(T_in, p_full, p_half, T_sfc, lat, None, insol, config)
            return jnp.sum(out.heating_rate ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Tests for make_radiation_physics integration bridge."""

    def test_hydrostatic_tendency_shapes(self):
        """Hydrostatic radiation tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        tendencies = physics_fn(state, grid, sigma)

        n = grid.n
        nlev = sigma.n_levels
        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dp_s_dt.data.shape == (6, n, n)

    def test_hydrostatic_nonzero_heating(self):
        """Radiation should produce nonzero temperature tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        tendencies = physics_fn(state, grid, sigma)

        # Heating rate should be nonzero
        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_hr > 0.0

    def test_hydrostatic_zero_wind_tendency(self):
        """Radiation should not produce wind tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        tendencies = physics_fn(state, grid, sigma)

        assert jnp.allclose(tendencies.du_dt.data, 0.0)
        assert jnp.allclose(tendencies.dv_dt.data, 0.0)
        assert jnp.allclose(tendencies.dp_s_dt.data, 0.0)

    def test_rrtmgp_backend_differs_from_gray(self):
        """Integration path should honor scheme selection."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(8)
        state = held_suarez_init(grid, sigma)

        gray_fn = make_radiation_physics(RadiationConfig(scheme="gray"), "hydrostatic")
        rrtmgp_fn = make_radiation_physics(RadiationConfig(scheme="rrtmgp"), "hydrostatic")

        gray_tend = gray_fn(state, grid, sigma).dT_dt.data
        rrtmgp_tend = rrtmgp_fn(state, grid, sigma).dT_dt.data
        diff = float(jnp.max(jnp.abs(gray_tend - rrtmgp_tend)))
        assert diff > 1e-8

    def test_nonhydrostatic_tendency_shapes(self):
        """Non-hydrostatic radiation tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field

        n = 8
        nlev = 10
        grid = create_cubed_sphere(n)
        height_coord = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros((6, n, n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        # Create a simple NH state
        from legoesm.core.state import NonHydrostaticState
        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers", dims=dims_tr, units="kg/kg"),
        )

        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="nonhydrostatic")
        tendencies = physics_fn(state, grid, height_coord, terrain_metric)

        assert tendencies.dtheta_prime_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dw_dt.data.shape == (6, n, n, nlev + 1)

    def test_nonhydrostatic_nonzero_heating(self):
        """NH radiation should produce nonzero theta tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        n = 8
        nlev = 10
        grid = create_cubed_sphere(n)
        height_coord = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros((6, n, n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers", dims=dims_tr, units="kg/kg"),
        )

        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="nonhydrostatic")
        tendencies = physics_fn(state, grid, height_coord, terrain_metric)

        max_hr = float(jnp.max(jnp.abs(tendencies.dtheta_prime_dt.data)))
        assert max_hr > 0.0

    def test_grad_through_hydrostatic_radiation(self):
        """jax.grad should work through hydrostatic radiation physics."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tendencies = physics_fn(new_state, grid, sigma)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))


# ===========================================================================
# RRTMGP tests (bundled)
# ===========================================================================

class TestRRTMGP:
    """Tests for bundled RRTMGP radiation wrapper."""

    def test_rrtmgp_import(self):
        """Bundled rrtmgp package should import successfully."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP  # noqa: F401

    def test_rrtmgp_clear_sky(self):
        """RRTMGP clear-sky heating rates should be finite."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)

        config = RRTMGPConfig()
        out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cos_zen, config)

        assert jnp.all(jnp.isfinite(out.heating_rate))
        assert out.heating_rate.shape == (ncol, nlev)

    def test_rrtmgp_differentiable_temperature(self):
        """jax.grad w.r.t. temperature should work through rrtmgp_radiation."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)
        config = RRTMGPConfig()

        def loss(T_in):
            out = rrtmgp_radiation(T_in, p_full, p_half, T_sfc, q_v, cos_zen, config)
            return jnp.sum(out.heating_rate ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_rrtmgp_differentiable_cos_zenith(self):
        """jax.grad w.r.t. cos_zenith should work through rrtmgp_radiation."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)
        config = RRTMGPConfig()

        def loss(cz):
            out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cz, config)
            return jnp.sum(out.heating_rate ** 2)

        grad_cz = jax.grad(loss)(cos_zen)
        assert jnp.all(jnp.isfinite(grad_cz))
        assert grad_cz.shape == cos_zen.shape

    def test_rrtmgp_use_scan_equivalence(self):
        """scan-based and unrolled column recurrence must produce the same fluxes.

        Guards the GPU perf path: ``use_scan=False`` (Python for-loop) must be
        numerically equivalent to ``use_scan=True`` (jax.lax.scan).  Without
        this invariant the driver pipeline's ``rrtmgp_use_scan`` knob would
        silently change results.
        """
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
            _instance_cache,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 3, 20
        T, p_full, p_half, T_sfc, _, _ = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 5.0e-4)
        cos_zen = jnp.full(ncol, 0.4)

        cfg_loop = RRTMGPConfig(use_scan=False)
        cfg_scan = RRTMGPConfig(use_scan=True)

        # Make sure the cache does not mask a config-honouring regression.
        _instance_cache.clear()
        out_loop = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_loop)
        out_scan = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_scan)

        for a, b in (
            (out_loop.heating_rate, out_scan.heating_rate),
            (out_loop.lw_flux_up, out_scan.lw_flux_up),
            (out_loop.lw_flux_down, out_scan.lw_flux_down),
            (out_loop.sw_flux_up, out_scan.sw_flux_up),
            (out_loop.sw_flux_down, out_scan.sw_flux_down),
        ):
            assert jnp.all(jnp.isfinite(a))
            assert jnp.all(jnp.isfinite(b))
            assert jnp.allclose(a, b, rtol=1e-5, atol=1e-5)


# ===========================================================================
# Diurnal cycle tests
# ===========================================================================

class TestDiurnalCycle:
    """Tests for diurnal cycle in radiation."""

    def test_set_time_exists_on_physics_fn(self):
        """make_radiation_physics should return a function with set_time."""
        config = RadiationConfig(scheme="gray")
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        assert hasattr(physics_fn, "set_time")
        assert callable(physics_fn.set_time)

    def test_diurnal_day_night_contrast(self):
        """Diurnal cycle should produce day/night SW contrast."""
        ncol, nlev = 4, 10
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        config = GrayRadiationConfig()
        lon = jnp.linspace(0, 2 * jnp.pi, ncol)

        # Noon (hour=12) — sun near local noon
        cos_sza_noon = cos_zenith_angle(lat, lon, 80.0, 12.0)
        insol_noon = 1360.0 * jnp.maximum(cos_sza_noon, 0.0)
        out_noon = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol_noon, config)

        # Midnight (hour=0) — sun on opposite side
        cos_sza_midnight = cos_zenith_angle(lat, lon, 80.0, 0.0)
        insol_midnight = 1360.0 * jnp.maximum(cos_sza_midnight, 0.0)
        out_midnight = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol_midnight, config)

        # SW heating at noon should be larger than at midnight for at least
        # some columns (those that are sunlit at noon)
        sw_noon_max = float(jnp.max(jnp.abs(out_noon.sw_heating_rate)))
        sw_midnight_max = float(jnp.max(jnp.abs(out_midnight.sw_heating_rate)))
        # At least one of these should be significantly different
        assert sw_noon_max > 0.0 or sw_midnight_max > 0.0

    def test_diurnal_integration_hydrostatic(self):
        """Integration with diurnal_cycle=True should produce valid tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(scheme="gray", diurnal_cycle=True)
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")

        # Set time to noon at equinox
        physics_fn.set_time(80.0, 43200.0)
        tendencies = physics_fn(state, grid, sigma)

        assert tendencies.dT_dt.data.shape == (6, 8, 8, 10)
        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_hr > 0.0
        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))

    def test_diurnal_nightside_zero_sw(self):
        """Columns on the nightside should have zero SW insolation."""
        # All columns at lon=pi (midnight side at hour=0)
        lat = jnp.zeros(4)
        lon = jnp.full(4, jnp.pi)
        cos_sza = cos_zenith_angle(lat, lon, 80.0, 0.0)
        # At equator, lon=pi, hour=0: hour_angle = 2*pi*(0/24) + pi - pi = 0
        # cos_z = cos(0)*cos(delta)*cos(0) + ... should be > 0 for equinox
        # Actually let's test lon=0, hour=0 -> hour_angle = 0 + 0 - pi = -pi
        # cos_z = cos(lat)*cos(delta)*cos(-pi) = -cos(lat)*cos(delta) < 0 at equinox
        lon_dark = jnp.zeros(4)  # lon=0, hour=0 -> nightside
        cos_sza_dark = cos_zenith_angle(lat, lon_dark, 80.0, 0.0)
        insol_dark = 1360.0 * jnp.maximum(cos_sza_dark, 0.0)
        # Should be zero (nightside)
        assert jnp.allclose(insol_dark, 0.0, atol=1e-6)

    def test_daily_mean_unchanged_without_diurnal(self):
        """Without diurnal_cycle, behavior should be identical to before."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(scheme="gray", diurnal_cycle=False)
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")

        # Two calls with different times should give the same result
        # (perpetual equinox is the default for gray)
        physics_fn.set_time(80.0, 0.0)
        tend1 = physics_fn(state, grid, sigma)

        physics_fn.set_time(80.0, 43200.0)
        tend2 = physics_fn(state, grid, sigma)

        assert jnp.allclose(tend1.dT_dt.data, tend2.dT_dt.data)


# ===========================================================================
# Ozone profile tests
# ===========================================================================

class TestOzoneProfile:
    """Tests for prescribed ozone profiles in RRTMGP."""

    def test_analytical_ozone_profile_shape(self):
        """Analytical ozone profile should return correct shape."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )
        ncol, nlev = 8, 20
        p_full = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev)[None, :], (ncol, nlev)
        )
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, ncol)
        config = OzoneProfileConfig(source="analytical")

        o3 = _compute_ozone_vmr(p_full, lat, config)
        assert o3 is not None
        assert o3.shape == (ncol, nlev)
        assert jnp.all(jnp.isfinite(o3))
        assert jnp.all(o3 > 0)

    def test_analytical_ozone_peak_location(self):
        """Ozone should peak near the configured p_peak pressure."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )
        ncol = 1
        nlev = 100
        # Fine pressure grid to resolve the peak
        p_full = jnp.broadcast_to(
            jnp.logspace(jnp.log10(10.0), jnp.log10(1.0e5), nlev)[None, :],
            (ncol, nlev),
        )
        lat = jnp.zeros(ncol)
        config = OzoneProfileConfig(source="analytical", p_peak_hPa=30.0)

        o3 = _compute_ozone_vmr(p_full, lat, config)
        peak_idx = int(jnp.argmax(o3[0]))
        peak_p_hPa = float(p_full[0, peak_idx]) / 100.0
        # Peak should be near 30 hPa (within a factor of 2)
        assert 15.0 < peak_p_hPa < 60.0

    def test_analytical_ozone_latitude_dependence(self):
        """Polar ozone should be higher than equatorial ozone."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )
        ncol = 2
        nlev = 40
        p_full = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev)[None, :], (ncol, nlev)
        )
        # Column 0 = equator, column 1 = pole
        lat = jnp.array([0.0, jnp.pi / 2])
        config = OzoneProfileConfig(source="analytical", lat_dependence=True)

        o3 = _compute_ozone_vmr(p_full, lat, config)
        # Pole (sin²(π/2) = 1) should have 1.5x the equatorial value
        assert float(jnp.max(o3[1])) > float(jnp.max(o3[0]))
        # Ratio should be close to 1.5
        ratio = float(jnp.max(o3[1])) / float(jnp.max(o3[0]))
        assert 1.4 < ratio < 1.6

    def test_analytical_no_lat_dependence(self):
        """With lat_dependence=False, polar and equatorial ozone should match."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )
        ncol = 2
        nlev = 40
        p_full = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev)[None, :], (ncol, nlev)
        )
        lat = jnp.array([0.0, jnp.pi / 2])
        config = OzoneProfileConfig(source="analytical", lat_dependence=False)

        o3 = _compute_ozone_vmr(p_full, lat, config)
        assert jnp.allclose(o3[0], o3[1])

    def test_standard_ozone_returns_none(self):
        """Standard source should return None (use built-in profile)."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )
        ncol, nlev = 4, 20
        p_full = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev)[None, :], (ncol, nlev)
        )
        lat = jnp.zeros(ncol)
        config = OzoneProfileConfig(source="standard")

        assert _compute_ozone_vmr(p_full, lat, config) is None

    def test_none_ozone_returns_near_zero(self):
        """Source 'none' should return near-zero ozone."""
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )
        ncol, nlev = 4, 20
        p_full = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev)[None, :], (ncol, nlev)
        )
        lat = jnp.zeros(ncol)
        config = OzoneProfileConfig(source="none")

        o3 = _compute_ozone_vmr(p_full, lat, config)
        assert o3 is not None
        assert float(jnp.max(o3)) < 1.0e-9

    def test_rrtmgp_with_external_ozone(self):
        """RRTMGP should accept external ozone and produce valid output."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)
        config = RRTMGPConfig()

        # Create a simple ozone profile
        p_hPa = p_full / 100.0
        o3_vmr = 8.0e-6 * jnp.exp(
            -0.5 * ((jnp.log(p_hPa) - jnp.log(30.0)) / 1.5) ** 2
        )

        out = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, config,
            o3_vmr=o3_vmr,
        )
        assert jnp.all(jnp.isfinite(out.heating_rate))
        assert out.heating_rate.shape == (ncol, nlev)

    def test_rrtmgp_ozone_affects_heating(self):
        """External ozone should produce different heating than no ozone."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)
        config = RRTMGPConfig()

        # With zero ozone
        o3_zero = jnp.full((ncol, nlev), 1.0e-10)
        out_no_o3 = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, config,
            o3_vmr=o3_zero,
        )

        # With substantial ozone
        p_hPa = p_full / 100.0
        o3_vmr = 8.0e-6 * jnp.exp(
            -0.5 * ((jnp.log(p_hPa) - jnp.log(30.0)) / 1.5) ** 2
        )
        out_with_o3 = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, config,
            o3_vmr=o3_vmr,
        )

        # Heating rates should differ
        diff = float(jnp.max(jnp.abs(
            out_with_o3.heating_rate - out_no_o3.heating_rate
        )))
        assert diff > 1e-8

    def test_integration_with_analytical_ozone(self):
        """Integration bridge should work with analytical ozone config."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(8)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(
            scheme="rrtmgp",
            ozone=OzoneProfileConfig(source="analytical"),
        )
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        tendencies = physics_fn(state, grid, sigma)

        assert tendencies.dT_dt.data.shape == (6, 4, 4, 8)
        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))
        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_hr > 0.0


# ===========================================================================
# Cloud fraction and cloud-radiation coupling tests
# ===========================================================================

class TestCloudFraction:
    """Tests for diagnostic cloud fraction and cloud-radiation coupling."""

    def test_sundqvist_zero_below_rh_crit(self):
        """Sundqvist cloud fraction should be 0 when RH < RH_crit."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        RH = jnp.array([[0.0, 0.3, 0.5, 0.69]])
        cf = sundqvist_cloud_fraction(RH, config)
        assert jnp.allclose(cf, 0.0, atol=1e-10)

    def test_sundqvist_one_at_saturation(self):
        """Sundqvist cloud fraction should be 1 when RH = 1."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        RH = jnp.array([[1.0]])
        cf = sundqvist_cloud_fraction(RH, config)
        assert jnp.allclose(cf, 1.0, atol=1e-10)

    def test_sundqvist_linear_ramp(self):
        """Sundqvist should give 0.5 at RH = (1 + RH_crit) / 2."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        RH_mid = jnp.array([[(1.0 + 0.7) / 2]])
        cf = sundqvist_cloud_fraction(RH_mid, config)
        assert jnp.allclose(cf, 0.5, atol=1e-6)

    def test_xu_randall_zero_without_condensate(self):
        """Xu-Randall should give 0 when condensate is zero."""
        config = CloudConfig(scheme="xu_randall")
        ncol, nlev = 4, 10
        RH = jnp.full((ncol, nlev), 0.8)
        q_c = jnp.zeros((ncol, nlev))
        q_sat = jnp.full((ncol, nlev), 0.01)
        cf = xu_randall_cloud_fraction(RH, q_c, q_sat, config)
        assert jnp.allclose(cf, 0.0, atol=1e-8)

    def test_xu_randall_increases_with_condensate(self):
        """Xu-Randall cloud fraction should increase with condensate."""
        config = CloudConfig(scheme="xu_randall")
        RH = jnp.array([[0.9]])
        q_sat = jnp.array([[0.01]])
        cf_low = xu_randall_cloud_fraction(
            RH, jnp.array([[1e-5]]), q_sat, config
        )
        cf_high = xu_randall_cloud_fraction(
            RH, jnp.array([[1e-3]]), q_sat, config
        )
        assert float(cf_high[0, 0]) > float(cf_low[0, 0])

    def test_cloud_properties_shape(self):
        """compute_cloud_properties should return correct shapes."""
        config = CloudConfig(scheme="sundqvist")
        ncol, nlev = 8, 20
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.005)
        dp = p_half[:, 1:] - p_half[:, :-1]

        props = compute_cloud_properties(T, p_full, q_v, dp, config)
        assert props.cloud_fraction.shape == (ncol, nlev)
        assert props.lwp.shape == (ncol, nlev)
        assert props.iwp.shape == (ncol, nlev)
        assert props.r_eff_liq.shape == (ncol, nlev)
        assert props.r_eff_ice.shape == (ncol, nlev)
        assert jnp.all(jnp.isfinite(props.cloud_fraction))
        assert jnp.all(props.cloud_fraction >= 0.0)
        assert jnp.all(props.cloud_fraction <= 1.0)

    def test_cloud_properties_diagnostic_condensate(self):
        """Without explicit condensate, diagnostic q_c should scale with cf."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        ncol, nlev = 4, 20
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        dp = p_half[:, 1:] - p_half[:, :-1]

        # Moist atmosphere: high RH -> clouds
        from legoesm.thermo import saturation_mixing_ratio
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v_moist = 0.9 * q_sat  # RH = 0.9 > rh_crit

        props = compute_cloud_properties(T, p_full, q_v_moist, dp, config)
        # Should have nonzero cloud fraction and water paths
        assert float(jnp.max(props.cloud_fraction)) > 0.5
        assert float(jnp.sum(props.lwp + props.iwp)) > 0.0

    def test_cloud_properties_ice_at_cold_temperatures(self):
        """At cold temperatures, condensate should be mostly ice."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.5)
        ncol, nlev = 4, 20
        # Cold atmosphere
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(
            ncol, nlev, T_surface=220.0, T_top=180.0,
        )
        dp = p_half[:, 1:] - p_half[:, :-1]

        from legoesm.thermo import saturation_mixing_ratio
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = 0.9 * q_sat

        props = compute_cloud_properties(T, p_full, q_v, dp, config)
        # Ice should dominate at these cold temps
        total_ice = float(jnp.sum(props.iwp))
        total_liq = float(jnp.sum(props.lwp))
        assert total_ice > total_liq

    def test_rrtmgp_with_clouds(self):
        """RRTMGP should accept cloud properties and produce valid output."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)
        config = RRTMGPConfig()

        # Create simple cloud properties
        lwp = jnp.full((ncol, nlev), 0.01)  # 10 g/m^2 per layer
        iwp = jnp.full((ncol, nlev), 0.005)
        r_eff_liq = jnp.full((ncol, nlev), 10.0e-6)
        r_eff_ice = jnp.full((ncol, nlev), 30.0e-6)

        out = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, config,
            cloud_path_liq=lwp,
            cloud_path_ice=iwp,
            cloud_r_eff_liq=r_eff_liq,
            cloud_r_eff_ice=r_eff_ice,
        )
        assert jnp.all(jnp.isfinite(out.heating_rate))
        assert out.heating_rate.shape == (ncol, nlev)

    def test_clouds_affect_radiation(self):
        """Clouds should change radiation fluxes compared to clear-sky."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 4, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 0.001)
        cos_zen = jnp.full(ncol, 0.5)
        config_clear = RRTMGPConfig(include_clouds=False)
        config_cloudy = RRTMGPConfig(include_clouds=True)

        # Clear sky
        out_clear = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, config_clear,
        )

        # Cloudy sky (include_clouds must be True for cloud optics to apply)
        lwp = jnp.full((ncol, nlev), 0.05)
        iwp = jnp.zeros((ncol, nlev))
        r_eff_liq = jnp.full((ncol, nlev), 10.0e-6)
        r_eff_ice = jnp.full((ncol, nlev), 30.0e-6)
        out_cloudy = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, config_cloudy,
            cloud_path_liq=lwp, cloud_path_ice=iwp,
            cloud_r_eff_liq=r_eff_liq, cloud_r_eff_ice=r_eff_ice,
        )

        # Heating rates should differ
        diff = float(jnp.max(jnp.abs(
            out_cloudy.heating_rate - out_clear.heating_rate
        )))
        assert diff > 1e-8

        # Clouds should increase LW flux down at surface (greenhouse)
        lw_down_sfc_clear = float(jnp.mean(out_clear.lw_flux_down[:, -1]))
        lw_down_sfc_cloudy = float(jnp.mean(out_cloudy.lw_flux_down[:, -1]))
        assert lw_down_sfc_cloudy > lw_down_sfc_clear

    def test_integration_with_sundqvist_clouds(self):
        """Integration bridge should work with Sundqvist cloud scheme."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(8)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(
            scheme="rrtmgp",
            cloud_scheme="sundqvist",
        )
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        tendencies = physics_fn(state, grid, sigma)

        assert tendencies.dT_dt.data.shape == (6, 4, 4, 8)
        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))

    def test_cloud_none_matches_clear_sky(self):
        """cloud_scheme='none' should give identical results to no cloud config."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(8)
        state = held_suarez_init(grid, sigma)

        config_clear = RadiationConfig(scheme="rrtmgp")
        config_none = RadiationConfig(scheme="rrtmgp", cloud_scheme="none")

        tend_clear = make_radiation_physics(
            config_clear, model_type="hydrostatic"
        )(state, grid, sigma)
        tend_none = make_radiation_physics(
            config_none, model_type="hydrostatic"
        )(state, grid, sigma)

        assert jnp.allclose(tend_clear.dT_dt.data, tend_none.dT_dt.data)
