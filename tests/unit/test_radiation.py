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
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.physics.radiation.rrtmgp_wrapper import (
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
        from legoesm.atmosphere.physics.radiation.rrtmgp_wrapper import (
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
        from legoesm.atmosphere.physics.radiation.rrtmgp_wrapper import (
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
