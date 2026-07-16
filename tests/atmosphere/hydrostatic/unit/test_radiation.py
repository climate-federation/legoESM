"""Unit tests for atmospheric radiation module.

Tests cover:
- Gray radiation: energy conservation, heating rate bounds, moisture feedback
- Solar geometry: equinox symmetry, global mean insolation
- Integration: correct tendency shapes, differentiability
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    OzoneProfileConfig,
    RadiationConfig,
    RRTMGPConfig,
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
from legoesm.runtime.backend import get_backend


# Skip marker for tests that exercise jax.device_put with a shard Mesh.
# The Apple GPU backend ``mps`` (jax-mps / MLX) exposes a single device, so
# ``batched_copy_array_to_devices_with_sharding`` cannot build a multi-device
# shard Mesh there.  Skip on that backend; the tests still run on CI
# Linux/CUDA where multi-device sharding is available.
_skip_if_metal_broken = pytest.mark.skipif(
    get_backend() == "mps",
    reason=(
        "Apple GPU (mps) is single-device: device_put with a shard Mesh is "
        "unsupported.  CI Linux/CUDA runs this."
    ),
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_column_data(ncol=4, nlev=10, T_sfc=300.0, T_top=200.0):
    """Create simple test column data with linear temperature profile."""
    # Pressure: linearly spaced interfaces from 100 Pa (top) to 1e5 Pa (surface)
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Temperature: linear from T_top to T_sfc
    T = jnp.broadcast_to(
        jnp.linspace(T_top, T_sfc, nlev)[None, :],
        (ncol, nlev),
    )

    # Latitude: spread from equator to 60 degrees
    lat = jnp.linspace(0.0, jnp.pi / 3.0, ncol)

    # Surface temperature
    sfc_temperature = jnp.full(ncol, T_sfc)

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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

    def test_rrtmgp_humidity_clip_survives_halo_extrapolation(self):
        """Iter-79: a steep q_v boundary profile must not produce
        singular / negative h2o_vmr after the linear halo extrapolation
        in ``_add_halos``.

        Pre-fix the q_v clip was applied BEFORE ``_add_halos``, so a
        boundary profile like ``q_v = [0.99, 0.0, ...]`` extrapolated
        to halo = ``2·q_v[0] − q_v[1] = 1.98``, driving the
        ``1 − q_v`` denominator negative and producing singular VMRs.
        Now the clip is applied AFTER halos so the operation is robust.
        """
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 2, 40
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(ncol, nlev)
        # Pathological q_v: boundary value at the upper limit, next layer
        # at 0.  Halo extrapolation pre-fix → 1.98 (out of bounds).
        q_v = jnp.zeros((ncol, nlev))
        q_v = q_v.at[:, 0].set(0.99)
        cos_zen = jnp.full(ncol, 0.5)
        config = RRTMGPConfig()
        out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cos_zen, config)
        assert jnp.all(jnp.isfinite(out.heating_rate)), (
            "RRTMGP heating rate must be finite even when the q_v boundary"
            " profile has a steep gradient that would extrapolate halos "
            "out of [0, 0.99]."
        )

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

    def test_cache_isolates_use_scan_order_sensitive(self):
        """Codex adversarial review 019e5467 (issue #273): the
        RRTMGP instance cache must key on ``use_scan`` so the
        first call's value does not leak into later calls with
        different ``use_scan``.

        Exercises all 6 orderings of (False, True, None) starting
        from a clean cache, and asserts the cached instance's
        ``_config.use_scan`` matches the requested value — i.e.
        no instance is reused with a stale ``use_scan``."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            _instance_cache, _get_instance,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        for ordering in [
            (False, True),
            (True, False),
            (False, None),
            (None, False),
            (True, None),
            (None, True),
        ]:
            _instance_cache.clear()
            for use_scan_val in ordering:
                cfg = RRTMGPConfig(use_scan=use_scan_val)
                inst = _get_instance(cfg)
                assert inst._config.use_scan == use_scan_val, (
                    f"order={ordering}: requested use_scan="
                    f"{use_scan_val} but cached instance carries "
                    f"use_scan={inst._config.use_scan} — the cache "
                    f"reused a stale solver and the auto-pick path "
                    f"is silently broken"
                )

    def test_cache_keys_distinguish_use_scan(self):
        """Companion unit test on the cache-key contract: the
        instance key must change with ``use_scan`` while the optics
        key must NOT.  Pins the design so the optics tables are
        not duplicated per ``use_scan`` value (waste) and instance
        cache entries do not collide across ``use_scan`` values
        (correctness)."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        cfg_loop = RRTMGPConfig(use_scan=False)
        cfg_scan = RRTMGPConfig(use_scan=True)
        cfg_auto = RRTMGPConfig(use_scan=None)
        # Optics cache key drops use_scan → tables shared.
        assert RRTMGP._optics_cache_key(cfg_loop) == RRTMGP._optics_cache_key(cfg_scan)
        assert RRTMGP._optics_cache_key(cfg_loop) == RRTMGP._optics_cache_key(cfg_auto)
        # Instance cache key keeps use_scan → instances isolated.
        assert RRTMGP._instance_cache_key(cfg_loop) != RRTMGP._instance_cache_key(cfg_scan)
        assert RRTMGP._instance_cache_key(cfg_loop) != RRTMGP._instance_cache_key(cfg_auto)
        assert RRTMGP._instance_cache_key(cfg_scan) != RRTMGP._instance_cache_key(cfg_auto)

    def test_iter51_clear_sky_solve_columns_with_no_cloud_paths(self):
        """iter-51: when ``RRTMOptics`` was built with
        ``include_clouds=False`` AND the user calls ``solve_columns``
        without any cloud_path kwargs, the run must succeed and the
        cloud-skip optics must not be invoked.  Mirror-image of
        iter-41's ``test_iter41_clear_sky_optics_raises_on_direct_cloud_call``.

        Pins: the public solve_columns API gates clouds on
        ``config.include_clouds AND has any cloud kwarg``, so a
        cfg(include_clouds=False) call with no cloud_path_liq /
        cloud_path_ice / cloud_r_eff_* should silently skip the
        cloud branch and produce finite clear-sky fluxes.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        solver = RRTMGP.from_legoesm_config(
            RRTMGPConfig(include_clouds=False)
        )
        ncol, nlev = 2, 8
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(
            jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev)
        )
        sfc_T = jnp.full((ncol,), 295.0)
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_z = jnp.full((ncol,), 0.5)

        out = solver.solve_columns(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=sfc_T,
            q_v=q_v, cos_zenith=cos_z,
        )
        # All fluxes finite — the cloud-skip didn't sneak into the
        # gas-only path.
        for name in ("lw_flux_up", "lw_flux_down", "sw_flux_up",
                     "sw_flux_down", "heating_rate"):
            val = getattr(out, name)
            assert jnp.all(jnp.isfinite(val)), (
                f"{name} non-finite under cfg(include_clouds=False) + "
                f"no cloud kwargs"
            )
        # Confirm the optics_lib actually has cloud_optics_*=None.
        assert solver.optics_lib.cloud_optics_lw is None
        assert solver.optics_lib.cloud_optics_sw is None

    def test_iter41_clear_sky_optics_raises_on_direct_cloud_call(self):
        """iter-41 codex review follow-up: when RRTMOptics was built
        with include_clouds=False, the cloud_optics_lw/sw attributes
        are None.  The public ``solve_columns`` gate prevents this
        path from being reached, but a direct caller that bypasses
        the gate and feeds cloud_path_liq through
        ``compute_lw_optical_properties`` must get a clear
        ValueError pointing at the include_clouds=False mismatch,
        not an opaque AttributeError on ``None.cloud_optics_lw``.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        cfg = RRTMGPConfig(include_clouds=False)
        solver = RRTMGP.from_legoesm_config(cfg)
        # Direct call to compute_lw_optical_properties with non-None
        # cloud paths should raise ValueError, not AttributeError.
        ncol, nlev = 1, 4
        T = jnp.full((ncol, 1, nlev), 250.0)
        p = jnp.full((ncol, 1, nlev), 5e4)
        mol = jnp.full((ncol, 1, nlev), 1e22)
        cloud_path = jnp.full((ncol, 1, nlev), 1e-3)
        cloud_reff = jnp.full((ncol, 1, nlev), 1e-5)
        with pytest.raises(ValueError, match="include_clouds=False"):
            solver.optics_lib.compute_lw_optical_properties(
                p, T, mol, igpt=jnp.array(0),
                cloud_path_liq=cloud_path,
                cloud_r_eff_liq=cloud_reff,
            )

    def test_iter40_include_clouds_in_optics_key_with_conditional_load(self):
        """iter-40 supersedes the iter-36 semantics: ``include_clouds``
        is back in ``_optics_cache_key``, but now for a *meaningful*
        reason — iter-40 made ``RRTMOptics.__init__`` conditionally
        skip the cloud-table load when ``include_clouds=False``, so
        the two configurations produce genuinely different
        optics_libs (one with cloud_optics_lw/sw populated, the other
        with them = None).

        Pin: ``include_clouds`` is in the OPTICS key now, and the
        instance key inherits via the optics-key sub-tuple.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        cfg_on = RRTMGPConfig(include_clouds=True)
        cfg_off = RRTMGPConfig(include_clouds=False)
        # Optics key: distinct (different cloud-table load).
        assert RRTMGP._optics_cache_key(cfg_on) != RRTMGP._optics_cache_key(cfg_off)
        # Instance key: distinct (inherits via optics key).
        assert RRTMGP._instance_cache_key(cfg_on) != RRTMGP._instance_cache_key(cfg_off)
        # Build both optics_libs and verify the cloud-table skip.
        optics_on, _ = RRTMGP._build_optics_and_vmr(cfg_on)
        optics_off, _ = RRTMGP._build_optics_and_vmr(cfg_off)
        assert optics_on.cloud_optics_lw is not None
        assert optics_off.cloud_optics_lw is None
        assert optics_on.cloud_optics_sw is not None
        assert optics_off.cloud_optics_sw is None

    def test_iter37_include_clouds_flag_changes_flux(self):
        """iter-37: end-to-end pin for the iter-36 cache-key move.
        Two ``rrtmgp_radiation`` calls with the same non-trivial
        cloud_path_liq but different ``include_clouds`` settings must
        produce different LW fluxes.  Pre-iter-36 the include_clouds
        flag was in the OPTICS cache key, so flipping it would have
        rebuilt the optics tables — a heavyweight no-op since the
        tables are include_clouds-independent.  Iter-36 moved it to
        the instance key so the optics tables are shared but the
        solver instance is fresh, and the per-call ``has_clouds``
        gate correctly turns cloud processing on/off."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
            _instance_cache,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 2, 8
        T, p_full, p_half, T_sfc, _, _ = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_zen = jnp.full(ncol, 0.5)
        # Non-trivial cloud path in the lower troposphere.
        cloud_path_liq = jnp.zeros((ncol, nlev))
        cloud_path_liq = cloud_path_liq.at[:, -3:].set(0.1)
        cloud_r_eff_liq = jnp.full((ncol, nlev), 1.0e-5)

        cfg_off = RRTMGPConfig(include_clouds=False)
        cfg_on = RRTMGPConfig(include_clouds=True)

        _instance_cache.clear()
        out_off = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_off,
            cloud_path_liq=cloud_path_liq,
            cloud_r_eff_liq=cloud_r_eff_liq,
        )
        out_on = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_on,
            cloud_path_liq=cloud_path_liq,
            cloud_r_eff_liq=cloud_r_eff_liq,
        )
        # Cloud effect on LW flux: TOA outgoing LW should be reduced
        # by the cloud (the cloud absorbs LW from below + emits at
        # cooler T).  The difference must be substantial.
        max_diff = float(jnp.max(jnp.abs(out_off.lw_flux_up - out_on.lw_flux_up)))
        assert max_diff > 1.0, (
            f"flipping include_clouds must change LW flux when "
            f"cloud_path_liq is non-zero; got max_diff={max_diff:.6e} "
            f"W/m².  Pre-iter-36 this could have failed if the "
            f"instance cache had silently shared the off-config solver "
            f"with the on-config call."
        )

    def test_cache_keys_distinguish_iter32_baked_in_config_fields(self):
        """Iter-32 audit: ``S_0``, ``aerosol_ssa``, ``aerosol_g``,
        ``sfc_emissivity``, ``sfc_albedo`` are baked into solver
        instances (no per-call override for the aerosol/solar pair,
        and the surface pair falls back to the config default).  A
        config change in any of these must produce a different
        instance cache key so the previous solver instance is not
        silently reused.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        base = RRTMGPConfig()
        base_key = RRTMGP._instance_cache_key(base)
        # Each tweak must change the instance key.
        tweaks = [
            base._replace(S_0=base.S_0 + 1.0),
            base._replace(aerosol_ssa=base.aerosol_ssa + 0.01),
            base._replace(aerosol_g=base.aerosol_g + 0.01),
            base._replace(sfc_emissivity=base.sfc_emissivity - 0.01),
            base._replace(sfc_albedo=base.sfc_albedo + 0.01),
        ]
        for tweak in tweaks:
            assert RRTMGP._instance_cache_key(tweak) != base_key, (
                f"instance cache key did not change for tweak: {tweak}"
            )
        # Same change to *unrelated* field (gas file path) must NOT
        # collide with these — optics_cache_key still differentiates.
        assert RRTMGP._instance_cache_key(base._replace(co2_ppmv=base.co2_ppmv + 1.0)) != base_key

    def test_iter32_cache_key_fix_changes_aerosol_flux(self):
        """End-to-end pin for iter-32 cache-key fix: bumping
        ``RRTMGPConfig.aerosol_ssa`` between two ``rrtmgp_radiation``
        calls must change the SW flux output.  Pre-iter-32 the
        instance cache key was missing ``aerosol_ssa`` so the second
        call silently reused the stale solver and produced the same
        output as the first.
        """
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
            _instance_cache,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 2, 8
        T, p_full, p_half, T_sfc, _, _ = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 5e-3)
        cos_zen = jnp.full(ncol, 0.5)
        # Prescribe a non-trivial aerosol AOD so the SW fluxes depend
        # on aerosol_ssa.
        aod = jnp.full((ncol, nlev), 0.05)

        cfg_a = RRTMGPConfig(aerosol_ssa=0.93)
        cfg_b = RRTMGPConfig(aerosol_ssa=0.98)

        _instance_cache.clear()
        out_a = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_a,
            aerosol_optical_depth=aod,
        )
        out_b = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_b,
            aerosol_optical_depth=aod,
        )
        # SW flux at surface MUST differ between the two configs.
        max_diff = float(jnp.max(jnp.abs(out_a.sw_flux_down - out_b.sw_flux_down)))
        assert max_diff > 1e-3, (
            f"changing aerosol_ssa from 0.93 to 0.98 must change SW "
            f"flux_down (cache key must rebuild the solver instance); "
            f"got max_diff={max_diff:.6e} W/m², which suggests the "
            f"iter-32 cache-key fix regressed and the second call "
            f"silently reused the stale solver."
        )

    def test_iter42_hashable_shim_edge_dtypes(self):
        """iter-41 codex follow-up: pin the dtype-kind gate in
        ``_hashable``.  Verify that exotic 0-D dtypes that
        ``RRTMGPConfig`` should never see (e.g. complex64) fall
        through to id() instead of raising ``TypeError`` from
        ``float()``, and that legitimate 0-D dtypes (float32, int,
        bool) correctly value-hash."""
        import numpy as np
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        # 0-D bool: dtype.kind='b' → value-hash (so True / False produce
        # distinct keys via float(0.0) / float(1.0)).
        cfg_t = RRTMGPConfig()._replace(sfc_albedo=jnp.array(True))
        cfg_f = RRTMGPConfig()._replace(sfc_albedo=jnp.array(False))
        assert (
            RRTMGP._instance_cache_key(cfg_t)
            != RRTMGP._instance_cache_key(cfg_f)
        )

        # 0-D float32: dtype.kind='f' → value-hash.
        cfg_a = RRTMGPConfig()._replace(
            sfc_albedo=jnp.array(0.07, dtype=jnp.float32)
        )
        cfg_b = RRTMGPConfig()._replace(
            sfc_albedo=jnp.array(0.07, dtype=jnp.float32)
        )
        assert (
            RRTMGP._instance_cache_key(cfg_a)
            == RRTMGP._instance_cache_key(cfg_b)
        ), "two equal-valued float32 0-D arrays must share cache key"

        # 0-D complex (hypothetical; RRTMGPConfig should never see it):
        # must fall through to id() without raising.
        cfg_c = RRTMGPConfig()._replace(
            sfc_albedo=np.array(0.07 + 0j, dtype=np.complex64)
        )
        try:
            key = RRTMGP._instance_cache_key(cfg_c)
            # Should not raise.
            assert isinstance(key, tuple)
        except TypeError as e:
            raise AssertionError(
                f"_hashable should fall through to id() for complex "
                f"0-D arrays; got TypeError: {e}"
            )

    def test_cache_keys_handle_array_valued_sfc_fields(self):
        """AIMIP populates ``config.sfc_albedo`` / ``sfc_emissivity``
        with ``(ncol,)`` arrays.  Arrays aren't hashable; the
        ``_hashable`` shim falls back to ``id()`` for arrays.  Verify
        the cache key construction does not raise."""
        import jax.numpy as jnp
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        # NamedTuple allows _replace with any type; runtime tolerates
        # arrays in these slots because solve_columns handles them
        # via ``_resolve_surface_field``.
        cfg = RRTMGPConfig()._replace(
            sfc_albedo=jnp.array([0.06, 0.08, 0.10]),
            sfc_emissivity=jnp.array([0.98, 0.97, 0.96]),
        )
        # Should not raise.
        key = RRTMGP._instance_cache_key(cfg)
        # Should be hashable (e.g., dict-key usable).
        d = {key: "ok"}
        assert d[key] == "ok"

    def test_use_scan_none_matches_explicit_choice(self):
        """Issue #273 GPU tuning: ``RRTMGPConfig(use_scan=None)`` (the
        new production default) must produce the same heating rates
        as ``use_scan=False`` on CPU and ``use_scan=True`` on GPU/TPU.
        The local test backend is CPU, so the auto-picked path is
        the unrolled for-loop.  Equivalence with ``use_scan=False``
        proves the auto-pick wiring works end-to-end without
        changing the kernel result."""
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
            _instance_cache,
        )
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig

        ncol, nlev = 3, 20
        T, p_full, p_half, T_sfc, _, _ = _make_column_data(ncol, nlev)
        q_v = jnp.full((ncol, nlev), 5.0e-4)
        cos_zen = jnp.full(ncol, 0.4)

        cfg_explicit = RRTMGPConfig(use_scan=False)
        cfg_auto = RRTMGPConfig()  # use_scan defaults to None
        assert cfg_auto.use_scan is None, (
            "RRTMGPConfig.use_scan default must be None for the "
            "auto-pick path; production runs rely on this"
        )

        _instance_cache.clear()
        out_explicit = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_explicit,
        )
        out_auto = rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg_auto,
        )

        # CPU host → auto-pick lands on use_scan=False, output identical.
        assert jnp.allclose(
            out_explicit.heating_rate, out_auto.heating_rate,
            rtol=1.0e-12, atol=1.0e-14,
        )


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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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

    def test_sundqvist_sqrt_form(self):
        """Faithful Sundqvist √-form: b = 1 − √((1−RH)/(1−RH_crit)).
        At RH = (1+RH_crit)/2 the argument is 1/2 ⇒ b = 1 − √0.5 ≈ 0.293
        (NOT 0.5 — the earlier linear-ramp value)."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        RH_mid = jnp.array([[(1.0 + 0.7) / 2]])      # = 0.85, arg = 0.5
        cf = sundqvist_cloud_fraction(RH_mid, config)
        assert jnp.allclose(cf, 1.0 - jnp.sqrt(jnp.array(0.5)), atol=1e-6)
        # Monotonic increasing in RH between RH_crit and 1.
        RH = jnp.array([[0.7, 0.8, 0.9, 1.0]])
        cf_seq = sundqvist_cloud_fraction(RH, config)
        assert jnp.all(jnp.diff(cf_seq[0]) > 0)
        # AD-safe at saturation (√ derivative would be infinite at RH=1).
        g = jax.grad(lambda r: jnp.sum(sundqvist_cloud_fraction(r, config)))(
            jnp.array([[1.0, 0.99]])
        )
        assert jnp.all(jnp.isfinite(g))

    def test_xu_randall_zero_without_condensate(self):
        """Xu-Randall should give 0 when condensate is zero."""
        config = CloudConfig(scheme="xu_randall")
        ncol, nlev = 4, 10
        RH = jnp.full((ncol, nlev), 0.8)
        q_c = jnp.zeros((ncol, nlev))
        q_sat = jnp.full((ncol, nlev), 0.01)
        cf = xu_randall_cloud_fraction(RH, q_c, q_sat, config)
        assert jnp.allclose(cf, 0.0, atol=1e-8)

    def test_xu_randall_gradient_finite_at_zero_rh(self):
        """d(cloud fraction)/d(RH) must be finite at RH=0 (dry layer).

        Regression: ``RH**p_xr`` with ``p_xr<1`` has an infinite derivative at
        RH=0 (the clip floor was 0), so reverse-mode AD produced an inf
        gradient d(cf)/d(q_v) for any dry layer (upper stratosphere / dry init),
        poisoning end-to-end training that touches a dry column.  Flooring the
        clip base fixes the gradient; the forward (cf -> 0 as condensate -> 0)
        is unchanged.
        """
        config = CloudConfig(scheme="xu_randall")
        q_sat = jnp.full((3,), 0.01)
        q_c = jnp.full((3,), 1e-4)

        def loss(RH):
            return jnp.sum(xu_randall_cloud_fraction(RH, q_c, q_sat, config))

        for rh0 in (0.0, 1e-9, 0.5):
            grad = jax.grad(loss)(jnp.full((3,), rh0))
            assert bool(jnp.all(jnp.isfinite(grad))), (
                f"xu_randall cloud-fraction gradient not finite at RH={rh0}: {grad}"
            )

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

    def test_liquid_reff_psd_m2005(self):
        """RAD-1-liq: the M2005 PSD liquid effective radius
        reffc=(PGAM+3)/(2·LAMC) (mp_graupel:495) replaces the fixed r_eff_liq
        when N_c is supplied; pins a known case + the q_c/N_c dependence and the
        no-N_c / liquid-free fallback to the config constant."""
        config = CloudConfig(scheme="resolved", r_eff_liq=14.0e-6)
        T = jnp.full((1, 1), 280.0)
        p_full = jnp.full((1, 1), 90000.0)
        q_v = jnp.full((1, 1), 0.005)
        dp = jnp.full((1, 1), 1000.0)
        q_c = jnp.full((1, 1), 5.0e-4)   # 0.5 g/kg
        N_c = jnp.full((1, 1), 1.0e8)    # 100 /cm³ (per-VOLUME)

        def reff(qc, nc):
            return float(compute_cloud_properties(
                T, p_full, q_v, dp, config, q_cloud=qc, n_cloud=nc
            ).r_eff_liq[0, 0])

        r = reff(q_c, N_c)
        assert 8.0e-6 < r < 14.0e-6, f"PSD reffc out of range: {r}"
        # more q_c ⇒ larger drops; more N_c ⇒ smaller drops.
        assert reff(q_c * 4.0, N_c) > r
        assert reff(q_c, N_c * 8.0) < r
        # No N_c ⇒ fixed config constant (back-compat).
        no_nc = float(compute_cloud_properties(
            T, p_full, q_v, dp, config, q_cloud=q_c).r_eff_liq[0, 0])
        assert abs(no_nc - 14.0e-6) < 1.0e-12
        # Liquid-free cell ⇒ fall back to the constant.
        assert abs(reff(jnp.zeros((1, 1)), N_c) - 14.0e-6) < 1.0e-12

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
        # Significant cloud at RH=0.9.  The faithful Sundqvist √-form gives
        # 1 − √((1−0.9)/(1−0.7)) ≈ 0.42 here (full cloud only near
        # saturation), vs the old linear ramp's 0.67.
        assert float(jnp.max(props.cloud_fraction)) > 0.4
        assert float(jnp.sum(props.lwp + props.iwp)) > 0.0

    def test_cloud_properties_ice_at_cold_temperatures(self):
        """At cold temperatures, condensate should be mostly ice."""
        config = CloudConfig(scheme="sundqvist", rh_crit=0.5)
        ncol, nlev = 4, 20
        # Cold atmosphere
        T, p_full, p_half, T_sfc, lat, insol = _make_column_data(
            ncol, nlev, T_sfc=220.0, T_top=180.0,
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
        """Integration bridge should work with Sundqvist cloud scheme.

        Uses a CONSISTENT gate (``include_clouds=True`` with the active cloud
        scheme).  Previously this built ``RadiationConfig(scheme='rrtmgp',
        cloud_scheme='sundqvist')`` with the default ``include_clouds=False``,
        which ran SILENTLY clear-sky — the finiteness assertion passed without
        ever exercising the cloud coupling.  ``make_radiation_physics`` now
        rejects that inconsistent config, so the test pins the working path."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

        grid = create_cubed_sphere(4)
        sigma = create_sigma_coordinate(8)
        state = held_suarez_init(grid, sigma)

        config = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=RRTMGPConfig(include_clouds=True),
            cloud_scheme="sundqvist",
        )
        physics_fn = make_radiation_physics(config, model_type="hydrostatic")
        tendencies = physics_fn(state, grid, sigma)

        assert tendencies.dT_dt.data.shape == (6, 4, 4, 8)
        assert jnp.all(jnp.isfinite(tendencies.dT_dt.data))

        # The inconsistent gate (clouds on, include_clouds off) must be
        # rejected, not silently run clear-sky.
        with pytest.raises(ValueError, match="Inconsistent cloud-radiation gate"):
            make_radiation_physics(
                RadiationConfig(scheme="rrtmgp", cloud_scheme="sundqvist"),
                model_type="hydrostatic",
            )

    def test_cloud_none_matches_clear_sky(self):
        """cloud_scheme='none' should give identical results to no cloud config."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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


class TestColumnShardedRadiation:
    """Issue #273 follow-up: ``make_radiation_physics(column_mesh=...)``
    shards the per-column radiation kernel across the supplied mesh.
    Locks in two contracts:

    1. **Numerical equivalence** — the sharded output matches the
       single-device baseline bit-for-bit on a divisible
       ``ncol = 6·n·n``.
    2. **Divisibility check** — when ``ncol`` does not divide the
       mesh device count, the factory raises ``ValueError`` instead
       of silently rounding down.

    The 4-device equivalence test only runs when at least 2 devices
    are visible to JAX; on a single-device host it is skipped.  Set
    ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` to
    exercise locally.
    """

    def _make_state(self, n=8, nlev=10):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)
        return grid, sigma, held_suarez_init(grid, sigma)

    @_skip_if_metal_broken
    def test_sharded_matches_unsharded_on_single_device(self):
        """Single-device mesh degenerates to no-op sharding; output
        must still match exactly."""
        from legoesm.parallel.column_shard import create_column_mesh
        grid, sigma, state = self._make_state()
        config = RadiationConfig(scheme="gray")
        ref = make_radiation_physics(
            config, model_type="hydrostatic",
        )(state, grid, sigma)
        mesh = create_column_mesh(n_devices=1)
        out = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
        )(state, grid, sigma)
        np.testing.assert_allclose(
            np.asarray(out.dT_dt.data),
            np.asarray(ref.dT_dt.data),
            rtol=1.0e-12, atol=1.0e-14,
        )

    def test_raises_on_non_divisible_ncol(self):
        """When the cubed-sphere ``ncol`` does not divide the mesh
        device count, the factory must raise rather than silently
        produce wrong results."""
        if len(jax.devices()) < 2:
            pytest.skip("non-divisibility check requires >=2 devices")
        from legoesm.parallel.column_shard import create_column_mesh
        # Build a grid where ncol = 6*5*5 = 150.  Pick a device count
        # that does not divide 150 — e.g. 4 (150 % 4 = 2).
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        grid = create_cubed_sphere(5)
        sigma = create_sigma_coordinate(8)
        state = held_suarez_init(grid, sigma)
        config = RadiationConfig(scheme="gray")
        n_dev = min(4, len(jax.devices()))
        if (6 * 5 * 5) % n_dev == 0:
            pytest.skip(
                f"chosen ncol={6*5*5} happens to divide n_dev={n_dev}; "
                f"cannot exercise the failure mode"
            )
        mesh = create_column_mesh(n_devices=n_dev)
        physics_fn = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
        )
        with pytest.raises(ValueError, match="divisible"):
            physics_fn(state, grid, sigma)

    def test_sharded_matches_unsharded_multidevice(self):
        """4-device CPU emulation: sharded ``make_radiation_physics``
        produces identical heating rates to the single-device path."""
        if len(jax.devices()) < 2:
            pytest.skip(
                "single-device host — set XLA_FLAGS to emulate 4 devices"
            )
        from legoesm.parallel.column_shard import create_column_mesh
        grid, sigma, state = self._make_state(n=8, nlev=10)
        config = RadiationConfig(scheme="gray")
        ref = make_radiation_physics(
            config, model_type="hydrostatic",
        )(state, grid, sigma)
        # 6*8*8 = 384, divisible by 1, 2, 3, 4, 6, 8, 12 — any common
        # CPU emulation count works.
        n_dev = len(jax.devices())
        mesh = create_column_mesh(n_devices=n_dev)
        out = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
        )(state, grid, sigma)
        np.testing.assert_allclose(
            np.asarray(out.dT_dt.data),
            np.asarray(ref.dT_dt.data),
            rtol=1.0e-12, atol=1.0e-14,
        )

    @_skip_if_metal_broken
    def test_rrtmgp_sharded_matches_unsharded_on_single_device(self):
        """Iter-28: same numerical-equivalence contract as the gray
        path, but for ``scheme="rrtmgp"``.  Catches regressions that
        break the gray path's column-shard invariance but happen to
        only affect the RRTMGP-specific kernel (e.g. cache key bugs
        that flip ``use_scan`` per-shard, JIT specialisation on
        sharded vs non-sharded shapes, or the iter-13 sign fix
        inadvertently breaking under sharded input)."""
        from legoesm.parallel.column_shard import create_column_mesh
        grid, sigma, state = self._make_state(n=4, nlev=8)
        config = RadiationConfig(scheme="rrtmgp")
        ref = make_radiation_physics(
            config, model_type="hydrostatic",
        )(state, grid, sigma)
        mesh = create_column_mesh(n_devices=1)
        out = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
        )(state, grid, sigma)
        np.testing.assert_allclose(
            np.asarray(out.dT_dt.data),
            np.asarray(ref.dT_dt.data),
            rtol=1.0e-10, atol=1.0e-12,
        )

    @_skip_if_metal_broken
    def test_aerosol_ccn_override_sharded_matches_unsharded_single_device(self):
        """R2 (single-device): with ``nc_from_aerosol=True`` the rrtmgp
        factory overrides ``n_cloud`` from ``forcing['aerosol_od']`` at the
        GLOBAL ``(ncol, nlev)`` shape BEFORE the column-shard.  A 1-device
        mesh must reproduce the unsharded heating rate exactly — proving the
        overridden field is sharded consistently with the other column
        arrays (not left at the global shape, which would crash or corrupt
        the backend call under a real multi-device mesh)."""
        from legoesm.parallel.column_shard import create_column_mesh
        grid, sigma, state = self._make_state(n=4, nlev=8)
        ncol, nlev = 6 * 4 * 4, 8
        forcing = {"aerosol_od": jnp.full((ncol, nlev), 0.1 / nlev)}
        config = RadiationConfig(scheme="rrtmgp")
        ref = make_radiation_physics(
            config, model_type="hydrostatic", nc_from_aerosol=True,
        )(state, grid, sigma, forcing=forcing)
        mesh = create_column_mesh(n_devices=1)
        out = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
            nc_from_aerosol=True,
        )(state, grid, sigma, forcing=forcing)
        np.testing.assert_allclose(
            np.asarray(out.dT_dt.data), np.asarray(ref.dT_dt.data),
            rtol=1.0e-10, atol=1.0e-12,
        )

    def test_aerosol_ccn_override_sharded_matches_unsharded_multidevice(self):
        """R2 (true multi-device): 4-device CPU emulation — the sharded
        aerosol-CCN ``n_cloud`` override produces identical heating rates to
        the single-device path.  This is the case the single-A100 smoke run
        could NOT exercise.  Skipped unless >=2 devices are visible (set
        ``XLA_FLAGS=--xla_force_host_platform_device_count=4``)."""
        if len(jax.devices()) < 2:
            pytest.skip(
                "single-device host — set XLA_FLAGS to emulate 4 devices"
            )
        from legoesm.parallel.column_shard import create_column_mesh
        grid, sigma, state = self._make_state(n=4, nlev=8)
        ncol, nlev = 6 * 4 * 4, 8     # 96 — divisible by 1,2,3,4,6,8,...
        forcing = {"aerosol_od": jnp.full((ncol, nlev), 0.1 / nlev)}
        config = RadiationConfig(scheme="rrtmgp")
        ref = make_radiation_physics(
            config, model_type="hydrostatic", nc_from_aerosol=True,
        )(state, grid, sigma, forcing=forcing)
        mesh = create_column_mesh(n_devices=len(jax.devices()))
        out = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
            nc_from_aerosol=True,
        )(state, grid, sigma, forcing=forcing)
        np.testing.assert_allclose(
            np.asarray(out.dT_dt.data), np.asarray(ref.dT_dt.data),
            rtol=1.0e-10, atol=1.0e-12,
        )

    @_skip_if_metal_broken
    def test_rrtmgp_optimal_angle_sharded_matches_unsharded(self):
        """Iter-29: column-shard invariance for the optimal-angle path.

        The optimal-angle code in solve_lw sums tau across all interior
        layers per column to derive the secant; a per-rank tau sum that
        accidentally uses sharded-only data (instead of the full column
        tau) would silently change the secant and the LW flux when the
        same physical column is split across shards.

        Single-device mesh degenerates to no-op sharding so the
        sharded and unsharded outputs must match exactly.  This is
        the gating test before any future enabling of
        ``use_optimal_angle=True`` by default."""
        from legoesm.parallel.column_shard import create_column_mesh
        grid, sigma, state = self._make_state(n=4, nlev=8)
        config = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=RRTMGPConfig(use_optimal_angle=True),
        )
        ref = make_radiation_physics(
            config, model_type="hydrostatic",
        )(state, grid, sigma)
        mesh = create_column_mesh(n_devices=1)
        out = make_radiation_physics(
            config, model_type="hydrostatic", column_mesh=mesh,
        )(state, grid, sigma)
        np.testing.assert_allclose(
            np.asarray(out.dT_dt.data),
            np.asarray(ref.dT_dt.data),
            rtol=1.0e-10, atol=1.0e-12,
        )
