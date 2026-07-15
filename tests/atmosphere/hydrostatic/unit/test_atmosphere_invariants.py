"""Atmospheric invariant/regression tests for core physical behaviors.

Requested coverage:
1. Rest-state invariance for zero-wind, hydrostatically stratified columns.
2. Solid-body rotation (barotropic SW) with stable energy/enstrophy behavior.
3. Single-column radiation equilibrium and heating-sign checks.
4. Diffusion-only column monotonic decay with no new extrema (Thomas solver).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
    cdgrid_hydrostatic_tendencies as hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel, CDGridShallowWaterConfig, CDGridShallowWaterState,
)
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.test_cases.williamson import williamson_test2
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import implicit_vertical_diffusion
from legoesm.core.conservation import compute_conservation_diagnostics
from legoesm.core.field import Field
from legoesm.core.operators import curl_z
from legoesm.core.state import HydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate


def _make_stratified_rest_state(grid, sigma) -> HydrostaticState:
    """Horizontally uniform, vertically stratified state at rest."""
    shape_3d = (6, grid.n, grid.n, sigma.n_levels)
    shape_2d = (6, grid.n, grid.n)

    # Stable stratification: cool aloft, warm near surface.
    t_profile = jnp.linspace(210.0, 290.0, sigma.n_levels, dtype=jnp.float32)
    t_data = jnp.broadcast_to(t_profile[None, None, None, :], shape_3d)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d, dtype=jnp.float32), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d, dtype=jnp.float32), name="v", dims=dims_3d, units="m/s"),
        T=Field(data=t_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(
            data=jnp.full(shape_2d, 1.0e5, dtype=jnp.float32),
            name="p_s",
            dims=dims_2d,
            units="Pa",
        ),
        phis=Field(
            data=jnp.zeros(shape_2d, dtype=jnp.float32),
            name="phis",
            dims=dims_2d,
            units="m^2/s^2",
        ),
    )


def _sw_enstrophy(state, grid) -> jnp.ndarray:
    """Potential enstrophy-like diagnostic for shallow-water state."""
    zeta = curl_z(state.u, state.v, grid).data
    abs_vor = zeta + grid.f
    h = jnp.clip(state.h.data, 1.0, None)
    return jnp.sum(0.5 * abs_vor**2 / h * grid.area)


class TestRestStateInvariance:
    """Resting hydrostatic stratification should have near-zero dynamics."""

    def test_zero_wind_hydrostatic_stratification_tendencies_near_zero(self):
        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(12)
        state = _make_stratified_rest_state(grid, sigma)
        config = PrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            hyperdiff_ps_coeff=0.0,
            use_conservation_fixer=False,
        )

        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        cdgrid = create_cubed_sphere_cdgrid(grid)
        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        assert float(jnp.max(jnp.abs(tend.du_dt.data))) < 1e-7
        assert float(jnp.max(jnp.abs(tend.dv_dt.data))) < 1e-7
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) < 1e-7
        assert float(jnp.max(jnp.abs(tend.dp_s_dt.data))) < 1e-7


class TestSolidBodyRotationBehavior:
    """Solid-body flow should remain stable in energy/enstrophy statistics."""

    def test_williamson2_energy_enstrophy_stable(self):
        from legoesm.core.operators_cdgrid import center_to_dgrid_vector
        from legoesm.grids.halo import pad_halo_vector

        grid = create_cubed_sphere(12)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sw_state = williamson_test2(grid)

        # Convert A-grid state to CDGrid state
        h = sw_state.h.data
        u_a, v_a = sw_state.u.data, sw_state.v.data
        u_pad, v_pad = pad_halo_vector(
            u_a, v_a,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=grid.halo_interp_offsets,
        )
        u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                       u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
        v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                       v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
        state = CDGridShallowWaterState(
            h=h, u_d=u_d, v_d=v_d, h_s=sw_state.h_s.data,
        )

        model = CDGridShallowWaterModel(
            grid,
            CDGridShallowWaterConfig(
                hyperdiff_coeff=1e15,
                use_conservation_fixer=False,
                time_integrator="ssp45",
            ),
        )

        # Use CDGrid diagnostics for conservation
        area = grid.area
        from legoesm.core.operators_cdgrid import dgrid_to_cgrid
        def _total_energy(s):
            u_c, v_c = dgrid_to_cgrid(s.u_d, s.v_d, cdgrid)
            u_cell = 0.5 * (u_c[:, :-1] + u_c[:, 1:])
            v_cell = 0.5 * (v_c[:, :, :-1] + v_c[:, :, 1:])
            ke = 0.5 * (u_cell**2 + v_cell**2)
            g = constants.g
            pe = 0.5 * g * (s.h + s.h_s)**2
            return float(jnp.sum((ke * s.h + pe) * area))

        e0 = _total_energy(state)

        for _ in range(120):
            state = model.step(state, dt=120.0)

        ef = _total_energy(state)
        energy_rel_drift = (ef - e0) / e0

        assert jnp.isfinite(jnp.array(ef))
        assert abs(energy_rel_drift) < 5e-2, f"Energy drift too large: {energy_rel_drift:.3e}"
        assert float(jnp.min(state.h)) > 100.0


class TestSingleColumnRadiation:
    """Single-column checks for known equilibrium and heating sign."""

    def test_transparent_column_has_zero_radiative_heating(self):
        ncol, nlev = 2, 16
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1, dtype=jnp.float32)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        t = jnp.broadcast_to(
            jnp.linspace(210.0, 290.0, nlev, dtype=jnp.float32)[None, :],
            (ncol, nlev),
        )
        lat = jnp.zeros(ncol, dtype=jnp.float32)
        t_sfc = jnp.full(ncol, 290.0, dtype=jnp.float32)
        insol = jnp.full(ncol, 340.0, dtype=jnp.float32)

        # Fully transparent atmosphere: no LW/SW absorption in layers.
        config = GrayRadiationConfig(
            tau_equator=0.0,
            tau_pole=0.0,
            tau_moist_coeff=0.0,
            sw_tau_0=0.0,
            sfc_albedo=0.0,
        )
        out = gray_radiation(t, p_full, p_half, t_sfc, lat, None, insol, config)

        assert jnp.allclose(out.heating_rate, 0.0, atol=1e-12)
        assert jnp.allclose(out.lw_heating_rate, 0.0, atol=1e-12)
        assert jnp.allclose(out.sw_heating_rate, 0.0, atol=1e-12)

    def test_longwave_cools_warm_column_over_cool_surface(self):
        ncol, nlev = 1, 24
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1, dtype=jnp.float32)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

        # Atmosphere warmer than surface with no SW forcing: LW cooling expected.
        t = jnp.full((ncol, nlev), 300.0, dtype=jnp.float32)
        t_sfc = jnp.full(ncol, 280.0, dtype=jnp.float32)
        lat = jnp.zeros(ncol, dtype=jnp.float32)
        insol = jnp.zeros(ncol, dtype=jnp.float32)

        out = gray_radiation(t, p_full, p_half, t_sfc, lat, None, insol, GrayRadiationConfig())

        assert float(jnp.max(out.lw_heating_rate)) < 1e-8
        assert float(jnp.min(out.lw_heating_rate)) < -1e-8


class TestDiffusionOnlyColumn:
    """Diffusion-only column should smooth monotonically without new extrema."""

    def test_monotonic_decay_and_no_new_extrema(self):
        ncol, nlev = 1, 30
        phi0 = jnp.full((ncol, nlev), 300.0, dtype=jnp.float32)
        phi0 = phi0.at[:, 12:18].set(320.0)  # localized warm anomaly

        k_half = jnp.full((ncol, nlev - 1), 12.0, dtype=jnp.float32)
        rho = jnp.ones((ncol, nlev), dtype=jnp.float32)
        dz = jnp.full((ncol, nlev), 400.0, dtype=jnp.float32)
        dz_half = jnp.full((ncol, nlev - 1), 400.0, dtype=jnp.float32)
        surface_flux = jnp.zeros(ncol, dtype=jnp.float32)
        dt = 300.0

        phi_prev = phi0
        max_prev = float(jnp.max(phi_prev))
        min_prev = float(jnp.min(phi_prev))
        tv_prev = float(jnp.sum(jnp.abs(jnp.diff(phi_prev, axis=1))))

        for _ in range(8):
            phi_new = implicit_vertical_diffusion(
                phi_prev, k_half, rho, dz, dz_half, dt, surface_flux
            )

            max_new = float(jnp.max(phi_new))
            min_new = float(jnp.min(phi_new))
            tv_new = float(jnp.sum(jnp.abs(jnp.diff(phi_new, axis=1))))

            assert max_new <= max_prev + 5e-4
            assert min_new >= min_prev - 5e-4
            assert tv_new <= tv_prev + 1e-6

            phi_prev = phi_new
            max_prev = max_new
            min_prev = min_new
            tv_prev = tv_new

        # No new extrema relative to initial profile.
        assert float(jnp.max(phi_prev)) <= float(jnp.max(phi0)) + 5e-4
        assert float(jnp.min(phi_prev)) >= float(jnp.min(phi0)) - 5e-4
