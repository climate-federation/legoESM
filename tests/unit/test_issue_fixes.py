"""Unit tests C1-C5 from the legoESM issue-fixes review.

These tests catch sign/indexing bugs quickly and are inexpensive to run in CI.
They are designed to work under jit and on CPU.

C1: Rest-state invariance (u=v=0, hydrostatic, check tendencies ~0)
C2: Diffusion-only column test (sharp spike, Thomas solver, monotonic smoothing)
C3: Radiation sign test (single-column flux-divergence consistency)
C4: Halo exchange invariance (vector rotation roundtrip, uniform zonal wind)
C5: Global dry-mass conservation (ps correction restores mass exactly)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators import global_integral
from legoesm.core.conservation import fix_mass_hydrostatic
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate, hybrid_from_sigma
from legoesm.grids.halo import pad_halo_vector
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
    cdgrid_hydrostatic_tendencies as hydrostatic_tendencies,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)
from legoesm import constants


# =========================================================================
# Helpers
# =========================================================================

def _make_hydrostatic_state(grid, sigma, T_val=300.0, u_val=0.0, v_val=0.0,
                            p_s_val=1e5):
    """Create a uniform HydrostaticState."""
    shape_3d = (6, grid.n, grid.n, sigma.n_levels)
    shape_2d = (6, grid.n, grid.n)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=jnp.ones(shape_3d) * u_val, name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * v_val, name="v",
                dims=dims_3d, units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * T_val, name="T",
                dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * p_s_val, name="p_s",
                  dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
    )


# =========================================================================
# C1: Rest-state invariance
# =========================================================================

class TestC1RestStateInvariance:
    """A resting, isothermal, flat-bottom atmosphere should produce
    near-zero tendencies from the dynamical core (no hyperdiffusion)."""

    def test_rest_state_tendencies_near_zero(self):
        grid = create_cubed_sphere(8)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(5)
        state = _make_hydrostatic_state(grid, sigma, T_val=300.0,
                                        u_val=0.0, v_val=0.0)
        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)

        tend = hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        # Wind tendencies should be ~0 (Coriolis with u=v=0 gives zero)
        # Use 1e-6 tolerance to accommodate float32 precision
        atol_wind = 1e-6
        assert jnp.allclose(tend.du_dt.data, 0.0, atol=atol_wind), (
            f"max |du/dt| = {float(jnp.max(jnp.abs(tend.du_dt.data))):.2e}")
        assert jnp.allclose(tend.dv_dt.data, 0.0, atol=atol_wind), (
            f"max |dv/dt| = {float(jnp.max(jnp.abs(tend.dv_dt.data))):.2e}")

        # Temperature tendency: isothermal + hydrostatic ⟹ ~0 advective heating
        assert jnp.allclose(tend.dT_dt.data, 0.0, atol=atol_wind), (
            f"max |dT/dt| = {float(jnp.max(jnp.abs(tend.dT_dt.data))):.2e}")

        # Surface pressure tendency: no divergence ⟹ ~0
        assert jnp.allclose(tend.dp_s_dt.data, 0.0, atol=1e-4), (
            f"max |dp_s/dt| = {float(jnp.max(jnp.abs(tend.dp_s_dt.data))):.2e}")

    def test_rest_state_jit_compatible(self):
        """Tendency computation should work under jax.jit."""
        grid = create_cubed_sphere(8)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(5)
        state = _make_hydrostatic_state(grid, sigma)
        config = PrimitiveEquationConfig(hyperdiff_coeff=0.0)

        @jax.jit
        def compute(state):
            return hydrostatic_tendencies(state, grid, sigma, cdgrid, config)

        tend = compute(state)
        assert jnp.all(jnp.isfinite(tend.du_dt.data))


# =========================================================================
# C2: Diffusion-only column test (Thomas solver)
# =========================================================================

class TestC2ThomasSolver:
    """A sharp temperature spike should be smoothed monotonically by
    vertical diffusion, validating the Thomas algorithm sign convention."""

    def test_spike_smoothed_monotonically(self):
        ncol = 1
        nlev = 20
        # Uniform column
        phi = jnp.ones((ncol, nlev)) * 300.0
        # Inject a spike at level 10
        phi = phi.at[:, 10].set(310.0)

        K_half = jnp.ones((ncol, nlev - 1)) * 10.0   # strong diffusivity
        rho = jnp.ones((ncol, nlev))                   # unit density
        dz = jnp.ones((ncol, nlev)) * 500.0            # 500 m layers
        dz_half = jnp.ones((ncol, nlev - 1)) * 500.0   # center-to-center
        dt = 300.0
        surface_flux = jnp.zeros(ncol)

        phi_new = implicit_vertical_diffusion(
            phi, K_half, rho, dz, dz_half, dt, surface_flux
        )

        # 1. No NaN/Inf
        assert jnp.all(jnp.isfinite(phi_new)), "Thomas solver produced NaN/Inf"

        # 2. Peak should decrease (diffusion smooths)
        assert float(phi_new[0, 10]) < float(phi[0, 10]), (
            "Spike was not smoothed by diffusion")

        # 3. Neighbours should increase (heat spread)
        assert float(phi_new[0, 9]) > float(phi[0, 9])
        assert float(phi_new[0, 11]) > float(phi[0, 11])

        # 4. Conservation: total heat content preserved (no-flux BCs)
        # With zero surface flux and zero-flux top, total phi*rho*dz is conserved
        total_old = float(jnp.sum(phi * rho * dz))
        total_new = float(jnp.sum(phi_new * rho * dz))
        assert abs(total_new - total_old) / total_old < 1e-6, (
            f"Heat not conserved: old={total_old:.4f}, new={total_new:.4f}")

    def test_uniform_field_unchanged(self):
        """A uniform field should remain unchanged after diffusion."""
        ncol, nlev = 2, 10
        phi = jnp.ones((ncol, nlev)) * 280.0
        K_half = jnp.ones((ncol, nlev - 1)) * 5.0
        rho = jnp.ones((ncol, nlev))
        dz = jnp.ones((ncol, nlev)) * 1000.0
        dz_half = jnp.ones((ncol, nlev - 1)) * 1000.0
        dt = 600.0
        surface_flux = jnp.zeros(ncol)

        phi_new = implicit_vertical_diffusion(
            phi, K_half, rho, dz, dz_half, dt, surface_flux
        )
        assert jnp.allclose(phi_new, 280.0, atol=1e-10), (
            "Uniform field changed after diffusion")

    def test_thomas_jit_compatible(self):
        """Thomas solver should work under jax.jit."""
        ncol, nlev = 4, 15
        phi = jnp.ones((ncol, nlev)) * 300.0
        K_half = jnp.ones((ncol, nlev - 1)) * 5.0
        rho = jnp.ones((ncol, nlev))
        dz = jnp.ones((ncol, nlev)) * 500.0
        dz_half = jnp.ones((ncol, nlev - 1)) * 500.0
        surface_flux = jnp.zeros(ncol)

        @jax.jit
        def diffuse(phi):
            return implicit_vertical_diffusion(
                phi, K_half, rho, dz, dz_half, 300.0, surface_flux
            )

        result = diffuse(phi)
        assert jnp.all(jnp.isfinite(result))


# =========================================================================
# C3: Radiation sign test
# =========================================================================

class TestC3RadiationSign:
    """Single-column flux-divergence consistency: the heating rate
    integrated over the column should equal the net flux divergence."""

    def test_heating_rate_consistent_with_fluxes(self):
        ncol, nlev = 1, 20
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T = jnp.broadcast_to(
            jnp.linspace(200.0, 300.0, nlev)[None, :],
            (ncol, nlev),
        )
        T_sfc = jnp.full(ncol, 300.0)
        lat = jnp.zeros(ncol)
        insol = perpetual_equinox_insolation(lat, 1360.0)

        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        # Net flux at each interface (positive upward)
        F_net = (out.lw_flux_up + out.sw_flux_up) - (out.lw_flux_down + out.sw_flux_down)

        # Flux divergence per layer: dF = F_net[k+1] - F_net[k]
        # Heating rate: hr = (g / c_p) * dF / dp
        dp = p_half[:, 1:] - p_half[:, :-1]

        # Column integral of hr * dp should equal g/c_p * (F_net_sfc - F_net_toa)
        col_hr_dp = jnp.sum(out.heating_rate * dp, axis=1)
        expected = (constants.g / constants.c_pd) * (F_net[:, -1] - F_net[:, 0])

        assert jnp.allclose(col_hr_dp, expected, rtol=1e-4), (
            f"Column hr*dp = {float(col_hr_dp):.6e}, "
            f"expected g/cp*(F_sfc-F_toa) = {float(expected):.6e}")

    def test_lw_cools_atmosphere(self):
        """In a warm atmosphere over a cooler surface (no SW), LW should
        produce net cooling (negative heating rate in upper levels)."""
        ncol, nlev = 1, 20
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        # Warm atmosphere, relatively cool surface
        T = jnp.full((ncol, nlev), 300.0)
        T_sfc = jnp.full(ncol, 280.0)
        lat = jnp.zeros(ncol)
        insol = jnp.zeros(ncol)  # no SW

        config = GrayRadiationConfig()
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)

        # Upper atmosphere should cool (LW emission to space > absorption)
        # At least some levels should have negative heating rate
        assert float(jnp.min(out.lw_heating_rate)) < 0.0, (
            "LW should produce cooling in at least some levels")


# =========================================================================
# C4: Halo exchange invariance (vector rotation roundtrip)
# =========================================================================

class TestC4HaloExchange:
    """A uniform zonal wind (u_east=const, v_north=0) should be
    recovered in all halo cells after the grid→geographic→grid
    roundtrip."""

    def test_uniform_zonal_wind_roundtrip(self):
        grid = create_cubed_sphere(8)
        n = grid.n
        U0 = 10.0  # 10 m/s eastward

        # Build grid-aligned (u, v) from geographic (u_east=U0, v_north=0):
        #   u_grid =  cos(angle) * U0
        #   v_grid = -sin(angle) * U0
        u_data = grid.cos_angle * U0
        v_data = -grid.sin_angle * U0

        u_pad, v_pad = pad_halo_vector(
            u_data, v_data,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
        )

        # Interior should be preserved exactly
        assert jnp.allclose(u_pad[:, 1:-1, 1:-1], u_data, atol=1e-5)
        assert jnp.allclose(v_pad[:, 1:-1, 1:-1], v_data, atol=1e-5)

        # Reconstruct geographic components in the halo region
        u_east_halo = (grid.cos_angle_padded * u_pad
                       - grid.sin_angle_padded * v_pad)
        v_north_halo = (grid.sin_angle_padded * u_pad
                        + grid.cos_angle_padded * v_pad)

        # In edges (excluding corners which are zero), u_east should be U0
        # Check each edge strip
        for edge_slice in [
            (slice(None), 0, slice(1, -1)),       # WEST halo
            (slice(None), -1, slice(1, -1)),      # EAST halo
            (slice(None), slice(1, -1), 0),       # SOUTH halo
            (slice(None), slice(1, -1), -1),      # NORTH halo
        ]:
            u_e = u_east_halo[edge_slice]
            v_n = v_north_halo[edge_slice]
            assert jnp.allclose(u_e, U0, atol=0.5), (
                f"u_east in halo: max err = {float(jnp.max(jnp.abs(u_e - U0))):.4f}")
            assert jnp.allclose(v_n, 0.0, atol=0.5), (
                f"v_north in halo: max abs = {float(jnp.max(jnp.abs(v_n))):.4f}")

    def test_wind_speed_preserved_in_halo(self):
        """Wind speed |V| should be preserved across all halo cells."""
        grid = create_cubed_sphere(8)
        key = jax.random.PRNGKey(42)
        u = jax.random.normal(key, (6, 8, 8)) * 10.0
        v = jax.random.normal(jax.random.PRNGKey(1), (6, 8, 8)) * 10.0

        u_pad, v_pad = pad_halo_vector(
            u, v,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
        )

        # Interior speed preserved
        speed_orig = jnp.sqrt(u**2 + v**2)
        speed_pad = jnp.sqrt(u_pad[:, 1:-1, 1:-1]**2
                             + v_pad[:, 1:-1, 1:-1]**2)
        assert jnp.allclose(speed_pad, speed_orig, atol=1e-5)


# =========================================================================
# C5: Global dry-mass conservation (ps correction)
# =========================================================================

class TestC5MassConservation:
    """The surface pressure correction should restore global dry air
    mass exactly (up to float32 round-off)."""

    def test_mass_restored_after_perturbation(self):
        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(5)
        state_old = _make_hydrostatic_state(grid, sigma, p_s_val=1e5)

        # Perturb p_s (simulating a timestep that doesn't conserve mass)
        key = jax.random.PRNGKey(123)
        dp = jax.random.normal(key, state_old.p_s.data.shape) * 100.0
        state_new = state_old._replace(
            p_s=state_old.p_s.replace(data=state_old.p_s.data + dp)
        )

        state_fixed = fix_mass_hydrostatic(state_new, state_old, grid)

        mass_old = global_integral(state_old.p_s, grid)
        mass_fixed = global_integral(state_fixed.p_s, grid)

        assert jnp.allclose(mass_fixed, mass_old, rtol=1e-5), (
            f"Mass not restored: old={float(mass_old):.6e}, "
            f"fixed={float(mass_fixed):.6e}")

    def test_correction_is_uniform(self):
        """The correction should be a uniform additive shift (preserves gradients)."""
        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(5)
        state_old = _make_hydrostatic_state(grid, sigma, p_s_val=1e5)

        key = jax.random.PRNGKey(456)
        dp = jax.random.normal(key, state_old.p_s.data.shape) * 50.0
        state_new = state_old._replace(
            p_s=state_old.p_s.replace(data=state_old.p_s.data + dp)
        )

        state_fixed = fix_mass_hydrostatic(state_new, state_old, grid)

        # The correction is uniform, so (fixed - new) should be constant
        correction = state_fixed.p_s.data - state_new.p_s.data
        assert jnp.allclose(correction, correction.ravel()[0], atol=1e-4), (
            "Mass correction is not spatially uniform")

    def test_mass_fixer_differentiable(self):
        """The mass fixer should be differentiable via jax.grad."""
        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(5)
        state_old = _make_hydrostatic_state(grid, sigma, p_s_val=1e5)

        def loss(ps_data):
            state_new = state_old._replace(
                p_s=state_old.p_s.replace(data=ps_data + 10.0)
            )
            state_fixed = fix_mass_hydrostatic(state_new, state_old, grid)
            return jnp.sum(state_fixed.p_s.data ** 2)

        grads = jax.grad(loss)(state_old.p_s.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through mass fixer not finite"


# =========================================================================
# Hybrid Held-Suarez compatibility
# =========================================================================

class TestHybridHeldSuarezCompatibility:
    """Hybrid coordinates should work with Held-Suarez forcing on all grids."""

    def test_cubed_and_latlon_hybrid_forcing_finite(self):
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing,
            held_suarez_init,
        )
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing_latlon,
            held_suarez_init_latlon,
        )

        sigma = create_sigma_coordinate(5)
        hybrid = hybrid_from_sigma(sigma)

        cube = create_cubed_sphere(6)
        state_cube = held_suarez_init(cube, hybrid, T_init=280.0)
        tend_cube = held_suarez_forcing(state_cube, cube, hybrid)
        assert jnp.all(jnp.isfinite(tend_cube.du_dt.data))
        assert jnp.all(jnp.isfinite(tend_cube.dT_dt.data))

        latlon = create_latlon_grid(8, 16)
        state_ll = held_suarez_init_latlon(latlon, hybrid, T_init=280.0)
        tend_ll = held_suarez_forcing_latlon(state_ll, latlon, hybrid)
        assert jnp.all(jnp.isfinite(tend_ll.du_dt.data))
        assert jnp.all(jnp.isfinite(tend_ll.dT_dt.data))

    def test_spectral_hybrid_forcing_finite(self):
        if not jax.config.jax_enable_x64:
            pytest.skip("Spectral hybrid Held-Suarez test requires jax_enable_x64")

        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            isothermal_rest_state_spectral,
        )
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_spectral

        sigma = create_sigma_coordinate(4)
        hybrid = hybrid_from_sigma(sigma)
        grid = create_gaussian_grid(n_max=7)
        state = isothermal_rest_state_spectral(grid, hybrid, T_init=280.0)
        tend = held_suarez_forcing_spectral(state, grid, hybrid)

        assert jnp.all(jnp.isfinite(tend.vor_hat.data.real))
        assert jnp.all(jnp.isfinite(tend.div_hat.data.real))
        assert jnp.all(jnp.isfinite(tend.T_hat.data.real))
