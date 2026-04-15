"""Unit tests for atmospheric turbulence / boundary layer module.

Tests cover:
- Vertical diffusion: shape, conservation, smoothing, differentiability
- Surface layer: shape/sign, wind sensitivity, warm surface
- Smagorinsky: output shapes, mixing, differentiability
- Louis: stable vs unstable Ri, shapes, differentiability
- TKE: shear response, minimum TKE, shapes, differentiability
- Integration: hydrostatic/NH shapes, nonzero tendencies, jax.grad, scheme selection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.turbulence.config import (
    SurfaceLayerConfig,
    SmagorinskyConfig,
    LouisConfig,
    TKEConfig,
    CLUBBLiteConfig,
    HoltslagBovilleConfig,
    YSUConfig,
    EDMFConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    PBLHeightConfig,
    compute_bulk_richardson,
    diagnose_pbl_height,
    diagnose_pbl_height_interp,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.atmosphere.physics.turbulence.integration import (
    make_turbulence_physics,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_column_data(ncol=4, nlev=10):
    """Create test column data with a sheared wind profile.

    Returns u, v, T, q_v, p_full, p_half, z_full, z_half, rho.
    Levels ordered top (index 0) to bottom (index nlev-1).
    """
    # Pressure: linearly spaced interfaces from 100 Pa (top) to 1e5 Pa (surface)
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Temperature: warm at surface, cool aloft
    T = jnp.broadcast_to(
        jnp.linspace(220.0, 290.0, nlev)[None, :],
        (ncol, nlev),
    )

    # Heights: approximate from hydrostatic balance
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)

    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])

    # Density from ideal gas
    rho = p_full / (constants.R_d * T)

    # Wind: sheared profile (jet aloft)
    u = jnp.broadcast_to(
        jnp.linspace(20.0, 2.0, nlev)[None, :],
        (ncol, nlev),
    )
    v = jnp.broadcast_to(
        jnp.linspace(5.0, 1.0, nlev)[None, :],
        (ncol, nlev),
    )

    # Moisture
    q_sat = saturation_mixing_ratio(T, p_full)
    rh = jnp.linspace(0.1, 0.8, nlev)[None, :]
    q_v = rh * q_sat

    return u, v, T, q_v, p_full, p_half, z_full, z_half, rho


# ===========================================================================
# Vertical Diffusion tests
# ===========================================================================

class TestVerticalDiffusion:
    """Tests for the implicit Thomas solver."""

    def test_shape_preservation(self):
        """Output should have the same shape as input."""
        ncol, nlev = 4, 10
        phi = jnp.ones((ncol, nlev))
        K_half = jnp.full((ncol, nlev - 1), 10.0)
        rho = jnp.ones((ncol, nlev))
        dz = jnp.full((ncol, nlev), 1000.0)
        dz_half = jnp.full((ncol, nlev - 1), 1000.0)
        sflx = jnp.zeros(ncol)

        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
        assert result.shape == (ncol, nlev)

    def test_conserves_column_integral(self):
        """With zero surface flux, column integral should be conserved."""
        ncol, nlev = 3, 8
        key = jax.random.PRNGKey(42)
        phi = jax.random.normal(key, (ncol, nlev)) + 10.0
        K_half = jnp.full((ncol, nlev - 1), 5.0)
        rho = jnp.ones((ncol, nlev))
        dz = jnp.full((ncol, nlev), 500.0)
        dz_half = jnp.full((ncol, nlev - 1), 500.0)
        sflx = jnp.zeros(ncol)

        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 300.0, sflx)

        # Column integral: sum(phi * rho * dz) should be conserved
        integral_before = jnp.sum(phi * rho * dz, axis=1)
        integral_after = jnp.sum(result * rho * dz, axis=1)
        # Use rtol=1e-4 to accommodate float32 precision
        assert jnp.allclose(integral_before, integral_after, rtol=1e-4)

    def test_smooths_sharp_gradient(self):
        """Diffusion should smooth a step function."""
        ncol, nlev = 2, 20
        phi = jnp.zeros((ncol, nlev))
        phi = phi.at[:, nlev // 2:].set(10.0)  # step function

        K_half = jnp.full((ncol, nlev - 1), 50.0)
        rho = jnp.ones((ncol, nlev))
        dz = jnp.full((ncol, nlev), 200.0)
        dz_half = jnp.full((ncol, nlev - 1), 200.0)
        sflx = jnp.zeros(ncol)

        result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)

        # The max gradient should be reduced
        max_grad_before = float(jnp.max(jnp.abs(jnp.diff(phi, axis=1))))
        max_grad_after = float(jnp.max(jnp.abs(jnp.diff(result, axis=1))))
        assert max_grad_after < max_grad_before

    def test_differentiable(self):
        """jax.grad should work through the solver."""
        ncol, nlev = 2, 6

        def loss(phi):
            K_half = jnp.full((ncol, nlev - 1), 10.0)
            rho = jnp.ones((ncol, nlev))
            dz = jnp.full((ncol, nlev), 500.0)
            dz_half = jnp.full((ncol, nlev - 1), 500.0)
            sflx = jnp.zeros(ncol)
            result = implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, 60.0, sflx)
            return jnp.sum(result ** 2)

        phi = jax.random.normal(jax.random.PRNGKey(0), (ncol, nlev)) + 5.0
        g = jax.grad(loss)(phi)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == phi.shape


# ===========================================================================
# Surface Layer tests
# ===========================================================================

class TestSurfaceLayer:
    """Tests for bulk aerodynamic surface fluxes."""

    def test_shapes(self):
        """Surface flux outputs should have correct shapes."""
        ncol = 4
        u = jnp.full(ncol, 5.0)
        v = jnp.full(ncol, 2.0)
        T = jnp.full(ncol, 290.0)
        q_v = jnp.full(ncol, 0.01)
        T_sfc = jnp.full(ncol, 295.0)
        q_sfc = jnp.full(ncol, 0.015)
        rho = jnp.full(ncol, 1.2)
        config = SurfaceLayerConfig()

        tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho, config,
        )

        assert tau_x.shape == (ncol,)
        assert tau_y.shape == (ncol,)
        assert shflx.shape == (ncol,)
        assert lhflx.shape == (ncol,)
        assert ustar.shape == (ncol,)

    def test_sign_conventions(self):
        """Verify sign: tau_x < 0 for u > 0, shflx > 0 for warm surface."""
        ncol = 2
        u = jnp.full(ncol, 10.0)
        v = jnp.zeros(ncol)
        T = jnp.full(ncol, 285.0)
        q_v = jnp.full(ncol, 0.005)
        T_sfc = jnp.full(ncol, 295.0)  # warm surface
        q_sfc = jnp.full(ncol, 0.015)
        rho = jnp.full(ncol, 1.2)
        config = SurfaceLayerConfig()

        tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho, config,
        )

        assert jnp.all(tau_x < 0)  # drag opposes wind
        assert jnp.all(shflx > 0)  # warm surface → upward heat
        assert jnp.all(lhflx > 0)  # q_sfc > q_v → upward moisture
        assert jnp.all(ustar > 0)

    def test_stronger_wind_larger_fluxes(self):
        """Stronger wind should produce larger magnitude fluxes."""
        ncol = 1
        T = jnp.full(ncol, 285.0)
        q_v = jnp.full(ncol, 0.005)
        T_sfc = jnp.full(ncol, 295.0)
        q_sfc = jnp.full(ncol, 0.015)
        rho = jnp.full(ncol, 1.2)
        config = SurfaceLayerConfig()

        # Weak wind
        tau_x_weak, _, shflx_weak, _, _ = compute_surface_fluxes(
            jnp.full(ncol, 2.0), jnp.zeros(ncol),
            T, q_v, T_sfc, q_sfc, rho, config,
        )

        # Strong wind
        tau_x_strong, _, shflx_strong, _, _ = compute_surface_fluxes(
            jnp.full(ncol, 20.0), jnp.zeros(ncol),
            T, q_v, T_sfc, q_sfc, rho, config,
        )

        assert float(jnp.abs(tau_x_strong[0])) > float(jnp.abs(tau_x_weak[0]))
        assert float(shflx_strong[0]) > float(shflx_weak[0])


# ===========================================================================
# Smagorinsky tests
# ===========================================================================

class TestSmagorinsky:
    """Tests for constant-Km Smagorinsky turbulence."""

    def test_output_shapes(self):
        """Smagorinsky output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = SmagorinskyConfig()

        out = smagorinsky_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)
        assert out.lhflx.shape == (ncol,)
        assert out.ustar.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """Smagorinsky should produce nonzero tendencies for sheared profiles."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0  # warm surface
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = SmagorinskyConfig()

        out = smagorinsky_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through Smagorinsky turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = SmagorinskyConfig()

        def loss(T_in):
            out = smagorinsky_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape


# ===========================================================================
# Louis tests
# ===========================================================================

class TestLouis:
    """Tests for Louis (1979) stability-dependent turbulence."""

    def test_output_shapes(self):
        """Louis output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = LouisConfig()

        out = louis_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)

    def test_stable_reduces_Km(self):
        """Stable stratification should reduce Km compared to unstable."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        # Stable: inversion (T increases with height at bottom)
        T_stable = T.at[:, -1].set(T[:, -2] - 5.0)
        T_sfc = T_stable[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = LouisConfig()

        out_stable = louis_turbulence(
            u, v, T_stable, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        # Unstable: very warm surface
        T_unstable = T.at[:, -1].set(T[:, -2] + 20.0)
        T_sfc_u = T_unstable[:, -1]
        q_sfc_u = saturation_mixing_ratio(T_sfc_u, p_full[:, -1])

        out_unstable = louis_turbulence(
            u, v, T_unstable, q_v, p_full, p_half, z_full, z_half,
            T_sfc_u, q_sfc_u, rho, dt=300.0, config=config,
        )

        # Unstable should have larger Km near surface
        mean_Km_stable = float(jnp.mean(out_stable.Km[:, -2:]))
        mean_Km_unstable = float(jnp.mean(out_unstable.Km[:, -2:]))
        assert mean_Km_unstable > mean_Km_stable

    def test_differentiable(self):
        """jax.grad should work through Louis turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = LouisConfig()

        def loss(T_in):
            out = louis_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))


# ===========================================================================
# TKE tests
# ===========================================================================

class TestTKE:
    """Tests for prognostic TKE turbulence."""

    def test_output_shapes(self):
        """TKE output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TKEConfig()

        out, tke_new = tke_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert tke_new.shape == (ncol, nlev)

    def test_tke_increases_with_shear(self):
        """More shear should produce higher TKE."""
        ncol, nlev = 2, 10
        _, _, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TKEConfig()

        # Weak shear
        u_weak = jnp.full((ncol, nlev), 1.0)
        v_weak = jnp.zeros((ncol, nlev))

        _, tke_weak = tke_turbulence(
            u_weak, v_weak, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        # Strong shear
        u_strong = jnp.broadcast_to(
            jnp.linspace(30.0, 0.0, nlev)[None, :], (ncol, nlev),
        )
        v_strong = jnp.zeros((ncol, nlev))

        _, tke_strong = tke_turbulence(
            u_strong, v_strong, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.mean(tke_strong)) > float(jnp.mean(tke_weak))

    def test_tke_stays_above_minimum(self):
        """TKE should never fall below tke_min."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = TKEConfig(tke_min=1e-4)

        # Start with very small TKE
        tke = jnp.full((ncol, nlev), 1e-8)

        _, tke_new = tke_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(tke_new >= config.tke_min)

    def test_differentiable(self):
        """jax.grad should work through TKE turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1]
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = TKEConfig()

        def loss(T_in):
            out, _ = tke_turbulence(
                u, v, T_in, q_v, tke, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Tests for make_turbulence_physics integration bridge."""

    def test_hydrostatic_tendency_shapes(self):
        """Hydrostatic turbulence tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        n = grid.n
        nlev = sigma.n_levels
        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dv_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dp_s_dt.data.shape == (6, n, n)

    def test_hydrostatic_nonzero_momentum_tendency(self):
        """Hydrostatic turbulence should produce nonzero wind tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        # Give state some wind to mix
        n = grid.n
        nlev = sigma.n_levels
        u_data = jnp.ones((6, n, n, nlev)) * 10.0
        state = state._replace(
            u=Field(data=u_data, name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        # Should have nonzero du_dt (from surface drag at minimum)
        max_du = float(jnp.max(jnp.abs(tendencies.du_dt.data)))
        assert max_du > 0.0

    def test_hydrostatic_nonzero_heat_tendency(self):
        """Hydrostatic turbulence should produce nonzero T tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        max_dT = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_dT > 0.0

    def test_nonhydrostatic_tendency_shapes(self):
        """Non-hydrostatic turbulence tendencies should have correct shapes."""
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
            u=Field(data=jnp.ones((6, n, n, nlev)) * 5.0, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.ones((6, n, n, nlev)) * 2.0, name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 1)), name="tracers", dims=dims_tr, units="kg/kg"),
        )

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="nonhydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, height_coord, terrain_metric)

        assert tendencies.dtheta_prime_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dw_dt.data.shape == (6, n, n, nlev + 1)
        assert tendencies.dtracers_dt.data.shape == (6, n, n, nlev, 1)

    def test_grad_through_hydrostatic_turbulence(self):
        """jax.grad should work through hydrostatic turbulence physics."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="smagorinsky")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tendencies, _ = physics_fn(new_state, grid, sigma)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))

    def test_scheme_selection(self):
        """Different schemes should produce different tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_init
        from legoesm.core.field import Field

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        # Give state some wind
        n = grid.n
        nlev = sigma.n_levels
        u_data = jnp.ones((6, n, n, nlev)) * 10.0
        state = state._replace(
            u=Field(data=u_data, name="u", dims=("face", "x", "y", "level"), units="m/s"),
        )

        fn_smag = make_turbulence_physics(
            TurbulenceConfig(scheme="smagorinsky"), model_type="hydrostatic", dt=300.0,
        )
        fn_louis = make_turbulence_physics(
            TurbulenceConfig(scheme="louis"), model_type="hydrostatic", dt=300.0,
        )

        tend_smag, _ = fn_smag(state, grid, sigma)
        tend_louis, _ = fn_louis(state, grid, sigma)

        # Both nonzero but different
        assert float(jnp.max(jnp.abs(tend_smag.du_dt.data))) > 0
        assert float(jnp.max(jnp.abs(tend_louis.du_dt.data))) > 0
        assert not jnp.allclose(tend_smag.du_dt.data, tend_louis.du_dt.data, atol=1e-10)

    def test_all_scheme_strings_accepted(self):
        """All supported scheme strings plus 'none' should be accepted."""
        schemes = [
            "smagorinsky", "louis", "tke", "clubb_lite",
            "holtslag_boville", "ysu", "edmf", "none",
        ]
        for scheme in schemes:
            config = TurbulenceConfig(scheme=scheme)
            fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
            assert callable(fn), f"Factory should return callable for scheme={scheme!r}"

    def test_none_scheme_gives_zeros(self):
        """scheme='none' should produce zero tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = TurbulenceConfig(scheme="none")
        physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tendencies.dT_dt.data, 0.0)
        assert jnp.allclose(tendencies.du_dt.data, 0.0)
        assert jnp.allclose(tendencies.dv_dt.data, 0.0)

# ===========================================================================
# Holtslag-Boville tests
# ===========================================================================

class TestHoltslagBoville:
    """Tests for Holtslag-Boville nonlocal K-profile turbulence."""

    def test_output_shapes(self):
        """HB output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)
        assert out.lhflx.shape == (ncol,)
        assert out.ustar.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """HB should produce nonzero tendencies for sheared profiles."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through HB turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        def loss(T_in):
            out = holtslag_boville_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_finite_outputs(self):
        """All outputs should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = HoltslagBovilleConfig()

        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.Km))
        assert jnp.all(jnp.isfinite(out.Kh))

    def test_counter_gradient_effect(self):
        """Warm surface should activate counter-gradient, changing T tendency."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 10.0  # very warm surface
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        # With counter-gradient
        config_cg = HoltslagBovilleConfig(gamma_h=10.0)
        out_cg = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_cg,
        )

        # Without counter-gradient
        config_no_cg = HoltslagBovilleConfig(gamma_h=0.0)
        out_no_cg = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_no_cg,
        )

        # T tendencies should differ when counter-gradient is active
        assert not jnp.allclose(out_cg.dT_dt, out_no_cg.dT_dt, atol=1e-10)


# ===========================================================================
# YSU tests
# ===========================================================================

class TestYSU:
    """Tests for YSU PBL turbulence scheme."""

    def test_output_shapes(self):
        """YSU output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """YSU should produce nonzero tendencies for sheared profiles."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through YSU turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        def loss(T_in):
            out = ysu_turbulence(
                u, v, T_in, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_finite_outputs(self):
        """All outputs should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        config = YSUConfig()

        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.Km))

    def test_entrainment_near_pbl_top(self):
        """YSU with entrainment should differ from zero-entrainment."""
        ncol, nlev = 2, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 15.0  # very warm surface for strong convective BL
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])

        # With strong entrainment
        config_ent = YSUConfig(entrainment_coeff=1.0)
        out_ent = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_ent,
        )

        # Without entrainment
        config_no_ent = YSUConfig(entrainment_coeff=0.0)
        out_no_ent = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_no_ent,
        )

        # Tendencies should differ when entrainment is active
        assert not jnp.allclose(out_ent.dT_dt, out_no_ent.dT_dt, atol=1e-12)


# ===========================================================================
# EDMF tests
# ===========================================================================

class TestEDMF:
    """Tests for EDMF eddy-diffusivity mass-flux turbulence."""

    def test_output_shapes(self):
        """EDMF output should have correct shapes."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = EDMFConfig()

        out, tke_new = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert out.du_dt.shape == (ncol, nlev)
        assert out.dv_dt.shape == (ncol, nlev)
        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.Km.shape == (ncol, nlev)
        assert out.Kh.shape == (ncol, nlev)
        assert out.shflx.shape == (ncol,)

    def test_nonzero_tendencies(self):
        """EDMF should produce nonzero tendencies."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = EDMFConfig()

        out, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert float(jnp.max(jnp.abs(out.du_dt))) > 1e-10
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through EDMF turbulence."""
        ncol, nlev = 2, 8
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = EDMFConfig()

        def loss(T_in):
            out, _ = edmf_turbulence(
                u, v, T_in, q_v, tke, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_finite_outputs(self):
        """All outputs should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = EDMFConfig()

        out, tke_new = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert jnp.all(jnp.isfinite(out.du_dt))
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.Km))
        assert jnp.all(jnp.isfinite(tke_new))

    def test_returns_tke(self):
        """EDMF should return a TKE array with correct shape."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)
        config = EDMFConfig()

        out, tke_new = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config,
        )

        assert isinstance(tke_new, jax.Array)
        assert tke_new.shape == (ncol, nlev)
        assert jnp.all(tke_new >= config.tke_min)

    def test_mass_flux_active(self):
        """Nonzero MF contribution with warm (unstable) surface."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 15.0  # very warm surface for strong updrafts
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)

        # With mass flux
        config_mf = EDMFConfig(a_updraft=0.1)
        out_mf, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_mf,
        )

        # Without mass flux (zero updraft area)
        config_no_mf = EDMFConfig(a_updraft=0.0)
        out_no_mf, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=config_no_mf,
        )

        # T tendencies should differ when MF is active
        assert not jnp.allclose(out_mf.dT_dt, out_no_mf.dT_dt, atol=1e-10)




# ===========================================================================
# PBL Height Diagnosis tests (Task 9)
# ===========================================================================

class TestPBLHeight:
    """Tests for PBL height diagnosis via bulk Richardson method."""

    def test_bulk_richardson_shape(self):
        """Bulk Ri should have shape (ncol, nlev)."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        Ri_bulk, theta_v = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
        assert Ri_bulk.shape == (ncol, nlev)
        assert theta_v.shape == (ncol, nlev)

    def test_bulk_richardson_surface_zero(self):
        """Ri at the surface level should be near zero (no buoyancy difference)."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        Ri_bulk, _ = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
        # Bottom level (surface): theta_v(sfc) - theta_v(sfc) = 0 so Ri ~ 0
        assert float(jnp.max(jnp.abs(Ri_bulk[:, -1]))) < 0.1

    def test_bulk_richardson_finite(self):
        """Ri should be finite everywhere."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        Ri_bulk, theta_v = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
        assert jnp.all(jnp.isfinite(Ri_bulk))
        assert jnp.all(jnp.isfinite(theta_v))

    def test_diagnose_pbl_height_shape(self):
        """h_pbl should have shape (ncol,)."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
        assert h_pbl.shape == (ncol,)

    def test_diagnose_pbl_height_bounds(self):
        """h_pbl should be within [h_min, h_max]."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        config = PBLHeightConfig(h_min=100.0, h_max=5000.0)

        h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full, config)
        assert jnp.all(h_pbl >= config.h_min)
        assert jnp.all(h_pbl <= config.h_max)

    def test_diagnose_pbl_height_finite(self):
        """h_pbl should be finite."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
        assert jnp.all(jnp.isfinite(h_pbl))

    def test_interp_method_shape_and_bounds(self):
        """Interp method should return correct shape within bounds."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        config = PBLHeightConfig(h_min=100.0, h_max=5000.0)

        h_pbl = diagnose_pbl_height_interp(T, q_v, u, v, p_full, z_full, config)
        assert h_pbl.shape == (ncol,)
        assert jnp.all(h_pbl >= config.h_min)
        assert jnp.all(h_pbl <= config.h_max)
        assert jnp.all(jnp.isfinite(h_pbl))

    def test_strong_shear_deeper_pbl(self):
        """Stronger wind shear should produce a deeper PBL (more Ri < Ri_crit)."""
        ncol, nlev = 4, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        # Weak wind -> shallow PBL
        u_weak = u * 0.2
        v_weak = v * 0.2
        h_weak = diagnose_pbl_height(T, q_v, u_weak, v_weak, p_full, z_full)

        # Strong wind -> deeper PBL
        u_strong = u * 3.0
        v_strong = v * 3.0
        h_strong = diagnose_pbl_height(T, q_v, u_strong, v_strong, p_full, z_full)

        # Stronger shear means larger V^2, so Ri is smaller -> more levels
        # where Ri < Ri_crit -> deeper PBL
        assert float(jnp.mean(h_strong)) > float(jnp.mean(h_weak))

    def test_barotropic_wind_does_not_deepen_pbl(self):
        """Adding a uniform wind offset (no extra shear) should not change PBL height.

        The bulk Ri denominator must use wind shear from surface, not absolute
        wind speed.  A barotropic wind adds the same vector to every level,
        so the shear is unchanged and the PBL height should be identical.
        """
        ncol, nlev = 4, 20
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        h_base = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
        # Add a large uniform barotropic wind (no shear added)
        h_baro = diagnose_pbl_height(T, q_v, u + 50.0, v + 30.0, p_full, z_full)

        # PBL height should be essentially unchanged
        assert jnp.allclose(h_base, h_baro, rtol=1e-4), (
            f"Barotropic wind changed PBL: {float(jnp.mean(h_base)):.1f} -> "
            f"{float(jnp.mean(h_baro)):.1f}"
        )

    def test_differentiable(self):
        """jax.grad should work through PBL height diagnosis."""
        ncol, nlev = 2, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)

        def loss(T_in):
            h = diagnose_pbl_height(T_in, q_v, u, v, p_full, z_full)
            return jnp.sum(h ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_backends_return_h_pbl(self):
        """All turbulence backends should return h_pbl in TurbulenceOutput."""
        ncol, nlev = 4, 10
        u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _make_column_data(ncol, nlev)
        T_sfc = T[:, -1] + 5.0
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        tke = jnp.full((ncol, nlev), 0.1)

        # Smagorinsky
        out = smagorinsky_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=SmagorinskyConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # Louis
        out = louis_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=LouisConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # YSU
        out = ysu_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=YSUConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # Holtslag-Boville
        out = holtslag_boville_turbulence(
            u, v, T, q_v, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=HoltslagBovilleConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # TKE
        out, _ = tke_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=TKEConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))

        # EDMF
        out, _ = edmf_turbulence(
            u, v, T, q_v, tke, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt=300.0, config=EDMFConfig(),
        )
        assert out.h_pbl.shape == (ncol,)
        assert jnp.all(jnp.isfinite(out.h_pbl))
