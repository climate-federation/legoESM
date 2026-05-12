"""Differentiability tests for atmosphere dynamical cores.

Tests that jax.grad produces finite, non-zero, spatially structured gradients
through single and multi-step integration of every dynamical core.

Categories:
  1a) Shallow water — all grids (CD-grid, lat-lon, spectral, MPAS)
  1b) Hydrostatic PE — cubed-sphere and lat-lon
  1c) Nonhydrostatic compressible Euler — cubed-sphere
  1d) Spectral PE
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.field import Field

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    """Check that a gradient array is finite, non-trivially non-zero, and has
    spatial structure (not all identical)."""
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )
    assert not jnp.all(grad_array == grad_array.ravel()[0]), (
        f"{name}: gradient is spatially uniform (no structure)"
    )


# ============================================================================
# 1a  Shallow Water — CD-grid (cubed-sphere C4)
# ============================================================================

class TestCDGridShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel, CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )
        n = 4
        base = create_cubed_sphere(n)
        config = CDGridShallowWaterConfig()
        self.model = CDGridShallowWaterModel(base, config)
        # Initial state: rest + Gaussian perturbation in h
        h0 = 1000.0 * jnp.ones((6, n, n))
        key = jax.random.PRNGKey(0)
        h0 = h0 + 10.0 * jax.random.normal(key, (6, n, n))
        u_d = jnp.zeros((6, n + 1, n + 1))
        v_d = jnp.zeros((6, n + 1, n + 1))
        h_s = jnp.zeros((6, n, n))
        self.state = CDGridShallowWaterState(h=h0, u_d=u_d, v_d=v_d, h_s=h_s)
        self.dt = 60.0
        self.n = n

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_init):
            s = state._replace(h=h_init)
            out = model.step(s, dt)
            return jnp.sum(out.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, "CDGrid SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_init):
            s = state._replace(h=h_init)
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(s_final.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, "CDGrid SW 5 steps")


# ============================================================================
# 1a  Shallow Water — spectral (T5)
# ============================================================================

class TestSpectralShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.spectral_sw import (
            SpectralShallowWaterModel, SpectralSWConfig, SpectralSWState,
        )

        grid = create_gaussian_grid(5, allow_unsupported_backend=True)
        config = SpectralSWConfig()
        self.model = SpectralShallowWaterModel(grid, config, allow_unsupported_backend=True)
        self.dt = 120.0

        # Build a spectral state: phi_hat = g * mean_depth (constant field)
        n_sh = grid.n_sh
        g = constants.g
        H0 = 5960.0
        phi_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
        phi_hat = phi_hat.at[0].set(g * H0 * jnp.sqrt(4 * jnp.pi))  # n=0,m=0
        key = jax.random.PRNGKey(2)
        phi_hat = phi_hat + 1.0 * jax.random.normal(key, (n_sh,))
        self.state = SpectralSWState(
            vor_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="vor_hat"),
            div_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="div_hat"),
            phi_hat=Field(phi_hat, name="phi_hat"),
            phis_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="phis_hat"),
        )
        self.n_sh = n_sh

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(phi_real):
            phi_hat_new = state.phi_hat.data.real * 0 + phi_real + 1j * state.phi_hat.data.imag
            s = state._replace(phi_hat=state.phi_hat.replace(data=phi_hat_new))
            out = model.step(s, dt)
            return jnp.sum(jnp.abs(out.phi_hat.data) ** 2)

        grad = jax.grad(loss)(state.phi_hat.data.real)
        assert_gradient_ok(grad, "Spectral SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(phi_real):
            phi_hat_new = phi_real + 1j * state.phi_hat.data.imag
            s = state._replace(phi_hat=state.phi_hat.replace(data=phi_hat_new))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(jnp.abs(s_final.phi_hat.data) ** 2)

        grad = jax.grad(loss)(state.phi_hat.data.real)
        assert_gradient_ok(grad, "Spectral SW 5 steps")


# ============================================================================
# 1a  Shallow Water — MPAS (level-2 Voronoi)
# ============================================================================

class TestMPASShallowWater:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.shallow_water_mpas import (
            MPASShallowWaterModel, MPASShallowWaterConfig,
        )
        from legoesm.core.state import MPASShallowWaterState

        mesh = create_voronoi_mesh(2)
        config = MPASShallowWaterConfig()
        self.model = MPASShallowWaterModel(mesh, config)
        self.dt = 120.0

        key = jax.random.PRNGKey(3)
        h_data = 1000.0 * jnp.ones(mesh.nCells) + 10.0 * jax.random.normal(key, (mesh.nCells,))
        self.state = MPASShallowWaterState(
            h=Field(h_data, name="h"),
            u=Field(jnp.zeros(mesh.nEdges), name="u"),
            h_s=Field(jnp.zeros(mesh.nCells), name="h_s"),
        )

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            out = model.step(s, dt)
            return jnp.sum(out.h.data ** 2)

        grad = jax.grad(loss)(state.h.data)
        assert_gradient_ok(grad, "MPAS SW single step")

    def test_grad_5_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=5)
            return jnp.sum(s_final.h.data ** 2)

        grad = jax.grad(loss)(state.h.data)
        assert_gradient_ok(grad, "MPAS SW 5 steps")


# ============================================================================
# 1b  Hydrostatic PE — CD-grid (C4, 5 levels)
# ============================================================================

class TestCDGridPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
        )
        from legoesm.core.state import FV3HydrostaticState

        n = 4
        nlev = 5
        base = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)
        self.model = CDGridPrimitiveEquationModel(base, sigma)
        self.dt = 60.0

        # Isothermal at-rest state
        T_data = 250.0 * jnp.ones((6, n, n, nlev))
        key = jax.random.PRNGKey(10)
        T_data = T_data + 1.0 * jax.random.normal(key, T_data.shape)
        self.state = FV3HydrostaticState(
            u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
            v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
        )
        self.n = n
        self.nlev = nlev

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "CDGrid PE single step")

    def test_grad_3_steps(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "CDGrid PE 3 steps")


# ============================================================================
# 1c  Nonhydrostatic Compressible Euler — CD-grid (C4, 5 levels)
# ============================================================================

class TestCompressibleEuler:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric
        from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
        )
        from legoesm.core.state import NonHydrostaticState

        n = 4
        nlev = 5
        H = 30000.0  # 30 km model top
        base = create_cubed_sphere(n)
        height = create_height_coordinate(nlev, H)
        z_s = jnp.zeros((6, n, n))
        terrain = compute_terrain_metric(z_s, height)
        config = CDGridCompressibleEulerConfig(use_coriolis=False)
        self.model = CDGridCompressibleEulerModel(base, height, terrain, config)
        self.dt = 1.0  # small dt for compressible

        key = jax.random.PRNGKey(20)
        theta_prime = 0.5 * jax.random.normal(key, (6, n, n, nlev))
        self.state = NonHydrostaticState(
            u=Field(jnp.zeros((6, n, n, nlev)), name="u"),
            v=Field(jnp.zeros((6, n, n, nlev)), name="v"),
            w=Field(jnp.zeros((6, n, n, nlev + 1)), name="w"),
            theta_prime=Field(theta_prime, name="theta_prime"),
            rho_prime=Field(jnp.zeros((6, n, n, nlev)), name="rho_prime"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
            tracers=Field(jnp.zeros((6, n, n, nlev, 0)), name="tracers"),
        )

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(theta_data):
            s = state._replace(theta_prime=state.theta_prime.replace(data=theta_data))
            out = model.step(s, dt)
            return jnp.sum(out.theta_prime.data ** 2)

        grad = jax.grad(loss)(state.theta_prime.data)
        assert_gradient_ok(grad, "CompEuler single step")


# ============================================================================
# 1d  Spectral PE (T5, 5 levels)
# ============================================================================

class TestSpectralPE:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
            SpectralHydrostaticState,
        )

        nlev = 5
        grid = create_gaussian_grid(5, allow_unsupported_backend=True)
        sigma = create_sigma_coordinate(nlev)
        config = SpectralPEConfig()
        self.model = SpectralPrimitiveEquationModel(
            grid, sigma, config, allow_unsupported_backend=True,
        )
        self.dt = 120.0
        n_sh = grid.n_sh

        key = jax.random.PRNGKey(30)
        T_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
        # Set mean temperature ~ 250 K
        T_hat = T_hat.at[0, :].set(250.0 * jnp.sqrt(4 * jnp.pi))
        T_hat = T_hat + 0.1 * jax.random.normal(key, (n_sh, nlev))
        # log(p_s) ~ log(1e5) ≈ 11.51
        lnps_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
        lnps_hat = lnps_hat.at[0].set(jnp.log(1e5) * jnp.sqrt(4 * jnp.pi))

        self.state = SpectralHydrostaticState(
            vor_hat=Field(jnp.zeros((n_sh, nlev), dtype=jnp.complex128), name="vor_hat"),
            div_hat=Field(jnp.zeros((n_sh, nlev), dtype=jnp.complex128), name="div_hat"),
            T_hat=Field(T_hat, name="T_hat"),
            lnps_hat=Field(lnps_hat, name="lnps_hat"),
            phis_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="phis_hat"),
        )
        self.n_sh = n_sh

    def test_grad_single_step(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_real):
            T_hat_new = T_real + 1j * state.T_hat.data.imag
            s = state._replace(T_hat=state.T_hat.replace(data=T_hat_new))
            out = model.step(s, dt)
            return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

        grad = jax.grad(loss)(state.T_hat.data.real)
        assert_gradient_ok(grad, "Spectral PE single step")
