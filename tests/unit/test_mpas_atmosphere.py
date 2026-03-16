"""Tests for MPAS atmosphere dynamical cores (hydrostatic PE + non-hydrostatic CE).

Tests verify:
1. Tendency shapes are correct
2. Model step runs without errors
3. Finite output (no NaN/Inf)
4. Mass conservation (hydrostatic)
5. Multi-step stability
6. JAX differentiability
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import unittest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.core.field import Field
from legoesm.core.state import (
    MPASHydrostaticState,
    MPASHydrostaticTendencies,
    MPASNonHydrostaticState,
    MPASNonHydrostaticTendencies,
)
from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    MPASPrimitiveEquationModel,
    mpas_hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.compressible_euler_mpas import (
    MPASCompressibleEulerConfig,
    MPASCompressibleEulerModel,
    mpas_compressible_euler_slow_tendencies,
)


# ============================================================================
# Shared test fixtures
# ============================================================================

def _make_mesh(level=2):
    """Small icosahedral mesh for testing (level 2 = 162 cells)."""
    return create_voronoi_mesh(level, lloyd_iterations=5)


def _make_hydrostatic_state(mesh, nlev=5):
    """Create a simple rest state for the hydrostatic PE."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges

    T0 = 250.0  # reference temperature [K]
    p_s0 = 1e5  # surface pressure [Pa]

    u = Field(data=jnp.zeros((nEdges, nlev), dtype=jnp.float64),
              name="u", dims=("nEdges", "nlev"), units="m/s")
    T = Field(data=jnp.full((nCells, nlev), T0, dtype=jnp.float64),
              name="T", dims=("nCells", "nlev"), units="K")
    p_s = Field(data=jnp.full((nCells,), p_s0, dtype=jnp.float64),
                name="p_s", dims=("nCells",), units="Pa")
    phis = Field(data=jnp.zeros((nCells,), dtype=jnp.float64),
                 name="phis", dims=("nCells",), units="m²/s²")

    return MPASHydrostaticState(u=u, T=T, p_s=p_s, phis=phis)


def _make_nh_state(mesh, height_coord, terrain_metric, nlev=5):
    """Create a simple rest state for the non-hydrostatic CE."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges

    u = Field(data=jnp.zeros((nEdges, nlev), dtype=jnp.float64),
              name="u", dims=("nEdges", "nlev"), units="m/s")
    w = Field(data=jnp.zeros((nCells, nlev + 1), dtype=jnp.float64),
              name="w", dims=("nCells", "nlev_half"), units="m/s")
    theta_prime = Field(
        data=jnp.zeros((nCells, nlev), dtype=jnp.float64),
        name="theta_prime", dims=("nCells", "nlev"), units="K")
    rho_prime = Field(
        data=jnp.zeros((nCells, nlev), dtype=jnp.float64),
        name="rho_prime", dims=("nCells", "nlev"), units="kg/m³")
    phis = Field(data=jnp.zeros((nCells,), dtype=jnp.float64),
                 name="phis", dims=("nCells",), units="m²/s²")
    tracers = Field(
        data=jnp.zeros((nCells, nlev, 0), dtype=jnp.float64),
        name="tracers", dims=("nCells", "nlev", "tracer"), units="kg/kg")

    return MPASNonHydrostaticState(
        u=u, w=w, theta_prime=theta_prime, rho_prime=rho_prime,
        phis=phis, tracers=tracers,
    )


def _add_perturbation_hydro(state, mesh, nlev):
    """Add a small wind perturbation for non-trivial tendencies."""
    nEdges = mesh.nEdges
    key = jax.random.PRNGKey(42)
    u_pert = 1.0 * jax.random.normal(key, (nEdges, nlev))
    return state._replace(
        u=state.u.replace(data=state.u.data + u_pert),
    )


def _add_perturbation_nh(state, mesh, nlev):
    """Add small perturbations for non-trivial NH tendencies."""
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    u_pert = 1.0 * jax.random.normal(k1, (nEdges, nlev))
    theta_pert = 0.5 * jax.random.normal(k2, (nCells, nlev))
    rho_pert = 0.001 * jax.random.normal(k3, (nCells, nlev))
    return state._replace(
        u=state.u.replace(data=state.u.data + u_pert),
        theta_prime=state.theta_prime.replace(
            data=state.theta_prime.data + theta_pert),
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + rho_pert),
    )


# ============================================================================
# Hydrostatic PE tests
# ============================================================================

class TestMPASHydrostaticPE(unittest.TestCase):
    """Tests for the MPAS hydrostatic primitive equations."""

    @classmethod
    def setUpClass(cls):
        cls.nlev = 5
        cls.mesh = _make_mesh(level=2)
        cls.sigma = create_sigma_coordinate(cls.nlev)
        cls.config = MPASPrimitiveEquationConfig(
            nu_del2=1e4, K_h=1e3,
        )
        cls.state = _make_hydrostatic_state(cls.mesh, cls.nlev)
        cls.state_pert = _add_perturbation_hydro(
            cls.state, cls.mesh, cls.nlev)

    def test_tendency_shapes(self):
        """Tendencies have correct shapes."""
        tend = mpas_hydrostatic_tendencies(
            self.state_pert, self.mesh, self.sigma, self.config,
        )
        self.assertEqual(tend.du_dt.data.shape,
                         (self.mesh.nEdges, self.nlev))
        self.assertEqual(tend.dT_dt.data.shape,
                         (self.mesh.nCells, self.nlev))
        self.assertEqual(tend.dp_s_dt.data.shape,
                         (self.mesh.nCells,))
        self.assertEqual(tend.dphis_dt.data.shape,
                         (self.mesh.nCells,))

    def test_tendencies_finite(self):
        """Tendencies contain no NaN or Inf."""
        tend = mpas_hydrostatic_tendencies(
            self.state_pert, self.mesh, self.sigma, self.config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dT_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dp_s_dt.data)))

    def test_rest_state_zero_tendencies(self):
        """Rest state (u=0, uniform T, uniform p_s) has near-zero tendencies."""
        tend = mpas_hydrostatic_tendencies(
            self.state, self.mesh, self.sigma, self.config,
        )
        # At rest, all tendencies should be very small
        self.assertLess(float(jnp.max(jnp.abs(tend.du_dt.data))), 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(tend.dp_s_dt.data))), 1e-1)

    def test_model_step(self):
        """Model.step() produces finite state."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, self.config,
        )
        state_new = model.step(self.state_pert, dt=60.0)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.p_s.data)))

    def test_mass_conservation(self):
        """Mass (sum of p_s * area) is conserved after one step."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, self.config,
        )
        area = self.mesh.areaCell
        mass_old = float(jnp.sum(self.state_pert.p_s.data * area))
        state_new = model.step(self.state_pert, dt=60.0)
        mass_new = float(jnp.sum(state_new.p_s.data * area))
        rel_err = abs(mass_new - mass_old) / abs(mass_old)
        self.assertLess(rel_err, 1e-12)

    def test_multi_step_stability(self):
        """3 steps without NaN."""
        model = MPASPrimitiveEquationModel(
            self.mesh, self.sigma, self.config,
        )
        state = self.state_pert
        for _ in range(3):
            state = model.step(state, dt=60.0)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.T.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.p_s.data)))

    def test_differentiability(self):
        """Tendencies are differentiable w.r.t. velocity."""
        def loss(u_data):
            s = self.state._replace(
                u=self.state.u.replace(data=u_data))
            tend = mpas_hydrostatic_tendencies(
                s, self.mesh, self.sigma, self.config)
            return jnp.sum(tend.du_dt.data ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(self.state_pert.u.data)
        self.assertEqual(g.shape, self.state_pert.u.data.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


# ============================================================================
# Non-hydrostatic CE tests
# ============================================================================

class TestMPASNonHydrostaticCE(unittest.TestCase):
    """Tests for the MPAS non-hydrostatic compressible Euler equations."""

    @classmethod
    def setUpClass(cls):
        cls.nlev = 5
        cls.H = 30000.0
        cls.mesh = _make_mesh(level=2)
        cls.height_coord = create_height_coordinate(cls.nlev, cls.H)
        z_s = jnp.zeros(cls.mesh.nCells, dtype=jnp.float64)
        cls.terrain_metric = compute_terrain_metric(z_s, cls.height_coord)
        cls.config = MPASCompressibleEulerConfig(
            nu_del2=1e4, n_acoustic_substeps=4,
        )
        cls.state = _make_nh_state(
            cls.mesh, cls.height_coord, cls.terrain_metric, cls.nlev,
        )
        cls.state_pert = _add_perturbation_nh(
            cls.state, cls.mesh, cls.nlev)

    def test_slow_tendency_shapes(self):
        """Slow tendencies have correct shapes."""
        tend = mpas_compressible_euler_slow_tendencies(
            self.state_pert, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertEqual(tend.du_dt.data.shape,
                         (self.mesh.nEdges, self.nlev))
        self.assertEqual(tend.dw_dt.data.shape,
                         (self.mesh.nCells, self.nlev + 1))
        self.assertEqual(tend.dtheta_prime_dt.data.shape,
                         (self.mesh.nCells, self.nlev))
        self.assertEqual(tend.drho_prime_dt.data.shape,
                         (self.mesh.nCells, self.nlev))
        self.assertEqual(tend.dtracers_dt.data.shape,
                         (self.mesh.nCells, self.nlev, 0))

    def test_slow_tendencies_finite(self):
        """Slow tendencies contain no NaN or Inf."""
        tend = mpas_compressible_euler_slow_tendencies(
            self.state_pert, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertTrue(jnp.all(jnp.isfinite(tend.du_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dw_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.drho_prime_dt.data)))

    def test_model_step(self):
        """Model.step() produces finite state."""
        model = MPASCompressibleEulerModel(
            self.mesh, self.height_coord, self.terrain_metric, self.config,
        )
        state_new = model.step(self.state_pert, dt=5.0)
        self.assertTrue(jnp.all(jnp.isfinite(state_new.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.w.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.theta_prime.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state_new.rho_prime.data)))

    def test_rest_state_small_tendencies(self):
        """Rest state has near-zero slow tendencies."""
        tend = mpas_compressible_euler_slow_tendencies(
            self.state, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertLess(float(jnp.max(jnp.abs(tend.du_dt.data))), 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))), 1e-3)

    def test_multi_step_stability(self):
        """3 steps without NaN."""
        model = MPASCompressibleEulerModel(
            self.mesh, self.height_coord, self.terrain_metric, self.config,
        )
        state = self.state_pert
        for _ in range(3):
            state = model.step(state, dt=5.0)
        self.assertTrue(jnp.all(jnp.isfinite(state.u.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.w.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.theta_prime.data)))
        self.assertTrue(jnp.all(jnp.isfinite(state.rho_prime.data)))

    def test_with_tracers(self):
        """Model runs with tracers."""
        nCells = self.mesh.nCells
        n_tracers = 2
        tracers = Field(
            data=jnp.ones((nCells, self.nlev, n_tracers), dtype=jnp.float64),
            name="tracers", dims=("nCells", "nlev", "tracer"), units="kg/kg",
        )
        state = self.state_pert._replace(tracers=tracers)
        tend = mpas_compressible_euler_slow_tendencies(
            state, self.mesh, self.height_coord,
            self.terrain_metric, self.config,
        )
        self.assertEqual(tend.dtracers_dt.data.shape,
                         (nCells, self.nlev, n_tracers))
        self.assertTrue(jnp.all(jnp.isfinite(tend.dtracers_dt.data)))

    def test_differentiability(self):
        """Slow tendencies are differentiable w.r.t. theta_prime."""
        def loss(theta_p_data):
            s = self.state._replace(
                theta_prime=self.state.theta_prime.replace(data=theta_p_data))
            tend = mpas_compressible_euler_slow_tendencies(
                s, self.mesh, self.height_coord,
                self.terrain_metric, self.config)
            return jnp.sum(tend.du_dt.data ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(self.state_pert.theta_prime.data)
        self.assertEqual(g.shape, self.state_pert.theta_prime.data.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


if __name__ == "__main__":
    unittest.main()
