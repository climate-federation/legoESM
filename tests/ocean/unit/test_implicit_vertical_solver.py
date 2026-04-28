"""Tests for the implicit backward-Euler vertical diffusion solver (#204).

This is the building block that lets the ocean dycore decouple the
physical vertical viscosity / diffusivity from the numerical diffusion
of 1st-order upwind vertical advection.

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_implicit_vertical_solver.py -v
"""

from __future__ import annotations

import math
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.vertical_mixing import (
    implicit_vertical_diffusion_ocean,
    build_dz_half,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# 1. Plumbing & shape handling
# ---------------------------------------------------------------------------


class TestShape:
    def test_single_column_shape(self):
        phi = jnp.linspace(0.0, 1.0, 5)
        dz = jnp.full((5,), 10.0)
        dzh = build_dz_half(dz)
        K = 1.0e-3
        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=60.0)
        assert out.shape == phi.shape

    def test_batched_2d(self):
        rng = np.random.RandomState(0)
        phi = jnp.array(rng.randn(4, 8))             # (ncol, nlev)
        dz = jnp.full((8,), 5.0)
        dzh = build_dz_half(dz)
        K = jnp.full((4, 7), 2e-3)
        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=60.0)
        assert out.shape == phi.shape

    def test_batched_3d_latlon(self):
        rng = np.random.RandomState(1)
        phi = jnp.array(rng.randn(3, 5, 7))          # (nlat, nlon, nlev)
        dz = jnp.full((7,), 8.0)
        dzh = build_dz_half(dz)
        K = jnp.full((3, 5, 6), 1e-3)
        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=100.0)
        assert out.shape == phi.shape

    def test_cubed_sphere_shape(self):
        rng = np.random.RandomState(2)
        phi = jnp.array(rng.randn(6, 4, 4, 10))
        dz = jnp.full((10,), 50.0)
        dzh = build_dz_half(dz)
        K = 5e-4
        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=300.0)
        assert out.shape == phi.shape

    def test_invalid_dt_raises(self):
        phi = jnp.ones(4)
        dz = jnp.ones(4)
        dzh = build_dz_half(dz)
        with pytest.raises(ValueError):
            implicit_vertical_diffusion_ocean(phi, 1.0, dz, dzh, dt=0.0)
        with pytest.raises(ValueError):
            implicit_vertical_diffusion_ocean(phi, 1.0, dz, dzh, dt=-1.0)

    def test_mismatched_K_length_raises(self):
        phi = jnp.ones(5)
        dz = jnp.ones(5)
        dzh = build_dz_half(dz)
        K = jnp.ones(5)   # wrong — should be nlev-1 = 4
        with pytest.raises(ValueError):
            implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=1.0)

    def test_single_level_is_noop(self):
        phi = jnp.array([7.5])
        dz = jnp.array([10.0])
        dzh = jnp.array([], dtype=jnp.float64)
        K = jnp.array([], dtype=jnp.float64)
        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=100.0)
        assert jnp.allclose(out, phi)


# ---------------------------------------------------------------------------
# 2. Conservation (no-flux BCs ⇒ total φ preserved)
# ---------------------------------------------------------------------------


class TestConservation:
    def test_mass_conserved_uniform_K(self):
        rng = np.random.RandomState(3)
        phi = jnp.array(rng.randn(16))
        dz = jnp.full((16,), 12.5)
        dzh = build_dz_half(dz)
        K = jnp.full((15,), 1e-2)
        out = implicit_vertical_diffusion_ocean(phi, K, dz, dzh, dt=60.0)
        # Volume-weighted total is what the no-flux BC conserves.
        m0 = float(jnp.sum(phi * dz))
        m1 = float(jnp.sum(out * dz))
        assert abs(m0 - m1) < 1e-10

    def test_mass_conserved_nonuniform_K(self):
        rng = np.random.RandomState(4)
        phi = jnp.array(rng.randn(20))
        dz = jnp.linspace(2.0, 40.0, 20)   # stretched
        dzh = build_dz_half(dz)
        K_profile = jnp.exp(-jnp.linspace(0, 3, 19))  # surface-enhanced
        out = implicit_vertical_diffusion_ocean(
            phi, K_profile * 1e-1, dz, dzh, dt=600.0)
        m0 = float(jnp.sum(phi * dz))
        m1 = float(jnp.sum(out * dz))
        assert abs(m0 - m1) < 1e-10

    def test_mass_conserved_many_steps(self):
        rng = np.random.RandomState(5)
        phi = jnp.array(rng.randn(12))
        dz = jnp.full((12,), 20.0)
        dzh = build_dz_half(dz)
        K = 5e-3
        m0 = float(jnp.sum(phi * dz))
        for _ in range(50):
            phi = implicit_vertical_diffusion_ocean(
                phi, K, dz, dzh, dt=600.0)
        m1 = float(jnp.sum(phi * dz))
        assert abs(m0 - m1) < 1e-9


# ---------------------------------------------------------------------------
# 3. Physics — approach to the well-mixed state, analytical decay
# ---------------------------------------------------------------------------


class TestPhysicsCorrectness:
    def test_well_mixed_limit(self):
        """Repeated implicit steps collapse the field to the
        volume-weighted mean — each step's first-mode damping factor
        is ``1 / (1 + dt·K·(π/H)²)``, so a few large steps suffice to
        push the residual below 1e-4."""
        phi = jnp.linspace(10.0, 0.0, 20)
        dz = jnp.full((20,), 5.0)
        dzh = build_dz_half(dz)
        K = 1.0                           # ridiculous mixing coeff [m²/s]
        out = phi
        for _ in range(5):
            out = implicit_vertical_diffusion_ocean(
                out, K, dz, dzh, dt=1e5)
        phi_mean = float(jnp.sum(phi * dz) / jnp.sum(dz))
        assert jnp.max(jnp.abs(out - phi_mean)) < 1e-4

    def test_exponential_decay_of_first_mode(self):
        """For a long enough column with uniform dz and K, a sinusoidal
        mode ``cos(π z / H)`` decays by ``exp(−K π² dt / H²)``.  Tested
        against two backward-Euler steps to avoid leading-order dt bias.
        """
        nlev = 128
        H = 100.0
        dz = jnp.full((nlev,), H / nlev)
        dzh = build_dz_half(dz)
        z = (jnp.arange(nlev) + 0.5) * dz[0]
        phi = jnp.cos(jnp.pi * z / H)         # first even cosine mode
        K = 1e-2                              # m²/s
        dt = 10.0
        # Backward-Euler damping factor per step: 1 / (1 + dt·K·(π/H)²)
        kpi = jnp.pi / H
        decay = 1.0 / (1.0 + dt * K * kpi ** 2)
        # Three steps
        out = phi
        for _ in range(3):
            out = implicit_vertical_diffusion_ocean(out, K, dz, dzh, dt)
        expected = decay ** 3 * phi
        # Higher modes leak into the solution via the centred-difference
        # stencil; check the inner 80 % of the column where the boundary
        # effect is negligible.
        err = jnp.max(jnp.abs(out[20:-20] - expected[20:-20]))
        assert float(err) < 5e-3

    def test_unconditional_stability(self):
        """Backward-Euler is stable for *any* dt.  Pick a dt far beyond
        the explicit CFL, confirm the solution stays bounded, and show
        that after a few such giant steps the residual is tiny.
        """
        rng = np.random.RandomState(7)
        phi = jnp.array(rng.randn(40)) * 5.0
        dz = jnp.full((40,), 1.0)                 # 1 m layers
        dzh = build_dz_half(dz)
        K = 1e-2                                   # CFL α = 1e-2 · dt → huge
        out = phi
        for _ in range(5):
            out = implicit_vertical_diffusion_ocean(
                out, K, dz, dzh, dt=1e6)
        assert jnp.all(jnp.isfinite(out))
        mean = float(jnp.sum(phi * dz) / jnp.sum(dz))
        # After 5 · 1e6 s of implicit mixing the residual should be
        # several orders of magnitude below the initial variance.
        assert jnp.max(jnp.abs(out - mean)) < 1e-4

    def test_zero_K_is_identity(self):
        rng = np.random.RandomState(9)
        phi = jnp.array(rng.randn(30))
        dz = jnp.full((30,), 3.0)
        dzh = build_dz_half(dz)
        out = implicit_vertical_diffusion_ocean(phi, 0.0, dz, dzh, dt=3600.0)
        assert jnp.allclose(out, phi, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_no_flux_top_boundary(self):
        """Initial state with a discontinuity at the *surface* must
        relax into the interior without generating a spurious
        upward flux — i.e. the surface-layer φ does not diverge."""
        phi = jnp.concatenate([jnp.array([10.0]), jnp.zeros(19)])
        dz = jnp.full((20,), 5.0)
        dzh = build_dz_half(dz)
        K = 1e-3
        out = implicit_vertical_diffusion_ocean(
            phi, K, dz, dzh, dt=6 * 3600.0)
        # Total conserved
        m0 = float(jnp.sum(phi * dz))
        m1 = float(jnp.sum(out * dz))
        assert abs(m0 - m1) < 1e-10
        # Surface value must have *decreased* (diffused down).
        assert out[0] < phi[0]
        # And bottom must have *increased* (some flux reached it —
        # given the large dt).
        assert out[-1] > phi[-1]

    def test_no_flux_bottom_boundary(self):
        """Symmetric test: discontinuity at the bottom."""
        phi = jnp.concatenate([jnp.zeros(19), jnp.array([10.0])])
        dz = jnp.full((20,), 5.0)
        dzh = build_dz_half(dz)
        K = 1e-3
        out = implicit_vertical_diffusion_ocean(
            phi, K, dz, dzh, dt=6 * 3600.0)
        m0 = float(jnp.sum(phi * dz))
        m1 = float(jnp.sum(out * dz))
        assert abs(m0 - m1) < 1e-10
        assert out[-1] < phi[-1]
        assert out[0] > phi[0]


# ---------------------------------------------------------------------------
# 5. JAX compatibility
# ---------------------------------------------------------------------------


class TestJAX:
    def test_grad_finite_wrt_field(self):
        phi = jnp.linspace(0.0, 1.0, 12)
        dz = jnp.full((12,), 4.0)
        dzh = build_dz_half(dz)
        K = 1e-2

        def loss(p):
            out = implicit_vertical_diffusion_ocean(p, K, dz, dzh, dt=60.0)
            return jnp.sum(out ** 2)

        g = jax.grad(loss)(phi)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_wrt_K(self):
        phi = jnp.linspace(0.0, 1.0, 8)
        dz = jnp.full((8,), 4.0)
        dzh = build_dz_half(dz)
        K = jnp.full((7,), 1e-2)

        def loss(kk):
            out = implicit_vertical_diffusion_ocean(phi, kk, dz, dzh, dt=60.0)
            return jnp.sum(out ** 2)

        g = jax.grad(loss)(K)
        assert jnp.all(jnp.isfinite(g))

    def test_jit_runs(self):
        phi = jnp.linspace(0.0, 1.0, 10)
        dz = jnp.full((10,), 3.0)
        dzh = build_dz_half(dz)
        K = 1e-3

        @jax.jit
        def step(p):
            return implicit_vertical_diffusion_ocean(p, K, dz, dzh, dt=60.0)

        out = step(phi)
        assert out.shape == phi.shape

    def test_jit_with_traced_dt(self):
        """Regression: ``dt`` must be allowed as a *traced* argument
        inside a jitted timestep without tripping ``TracerBoolConversionError``.
        """
        phi = jnp.linspace(0.0, 1.0, 10)
        dz = jnp.full((10,), 3.0)
        dzh = build_dz_half(dz)
        K = 1e-3

        @jax.jit
        def step(p, dt):
            return implicit_vertical_diffusion_ocean(p, K, dz, dzh, dt=dt)

        out = step(phi, 60.0)
        assert out.shape == phi.shape
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# 6. Public-API plumbing
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_exported_from_vertical_mixing(self):
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean as f1,
            build_dz_half as f2,
        )
        assert callable(f1)
        assert callable(f2)

    def test_build_dz_half_mean(self):
        dz = jnp.array([5.0, 15.0, 25.0])
        dzh = build_dz_half(dz)
        assert dzh.shape == (2,)
        assert jnp.allclose(dzh, jnp.array([10.0, 20.0]))
