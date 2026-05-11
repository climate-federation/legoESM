"""Tests for ``_shared.compute_moisture_convergence``.

The diagnostic returns the column-flattened ``MC = -∇·(q_v u)`` per
level by reusing the dycore's FV-flux-divergence operator with the
slope limiter disabled.  Tests pin:

* the constant-q_v identity ``MC = -q_v * div(u)`` to numerical
  precision;
* sign convention — convergent flow into a moist column gives
  ``MC > 0`` somewhere in the column;
* the result is finite for typical synthetic fields and matches the
  expected ``(ncol, nlev)`` shape;
* differentiability through ``q_v`` (the diagnostic is part of the
  Tiedtke / Bechtold AD path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.physics._shared import compute_moisture_convergence


# ---------------------------------------------------------------------------
# Cubed-sphere fixture
# ---------------------------------------------------------------------------

def _cubed_sphere_field(n: int = 4, nlev: int = 6, fill: float = 1.0):
    grid = create_cubed_sphere(n)
    shape = (6, n, n, nlev)
    return grid, jnp.full(shape, fill)


# ---------------------------------------------------------------------------
# Shape / finiteness
# ---------------------------------------------------------------------------

def test_moisture_convergence_shape_and_finite():
    grid, q_v = _cubed_sphere_field(n=4, nlev=8, fill=1.0e-2)
    u = 5.0 * jnp.ones_like(q_v)
    v = 0.0 * jnp.ones_like(q_v)
    mc = compute_moisture_convergence(q_v, u, v, grid)
    n = 4; nlev = 8
    assert mc.shape == (6 * n * n, nlev)
    assert jnp.all(jnp.isfinite(mc))


# ---------------------------------------------------------------------------
# Constant q_v: MC = -q_v * div(u)
# ---------------------------------------------------------------------------

def test_constant_q_v_factors_cleanly():
    """For a constant ``q_v`` field, ``MC = -∇·(q_v u) = -q_v ∇·u`` —
    the FV operator should produce a result that scales linearly with
    the constant ``q_v``."""
    grid, q1 = _cubed_sphere_field(n=4, nlev=4, fill=1.0)
    _, q2 = _cubed_sphere_field(n=4, nlev=4, fill=2.0)
    # Use a wind field with non-trivial divergence (linear in face-x).
    n = 4; nlev = 4
    u_pattern = jnp.linspace(-5.0, 5.0, n)[None, :, None, None]
    u = jnp.broadcast_to(u_pattern, (6, n, n, nlev))
    v = jnp.zeros_like(u)
    mc1 = compute_moisture_convergence(q1, u, v, grid)
    mc2 = compute_moisture_convergence(q2, u, v, grid)
    # mc2 should be ~ 2x mc1 (linear in q_v).
    assert jnp.allclose(mc2, 2.0 * mc1, atol=1e-12)


# ---------------------------------------------------------------------------
# Sign convention — convergent flow gives MC > 0 somewhere
# ---------------------------------------------------------------------------

def test_convergent_flow_positive_mc_somewhere():
    """A converging wind pattern (winds pointing inward) into a
    uniformly-moist column produces ``MC > 0`` at some grid point."""
    grid, q_v = _cubed_sphere_field(n=4, nlev=4, fill=1.0e-2)
    n = 4; nlev = 4
    # Winds converging on the cube-face center: u positive on left
    # half, negative on right half (and vice versa for v).  This is
    # a convergent flow on each face center.
    u_x = jnp.linspace(1.0, -1.0, n)
    v_y = jnp.linspace(1.0, -1.0, n)
    u = jnp.broadcast_to(u_x[None, None, :, None], (6, n, n, nlev))
    v = jnp.broadcast_to(v_y[None, :, None, None], (6, n, n, nlev))
    mc = compute_moisture_convergence(q_v, u, v, grid)
    # At least one point has positive MC.  (Convergent flow ⇒ +MC at
    # the convergence point.)
    assert float(jnp.max(mc)) > 0.0


def test_mc_sign_matches_dycore_tracer_tendency():
    """Audit cycle iter-35 finding F1 (CRITICAL): the cubed-sphere
    branch of ``compute_moisture_convergence`` had been
    double-negating ``fv_flux_divergence_3d``'s output, returning
    ``+div(q·V)`` (moisture DIVERGENCE) instead of ``-div(q·V)``
    (moisture CONVERGENCE).  The dycore tracer tendency uses the
    convention ``dq/dt = -div(q·V)`` (see ``fv_flux_divergence_3d``
    docstring).  After the iter-35 fix, ``compute_moisture_convergence``
    must return the SAME quantity (up to per-column reshape) as the
    underlying flux-divergence operator — both represent
    ``dq/dt = -div(q·V)``.

    Pin this by comparing the two outputs directly.
    """
    from legoesm.core.operators_3d import fv_flux_divergence_3d
    grid, q_v = _cubed_sphere_field(n=8, nlev=2, fill=1.0e-2)
    n = 8; nlev = 2
    u = jnp.linspace(-5.0, 5.0, n)
    u = jnp.broadcast_to(u[None, None, :, None], (6, n, n, nlev))
    v = jnp.zeros_like(u)

    mc = compute_moisture_convergence(q_v, u, v, grid)
    fv_tendency = fv_flux_divergence_3d(
        q_v, u, v, grid, limiter=False,
    ).reshape(6 * n * n, nlev)

    # The two arrays must be IDENTICAL.  Pre-fix the cubed-sphere
    # branch returned -fv_tendency; this assertion would have
    # caught the regression.
    assert jnp.allclose(mc, fv_tendency, rtol=1e-12), (
        "compute_moisture_convergence must return the same value "
        "as fv_flux_divergence_3d (both = dq/dt = -div(q·V) = MC). "
        "The iter-35 F1 fix removed a stale double-negation."
    )


# ---------------------------------------------------------------------------
# Differentiability — gradient through q_v
# ---------------------------------------------------------------------------

def test_grad_through_q_v_finite():
    """``d (sum MC) / d q_v`` is finite (the limiter-off variant is
    smooth)."""
    grid, q_v = _cubed_sphere_field(n=4, nlev=4, fill=1.0e-2)
    n = 4; nlev = 4
    u = jnp.full((6, n, n, nlev), 3.0)
    v = jnp.zeros_like(u)

    def f(qv):
        return jnp.sum(compute_moisture_convergence(qv, u, v, grid))

    g = jax.grad(f)(q_v)
    assert g.shape == q_v.shape
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Unsupported grid raises clearly
# ---------------------------------------------------------------------------

def test_unsupported_grid_raises_typeerror():
    class _DummyGrid:
        pass

    qv = jnp.zeros((1, 1, 1))
    with pytest.raises(TypeError, match="Supported"):
        compute_moisture_convergence(qv, qv, qv, _DummyGrid())


# ---------------------------------------------------------------------------
# Gaussian grid (spectral PE) branch
# ---------------------------------------------------------------------------

class TestGaussianGrid:
    """Tests for the GaussianGrid branch of compute_moisture_convergence.

    The spectral PE bridge feeds q_v · u and q_v · v through the
    transform pathway: synthesize the product, take spectral div via
    ``vordiv_from_uv_3d``, synthesize back, negate.  The pole-safe
    ``sh_analysis_oc2`` / ``sh_analysis_dmu`` machinery does the heavy
    lifting.  These tests pin shape, sign, and differentiability.
    """

    def _grid_field(self, n_max: int = 21, nlev: int = 4, fill: float = 1.0):
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(n_max=n_max)
        shape = (grid.n_lat, grid.n_lon, nlev)
        return grid, jnp.full(shape, fill, dtype=jnp.float64)

    def test_gaussian_shape_and_finite(self):
        grid, q_v = self._grid_field()
        u = 5.0 * jnp.ones_like(q_v)
        v = 0.0 * jnp.ones_like(q_v)
        mc = compute_moisture_convergence(q_v, u, v, grid)
        nlev = q_v.shape[-1]
        assert mc.shape == (grid.n_lat * grid.n_lon, nlev)
        assert jnp.all(jnp.isfinite(mc))

    def test_gaussian_zero_winds_yield_zero_mc(self):
        grid, q_v = self._grid_field(fill=1.5e-2)
        u = jnp.zeros_like(q_v)
        v = jnp.zeros_like(q_v)
        mc = compute_moisture_convergence(q_v, u, v, grid)
        # Synthesizing zero on the grid through the SH cycle is zero
        # to machine precision in float64.
        assert float(jnp.max(jnp.abs(mc))) < 1e-12

    def test_gaussian_constant_q_v_factors_linearly(self):
        """For constant q_v, MC ∝ q_v."""
        grid, q1 = self._grid_field(fill=1.0)
        _, q2 = self._grid_field(fill=2.5)
        # Wave-1 zonal flow gives non-trivial divergence.
        a = grid.radius
        u0 = 5.0
        lon = grid.lon[None, :, None]
        u = u0 * jnp.cos(lon) * jnp.ones_like(q1)
        v = jnp.zeros_like(q1)
        mc1 = compute_moisture_convergence(q1, u, v, grid)
        mc2 = compute_moisture_convergence(q2, u, v, grid)
        assert bool(jnp.allclose(mc2, 2.5 * mc1, atol=1e-10))

    def test_gaussian_grad_through_q_v_finite(self):
        grid, q_v = self._grid_field(fill=1.5e-2)
        u = 3.0 * jnp.ones_like(q_v)
        v = jnp.zeros_like(q_v)

        def f(qv):
            return jnp.sum(compute_moisture_convergence(qv, u, v, grid))

        g = jax.grad(f)(q_v)
        assert g.shape == q_v.shape
        assert jnp.all(jnp.isfinite(g))
