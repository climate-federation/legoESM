"""FV3_3D iter 718: get_vorticity_fv3 port.

Faithful JAX port of FV3 ``get_vorticity`` (tools/fv_diagnostics.F90:
3877-3908).  Diagnostic circulation-form vorticity at cell center.

Tests
-----

1. ``test_vort_zero_wind``.
2. ``test_vort_uniform_wind_no_curl``.
3. ``test_vort_solid_rotation_2d``.
4. ``test_vort_shapes_4d``.
5. ``test_vort_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import get_vorticity_fv3


def test_vort_zero_wind():
    """Zero wind → vort = 0."""
    n = 6
    u = jnp.zeros((n, n + 1))
    v = jnp.zeros((n + 1, n))
    dx = jnp.full((n, n + 1), 1000.0)
    dy = jnp.full((n + 1, n), 1000.0)
    rarea = jnp.full((n, n), 1.0 / (1000.0 * 1000.0))
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert vort.shape == (n, n)
    assert jnp.all(jnp.abs(vort) < 1e-15)


def test_vort_uniform_wind_no_curl():
    """Uniform u, v on a uniform grid → vort = 0 (no curl)."""
    n = 6
    u = jnp.full((n, n + 1), 5.0)
    v = jnp.full((n + 1, n), 3.0)
    dx = jnp.full((n, n + 1), 1000.0)
    dy = jnp.full((n + 1, n), 1000.0)
    rarea = jnp.full((n, n), 1.0 / (1000.0 * 1000.0))
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert jnp.all(jnp.abs(vort) < 1e-12)


def test_vort_solid_rotation_2d():
    """Solid-body rotation ω: u(y) = -ω·y, v(x) = +ω·x → curl = 2ω.

    Build u as a function of j-position (y), v as a function of
    i-position (x).  D-grid: u lives at y-edges (j ∈ [0, n]),
    v lives at x-edges (i ∈ [0, n]).
    """
    n = 6
    omega = 0.01
    dx_cell = 1000.0
    # u[i, j] at y_j = (j - n/2)*dx (centered at grid centre)
    # j ranges over n+1 edges
    j_edge = jnp.arange(n + 1) - n / 2.0
    u = -omega * j_edge[None, :] * dx_cell  # (n, n+1), constant in i
    u = jnp.broadcast_to(u, (n, n + 1))
    # v[i, j] at x_i: i ranges over n+1 edges
    i_edge = jnp.arange(n + 1) - n / 2.0
    v = omega * i_edge[:, None] * dx_cell    # (n+1, n)
    v = jnp.broadcast_to(v, (n + 1, n))
    dx = jnp.full((n, n + 1), dx_cell)
    dy = jnp.full((n + 1, n), dx_cell)
    rarea = jnp.full((n, n), 1.0 / (dx_cell * dx_cell))
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    # Solid-body rotation → vort = 2·ω everywhere
    expected = 2.0 * omega
    assert jnp.all(jnp.abs(vort - expected) < 1e-12)


def test_vort_shapes_4d():
    """4-D (n, n+1, km) input → (n, n, km) output."""
    rng = np.random.default_rng(seed=718)
    n, km = 8, 5
    u = jnp.asarray(rng.normal(scale=10.0, size=(n, n + 1, km)))
    v = jnp.asarray(rng.normal(scale=10.0, size=(n + 1, n, km)))
    dx = jnp.full((n, n + 1), 1000.0)
    dy = jnp.full((n + 1, n), 1000.0)
    rarea = jnp.full((n, n), 1.0 / (1000.0 * 1000.0))
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert vort.shape == (n, n, km)


def test_vort_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=719)
    n, km = 12, 20
    u = jnp.asarray(rng.normal(scale=15.0, size=(n, n + 1, km)))
    v = jnp.asarray(rng.normal(scale=15.0, size=(n + 1, n, km)))
    dx = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n, n + 1)))
    dy = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n + 1, n)))
    rarea = jnp.full((n, n), 1.0 / (1000.0 * 1000.0))
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert jnp.all(jnp.isfinite(vort))
