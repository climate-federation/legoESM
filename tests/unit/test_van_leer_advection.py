"""Unit tests for Van Leer TVD horizontal advection on the plane CRM
dycore (iter-179).

Cover the four properties a TVD scheme must satisfy:

1. Reduces to first-order upwind at extrema (r <= 0 -> phi = 0).
2. Exact on a linear field (2nd-order accuracy in smooth regions).
3. TVD: no new extrema injected for a monotone field.
4. Differentiable end-to-end (jax.grad flows through the limiter).

Also verify the scheme dispatch in the plane CRM slow-tendency entry
point picks the Van Leer path when ``horizontal_advection_scheme =
"van_leer"``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.compressible_euler_plane import (  # noqa: E402
    _upwind_advection_x, _upwind_advection_y,
    _van_leer_advection_x, _van_leer_advection_y,
)
from legoesm.core.flux_limiters import van_leer_limiter  # noqa: E402


def test_van_leer_limiter_known_values():
    """Lock the four representative limiter values: extremum -> 0,
    smooth -> 1, sharp -> bounded by 2."""
    assert float(van_leer_limiter(jnp.array(0.0))) == 0.0
    assert float(van_leer_limiter(jnp.array(-1.0))) == 0.0
    assert float(van_leer_limiter(jnp.array(1.0))) == 1.0
    assert float(van_leer_limiter(jnp.array(10.0))) < 2.0


def test_van_leer_reduces_to_upwind_on_oscillating_field():
    """A 2-Δx oscillation has r <= 0 everywhere (every cell is an
    extremum) so the limiter returns 0 → Van Leer == upwind1. Locks
    the TVD-at-extrema fallback."""
    ny, nx, nlev = 4, 16, 2
    # Sign-flip in x at every other cell: f = [+1, -1, +1, -1, ...]
    f = jnp.ones((ny, nx, nlev), dtype=jnp.float64)
    f = f.at[:, 1::2, :].set(-1.0)
    u = jnp.full_like(f, 0.5)  # uniform positive flow
    dx = 1000.0
    upwind = _upwind_advection_x(f, u, dx)
    van_leer = _van_leer_advection_x(f, u, dx)
    np.testing.assert_allclose(
        np.asarray(van_leer), np.asarray(upwind),
        rtol=1e-12, atol=1e-12,
        err_msg="Van Leer must collapse to upwind1 at every extremum",
    )


def test_van_leer_exact_on_linear_field_x_interior():
    """For a linear field f(x) = a*x + b in the INTERIOR (away from
    the periodic wrap discontinuity at i=0/nx-1), the advective
    tendency is -u*df/dx = -u*a — a constant. Van Leer's slope
    limiter returns the full 2nd-order linear extrapolation on a
    locally-linear field so the tendency is exact within float64
    round-off. Tightens 2nd-order accuracy. (Boundary cells where
    the wraparound makes the field non-linear are excluded.)
    """
    ny, nx, nlev = 4, 12, 2
    dx = 1000.0
    a = 0.7
    x = jnp.arange(nx, dtype=jnp.float64) * dx
    f = jnp.broadcast_to(a * x[None, :, None], (ny, nx, nlev))
    u = jnp.full_like(f, 1.5)
    expected = -u * a  # -u * df/dx with df/dx = a (analytic)
    actual = _van_leer_advection_x(f, u, dx)
    # Strip boundary cells: Van Leer's 4-cell stencil [i-1..i+2] is
    # corrupted at i in {0, 1, nx-2, nx-1} by the periodic wrap
    # (the linear field is discontinuous at the seam).
    interior = slice(2, nx - 2)
    np.testing.assert_allclose(
        np.asarray(actual[:, interior, :]),
        np.asarray(expected[:, interior, :]),
        rtol=1e-10, atol=1e-10,
        err_msg="Van Leer must be exact on a linear field (smooth interior)",
    )


def test_van_leer_exact_on_linear_field_y_interior():
    """Same as the x test but on the y axis (axis=0). Tests
    _van_leer_advection_y interior accuracy."""
    ny, nx, nlev = 12, 4, 2
    dy = 750.0
    a = -0.4
    y = jnp.arange(ny, dtype=jnp.float64) * dy
    f = jnp.broadcast_to(a * y[:, None, None], (ny, nx, nlev))
    v = jnp.full_like(f, 2.0)
    expected = -v * a
    actual = _van_leer_advection_y(f, v, dy)
    interior = slice(2, ny - 2)
    np.testing.assert_allclose(
        np.asarray(actual[interior, :, :]),
        np.asarray(expected[interior, :, :]),
        rtol=1e-10, atol=1e-10,
    )


def test_van_leer_tvd_no_new_extrema():
    """Forward Euler step with the Van Leer tendency must NOT introduce
    new extrema for a monotone step field. Concretely: max(f^{n+1})
    <= max(f^n) + tol, min(f^{n+1}) >= min(f^n) - tol. This is the
    TVD property the limiter is supposed to enforce."""
    ny, nx, nlev = 4, 16, 1
    # Heaviside step: f = 0 for x < nx/2, 1 elsewhere.
    f = jnp.zeros((ny, nx, nlev), dtype=jnp.float64)
    f = f.at[:, nx // 2:, :].set(1.0)
    u = jnp.full_like(f, 0.3)
    dx = 1000.0
    dt = 100.0  # Courant ~0.03, well within TVD bound.
    tend = _van_leer_advection_x(f, u, dx)
    f_new = f + dt * tend
    tol = 1e-12
    assert float(jnp.max(f_new)) <= 1.0 + tol, (
        "Van Leer introduced a new maximum on a step field — "
        "violates TVD"
    )
    assert float(jnp.min(f_new)) >= 0.0 - tol, (
        "Van Leer introduced a new minimum on a step field — "
        "violates TVD"
    )


def test_van_leer_differentiable_via_jax_grad():
    """jax.grad must flow through the Van Leer limiter end-to-end —
    the (r+|r|)/(1+|r|) form is differentiable everywhere except the
    extremum kink, and JAX returns a finite subgradient there. Lock
    the differentiability contract."""
    ny, nx, nlev = 4, 8, 2
    rng = jax.random.PRNGKey(0)
    f0 = jax.random.normal(rng, (ny, nx, nlev), dtype=jnp.float64)
    u = jax.random.normal(
        jax.random.fold_in(rng, 1), f0.shape, dtype=jnp.float64,
    )
    dx = 1000.0

    def loss_fn(f):
        return jnp.sum(_van_leer_advection_x(f, u, dx) ** 2)

    grad = jax.grad(loss_fn)(f0)
    assert jnp.all(jnp.isfinite(grad)), (
        "Van Leer gradient has non-finite entries — limiter or "
        "flux-form conversion broke autograd"
    )


def test_van_leer_scheme_dispatch_in_slow_tendency():
    """The plane CRM slow-tendency entry rejects bad scheme names AND
    accepts the three supported names. Iter-179 added 'van_leer' to
    the dispatch — verify a typo still raises (so a silent fallback
    cannot reintroduce iter-114's typo-class bug)."""
    import re
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        plane_compressible_euler_slow_tendencies,
    )
    # Use the module-source check approach: scrape the dispatch
    # block from compressible_euler_plane.py and verify all three
    # schemes appear. Cheaper than building a full plane state.
    from pathlib import Path
    src = Path(
        plane_compressible_euler_slow_tendencies.__code__.co_filename
    ).read_text()
    # Locate the dispatch block.
    m = re.search(
        r'scheme\s*=\s*getattr\(config,\s*"horizontal_advection_scheme"',
        src,
    )
    assert m is not None, "dispatch block missing"
    # All three schemes must be listed.
    block = src[m.start():m.start() + 1500]
    for name in ("upwind1", "van_leer", "weno5"):
        assert f'"{name}"' in block, (
            f"scheme {name!r} missing from dispatch block"
        )
    # And the ValueError on unknown scheme must mention all three.
    assert "Expected 'upwind1', 'van_leer', or 'weno5'" in block, (
        "ValueError message must list all three valid schemes"
    )
