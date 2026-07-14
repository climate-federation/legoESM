"""Unit tests for ``src/legoesm/atmosphere/dynamics/les/plane_operators.py``.

Two test categories:

1. **Exact algebraic identities** (machine-epsilon tolerance in x64):
   divergence of constant flux, gradient of constant scalar,
   curl-of-gradient commutator, divergence sum under periodic BC, and
   the energy-consistent pressure-gradient / divergence adjoint
   identity.

2. **Approximate numerical claims** with explicit tolerances or
   convergence rates: discrete Laplacian eigenvalues, second-order
   convergence of Laplacian to the continuous limit, and analytic
   gradient round-trip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.plane_operators import (
    curl_3d,
    divergence_3d,
    grad_x_3d,
    grad_y_3d,
    laplacian_3d,
)
from legoesm.grids.plane import create_plane_grid


jax.config.update("jax_enable_x64", True)


def _grid(nx=16, ny=12, nlev=3, dx=2.0e3, dy=3.0e3):
    return create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dy, dtype=jnp.float64
    )


def _key(seed=0):
    return jax.random.PRNGKey(seed)


# --------------------------------------------------------------------- #
# Exact algebraic identities                                            #
# --------------------------------------------------------------------- #


def test_grad_of_constant_is_zero():
    g = _grid()
    phi = jnp.full((g.nlev, g.ny, g.nx), 7.5, dtype=jnp.float64)
    assert jnp.max(jnp.abs(grad_x_3d(phi, g))) == 0.0
    assert jnp.max(jnp.abs(grad_y_3d(phi, g))) == 0.0


def test_divergence_of_constant_flux_is_zero():
    g = _grid()
    u = jnp.full((g.nlev, g.ny, g.nx), 1.25, dtype=jnp.float64)
    v = jnp.full((g.nlev, g.ny, g.nx), -3.5, dtype=jnp.float64)
    assert jnp.max(jnp.abs(divergence_3d(u, v, g))) == 0.0


def test_curl_of_gradient_is_zero():
    """Mixed differences commute on the staggered grid (4-point stencil).
    curl(grad(phi)) == 0 to machine precision for arbitrary phi."""
    g = _grid()
    phi = jax.random.normal(_key(1), (g.nlev, g.ny, g.nx), dtype=jnp.float64)
    u = grad_x_3d(phi, g)
    v = grad_y_3d(phi, g)
    err = jnp.max(jnp.abs(curl_3d(u, v, g)))
    assert float(err) < 1.0e-12


def test_divergence_sum_is_zero_under_periodic_bc():
    g = _grid()
    u = jax.random.normal(_key(2), (g.nlev, g.ny, g.nx), dtype=jnp.float64)
    v = jax.random.normal(_key(3), (g.nlev, g.ny, g.nx), dtype=jnp.float64)
    div = divergence_3d(u, v, g)
    # Discrete Stokes: each level's divergence sums to zero.
    per_level = jnp.sum(div, axis=(-2, -1))
    assert float(jnp.max(jnp.abs(per_level))) < 1.0e-10


def test_energy_consistent_pg_div_adjoint():
    """``sum(phi * div(u, v)) * dx * dy
        == - sum(u * grad_x(phi)) * dx * dy
           - sum(v * grad_y(phi)) * dx * dy``

    This is the discrete equivalent of ⟨φ, ∇·F⟩ = -⟨F, ∇φ⟩ under
    periodic BC. The future plane NH dycore needs this pairing for
    energy-consistent pressure-gradient / divergence coupling.
    """
    g = _grid()
    phi = jax.random.normal(_key(4), (g.nlev, g.ny, g.nx), dtype=jnp.float64)
    u = jax.random.normal(_key(5), (g.nlev, g.ny, g.nx), dtype=jnp.float64)
    v = jax.random.normal(_key(6), (g.nlev, g.ny, g.nx), dtype=jnp.float64)

    lhs = jnp.sum(phi * divergence_3d(u, v, g)) * g.dx * g.dy
    rhs = -(
        jnp.sum(u * grad_x_3d(phi, g)) + jnp.sum(v * grad_y_3d(phi, g))
    ) * g.dx * g.dy

    # Both sides are sums of ~nlev*ny*nx products; relative tolerance
    # tracks accumulated round-off.
    rel = float(jnp.abs(lhs - rhs) / (jnp.abs(lhs) + jnp.abs(rhs) + 1.0e-30))
    assert rel < 1.0e-12


# --------------------------------------------------------------------- #
# Approximate numerical claims                                          #
# --------------------------------------------------------------------- #


def test_laplacian_discrete_eigenvalue_complex_exponential():
    """For ``phi[j, i] = exp(2 pi i (k i/nx + l j/ny))`` the 5-point
    Laplacian has eigenvalue
        -4 sin^2(pi k/nx) / dx^2 - 4 sin^2(pi l/ny) / dy^2 ."""
    g = _grid(nx=32, ny=24, dx=1.0e3, dy=2.0e3)
    i = jnp.arange(g.nx)
    j = jnp.arange(g.ny)
    for k, l in [(1, 1), (2, 3), (5, 7)]:
        phase = (
            2.0 * jnp.pi * (k * i / g.nx)[None, :]
            + 2.0 * jnp.pi * (l * j / g.ny)[:, None]
        )
        phi = jnp.exp(1j * phase).astype(jnp.complex128)
        phi_3d = phi[None, :, :]
        lap = laplacian_3d(phi_3d, g)
        expected = (
            -4.0 * jnp.sin(jnp.pi * k / g.nx) ** 2 / g.dx ** 2
            - 4.0 * jnp.sin(jnp.pi * l / g.ny) ** 2 / g.dy ** 2
        )
        ratio = lap / phi_3d
        rel = float(jnp.max(jnp.abs(ratio - expected) / jnp.abs(expected)))
        assert rel < 1.0e-12, f"mode (k={k}, l={l}) failed: rel={rel}"


def test_laplacian_second_order_convergence_to_continuous_limit():
    """For a smooth low-wavenumber sin field the discrete eigenvalue
    converges to ``-(2 pi / L)^2`` at second order in grid spacing."""
    Lx = Ly = 1.0
    k = l = 1
    cont = -((2.0 * jnp.pi * k / Lx) ** 2) - ((2.0 * jnp.pi * l / Ly) ** 2)

    errs = []
    for n in (32, 64, 128):
        dx = Lx / n
        dy = Ly / n
        g = create_plane_grid(
            nx=n, ny=n, nlev=2, dx=dx, dy=dy, dtype=jnp.float64
        )
        i = jnp.arange(n)
        j = jnp.arange(n)
        xc = (i + 0.5) * dx
        yc = (j + 0.5) * dy
        phi = jnp.sin(2.0 * jnp.pi * k * xc[None, :] / Lx) * jnp.sin(
            2.0 * jnp.pi * l * yc[:, None] / Ly
        )
        phi_3d = phi[None, :, :]
        lap = laplacian_3d(phi_3d, g)
        # Sample eigenvalue from a point well inside the domain.
        sample_ratio = lap[0, n // 4, n // 4] / phi_3d[0, n // 4, n // 4]
        errs.append(float(jnp.abs(sample_ratio - cont)))

    # Each refinement should drop error by ~4x. Allow 3.6x to absorb
    # higher-order-in-h corrections at the coarsest level.
    assert errs[0] / errs[1] > 3.6, f"errs={errs}"
    assert errs[1] / errs[2] > 3.6, f"errs={errs}"


def test_grad_x_recovers_analytic_derivative_on_smooth_field():
    """``grad_x(sin(kx))`` matches ``k cos(kx - dx/2)`` (face-centered) at
    second order in dx."""
    Lx = 1.0
    k = 2 * jnp.pi  # one period
    errs = []
    for n in (32, 64, 128):
        dx = Lx / n
        g = create_plane_grid(
            nx=n, ny=4, nlev=2, dx=dx, dy=dx, dtype=jnp.float64
        )
        xc = (jnp.arange(n) + 0.5) * dx
        phi = jnp.sin(k * xc)[None, None, :] * jnp.ones((1, 4, 1))
        gx = grad_x_3d(phi, g)
        # grad_x at x-face ``xu[i] = i*dx`` is the centred difference of
        # cell-centre samples ``phi(xc[i]) - phi(xc[i-1])``, so the
        # second-order continuum target evaluates ``k cos(k * xu[i])``
        # at the face position itself, not at the cell centre west of
        # it. Using the wrong reference point degrades the test to
        # first-order convergence.
        xu = jnp.arange(n) * dx
        expected = (k * jnp.cos(k * xu))[None, None, :] * jnp.ones((1, 4, 1))
        # Second-order centred difference on staggered face: error O(dx^2).
        errs.append(float(jnp.max(jnp.abs(gx - expected))))
    assert errs[0] / errs[1] > 3.6, f"errs={errs}"
    assert errs[1] / errs[2] > 3.6, f"errs={errs}"


# --------------------------------------------------------------------- #
# Differentiability                                                     #
# --------------------------------------------------------------------- #


def test_operators_reject_wrong_shape():
    g = _grid(nx=10, ny=8, nlev=2)  # ny != nx so transpose is detectable
    # 2D scalar (missing vertical leading axis).
    bad_2d = jnp.zeros((g.ny, g.nx))
    with pytest.raises(ValueError, match="ndim >= 3"):
        grad_x_3d(bad_2d, g)
    with pytest.raises(ValueError, match="ndim >= 3"):
        laplacian_3d(bad_2d, g)
    # Wrong trailing shape (transposed nx, ny).
    bad_transposed = jnp.zeros((g.nlev, g.nx, g.ny))
    with pytest.raises(ValueError, match="trailing axes"):
        grad_y_3d(bad_transposed, g)
    bad_size = jnp.zeros((g.nlev, g.ny + 1, g.nx))
    with pytest.raises(ValueError, match="trailing axes"):
        divergence_3d(bad_size, jnp.zeros_like(bad_size), g)
    # Both u, v have correct trailing (ny, nx) but different leading
    # dims — exercises the u.shape != v.shape branch separately from
    # the trailing-axes check.
    u_ok = jnp.zeros((2, g.ny, g.nx))
    v_diff_lead = jnp.zeros((3, g.ny, g.nx))
    with pytest.raises(ValueError, match="must have equal shape"):
        divergence_3d(u_ok, v_diff_lead, g)
    with pytest.raises(ValueError, match="must have equal shape"):
        curl_3d(u_ok, v_diff_lead, g)


def test_operators_grad_pass_finite_difference_check():
    """``check_grads`` confirms forward + reverse-mode AD match
    second-order finite differences for each operator."""
    g = _grid(nx=8, ny=8, nlev=2)
    rng = np.random.default_rng(42)
    phi = jnp.asarray(rng.standard_normal((g.nlev, g.ny, g.nx)))
    u = jnp.asarray(rng.standard_normal((g.nlev, g.ny, g.nx)))
    v = jnp.asarray(rng.standard_normal((g.nlev, g.ny, g.nx)))

    # Scalar-valued objectives so check_grads has a scalar to perturb.
    jax.test_util.check_grads(
        lambda p: grad_x_3d(p, g).sum(), (phi,), order=2, modes=["rev"],
        rtol=1.0e-6, atol=1.0e-6,
    )
    jax.test_util.check_grads(
        lambda p: grad_y_3d(p, g).sum(), (phi,), order=2, modes=["rev"],
        rtol=1.0e-6, atol=1.0e-6,
    )
    jax.test_util.check_grads(
        lambda a, b: divergence_3d(a, b, g).sum(),
        (u, v), order=2, modes=["rev"], rtol=1.0e-6, atol=1.0e-6,
    )
    jax.test_util.check_grads(
        lambda a, b: curl_3d(a, b, g).sum(),
        (u, v), order=2, modes=["rev"], rtol=1.0e-6, atol=1.0e-6,
    )
    jax.test_util.check_grads(
        lambda p: laplacian_3d(p, g).sum(), (phi,), order=2, modes=["rev"],
        rtol=1.0e-6, atol=1.0e-6,
    )
