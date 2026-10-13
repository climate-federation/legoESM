"""Direct test of barotropic_common._fixed_iteration_chebyshev on a small
area-weighted Helmholtz system (single process, trivial halo)."""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.barotropic_common import _fixed_iteration_chebyshev


def _system(n=40, c=50.0, seed=0):
    """A = I + c W^-1 L, L a weighted 1-D Neumann Laplacian: W-self-adjoint,
    off-diagonal row sum = diag - 1 (the MPAS Helmholtz structure)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, n)
    k = rng.uniform(0.2, 1.0, n - 1)
    L = np.zeros((n, n))
    for i, ki in enumerate(k):
        L[i, i] += ki; L[i + 1, i + 1] += ki; L[i, i + 1] -= ki; L[i + 1, i] -= ki
    A = np.eye(n) + c * L / w[:, None]
    return A, w, rng.standard_normal(n)


def test_chebyshev_converges_to_direct_solve():
    # c = 5: D^-1 A window ratio ~40, contraction ~0.73 per update, so 150
    # updates reach round-off (c = 50 gives ~400 and needs ~300 updates).
    A, w, b = _system(c=5.0)
    Aj, W = jnp.asarray(A), jnp.asarray(w)
    inv_d = jnp.asarray(1.0 / np.diag(A))
    x, rr = _fixed_iteration_chebyshev(
        lambda v: Aj @ v, jnp.asarray(b), lambda r: inv_d * r, jnp.zeros_like(W),
        max_iter=150, dot_weight=W, deep_halo=(lambda *f: f, jnp.ones_like(W), 1))
    ref = np.linalg.solve(A, b)
    np.testing.assert_allclose(np.asarray(x), ref, rtol=1e-10, atol=1e-10)
    assert float(rr) < 1e-18 * float(np.sum(w * b * b))


def test_chebyshev_error_drops_with_updates_and_cadence_is_exact():
    """Error falls with the update count, and the update count does not
    depend on the exchange cadence (the exchange is the identity here; the
    halo cadence itself is tested on the SPMD lane)."""
    A, w, b = _system(seed=1)
    Aj, W = jnp.asarray(A), jnp.asarray(w)
    inv_d = jnp.asarray(1.0 / np.diag(A))
    ref = np.linalg.solve(A, b)

    def err(m, rings):
        x, _ = _fixed_iteration_chebyshev(
            lambda v: Aj @ v, jnp.asarray(b), lambda r: inv_d * r, jnp.zeros_like(W),
            max_iter=m, dot_weight=W, deep_halo=(lambda *f: f, jnp.ones_like(W), rings))
        return np.max(np.abs(np.asarray(x) - ref)), np.asarray(x)

    e = [err(m, 1)[0] for m in (5, 20, 60)]
    assert e[0] > e[1] > e[2]
    np.testing.assert_allclose(err(37, 3)[1], err(37, 1)[1], rtol=0, atol=1e-14)


def test_returned_residual_is_that_of_returned_x_and_identity_is_safe():
    A, w, b = _system(c=5.0, seed=2)
    Aj, W = jnp.asarray(A), jnp.asarray(w)
    inv_d = jnp.asarray(1.0 / np.diag(A))
    x, rr = _fixed_iteration_chebyshev(
        lambda v: Aj @ v, jnp.asarray(b), lambda r: inv_d * r, jnp.zeros_like(W),
        max_iter=7, dot_weight=W, deep_halo=(lambda *f: f, jnp.ones_like(W), 1))
    r = b - A @ np.asarray(x)
    np.testing.assert_allclose(float(rr), float(np.sum(w * r * r)), rtol=1e-10)
    eye = jnp.eye(len(w))
    x, rr = _fixed_iteration_chebyshev(
        lambda v: eye @ v, jnp.asarray(b), lambda r: r, jnp.zeros_like(W),
        max_iter=5, dot_weight=W, deep_halo=(lambda *f: f, jnp.ones_like(W), 1))
    np.testing.assert_allclose(np.asarray(x), b, rtol=1e-12)
