"""Single-rank gates for the distributed pseudo-incompressible Poisson solver.

At ``comm.Get_size() == 1`` the y-ring halo reduces to a periodic pad, so
every distributed entry point runs WITHOUT an mpirun launcher — which lets
these unit tests pin three contracts cheaply:

1. distributed solve == serial solver (same operator, same Krylov method);
2. the chunked masked-``fori_loop`` driver reproduces the serial answer at
   any chunk size (masking after convergence is a no-op);
3. :func:`solve_pressure_mpi_fixed_iters` is reverse-mode differentiable
   with finite gradients (the AD contract of the ``custom_vjp`` halo +
   ``allreduce(SUM)`` dots).

Multi-rank transport parity lives in
``tests/distributed/test_pseudo_incompressible_poisson_mpi.py``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("mpi4py.MPI")
pytest.importorskip("mpi4jax")
from legoesm.atmosphere.dynamics.les import (  # noqa: E402
    pseudo_incompressible_poisson as ser,
)
from legoesm.atmosphere.dynamics.les import (  # noqa: E402
    pseudo_incompressible_poisson_mpi as mpi,
)
from mpi4py import MPI  # noqa: E402

NY, NX, NZ = 8, 6, 5
DX, DY, DZ = 100.0, 120.0, 90.0


def _fields(dtype=jnp.float64):
    x = (jnp.arange(NX) + 0.5) * DX
    z = (jnp.arange(NZ) + 0.5) * DZ
    xg, zg = jnp.meshgrid(x, z, indexing="ij")
    c2d = 1.0 + 0.4 * jnp.cos(2 * np.pi * xg / (NX * DX)) * jnp.cos(
        np.pi * zg / (NZ * DZ))
    c = jnp.broadcast_to(c2d[None, :, :], (NY, NX, NZ)).astype(dtype)
    rhs = jax.random.normal(jax.random.PRNGKey(0), (NY, NX, NZ)).astype(dtype)
    return c, rhs


@pytest.fixture
def comm():
    c = MPI.COMM_WORLD
    if c.Get_size() != 1:
        pytest.skip("single-rank gate (multi-rank covered in tests/distributed)")
    return c


def test_laplacian_matches_serial(comm):
    c, rhs = _fields()
    lap_mpi = mpi.laplace_pi_mpi(rhs, c, DX, DY, DZ, comm)
    lap_ser = ser.laplace_pi(rhs, c, DX, DY, DZ)
    np.testing.assert_allclose(
        np.asarray(lap_mpi), np.asarray(lap_ser), rtol=0.0, atol=1e-12,
    )


def test_solve_matches_serial(comm):
    c, rhs = _fields()
    n_global = NY * NX * NZ
    pi_mpi = mpi.solve_pressure_mpi(
        rhs, c, DX, DY, DZ, comm, n_global, tol=1e-10, atol=1e-12,
        maxiter=400,
    )
    pi_ser = ser.solve_pressure(
        rhs, c, DX, DY, DZ, tol=1e-10, atol=1e-12, maxiter=400,
    )[0]
    a = np.asarray(pi_mpi) - float(np.mean(np.asarray(pi_mpi)))
    b = np.asarray(pi_ser) - float(np.mean(np.asarray(pi_ser)))
    assert np.max(np.abs(a - b)) < 1e-6


@pytest.mark.parametrize("check_every", [1, 3, 64])
def test_chunk_size_invariance(comm, check_every):
    """The masked fori/chunk driver must give the same solution at any
    host-check cadence — iterations past convergence are frozen no-ops."""
    c, rhs = _fields()
    n_global = NY * NX * NZ
    ref = mpi.solve_pressure_mpi(
        rhs, c, DX, DY, DZ, comm, n_global, tol=1e-10, atol=1e-12,
        maxiter=200, check_every=8,
    )
    got = mpi.solve_pressure_mpi(
        rhs, c, DX, DY, DZ, comm, n_global, tol=1e-10, atol=1e-12,
        maxiter=200, check_every=check_every,
    )
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=0.0, atol=1e-11,
    )


def test_fixed_iters_matches_forward(comm):
    c, rhs = _fields()
    n_global = NY * NX * NZ
    ref = mpi.solve_pressure_mpi(
        rhs, c, DX, DY, DZ, comm, n_global, tol=1e-10, atol=1e-12,
        maxiter=200,
    )
    got = mpi.solve_pressure_mpi_fixed_iters(
        rhs, c, DX, DY, DZ, comm, n_global, n_iters=200,
        tol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=0.0, atol=1e-9,
    )


def test_fixed_iters_grad_finite(comm):
    """Reverse-mode AD through the full solve: halo (custom_vjp sendrecv
    ring → serial pad at 1 rank) + batched allreduce dots + masked loop.
    The gradient must be finite even though the loop converges (and
    freezes) well before n_iters — the double-where guards keep the
    frozen iterations out of the cotangent."""
    c, rhs = _fields()
    n_global = NY * NX * NZ

    def loss(r):
        pi = mpi.solve_pressure_mpi_fixed_iters(
            r, c, DX, DY, DZ, comm, n_global, n_iters=40,
            tol=1e-8, atol=1e-10,
        )
        return jnp.sum(pi ** 2)

    g = jax.grad(loss)(rhs)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0

    # Directional finite-difference cross-check on the scalar loss.
    v = jax.random.normal(jax.random.PRNGKey(3), rhs.shape)
    v = v / jnp.linalg.norm(v)
    eps = 1e-5
    fd = (loss(rhs + eps * v) - loss(rhs - eps * v)) / (2 * eps)
    ad = jnp.vdot(g, v)
    np.testing.assert_allclose(float(ad), float(fd), rtol=1e-5, atol=1e-7)


def test_x0_already_converged_short_circuit(comm):
    """rhs == L(x0) with x0 zero-mean ⇒ the entry check exits immediately
    and returns x0 (up to the zero-mean gauge)."""
    c, _ = _fields()
    n_global = NY * NX * NZ
    x0 = jax.random.normal(jax.random.PRNGKey(1), (NY, NX, NZ))
    x0 = x0 - jnp.mean(x0)
    rhs = mpi.laplace_pi_mpi(x0, c, DX, DY, DZ, comm)
    pi = mpi.solve_pressure_mpi(
        rhs, c, DX, DY, DZ, comm, n_global, x0=x0, tol=1e-8, atol=1e-10,
        maxiter=50,
    )
    np.testing.assert_allclose(
        np.asarray(pi), np.asarray(x0), rtol=0.0, atol=1e-7,
    )
