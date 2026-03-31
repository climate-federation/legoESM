"""MPI reverse-mode AD (gradient) tests.

Run with:
    mpirun -np 2 python -m pytest tests/distributed/test_mpi_differentiability.py -v
    mpirun -np 3 python -m pytest tests/distributed/test_mpi_differentiability.py -v
    mpirun -np 6 python -m pytest tests/distributed/test_mpi_differentiability.py -v

Verifies that ``jax.grad`` flows correctly through:
- ``global_sum_mpi`` (allreduce with SUM)
- ``pad_halo`` with MPI backend (sendrecv with custom_vjp)
- ``pad_halo_4d`` with MPI backend
- Conservation fixers (``fix_mass_hydrostatic``) under MPI

The single-process equivalents are already tested in:
- tests/unit/test_halo.py (test_grad_compatible)
- tests/unit/test_scale_global_reductions.py (test_global_integral_differentiable)
- tests/validation/test_differentiability_regression.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.parallel.comm import build_comm_topology
from legoesm.parallel.reductions import global_sum_mpi
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    pad_halo,
    pad_halo_4d,
    set_halo_backend,
)


@pytest.fixture(autouse=True)
def reset_halo_backend():
    """Reset halo backend to local after each test."""
    yield
    set_halo_backend("local")


@pytest.fixture
def topology():
    rank = MPI.COMM_WORLD.Get_rank()
    n_processes = MPI.COMM_WORLD.Get_size()
    return build_comm_topology(rank, n_processes)


class TestGlobalSumMPIGrad:
    """Gradient through allreduce(SUM)."""

    def test_grad_scalar(self):
        """Gradient of global_sum_mpi should be all-ones (like jnp.sum)."""
        def f(x):
            return global_sum_mpi(jnp.sum(x))

        x = jnp.ones(5, dtype=jnp.float64)
        g = jax.grad(f)(x)
        np.testing.assert_allclose(g, jnp.ones(5), atol=1e-12)

    def test_grad_nonzero(self):
        """Gradient of quadratic loss through global_sum_mpi is nonzero."""
        def f(x):
            s = global_sum_mpi(jnp.sum(x ** 2))
            return s

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        g = jax.grad(f)(x)
        assert jnp.all(jnp.isfinite(g))
        assert not jnp.allclose(g, 0.0)
        # d/dx sum(x^2) = 2*x, scaled by world_size from allreduce
        world_size = MPI.COMM_WORLD.Get_size()
        np.testing.assert_allclose(g, 2.0 * x * world_size, atol=1e-12)


class TestPadHaloMPIGrad:
    """Gradient through MPI halo exchange (sendrecv with custom_vjp)."""

    def test_pad_halo_mpi_grad_finite(self, topology):
        """jax.grad through pad_halo with MPI backend produces finite results."""
        n = 8
        set_halo_backend("mpi", topology)

        def loss(data):
            padded = pad_halo(data)
            return jnp.sum(padded ** 2)

        data = jnp.ones((6, n, n), dtype=jnp.float64)
        g = jax.grad(loss)(data)
        assert g.shape == (6, n, n)
        assert jnp.all(jnp.isfinite(g))
        assert not jnp.allclose(g, 0.0)

    def test_pad_halo_mpi_grad_matches_local(self, topology):
        """MPI gradient matches local-backend gradient on rank 0."""
        rank = MPI.COMM_WORLD.Get_rank()
        n = 8

        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float64)

        def loss(d):
            padded = pad_halo(d)
            return jnp.sum(padded ** 2)

        # Local gradient (reference)
        set_halo_backend("local")
        g_local = jax.grad(loss)(data)

        # MPI gradient
        set_halo_backend("mpi", topology)
        g_mpi = jax.grad(loss)(data)

        if rank == 0:
            np.testing.assert_allclose(
                np.asarray(g_mpi), np.asarray(g_local),
                atol=1e-12,
                err_msg="MPI halo gradient does not match local gradient",
            )

    def test_pad_halo_4d_mpi_grad_finite(self, topology):
        """jax.grad through pad_halo_4d with MPI backend produces finite results."""
        n, nlev = 8, 5
        set_halo_backend("mpi", topology)

        def loss(data):
            padded = pad_halo_4d(data)
            return jnp.sum(padded ** 2)

        data = jnp.ones((6, n, n, nlev), dtype=jnp.float64)
        g = jax.grad(loss)(data)
        assert g.shape == (6, n, n, nlev)
        assert jnp.all(jnp.isfinite(g))
        assert not jnp.allclose(g, 0.0)

    def test_pad_halo_4d_mpi_grad_matches_local(self, topology):
        """4D MPI gradient matches local-backend gradient on rank 0."""
        rank = MPI.COMM_WORLD.Get_rank()
        n, nlev = 8, 3

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (6, n, n, nlev), dtype=jnp.float64)

        def loss(d):
            padded = pad_halo_4d(d)
            return jnp.sum(padded ** 2)

        set_halo_backend("local")
        g_local = jax.grad(loss)(data)

        set_halo_backend("mpi", topology)
        g_mpi = jax.grad(loss)(data)

        if rank == 0:
            np.testing.assert_allclose(
                np.asarray(g_mpi), np.asarray(g_local),
                atol=1e-12,
                err_msg="4D MPI halo gradient does not match local gradient",
            )


class TestConservationFixerMPIGrad:
    """Gradient through conservation fixers with MPI reductions."""

    def test_fix_mass_mpi_grad(self, topology):
        """jax.grad flows through fix_mass_hydrostatic with MPI backend."""
        set_halo_backend("mpi", topology)
        grid = create_cubed_sphere(8)

        from legoesm.core.conservation import fix_ps_mass_target

        target_mass = jnp.float64(1e5) * grid.total_area

        def loss(p_s):
            p_s_fixed = fix_ps_mass_target(p_s, target_mass, grid)
            return jnp.sum(p_s_fixed ** 2)

        p_s = jnp.ones((6, 8, 8), dtype=jnp.float64) * 1e5
        g = jax.grad(loss)(p_s)
        assert g.shape == (6, 8, 8)
        assert jnp.all(jnp.isfinite(g))
        assert not jnp.allclose(g, 0.0)
