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
        """Gradient of quadratic loss through global_sum_mpi is nonzero.

        FV3_3D iter-1061: contract clarification.  mpi4jax 0.8/0.9's
        ``allreduce(SUM)`` VJP is identity passthrough: each rank
        receives the upstream cotangent unchanged, so the per-rank
        gradient of ``allreduce_sum(sum(x_local^2))`` w.r.t. ``x_local``
        is ``2*x_local`` — regardless of ``world_size``.  The previous
        assertion ``2*x*world_size`` confused two different semantics:

        - Rank-local view: each rank has its own ``x``; the per-rank
          gradient is ``2*x``.  Summing those gradients across ranks
          (via an explicit second allreduce) would give ``2*x*world_size``,
          which is the "single-shared-x" gradient.

        - The implementation does not insert that second allreduce.
          Callers who treat ``x`` as a shared replicated variable must
          allreduce(SUM) the gradient themselves after ``jax.grad``.

        The "scaled by world_size" comment in the original test was
        documenting the SHARED-x interpretation, but the assertion ran
        on each rank without doing the post-grad allreduce — so the
        test was never well-defined.  Fixed to assert the per-rank
        gradient ``2*x`` (no scaling).  See iter-1061 for the
        complete contract discussion.
        """
        def f(x):
            s = global_sum_mpi(jnp.sum(x ** 2))
            return s

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        g = jax.grad(f)(x)
        assert jnp.all(jnp.isfinite(g))
        assert not jnp.allclose(g, 0.0)
        # d/dx sum(x_local^2) = 2*x_local; allreduce VJP is identity.
        np.testing.assert_allclose(g, 2.0 * x, atol=1e-12)


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
        """MPI gradient matches local-backend gradient on locally-owned faces.

        FV3_3D iter-1061: contract clarification.  With replicated
        ``(6, n, n)`` input on every rank, the MPI ``pad_halo``
        forward fills each rank's local-face interior from its own
        data and the cross-face halos from neighboring ranks via
        ``mpi4jax.sendrecv``.  The backward VJP propagates cotangents
        through ``sendrecv`` (via ``_sendrecv_vjp`` custom_vjp), but
        does not allreduce the upstream cotangent — each rank's
        gradient at a given face position is correct ONLY for the
        faces it owns.

        Non-owned faces' gradients miss the cross-face halo
        contributions that the local-backend would have computed
        across ALL 6 faces locally.  The previous test compared the
        full (6, n, n) gradient between MPI and local on rank 0,
        which fails for faces ∈ ``{3, 4, 5}`` (rank 0 owns 0-2 at
        np=2).

        Fixed to compare only ``topology.local_face_ids`` on each
        rank — matches the standard MPI bit-for-bit pattern from the
        FV3 step-fidelity tests.
        """
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

        # Compare only on locally-owned faces.
        for f in topology.local_face_ids:
            np.testing.assert_allclose(
                np.asarray(g_mpi[f]), np.asarray(g_local[f]),
                atol=1e-12,
                err_msg=(
                    f"MPI halo gradient on owned face {f} (rank {rank}) "
                    f"does not match local-backend gradient"
                ),
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
        """4D MPI gradient matches local-backend gradient on owned faces.

        FV3_3D iter-1061: see sibling 2D test for the contract.
        """
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

        for f in topology.local_face_ids:
            np.testing.assert_allclose(
                np.asarray(g_mpi[f]), np.asarray(g_local[f]),
                atol=1e-12,
                err_msg=(
                    f"4D MPI halo gradient on owned face {f} (rank {rank}) "
                    f"does not match local-backend gradient"
                ),
            )

    def test_pad_halo_mpi_grad_allreduce_recovers_full(self, topology):
        """FV3_3D iter-1062 (codex iter-1061 WARN #1): allreduce of the
        per-rank MPI gradient must recover the full local-backend
        gradient on ALL 6 faces.

        Without this stronger test, a bug in ``_sendrecv_vjp`` that
        zeros out non-owned-face cotangents (or fails to propagate
        them via MPI) would pass ``test_pad_halo_mpi_grad_matches_local``
        (which only inspects owned faces) but corrupt the canonical
        replicated-state data-parallel training pattern where the
        full gradient is reconstructed via ``allreduce(SUM)`` at the
        end.
        """
        from legoesm.parallel.reductions import global_sum_mpi

        n = 8
        key = jax.random.PRNGKey(2026)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float64)

        def loss(d):
            padded = pad_halo(d)
            return jnp.sum(padded ** 2)

        set_halo_backend("local")
        g_local = jax.grad(loss)(data)

        set_halo_backend("mpi", topology)
        g_mpi = jax.grad(loss)(data)

        # Mask each rank's gradient to its owned faces only, then
        # allreduce(SUM) to assemble the full (6, n, n) gradient.
        owned_mask = jnp.zeros((6, 1, 1), dtype=g_mpi.dtype)
        for f in topology.local_face_ids:
            owned_mask = owned_mask.at[f].set(1.0)
        g_mpi_masked = g_mpi * owned_mask
        g_mpi_full = global_sum_mpi(g_mpi_masked)

        np.testing.assert_allclose(
            np.asarray(g_mpi_full), np.asarray(g_local),
            atol=1e-12,
            err_msg=(
                "Allreduced per-rank MPI gradient does not recover the "
                "full local-backend gradient.  Likely cause: "
                "_sendrecv_vjp drops or mis-routes non-owned-face "
                "cotangents."
            ),
        )

    def test_pad_halo_4d_mpi_grad_allreduce_recovers_full(self, topology):
        """FV3_3D iter-1062 (codex iter-1061 WARN #1): 4D allreduce
        check for full-gradient recovery.  Mirror of the 2D test.
        """
        from legoesm.parallel.reductions import global_sum_mpi

        n, nlev = 8, 3
        key = jax.random.PRNGKey(1234)
        data = jax.random.normal(key, (6, n, n, nlev), dtype=jnp.float64)

        def loss(d):
            padded = pad_halo_4d(d)
            return jnp.sum(padded ** 2)

        set_halo_backend("local")
        g_local = jax.grad(loss)(data)

        set_halo_backend("mpi", topology)
        g_mpi = jax.grad(loss)(data)

        owned_mask = jnp.zeros((6, 1, 1, 1), dtype=g_mpi.dtype)
        for f in topology.local_face_ids:
            owned_mask = owned_mask.at[f].set(1.0)
        g_mpi_masked = g_mpi * owned_mask
        g_mpi_full = global_sum_mpi(g_mpi_masked)

        np.testing.assert_allclose(
            np.asarray(g_mpi_full), np.asarray(g_local),
            atol=1e-12,
            err_msg=(
                "Allreduced per-rank 4D MPI gradient does not recover "
                "the full local-backend gradient."
            ),
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
