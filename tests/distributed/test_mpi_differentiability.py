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
- The Voronoi/MPAS halo exchange on BOTH state-exchange paths: the
  DEFAULT ``batched_halo_exchange`` union-neighbor path and the opt-OUT
  per-entity path (``VoronoiHaloExchange`` -> ``_exchange_mpi``) (both
  sendrecv with custom_vjp; per-neighbor send/recv counts differ,
  exercising the asymmetric-shape backward template)

The single-process equivalents are already tested in:
- tests/unit/test_halo.py (test_grad_compatible)
- tests/unit/test_scale_global_reductions.py (test_global_integral_differentiable)
- tests/validation/test_differentiability_regression.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import NamedTuple

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
        """Gradient of a final-loss global_sum_mpi is all-ones (like jnp.sum)."""
        def f(x):
            return global_sum_mpi(jnp.sum(x), final_loss=True)

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
            s = global_sum_mpi(jnp.sum(x ** 2), final_loss=True)
            return s

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        g = jax.grad(f)(x)
        assert jnp.all(jnp.isfinite(g))
        assert not jnp.allclose(g, 0.0)
        # d/dx sum(x_local^2) = 2*x_local; allreduce VJP is identity.
        np.testing.assert_allclose(g, 2.0 * x, atol=1e-12)


class TestNormalisedFinalLoss:
    """#1814: a global MEAN that every rank evaluates as the loss also needs
    final_loss=True; the default backward would scale it by the rank count."""

    @pytest.fixture(autouse=True)
    def _two_ranks(self):
        if MPI.COMM_WORLD.Get_size() < 2:
            pytest.skip("needs >= 2 MPI ranks")

    def test_global_mean_loss_matches_serial(self):
        comm = MPI.COMM_WORLD
        n, rank = comm.Get_size(), comm.Get_rank()
        x = jnp.arange(3.0, dtype=jnp.float64) + 10.0 * rank + 1.0

        def loss(x):
            return global_sum_mpi(jnp.sum(x ** 2), final_loss=True) / n

        g = jax.grad(loss)(x)
        np.testing.assert_allclose(np.asarray(g), np.asarray(2.0 * x / n),
                                   rtol=1e-12)


class TestReusedSumGrad:
    """#1814: a sum that every rank holds and reuses needs the allreduce
    backward pass.  Each rank differentiates its OWN local loss; the gradient
    must equal the matching row of the serial gradient of the total loss.
    With mpi4jax's identity backward pass these fail for >= 2 ranks."""

    @pytest.fixture(autouse=True)
    def _two_ranks(self):
        if MPI.COMM_WORLD.Get_size() < 2:
            pytest.skip("needs >= 2 MPI ranks")

    @staticmethod
    def _x(rank):
        return jnp.arange(3.0, dtype=jnp.float64) + 10.0 * rank + 1.0

    def _serial_row(self, total_loss):
        comm = MPI.COMM_WORLD
        xs = jnp.stack([self._x(r) for r in range(comm.Get_size())])
        return np.asarray(jax.grad(total_loss)(xs))[comm.Get_rank()]

    def test_reused_global_sum(self):
        def local_loss(x):
            s = global_sum_mpi(jnp.sum(x))
            return jnp.sum(jnp.sin(x) * s)

        def total(xs):
            return jnp.sum(jnp.sin(xs) * jnp.sum(xs))

        g = jax.grad(local_loss)(self._x(MPI.COMM_WORLD.Get_rank()))
        np.testing.assert_allclose(np.asarray(g), self._serial_row(total),
                                   rtol=1e-12)

    def test_reused_batch_allreduce(self):
        from legoesm.parallel.reductions import batch_allreduce_mpi

        def local_loss(x):
            s1, s2 = batch_allreduce_mpi([jnp.sum(x), jnp.sum(x ** 2)])
            return jnp.sum(jnp.sin(x) * s1 + x * s2)

        def total(xs):
            return jnp.sum(jnp.sin(xs) * jnp.sum(xs) + xs * jnp.sum(xs ** 2))

        g = jax.grad(local_loss)(self._x(MPI.COMM_WORLD.Get_rank()))
        np.testing.assert_allclose(np.asarray(g), self._serial_row(total),
                                   rtol=1e-12)

    def test_backward_uses_the_sub_communicator(self):
        """Singleton sub-communicators: the backward allreduce must use
        ``comm`` too, or it would mix in the other ranks' cotangents."""
        world = MPI.COMM_WORLD
        sub = world.Split(color=world.Get_rank(), key=0)
        try:
            def local_loss(x):
                s = global_sum_mpi(jnp.sum(x), comm=sub)
                return jnp.sum(jnp.sin(x) * s)

            x = self._x(world.Get_rank())
            g = jax.grad(local_loss)(x)
            want = jnp.cos(x) * jnp.sum(x) + jnp.sum(jnp.sin(x))
            np.testing.assert_allclose(np.asarray(g), np.asarray(want),
                                       rtol=1e-12)
        finally:
            sub.Free()

    def test_rank_that_discards_the_sum_still_joins_backward(self):
        """Only rank 0 uses the sum; the others discard it.  Their cotangent
        for it is a symbolic zero, but they must still enter the backward
        allreduce, or rank 0 deadlocks / misses nothing it needs."""
        rank = MPI.COMM_WORLD.Get_rank()

        def local_loss(x):
            s = global_sum_mpi(jnp.sum(x))
            if rank == 0:
                return jnp.sum(jnp.sin(x) * s)
            return jnp.sum(jnp.cos(x))

        def total(xs):
            return (jnp.sum(jnp.sin(xs[0]) * jnp.sum(xs))
                    + jnp.sum(jnp.cos(xs[1:])))

        g = jax.grad(local_loss)(self._x(rank))
        np.testing.assert_allclose(np.asarray(g), self._serial_row(total),
                                   rtol=1e-12)

    def test_reused_sum_inside_jit_scan(self):
        def step(c, s):
            return 0.5 * c + jnp.sin(c) * s / 100.0

        def local_loss(x):
            def body(c, _):
                return step(c, global_sum_mpi(jnp.sum(c))), None
            c, _ = jax.lax.scan(body, x, None, length=3)
            return jnp.sum(c ** 2)

        def total(xs):
            def body(c, _):
                return step(c, jnp.sum(c)), None
            c, _ = jax.lax.scan(body, xs, None, length=3)
            return jnp.sum(c ** 2)

        g = jax.jit(jax.grad(local_loss))(self._x(MPI.COMM_WORLD.Get_rank()))
        np.testing.assert_allclose(np.asarray(g), self._serial_row(total),
                                   rtol=1e-12)


class TestForwardMode:
    """#1814 follow-up: forward mode through the MPI global sum.  The tangent of
    S = sum_r x_r is sum_r v_r, so each rank's JVP must equal the matching row
    of the serial JVP with every rank's tangent stacked."""

    @pytest.fixture(autouse=True)
    def _two_ranks(self):
        if MPI.COMM_WORLD.Get_size() < 2:
            pytest.skip("needs >= 2 MPI ranks")

    @staticmethod
    def _xv(rank):
        x = jnp.arange(3.0, dtype=jnp.float64) + 10.0 * rank + 1.0
        return x, jnp.cos(x) + rank

    def _serial_jvp_row(self, total_rows):
        comm = MPI.COMM_WORLD
        xs, vs = (jnp.stack(a) for a in
                  zip(*(self._xv(r) for r in range(comm.Get_size()))))
        return np.asarray(jax.jvp(total_rows, (xs,), (vs,))[1])[comm.Get_rank()]

    @pytest.mark.parametrize("jit", [False, True])
    def test_jvp_global_sum_matches_serial(self, jit):
        def local(x):
            return jnp.sin(x) * global_sum_mpi(jnp.sum(x ** 2))

        def rows(xs):
            return jnp.sin(xs) * jnp.sum(xs ** 2)

        x, v = self._xv(MPI.COMM_WORLD.Get_rank())
        fn = jax.jit(local) if jit else local
        t = jax.jvp(fn, (x,), (v,))[1]
        np.testing.assert_allclose(np.asarray(t), self._serial_jvp_row(rows),
                                   rtol=1e-12)

    def test_jvp_batch_allreduce_matches_serial(self):
        from legoesm.parallel.reductions import batch_allreduce_mpi

        def local(x):
            s1, s2 = batch_allreduce_mpi([jnp.sum(x), jnp.sum(x ** 2)])
            return jnp.sin(x) * s1 + x * s2

        def rows(xs):
            return jnp.sin(xs) * jnp.sum(xs) + xs * jnp.sum(xs ** 2)

        x, v = self._xv(MPI.COMM_WORLD.Get_rank())
        t = jax.jvp(local, (x,), (v,))[1]
        np.testing.assert_allclose(np.asarray(t), self._serial_jvp_row(rows),
                                   rtol=1e-12)

    def test_jvp_final_loss_matches_serial(self):
        def local(x):
            return global_sum_mpi(jnp.sum(x ** 2), final_loss=True)

        def rows(xs):
            return jnp.broadcast_to(jnp.sum(xs ** 2), xs.shape[:1])

        x, v = self._xv(MPI.COMM_WORLD.Get_rank())
        t = jax.jvp(local, (x,), (v,))[1]
        np.testing.assert_allclose(float(t), self._serial_jvp_row(rows),
                                   rtol=1e-12)

    def test_check_grads_fwd_and_rev_order2(self):
        """Inputs identical on every rank (check_grads draws its direction from
        a fixed seed, so all ranks perturb alike); then the local JVP/VJP
        identity <ct, J v> = <J^T ct, v> holds rank by rank."""
        from jax.test_util import check_grads

        def local(x):
            return jnp.sin(x) * global_sum_mpi(jnp.sum(x ** 2))

        check_grads(local, (jnp.arange(1.0, 4.0, dtype=jnp.float64),),
                    order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)

    def test_check_grads_final_loss_fwd_order2(self):
        """final_loss keeps the per-rank identity backward by convention, so
        only forward mode has a finite-difference reference."""
        from jax.test_util import check_grads

        check_grads(lambda x: global_sum_mpi(jnp.sum(x ** 3), final_loss=True),
                    (jnp.arange(1.0, 4.0, dtype=jnp.float64),),
                    order=2, modes=("fwd",), atol=1e-6, rtol=1e-6)


class TestFixerGradScatteredMatchesSerial:
    """#1814, measured: on face-scattered cube MPI (each rank owns some faces),
    each rank's gradient of its OWN local loss through a mass fixer must equal
    the serial gradient on those faces.  The fixer's global sum is reused by
    every rank (the correction is added to every cell), so an identity
    backward pass misses the other ranks' terms."""

    N = 4

    class _F(NamedTuple):
        data: jax.Array

        def replace(self, data):
            return self._replace(data=data)

    class _S(NamedTuple):
        p_s: "TestFixerGradScatteredMatchesSerial._F"

    class _G(NamedTuple):
        area: jax.Array
        grid_total_area: jax.Array

    def _setup(self, topology):
        n = self.N
        rng = np.random.default_rng(1814)
        area = jnp.asarray(rng.uniform(0.5, 1.5, (6, n, n)))
        p_old = jnp.asarray(1e5 + 100 * rng.standard_normal((6, n, n)))
        p_new = jnp.asarray(1e5 + 100 * rng.standard_normal((6, n, n)))
        w = jnp.asarray(rng.standard_normal((6, n, n)))
        faces = np.asarray(topology.local_face_ids)
        return area, p_old, p_new, w, faces

    def _compare(self, topology, fixed_fn):
        area, p_old, p_new, w, faces = self._setup(topology)

        def loss(a, wt, po, pn):
            g = self._G(area=a, grid_total_area=jnp.sum(a))
            return jnp.sum(wt * fixed_fn(po, pn, g) ** 2)

        set_halo_backend("local")
        g_ref = jax.grad(loss, argnums=(2, 3))(area, w, p_old, p_new)
        set_halo_backend("mpi", topology)
        g_mpi = jax.grad(loss, argnums=(2, 3))(
            area[faces], w[faces], p_old[faces], p_new[faces])
        for got, ref in zip(g_mpi, g_ref):
            np.testing.assert_allclose(np.asarray(got),
                                       np.asarray(ref)[faces], rtol=1e-9)

    def test_fix_mass_hydrostatic(self, topology):
        if len(topology.local_face_ids) >= 6:
            pytest.skip("needs face scatter (>= 2 ranks)")
        from legoesm.core.conservation import fix_mass_hydrostatic
        F, S = self._F, self._S
        self._compare(topology, lambda po, pn, g: fix_mass_hydrostatic(
            S(F(pn)), S(F(po)), g).p_s.data)

    def test_fix_ps_mass(self, topology):
        if len(topology.local_face_ids) >= 6:
            pytest.skip("needs face scatter (>= 2 ranks)")
        from legoesm.core.conservation import fix_ps_mass
        self._compare(topology, lambda po, pn, g: fix_ps_mass(pn, po, g))

    def test_fix_ps_mass_target(self, topology):
        if len(topology.local_face_ids) >= 6:
            pytest.skip("needs face scatter (>= 2 ranks)")
        from legoesm.core.conservation import fix_ps_mass_target
        self._compare(topology, lambda po, pn, g: fix_ps_mass_target(
            pn, jnp.asarray(1e5) * 6.0 * self.N ** 2, g))


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


class TestVoronoiHaloMPIGrad:
    """Gradient through the Voronoi (MPAS) halo exchange — BOTH paths.

    Mirrors ``TestPadHaloMPIGrad.test_pad_halo_mpi_grad_matches_local``
    for the unstructured Voronoi mesh: the per-rank gradient of a global
    (allreduced) loss through the halo exchange must match the serial
    gradient of the equivalent all-partitions-in-one-process
    composition.  Parametrized over the two state-exchange paths of
    ``voronoi_mpi.make_voronoi_mpi_step`` (selected there at import by
    ``LEGOESM_VORONOI_BATCHED_HALO`` / ``_USE_BATCHED_HALO``):

    - ``per_entity_legacy`` — the opt-OUT model-step path
      (``VoronoiHaloExchange`` -> ``_exchange_mpi`` ->
      ``get_sendrecv_vjp``), composed exactly as the legacy
      ``_exchange_mpas_state`` packs fields (u edge; T+p_s packed cell;
      tracers stacked cell);
    - ``batched`` — the DEFAULT union-neighbor exchange
      (``batched_halo_exchange``), called as a direct closure (no env
      var needed).  Its UNEQUAL per-neighbor send/recv counts also gate
      the asymmetric-shape backward recv template in ``_sendrecv_vjp``
      (cotangents must come back SEND-shaped, not recv-shaped).

    Real gate: ``mpirun -np 2`` (and ``-np 3``/``-np 6``).  At np=1 the
    schedules have no neighbors, the exchange is the identity, and the
    test degrades gracefully (same convention as the other classes
    here).
    """

    @pytest.fixture(scope="class")
    def voronoi_setup(self):
        """Mesh + every rank's partition/schedule (deterministic, no comm)."""
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.parallel.voronoi_partition import (
            build_batched_halo_schedule,
            partition_cells_geometric,
            partition_voronoi_mesh,
        )

        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
        n_ranks = MPI.COMM_WORLD.Get_size()
        cell_owner = partition_cells_geometric(mesh, n_ranks)
        parts = [
            partition_voronoi_mesh(mesh, n_ranks, r, cell_owner=cell_owner)
            for r in range(n_ranks)
        ]
        scheds = [
            build_batched_halo_schedule(p.cell_comm, p.edge_comm)
            for p in parts
        ]
        return mesh, parts, scheds

    @pytest.mark.parametrize(
        "exchange_path", ["per_entity_legacy", "batched"])
    def test_voronoi_exchange_grad_matches_serial(
            self, voronoi_setup, exchange_path):
        """Per-rank MPI gradient == serial-composition gradient.

        Loss: sum of squares of the post-exchange local fields on every
        rank, allreduced.  Serially that is ``sum_p sum(x[p.local]**2)``
        over all partitions, so the serial gradient at global entity e is
        ``2*x[e] * (#partitions whose local set contains e)``.  Under MPI
        the extra copies live on neighbor ranks, and their cotangents
        reach the owner ONLY through the sendrecv VJP — a dropped or
        mis-routed backward message fails this test.  Halo-row gradients
        must be exactly zero (the forward overwrites every halo row on
        both paths: per-entity and union recv_idx cover the same rows).
        """
        from legoesm.parallel.halo_exchange_voronoi import (
            VoronoiHaloExchange,
            batched_halo_exchange,
        )

        mesh, parts, scheds = voronoi_setup
        rank = MPI.COMM_WORLD.Get_rank()
        part, sched = parts[rank], scheds[rank]
        nlev = 3

        if exchange_path == "batched":
            # Opt-in path, exercised as a direct closure.
            def _exchange(u, T, ps, qv):
                (u_e,), (T_e, ps_e, qv_e) = batched_halo_exchange(
                    (u,), (T, ps, qv), sched, rank)
                return u_e, T_e, ps_e, qv_e
        else:
            # DEFAULT model-step path: compose EXACTLY as the legacy
            # ``voronoi_mpi._exchange_mpas_state`` does — u edge exchange;
            # T+p_s packed into ONE cell exchange; tracers stacked into
            # one further cell exchange — through ``VoronoiHaloExchange``
            # -> ``_exchange_mpi`` -> ``get_sendrecv_vjp``.
            halo = VoronoiHaloExchange(part, backend="mpi")

            def _exchange(u, T, ps, qv):
                u_e = halo.exchange_edge_field(u)
                Tp_e = halo.exchange_cell_field(
                    jnp.concatenate([T, ps[..., None]], axis=-1))
                T_e, ps_e = Tp_e[..., :nlev], Tp_e[..., nlev]
                qv_e = halo.exchange_cell_field(
                    jnp.stack([qv], axis=-1))[..., 0]
                return u_e, T_e, ps_e, qv_e

        k = jax.random.split(jax.random.PRNGKey(11), 4)
        u_g = jax.random.normal(k[0], (mesh.nEdges, nlev), dtype=jnp.float64)
        T_g = jax.random.normal(k[1], (mesh.nCells, nlev), dtype=jnp.float64)
        ps_g = jax.random.normal(k[2], (mesh.nCells,), dtype=jnp.float64)
        qv_g = jax.random.normal(k[3], (mesh.nCells, nlev), dtype=jnp.float64)

        # --- Serial reference (no collectives; identical on every rank) ---
        edge_ids = [jnp.asarray(p.local_edges) for p in parts]
        cell_ids = [jnp.asarray(p.local_cells) for p in parts]

        def serial_loss(u, T, ps, qv):
            total = 0.0
            for ce, cc in zip(edge_ids, cell_ids):
                total = (total + jnp.sum(u[ce] ** 2) + jnp.sum(T[cc] ** 2)
                         + jnp.sum(ps[cc] ** 2) + jnp.sum(qv[cc] ** 2))
            return total

        g_serial = jax.grad(serial_loss, argnums=(0, 1, 2, 3))(
            u_g, T_g, ps_g, qv_g)

        # --- MPI leg: rank-local inputs with halo rows ZEROED, so the
        # forward loss matches serial only if the exchange fills halos ---
        def _local_zero_halo(arr, ids, n_owned):
            loc = arr[jnp.asarray(ids)]
            return loc.at[n_owned:].set(0.0)

        u_l = _local_zero_halo(u_g, part.local_edges, part.n_owned_edges)
        T_l = _local_zero_halo(T_g, part.local_cells, part.n_owned_cells)
        ps_l = _local_zero_halo(ps_g, part.local_cells, part.n_owned_cells)
        qv_l = _local_zero_halo(qv_g, part.local_cells, part.n_owned_cells)

        def mpi_loss(u, T, ps, qv):
            u_e, T_e, ps_e, qv_e = _exchange(u, T, ps, qv)
            local = (jnp.sum(u_e ** 2) + jnp.sum(T_e ** 2)
                     + jnp.sum(ps_e ** 2) + jnp.sum(qv_e ** 2))
            return global_sum_mpi(local, final_loss=True)

        loss_mpi, g_mpi = jax.value_and_grad(
            mpi_loss, argnums=(0, 1, 2, 3))(u_l, T_l, ps_l, qv_l)

        # Forward parity also certifies the exchanged values.
        np.testing.assert_allclose(
            float(loss_mpi), float(serial_loss(u_g, T_g, ps_g, qv_g)),
            rtol=1e-12,
            err_msg=(
                f"forward loss through {exchange_path} exchange != serial"
            ),
        )

        checks = [
            (g_mpi[0], g_serial[0], part.local_edges,
             part.n_owned_edges, "u"),
            (g_mpi[1], g_serial[1], part.local_cells,
             part.n_owned_cells, "T"),
            (g_mpi[2], g_serial[2], part.local_cells,
             part.n_owned_cells, "p_s"),
            (g_mpi[3], g_serial[3], part.local_cells,
             part.n_owned_cells, "q_v"),
        ]
        for g_m, g_s, ids, n_owned, name in checks:
            np.testing.assert_allclose(
                np.asarray(g_m[:n_owned]),
                np.asarray(g_s)[np.asarray(ids[:n_owned])],
                atol=1e-12,
                err_msg=(
                    f"{name}: owned-row MPI gradient != serial gradient "
                    f"(rank {rank}) — sendrecv VJP dropped/mis-routed a "
                    f"halo cotangent"
                ),
            )
            np.testing.assert_array_equal(
                np.asarray(g_m[n_owned:]), 0.0,
                err_msg=(
                    f"{name}: halo-row gradient must be exactly zero "
                    f"(forward .at[recv_idx].set overwrites every halo row)"
                ),
            )

        # Production composition: the real step traces the exchange under
        # @jax.jit, so gate jit(grad(...)) too — XLA must compile the
        # custom_vjp sendrecv pair (fwd+bwd) into one program on every
        # rank without reordering the matched message sequence.
        g_jit = jax.jit(jax.grad(mpi_loss, argnums=(0, 1, 2, 3)))(
            u_l, T_l, ps_l, qv_l)
        for g_e, g_j, name in zip(
                g_mpi, g_jit, ["u", "T", "p_s", "q_v"]):
            np.testing.assert_allclose(
                np.asarray(g_j), np.asarray(g_e), atol=1e-12,
                err_msg=f"{name}: jit(grad) != eager grad (rank {rank})",
            )

    def test_voronoi_batched_grad_finite_and_message_count(
            self, voronoi_setup):
        """Finite, nonzero grads; eager exchange posts EXACTLY the
        schedule-math message count (n_union_neighbors for one dtype
        group) — strictly fewer than the legacy per-entity pattern."""
        from legoesm.parallel.halo_exchange_voronoi import (
            batched_halo_exchange,
            count_batched_messages,
            get_halo_message_count,
            reset_halo_message_count,
        )

        _mesh, parts, scheds = voronoi_setup
        rank = MPI.COMM_WORLD.Get_rank()
        part, sched = parts[rank], scheds[rank]
        nlev = 2
        u = jnp.ones((part.n_local_edges, nlev), dtype=jnp.float64)
        T = jnp.ones((part.n_local_cells, nlev), dtype=jnp.float64)
        ps = jnp.ones((part.n_local_cells,), dtype=jnp.float64)
        qv = jnp.ones((part.n_local_cells, nlev), dtype=jnp.float64)

        # Message-count instrumentation (counter increments per posted
        # sendrecv when running eagerly).
        reset_halo_message_count()
        _ = batched_halo_exchange((u,), (T, ps, qv), sched, rank)
        posted = get_halo_message_count()
        assert posted == count_batched_messages((u,), (T, ps, qv), sched)
        assert posted == sched.messages_per_exchange(n_dtype_groups=1)
        if MPI.COMM_WORLD.Get_size() > 1:
            legacy = (len(part.edge_comm.neighbor_ranks)
                      + 2 * len(part.cell_comm.neighbor_ranks))
            assert posted < legacy, (
                f"batched path posted {posted} messages, legacy per-entity "
                f"pattern posts {legacy}"
            )

        def loss(u, T, ps, qv):
            (u_e,), (T_e, ps_e, qv_e) = batched_halo_exchange(
                (u,), (T, ps, qv), sched, rank)
            return (jnp.sum(u_e ** 2) + jnp.sum(T_e ** 2)
                    + jnp.sum(ps_e ** 2) + jnp.sum(qv_e ** 2))

        g = jax.grad(loss, argnums=(0, 1, 2, 3))(u, T, ps, qv)
        for arr in g:
            assert jnp.all(jnp.isfinite(arr))
        assert not jnp.allclose(g[0], 0.0)


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
