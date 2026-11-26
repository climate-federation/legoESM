"""Second-order AD through the MPI halo sendrecv and the FFT all-to-all.

(a) the plane-CRM halo exchange (``exchange_halo_plane_yxz``, the ``step_halo``
path's sendrecv), (b) the spectral-LES slab / pencil FFT transposes
(``spectral_test_filter`` -> ``layout_rfft2``/``layout_irfft2`` -> all-to-all).

Two kinds of check per operator, both at 2 ranks on each rank mesh:

* ``jax.test_util.check_grads(order=2, modes=("fwd", "rev"))`` on inputs that
  are IDENTICAL on every rank. check_grads draws its directions from a fixed
  seed and compares LOCAL inner products; a distributed map satisfies that
  per-rank identity only when every rank holds the same data and the operator
  commutes with the rank shift (periodic halo, translation-invariant filter).
* a GLOBAL Hessian-vector check on rank-dependent data: reverse-over-reverse, forward-over-reverse and
  forward-over-forward second derivatives vs central differences of the
  (first-order, already verified) gradient / loss. At 2 ranks the two
  neighbours along a split axis are the same rank, so the exchange direction
  is checked by the ring test, which needs 3+ ranks for source != dest.

    mpiexec -n 2 python -m pytest tests/distributed/test_mpi_second_order_ad.py
    mpiexec -n 3 python -m pytest tests/distributed/test_mpi_second_order_ad.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

MPI = pytest.importorskip("mpi4py.MPI")
pytest.importorskip("mpi4jax")

from jax.test_util import check_grads  # noqa: E402

from legoesm.atmosphere.dynamics.les.spectral_les_plane import (  # noqa: E402
    make_spectral_les_pencil_layout)
from legoesm.atmosphere.physics.turbulence.lasd_core import (  # noqa: E402
    spectral_test_filter)
from legoesm.parallel.plane_mpi import (  # noqa: E402
    exchange_halo_plane_yxz, make_plane_pencil_layout)
from legoesm.parallel.reductions import global_sum_mpi  # noqa: E402

COMM = MPI.COMM_WORLD
RANK, NPROC = COMM.Get_rank(), COMM.Get_size()
pytestmark = pytest.mark.skipif(NPROC < 2, reason="needs >= 2 MPI ranks")
two_ranks = pytest.mark.skipif(NPROC != 2, reason="rank meshes are for 2 ranks")

MESHES = [(2, 1), (1, 2)]
NY, NX, NZ, HALO = 8, 8, 3, 2
CUT_Y, CUT_X = 3, 3
TOL = 1e-6          # check_grads atol/rtol and the global HVP tolerance
EPS = 1e-4          # central-difference step of the global HVP checks


def _halo_op(mesh):
    L = make_plane_pencil_layout(RANK, NPROC, *mesh, NY, NX, halo=HALO)
    return (lambda x: exchange_halo_plane_yxz(x, L)), (L.ny_local, L.nx_local, NZ)


def _filter_op(mesh):
    L = make_spectral_les_pencil_layout(COMM, *mesh, NY, NX)
    return (lambda x: spectral_test_filter(x, CUT_Y, CUT_X, L)), \
        (NY // mesh[0], NX // mesh[1], NZ)


OPS = {"halo": _halo_op, "fft_filter": _filter_op}


def _rand(seed, shape):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


@two_ranks
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("name", list(OPS))
def test_check_grads_order2_identical_inputs(name, mesh):
    op, shape = OPS[name](mesh)
    x = _rand(0, shape)                              # same on every rank
    check_grads(lambda x: jnp.sin(op(x)) * op(x), (x,), order=2,
                modes=("fwd", "rev"), atol=TOL, rtol=TOL)


def _check_global_hvp(op, x, v, w, jit=False):
    """Hv by rev-over-rev and fwd-over-rev vs central differences of the
    gradient (block by block), and v^T H v by fwd-over-fwd vs the global
    <v, Hv_fd>. The loss is the replicated global sum every rank evaluates."""
    def loss(x):
        y = op(x)
        return global_sum_mpi(jnp.sum(w * jnp.sin(y) * y ** 2), final_loss=True)

    jt = jax.jit if jit else (lambda f: f)
    grad = jt(jax.grad(loss))
    hv_fd = (grad(x + EPS * v) - grad(x - EPS * v)) / (2 * EPS)
    hv_rr = jt(jax.grad(lambda x: global_sum_mpi(jnp.vdot(grad(x), v),
                                                 final_loss=True)))(x)
    hv_fr = jt(lambda x: jax.jvp(grad, (x,), (v,))[1])(x)
    scale = float(COMM.allreduce(float(jnp.max(jnp.abs(hv_fd))), op=MPI.MAX))
    for label, hv in (("rev-over-rev", hv_rr), ("fwd-over-rev", hv_fr)):
        np.testing.assert_allclose(np.asarray(hv), np.asarray(hv_fd),
                                   atol=TOL * scale, rtol=0, err_msg=label)
    vhv_ff = jt(lambda x: jax.jvp(lambda x: jax.jvp(loss, (x,), (v,))[1],
                                  (x,), (v,))[1])(x)
    vhv_fd = COMM.allreduce(float(jnp.vdot(v, hv_fd)))
    assert abs(float(vhv_ff) - vhv_fd) <= TOL * max(abs(vhv_fd), 1.0), \
        ("fwd-over-fwd", float(vhv_ff), vhv_fd)


@two_ranks
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("name", list(OPS))
def test_global_hessian_vector_products(name, mesh):
    op, shape = OPS[name](mesh)
    x, v = _rand(10 + RANK, shape), _rand(20 + RANK, shape)   # rank-dependent
    _check_global_hvp(op, x, v, _rand(30 + RANK, op(x).shape))


@pytest.mark.parametrize("jit", [False, True])
def test_ring_sendrecv_unequal_sizes_and_tags(jit):
    """The raw shared sendrecv on a ring (send to r+1, receive from r-1) with a
    rank-dependent message length and unequal send/recv tags: the transpose
    must swap endpoints AND tags and receive a SEND-shaped buffer. At 3+ ranks
    source != dest, so a transpose that kept the direction fails here."""
    import mpi4jax

    from legoesm.parallel.halo_exchange import get_sendrecv_vjp

    sr = get_sendrecv_vjp(mpi4jax)
    src, dst = (RANK - 1) % NPROC, (RANK + 1) % NPROC
    n_send, n_recv = 3 + RANK, 3 + src

    def op(x):
        return sr(x, jnp.zeros(n_recv, x.dtype), src, dst, 100 + RANK,
                  100 + src, COMM)

    x, v = _rand(50 + RANK, (n_send,)), _rand(60 + RANK, (n_send,))
    np.testing.assert_array_equal(
        np.asarray(op(jnp.full(n_send, float(RANK)))), np.full(n_recv, src))
    _check_global_hvp(op, x, v, _rand(70 + RANK, (n_recv,)), jit=jit)


@two_ranks
@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("name", list(OPS))
def test_vmap_matches_loop(name, mesh):
    op, shape = OPS[name](mesh)
    xs = jnp.stack([_rand(40 + 2 * RANK + i, shape) for i in range(2)])
    np.testing.assert_allclose(np.asarray(jax.vmap(op)(xs)),
                               np.stack([np.asarray(op(x)) for x in xs]),
                               rtol=1e-13, atol=1e-13)
