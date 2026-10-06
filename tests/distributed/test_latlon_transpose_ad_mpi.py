"""Reverse-mode AD correctness for the lat-pencil transpose
(``lon_gather_full``) under MPI — the custom-VJP that makes the
lon-split pole-fold / polar-filter differentiable (was forward-only;
``mpi4jax.allgather`` has no native VJP).

The gather ``G`` is a LINEAR map (each rank's lon block → the full-lon
array, replicated across the row ring).  Its adjoint is exact:
``x̄_s = Σ_r ȳ_r[:, block_s]`` = ``allreduce(SUM)`` over the row ring
then slice the rank's block.  We verify the implemented VJP IS that
adjoint with the dot-product identity ``<Gx, y> == <x, Gᵀy>`` summed
globally (the gold-standard linear-operator adjoint test — no serial
reference needed), plus a finiteness check on a scalar-loss gradient.

Run: ``mpirun -np {2,3,6} python -m pytest <thisfile>`` (wired into
mpi-distributed.yml).  Single rank skips (no lon ring to gather).
Size-adaptive: np2→1×2, np3→1×3, np6→2×3 (proc_lon≥2 always, so a real
ring); n_lon = 4·proc_lon keeps the allgather's equal-split requirement.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    make_lon_row_comm,
    lon_gather_full,
    scatter_field_latlon_2d,
)
from legoesm.parallel.reductions import global_sum_mpi


def _pick_grid(n_ranks: int) -> tuple[int, int]:
    """(proc_lat, proc_lon) with proc_lon holding the larger factor so a
    real lon ring (proc_lon>=2) is always exercised: np2→(1,2),
    np3→(1,3), np6→(2,3)."""
    best = (1, n_ranks)
    for pr in range(2, int(n_ranks ** 0.5) + 1):
        if n_ranks % pr == 0:
            best = (pr, n_ranks // pr)
    return best


def test_transpose_adjoint_dotproduct():
    """<G x, y> == <x, Gᵀ y> (global) — the implemented custom VJP is the
    exact adjoint of the lat-pencil gather."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks (lon ring to gather)")
    pr, pc = _pick_grid(n)
    n_lat, n_lon = 2 * pr, 4 * pc          # equal lon split (n_lon % pc == 0)
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    row_comm = make_lon_row_comm(L)

    rng = np.random.default_rng(100 + rank)
    x = jnp.asarray(rng.standard_normal((L.n_lat_local, L.n_lon_local)))
    # y is the cotangent seed on the FULL-lon (gathered) shape.
    y = jnp.asarray(rng.standard_normal((L.n_lat_local, n_lon)))

    fx = lon_gather_full(x, L, row_comm)              # G x  (full-lon)
    _, vjp_fn = jax.vjp(lambda z: lon_gather_full(z, L, row_comm), x)
    xbar = vjp_fn(y)[0]                                # Gᵀ y  (rank's block)

    lhs = float(np.asarray(global_sum_mpi(jnp.sum(fx * y), comm)))
    rhs = float(np.asarray(global_sum_mpi(jnp.sum(x * xbar), comm)))
    np.testing.assert_allclose(
        lhs, rhs, rtol=1e-11, atol=1e-11,
        err_msg=f"rank {rank}: adjoint identity <Gx,y> != <x,Gᵀy> "
                f"(pr={pr} pc={pc}): {lhs} vs {rhs}")
    if rank == 0:
        print(f"TRANSPOSE_ADJOINT_OK pr={pr} pc={pc} lhs={lhs:.6e}",
              flush=True)


def test_transpose_grad_finite():
    """A scalar loss through gather → fold(roll, block-mixing) → grad is
    finite on every rank (no NaN/inf from the collective VJP; the roll
    makes the loss non-separable across lon blocks so the ring adjoint is
    genuinely exercised)."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    n_lat, n_lon = 2 * pr, 4 * pc
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    row_comm = make_lon_row_comm(L)
    g = np.random.default_rng(7).standard_normal((n_lat, n_lon))
    local = scatter_field_latlon_2d(jnp.asarray(g), L)

    def loss(x):
        full = lon_gather_full(x, L, row_comm)
        # 180° lon roll = the pole-fold shift; mixes blocks so grad must
        # route through the ring adjoint.
        mixed = full * jnp.roll(full, n_lon // 2, axis=1)
        return global_sum_mpi(jnp.sum(mixed), comm, final_loss=True)

    grd = np.asarray(jax.grad(loss)(local))
    assert np.all(np.isfinite(grd)), f"rank {rank}: non-finite transpose grad"
    if rank == 0:
        print(f"TRANSPOSE_GRAD_OK pr={pr} pc={pc}", flush=True)
