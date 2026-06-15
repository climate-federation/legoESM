"""E/W periodic longitude halo (lat-lon 2-D decomposition, increment 1).

exchange_halo_lon adds ghost lon columns from the W/E ring neighbours
(globally periodic).  Serial part runs without MPI; the ring-parity part
needs ``mpirun -np N`` (split a global field by longitude, exchange,
verify each rank's ghost columns equal the GLOBAL periodic neighbours).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.parallel.latlon_mpi import exchange_halo_lon


def test_single_ring_local_wrap_matches_roll():
    # proc_lon == 1 (west==east==self): byte-identical to the legacy
    # periodic jnp.roll wrap.
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.standard_normal((6, 8, 3)))
    for halo in (1, 2):
        out = exchange_halo_lon(f, 0, 0, 0, halo=halo)
        ref = jnp.concatenate([f[:, -halo:], f, f[:, :halo]], axis=1)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
        assert out.shape == (6, 8 + 2 * halo, 3)


def test_single_ring_2d_field():
    rng = np.random.default_rng(1)
    f = jnp.asarray(rng.standard_normal((5, 7)))
    out = exchange_halo_lon(f, 0, 0, 0, halo=1)
    ref = jnp.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_single_ring_grad_finite():
    f = jnp.asarray(np.linspace(0.0, 1.0, 6 * 8).reshape(6, 8))

    def loss(x):
        return jnp.sum(exchange_halo_lon(x, 0, 0, 0, halo=1) ** 2)

    g = jax.grad(loss)(f)
    assert np.all(np.isfinite(np.asarray(g)))
    # interior columns are read once (own) + twice (as neighbours' ghost
    # via the wrap), so the gradient is strictly positive where x != 0.
    assert float(jnp.sum(g)) != 0.0


def test_halo_exceeds_width_raises():
    f = jnp.ones((4, 2))
    try:
        exchange_halo_lon(f, 0, 0, 0, halo=3)
    except ValueError as e:
        assert "halo" in str(e)
    else:
        raise AssertionError("expected ValueError for halo > n_lon_local")


def _mpi_ring_parity():
    """Run under mpirun -np N: split a global field across the lon ring,
    exchange, assert ghosts == global periodic neighbours."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n == 1:
        return
    n_lat, n_lon = 4, 6 * n          # even split, n_lon_local = 6
    rng = np.random.default_rng(42)
    g = rng.standard_normal((n_lat, n_lon))   # same on every rank (seeded)
    nl = n_lon // n
    s = rank * nl
    local = jnp.asarray(g[:, s:s + nl])
    west, east = (rank - 1) % n, (rank + 1) % n
    out = np.asarray(exchange_halo_lon(local, west, east, rank, halo=1))
    # west ghost == global column (s-1) mod n_lon; east ghost == (s+nl) %
    want_w = g[:, (s - 1) % n_lon]
    want_e = g[:, (s + nl) % n_lon]
    np.testing.assert_allclose(out[:, 0], want_w, atol=1e-12,
                               err_msg=f"rank {rank} west ghost")
    np.testing.assert_allclose(out[:, -1], want_e, atol=1e-12,
                               err_msg=f"rank {rank} east ghost")
    np.testing.assert_allclose(out[:, 1:-1], np.asarray(local), atol=1e-12,
                               err_msg=f"rank {rank} interior preserved")
    if rank == 0:
        print(f"LON_RING_PARITY_OK n={n}", flush=True)


def _mpi_ring_grad():
    """Run under mpirun -np N: gradient through exchange_halo_lon must
    (a) not hang (the AD tag bug for proc_lon>=3 would deadlock the
    backward sendrecv) and (b) match the serial periodic-pad reference —
    every BLOCK-boundary column (all 2n: each block's first + last) is a
    neighbour's ghost so gets 4*g (interior read + ghost read), the rest
    2*g (see the reads-accumulation below)."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n == 1:
        return
    n_lat, n_lon = 4, 6 * n
    rng = np.random.default_rng(7)
    g = rng.standard_normal((n_lat, n_lon))
    nl = n_lon // n
    s = rank * nl
    local = jnp.asarray(g[:, s:s + nl])
    west, east = (rank - 1) % n, (rank + 1) % n

    def loss(x):
        return jnp.sum(exchange_halo_lon(x, west, east, rank, halo=1) ** 2)

    gr = np.asarray(jax.grad(loss)(local))   # would HANG if tags mismatch
    assert np.all(np.isfinite(gr)), f"rank {rank} non-finite grad"

    # Serial reference: the global loss is sum over EVERY rank's padded
    # field [west_ghost | block | east_ghost].  dL/dg[:,j] = 2*g[:,j] x
    # (number of times column j is READ across all ranks).  Each column
    # is read once as its owner's interior, and additionally each
    # BLOCK-BOUNDARY column is read as a neighbour's ghost — block r's
    # first col (r*nl) = rank(r-1)'s east ghost, last col (r*nl+nl-1) =
    # rank(r+1)'s west ghost.  Accumulate the actual reads (2n boundary
    # columns, not just the 2 global seam columns).
    reads = np.zeros(n_lon)
    for rr in range(n):
        ss = rr * nl
        cols = [(ss - 1) % n_lon] + list(range(ss, ss + nl)) + \
               [(ss + nl) % n_lon]
        for c in cols:
            reads[c] += 1.0
    ref = 2.0 * g * reads[None, :]
    ref_local = ref[:, s:s + nl]
    np.testing.assert_allclose(gr, ref_local, atol=1e-10,
                               err_msg=f"rank {rank} grad != serial ref")
    if rank == 0:
        print(f"LON_RING_GRAD_OK n={n}", flush=True)


if __name__ == "__main__":
    _mpi_ring_parity()
    _mpi_ring_grad()
