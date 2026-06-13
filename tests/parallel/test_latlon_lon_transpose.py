"""lat-pencil transpose: lon_gather_full / lon_scatter_full (the crux
primitive for pole-fold + polar filter under lon-split, increment 3b).

mpirun -np N: each rank assembles the FULL longitude axis for its lat
band from the proc_row lon-ring (mpi4jax.allgather over a row sub-comm),
verifies it == the global field's lat rows, and that scatter-back == id.
Forward-only (allgather has no VJP); equal lon split required.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    make_lon_row_comm,
    lon_gather_full,
    lon_scatter_full,
    scatter_field_latlon_2d,
)


def test_single_proc_transpose_is_identity():
    """pytest-lane coverage (no mpirun): proc_lon==1 -> a 1-rank lon ring
    -> lon_gather_full returns the (already-full) local block, scatter
    back == input.  (The multi-rank parity runs via __main__ under
    mpirun -np N — codex CI-gap fix: this file otherwise collected zero
    pytest tests.)"""
    import jax.numpy as jnp

    g = np.arange(4 * 6 * 2).reshape(4, 6, 2).astype(float)
    L = make_latlon_2d_layout(0, 1, 1, 4, 6)   # 1x1 grid
    row_comm = make_lon_row_comm(L)
    local = scatter_field_latlon_2d(jnp.asarray(g), L)
    full = np.asarray(lon_gather_full(local, L, row_comm))
    np.testing.assert_array_equal(full, g)             # already full lon
    back = np.asarray(lon_scatter_full(jnp.asarray(full), L))
    np.testing.assert_array_equal(back, np.asarray(local))


def _mpi_transpose():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n == 1:
        return
    pc = 2 if n % 2 == 0 else 1
    pr = n // pc
    n_lat, n_lon = 4 * pr, 6 * pc           # equal lon split (6/block)
    g = np.random.default_rng(11).standard_normal((n_lat, n_lon, 3))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    row_comm = make_lon_row_comm(L)         # collective over COMM_WORLD
    local = scatter_field_latlon_2d(g, L)

    full = np.asarray(lon_gather_full(local, L, row_comm))
    # full longitude for THIS rank's lat band
    want = g[L.lat_start:L.lat_end, :, :]
    np.testing.assert_allclose(full, want, atol=1e-12,
                               err_msg=f"rank {rank} lon_gather != global lat band")

    back = np.asarray(lon_scatter_full(full, L))
    np.testing.assert_allclose(back, np.asarray(local), atol=1e-12,
                               err_msg=f"rank {rank} scatter-back != local")
    if rank == 0:
        print(f"LON_TRANSPOSE_OK pr={pr} pc={pc}", flush=True)


if __name__ == "__main__":
    _mpi_transpose()
