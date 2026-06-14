"""LatLon2DLayout — 2-D pencil decomposition (lat-lon 2-D, increment 2).

Serial part (no MPI): the per-rank blocks TILE the global exactly, the
N/S line terminates at the poles, the W/E ring is periodic, and
proc_lon==1 reproduces the band.  MPI part (mpirun -np N): a 2-D
scatter→gather roundtrip == identity.
"""
from __future__ import annotations

import numpy as np

from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    make_latlon_band_layout,
    scatter_field_latlon_2d,
    gather_field_latlon_2d,
)


def test_blocks_tile_the_global_exactly():
    n_lat, n_lon, pr, pc = 16, 24, 4, 2
    cover = np.zeros((n_lat, n_lon), dtype=int)
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
        cover[L.lat_start:L.lat_end, L.lon_start:L.lon_end] += 1
        assert L.n_lat_local == L.lat_end - L.lat_start > 0
        assert L.n_lon_local == L.lon_end - L.lon_start > 0
    # every global cell owned exactly once (partition, no overlap/gap)
    assert np.all(cover == 1), "blocks do not tile the global exactly"


def test_ns_line_ew_ring_topology():
    pr, pc = 3, 4
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, 12, 16)
        # N/S = pole-terminated line
        if L.proc_row == 0:
            assert L.south_rank is None
        else:
            assert L.south_rank == rank - pc
        if L.proc_row == pr - 1:
            assert L.north_rank is None
        else:
            assert L.north_rank == rank + pc
        # E/W = periodic ring (always defined, wraps)
        assert L.west_rank == L.proc_row * pc + (L.proc_col - 1) % pc
        assert L.east_rank == L.proc_row * pc + (L.proc_col + 1) % pc


def test_proc_lon_1_is_band_equivalent():
    # proc_lon==1: the 2-D layout's lat split + N/S neighbours must match
    # make_latlon_band_layout EXACTLY, including the uneven-remainder
    # case (codex review: assert parity vs the band, not just divisible).
    for n_lat, nproc in ((16, 4), (17, 4), (15, 4), (10, 3)):
        for rank in range(nproc):
            L = make_latlon_2d_layout(rank, nproc, 1, n_lat, 24)
            B = make_latlon_band_layout(rank, nproc, n_lat, 24)
            assert L.n_lon_local == 24
            assert L.west_rank == rank and L.east_rank == rank
            assert (L.lat_start, L.lat_end, L.n_lat_local) == (
                B.lat_start, B.lat_end, B.n_lat_local), (
                f"lat split != band at n_lat={n_lat} rank={rank}")
            assert (L.south_rank, L.north_rank) == (
                B.south_rank, B.north_rank)


def test_scatter_slices_block():
    g = np.arange(16 * 24).reshape(16, 24).astype(float)
    L = make_latlon_2d_layout(5, 4, 2, 16, 24)  # row 2, col 1
    blk = np.asarray(scatter_field_latlon_2d(g, L))
    np.testing.assert_array_equal(
        blk, g[L.lat_start:L.lat_end, L.lon_start:L.lon_end])


def test_rejects_too_few_cells():
    for bad in (dict(proc_lat=8, proc_lon=1, n_lat=4, n_lon=24),
                dict(proc_lat=1, proc_lon=8, n_lat=16, n_lon=4)):
        try:
            make_latlon_2d_layout(0, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def _mpi_roundtrip():
    """mpirun -np N (N=proc_lat*proc_lon): scatter→gather == identity."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n == 1:
        return
    # factor n into (proc_lat, proc_lon) — prefer proc_lon>1 to exercise
    # the 2-D path (n=4 -> 2x2; n=2 -> 1x2; n=6 -> 3x2).
    pc = 2 if n % 2 == 0 else 1
    pr = n // pc
    n_lat, n_lon = 4 * pr, 6 * pc
    g = np.random.default_rng(0).standard_normal((n_lat, n_lon, 2))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(g, L)
    back = gather_field_latlon_2d(local, L)
    np.testing.assert_allclose(back, g, atol=1e-12,
                               err_msg=f"rank {rank} roundtrip != id")
    if rank == 0:
        print(f"LATLON_2D_ROUNDTRIP_OK pr={pr} pc={pc}", flush=True)


if __name__ == "__main__":
    _mpi_roundtrip()
