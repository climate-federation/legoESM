"""NAMED MPI GATE for the water-budget tripwire global reductions (tripwire C).

Cannot run on the login node -- run on a compute node:
    mpirun -np 2 $PY_MPI -m pytest -q tests/distributed/test_water_budget_global_reduction_mpi.py

This is the ONLY test that certifies the disjoint-interior-shard contract the
distributed correctness of ``area_integral`` / ``atm_moisture_residual`` rests
on: each rank holds a DISJOINT chunk of a known global field (no replication ->
no allreduce N-count; no halo -> no boundary double-count), and the globally
reduced integral must equal the analytic serial integral. The CPU unit tests
only prove the serial no-op + that the collective is wired.

Arming mirrors tests/distributed/test_latlon_2d_mpi_step.py: initialize a lat-lon
band layout so ``is_distributed()`` is True, which makes
``global_sum_if_distributed`` route through ``global_sum_mpi`` (allreduce SUM).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.diagnostics.water_budget import area_integral, atm_moisture_residual
from legoesm.grids.halo import set_halo_backend
from legoesm.parallel.distributed import initialize_distributed_latlon
from legoesm.parallel.reductions import is_multi_process


def _disjoint(rank, size, n):
    """This rank's DISJOINT half-open [lo, hi) slice of a length-n global axis."""
    per = n // size
    lo = rank * per
    hi = n if rank == size - 1 else (rank + 1) * per
    return slice(lo, hi)


@pytest.mark.mpi
def test_area_integral_np2_equals_serial():
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    if size < 2:
        pytest.skip("needs np>=2")
    # Arm the mpi halo backend so is_distributed()/is_multi_process() are True.
    initialize_distributed_latlon(global_n_lat=max(size, 4))
    try:
        assert is_multi_process(), "mpi backend not armed"
        n = 8
        full = np.arange(1.0, n + 1.0)          # 1..8, a known global field
        area = np.full(n, 2.0)
        sl = _disjoint(rank, size, n)
        # area_integral = allreduce(SUM) of each rank's local shard integral.
        got = float(area_integral(jnp.asarray(full[sl]), jnp.asarray(area[sl])))
        expect = float((full * area).sum())     # true GLOBAL integral (== 72.0)
        assert abs(got - expect) < 1e-9, (rank, got, expect)
        # Every rank sees the SAME global value (not its rank-local shard sum).
        assert abs(got - comm.bcast(got, root=0)) < 1e-12
    finally:
        set_halo_backend("local")               # reset for any following test


@pytest.mark.mpi
def test_atm_moisture_residual_np2_equals_serial():
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    if size < 2:
        pytest.skip("needs np>=2")
    initialize_distributed_latlon(global_n_lat=max(size, 4))
    try:
        assert is_multi_process()
        n, dt = 8, 3600.0
        cwv_now = np.linspace(20.0, 27.0, n)
        cwv_prev = np.linspace(19.0, 26.0, n)
        evap = np.full(n, 4.0e-5)
        precip = np.full(n, 3.5e-5)
        area = np.full(n, 2.0)
        sl = _disjoint(rank, size, n)
        got = float(atm_moisture_residual(
            jnp.asarray(cwv_now[sl]), jnp.asarray(cwv_prev[sl]),
            jnp.asarray(evap[sl]), jnp.asarray(precip[sl]),
            jnp.asarray(area[sl]), dt))
        # Analytic global residual R = d<CWV>/dt - (<E> - <P>).
        asum = area.sum()
        dcwv = ((cwv_now * area).sum() - (cwv_prev * area).sum()) / asum / dt
        emp = ((evap * area).sum() - (precip * area).sum()) / asum
        expect = float(dcwv - emp)
        assert abs(got - expect) < 1e-12, (rank, got, expect)
    finally:
        set_halo_backend("local")
