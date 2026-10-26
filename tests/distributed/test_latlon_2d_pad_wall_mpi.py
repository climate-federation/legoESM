"""MPI-collected parity + AD tests for the full 2-D halo pad
(``pad_halo_latlon_2d``, ``pole_bc="wall"`` — regular lat-lon /
closed-pole ocean).  This is the CI home (codex re-review 2026-06-13:
the parallel-lane file ran its MPI checks only under ``__main__`` so
they were never pytest-collected → an AD regression could slip through).

Run under MPI:  ``mpirun -np {2,3,6} python -m pytest <thisfile>``
(wired into ``.github/workflows/mpi-distributed.yml`` alongside the
1-D halo tests).  Under a single rank (plain pytest) every test skips —
the single-proc forward/grad/raise paths live in the pytest lane
``tests/parallel/test_latlon_2d_pad_wall.py``.

Size-adaptive: the process grid is factored from the world size so the
SAME file covers np=2 (1×2 ring), np=3 (1×3 ring — the phase-constant
tag reverse-ring case, proc_lon≥3), and np=6 (2×3 — N/S reverse sendrecv
AND the proc_lon=3 reverse ring together).  ``n_lat=2·pr+1`` /
``n_lon=2·pc+1`` give UNEVEN splits, so the remainder path is exercised
too (smallest block = 2 ≥ halo).
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
    scatter_field_latlon_2d,
    pad_halo_latlon_2d,
)
from legoesm.parallel.reductions import global_sum_mpi

SV, NV = -1.0, -2.0   # distinct non-zero walls to catch S/N confusion


def _pick_grid(n_ranks: int) -> tuple[int, int]:
    """Factor the world size into ``(proc_lat, proc_lon)`` maximising
    halo+VJP coverage: a balanced 2-D split (proc_lat≥2 AND proc_lon≥2)
    when ``n_ranks`` is composite, else a 1×n ring for primes.  Always
    prefers the LARGER factor on proc_lon so the periodic-ring reverse
    VJP (phase-constant tags at proc_lon≥3) gets exercised whenever a
    factor allows it."""
    best = (1, n_ranks)
    for pr in range(2, int(n_ranks ** 0.5) + 1):
        if n_ranks % pr == 0:
            best = (pr, n_ranks // pr)   # pr ≤ pc, so pc holds the larger factor
    return best


def _serial_ref(g, halo):
    """lat-wall (axis0) THEN lon-periodic (axis1) — the serial twin of
    the 2-D pad's N/S-then-E/W order (wall rows are constant so their
    lon-wrap is the same constant ⇒ corners = wall)."""
    n_lon = g.shape[1]
    sw = np.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    nw = np.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    latw = np.concatenate([sw, g, nw], axis=0)
    pads = [(0, 0)] * latw.ndim
    pads[1] = (halo, halo)
    return np.pad(latw, pads, mode="wrap")


def _serial_ref_jax(g, halo):
    """Differentiable twin of :func:`_serial_ref` — pad is a LINEAR
    gather (const wall rows + lon wrap), so jnp.pad(mode='wrap') gives
    the exact serial forward AND its adjoint for the grad reference."""
    n_lon = g.shape[1]
    sw = jnp.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    nw = jnp.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    latw = jnp.concatenate([sw, g, nw], axis=0)
    pads = [(0, 0)] * latw.ndim
    pads[1] = (halo, halo)
    return jnp.pad(latw, pads, mode="wrap")


def test_mpi_wall_pad_parity():
    """Each rank's wall 2-D pad == the corresponding window of the
    serial lat-wall+lon-periodic global pad."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks (single-proc path is in "
                    "tests/parallel/test_latlon_2d_pad_wall.py)")
    pr, pc = _pick_grid(n)
    h = 1
    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1     # uneven splits, min block 2
    g = np.random.default_rng(5).standard_normal((n_lat, n_lon))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(jnp.asarray(g), L)
    out = np.asarray(pad_halo_latlon_2d(
        local, L, halo=h, south_value=SV, north_value=NV))
    ref = _serial_ref(g, h)[
        L.lat_start:L.lat_end + 2 * h, L.lon_start:L.lon_end + 2 * h]
    np.testing.assert_allclose(
        out, ref, atol=1e-12,
        err_msg=f"rank {rank} wall 2-D pad != serial (pr={pr} pc={pc})")
    if rank == 0:
        print(f"WALL_2D_PAD_OK pr={pr} pc={pc} n_lat={n_lat} n_lon={n_lon}",
              flush=True)


def test_mpi_wall_grad_parity():
    """Reverse-mode AD through BOTH sendrecv VJPs.  The loss is
    ``global_sum_mpi(sum(pad_local**2))`` (AD-safe allreduce SUM) —
    WITHOUT the global sum, jax.grad of a rank's OWN loss collapses to
    ``2*local`` (ghost cotangents flow AWAY to neighbours and never
    re-enter ``d(own_loss)/d(own_block)``, leaving the VJP untested).
    Summing every rank's loss makes each neighbour's ghost-loss cotangent
    flow BACK through the halo VJP into this rank's owned cells, so the
    distributed grad on a rank's OWNED block == the GLOBAL grad of
    ``sum_r sum(P[window_r]**2)`` (P=_serial_ref_jax) restricted to that
    block.  ``proc_lon≥3`` (np=3, np=6) drives the periodic ring through
    REVERSE with the phase-constant tags it requires; ``proc_lat≥2``
    (np=6) drives the N/S reverse sendrecv."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    h = 1
    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1
    g = np.random.default_rng(7).standard_normal((n_lat, n_lon))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(jnp.asarray(g), L)

    def loss(x):
        local_loss = jnp.sum(pad_halo_latlon_2d(
            x, L, halo=h, south_value=SV, north_value=NV) ** 2)
        return global_sum_mpi(local_loss, comm, final_loss=True)

    grad_local = np.asarray(jax.grad(loss)(local))
    assert np.all(np.isfinite(grad_local)), \
        f"rank {rank} non-finite distributed grad"

    def total_serial(gg):
        P = _serial_ref_jax(gg, h)
        tot = 0.0
        for r in range(pr * pc):
            Lr = make_latlon_2d_layout(r, pr, pc, n_lat, n_lon)
            tot = tot + jnp.sum(
                P[Lr.lat_start:Lr.lat_end + 2 * h,
                  Lr.lon_start:Lr.lon_end + 2 * h] ** 2)
        return tot

    ref_grad = np.asarray(jax.grad(total_serial)(jnp.asarray(g)))
    want = ref_grad[L.lat_start:L.lat_end, L.lon_start:L.lon_end]
    np.testing.assert_allclose(
        grad_local, want, atol=1e-9,
        err_msg=f"rank {rank} distributed grad != serial ref (pr={pr} pc={pc})")
    if rank == 0:
        print(f"WALL_2D_GRAD_OK pr={pr} pc={pc} n_lat={n_lat} n_lon={n_lon}",
              flush=True)
