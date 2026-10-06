"""MPI parity + AD for the lat-lon 2-D pencil DISPATCH + the lat-only wall
pad (``pad_with_pole_bc_lat_2d``).

Two layers, both run under ``mpirun -np {2,3,6} python -m pytest <thisfile>``
(wired into the MPI CI workflow next to ``test_latlon_2d_pad_wall_mpi.py``):

1. ``pad_with_pole_bc_lat_2d`` — the lat-axis-ONLY wall pad (the 2-D twin of
   the band ``pad_with_pole_bc_lat`` MPI path; longitude is NOT padded).  The
   full lat+lon ``pad_halo_latlon_2d`` is already MPI-tested; this pins the
   lat-only variant (interior-cut sendrecv + pole wall, lon untouched) +
   its sendrecv VJP.
2. The BACKEND DISPATCH: with ``set_halo_backend("mpi", LatLon2DLayout)``
   armed, the public ``halo_latlon`` entry points must route to the 2-D
   primitives under a REAL distributed layout (the 1×1 serial lane in
   ``tests/parallel/test_latlon_2d_dispatch_serial.py`` cannot exercise the
   sendrecv branch / deadlock-freedom).

Single rank => skip (the serial routing is the parallel-lane file).
Size-adaptive process grid (1×n ring for primes, balanced 2-D for
composites; uneven splits via n=2·p+1) — same factoring as
``test_latlon_2d_pad_wall_mpi.py``.
"""
from __future__ import annotations

from collections import namedtuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_with_pole_bc_lat,
)
from legoesm.parallel.latlon_mpi import (
    gather_field_latlon_2d,
    make_latlon_2d_layout,
    make_latlon_band_layout,
    pad_halo_latlon_2d,
    pad_halo_latlon_mpi,
    pad_with_pole_bc_lat_2d,
    scatter_field_latlon_2d,
    scatter_state_latlon_2d,
)
from legoesm.parallel.reductions import global_sum_mpi

SV, NV = -1.0, -2.0   # distinct non-zero walls to catch S/N confusion


def _pick_grid(n_ranks: int) -> tuple[int, int]:
    best = (1, n_ranks)
    for pr in range(2, int(n_ranks ** 0.5) + 1):
        if n_ranks % pr == 0:
            best = (pr, n_ranks // pr)
    return best


def _serial_lat_wall(g, halo):
    """lat-axis-only wall pad (NO lon pad) — the serial twin of
    ``pad_with_pole_bc_lat_2d``."""
    n_lon = g.shape[1]
    sw = np.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    nw = np.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    return np.concatenate([sw, g, nw], axis=0)


def _serial_lat_wall_jax(g, halo):
    n_lon = g.shape[1]
    sw = jnp.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    nw = jnp.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    return jnp.concatenate([sw, g, nw], axis=0)


def test_lat_only_wall_pad_parity():
    """Each rank's lat-only wall pad == the lat-wall global pad windowed to
    the rank's lat band AND its lon block (longitude NOT padded)."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks (serial lane covers 1×1)")
    pr, pc = _pick_grid(n)
    h = 1
    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1     # uneven splits, min block 2
    g = np.random.default_rng(11).standard_normal((n_lat, n_lon))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(jnp.asarray(g), L)
    out = np.asarray(pad_with_pole_bc_lat_2d(
        local, L, halo=h, south_value=SV, north_value=NV))
    ref = _serial_lat_wall(g, h)[
        L.lat_start:L.lat_end + 2 * h, L.lon_start:L.lon_end]
    np.testing.assert_allclose(
        out, ref, atol=1e-12,
        err_msg=f"rank {rank} lat-only wall pad != serial (pr={pr} pc={pc})")
    # lon width unchanged (lat-only pad), lat grew by 2h.
    assert out.shape == (L.n_lat_local + 2 * h, L.n_lon_local)
    if rank == 0:
        print(f"LATONLY_WALL_OK pr={pr} pc={pc}", flush=True)


def test_lat_only_wall_pad_grad_parity():
    """Reverse-mode AD through the lat-axis sendrecv VJP.  Loss is
    ``global_sum_mpi(sum(pad_local**2))`` so each neighbour's ghost-loss
    cotangent flows BACK through the halo VJP into this rank's owned cells
    (the same construction as ``test_latlon_2d_pad_wall_mpi`` — without the
    global sum a rank's own-loss grad collapses to ``2*local``)."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    h = 1
    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1
    g = np.random.default_rng(13).standard_normal((n_lat, n_lon))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(jnp.asarray(g), L)

    def loss(x):
        local_loss = jnp.sum(pad_with_pole_bc_lat_2d(
            x, L, halo=h, south_value=SV, north_value=NV) ** 2)
        return global_sum_mpi(local_loss, comm, final_loss=True)

    grad_local = np.asarray(jax.grad(loss)(local))
    assert np.all(np.isfinite(grad_local)), \
        f"rank {rank} non-finite distributed grad"

    def total_serial(gg):
        P = _serial_lat_wall_jax(gg, h)
        tot = 0.0
        for r in range(pr * pc):
            Lr = make_latlon_2d_layout(r, pr, pc, n_lat, n_lon)
            tot = tot + jnp.sum(
                P[Lr.lat_start:Lr.lat_end + 2 * h,
                  Lr.lon_start:Lr.lon_end] ** 2)
        return tot

    ref_grad = np.asarray(jax.grad(total_serial)(jnp.asarray(g)))
    want = ref_grad[L.lat_start:L.lat_end, L.lon_start:L.lon_end]
    np.testing.assert_allclose(
        grad_local, want, atol=1e-9,
        err_msg=f"rank {rank} lat-only grad != serial (pr={pr} pc={pc})")
    if rank == 0:
        print(f"LATONLY_GRAD_OK pr={pr} pc={pc}", flush=True)


def test_dispatch_routes_to_2d_under_mpi():
    """With the 2-D layout armed as the MPI backend, the public dispatch
    entry points must produce the SAME result as calling the 2-D primitive
    directly — i.e. the ``LatLon2DLayout`` branch fires (not the serial
    pole-fold / band path) AND its sendrecvs do not deadlock."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks (serial lane covers 1×1)")
    pr, pc = _pick_grid(n)
    h = 1
    n_lat, n_lon = 2 * pr + 1, 2 * pc + 1
    g = np.random.default_rng(17).standard_normal((n_lat, n_lon))
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(jnp.asarray(g), L)

    # Direct primitives (the reference the dispatch must reproduce).
    want_full = np.asarray(pad_halo_latlon_2d(local, L, halo=h, pole_bc="wall"))
    want_lat = np.asarray(pad_with_pole_bc_lat_2d(local, L, halo=h))

    set_halo_backend("mpi", L)
    try:
        got_full = np.asarray(pad_halo_latlon(local, halo=h))
        got_lat = np.asarray(pad_with_pole_bc_lat(local, halo=h))
    finally:
        set_halo_backend("local")

    np.testing.assert_allclose(
        got_full, want_full, atol=1e-12,
        err_msg=f"rank {rank} pad_halo_latlon dispatch != 2-D primitive")
    np.testing.assert_allclose(
        got_lat, want_lat, atol=1e-12,
        err_msg=f"rank {rank} pad_with_pole_bc_lat dispatch != 2-D primitive")
    if rank == 0:
        print(f"DISPATCH_2D_OK pr={pr} pc={pc}", flush=True)


def test_fold_family_reuses_band_fold_at_proc_lon1():
    """At proc_lon==1 (latitude split only, lon full per rank) the fold-family
    ``pad_halo_latlon`` dispatch must reuse the validated band POLE-FOLD —
    NOT the wall-pole pad.  Compared bit-for-bit against ``pad_halo_latlon_mpi``
    on the equivalent ``LatLonBandLayout`` (the path the 2-D dispatch delegates
    to).  Exercises the real lat sendrecv at interior cuts AND the local
    180-deg fold at pole-touching ranks under proc_lat>1 MPI."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = n, 1                              # latitude split only
    h = 1
    n_lat, n_lon = 2 * pr + 1, 6               # uneven lat split, full lon
    g = np.random.default_rng(19).standard_normal((n_lat, n_lon))
    L2d = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    local = scatter_field_latlon_2d(jnp.asarray(g), L2d)
    band_ref = make_latlon_band_layout(rank, pr, n_lat, n_lon)

    # Reference: the validated band fold on the equivalent band layout
    # (computed BEFORE arming the 2-D backend; both legs sendrecv in the
    # same SPMD-symmetric order so there is no deadlock).
    want = np.asarray(pad_halo_latlon_mpi(local, band_ref, halo=h))

    set_halo_backend("mpi", L2d)
    try:
        got = np.asarray(pad_halo_latlon(local, halo=h))
    finally:
        set_halo_backend("local")

    np.testing.assert_allclose(
        got, want, atol=1e-12,
        err_msg=f"rank {rank} proc_lon==1 fold dispatch != band fold")
    # lat+lon halo present (full 2-D pad shape), lon wrapped locally.
    assert got.shape == (L2d.n_lat_local + 2 * h, n_lon + 2 * h)
    if rank == 0:
        print(f"FOLD_REUSE_OK pr={pr}", flush=True)


_St = namedtuple("_St", "u v T p_s phis tracers")


def test_scatter_gather_state_faces_roundtrip():
    """scatter_state_latlon_2d -> gather_field_latlon_2d reproduces the global
    C-grid state EXACTLY for the staggered faces under a real 2-D split: u is
    a lon-face (n_lon+1, shared east column, periodic closure) gathered with
    ``is_u_face``; v is a lat-face (n_lat+1, shared row) gathered with
    ``is_v_face``; cell fields tile plainly.  Guards the u-face scatter fix +
    the gather lon-face mode (codex round-2: the old gather overran for u)."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    n_lat, n_lon, nlev = 2 * pr, 2 * pc, 2
    rng = np.random.default_rng(23)
    T = rng.standard_normal((n_lat, n_lon, nlev))
    u = rng.standard_normal((n_lat, n_lon + 1, nlev))
    u[:, n_lon, :] = u[:, 0, :]                       # periodic closure invariant
    v = rng.standard_normal((n_lat + 1, n_lon, nlev))
    p_s = rng.standard_normal((n_lat, n_lon))
    g = _St(u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.asarray(T),
            p_s=jnp.asarray(p_s), phis=jnp.zeros((n_lat, n_lon)), tracers={})

    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    loc = scatter_state_latlon_2d(g, L)
    u_g = gather_field_latlon_2d(loc.u, L, is_u_face=True)
    v_g = gather_field_latlon_2d(loc.v, L, is_v_face=True)
    T_g = gather_field_latlon_2d(loc.T, L)

    np.testing.assert_allclose(np.asarray(u_g), u, atol=1e-12,
                               err_msg=f"rank {rank} u-face roundtrip")
    np.testing.assert_allclose(np.asarray(v_g), v, atol=1e-12,
                               err_msg=f"rank {rank} v-face roundtrip")
    np.testing.assert_allclose(np.asarray(T_g), T, atol=1e-12,
                               err_msg=f"rank {rank} cell roundtrip")
    if rank == 0:
        print(f"FACE_ROUNDTRIP_OK pr={pr} pc={pc}", flush=True)
