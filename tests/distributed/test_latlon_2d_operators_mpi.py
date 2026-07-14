"""MPI equivariance for the C-grid operators converted to a dispatched
LONGITUDE halo (``pad_lon_cgrid``) — the task #14 lon-op conversion that lets
the lat-lon C-grid step run on a 2-D ``(proc_lat x proc_lon)`` pencil.

Each converted operator (``gradient_x_cgrid``, ``interp_cell_to_uface``,
``curl_vertex_cgrid``) is run on this rank's 2-D block with the MPI 2-D
backend armed, and its output is compared to the WINDOW of the serial
(local-backend) operator on the global field — the same per-rank-window
strategy as ``test_latlon_2d_pad_wall_mpi``, so no face-field gather is
needed.  A genuine longitude split (proc_lon>1 for np={2,3,6}) drives the
``exchange_halo_lon`` ring; matching the serial window proves the local
``jnp.roll`` wrap was correctly replaced (and stays bit-identical where lon
is full).

The serial reference is computed FIRST on every rank under the LOCAL backend
(deterministic, collective-free) BEFORE arming the 2-D MPI backend — a serial
op traced after arming would embed sendrecvs no other rank matches and
deadlock (same ordering contract as ``test_latlon_mpi_step``).

Run: ``mpirun -np {2,3,6} python -m pytest <thisfile>``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    absolute_vorticity_coriolis,
)
from legoesm.grids.operators_latlon_cgrid import (
    compute_vertex_mask,
    curl_vertex_cgrid,
    gradient_x_cgrid,
    interp_cell_to_uface,
    laplacian_cgrid,
)
from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    slice_latlon_grid_to_block_2d,
)


def _pick_grid(n_ranks: int) -> tuple[int, int]:
    best = (1, n_ranks)
    for pr in range(2, int(n_ranks ** 0.5) + 1):
        if n_ranks % pr == 0:
            best = (pr, n_ranks // pr)
    return best


def test_2d_operators_match_serial_window():
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    grid = create_latlon_grid(8)                 # n_lat=8, n_lon=16
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(29)
    # Cell field f; u on lon-faces (n_lon+1, periodic closure); v on lat-faces.
    f = jnp.asarray(rng.standard_normal((n_lat, n_lon)))
    u = rng.standard_normal((n_lat, n_lon + 1))
    u[:, n_lon] = u[:, 0]
    u = jnp.asarray(u)
    v = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))

    # --- Serial references FIRST, every rank, LOCAL backend ---
    set_halo_backend("local")
    gx_s = np.asarray(gradient_x_cgrid(f, grid))            # (n_lat, n_lon+1)
    iu_s = np.asarray(interp_cell_to_uface(f))              # (n_lat, n_lon+1)
    cz_s = np.asarray(curl_vertex_cgrid(u, v, grid))        # (n_lat+1, n_lon+1)
    # absolute_vorticity_coriolis: the Coriolis v->u 4-pt average carried the
    # missed local lon roll that broke the 2-D u-momentum (dycore-gate
    # finding) — pin it at the operator level too.
    cu_s, cv_s = (np.asarray(a) for a in absolute_vorticity_coriolis(u, v, grid))
    gx_s = jax.block_until_ready(gx_s)

    # --- This rank's 2-D block ---
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    s, e, w, x = L.lat_start, L.lat_end, L.lon_start, L.lon_end
    gblk = slice_latlon_grid_to_block_2d(grid, L, skip_total_area_reduce=True)
    f_b = f[s:e, w:x]
    u_b = u[s:e, w:x + 1]              # lon-face: shared east column
    v_b = v[s:e + 1, w:x]             # lat-face: shared row

    set_halo_backend("mpi", L)
    try:
        gx_b = np.asarray(gradient_x_cgrid(f_b, gblk))
        iu_b = np.asarray(interp_cell_to_uface(f_b))
        cz_b = np.asarray(curl_vertex_cgrid(u_b, v_b, gblk))
        cu_b, cv_b = (np.asarray(a)
                      for a in absolute_vorticity_coriolis(u_b, v_b, gblk))
    finally:
        set_halo_backend("local")

    # Block output == serial output windowed to the block's faces.
    np.testing.assert_allclose(
        gx_b, gx_s[s:e, w:x + 1], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} gradient_x_cgrid != serial window (pr={pr} pc={pc})")
    np.testing.assert_allclose(
        iu_b, iu_s[s:e, w:x + 1], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} interp_cell_to_uface != serial window")
    np.testing.assert_allclose(
        cz_b, cz_s[s:e + 1, w:x + 1], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} curl_vertex_cgrid != serial window")
    np.testing.assert_allclose(
        cu_b, cu_s[s:e, w:x + 1], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} coriolis cor_u != serial window (the v->u 4-pt)")
    np.testing.assert_allclose(
        cv_b, cv_s[s:e + 1, w:x], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} coriolis cor_v != serial window")
    if rank == 0:
        print(f"OP_2D_EQUIVARIANCE_OK pr={pr} pc={pc}", flush=True)


def test_2d_masked_operators_match_serial_window():
    """The MASK-path lon ops converted alongside the dry ops — masked
    ``laplacian_cgrid`` (u_mask) and ``compute_vertex_mask`` (4-cell vertex
    product) — also match the serial window under a 2-D split (codex flagged
    these as latent local-roll footguns for masked 2-D runs)."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    grid = create_latlon_grid(8)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(37)
    f = jnp.asarray(rng.standard_normal((n_lat, n_lon)))
    # 0/1 land mask with both phases present.
    mask = jnp.asarray((rng.random((n_lat, n_lon)) > 0.3).astype(float))

    set_halo_backend("local")
    lap_s = np.asarray(laplacian_cgrid(f, grid, mask=mask))   # (n_lat, n_lon)
    vm_s = np.asarray(compute_vertex_mask(mask, grid))        # (n_lat+1, n_lon+1)
    lap_s = jax.block_until_ready(lap_s)

    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    s, e, w, x = L.lat_start, L.lat_end, L.lon_start, L.lon_end
    gblk = slice_latlon_grid_to_block_2d(grid, L, skip_total_area_reduce=True)
    f_b, m_b = f[s:e, w:x], mask[s:e, w:x]

    set_halo_backend("mpi", L)
    try:
        lap_b = np.asarray(laplacian_cgrid(f_b, gblk, mask=m_b))
        vm_b = np.asarray(compute_vertex_mask(m_b, gblk))
    finally:
        set_halo_backend("local")

    np.testing.assert_allclose(
        lap_b, lap_s[s:e, w:x], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} masked laplacian_cgrid != serial window")
    np.testing.assert_allclose(
        vm_b, vm_s[s:e + 1, w:x + 1], rtol=1e-7, atol=1e-9,
        err_msg=f"rank {rank} compute_vertex_mask != serial window")
    if rank == 0:
        print(f"MASK_OP_2D_OK pr={pr} pc={pc}", flush=True)


def test_2d_gradient_x_grad_finite():
    """Reverse-mode AD through the lon-ring exchange inside gradient_x stays
    finite under a real 2-D split (the exchange_halo_lon VJP is exercised via
    the operator)."""
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n < 2:
        pytest.skip("needs mpirun with >=2 ranks")
    pr, pc = _pick_grid(n)
    grid = create_latlon_grid(8)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
    s, e, w, x = L.lat_start, L.lat_end, L.lon_start, L.lon_end
    gblk = slice_latlon_grid_to_block_2d(grid, L, skip_total_area_reduce=True)
    f_b = jnp.asarray(np.random.default_rng(31 + rank).standard_normal(
        (e - s, x - w)))

    from legoesm.parallel.reductions import global_sum_mpi

    def loss(fb):
        # global_sum_mpi(sum**2) so neighbour cotangents flow back through the
        # lon-ring VJP into this rank's owned cells (else grad collapses to a
        # local form — same construction as the pad-wall AD test).
        return global_sum_mpi(jnp.sum(gradient_x_cgrid(fb, gblk) ** 2), comm)

    set_halo_backend("mpi", L)            # armed for the whole grad trace+run
    try:
        g = np.asarray(jax.grad(loss)(f_b))
    finally:
        set_halo_backend("local")
    assert np.all(np.isfinite(g)), f"rank {rank} non-finite gradient_x 2-D grad"
    if rank == 0:
        print(f"OP_2D_GRAD_OK pr={pr} pc={pc}", flush=True)
