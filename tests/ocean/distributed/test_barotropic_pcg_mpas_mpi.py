"""np=2 parity for the DISTRIBUTED MPAS fixed-M PCG (Voronoi partition).

Run:  mpirun -np 2 python -m pytest \
          tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py

Exercises exactly the machinery ``barotropic_implicit_mpas`` assembles
on the distributed path — the halo-composed TRiSK Helmholtz ``A_op``,
the owned-cell-masked area-weighted ``dot_weight``, and the shared
fixed-M PCG — at the SOLVER level (raw cell arrays; the full
ocean-state scatter is a separate follow-on).  The serial reference is
the same solve on the GLOBAL mesh, computed independently on every
rank (no communication), fully converged on both sides.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

mpi4py = pytest.importorskip("mpi4py")
pytest.importorskip("mpi4jax")

from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK, SIZE = comm.Get_rank(), comm.Get_size()

pytestmark = pytest.mark.skipif(SIZE < 2, reason="needs mpirun -np 2")

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.barotropic_common import (
    solve_helmholtz_implicit,
)
from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
    _make_diag_preconditioner,
    _make_helmholtz,
)

N_CELLS_RES = 3        # icosahedral refinement level for create_voronoi_mesh
COEFF = 5.0e6
M_ITERS = 150          # generous => fully converged both sides
TOL = 1.0e-9


def _global_problem():
    mesh = create_voronoi_mesh(N_CELLS_RES)
    n_cells = int(mesh.areaCell.shape[0])
    n_edges = int(mesh.cellsOnEdge.shape[1])
    rng = np.random.default_rng(7)
    mask = jnp.ones((n_cells,))
    H_e = jnp.asarray(2000.0 + 1000.0 * rng.random((n_edges,)))
    rhs = jnp.asarray(rng.standard_normal((n_cells,)))
    return mesh, mask, H_e, rhs


def test_distributed_mpas_pcg_matches_serial():
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    mesh, mask, H_e, rhs = _global_problem()
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    coeff = jnp.asarray(COEFF)

    # ---- Serial reference: global-mesh PCG, every rank, no comm. ----
    A_glob = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    M_glob = _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask)
    w_glob = mesh.areaCell.astype(rhs.dtype) * mask
    eta_ref, diag_ref = solve_helmholtz_implicit(
        A_glob, rhs, M_glob, jnp.zeros_like(rhs),
        distributed=False, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
    )
    eta_ref = np.asarray(eta_ref)

    # ---- Partition + rank-local problem (same slicing the driver
    # does: layout.local_mesh + global-id gathers of the inputs). ----
    _rank, _n, layout = initialize_voronoi_mpi(mesh)
    assert get_active_voronoi_layout() is layout
    part = layout.partition
    lmesh = layout.local_mesh
    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)

    mask_l = jnp.asarray(np.asarray(mask)[lc])
    rhs_l = jnp.asarray(np.asarray(rhs)[lc])
    H_e_l = jnp.asarray(np.asarray(H_e)[le])
    c1l, c2l = lmesh.cellsOnEdge[0], lmesh.cellsOnEdge[1]
    edge_mask_l = mask_l[c1l] * mask_l[c2l]

    A_loc = _make_helmholtz(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    M_loc = _make_diag_preconditioner(
        H_e_l, coeff, lmesh, mask_l, edge_mask_l)

    # ---- EXACTLY the production distributed assembly. ----
    exchanger = VoronoiHaloExchange(part, backend="mpi")

    def A_dist(x):
        return A_loc(exchanger.exchange_cell_field(x))

    owned = layout.owned_mask_cells.astype(rhs_l.dtype)
    w_dots = owned * lmesh.areaCell.astype(rhs_l.dtype) * mask_l
    eta_loc, diag = solve_helmholtz_implicit(
        A_dist, rhs_l, M_loc, jnp.zeros_like(rhs_l),
        distributed=True, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
        dot_weight=w_dots,
    )
    eta_loc = exchanger.exchange_cell_field(eta_loc)
    jax.block_until_ready(eta_loc)
    assert bool(diag.converged), f"rel_residual={float(diag.rel_residual)}"

    # ---- Gather owned cells to the global layout on rank 0. ----
    n_owned = int(part.n_owned_cells)
    owned_ids = lc[:n_owned]
    owned_vals = np.asarray(eta_loc)[:n_owned]
    pieces = comm.gather((owned_ids, owned_vals), root=0)
    if RANK != 0:
        return
    eta_glob = np.full(eta_ref.shape, np.nan)
    for ids, vals in pieces:
        eta_glob[ids] = vals
    assert not np.isnan(eta_glob).any(), "gather left unowned cells"
    np.testing.assert_allclose(
        eta_glob, eta_ref, rtol=TOL, atol=TOL,
        err_msg="distributed MPAS PCG diverged from the serial solve",
    )


def test_distributed_mpas_pcg_single_reduce_variant():
    """Same parity through the single_reduce body (W-dots required)."""
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    mesh, mask, H_e, rhs = _global_problem()
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    coeff = jnp.asarray(COEFF)

    A_glob = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    M_glob = _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask)
    eta_ref, _ = solve_helmholtz_implicit(
        A_glob, rhs, M_glob, jnp.zeros_like(rhs),
        distributed=False, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
    )
    eta_ref = np.asarray(eta_ref)

    layout = get_active_voronoi_layout()
    if layout is None:
        _r, _n, layout = initialize_voronoi_mpi(mesh)
    part = layout.partition
    lmesh = layout.local_mesh
    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)
    mask_l = jnp.asarray(np.asarray(mask)[lc])
    rhs_l = jnp.asarray(np.asarray(rhs)[lc])
    H_e_l = jnp.asarray(np.asarray(H_e)[le])
    c1l, c2l = lmesh.cellsOnEdge[0], lmesh.cellsOnEdge[1]
    edge_mask_l = mask_l[c1l] * mask_l[c2l]
    A_loc = _make_helmholtz(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    M_loc = _make_diag_preconditioner(
        H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    exchanger = VoronoiHaloExchange(part, backend="mpi")

    def A_dist(x):
        return A_loc(exchanger.exchange_cell_field(x))

    owned = layout.owned_mask_cells.astype(rhs_l.dtype)
    w_dots = owned * lmesh.areaCell.astype(rhs_l.dtype) * mask_l
    eta_loc, diag = solve_helmholtz_implicit(
        A_dist, rhs_l, M_loc, jnp.zeros_like(rhs_l),
        distributed=True, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
        pcg_variant="single_reduce", dot_weight=w_dots,
    )
    jax.block_until_ready(eta_loc)
    assert bool(diag.converged)

    n_owned = int(part.n_owned_cells)
    pieces = comm.gather(
        (lc[:n_owned], np.asarray(eta_loc)[:n_owned]), root=0)
    if RANK != 0:
        return
    eta_glob = np.full(eta_ref.shape, np.nan)
    for ids, vals in pieces:
        eta_glob[ids] = vals
    np.testing.assert_allclose(eta_glob, eta_ref, rtol=TOL, atol=TOL)


def test_local_Aop_matches_global_restriction():
    """Tripwire: the halo-composed LOCAL A_op (and the Jacobi diag) must
    equal the GLOBAL operator restricted to owned cells — isolates
    operator-assembly errors from solver errors."""
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    mesh, mask, H_e, rhs = _global_problem()
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    coeff = jnp.asarray(COEFF)
    A_glob = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    M_glob = _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask)

    layout = get_active_voronoi_layout()
    if layout is None:
        _r, _n, layout = initialize_voronoi_mpi(mesh)
    part = layout.partition
    lmesh = layout.local_mesh
    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)
    mask_l = jnp.asarray(np.asarray(mask)[lc])
    H_e_l = jnp.asarray(np.asarray(H_e)[le])
    c1l, c2l = lmesh.cellsOnEdge[0], lmesh.cellsOnEdge[1]
    edge_mask_l = mask_l[c1l] * mask_l[c2l]
    A_loc = _make_helmholtz(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    M_loc = _make_diag_preconditioner(
        H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    exchanger = VoronoiHaloExchange(part, backend="mpi")

    rng = np.random.default_rng(99)
    x_glob = jnp.asarray(rng.standard_normal(int(mesh.areaCell.shape[0])))
    x_loc = jnp.asarray(np.asarray(x_glob)[lc])

    y_loc = A_loc(exchanger.exchange_cell_field(x_loc))
    y_glob_restr = np.asarray(A_glob(x_glob))[lc]
    n_owned = int(part.n_owned_cells)
    np.testing.assert_allclose(
        np.asarray(y_loc)[:n_owned], y_glob_restr[:n_owned],
        rtol=1e-12, atol=1e-12,
        err_msg="local halo-composed A_op != global restriction on owned",
    )

    ones_loc = jnp.ones_like(x_loc)
    np.testing.assert_allclose(
        np.asarray(M_loc(ones_loc))[:n_owned],
        np.asarray(M_glob(jnp.ones_like(x_glob)))[lc][:n_owned],
        rtol=1e-12, atol=1e-12,
        err_msg="local Jacobi diag != global restriction on owned",
    )


def test_triangulate_solver_disagreement():
    """Diagnostic triangulation: serial stock-CG vs serial fixed-M PCG
    vs np=2 distributed PCG — identifies WHICH solve diverges."""
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    mesh, mask, H_e, rhs = _global_problem()
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    coeff = jnp.asarray(COEFF)
    A_glob = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    M_glob = _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask)
    w_glob = mesh.areaCell.astype(rhs.dtype) * mask

    eta_stock, _ = solve_helmholtz_implicit(
        A_glob, rhs, M_glob, jnp.zeros_like(rhs),
        distributed=False, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
    )
    # Single-process 'distributed' PCG on the GLOBAL mesh (local dots —
    # is_multi_process() False): the fixed-M body without any halo
    # machinery.
    eta_spcg, _ = solve_helmholtz_implicit(
        A_glob, rhs, M_glob, jnp.zeros_like(rhs),
        distributed=True, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
        dot_weight=w_glob,
    )
    d_stock_spcg = float(jnp.max(jnp.abs(eta_stock - eta_spcg)))

    layout = get_active_voronoi_layout()
    if layout is None:
        _r, _n, layout = initialize_voronoi_mpi(mesh)
    part = layout.partition
    lmesh = layout.local_mesh
    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)
    mask_l = jnp.asarray(np.asarray(mask)[lc])
    rhs_l = jnp.asarray(np.asarray(rhs)[lc])
    H_e_l = jnp.asarray(np.asarray(H_e)[le])
    c1l, c2l = lmesh.cellsOnEdge[0], lmesh.cellsOnEdge[1]
    edge_mask_l = mask_l[c1l] * mask_l[c2l]
    A_loc = _make_helmholtz(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    M_loc = _make_diag_preconditioner(
        H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    exchanger = VoronoiHaloExchange(part, backend="mpi")

    def A_dist(x):
        return A_loc(exchanger.exchange_cell_field(x))

    owned = layout.owned_mask_cells.astype(rhs_l.dtype)
    w_dots = owned * lmesh.areaCell.astype(rhs_l.dtype) * mask_l
    eta_loc, diag = solve_helmholtz_implicit(
        A_dist, rhs_l, M_loc, jnp.zeros_like(rhs_l),
        distributed=True, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
        dot_weight=w_dots,
    )
    jax.block_until_ready(eta_loc)

    n_owned = int(part.n_owned_cells)
    pieces = comm.gather(
        (lc[:n_owned], np.asarray(eta_loc)[:n_owned]), root=0)
    if RANK != 0:
        return
    eta_glob = np.full(np.asarray(eta_stock).shape, np.nan)
    for ids, vals in pieces:
        eta_glob[ids] = vals
    d_spcg_dist = float(np.max(np.abs(eta_glob - np.asarray(eta_spcg))))
    d_stock_dist = float(np.max(np.abs(eta_glob - np.asarray(eta_stock))))
    print(f"\nTRIANGULATE: |stock-serialPCG|={d_stock_spcg:.3e}  "
          f"|serialPCG-dist|={d_spcg_dist:.3e}  "
          f"|stock-dist|={d_stock_dist:.3e}  "
          f"dist_rel_res={float(diag.rel_residual):.3e}")
    assert d_spcg_dist <= 1.0e-9, (
        f"distributed PCG != serial fixed-M PCG: {d_spcg_dist:.3e}"
    )


def test_deep_halo_variant_matches_per_iteration_exchange():
    """The deep-halo variant on the MPI-per-rank partition (certifies 1 ring,
    so (r, s) are refreshed every iteration) equals the per-iteration
    exchange to round-off at an UNCONVERGED iteration count; claiming two
    rings more than certified must break it.  (One more does NOT break it
    HERE: this fixture slices H_e from the global mesh, so the outer ring's
    diagonal is exact; in the model H_e is built locally and is wrong on the
    outer ring's cut edges, which is why the certificate requires every
    neighbour to be local.  Measured: +1 ring -> 4.4e-16.)"""
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    mesh, mask, H_e, rhs = _global_problem()
    coeff = jnp.asarray(COEFF)
    layout = get_active_voronoi_layout()
    if layout is None:
        _r, _n, layout = initialize_voronoi_mpi(mesh)
    rings = int(layout.complete_cell_rings)
    assert rings == 1
    part = layout.partition
    lmesh = layout.local_mesh
    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)
    mask_l = jnp.asarray(np.asarray(mask)[lc])
    rhs_l = jnp.asarray(np.asarray(rhs)[lc])
    H_e_l = jnp.asarray(np.asarray(H_e)[le])
    c1l, c2l = lmesh.cellsOnEdge[0], lmesh.cellsOnEdge[1]
    edge_mask_l = mask_l[c1l] * mask_l[c2l]
    A_loc = _make_helmholtz(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    M_loc = _make_diag_preconditioner(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    ex = VoronoiHaloExchange(part, backend="mpi")
    owned = layout.owned_mask_cells.astype(rhs_l.dtype)
    w_dots = owned * lmesh.areaCell.astype(rhs_l.dtype) * mask_l
    kw = dict(distributed=True, fixed_iters=7, residual_tol=1.0e-9,
              stock_cg_tol=1.0e-12, stock_cg_maxiter=600, dot_weight=w_dots)
    x0 = jnp.zeros_like(rhs_l)

    def exch(*fs):
        return tuple(ex.exchange_cell_field(f) for f in fs)

    ref, _ = solve_helmholtz_implicit(
        lambda x: A_loc(ex.exchange_cell_field(x)), rhs_l, M_loc, x0,
        pcg_variant="single_reduce", **kw)
    deep, _ = solve_helmholtz_implicit(
        A_loc, rhs_l, M_loc, x0, pcg_variant="single_reduce_deep",
        deep_halo=(exch, owned, rings), **kw)
    bad, _ = solve_helmholtz_implicit(
        A_loc, rhs_l, M_loc, x0, pcg_variant="single_reduce_deep",
        deep_halo=(exch, owned, rings + 2), **kw)
    n = int(part.n_owned_cells)
    gap = comm.allreduce(float(np.max(np.abs(np.asarray(ref - deep)[:n]))), op=MPI.MAX)
    mut = comm.allreduce(float(np.max(np.abs(np.asarray(ref - bad)[:n]))), op=MPI.MAX)
    scale = comm.allreduce(float(np.max(np.abs(np.asarray(ref)[:n]))), op=MPI.MAX)
    assert gap <= 1e-12 * scale, (gap, scale)
    assert mut > 1e-6 * scale, (mut, scale)


def test_chebyshev_deep_on_mpi_lane_matches_serial_and_overclaim_breaks():
    """chebyshev_deep on the MPI-per-rank partition (1 certified ring; its one
    global max goes through global_max_mpi) converges to the serial solve, and
    claiming two rings more than certified changes the iterate."""
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    from legoesm.parallel.voronoi_mpi import (
        get_active_voronoi_layout,
        initialize_voronoi_mpi,
    )

    mesh, mask, H_e, rhs = _global_problem()
    coeff = jnp.asarray(COEFF)
    edge_mask = mask[mesh.cellsOnEdge[0]] * mask[mesh.cellsOnEdge[1]]
    eta_ref, _ = solve_helmholtz_implicit(
        _make_helmholtz(H_e, coeff, mesh, mask, edge_mask), rhs,
        _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask),
        jnp.zeros_like(rhs), distributed=False, fixed_iters=M_ITERS,
        residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600)
    eta_ref = np.asarray(eta_ref)

    layout = get_active_voronoi_layout()
    if layout is None:
        _r, _n, layout = initialize_voronoi_mpi(mesh)
    rings = int(layout.complete_cell_rings)
    part, lmesh = layout.partition, layout.local_mesh
    lc, le = np.asarray(part.local_cells), np.asarray(part.local_edges)
    mask_l = jnp.asarray(np.asarray(mask)[lc])
    rhs_l = jnp.asarray(np.asarray(rhs)[lc])
    H_e_l = jnp.asarray(np.asarray(H_e)[le])
    edge_mask_l = mask_l[lmesh.cellsOnEdge[0]] * mask_l[lmesh.cellsOnEdge[1]]
    A_loc = _make_helmholtz(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    M_loc = _make_diag_preconditioner(H_e_l, coeff, lmesh, mask_l, edge_mask_l)
    ex = VoronoiHaloExchange(part, backend="mpi")
    owned = layout.owned_mask_cells.astype(rhs_l.dtype)
    w_dots = owned * lmesh.areaCell.astype(rhs_l.dtype) * mask_l
    x0 = jnp.zeros_like(rhs_l)

    def exch(*fs):
        return tuple(ex.exchange_cell_field(f) for f in fs)

    def cheb(n, r):
        return solve_helmholtz_implicit(
            A_loc, rhs_l, M_loc, x0, distributed=True, fixed_iters=n,
            residual_tol=1.0e-9, stock_cg_tol=1.0e-12, stock_cg_maxiter=600,
            pcg_variant="chebyshev_deep", dot_weight=w_dots, deep_halo=(exch, owned, r))

    eta, diag = cheb(400, rings)
    assert bool(diag.converged), f"rel_residual={float(diag.rel_residual)}"
    n = int(part.n_owned_cells)
    good, _ = cheb(7, rings)
    bad, _ = cheb(7, rings + 2)
    mut = comm.allreduce(float(np.max(np.abs(np.asarray(good - bad)[:n]))), op=MPI.MAX)
    scale = comm.allreduce(float(np.max(np.abs(np.asarray(good)[:n]))), op=MPI.MAX)
    assert mut > 1e-6 * scale, (mut, scale)
    pieces = comm.gather((lc[:n], np.asarray(eta)[:n]), root=0)
    if RANK != 0:
        return
    eta_glob = np.full(eta_ref.shape, np.nan)
    for ids, vals in pieces:
        eta_glob[ids] = vals
    assert not np.isnan(eta_glob).any(), "gather left unowned cells"
    np.testing.assert_allclose(eta_glob, eta_ref, rtol=TOL, atol=TOL)
