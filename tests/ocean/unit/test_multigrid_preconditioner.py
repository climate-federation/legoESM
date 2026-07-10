"""Anisotropic geometric-multigrid barotropic preconditioner (task #26).

POC 8487762 / commit 81673f05: a geometric V-cycle with a ZONAL-LINE smoother
cuts the barotropic-PCG outer iteration count from M~60 (jacobi) to M~2-4 on
the polar-anisotropic lat-lon Helmholtz — the textbook O(log n) multigrid
convergence the pointwise-jacobi smoother cannot achieve (it gives ~3x).  This
pins the PRODUCTION ``_make_multigrid_preconditioner`` on a COASTAL grid (the
real case, with land masks in the transfer operators):

  * MG at a SMALL fixed M reaches (or beats) jacobi's M=60 residual — the
    iteration-count cut that beats the barotropic reduction-latency wall.
  * MG residual is monotone-decreasing in M (a valid preconditioned solve).
  * Non-vacuity: jacobi/M60 is NOT already at machine precision, so the cut
    is real.

Also asserts the transfer pair is Galerkin-symmetric (R = 0.25 P^T) on the
masked grid (the property that keeps the V-cycle CG-friendly).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _make_helmholtz, _helmholtz_inv_diag, _faces_from_cell_depth,
    _faces_from_cell_depth_banded,
    _make_multigrid_preconditioner, _make_multigrid_preconditioner_banded,
    _mg_restrict, _mg_prolong,
    _coarse_band_layout, _coarse_band_hierarchy,
)
from legoesm.parallel.latlon_mpi import (
    LatLonBandLayout, make_latlon_band_layout,
)
from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit

N_LAT, N_LON = 48, 96
COEFF = 5.0e7


def _setup():
    grid = ensure_geometry(create_latlon_grid(N_LAT, N_LON))
    rng = np.random.default_rng(13)
    H_cell = jnp.asarray(1000.0 + 500.0 * rng.random((N_LAT, N_LON)))
    m = np.ones((N_LAT, N_LON))
    m[0, :] = 0.0
    m[-1, :] = 0.0                       # pole rows
    m[:, 12:16] = 0.0                    # meridional coast
    m[20:26, 40:55] = 0.0               # interior basin
    mask = jnp.asarray(m)
    H_u, H_v, u_mask, v_mask = _faces_from_cell_depth(H_cell, mask, N_LAT, N_LON)
    coeff = jnp.asarray(COEFF)
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    return grid, coeff, mask, H_cell, A_op, inv_diag


def test_multigrid_transfer_galerkin_symmetric():
    """R == 0.25 * P^T on the masked grid: <R rf, xc> == 0.25 <rf, P xc>."""
    _, _, mask, _, _, _ = _setup()
    nlc, nloc = N_LAT // 2, N_LON // 2
    mask_c = (_mg_restrict(mask * 4.0, jnp.ones_like(mask), nlc, nloc) > 0.5
              ).astype(mask.dtype)
    rng = np.random.default_rng(5)
    rf = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    xc = jnp.asarray(rng.standard_normal((nlc, nloc))) * mask_c
    lhs = float(jnp.sum(_mg_restrict(rf, mask, nlc, nloc) * xc))
    rhs = 0.25 * float(jnp.sum(rf * _mg_prolong(xc, mask)))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs), 1.0), (
        f"R != 0.25 P^T: {lhs} vs {rhs}")


def test_multigrid_beats_jacobi_iteration_count():
    """MG at small M reaches jacobi-M60 accuracy — the O(log n) M-cut."""
    grid, coeff, mask, H_cell, A_op, inv_diag = _setup()
    w = grid.area * mask
    rng = np.random.default_rng(1)
    rhs = jnp.asarray(rng.standard_normal((N_LAT, N_LON))) * mask
    x0 = jnp.zeros_like(rhs)
    mg = _make_multigrid_preconditioner(H_cell, coeff, grid, mask)

    def _resid(M_inv, M):
        _, diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, distributed=True, fixed_iters=M,
            residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
            pcg_variant="standard", dot_weight=w)
        return float(diag.rel_residual)

    def jac(r):
        return r * inv_diag.astype(r.dtype)

    jac_m60 = _resid(jac, 60)
    mg_m6 = _resid(mg, 6)
    mg_m4 = _resid(mg, 4)
    mg_m2 = _resid(mg, 2)
    # Non-vacuity: jacobi M60 not already at machine precision.
    assert jac_m60 > 1e-8, f"jacobi M60 too converged ({jac_m60}) — vacuous"
    # The lever: MG at M<=6 beats jacobi at M=60.
    assert mg_m6 <= jac_m60, (
        f"MG/M6 ({mg_m6:.2e}) should beat jacobi/M60 ({jac_m60:.2e})")
    # Monotone decreasing (valid preconditioned solve).
    assert mg_m4 < mg_m2 and mg_m6 < mg_m4, (
        f"MG residual not monotone: M2={mg_m2:.2e} M4={mg_m4:.2e} M6={mg_m6:.2e}")
    # Finite (no NaN from the masked transfers / coarse solve).
    assert np.isfinite(mg_m2) and np.isfinite(mg_m6)


def test_multigrid_dispatch_wiring():
    """_select_preconditioner('multigrid', ..., H_cell=...) returns a working
    V-cycle M_inv (the production dispatch path); missing H_cell fails loud."""
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        _select_preconditioner,
    )
    grid, coeff, mask, H_cell, A_op, inv_diag = _setup()
    H_u, H_v, _, _ = _faces_from_cell_depth(H_cell, mask, N_LAT, N_LON)
    M_inv = _select_preconditioner(
        "multigrid", inv_diag, H_u, H_v, coeff, grid, mask,
        A_op=A_op, H_cell=H_cell)
    r = jnp.asarray(np.random.default_rng(2).standard_normal((N_LAT, N_LON))) * mask
    out = M_inv(r)
    assert out.shape == r.shape and np.all(np.isfinite(np.asarray(out)))
    # V-cycle is a real approximate solve: M_inv(r) is a non-trivial response.
    assert float(jnp.max(jnp.abs(out))) > 0.0
    # H_cell required (dispatch-hardening fail-loud).
    with pytest.raises(ValueError, match="multigrid preconditioner requires"):
        _select_preconditioner("multigrid", inv_diag, H_u, H_v, coeff, grid,
                               mask, A_op=A_op, H_cell=None)


def test_multigrid_dispatch_banded_under_mpi(monkeypatch):
    """_select_preconditioner('multigrid', layout=...) auto-routes to the BANDED
    factory when is_distributed(); a missing band layout fails loud (the serial
    rank-local V-cycle is not halo-aware)."""
    import legoesm.core.operators as _ops
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        _select_preconditioner,
    )
    grid, coeff, mask, H_cell, A_op, inv_diag = _setup()
    H_u, H_v, _, _ = _faces_from_cell_depth(H_cell, mask, N_LAT, N_LON)
    monkeypatch.setattr(_ops, "is_distributed", lambda: True)
    # With an (equal, n_ranks=1) band layout -> banded V-cycle M_inv.  Under the
    # local halo backend pad_halo_latlon pole-folds, so this builds serially.
    layout = make_latlon_band_layout(0, 1, N_LAT, N_LON)
    M_inv = _select_preconditioner(
        "multigrid", inv_diag, H_u, H_v, coeff, grid, mask,
        A_op=A_op, H_cell=H_cell, layout=layout)
    r = jnp.asarray(np.random.default_rng(4).standard_normal((N_LAT, N_LON))) * mask
    out = M_inv(r)
    assert out.shape == r.shape and np.all(np.isfinite(np.asarray(out)))
    assert float(jnp.max(jnp.abs(out))) > 0.0
    # Missing band layout under MPI -> fail loud (no silent serial fallback).
    with pytest.raises(ValueError, match="needs the LatLonBandLayout"):
        _select_preconditioner(
            "multigrid", inv_diag, H_u, H_v, coeff, grid, mask,
            A_op=A_op, H_cell=H_cell, layout=None)


def test_multigrid_refuses_distributed(monkeypatch):
    """SCOPE guard (codex 38fec66b HIGH #2): the rank-local 2x2 transfers are
    not halo-aware, so 'multigrid' must FAIL LOUD under band MPI / SPMD rather
    than silently build a wrong coarse problem."""
    import legoesm.core.operators as _ops
    grid, coeff, mask, H_cell, _, _ = _setup()
    monkeypatch.setattr(_ops, "is_distributed", lambda: True)
    with pytest.raises(ValueError, match="not yet halo-aware under band MPI"):
        _make_multigrid_preconditioner(H_cell, coeff, grid, mask)


def test_multigrid_refuses_spmd_backend():
    """route-B guard gap: lat-lon SPMD arms the 'spmd' halo backend with
    is_distributed()==False (single process) — the 2x2 restriction still
    straddles shards, so 'multigrid' must refuse on the spmd backend too."""
    from legoesm.grids.halo import (
        get_halo_backend, set_halo_backend,
    )
    grid, coeff, mask, H_cell, _, _ = _setup()
    prev = get_halo_backend()
    set_halo_backend("spmd")
    try:
        with pytest.raises(ValueError, match="band MPI / lat-lon SPMD"):
            _make_multigrid_preconditioner(H_cell, coeff, grid, mask)
    finally:
        set_halo_backend(prev)


# ---------------------------------------------------------------------------
# Banded-distributed MG foundation (task #26): _coarse_band_layout
# ---------------------------------------------------------------------------
def test_coarse_band_layout_even_aligned_halves():
    """An EVEN-ALIGNED band 2x-coarsens to a band with every size halved, the
    rank topology (rank/n_ranks/neighbours) preserved, and ``fold`` dropped —
    so the coarse 2x2 restriction stays band-local (the distributed-MG
    requirement)."""
    # 2-rank even split of 48x96: each band is 24 rows, lat_start in {0,24}.
    for rank in (0, 1):
        fine = make_latlon_band_layout(rank, 2, N_LAT, N_LON)
        assert fine.lat_start % 2 == 0 and fine.n_lat_local % 2 == 0  # even-aligned
        coarse = _coarse_band_layout(fine)
        assert coarse is not None
        assert coarse.n_lat_global == N_LAT // 2
        assert coarse.n_lon_global == N_LON // 2
        assert coarse.n_lat_local == fine.n_lat_local // 2
        assert coarse.lat_start == fine.lat_start // 2
        assert coarse.lat_end == fine.lat_end // 2
        # band stays self-consistent: lat_end == lat_start + n_lat_local.
        assert coarse.lat_end == coarse.lat_start + coarse.n_lat_local
        # rank topology untouched (transfers reuse the same neighbour cuts).
        assert (coarse.rank, coarse.n_ranks) == (fine.rank, fine.n_ranks)
        assert coarse.south_rank == fine.south_rank
        assert coarse.north_rank == fine.north_rank
        assert coarse.fold is None


def test_coarse_band_layout_partition_is_a_cover():
    """The per-rank coarse bands TILE the coarse global grid with no gap/overlap
    (sum of coarse local rows == coarse global rows; bands are contiguous) — the
    invariant that makes the band-local restriction equal the global one."""
    n_ranks = 4
    n_lat = 64  # 64/4 = 16 rows/rank, all even-aligned
    coarse = [_coarse_band_layout(make_latlon_band_layout(r, n_ranks, n_lat, N_LON))
              for r in range(n_ranks)]
    assert all(c is not None for c in coarse)
    assert sum(c.n_lat_local for c in coarse) == n_lat // 2
    # contiguous cover: each band starts where the previous ended.
    assert coarse[0].lat_start == 0
    for r in range(1, n_ranks):
        assert coarse[r].lat_start == coarse[r - 1].lat_end
    assert coarse[-1].lat_end == n_lat // 2


@pytest.mark.parametrize("field,bad", [
    ("lat_start", 1),      # band offset odd -> 2x2 straddles a rank cut
    ("n_lat_local", 3),    # odd row count -> last coarse row has 1 fine child
    ("n_lat_global", 47),  # odd global lat -> coarse grid ill-defined
    ("n_lon_global", 95),  # odd global lon -> coarse grid ill-defined
])
def test_coarse_band_layout_odd_returns_none(field, bad):
    """A band that is NOT even-aligned in any of the 4 size axes returns
    ``None`` (the caller must then refuse / fall back rather than build a wrong
    coarse problem) — the fail-safe for the offset-halo case we don't yet
    handle."""
    base = make_latlon_band_layout(0, 1, N_LAT, N_LON)  # global, even
    bad_layout = base._replace(**{field: bad})
    assert _coarse_band_layout(bad_layout) is None


def test_coarse_band_hierarchy_single_rank_halving_chain():
    """Single-rank 48-row grid 2x-coarsens 48->24->12 with min_coarse_rows=8
    (12 < 2*8 so 6 is not reached): a fine->coarsest list whose local rows
    halve each level and whose first entry is the input."""
    h = _coarse_band_hierarchy(make_latlon_band_layout(0, 1, N_LAT, N_LON),
                               min_coarse_rows=8)
    assert [lvl.n_lat_local for lvl in h] == [48, 24, 12]
    assert h[0].n_lat_local == N_LAT          # L0 is the input (always present)
    assert all(h[i + 1].n_lat_local == h[i].n_lat_local // 2
               for i in range(len(h) - 1))


def test_coarse_band_hierarchy_distributed_lockstep():
    """4-rank 64-row decomposition: every rank produces the SAME depth and stays
    even-aligned at every level (the lock-step requirement so the band-local
    restriction equals the global one).  Depth is set by the GLOBAL row count
    (coarsen until global < 2*min_coarse_rows): 64->32->16->8 (depth 4)."""
    n_ranks, n_lat = 4, 64
    hs = [_coarse_band_hierarchy(make_latlon_band_layout(r, n_ranks, n_lat, N_LON),
                                 min_coarse_rows=8) for r in range(n_ranks)]
    depths = {len(h) for h in hs}
    assert depths == {4}, f"ranks disagree on depth: {depths}"
    for h in hs:
        assert [lvl.n_lat_local for lvl in h] == [16, 8, 4, 2]
        # global lat halves with the band; depth driven by global, not local.
        assert [lvl.n_lat_global for lvl in h] == [64, 32, 16, 8]


def test_coarse_band_hierarchy_stops_on_odd():
    """Coarsening that would produce an ODD band terminates the hierarchy (the
    next level's 2x2 would straddle a cut) — here 6 local rows can coarsen to 3
    once, but 3 is odd so the chain stops, independent of min_coarse_rows."""
    h = _coarse_band_hierarchy(make_latlon_band_layout(0, 1, N_LAT, N_LON),
                               min_coarse_rows=1)
    # 48->24->12->6->3 then 3 is odd -> stop. Last level is odd-but-reachable.
    assert [lvl.n_lat_local for lvl in h] == [48, 24, 12, 6, 3]
    assert _coarse_band_layout(h[-1]) is None   # cannot coarsen past the tail


def test_coarse_band_hierarchy_single_level_when_too_small():
    """A GLOBAL grid already below 2*min_coarse_rows yields a depth-1 hierarchy
    (just the input) — the caller falls back to a single-level smoother."""
    # 12 global rows / 2 ranks = 6 local; 12 < 2*8 -> no admissible coarsening.
    h = _coarse_band_hierarchy(make_latlon_band_layout(0, 2, 12, N_LON),
                               min_coarse_rows=8)
    assert len(h) == 1 and h[0].n_lat_local == 6
    # And 24 global rows DO coarsen once (global 24 -> 12), depth 2.
    h2 = _coarse_band_hierarchy(make_latlon_band_layout(0, 2, 24, N_LON),
                                min_coarse_rows=8)
    assert [lvl.n_lat_local for lvl in h2] == [12, 6]
    assert [lvl.n_lat_global for lvl in h2] == [24, 12]


def test_banded_faces_reduce_to_serial_on_global_band():
    """``_faces_from_cell_depth_banded`` on a GLOBAL single band (rank 0 of 1,
    both edges = true poles) is BIT-IDENTICAL to the serial
    ``_faces_from_cell_depth`` — the equivalence that lets the distributed MG
    reuse the production min-rule.  (Cross-band coupling at an INTERIOR cut is
    validated by the mpirun -np 2 distributed test.)"""
    rng = np.random.default_rng(7)
    H_cell = jnp.asarray(800.0 + 400.0 * rng.random((N_LAT, N_LON)))
    m = np.ones((N_LAT, N_LON))
    m[0, :] = 0.0
    m[-1, :] = 0.0
    m[10:14, 30:40] = 0.0
    mask = jnp.asarray(m)
    glob = make_latlon_band_layout(0, 1, N_LAT, N_LON)  # south_rank==north_rank==None
    Hu_b, Hv_b, um_b, vm_b = _faces_from_cell_depth_banded(H_cell, mask, glob)
    Hu_s, Hv_s, um_s, vm_s = _faces_from_cell_depth(H_cell, mask, N_LAT, N_LON)
    for got, ref, nm in [(Hu_b, Hu_s, "H_u"), (Hv_b, Hv_s, "H_v"),
                          (um_b, um_s, "u_mask"), (vm_b, vm_s, "v_mask")]:
        assert got.shape == ref.shape, f"{nm} shape {got.shape} != {ref.shape}"
        assert np.allclose(np.asarray(got), np.asarray(ref), atol=0, rtol=0), (
            f"{nm} banded != serial on a global band")
    # The true-pole v-faces are walls in both.
    assert float(jnp.max(jnp.abs(Hv_b[0]))) == 0.0
    assert float(jnp.max(jnp.abs(Hv_b[-1]))) == 0.0


def test_banded_faces_open_at_interior_cut():
    """At an INTERIOR band edge (a neighbour exists) the v-face is NOT zeroed —
    the wall BC applies only at true poles.  Serial single-process: the halo
    pad wraps to this rank's own opposite edge, but the KEY property under test
    is that the edge face is LEFT OPEN (non-wall) when a neighbour rank is
    present, unlike the serial helper which always walls both ends."""
    rng = np.random.default_rng(9)
    nl = N_LAT // 2
    H_cell = jnp.asarray(900.0 + 300.0 * rng.random((nl, N_LON)))
    mask = jnp.ones((nl, N_LON))
    # rank 0 of 2: south edge IS the south pole (wall); north edge is an
    # interior cut to rank 1 (must stay open).
    layout = make_latlon_band_layout(0, 2, N_LAT, N_LON)
    assert layout.south_rank is None and layout.north_rank == 1
    _, Hv, _, vm = _faces_from_cell_depth_banded(H_cell, mask, layout)
    assert Hv.shape == (nl + 1, N_LON)
    assert float(jnp.max(jnp.abs(Hv[0]))) == 0.0          # south pole wall
    assert float(jnp.min(np.asarray(vm[-1]))) > 0.5       # north cut OPEN (wet)
    assert float(jnp.min(np.asarray(Hv[-1]))) > 0.0       # north cut carries depth


def test_banded_mg_reduces_to_serial_on_global_band():
    """The BANDED V-cycle on a GLOBAL single band (n_ranks=1) matches the SERIAL
    ``_make_multigrid_preconditioner`` M_inv — same hierarchy depth, faces, and
    transfers — so the banded engine is correct in the degenerate case.  (The
    cross-rank reduction-latency win is validated by the mpirun -np 2 distributed
    test.)"""
    grid, coeff, mask, H_cell, _, _ = _setup()
    layout = make_latlon_band_layout(0, 1, N_LAT, N_LON)
    mg_serial = _make_multigrid_preconditioner(H_cell, coeff, grid, mask)
    mg_banded = _make_multigrid_preconditioner_banded(
        H_cell, coeff, grid, mask, layout)
    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.standard_normal((N_LAT, N_LON))) * mask
    out_s = np.asarray(mg_serial(r))
    out_b = np.asarray(mg_banded(r))
    assert out_b.shape == out_s.shape
    assert np.allclose(out_b, out_s, rtol=1e-9, atol=1e-12), (
        f"banded != serial on a global band: max|Δ|="
        f"{np.max(np.abs(out_b - out_s)):.2e}")


def test_banded_mg_refuses_unequal_bands():
    """Lock-step guard: an UNEQUAL decomposition (n_lat_global % n_ranks != 0)
    would coarsen ranks to different depths, so the banded factory FAILS LOUD
    rather than deadlock on mismatched halo schedules."""
    grid, coeff, mask, H_cell, _, _ = _setup()
    # 48 rows over 5 ranks = 9/10/10/... unequal; rank-0 layout carries n_ranks.
    layout = make_latlon_band_layout(0, 5, N_LAT, N_LON)
    with pytest.raises(ValueError, match="EQUAL bands"):
        _make_multigrid_preconditioner_banded(
            H_cell, coeff, grid, mask, layout)


def test_banded_mg_refuses_wet_balanced_bands():
    """Lock-step guard vs EXPLICIT boundaries (codex): divisibility alone no
    longer proves equal bands — wet-cell-balanced boundaries on a divisible
    global (48 % 4 == 0) still give unequal/odd-aligned bands, which would
    coarsen ranks to different depths. The guard must check the ACTUAL band
    span, and it must fire even for a rank whose OWN band happens to be
    base-sized but mis-aligned."""
    grid, coeff, mask, H_cell, _, _ = _setup()
    b = (0, 10, 22, 34, N_LAT)               # 48 rows, 4 bands: 10/12/12/14
    # Rank 1 owns 12 rows == base, but lat_start 10 != 1*12 — mis-aligned.
    layout = make_latlon_band_layout(1, 4, N_LAT, N_LON, boundaries=b)
    with pytest.raises(ValueError, match="EQUAL bands"):
        _make_multigrid_preconditioner_banded(
            H_cell, coeff, grid, mask, layout)
