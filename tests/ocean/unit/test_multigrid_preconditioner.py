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
    _make_multigrid_preconditioner, _mg_restrict, _mg_prolong,
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
