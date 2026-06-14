"""Seam-reduced adjoint of the rigid-lid CG solve (lat-lon C-grid).

The vertex layout (n_lat+1, n_lon+1) stores the periodic wrap REDUNDANTLY
(column n_lon duplicates column 0).  The symmetrized solve operator
``S = -A_vertex·L`` is exactly symmetric on the seam-REDUCED space but NOT
Euclidean-symmetric on the redundant layout, which biased every reverse-mode
gradient through ``jax.scipy.sparse.linalg.cg``'s symmetric-reuse VJP (a few %
up to 25%; discovered by the adjoint-matching harness,
.physics-validator/adjoint_oracle_match/RESULTS.md).  The production fix keeps
the forward CG byte-identical and overrides only the VJP with the exact
seam-reduced adjoint ``Fᵀ = Pᵀ·S_r⁻¹·Eᵀ``.

Source file exercised:
  - legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py
    (solve_streamfunction_interior custom VJP, _seam_expand,
     _symmetric_solve_operator)

Tests (each gate of the fix):
  1. Symmetry self-test + non-vacuity: the reduced operator passes an explicit
     <x,Ay> == <Ax,y> check; the redundant one FAILS it.
  2. Forward bit-identity: the custom-VJP solve equals the stock CG call
     bit-for-bit (eager and jit).
  3. Adjoint correctness: AD/FD == 1 through the solve; the stock-CG version
     is measurably biased (non-vacuity at gradient level).
  4. x0 cotangent stays exactly zero (stock-cg behavior preserved).
  5. AD/FD through TWO chained rigid_lid_step calls (the production
     composition: history feed-forward, leapfrog guess, island constraint).

The model-level dJ/dT0 / dJ/d(wind) AD-vs-FD gate runs on the spun-up faithful
ACC recipe (kink-free linearization point) in the fix's validation harness —
a from-rest coupled state is advection-limiter-kink-dominated and FD there is
meaningless at any eps.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.latlon_cgrid_operators import curl_vertex_cgrid
from legoesm.ocean.dynamics.rigid_lid_latlon_cgrid import (
    solve_streamfunction_interior, _symmetric_solve_operator, _seam_expand,
    rigid_lid_step,
)
from legoesm.ocean.dynamics.rigid_lid_islands import build_rigid_lid_data


N_LAT, N_LON = 18, 36


def _channel():
    """Periodic channel: south + north 2-row land walls, varying-depth interior.

    The bathymetry varies in latitude so the 1/H weights make the operator
    non-trivial (a flat-bottom channel hides part of the seam asymmetry)."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    lm = np.ones((N_LAT, N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    depth = 3000.0 + 1500.0 * np.sin(np.linspace(0, np.pi, N_LAT))[:, None]
    Hb = depth * np.ones((1, N_LON)) * lm
    mE = lm; mW = np.roll(lm, 1, axis=1)
    u_int = (mE * mW) > 0.5
    u_mask = np.concatenate([u_int, u_int[:, :1]], axis=1).astype(float)
    v_int = (lm[:-1] * lm[1:]) > 0.5
    v_mask = np.concatenate([np.zeros((1, N_LON)), v_int, np.zeros((1, N_LON))],
                            axis=0).astype(float)
    cfg = LatLonCGridOceanConfig()
    rl = build_rigid_lid_data(Hb, lm, u_mask, v_mask, cfg, grid, periodic_x=True)
    return grid, lm, u_mask, v_mask, cfg, rl


def _wrap_consistent_pair(rl, rng):
    """Random wrap-consistent vertex fields on the solve mask."""
    shape = np.asarray(rl.solve_mask).shape
    sm = np.asarray(rl.solve_mask)
    x = rng.standard_normal(shape); x[:, -1] = x[:, 0]; x *= sm
    w = rng.standard_normal(shape); w[:, -1] = w[:, 0]; w *= sm
    return jnp.asarray(x), jnp.asarray(w)


# ------------------------------------------------- 1. operator symmetry ----
def test_reduced_operator_symmetric_redundant_fails():
    """S_r = P·S·E is Euclidean-symmetric; S on the redundant layout is NOT.

    The redundant-side check is the NON-VACUITY guard: if the layout or the
    stencils ever change so the redundant operator becomes symmetric, the
    custom VJP here is no longer needed and this test must be revisited."""
    grid, lm, u_mask, v_mask, cfg, rl = _channel()
    rng = np.random.default_rng(1)

    def S(p):
        return _symmetric_solve_operator(p, rl, grid)

    def S_r(v):
        return S(_seam_expand(v))[:, :-1]

    max_asym_reduced = 0.0
    max_asym_redundant = 0.0
    for _ in range(5):
        x, w = _wrap_consistent_pair(rl, rng)
        # redundant Euclidean inner product
        a = float(jnp.sum(S(x) * w)); b = float(jnp.sum(x * S(w)))
        max_asym_redundant = max(max_asym_redundant,
                                 abs(a - b) / max(abs(a), abs(b)))
        # seam-reduced inner product
        xr, wr = x[:, :-1], w[:, :-1]
        ar = float(jnp.sum(S_r(xr) * wr)); br = float(jnp.sum(xr * S_r(wr)))
        max_asym_reduced = max(max_asym_reduced,
                               abs(ar - br) / max(abs(ar), abs(br)))
    assert max_asym_reduced < 1e-12, max_asym_reduced
    # non-vacuity: the redundant layout REALLY is asymmetric (this is the bug)
    assert max_asym_redundant > 1e-3, max_asym_redundant


# --------------------------------------------- 2. forward bit-identity ----
def _stock_cg_solve(rhs, rl, grid, x0, *, tol, maxiter):
    """The pre-fix forward solve, verbatim (stock cg, redundant layout)."""
    sm = rl.solve_mask
    rhs_sym = jnp.where(sm > 0.5, -rl.A_vertex * rhs, 0.0)
    x0_in = x0 * sm

    def op(psi):
        return _symmetric_solve_operator(psi, rl, grid)

    def precond(r):
        return r * rl.inv_diag

    dpsi, _info = jax.scipy.sparse.linalg.cg(
        op, rhs_sym, x0=x0_in, tol=tol, atol=0.0, maxiter=maxiter, M=precond)
    return dpsi * sm


def _production_rhs_and_guess(grid, rl, rng):
    """A wrap-consistent (rhs, x0) pair through the production RHS path."""
    Fu = jnp.asarray(rng.standard_normal((grid.n_lat, grid.n_lon + 1)))
    Fu = Fu.at[:, -1].set(Fu[:, 0]) * rl.u_mask
    Fv = jnp.asarray(rng.standard_normal((grid.n_lat + 1, grid.n_lon))) * rl.v_mask
    rhs = curl_vertex_cgrid(1e-6 * Fu, 1e-6 * Fv, grid)
    x0 = jnp.asarray(rng.standard_normal((grid.n_lat + 1, grid.n_lon + 1)))
    x0 = x0.at[:, -1].set(x0[:, 0]) * rl.solve_mask
    return rhs, x0


def test_forward_bit_identical_to_stock_cg():
    grid, lm, u_mask, v_mask, cfg, rl = _channel()
    rng = np.random.default_rng(2)
    rhs, x0 = _production_rhs_and_guess(grid, rl, rng)
    kw = dict(tol=cfg.rigid_lid_cg_tol, maxiter=cfg.rigid_lid_cg_maxiter)
    a = _stock_cg_solve(rhs, rl, grid, x0, **kw)
    b = solve_streamfunction_interior(rhs, rl, grid, x0, **kw)
    assert bool(jnp.all(a == b)), "forward changed (eager)"
    aj = jax.jit(lambda r, x: _stock_cg_solve(r, rl, grid, x, **kw))(rhs, x0)
    bj = jax.jit(
        lambda r, x: solve_streamfunction_interior(r, rl, grid, x, **kw)
    )(rhs, x0)
    assert bool(jnp.all(aj == bj)), "forward changed (jit)"


# ------------------------------------------------ 3. adjoint AD-vs-FD ----
def test_adjoint_ad_fd_exact_and_stock_biased():
    """AD/FD = 1 through the fixed solve; the stock VJP is measurably biased."""
    grid, lm, u_mask, v_mask, cfg, rl = _channel()
    rng = np.random.default_rng(3)
    rhs0, x0 = _production_rhs_and_guess(grid, rl, rng)
    kw = dict(tol=cfg.rigid_lid_cg_tol, maxiter=cfg.rigid_lid_cg_maxiter)
    scale = 1.0 / float(jnp.max(jnp.abs(
        solve_streamfunction_interior(rhs0, rl, grid, x0, **kw))))

    def make_J(solver):
        def J(s):
            d = solver(rhs0 * s, rl, grid, x0, **kw)
            # nonquadratic so the bias cannot cancel by symmetry
            return jnp.sum(jnp.tanh(d * scale) * rl.A_vertex)
        return J

    s0 = 1.0
    eps = 1e-5
    for solver, expect_exact in ((solve_streamfunction_interior, True),
                                 (_stock_cg_solve, False)):
        J = make_J(solver)
        g = float(jax.grad(J)(s0))
        fd = (float(J(s0 + eps)) - float(J(s0 - eps))) / (2 * eps)
        ratio = g / fd
        if expect_exact:
            assert abs(ratio - 1.0) < 1e-5, ratio
        else:
            # non-vacuity: the pre-fix adjoint REALLY is biased on this
            # operator (if this starts passing, the redundant layout went
            # away and the custom VJP should be revisited).
            assert abs(ratio - 1.0) > 1e-4, ratio


def test_x0_cotangent_exactly_zero():
    """The Krylov guess gets a zero gradient — identical to stock cg (IFT)."""
    grid, lm, u_mask, v_mask, cfg, rl = _channel()
    rng = np.random.default_rng(4)
    rhs0, x0 = _production_rhs_and_guess(grid, rl, rng)
    kw = dict(tol=cfg.rigid_lid_cg_tol, maxiter=cfg.rigid_lid_cg_maxiter)

    def J(x0v):
        d = solve_streamfunction_interior(rhs0, rl, grid, x0v, **kw)
        return jnp.sum(d ** 3)

    g = jax.grad(J)(x0)
    assert float(jnp.max(jnp.abs(g))) == 0.0


# --------------------------------- 5. two chained rigid-lid steps, AD/FD ----
def test_two_step_rigid_lid_chain_ad_fd():
    """AD == FD through TWO chained rigid_lid_step calls (ψ/dψ/dpsin history
    fed forward, leapfrog CG guess active) — the exact production composition
    of the fixed solve, free of advection-limiter kinks.  The KE-of-velocity
    cotangent enters the solve through both the interior dψ and the island
    (transport) constraint, the configuration the seam bias corrupted."""
    grid, lm, u_mask, v_mask, cfg, rl = _channel()
    rng = np.random.default_rng(6)
    Fu = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1)))
    Fu = (1e-6 * Fu.at[:, -1].set(Fu[:, 0])) * rl.u_mask
    Fv = (1e-6 * jnp.asarray(rng.standard_normal((N_LAT + 1, N_LON)))) * rl.v_mask
    zV = jnp.zeros((N_LAT + 1, N_LON + 1))
    zI = jnp.zeros((rl.nisle,))
    dt = 4800.0

    def J(amp):
        psi, dpsi, dpsi_prev, dpsin, dpsin_prev = zV, zV, zV, zI, zI
        for _ in range(2):
            psi, dpsi, dpsi_prev, dpsin, dpsin_prev, u_bt, v_bt = rigid_lid_step(
                psi, dpsi, dpsi_prev, dpsin, dpsin_prev,
                amp * Fu, Fv, rl, dt, cfg, grid)
        # each physical u-face once (drop the wrap duplicate)
        return 0.5 * (jnp.sum(u_bt[:, :-1] ** 2) + jnp.sum(v_bt ** 2))

    a0 = 1.0
    g = float(jax.grad(J)(a0))
    for eps in (1e-4, 1e-5):
        fd = (float(J(a0 + eps)) - float(J(a0 - eps))) / (2 * eps)
        assert abs(g / fd - 1.0) < 1e-5, (eps, g / fd)
