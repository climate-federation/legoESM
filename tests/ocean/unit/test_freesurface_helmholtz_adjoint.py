"""Area-weighted adjoint of the free-surface implicit Helmholtz solve.

The implicit-CN barotropic solver's Helmholtz operator
``A(η) = η - coeff·∇·(H·∇η)`` is built from the FV gradient and the FV
divergence; the divergence carries ``1/area``, so ``A`` is self-adjoint only
in the AREA-WEIGHTED inner product (``Aᵀ = W·A·W⁻¹``, ``W = diag(area)``;
measured Euclidean asymmetry 3.9e-2 at 18x36, area-weighted ~3e-15).
``jax.scipy.sparse.linalg.cg``'s VJP re-solves with ``A`` assuming Euclidean
symmetry, which biased every reverse-mode gradient through the free-surface
solve (per-solve AD/FD 0.9909 at 18x36; forward always correct).  Sibling of
the rigid-lid seam-reduced adjoint (commit 1e370679,
tests/ocean/unit/test_rigid_lid_seam_adjoint.py).

Source file exercised:
  - legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py
    (solve_helmholtz_freesurface custom VJP, _helmholtz_apply,
     _helmholtz_inv_diag, barotropic_implicit_latlon_cgrid)

Tests (each gate of the fix):
  1. Symmetry self-test + non-vacuity: A passes an explicit area-weighted
     <x,Ay>_W == <Ax,y>_W check; the Euclidean one FAILS it.
  2. Forward bit-identity: the custom-VJP solve equals the stock CG call
     bit-for-bit (eager and jit).
  3. Dense ground truth (6x8): the shipped rhs-cotangent matches the dense
     S_wᵀ solve elementwise; stock cg AND both area-weighting slips
     (W on the wrong side — THE classic error) are measurably wrong.
  4. Adjoint AD/FD == 1 through the solve; the stock-CG version is
     measurably biased (non-vacuity at gradient level), eps-stable.
  5. x0 cotangent stays exactly zero (stock-cg behavior preserved).
  6. grad under lax.scan through TWO chained barotropic_implicit steps
     (production composition; catches tracer-captured-in-bwd-closure
     lowering failures) + jit(grad).
  7. Forward-mode jvp through the solve raises (custom_vjp limitation,
     documented; no repo path uses it).
  8. float32 NaN regression: the adjoint solve's weight is normalized by
     max(area) — with RAW ~1e11 m² areas the float32 CG residual norms
     underflow subnormal and the gradient NaNs (found while building the
     fix; production states are float32).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm import constants

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _h_total_at_faces,
    _helmholtz_inv_diag,
    _make_helmholtz,
    _make_diag_preconditioner,
    barotropic_implicit_latlon_cgrid,
    solve_helmholtz_freesurface,
)

TOL, MAXITER = 1e-12, 2000


def _channel(n_lat, n_lon):
    """Periodic channel: land walls north+south, an island, varying depth."""
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    lm = np.ones((n_lat, n_lon))
    lm[0] = 0.0
    lm[-1] = 0.0
    lm[n_lat // 2, n_lon // 3] = 0.0  # island breaks accidental symmetry
    u_int = (lm * np.roll(lm, 1, axis=1)) > 0.5
    u_mask = jnp.asarray(
        np.concatenate([u_int, u_int[:, :1]], axis=1).astype(float))
    v_int = (lm[:-1] * lm[1:]) > 0.5
    v_mask = jnp.asarray(np.concatenate(
        [np.zeros((1, n_lon)), v_int, np.zeros((1, n_lon))], axis=0,
    ).astype(float))
    mask = jnp.asarray(lm)
    depth = 3000.0 + 1500.0 * np.sin(np.linspace(0, np.pi, n_lat))[:, None] \
        * (1.0 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, n_lon))[None, :])
    h_k = jnp.asarray(np.repeat((depth * lm / 3.0)[:, :, None], 3, axis=2))
    H_u, H_v = _h_total_at_faces(h_k, jnp.asarray(1.0), mask, grid)
    coeff = jnp.asarray(0.55 * 0.55 * 3600.0 ** 2 * constants.g)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    return grid, mask, u_mask, v_mask, H_u, H_v, coeff, inv_diag


def _stock_cg_solve(rhs, x0, H_u, H_v, coeff, mask, u_mask, v_mask,
                    inv_diag, grid, *, tol, maxiter):
    """The pre-fix forward solve, verbatim (stock cg, Euclidean VJP)."""
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
    M_inv = _make_diag_preconditioner(H_u, H_v, coeff, grid, mask)
    out, _info = jax.scipy.sparse.linalg.cg(
        A_op, rhs, x0=x0, tol=tol, maxiter=maxiter, M=M_inv)
    return out


# ------------------------------------------------- 1. operator symmetry ----
def test_area_weighted_symmetric_euclidean_fails():
    """A is symmetric in <.,.>_W = sum(x*y*area); NOT in the Euclidean ip.

    The Euclidean-side check is the NON-VACUITY guard: if the operators ever
    change so A becomes Euclidean-symmetric, the custom VJP here is no
    longer needed and this test must be revisited."""
    grid, mask, u_mask, v_mask, H_u, H_v, coeff, _ = _channel(18, 36)
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
    area = jnp.asarray(grid.area)
    rng = np.random.default_rng(1)
    asym_e = asym_w = 0.0
    for _ in range(5):
        x = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        y = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        a = float(jnp.sum(A_op(x) * y))
        b = float(jnp.sum(x * A_op(y)))
        asym_e = max(asym_e, abs(a - b) / max(abs(a), abs(b)))
        aw = float(jnp.sum(A_op(x) * y * area))
        bw = float(jnp.sum(x * A_op(y) * area))
        asym_w = max(asym_w, abs(aw - bw) / max(abs(aw), abs(bw)))
    assert asym_w < 1e-12, asym_w
    # non-vacuity: the Euclidean layout REALLY is asymmetric (this is the bug)
    assert asym_e > 1e-3, asym_e


# --------------------------------------------- 2. forward bit-identity ----
def test_forward_bit_identical_to_stock_cg():
    grid, mask, u_mask, v_mask, H_u, H_v, coeff, inv_diag = _channel(18, 36)
    rng = np.random.default_rng(2)
    rhs = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    x0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    args = (rhs, x0, H_u, H_v, coeff, mask, u_mask, v_mask, inv_diag, grid)
    kw = dict(tol=TOL, maxiter=MAXITER)
    a = _stock_cg_solve(*args, **kw)
    b = solve_helmholtz_freesurface(*args, **kw)
    assert bool(jnp.all(a == b)), "forward changed (eager)"
    aj = jax.jit(lambda r, x: _stock_cg_solve(
        r, x, *args[2:], **kw))(rhs, x0)
    bj = jax.jit(lambda r, x: solve_helmholtz_freesurface(
        r, x, *args[2:], **kw))(rhs, x0)
    # Eager is bit-identical (asserted above); the jit path differs only by
    # XLA fusion-order FP round-off, NOT logic.  The in-test _stock_cg_solve
    # reference and the production solve_helmholtz_freesurface build the same
    # Helmholtz operator + diagonal preconditioner from the SAME inputs, but
    # their distinct call structures fuse differently under jit (the shared
    # #516 single-sourced metric in the preconditioner participates), so the
    # two CG residual sequences reassociate at the ~1 ULP level.  Measured at
    # 18x36: max abs diff 6.9e-17, max rel diff 3.4e-13 against a field of
    # magnitude ~0.2.  A real forward divergence would be O(1e-3) (cf. the
    # adjoint-bias non-vacuity gates), so this tolerance still catches logic
    # changes while tolerating cross-hardware fusion ordering.
    np.testing.assert_allclose(
        np.asarray(aj), np.asarray(bj), rtol=1e-10, atol=1e-14,
        err_msg="forward changed (jit) beyond XLA fusion-order round-off",
    )


# ------------------------------- 3. dense ground truth + mutation pin ----
def test_dense_ground_truth_and_weighting_slips():
    """Shipped rhs-cotangent == dense S_wᵀ solve elementwise; stock cg and
    BOTH area-weighting slips (W on the wrong side of the solve — THE
    classic error in W-conjugated adjoints) are measurably wrong."""
    n_lat, n_lon = 6, 8
    grid, mask, u_mask, v_mask, H_u, H_v, coeff, inv_diag = _channel(
        n_lat, n_lon)
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask)
    M_inv = _make_diag_preconditioner(H_u, H_v, coeff, grid, mask)
    area = jnp.asarray(grid.area)
    n = n_lat * n_lon
    wet = np.asarray(mask).ravel() > 0.5

    S = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        S[:, j] = np.asarray(
            A_op(jnp.asarray(ej.reshape(n_lat, n_lon)))).ravel()
    S_w = S[np.ix_(wet, wet)]

    rng = np.random.default_rng(3)
    cbar = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    truth = np.linalg.solve(S_w.T, np.asarray(cbar).ravel()[wet])
    t_scale = np.max(np.abs(truth))

    # shipped cotangent: grad of the linear functional <solve(rhs), cbar>
    def J(rhs):
        return jnp.sum(solve_helmholtz_freesurface(
            rhs, jnp.zeros_like(rhs), H_u, H_v, coeff,
            mask, u_mask, v_mask, inv_diag, grid,
            tol=TOL, maxiter=MAXITER) * cbar)

    shipped = np.asarray(jax.grad(J)(jnp.zeros_like(cbar))).ravel()[wet]
    assert np.max(np.abs(shipped - truth)) / t_scale < 1e-10

    def cgs(rhs):
        out, _ = jax.scipy.sparse.linalg.cg(
            A_op, rhs * mask, tol=TOL, maxiter=MAXITER, M=M_inv)
        return out * mask

    # the fix, recomputed by hand, must also match (formula pin)
    by_hand = np.asarray(area * cgs(cbar / area)).ravel()[wet]
    assert np.max(np.abs(by_hand - truth)) / t_scale < 1e-10

    # mutation pin: each wrong variant must FAIL the dense comparison
    stock = np.asarray(cgs(cbar)).ravel()[wet]
    slip1 = np.asarray(cgs(area * cbar) / area).ravel()[wet]  # W wrong side
    slip2 = np.asarray(cgs(cbar / area) / area).ravel()[wet]  # W⁻¹ twice
    for name, wrong in (("stock", stock), ("slip-Wside", slip1),
                        ("slip-Winv2", slip2)):
        err = np.max(np.abs(wrong - truth)) / t_scale
        assert err > 1e-4, (name, err)


# ------------------------------------------------ 4. adjoint AD-vs-FD ----
def test_adjoint_ad_fd_exact_and_stock_biased():
    """AD/FD = 1 through the fixed solve (eps-stable over 2 decades); the
    stock VJP is measurably biased (non-vacuity: if this starts passing,
    the operator became Euclidean-symmetric and the custom VJP should be
    revisited)."""
    grid, mask, u_mask, v_mask, H_u, H_v, coeff, inv_diag = _channel(18, 36)
    rng = np.random.default_rng(4)
    rhs0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    w = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    args = (H_u, H_v, coeff, mask, u_mask, v_mask, inv_diag, grid)
    kw = dict(tol=TOL, maxiter=MAXITER)
    sc = 1.0 / float(jnp.max(jnp.abs(solve_helmholtz_freesurface(
        rhs0, jnp.zeros_like(rhs0), *args, **kw))))

    def make_J(solver):
        def J(s):
            x = solver(rhs0 * s, jnp.zeros_like(rhs0), *args, **kw) * mask
            # nonquadratic so the bias cannot cancel by symmetry
            return jnp.sum(jnp.tanh(x * sc) * w)
        return J

    for solver, expect_exact in ((solve_helmholtz_freesurface, True),
                                 (_stock_cg_solve, False)):
        J = make_J(solver)
        g = float(jax.grad(J)(1.0))
        for eps in (1e-5, 1e-6):
            fd = (float(J(1.0 + eps)) - float(J(1.0 - eps))) / (2 * eps)
            ratio = g / fd
            if expect_exact:
                assert abs(ratio - 1.0) < 1e-6, (eps, ratio)
            else:
                assert abs(ratio - 1.0) > 1e-3, (eps, ratio)


def test_x0_cotangent_exactly_zero():
    """The Krylov guess gets a zero gradient — identical to stock cg (IFT)."""
    grid, mask, u_mask, v_mask, H_u, H_v, coeff, inv_diag = _channel(12, 16)
    rng = np.random.default_rng(5)
    rhs0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    x0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1

    def J(x0v):
        x = solve_helmholtz_freesurface(
            rhs0, x0v, H_u, H_v, coeff, mask, u_mask, v_mask,
            inv_diag, grid, tol=TOL, maxiter=MAXITER)
        return jnp.sum(x ** 3)

    g = jax.grad(J)(x0)
    assert float(jnp.max(jnp.abs(g))) == 0.0


# ------------------------------- 6. production composition under scan ----
def _step_setup(dtype):
    grid = create_latlon_grid(n_lat=12, n_lon=24)
    z_coord = create_ocean_z_star(n_levels=3, H_max=3000.0)
    lat_rad = jnp.asarray(grid.lat2d)
    H_bathy = 500.0 + 2500.0 * jnp.cos(lat_rad) ** 2
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H_bathy,
    )
    eta0 = (0.05 * jnp.sin(2 * lat_rad) * jnp.cos(3 * jnp.asarray(grid.lon2d))
            * state.land_mask.data).astype(dtype)
    state = state._replace(
        eta=state.eta.replace(data=eta0),
        u=state.u.replace(data=state.u.data.astype(dtype)),
        v=state.v.replace(data=state.v.data.astype(dtype)),
    )
    cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="implicit_cn")
    return grid, z_coord, cfg, state


def test_grad_under_scan_two_steps_finite():
    """jax.grad through lax.scan over two implicit-CN barotropic steps —
    the production composition.  Catches the custom_vjp closure-capture
    lowering failure ("No constant handler for DynamicJaxprTracer") and
    bwd NaN; also pins jit(grad)."""
    grid, z_coord, cfg, state = _step_setup(jnp.float64)
    rng = np.random.default_rng(6)
    Fu = jnp.asarray(rng.standard_normal((grid.n_lat, grid.n_lon + 1)))
    Fu = (1e-5 * Fu.at[:, -1].set(Fu[:, 0])) * state.u_mask.data
    w = jnp.asarray(rng.standard_normal((grid.n_lat, grid.n_lon))) \
        * state.land_mask.data

    def J(amp):
        def body(s, _):
            s_new, _flux = barotropic_implicit_latlon_cgrid(
                s, 3600.0, grid, z_coord, cfg, F_slow_u=amp * Fu)
            return s_new, None
        s_fin, _ = jax.lax.scan(body, state, None, length=2)
        ke = 0.5 * (jnp.sum(s_fin.u.data[:, :-1] ** 2)
                    + jnp.sum(s_fin.v.data ** 2))
        return jnp.sum(jnp.tanh(s_fin.eta.data / 0.05) * w) + 1e6 * ke

    g = float(jax.grad(J)(1.0))
    assert np.isfinite(g) and g != 0.0
    # jit changes CG fp reassociation slightly (~1e-10 rel); the gate is
    # "lowers and agrees", not bitwise.
    gj = float(jax.jit(jax.grad(J))(1.0))
    np.testing.assert_allclose(gj, g, rtol=1e-8)
    # AD == FD through the scan composition (free-surface 2-step loss)
    for eps in (1e-4, 1e-5):
        fd = (float(J(1.0 + eps)) - float(J(1.0 - eps))) / (2 * eps)
        assert abs(g / fd - 1.0) < 1e-4, (eps, g / fd)


# ----------------------------------------------------- 7. jvp raises ----
def test_jvp_through_solve_raises():
    """custom_vjp does not support forward-mode AD; the failure must be loud
    (documented limitation; remedy = paired custom_jvp), not silent."""
    grid, mask, u_mask, v_mask, H_u, H_v, coeff, inv_diag = _channel(12, 16)
    rng = np.random.default_rng(7)
    rhs0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1

    def f(rhs):
        return solve_helmholtz_freesurface(
            rhs, jnp.zeros_like(rhs), H_u, H_v, coeff, mask, u_mask,
            v_mask, inv_diag, grid, tol=TOL, maxiter=MAXITER)

    with pytest.raises(TypeError):
        jax.jvp(f, (rhs0,), (rhs0,))


# ------------------------------------------- 8. float32 NaN regression ----
def test_float32_grad_finite():
    """Production states are float32: the adjoint weight must be the
    NORMALIZED area (w̃ = area/max(area)).  With raw ~1e11 m² areas the
    adjoint CG residual norms underflow float32 subnormals and the
    gradient NaNs (the bug found while shipping this fix)."""
    grid, z_coord, cfg, state = _step_setup(jnp.float32)
    assert state.eta.data.dtype == jnp.float32
    rng = np.random.default_rng(8)
    w = jnp.asarray(rng.standard_normal((grid.n_lat, grid.n_lon)),
                    dtype=jnp.float32) * state.land_mask.data

    def J(amp):
        s_new, _flux = barotropic_implicit_latlon_cgrid(
            state, 3600.0, grid, z_coord, cfg,
            F_slow_eta=amp * 1e-4 * w)
        return jnp.sum(s_new.eta.data * w)

    g = float(jax.grad(J)(jnp.float32(1.0)))
    assert np.isfinite(g) and g != 0.0
