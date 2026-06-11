"""Area-weighted adjoint of the MPAS free-surface implicit Helmholtz solve.

Exact mirror of tests/ocean/unit/test_freesurface_helmholtz_adjoint.py on the
Voronoi mesh.  The TRiSK Helmholtz operator ``A(η) = η - coeff·div(H·grad η)``
is self-adjoint only in the areaCell-weighted inner product (measured
Euclidean asymmetry 1.4e-2 on the level-2 icosahedral mesh, area-weighted
~2e-15); stock ``jax.scipy`` cg's symmetry-reusing VJP therefore biased
reverse-mode gradients through the free-surface solve.  Sibling of the
rigid-lid seam adjoint (commit 1e370679).

Source file exercised:
  - legoesm/ocean/dynamics/barotropic_implicit_mpas.py
    (solve_helmholtz_freesurface_mpas custom VJP, _helmholtz_apply_mpas,
     _helmholtz_inv_diag_mpas, barotropic_implicit_mpas)

Gates: operator symmetry self-test (+ Euclidean non-vacuity), forward
bit-identity vs stock cg, dense ground-truth VJP elementwise + weighting-slip
mutation pin, AD/FD exact vs stock biased, x0 zero cotangent, jvp raises,
float32 grad finite (areaCell ~1e12 m² underflow regression), grad through
the production barotropic_implicit_mpas step.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm import constants

jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
    _helmholtz_inv_diag_mpas,
    _make_helmholtz,
    _make_diag_preconditioner,
    barotropic_implicit_mpas,
    solve_helmholtz_freesurface_mpas,
)

TOL, MAXITER = 1e-12, 2000


@pytest.fixture(scope="module")
def setup():
    mesh = create_voronoi_mesh(subdivision_level=2)
    rng = np.random.default_rng(0)
    lat = np.asarray(mesh.latCell)
    mask = jnp.asarray((np.abs(lat) < 1.3).astype(float))  # polar land caps
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    edge_mask = jnp.asarray(
        ((np.asarray(mask)[c1] > 0.5) & (np.asarray(mask)[c2] > 0.5))
        .astype(float))
    # varying H so the operator weights are nontrivial
    H_e = jnp.asarray(
        3000.0 + 300.0 * rng.standard_normal(edge_mask.shape)) * edge_mask
    coeff = jnp.asarray(0.55 * 0.55 * 3600.0 ** 2 * constants.g)
    inv_diag = _helmholtz_inv_diag_mpas(H_e, coeff, mesh, mask, edge_mask)
    return mesh, mask, edge_mask, H_e, coeff, inv_diag


def _stock_cg_solve(rhs, x0, H_e, coeff, mask, edge_mask, inv_diag, mesh,
                    *, tol, maxiter):
    """The pre-fix forward solve, verbatim (stock cg, Euclidean VJP)."""
    A_op = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    M_inv = _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask)
    out, _info = jax.scipy.sparse.linalg.cg(
        A_op, rhs, x0=x0, tol=tol, maxiter=maxiter, M=M_inv)
    return out


def test_area_weighted_symmetric_euclidean_fails(setup):
    mesh, mask, edge_mask, H_e, coeff, _ = setup
    A_op = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    area = jnp.asarray(mesh.areaCell)
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
    # non-vacuity: Euclidean really is asymmetric (this is the bug)
    assert asym_e > 1e-4, asym_e


def test_forward_bit_identical_to_stock_cg(setup):
    mesh, mask, edge_mask, H_e, coeff, inv_diag = setup
    rng = np.random.default_rng(2)
    rhs = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    x0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    args = (H_e, coeff, mask, edge_mask, inv_diag, mesh)
    kw = dict(tol=TOL, maxiter=MAXITER)
    a = _stock_cg_solve(rhs, x0, *args, **kw)
    b = solve_helmholtz_freesurface_mpas(rhs, x0, *args, **kw)
    assert bool(jnp.all(a == b)), "forward changed (eager)"
    # Under jit, XLA fuses the TRiSK gather/scatter ops differently across
    # the two (distinct) programs — stock's OWN eager-vs-jit drift is
    # ~6e-17, so program-identity bitwise equality is not a meaningful
    # gate here.  The decisive forward gate is the SAME-program A/B on the
    # jitted production step: 18/18 sha256 match over 6 steps, fix vs
    # pre-fix file (review record:
    # .physics-validator/helmholtz_adjoint_review/).  Here: last-bit tol.
    aj = jax.jit(lambda r, x: _stock_cg_solve(r, x, *args, **kw))(rhs, x0)
    bj = jax.jit(
        lambda r, x: solve_helmholtz_freesurface_mpas(r, x, *args, **kw)
    )(rhs, x0)
    assert float(jnp.max(jnp.abs(aj - bj))) < 1e-15, "forward changed (jit)"


def test_dense_ground_truth_and_weighting_slips(setup):
    """Shipped rhs-cotangent == dense S_wᵀ solve elementwise; stock cg and
    the area-weighting slips are measurably wrong."""
    mesh, mask, edge_mask, H_e, coeff, inv_diag = setup
    A_op = _make_helmholtz(H_e, coeff, mesh, mask, edge_mask)
    M_inv = _make_diag_preconditioner(H_e, coeff, mesh, mask, edge_mask)
    area = jnp.asarray(mesh.areaCell)
    n = mask.shape[0]
    wet = np.asarray(mask) > 0.5

    S = np.asarray(jax.vmap(
        lambda e: A_op(e), in_axes=0, out_axes=1)(jnp.eye(n)))
    S_w = S[np.ix_(wet, wet)]

    rng = np.random.default_rng(3)
    cbar = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    truth = np.linalg.solve(S_w.T, np.asarray(cbar)[wet])
    t_scale = np.max(np.abs(truth))

    def J(rhs):
        return jnp.sum(solve_helmholtz_freesurface_mpas(
            rhs, jnp.zeros_like(rhs), H_e, coeff, mask, edge_mask,
            inv_diag, mesh, tol=TOL, maxiter=MAXITER) * cbar)

    shipped = np.asarray(jax.grad(J)(jnp.zeros_like(cbar)))[wet]
    assert np.max(np.abs(shipped - truth)) / t_scale < 1e-10

    def cgs(rhs):
        out, _ = jax.scipy.sparse.linalg.cg(
            A_op, rhs * mask, tol=TOL, maxiter=MAXITER, M=M_inv)
        return out * mask

    by_hand = np.asarray(area * cgs(cbar / area))[wet]
    assert np.max(np.abs(by_hand - truth)) / t_scale < 1e-10

    stock = np.asarray(cgs(cbar))[wet]
    slip1 = np.asarray(cgs(area * cbar) / area)[wet]   # W on the wrong side
    slip2 = np.asarray(cgs(cbar / area) / area)[wet]   # W⁻¹ twice
    for name, wrong in (("stock", stock), ("slip-Wside", slip1),
                        ("slip-Winv2", slip2)):
        err = np.max(np.abs(wrong - truth)) / t_scale
        assert err > 1e-5, (name, err)


def test_adjoint_ad_fd_exact_and_stock_biased(setup):
    mesh, mask, edge_mask, H_e, coeff, inv_diag = setup
    rng = np.random.default_rng(4)
    rhs0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    w = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    args = (H_e, coeff, mask, edge_mask, inv_diag, mesh)
    kw = dict(tol=TOL, maxiter=MAXITER)
    sc = 1.0 / float(jnp.max(jnp.abs(solve_helmholtz_freesurface_mpas(
        rhs0, jnp.zeros_like(rhs0), *args, **kw))))

    def make_J(solver):
        def J(s):
            x = solver(rhs0 * s, jnp.zeros_like(rhs0), *args, **kw) * mask
            return jnp.sum(jnp.tanh(x * sc) * w)
        return J

    for solver, expect_exact in ((solve_helmholtz_freesurface_mpas, True),
                                 (_stock_cg_solve, False)):
        J = make_J(solver)
        g = float(jax.grad(J)(1.0))
        for eps in (1e-5, 1e-6):
            fd = (float(J(1.0 + eps)) - float(J(1.0 - eps))) / (2 * eps)
            ratio = g / fd
            if expect_exact:
                assert abs(ratio - 1.0) < 1e-6, (eps, ratio)
            else:
                # non-vacuity: stock really is biased on this operator
                assert abs(ratio - 1.0) > 1e-5, (eps, ratio)


def test_x0_cotangent_exactly_zero(setup):
    mesh, mask, edge_mask, H_e, coeff, inv_diag = setup
    rng = np.random.default_rng(5)
    rhs0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1
    x0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1

    def J(x0v):
        x = solve_helmholtz_freesurface_mpas(
            rhs0, x0v, H_e, coeff, mask, edge_mask, inv_diag, mesh,
            tol=TOL, maxiter=MAXITER)
        return jnp.sum(x ** 3)

    g = jax.grad(J)(x0)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_jvp_through_solve_raises(setup):
    mesh, mask, edge_mask, H_e, coeff, inv_diag = setup
    rng = np.random.default_rng(6)
    rhs0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask * 0.1

    def f(rhs):
        return solve_helmholtz_freesurface_mpas(
            rhs, jnp.zeros_like(rhs), H_e, coeff, mask, edge_mask,
            inv_diag, mesh, tol=TOL, maxiter=MAXITER)

    with pytest.raises(TypeError):
        jax.jvp(f, (rhs0,), (rhs0,))


def test_production_step_grad_finite_f32_and_scan():
    """grad through barotropic_implicit_mpas (production composition):
    float32 (areaCell ~1e12 m² — the subnormal-underflow regression) and
    f64 under lax.scan (closure-capture lowering regression)."""
    mesh = create_voronoi_mesh(subdivision_level=2)
    z_coord = create_ocean_z_star(n_levels=3, H_max=3000.0)
    state = rest_state_mpas_ocean(
        mesh, z_coord, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
        H_max=3000.0, land_lat_threshold=75.0,
    )
    cfg = MPASOceanConfig(barotropic_solver="implicit_cn")
    rng = np.random.default_rng(7)
    w_cell = jnp.asarray(rng.standard_normal(state.eta.data.shape),
                         dtype=state.eta.data.dtype) * state.land_mask.data

    # float32 regression (production state dtype)
    def J32(amp):
        eta_new, u_bar_new, Hu_avg = barotropic_implicit_mpas(
            state, mesh, z_coord, cfg, 3600.0,
            F_slow_eta=amp * 1e-4 * w_cell)
        return jnp.sum(eta_new * w_cell)

    g32 = float(jax.grad(J32)(jnp.asarray(1.0, dtype=state.eta.data.dtype)))
    assert np.isfinite(g32) and g32 != 0.0

    # f64 grad-under-scan (chain eta through two steps)
    state64 = state._replace(
        eta=state.eta.replace(data=state.eta.data.astype(jnp.float64)))

    def J(amp):
        def body(eta, _):
            s = state64._replace(eta=state64.eta.replace(data=eta))
            eta_new, _u, _hu = barotropic_implicit_mpas(
                s, mesh, z_coord, cfg, 3600.0,
                F_slow_eta=amp * 1e-4 * w_cell)
            return eta_new, None
        eta_fin, _ = jax.lax.scan(
            body, state64.eta.data, None, length=2)
        return jnp.sum(jnp.tanh(eta_fin / 0.05) * w_cell)

    g = float(jax.grad(J)(1.0))
    assert np.isfinite(g) and g != 0.0
    # FD through the full step carries min()/floor kink noise that grows
    # as eps shrinks (the solver-level AD/FD gate is the sharp one).
    for eps, tol_r in ((1e-3, 2e-4), (1e-4, 2e-3)):
        fd = (float(J(1.0 + eps)) - float(J(1.0 - eps))) / (2 * eps)
        assert abs(g / fd - 1.0) < tol_r, (eps, g / fd)
