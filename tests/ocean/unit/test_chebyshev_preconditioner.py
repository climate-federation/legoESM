"""Chebyshev-polynomial preconditioner for the implicit-CN Helmholtz PCG
(scaling review 2026-06-13, ocean lever #1: cut the OUTER iteration count
with matvecs + scalar coefficients, NO per-iteration global reduction —
the right direction for the latency-bound cross-node fabric).

Pins:
1. M⁻¹ is W-self-adjoint (the single_reduce / Chronopoulos–Gear
   requirement) — a polynomial in the area-weighted-self-adjoint A is
   itself W-self-adjoint.
2. M⁻¹ approximates A⁻¹: the residual ``||A·M⁻¹b − b||_W`` SHRINKS as the
   Chebyshev degree rises (the iteration-cut mechanism is real, not vacuous).
3. Equal-M PCG: chebyshev beats jacobi on a pole/coastal-stiffened problem.
4. AD: jax.grad through a chebyshev-preconditioned solve is finite.
5. Dispatch guards: A_op-required + unknown-name refuse loudly.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _h_total_at_faces,
    _helmholtz_inv_diag,
    _make_helmholtz,
    _make_chebyshev_preconditioner,
    _select_preconditioner,
)
from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit


def _setup(n_lat=24, n_lon=48, coastal=True):
    grid = create_latlon_grid(n_lat, n_lon)
    rng = np.random.default_rng(11)
    H = 1000.0 + 500.0 * rng.random((n_lat, n_lon))
    mask = np.ones((n_lat, n_lon))
    if coastal:
        mask[:, 10:14] = 0.0
        mask[5:8, 30:40] = 0.0
    H_u = np.zeros((n_lat, n_lon + 1))
    H_u[:, 1:-1] = 0.5 * (H[:, 1:] + H[:, :-1])
    H_u[:, 0] = H_u[:, -1] = 0.5 * (H[:, 0] + H[:, -1])
    u_wet = np.zeros_like(H_u)
    u_wet[:, 1:-1] = mask[:, 1:] * mask[:, :-1]
    u_wet[:, 0] = u_wet[:, -1] = mask[:, 0] * mask[:, -1]
    H_u = H_u * u_wet
    H_v = np.zeros((n_lat + 1, n_lon))
    H_v[1:-1, :] = 0.5 * (H[1:, :] + H[:-1, :])
    v_wet = np.zeros_like(H_v)
    v_wet[1:-1, :] = mask[1:, :] * mask[:-1, :]
    H_v = H_v * v_wet
    coeff = jnp.asarray(5.0e7)
    return (grid, jnp.asarray(H_u), jnp.asarray(H_v), coeff,
            jnp.asarray(mask), jnp.asarray(u_wet), jnp.asarray(v_wet))


def _setup_production(n_lat=24, n_lon=48, nz=4):
    """Production-path fixture: H_u/H_v built by the SAME min-depth face
    builder the solver uses (``_h_total_at_faces``), from a 3D per-level
    thickness with coastal dry columns + a partial-cell step.  Exercises
    the real stencil weights the Gershgorin λ_max bound depends on — not a
    hand-rolled arithmetic-mean average (codex 2026-06-13 #7)."""
    grid = create_latlon_grid(n_lat, n_lon)
    rng = np.random.default_rng(13)
    mask = np.ones((n_lat, n_lon))
    mask[:, 10:14] = 0.0          # meridional coast
    mask[5:8, 30:40] = 0.0        # interior basin
    h_k = 100.0 + 50.0 * rng.random((n_lat, n_lon, nz))
    h_k = h_k * mask[:, :, None]
    h_k[3:6, 20:25, nz - 1] = 0.0   # partial cells (bottom level missing)
    coeff = jnp.asarray(5.0e7)
    min_water_col = jnp.asarray(1.0)
    H_u, H_v = _h_total_at_faces(
        jnp.asarray(h_k), min_water_col, jnp.asarray(mask), grid)
    u_wet = (H_u > 0).astype(jnp.float64)
    v_wet = (H_v > 0).astype(jnp.float64)
    return (grid, H_u, H_v, coeff, jnp.asarray(mask), u_wet, v_wet)


def _build(degree):
    grid, H_u, H_v, coeff, mask, u_wet, v_wet = _setup()
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    M_inv = _make_chebyshev_preconditioner(A_op, inv_diag, mask, degree)
    return grid, A_op, inv_diag, mask, M_inv


def test_chebyshev_preserves_input_dtype():
    """M_inv must return the INPUT dtype.  The spectral-window scalars
    (lmax/d/c) are computed in f64 (from inv_diag under x64); if alpha/beta
    derived from them are f64, ``x = x + alpha*p`` UPCASTS an f32 carry to f64
    and the distributed fixed-M PCG fori_loop carry-dtype check fails on the
    full ocean's f32 barotropic carry (job 8489358).  Regression: f32 in -> f32
    out (and f64 in -> f64 out, unchanged)."""
    grid, A_op, inv_diag, mask, M_inv = _build(4)
    rng = np.random.default_rng(7)
    for dt in (jnp.float32, jnp.float64):
        r = jnp.asarray(rng.standard_normal(mask.shape)).astype(dt) * mask.astype(dt)
        out = M_inv(r)
        assert out.dtype == dt, f"chebyshev M_inv upcast {dt} -> {out.dtype}"
        assert np.all(np.isfinite(np.asarray(out)))


def test_chebyshev_W_self_adjoint():
    """``M_inv`` is self-adjoint in the area-weighted inner product (the
    single_reduce / Chronopoulos–Gear requirement).  Checked over several
    random wet pairs AND vectors localized at the periodic seam / coastline
    — a single global random pair can average out localized boundary/fold
    asymmetry (codex 2026-06-13 #4)."""
    grid, _, _, mask, M_inv = _build(degree=4)
    w = grid.area * mask
    n_lat, n_lon = mask.shape

    def _sym_err(x, y):
        lhs = float(jnp.sum(w * M_inv(x) * y))
        rhs = float(jnp.sum(w * x * M_inv(y)))
        return abs(lhs - rhs), max(abs(lhs), abs(rhs))

    # several global random wet pairs (O(1) scale → non-vacuous rel-tol)
    for seed in (5, 17, 29):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        y = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        err, scale = _sym_err(x, y)
        assert err <= 1e-9 * scale, f"random seed {seed}: {err} vs {scale}"

    # vectors concentrated at the periodic seam and around the coastal band,
    # where fold/periodic/coast asymmetry would surface.  Windowed (many
    # nonzeros) so the relative tolerance stays meaningful.
    windows = [
        (slice(0, n_lat), slice(0, 3)),                 # periodic seam i≈0
        (slice(0, n_lat), slice(n_lon - 3, n_lon)),     # periodic seam wrap
        (slice(4, 9), slice(8, 16)),                    # around coastal band
    ]
    for wi, win in enumerate(windows):
        rng = np.random.default_rng(100 + wi)
        xv = np.zeros(mask.shape)
        yv = np.zeros(mask.shape)
        xv[win] = rng.standard_normal(xv[win].shape)
        yv[win] = rng.standard_normal(yv[win].shape)
        x = jnp.asarray(xv) * mask
        y = jnp.asarray(yv) * mask
        err, scale = _sym_err(x, y)
        assert err <= 1e-9 * scale, f"window {wi}: {err} vs {scale}"


def test_chebyshev_residual_shrinks_with_degree():
    """||A·M⁻¹b − b||_W decreases monotonically with the Chebyshev degree —
    the preconditioner genuinely approximates A⁻¹ better at higher degree
    (non-vacuous: a no-op M⁻¹ would not improve)."""
    grid, _, _, mask, _ = _build(degree=1)
    rng = np.random.default_rng(7)
    b = (jnp.asarray(rng.standard_normal(mask.shape)) * mask)
    w = grid.area * mask

    def _resid(degree):
        g, A_op, _, m, M_inv = _build(degree)
        r = A_op(M_inv(b)) - b
        return float(jnp.sqrt(jnp.sum(w * r * r)))

    r1, r2, r4 = _resid(1), _resid(2), _resid(4)
    assert r2 < r1, f"degree-2 residual {r2} not < degree-1 {r1}"
    assert r4 < r2, f"degree-4 residual {r4} not < degree-2 {r2}"


@pytest.mark.parametrize("variant", ["standard", "single_reduce"])
def test_chebyshev_beats_jacobi_equal_M(variant):
    """Equal fixed-M PCG: the chebyshev-preconditioned residual is no worse
    (and generally better) than jacobi on the pole/coastal-stiffened
    problem — the lever's point.  Also exercises the single_reduce path
    (the W-self-adjoint composition)."""
    grid, H_u, H_v, coeff, mask, u_wet, v_wet = _setup()
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    w = grid.area * mask
    rng = np.random.default_rng(9)
    rhs = (jnp.asarray(rng.standard_normal(mask.shape)) * mask)
    x0 = jnp.zeros_like(rhs)

    def _resid(M_inv, m_iters):
        _, diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, x0, distributed=True, fixed_iters=m_iters,
            residual_tol=1.0e-30, stock_cg_tol=1.0e-12,
            stock_cg_maxiter=200, pcg_variant=variant, dot_weight=w)
        return float(diag.rel_residual)

    m = 8
    jac = _resid(lambda r: r * inv_diag.astype(r.dtype), m)
    cheb = _resid(_make_chebyshev_preconditioner(A_op, inv_diag, mask, 4), m)
    assert np.isfinite(cheb) and cheb < jac, (
        f"chebyshev equal-M={m} residual {cheb:.3e} should beat jacobi "
        f"{jac:.3e} ({variant})")


def test_chebyshev_grad_minv_finite():
    """Focused: jax.grad through a bare ``M_inv`` apply is finite (the
    polynomial recurrence itself is AD-clean)."""
    grid, H_u, H_v, coeff, mask, u_wet, v_wet = _setup()

    def loss(scale):
        Hu = H_u * scale
        A_op = _make_helmholtz(Hu, H_v, coeff, grid, mask, u_wet, v_wet)
        inv_diag = _helmholtz_inv_diag(Hu, H_v, coeff, grid, mask)
        M_inv = _make_chebyshev_preconditioner(A_op, inv_diag, mask, 4)
        b = jnp.ones_like(mask) * mask
        return jnp.sum(M_inv(b) ** 2)   # jnp scalar (float() would break grad)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(np.asarray(g)), f"non-finite grad {g}"


def test_chebyshev_grad_through_solve():
    """jax.grad of a scalar of the SOLUTION through a full chebyshev-
    preconditioned PCG solve — exercises the solver call, the PCG
    recurrence and the weighted dot reductions, not just the bare
    ``M_inv`` apply (codex 2026-06-13 #5).

    Differentiated w.r.t. an RHS scale ``s``: the fixed-iteration PCG is
    EXACTLY degree-1 homogeneous in ``b`` — α,β are ratios of quantities
    quadratic in the residual, hence invariant under ``b→s·b``, so
    ``eta(s·b)=s·eta(b)``.  Thus ``loss(s)=s²·loss(1)`` and the autodiff
    gradient MUST equal ``2·loss(1)`` to round-off, regardless of how far
    the 6-iteration solve has converged.  This is a rigorous AD-CORRECTNESS
    check (not just finite/non-zero) whose cotangent path runs back through
    the chebyshev matvec recurrence (``M_inv`` is applied to ``r0∝s``)."""
    grid, H_u, H_v, coeff, mask, u_wet, v_wet = _setup()
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    M_inv = _make_chebyshev_preconditioner(A_op, inv_diag, mask, 4)
    w = grid.area * mask
    rhs = jnp.ones_like(mask) * mask

    def loss(s):
        eta, _ = solve_helmholtz_implicit(
            A_op, rhs * s, M_inv, jnp.zeros_like(rhs), distributed=True,
            fixed_iters=6, residual_tol=1.0e-30, stock_cg_tol=1.0e-12,
            stock_cg_maxiter=200, pcg_variant="standard", dot_weight=w)
        return jnp.sum(w * eta * eta)

    loss1 = float(loss(1.0))
    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g) and loss1 > 0.0, f"degenerate: g={g}, loss1={loss1}"
    assert abs(g - 2.0 * loss1) <= 1e-6 * (2.0 * loss1), (
        f"AD grad through solve {g:.6e} != homogeneity prediction "
        f"2·loss(1)={2.0 * loss1:.6e}")


def test_chebyshev_production_face_thickness():
    """W-self-adjoint + degree-monotone residual on an operator built by
    the PRODUCTION min-depth face-thickness path (``_h_total_at_faces``)
    with coastal dry columns and a partial-cell step.  The residual-shrink
    is the numerical proof that the Gershgorin λ_max = max(2·diag−1) is a
    TRUE upper bound on the PRODUCTION spectrum — if it under-estimated
    λ_max the Chebyshev polynomial would not contract there (codex
    2026-06-13 #7, the check codex could not run in-sandbox)."""
    grid, H_u, H_v, coeff, mask, u_wet, v_wet = _setup_production()
    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    w = grid.area * mask

    M_inv = _make_chebyshev_preconditioner(A_op, inv_diag, mask, 4)
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    y = jnp.asarray(rng.standard_normal(mask.shape)) * mask
    lhs = float(jnp.sum(w * M_inv(x) * y))
    rhs_ = float(jnp.sum(w * x * M_inv(y)))
    assert abs(lhs - rhs_) <= 1e-9 * max(abs(lhs), abs(rhs_)), (
        f"production operator: M_inv not W-self-adjoint: {lhs!r} vs {rhs_!r}")

    b = jnp.asarray(rng.standard_normal(mask.shape)) * mask

    def _resid(deg):
        Mi = _make_chebyshev_preconditioner(A_op, inv_diag, mask, deg)
        r = A_op(Mi(b)) - b
        return float(jnp.sqrt(jnp.sum(w * r * r)))

    r1, r2, r4 = _resid(1), _resid(2), _resid(4)
    assert r2 < r1 and r4 < r2, (
        f"production residuals not shrinking (λ_max bound too low?): "
        f"{r1}, {r2}, {r4}")


def test_chebyshev_dispatch_guards():
    grid, H_u, H_v, coeff, mask, _, _ = _setup()
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    # A_op required.
    with pytest.raises(ValueError, match="chebyshev preconditioner requires"):
        _select_preconditioner("chebyshev", inv_diag, H_u, H_v, coeff, grid,
                               mask, A_op=None)
    # Unknown name still refuses (now lists chebyshev).
    with pytest.raises(ValueError, match="unknown"):
        _select_preconditioner("bogus", inv_diag, H_u, H_v, coeff, grid, mask)
