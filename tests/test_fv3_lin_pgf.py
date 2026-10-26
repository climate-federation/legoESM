"""Unit tests for the FV3-faithful Lin (1997) hydrostatic PGF.

Validates the cross-product PGF port in
``legoesm.atmosphere._future._fv3_lin_pgf`` against:

1. **Exact hydrostatic cancellation** — for any spatially uniform
   hydrostatic state (T const, p_s const, phis const), the C-grid PGF
   must be zero at machine precision.  This is the construction
   property of the Lin (1997) cross-product (see
   ``../FV3/atmos_cubed_sphere-symmetryclean/model/dyn_core.F90:p_grad_c``
   line 2073).

2. **Surface-pressure tilt drives the expected face-aligned PGF** —
   when p_s has a controlled gradient along x, ∂u_c/∂t at x-faces
   should be non-zero with the correct sign (force points from high
   to low p_s); ∂v_c/∂t at y-faces should remain near zero.

3. **D-grid projection preserves zero** — applying
   ``project_cgrid_pgf_to_dgrid_corners`` to the zero PGF gives zero
   at corners.

4. **gz_half is consistent with the FV3 recurrence** — verify that
   ``gz_half(k+1) - gz_half(k) = -cp * theta(k) * (pk(k+1) - pk(k))``
   in any column (independent of horizontal structure).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere._future._fv3_lin_pgf import (
    compute_geopotential_half_fv3,
    compute_pkappa_half,
    fv3_lin1997_pgf_3d_cgrid,
    project_cgrid_pgf_to_dgrid_corners,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_cube():
    n = 8
    nlev = 10
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    return grid, cdgrid, coord, n, nlev


def test_uniform_hydrostatic_state_zero_pgf(small_cube):
    """Uniform IC: cross-product PGF is zero at machine precision."""
    _, cdgrid, coord, n, nlev = small_cube

    T = jnp.full((6, n, n, nlev), 280.0)
    p_s = jnp.full((6, n, n), constants.p_ref)
    phis = jnp.zeros((6, n, n))

    pgf_x_c, pgf_y_c = fv3_lin1997_pgf_3d_cgrid(T, p_s, phis, coord, cdgrid)

    # Cross-product gives EXACT cancellation in pure hydrostatic state.
    # Only floating-point roundoff remains.
    assert pgf_x_c.shape == (6, n + 1, n, nlev)
    assert pgf_y_c.shape == (6, n, n + 1, nlev)
    max_x = float(jnp.max(jnp.abs(pgf_x_c)))
    max_y = float(jnp.max(jnp.abs(pgf_y_c)))
    # The denominator is ~p_ref^κ ≈ 100 Pa^κ for the lowest layer; the
    # numerator scales with cp*T*δpk^2 ≈ 1004 * 280 * (10000)^2 = 2.8e13
    # in the LARGEST layer, divided by sum_wks ~ 20000 → ratio ~1.4e9.
    # Multiplied by rdxc ~ 1e-6 → tendency ~ 1e3.  Roundoff should be
    # ~1e-9 of that signal in float64, i.e. ~1e-6.  Use a generous bound.
    assert max_x < 1e-5, f"x-PGF max {max_x:.3e} > 1e-5 in uniform state"
    assert max_y < 1e-5, f"y-PGF max {max_y:.3e} > 1e-5 in uniform state"


def test_uniform_hydrostatic_dgrid_projection_zero(small_cube):
    """D-grid projection of zero C-grid PGF gives zero corners."""
    _, cdgrid, coord, n, nlev = small_cube

    T = jnp.full((6, n, n, nlev), 280.0)
    p_s = jnp.full((6, n, n), constants.p_ref)
    phis = jnp.zeros((6, n, n))

    pgf_x_c, pgf_y_c = fv3_lin1997_pgf_3d_cgrid(T, p_s, phis, coord, cdgrid)
    pgf_x_d, pgf_y_d = project_cgrid_pgf_to_dgrid_corners(pgf_x_c, pgf_y_c)

    assert pgf_x_d.shape == (6, n + 1, n + 1, nlev)
    assert pgf_y_d.shape == (6, n + 1, n + 1, nlev)
    assert float(jnp.max(jnp.abs(pgf_x_d))) < 1e-5
    assert float(jnp.max(jnp.abs(pgf_y_d))) < 1e-5


def test_geopotential_half_recurrence_consistency(small_cube):
    """gz_half satisfies the FV3 bottom-up recurrence."""
    _, _, coord, n, nlev = small_cube

    rng = np.random.default_rng(seed=42)
    T_np = rng.uniform(220.0, 310.0, size=(6, n, n, nlev))
    p_s_np = rng.uniform(0.7e5, 1.05e5, size=(6, n, n))
    phis_np = rng.uniform(0.0, 5e4, size=(6, n, n))

    T = jnp.asarray(T_np)
    p_s = jnp.asarray(p_s_np)
    phis = jnp.asarray(phis_np)

    gz_half = compute_geopotential_half_fv3(T, p_s, phis, coord)
    pk_half = compute_pkappa_half(p_s, coord)

    p_full = coord.A_full * coord.p_ref + coord.B_full * p_s[..., None]
    p_full_safe = jnp.maximum(p_full, 1.0)
    dpk = pk_half[..., 1:] - pk_half[..., :-1]
    # Correct dgz uses cp*T*p^(-kappa)*dpk (see compute_geopotential_half_fv3
    # docstring).  This matches the hydrostatic ∂Φ/∂p = -RT/p relation.
    dgz_expected = (
        constants.c_pd * T * p_full_safe ** (-constants.kappa) * dpk
    )

    # Recurrence: gz_half(k) - gz_half(k+1) = dgz(k).
    dgz_actual = gz_half[..., :-1] - gz_half[..., 1:]
    err = float(jnp.max(jnp.abs(dgz_actual - dgz_expected)))
    rel = err / float(jnp.max(jnp.abs(dgz_expected)) + 1e-30)
    # Cumsum + slicing is exact in float64.
    assert rel < 1e-12, f"gz_half recurrence rel err = {rel:.3e}"

    # Surface boundary: gz_half[..., -1] == phis.
    np.testing.assert_allclose(
        np.asarray(gz_half[..., -1]), phis_np, atol=1e-9,
    )


def test_surface_pressure_tilt_produces_pgf(small_cube):
    """A controlled p_s tilt drives ∂u_c/∂t with the correct sign."""
    _, cdgrid, coord, n, nlev = small_cube

    # Uniform T, but p_s linearly increasing in i on face 0.  This
    # mimics a low-altitude high-pressure ridge: PGF should push fluid
    # FROM high p_s TOWARD low p_s, i.e. from large-i toward small-i.
    # On our gnomonic face 0, larger i = larger longitude (eastward).
    # So ∂u_c/∂t should be NEGATIVE (westward force).
    T = jnp.full((6, n, n, nlev), 280.0)
    phis = jnp.zeros((6, n, n))

    p_s_base = jnp.full((6, n, n), constants.p_ref)
    # Add 100 Pa per cell along i on face 0 only.
    i_idx = jnp.arange(n, dtype=jnp.float64)
    p_s_face0_tilt = p_s_base[0] + 100.0 * i_idx[:, None]
    p_s = p_s_base.at[0].set(p_s_face0_tilt)

    pgf_x_c, pgf_y_c = fv3_lin1997_pgf_3d_cgrid(T, p_s, phis, coord, cdgrid)

    # On face 0, away from face boundaries (i ∈ [2, n-2], j ∈ [2, n-2]),
    # ∂u_c/∂t should be NEGATIVE in the lowest layer (largest |δpk|).
    interior_pgf_x = pgf_x_c[0, 2:n-1, 2:n-2, -1]   # bottom layer
    # mean direction
    mean_pgf_x = float(jnp.mean(interior_pgf_x))
    assert mean_pgf_x < 0, (
        f"Expected NEGATIVE x-PGF on face 0 with eastward p_s tilt, "
        f"got mean = {mean_pgf_x:.3e}"
    )

    # On other faces (1-5) p_s is uniform → x-PGF should be tiny.
    other_pgf_x = pgf_x_c[1:, 2:n-1, 2:n-2, -1]
    other_max = float(jnp.max(jnp.abs(other_pgf_x)))
    # Cross-face halo from face 0 will leak some non-zero PGF into face
    # 1's western edge.  Allow a generous bound but require it to be
    # much smaller than face 0's signal.
    face0_signal = float(jnp.max(jnp.abs(interior_pgf_x)))
    assert other_max < 0.5 * face0_signal, (
        f"Other faces show {other_max:.3e}, face 0 signal {face0_signal:.3e}"
    )


def test_denom_floor_is_strictly_nonzero_at_zero():
    """Regression: the denominator floor must be NONZERO when denom==0.

    ``jnp.sign(0.0) == 0.0`` so the old guard
    ``where(|d|>eps, d, sign(d)*eps)`` returned 0 at ``d==0`` and
    reintroduced a 0/0 (NaN value AND NaN gradient).  The fixed guard
    maps sign(0) -> +1.  We replicate the in-module floor expression and
    assert it is strictly positive at ``d==0`` and has a finite gradient.
    """
    from legoesm.atmosphere._future._fv3_lin_pgf import _PGF_DENOM_FLOOR

    def safe_denom(d):
        sgn = jnp.sign(d)
        floor_sign = sgn + (1.0 - jnp.abs(sgn))
        return jnp.where(jnp.abs(d) > _PGF_DENOM_FLOOR, d, floor_sign * _PGF_DENOM_FLOOR)

    # value at exactly zero is the +floor, not 0.
    val0 = float(safe_denom(jnp.array(0.0)))
    assert val0 == pytest.approx(_PGF_DENOM_FLOOR), val0
    assert val0 != 0.0

    # 1/denom and its gradient are finite at d==0 (the actual divide).
    def recip(d):
        return 1.0 / safe_denom(d)

    r = float(recip(jnp.array(0.0)))
    g = float(jax.grad(recip)(jnp.array(0.0)))
    assert jnp.isfinite(r), r
    assert jnp.isfinite(g), g


def test_pgf_gradient_finite_through_full_field(small_cube):
    """End-to-end: jax.grad through the full PGF is finite even when a
    column is pushed to a near-degenerate (zero-δp^κ) state.

    We differentiate a scalar of the PGF w.r.t. ``p_s`` and ``T`` at a
    realistic state and assert NO NaN/Inf gradients anywhere.
    """
    _, cdgrid, coord, n, nlev = small_cube

    T = jnp.full((6, n, n, nlev), 285.0)
    phis = jnp.zeros((6, n, n))
    p_s = jnp.full((6, n, n), constants.p_ref)

    def scalar_of_pgf(p_s_in, T_in):
        px, py = fv3_lin1997_pgf_3d_cgrid(T_in, p_s_in, phis, coord, cdgrid)
        return jnp.sum(px ** 2) + jnp.sum(py ** 2)

    gp, gt = jax.grad(scalar_of_pgf, argnums=(0, 1))(p_s, T)
    assert bool(jnp.all(jnp.isfinite(gp))), "non-finite grad wrt p_s"
    assert bool(jnp.all(jnp.isfinite(gt))), "non-finite grad wrt T"


def _coord_with_zero_thickness_layer(coord, k_dup=3):
    """Return a copy of ``coord`` with interface ``k_dup+1`` collapsed onto
    ``k_dup`` so layer ``k_dup`` has EXACTLY zero pressure thickness -> the
    production ``compute_pkappa_half`` yields δp^κ == 0 -> the Lin PGF
    denominator ``wk_W + wk_E`` is EXACTLY 0 at that level (drives the real
    divide-by-zero the floor guards).  ``A_full``/``B_full``/``dA``/``dB`` for
    the degenerate layer are set consistently so the geopotential recurrence
    stays finite.
    """
    A_half = coord.A_half.at[k_dup + 1].set(coord.A_half[k_dup])
    B_half = coord.B_half.at[k_dup + 1].set(coord.B_half[k_dup])
    A_full = 0.5 * (A_half[:-1] + A_half[1:])
    B_full = 0.5 * (B_half[:-1] + B_half[1:])
    dA = A_half[1:] - A_half[:-1]
    dB = B_half[1:] - B_half[:-1]
    return coord._replace(
        A_half=A_half, B_half=B_half, A_full=A_full, B_full=B_full,
        dA=dA, dB=dB,
    )


def test_pgf_production_denominator_exactly_zero_is_finite(small_cube):
    """Regression for the sign(0)->+1 denominator floor, driven through the
    PRODUCTION ``fv3_lin1997_pgf_3d_cgrid`` with a coordinate that makes the
    Lin cross-product denominator EXACTLY zero (a zero-thickness layer).

    Without the floor (or with the old ``sign(d)*eps`` that returns 0 at d==0)
    this produces a 0/0 -> NaN value AND NaN gradient.  Assert both the PGF and
    its gradient are finite at the degenerate level.
    """
    _, cdgrid, coord, n, nlev = small_cube
    k_dup = 3
    deg_coord = _coord_with_zero_thickness_layer(coord, k_dup=k_dup)

    # Confirm the production denominator really is exactly zero at k_dup
    # (non-vacuous): p^κ is equal across the collapsed interface.
    p_s = jnp.full((6, n, n), constants.p_ref)
    pk_half = compute_pkappa_half(p_s, deg_coord)        # (6,n,n,nlev+1)
    wk = pk_half[..., 1:] - pk_half[..., :-1]            # (6,n,n,nlev)
    assert float(jnp.max(jnp.abs(wk[..., k_dup]))) == 0.0, "layer not degenerate"

    T = jnp.full((6, n, n, nlev), 285.0)
    phis = jnp.zeros((6, n, n))

    px, py = fv3_lin1997_pgf_3d_cgrid(T, p_s, phis, deg_coord, cdgrid)
    assert bool(jnp.all(jnp.isfinite(px))), "non-finite PGF (x) at zero denom"
    assert bool(jnp.all(jnp.isfinite(py))), "non-finite PGF (y) at zero denom"

    def scalar_of_pgf(p_s_in, T_in):
        qx, qy = fv3_lin1997_pgf_3d_cgrid(T_in, p_s_in, phis, deg_coord, cdgrid)
        return jnp.sum(qx ** 2) + jnp.sum(qy ** 2)

    gp, gt = jax.grad(scalar_of_pgf, argnums=(0, 1))(p_s, T)
    assert bool(jnp.all(jnp.isfinite(gp))), "non-finite grad wrt p_s at zero denom"
    assert bool(jnp.all(jnp.isfinite(gt))), "non-finite grad wrt T at zero denom"


# Parked module: see its docstring.
pytestmark = pytest.mark.skip(
    reason="parked in _future/: not wired into production (ponytail #9)")
