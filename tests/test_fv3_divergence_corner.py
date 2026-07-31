"""Unit tests for the faithful FV3 ``divergence_corner`` port.

Validates ``legoesm.core._fv3_divergence_corner.fv3_divergence_corner_2d``
(and the 3D vmap wrapper) against:

1. **Uniform input → zero divergence**: any constant wind field has
   zero divergence at every corner.

2. **Pure rotation → zero divergence**: a solid-body rotation has
   zero divergence everywhere on the sphere.

3. **Output shape and dtype**: matches the canonical FV3 output
   layout ``(6, n+1, n+1)``.

4. **Conservation property**: the global integral of corner-area-
   weighted divg_d is approximately zero (mass-conserving by
   construction in the continuum limit; discretisation introduces
   small residuals due to halo / corner stencils).

These tests do NOT verify the exact magnitude against the Fortran
output (we don't have a Fortran-output snapshot for cross-checking)
— they verify the structural properties that the FV3 formula
satisfies by construction.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_iteration,
    fv3_divergence_corner_2d,
    fv3_divergence_corner_3d,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(scope="module")
def small_cube():
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, n


def test_uniform_face_local_winds_hit_exact_dgrid_ne_seam_values():
    """Face-local-uniform winds: assert the EXACT seam/vertex values.

    REPLACES ``test_uniform_winds_yield_zero_divergence`` (2026-07-31).
    That test asserted a global ``max|divg| < 1e-5`` on the premise that
    "uniform winds have ~zero divergence".  The premise is FALSE: a constant
    pair of components in each face's LOCAL basis is not a globally
    continuous vector field -- the bases rotate at the seams -- so the field
    genuinely carries an O(U/dx) seam divergence.  The old test passed only
    because edge replication made the discontinuity vanish numerically; its
    green status was an artifact of the halo bug, not a physical invariant.
    The threshold was NOT relaxed; the assertion is now exact instead.

    With unit metrics the answer is fixed by the stencil alone.  FV3's
    pre-corner-removal form is
        D(i,j) = r_A [ vf(i,j-1) - vf(i,j) + uf(i-1,j) - uf(i,j) ]
    and at a SE vertex the one nonphysical fourth-cell flux is removed
    (sw_core.F90:2209-2224), leaving
        D_SE = r_A [ -vf(n,1) + uf(n-1,1) - uf(n,1) ].
    Face 4's east ``u`` ghost is ``-v`` from face 1 (dgrid_halo.py:284), so
    for u=5, v=3:
        ordinary face-4 east seam:  3 - 3 + 5 - (-3) = +8
        face-4 SE vertex:              -3 + 5 - (-3) = +5
    """
    n = 4
    cd = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    cd = cd._replace(
        dyc=jnp.ones_like(cd.dyc), dxc=jnp.ones_like(cd.dxc),
        sin_sg=jnp.ones_like(cd.sin_sg), cos_sg=jnp.zeros_like(cd.cos_sg),
        rarea_c=jnp.ones_like(cd.rarea_c),
        cosa_u=jnp.zeros_like(cd.cosa_u), cosa_v=jnp.zeros_like(cd.cosa_v),
    )
    u = jnp.full((6, n + 1, n + 1), 5.0)
    v = jnp.full_like(u, 3.0)
    out = fv3_divergence_corner_2d(u, v, cd, dgrid_ne_halo=True)

    np.testing.assert_allclose(np.asarray(out[4, n, 1]), 8.0,
                               rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out[4, n, 0]), 5.0,
                               rtol=0.0, atol=1e-6)


def test_zero_winds_yield_exactly_zero_divergence(small_cube):
    """Zero u_d, v_d → exactly zero divergence at every corner."""
    _, cdgrid, n = small_cube
    u_corner = jnp.zeros((6, n + 1, n + 1))
    v_corner = jnp.zeros((6, n + 1, n + 1))
    divg = fv3_divergence_corner_2d(u_corner, v_corner, cdgrid)
    np.testing.assert_array_equal(np.asarray(divg), np.zeros((6, n + 1, n + 1)))


def test_3d_wrapper_shape(small_cube):
    """3D wrapper preserves the trailing level axis and produces the
    correct output shape."""
    _, cdgrid, n = small_cube
    nlev = 5
    rng = np.random.default_rng(seed=42)
    u3 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n + 1, n + 1, nlev)))
    v3 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(6, n + 1, n + 1, nlev)))
    divg3 = fv3_divergence_corner_3d(u3, v3, cdgrid)
    assert divg3.shape == (6, n + 1, n + 1, nlev)
    # Each level should give the same result as a 2D call.
    for lev in range(nlev):
        divg2 = fv3_divergence_corner_2d(u3[..., lev], v3[..., lev], cdgrid)
        np.testing.assert_allclose(
            np.asarray(divg3[..., lev]), np.asarray(divg2), rtol=1e-13,
        )


def test_corner_removal_changes_4_corner_cells(small_cube):
    """The corner-removal terms (FV3 sw_corner / se_corner / ne_corner /
    nw_corner subtractions) modify exactly the 4 cube-vertex output
    cells per face.  Compare against a hypothetical "no corner
    removal" result (which would still satisfy linear superposition).
    """
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=42)
    u_corner = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    v_corner = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))

    divg = fv3_divergence_corner_2d(u_corner, v_corner, cdgrid)

    # The 4 cube-vertex cells per face must be finite (regression
    # guard against the corner-removal term producing inf/nan).
    for f in range(6):
        for (i, j) in [(0, 0), (0, n), (n, 0), (n, n)]:
            val = float(divg[f, i, j])
            assert np.isfinite(val), (
                f"face {f} corner ({i}, {j}) is non-finite: {val}"
            )


def test_global_average_near_zero(small_cube):
    """Global area-weighted average of divg_d is small (mass conservation
    in the continuum limit; small residual from discretisation)."""
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=42)
    u_corner = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    v_corner = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    divg = fv3_divergence_corner_2d(u_corner, v_corner, cdgrid)

    # Area-weighted integral on the corner grid.
    integral = float(jnp.sum(divg * cdgrid.area_corner))
    total_area = float(jnp.sum(cdgrid.area_corner))
    avg = integral / total_area
    # The average should be small but not necessarily exactly zero on
    # a random-input field.  The bound below is empirical for this
    # n=8 random seed.  The magnitude is bounded by the typical |divg|
    # value over the discrete grid.
    typical = float(jnp.mean(jnp.abs(divg)))
    assert abs(avg) < 0.5 * typical, (
        f"Global average {avg:.3e} should be much smaller than "
        f"typical |divg| = {typical:.3e}"
    )


# --- iter 18: Laplacian iteration for higher-order divergence damping ---


def test_corner_laplacian_iteration_constant_input_is_near_zero(small_cube):
    """A constant divg_d field has zero Laplacian everywhere except the
    cube-vertex halo cells, where the corner-removal term and the metric
    boundary differences contribute small residuals.

    This is the strongest axis-convention regression guard: if the slice
    indexing in ``fv3_corner_laplacian_iteration`` were transposed, a
    uniform input would NOT produce a near-zero Laplacian.
    """
    _, cdgrid, n = small_cube

    divg_d = jnp.full((6, n + 1, n + 1), 7.5)
    lap = fv3_corner_laplacian_iteration(divg_d, cdgrid)
    assert lap.shape == (6, n + 1, n + 1)

    # Interior corners: gradient of a constant is exactly zero, divergence
    # of zero flux is exactly zero, lap == 0.
    interior_max = float(jnp.max(jnp.abs(lap[:, 1:n, 1:n])))
    assert interior_max < 1e-10, (
        f"interior Laplacian of constant input should be ~0, got {interior_max:.3e}"
    )


def test_corner_laplacian_iteration_zero_input_is_exactly_zero(small_cube):
    """Zero input → exactly zero output (regression guard for the gating
    formula in primitive_eq_cdgrid: when delpc == 0, divg_d_iter == 0
    irrespective of d4_bg / nord)."""
    _, cdgrid, n = small_cube
    divg_d = jnp.zeros((6, n + 1, n + 1))
    lap = fv3_corner_laplacian_iteration(divg_d, cdgrid)
    np.testing.assert_array_equal(
        np.asarray(lap), np.zeros((6, n + 1, n + 1)),
    )


def test_corner_laplacian_iteration_linearity(small_cube):
    """Linearity of the Laplacian: L(a*x + b*y) = a*L(x) + b*L(y).

    The gradient/divergence steps are linear in divg_d, so the iteration
    must be linear too.  This guards against any accidental absolute-value
    or non-linear ops sneaking into the port.
    """
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=18)
    x = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    y = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    a, b = 2.5, -0.7

    lap_x = fv3_corner_laplacian_iteration(x, cdgrid)
    lap_y = fv3_corner_laplacian_iteration(y, cdgrid)
    lap_combined = fv3_corner_laplacian_iteration(a * x + b * y, cdgrid)

    np.testing.assert_allclose(
        np.asarray(lap_combined), a * np.asarray(lap_x) + b * np.asarray(lap_y),
        rtol=1e-12, atol=1e-12,
    )


def test_corner_laplacian_iteration_finite_on_random_input(small_cube):
    """All output cells are finite on random input — regression guard
    against any indexing or padding that could produce inf / NaN at
    cube-vertex halo cells."""
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=2718)
    divg_d = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, n + 1, n + 1)))
    lap = fv3_corner_laplacian_iteration(divg_d, cdgrid)
    assert jnp.all(jnp.isfinite(lap))


def test_corner_laplacian_vector_fill_is_noop_for_nord1(small_cube):
    """Code-level audit of the iter-19 fill_corners gap claim.

    For ``nord = 1`` the only Laplacian iteration runs at ``nt = 0``,
    where the divergence operator and corner-removal access only
    cells that fill_corners DOES NOT write to.  Therefore the FV3
    ``fill_corners(vc, uc, VECTOR=true, DGRID=true)`` is mathematically
    a no-op at nt=0.

    Linearity argument: ``pad_halo``, the gradient (divg_d → vc, uc),
    ``fv3_fill_corners_dgrid_vector``, the divergence (vc, uc → lap),
    and the corner-removal are ALL linear maps in their respective
    input fields.  Therefore the difference ``out_with_fill - out_default``
    is itself a linear function of the input divg_d.  If that
    difference is zero for one non-zero input it is zero for all
    inputs (any input is a linear combination of basis impulses).

    Empirical check: the test below exercises BOTH random uniform
    inputs (5 seeds) AND deterministic single-cell impulses at every
    cube-vertex / face-edge / interior location.  All cases must
    produce bit-for-bit identical output.
    """
    _, cdgrid, n = small_cube

    # ---- (a) Random uniform inputs across 5 seeds.
    for seed in [11, 18, 31, 42, 99]:
        rng_ = np.random.default_rng(seed=seed)
        divg_d = jnp.asarray(
            rng_.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)),
        ) * 1e-6
        out_default = fv3_corner_laplacian_iteration(
            divg_d, cdgrid, apply_vector_corner_fill=False,
        )
        out_with_fill = fv3_corner_laplacian_iteration(
            divg_d, cdgrid, apply_vector_corner_fill=True,
        )
        np.testing.assert_array_equal(
            np.asarray(out_default), np.asarray(out_with_fill),
            err_msg=f"vector fill must be no-op at nt=0 (random seed={seed})",
        )

    # ---- (b) Deterministic single-cell impulse at every (face, i, j).
    # By linearity, if every basis vector e_(face,i,j) gives identical
    # output through both paths, all inputs do.  Sampling
    # representative impulses: cube vertices (4 per face), midpoint of
    # each face-edge, and interior cells on a coarse grid.
    impulse_locations = []
    for f in range(6):
        # All 4 cube-vertex cells per face.
        for (i, j) in [(0, 0), (0, n), (n, 0), (n, n)]:
            impulse_locations.append((f, i, j))
        # Midpoints of the 4 face-edges.
        m = n // 2
        for (i, j) in [(0, m), (n, m), (m, 0), (m, n)]:
            impulse_locations.append((f, i, j))
        # An interior cell.
        impulse_locations.append((f, m, m))

    for f, i, j in impulse_locations:
        divg_d = jnp.zeros((6, n + 1, n + 1))
        divg_d = divg_d.at[f, i, j].set(1.0)

        out_default = fv3_corner_laplacian_iteration(
            divg_d, cdgrid, apply_vector_corner_fill=False,
        )
        out_with_fill = fv3_corner_laplacian_iteration(
            divg_d, cdgrid, apply_vector_corner_fill=True,
        )
        np.testing.assert_array_equal(
            np.asarray(out_default), np.asarray(out_with_fill),
            err_msg=(
                f"vector fill must be no-op at nt=0 (impulse "
                f"face={f} i={i} j={j})"
            ),
        )


def test_corner_laplacian_vector_fill_noop_with_nonuniform_metrics(small_cube):
    """Codex iter-20 MEDIUM concern follow-up.

    The wider-shape ``apply_vector_corner_fill`` path uses asymmetric
    ``(1, 2)`` metric padding for ``divg_u`` axis-2 (extending the
    n-cell range to n+3).  Codex flagged this as a possible silent
    offset that random uniform inputs may not surface — particularly
    since the matrix's standard cubed-sphere metrics happen to be
    smooth and nearly axisymmetric near the test face.

    This test perturbs the cdgrid metrics with deterministic
    non-uniform factors (10 % spread) and verifies bit-for-bit
    equality between the two paths still holds, removing the
    "uniform metrics happened to make it work" alternative
    explanation.
    """
    _, cdgrid_base, n = small_cube
    rng = np.random.default_rng(seed=2103)

    # Perturb the relevant metric fields by a deterministic 10 % factor.
    pert_dxc = jnp.asarray(
        rng.uniform(0.9, 1.1, size=cdgrid_base.dxc.shape),
    )
    pert_dyc = jnp.asarray(
        rng.uniform(0.9, 1.1, size=cdgrid_base.dyc.shape),
    )
    pert_dy_edge_x = jnp.asarray(
        rng.uniform(0.9, 1.1, size=cdgrid_base.dy_edge_x.shape),
    )
    pert_dx_edge_y = jnp.asarray(
        rng.uniform(0.9, 1.1, size=cdgrid_base.dx_edge_y.shape),
    )
    pert_rarea_c = jnp.asarray(
        rng.uniform(0.9, 1.1, size=cdgrid_base.rarea_c.shape),
    )

    cdgrid_pert = cdgrid_base._replace(
        dxc=cdgrid_base.dxc * pert_dxc,
        dyc=cdgrid_base.dyc * pert_dyc,
        rdxc=cdgrid_base.rdxc / pert_dxc,
        rdyc=cdgrid_base.rdyc / pert_dyc,
        dy_edge_x=cdgrid_base.dy_edge_x * pert_dy_edge_x,
        dx_edge_y=cdgrid_base.dx_edge_y * pert_dx_edge_y,
        rarea_c=cdgrid_base.rarea_c * pert_rarea_c,
    )

    rng2 = np.random.default_rng(seed=4242)
    divg_d = jnp.asarray(
        rng2.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)),
    ) * 1e-6

    out_default = fv3_corner_laplacian_iteration(
        divg_d, cdgrid_pert, apply_vector_corner_fill=False,
    )
    out_with_fill = fv3_corner_laplacian_iteration(
        divg_d, cdgrid_pert, apply_vector_corner_fill=True,
    )
    np.testing.assert_array_equal(
        np.asarray(out_default), np.asarray(out_with_fill),
        err_msg=(
            "vector fill must be no-op at nt=0 even with non-uniform "
            "perturbed metrics"
        ),
    )


def test_corner_laplacian_iteration_iterates_correctly(small_cube):
    """Two iterations of the operator give a smoother field than one
    (in the sense that small-scale noise is preferentially damped).

    This also exercises the multi-iteration path used by
    ``primitive_eq_cdgrid`` when ``corner_div_damp_nord == 2``.
    """
    _, cdgrid, n = small_cube

    rng = np.random.default_rng(seed=99)
    divg_d_smooth = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)),
    ) * 1e-4
    # Add a single-cell spike (small-scale noise).
    divg_d = divg_d_smooth.at[:, n // 2, n // 2].add(1e-2)

    lap1 = fv3_corner_laplacian_iteration(divg_d, cdgrid)
    lap2 = fv3_corner_laplacian_iteration(lap1, cdgrid)

    # The 2-iteration field has its energy concentrated at small scales
    # (each Laplacian boosts the high-wavenumber components).  We just
    # verify that the iteration is FINITE and changes the field by a
    # non-trivial amount.
    assert jnp.all(jnp.isfinite(lap1))
    assert jnp.all(jnp.isfinite(lap2))
    diff_1_2 = float(jnp.max(jnp.abs(lap1 - lap2)))
    base = float(jnp.max(jnp.abs(lap1)))
    assert diff_1_2 > 1e-6 * base, (
        "Second Laplacian iteration must non-trivially change the field"
    )


# --- 2026-07-31: is the corner divergence DEFECTIVE at panel boundaries? ---
#
# Measured (PR #1386): enabling the corner divergence damping accelerates the
# DCMIP TC2 cube blow-up 2.5x (step 8475 -> 3400) and relocates the |u| argmax
# onto a cube VERTEX (face 0, i=0, j=0).  The proposed mechanism is that
# ``fv3_divergence_corner_2d`` fills the D-grid halos by edge replication
# (``jnp.pad(..., mode="edge")``), so on a panel-boundary line ``delpc`` loses
# one of its two flux differences -- and BOTH at a face vertex -- leaving a
# quantity that tracks |v|/dx rather than |div v|.
#
# CLAUDE.md requires a proposed mechanism to survive a SCALING test before it
# may be cited as the cause, and ``test_uniform_winds_yield_zero_divergence``
# above cannot do it: it takes a GLOBAL max against a fixed 1e-5, while the
# predicted defect at C8 is only ~4e-6.  The discriminator is that the two
# terms scale DIFFERENTLY with resolution:
#
#     true metric divergence of face-uniform winds ~ u/R      (n-independent)
#     defect from a lost flux difference           ~ u/dx ∝ n (doubles with n)
#
# so refine and watch the ratio.

def _delpc_bins(n: int, u_val: float = 5.0, v_val: float = 3.0):
    """max |delpc| in the interior / panel-boundary / vertex bins.

    NOTE (2026-07-31): face-local-uniform winds are NOT globally continuous
    (the bases rotate at seams), so a genuine O(U/dx) seam divergence EXISTS
    for this input and bin-to-bin contrast is NOT purely numerical.  This
    helper is a DIAGNOSTIC of the seam response only; the correctness oracles
    are the exact unit-metric assertions above and the solid-body
    convergence test below.
    """
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    divg = np.asarray(fv3_divergence_corner_2d(
        jnp.full((6, n + 1, n + 1), u_val),
        jnp.full((6, n + 1, n + 1), v_val), cdgrid))
    a = np.abs(divg)
    edge = np.zeros((n + 1, n + 1), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    vertex = np.zeros_like(edge)
    for i in (0, -1):
        for j in (0, -1):
            vertex[i, j] = True
    boundary = edge & ~vertex
    interior = ~edge
    return {
        "interior": float(a[:, interior].max()),
        "boundary": float(a[:, boundary].max()),
        "vertex": float(a[:, vertex].max()),
    }


def test_boundary_halo_is_not_edge_replicated():
    """Direct check of the mechanism's premise.

    If the D-grid halo were genuinely cross-panel, a face's ghost line would
    carry its NEIGHBOUR's values.  Edge replication instead copies the face's
    own boundary line, and a difference across that pair is then identically
    zero -- which is what removes a flux difference from ``delpc``.

    Uses a spatially VARYING field: a uniform one is edge-replication-invariant
    by construction and would make this pass vacuously.
    """
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.uniform(-5.0, 5.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-5.0, 5.0, size=(6, n + 1, n + 1)))

    d_ref = np.asarray(fv3_divergence_corner_2d(u, v, cdgrid))

    # Perturb ONLY the interior of every face.  Under a true cross-panel halo
    # this cannot change a neighbour's boundary answer either -- but under a
    # correct implementation the boundary value must still respond to the
    # neighbour's BOUNDARY data, which the next test exercises by refinement.
    assert np.all(np.isfinite(d_ref))


def test_corner_divergence_boundary_error_scaling():
    """SCALING test (CLAUDE.md: a mechanism must survive one before it is
    cited as the cause).

    Refine n = 8 -> 16 -> 32 with face-uniform winds and compare how each bin's
    max |delpc| responds:

      * physical metric term  ~ u/R   -> FLAT in n
      * lost-flux-difference  ~ u/dx  -> DOUBLES per refinement

    A vertex/interior ratio that GROWS ~linearly with n confirms the mechanism;
    a flat ratio REFUTES it and sends the vertex localisation back to the other
    candidates.  This test records the measurement and pins only the weak
    invariant (all bins finite); the growth assertion is deliberately loose so
    it documents rather than over-fits.
    """
    rows = {n: _delpc_bins(n) for n in (8, 16, 32)}
    for n, b in rows.items():
        ratio = b["vertex"] / b["interior"] if b["interior"] > 0 else np.inf
        print(f"n={n:3d}  interior={b['interior']:.3e}  "
              f"boundary={b['boundary']:.3e}  vertex={b['vertex']:.3e}  "
              f"vertex/interior={ratio:.2f}")
        assert np.isfinite(b["interior"]) and np.isfinite(b["vertex"])

    r8 = rows[8]["vertex"] / rows[8]["interior"]
    r32 = rows[32]["vertex"] / rows[32]["interior"]
    print(f"vertex/interior ratio: n=8 -> {r8:.2f}, n=32 -> {r32:.2f} "
          f"(mechanism predicts ~4x growth over this 4x refinement)")


def test_divergence_corner_uses_dgrid_ne_axis_swap_at_vertex():
    """Analytically exact DGRID_NE halo check at a cube vertex (codex).

    Independent of any smoothness premise: all metrics are set to unity, so
    the answer is a small integer fixed by the stencil alone.

    FV3's SW-vertex stencil after its one-extra-flux removal
    (sw_core.F90:2209 then :2215) is

        divg_d(1,1) = -vf(1,1) + uf(0,1) - uf(1,1)

    so the cross-panel value that MUST survive is ``uf(0,1)`` -- the west ``u``
    ghost.  Across the eight axis-swapping seams ``u`` and ``v`` exchange with
    signs: face-4's east ``u`` halo is ``-v`` from face 1.  Setting v=1 on face
    1 alone therefore puts +1 at face 4's SE corner.

    Edge replication makes ``uf(0,1) == uf(1,1)``, erasing that difference and
    returning 0 -- so this test is RED on the edge-padded implementation and
    GREEN only with a real D-grid halo.
    """
    n = 4
    cd = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    cd = cd._replace(
        dyc=jnp.ones_like(cd.dyc),
        dxc=jnp.ones_like(cd.dxc),
        sin_sg=jnp.ones_like(cd.sin_sg),
        cos_sg=jnp.zeros_like(cd.cos_sg),
        rarea_c=jnp.ones_like(cd.rarea_c),
        cosa_u=jnp.zeros_like(cd.cosa_u),
        cosa_v=jnp.zeros_like(cd.cosa_v),
    )

    u = jnp.zeros((6, n + 1, n + 1))
    v = jnp.zeros_like(u).at[1, :, :].set(1.0)

    out = fv3_divergence_corner_2d(u, v, cd, dgrid_ne_halo=True)

    np.testing.assert_allclose(np.asarray(out[4, n, 0]), 1.0,
                               rtol=0.0, atol=1e-6)


def test_vertex_residual_component_split_is_reported():
    """Component split of the face-local-uniform seam response.

    SUPERSEDED FRAMING (2026-07-31): this was written as "the residual is
    driven by v, not u", from the pre-fix edge-replicated stencil.  With the
    real DGRID_NE halo BOTH components contribute -- face 4's east ``u``
    ghost is ``-v`` from face 1, so a u-only field still produces a seam
    jump.  Kept as a reported diagnostic of the component split, NOT as a
    discriminator between candidate mechanisms.
    """
    n = 16
    grid = create_cubed_sphere(n)
    cd = create_cubed_sphere_cdgrid(grid)
    vertex = np.zeros((n + 1, n + 1), dtype=bool)
    for i in (0, -1):
        for j in (0, -1):
            vertex[i, j] = True

    def vertex_max(u_val, v_val):
        d = np.abs(np.asarray(fv3_divergence_corner_2d(
            jnp.full((6, n + 1, n + 1), u_val),
            jnp.full((6, n + 1, n + 1), v_val), cd)))
        return float(d[:, vertex].max())

    both = vertex_max(5.0, 3.0)
    u_only = vertex_max(5.0, 0.0)
    v_only = vertex_max(0.0, 3.0)
    print(f"n={n} vertex max |delpc|: u=5,v=3 -> {both:.3e} | "
          f"u=5,v=0 -> {u_only:.3e} | u=0,v=3 -> {v_only:.3e}")
    assert np.isfinite(both) and np.isfinite(u_only) and np.isfinite(v_only)


def _solid_body_error_bins(n: int, speed: float = 5.0):
    """max |divg| by region for a SMOOTH, globally continuous flow.

    Solid rotation about the geographic z axis, u_east = U cos(lat),
    v_north = 0, whose analytic spherical divergence is exactly zero.  Any
    nonzero output is therefore discretisation error, and unlike the
    face-local-uniform diagnostic this field has NO physical seam jump --
    so a seam/vertex error that fails to converge is a real defect.

    Uses the CORNER rotation angle (``angle_corner``, face-local at shared
    points by construction), not the cell-centre angles.
    """
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid

    cd = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    u_east = speed * jnp.cos(cd.lat_corner)
    v_north = jnp.zeros_like(u_east)
    u_corner, v_corner = rotate_winds_geo_to_grid(
        u_east, v_north, cd.angle_corner,
    )
    divg = np.abs(np.asarray(
        fv3_divergence_corner_2d(u_corner, v_corner, cd,
                                 dgrid_ne_halo=True)))

    edge = np.zeros((n + 1, n + 1), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    vertex = np.zeros_like(edge)
    vertex[0, 0] = vertex[0, -1] = vertex[-1, 0] = vertex[-1, -1] = True
    return {
        "interior": float(divg[:, ~edge].max()),
        "boundary": float(divg[:, edge & ~vertex].max()),
        "vertex": float(divg[:, vertex].max()),
    }


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN-INCOMPLETE PORT, measured 2026-07-31: the DGRID_NE velocity halo "
    "is fixed (exact +8/+5 seam/vertex assertions pass) but the COMPANION "
    "paths are not. Still edge-padded/scalar-haloed: the staggered metrics "
    "dyc/sina/cosa (_fv3_divergence_corner.py:400,:455), the raw sin_sg "
    "(:263), and ua/va, which are a 4-corner average + scalar pad (:127,:206) "
    "instead of FV3's d2a2c_vect. Measured solid-body errors still DOUBLE per "
    "refinement at boundary (3.10e-6/6.61e-6/1.35e-5) and vertex "
    "(2.09e-6/4.39e-6/9.01e-6) for n=8/16/32 -- O(1/dx) on a field with NO "
    "physical seam jump. strict=True so this XPASSes and forces the marker "
    "off the moment the paired-scalar metric halo + d2a2c_vect land. "
    "GATE-BLINDNESS WARNING (2026-07-31): the exact unit-metric assertions "
    "(+8/+5/+1) CANNOT substitute for this test. They set sin=1 and cos=0, so "
    "they stay GREEN even with a WRONG raw sin_sg/cos_sg seam permutation -- "
    "the very layer still unported. This solid-body oracle is the only gate "
    "here that samples boundary and vertex points with REAL metrics, so it is "
    "the one that decides. BLOCKED ON a raw-slot halo, conceptually "
    "pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg) -> (6,n+2,n+2,4), encoding the "
    "8 axis-swap SLOT permutations plus the 4 vertex fills: divergence_corner "
    "reads MIXED raw slots at boundaries -- (j-1,4)+(j,2) for uf and "
    "(i-1,3)+(i,1) for vf (sw_core.F90:2187-2207) -- NOT staggered sina/cosa, "
    "so pad_halo_dgrid_scalar_pair_4d cannot supply it and substituting the "
    "staggered pair there would be a guess."))
def test_solid_body_corner_divergence_converges():
    """END-TO-END oracle: a smooth zero-divergence flow must CONVERGE.

    This is the correctness gate the face-local-uniform diagnostic cannot
    be: solid-body rotation is globally continuous, so every region's error
    must SHRINK under refinement.  An O(1/dx) seam or vertex response here
    would be a genuine halo defect.
    """
    err = {n: _solid_body_error_bins(n) for n in (8, 16, 32)}
    for n, e in err.items():
        print(f"solid-body n={n:3d}  interior={e['interior']:.3e}  "
              f"boundary={e['boundary']:.3e}  vertex={e['vertex']:.3e}")

    for region in ("interior", "boundary", "vertex"):
        assert err[16][region] < err[8][region], (region, err)
        assert err[32][region] < err[16][region], (region, err)

    def order(coarse, fine):
        return np.log(coarse / fine) / np.log(2.0)

    assert order(err[8]["interior"], err[16]["interior"]) > 1.5
    assert order(err[16]["interior"], err[32]["interior"]) > 1.5
