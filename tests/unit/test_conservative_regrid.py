"""Tests for ``legoesm.grids.conservative_regrid``.

The conservative-regrid contract is sharper than for KD-tree
interpolation:

1. **Constant fields are exactly preserved.**  Regridding a uniform
   field returns the same uniform field everywhere the source covers
   the target.
2. **Linear-in-sin-lat fields are preserved.**  The lat-band area
   factor on the sphere is ``sin(top) − sin(bot)``, so a field that is
   linear in ``sin(lat)`` (or constant in lat) integrates linearly.
3. **Sum of weights per target cell is 1** to ~1e-14 wherever the
   source grid fully covers that target cell.
4. **Total spherical integral is conserved** to ~1e-14: any source
   field that doesn't extend beyond the target domain produces a
   regridded field whose area-weighted sum equals the source's
   area-weighted sum.

These four together are the standard correctness gates for a
conservative remap (cf. Jones 1999, "First- and Second-Order
Conservative Remapping Schemes for Grids in Spherical Coordinates").
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.conservative_regrid import (
    ConservativeRegridWeights,
    _COVERAGE_TOL,
    _MIN_LAT_FLOOR,
    _attainable_lat_coverage,
    apply_conservative_regrid,
    check_axis_span,
    compute_overlap_weights,
)


# Enable x64 for these tests so we can verify ~1e-14 conservation.
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Helpers
# ============================================================================

def _regular_grid_edges(lat_min, lat_max, n_lat, lon_min, lon_max, n_lon):
    """Return (lat_edges, lon_edges) in radians for a regular lat-lon grid."""
    lat_edges = np.deg2rad(np.linspace(lat_min, lat_max, n_lat + 1))
    lon_edges = np.deg2rad(np.linspace(lon_min, lon_max, n_lon + 1))
    return lat_edges, lon_edges


def _spherical_areas(lat_edges, lon_edges):
    """Cell areas on the unit sphere: shape (n_lat, n_lon)."""
    dsin = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])  # (n_lat,)
    dlon = lon_edges[1:] - lon_edges[:-1]                   # (n_lon,)
    return dsin[:, None] * dlon[None, :]                    # (n_lat, n_lon)


# ============================================================================
# Edge validation
# ============================================================================

def test_invalid_edges_raise():
    good_lat, good_lon = _regular_grid_edges(-90, 90, 4, 0, 360, 8)
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        compute_overlap_weights(good_lat[::-1], good_lon, good_lat, good_lon)
    with pytest.raises(ValueError, match="must have at least 2 entries"):
        compute_overlap_weights(np.array([0.0]), good_lon, good_lat, good_lon)


# ============================================================================
# 1. Constant field preservation
# ============================================================================

@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ((36, 72), (18, 36)),    # 5° -> 10°  (downsample by 2)
        ((720, 1440), (180, 360)),  # 0.25° -> 1° (downsample by 4)
        ((180, 360), (180, 360)),   # identity-grid (1° -> 1°)
        ((180, 360), (90, 180)),    # 1° -> 2°
    ],
)
def test_constant_field_is_preserved(src_shape, dst_shape):
    """Conservation gate: a uniform source field maps to the same
    uniform field on the target."""
    n_src_lat, n_src_lon = src_shape
    n_dst_lat, n_dst_lon = dst_shape
    src_lat_e, src_lon_e = _regular_grid_edges(
        -90, 90, n_src_lat, 0, 360, n_src_lon,
    )
    dst_lat_e, dst_lon_e = _regular_grid_edges(
        -90, 90, n_dst_lat, 0, 360, n_dst_lon,
    )
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)
    src_field = jnp.full(src_shape, 7.5, dtype=jnp.float64)
    out = apply_conservative_regrid(src_field, weights)
    assert out.shape == dst_shape
    np.testing.assert_allclose(np.asarray(out), 7.5, atol=1e-13)


# ============================================================================
# 2. Linear-in-sin(lat) field preservation
# ============================================================================

def test_linear_in_sin_lat_field_preserved():
    """A field f(lat) = sin(lat) is preserved cell-mean-wise after
    conservative regrid because the lat-band area factor is also
    sin-based — so the cell mean equals
    ``(sin(lat_top) + sin(lat_bot))/2`` on both grids."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 1)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 1)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)

    # Cell-mean of sin(lat) over a cell: (sin(top) - sin(bot)) / (top - bot)
    # but our regrid weights are area-weighted, so the conserved quantity
    # is (sin(top)^2 - sin(bot)^2)/2 / (sin(top) - sin(bot))
    #  = (sin(top) + sin(bot))/2.
    # We assign each src cell its area-mean of sin(lat) and check the
    # regridded value equals the area-mean on each dst cell.
    src_area_mean = (np.sin(src_lat_e[1:]) + np.sin(src_lat_e[:-1])) / 2.0
    src_field = jnp.asarray(src_area_mean[:, None], dtype=jnp.float64)
    out = apply_conservative_regrid(src_field, weights)

    dst_area_mean = (np.sin(dst_lat_e[1:]) + np.sin(dst_lat_e[:-1])) / 2.0
    np.testing.assert_allclose(
        np.asarray(out)[:, 0], dst_area_mean, atol=1e-13,
    )


# ============================================================================
# 3. Per-cell weight sum equals 1
# ============================================================================

def test_weight_sum_per_target_cell_is_one():
    """Σ_src w(src, dst) = 1 for every target cell that is fully
    covered by the source grid."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 72)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 36)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)
    # Sum weights into target cells.
    sum_per_dst = jax.ops.segment_sum(
        weights.weights, weights.dst_idx_flat,
        num_segments=weights.n_dst_cells,
    )
    np.testing.assert_allclose(np.asarray(sum_per_dst), 1.0, atol=1e-13)


# ============================================================================
# 4. Global integral conservation
# ============================================================================

def test_global_integral_conservation():
    """Σ field_dst * area_dst = Σ field_src * area_src to ~1e-13 for
    any source field, when the source covers the target completely."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 72)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 36)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)

    rng = np.random.default_rng(42)
    src_field = jnp.asarray(rng.normal(size=(36, 72)), dtype=jnp.float64)
    out = apply_conservative_regrid(src_field, weights)

    src_area = _spherical_areas(src_lat_e, src_lon_e)
    dst_area = _spherical_areas(dst_lat_e, dst_lon_e)
    src_total = float(np.sum(np.asarray(src_field) * src_area))
    dst_total = float(np.sum(np.asarray(out) * dst_area))
    rel_err = abs(dst_total - src_total) / abs(src_total)
    assert rel_err < 1e-12, (
        f"Global integral not conserved: src={src_total}, "
        f"dst={dst_total}, rel_err={rel_err:.3e}"
    )


def test_global_integral_conservation_realistic_omip_grids():
    """The OMIP-relevant case: 0.5° JRA55-do-like → 1° model grid."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 360, 0, 360, 720)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 180, 0, 360, 360)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)

    rng = np.random.default_rng(0)
    # Realistic precip-like field: positive, spatially varying.
    src_field = jnp.asarray(
        rng.uniform(0.0, 1e-4, size=(360, 720)), dtype=jnp.float64,
    )
    out = apply_conservative_regrid(src_field, weights)

    src_area = _spherical_areas(src_lat_e, src_lon_e)
    dst_area = _spherical_areas(dst_lat_e, dst_lon_e)
    src_total = float(np.sum(np.asarray(src_field) * src_area))
    dst_total = float(np.sum(np.asarray(out) * dst_area))
    rel_err = abs(dst_total - src_total) / abs(src_total)
    assert rel_err < 1e-12


# ============================================================================
# Identity grid: src == dst should give exact identity
# ============================================================================

def test_identity_grid_preserves_field_exactly():
    """When src and dst are the same grid, regrid is identity."""
    lat_e, lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 72)
    weights = compute_overlap_weights(lat_e, lon_e, lat_e, lon_e)
    rng = np.random.default_rng(123)
    src_field = jnp.asarray(rng.normal(size=(36, 72)), dtype=jnp.float64)
    out = apply_conservative_regrid(src_field, weights)
    np.testing.assert_allclose(np.asarray(out), np.asarray(src_field), atol=1e-13)


# ============================================================================
# JIT and vmap compatibility
# ============================================================================

def test_apply_is_jit_compatible():
    """apply_conservative_regrid must JIT-compile cleanly."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 72)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 36)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)
    src_field = jnp.full((36, 72), 3.14, dtype=jnp.float64)

    jit_apply = jax.jit(lambda f: apply_conservative_regrid(f, weights))
    out = jit_apply(src_field)
    np.testing.assert_allclose(np.asarray(out), 3.14, atol=1e-13)


def test_apply_handles_leading_time_axis():
    """Leading axes (time) must broadcast via vmap path."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 72)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 36)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)

    n_time = 5
    rng = np.random.default_rng(7)
    src_field = jnp.asarray(
        rng.normal(size=(n_time, 36, 72)), dtype=jnp.float64,
    )
    out = apply_conservative_regrid(src_field, weights)
    assert out.shape == (n_time, 18, 36)

    # Spot check: each time slice regrids to the same as a per-slice call.
    for t in range(n_time):
        per_slice = apply_conservative_regrid(src_field[t], weights)
        np.testing.assert_allclose(
            np.asarray(out[t]), np.asarray(per_slice), atol=1e-14,
        )


def test_apply_rejects_shape_mismatch():
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 36, 0, 360, 72)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 36)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)
    bad_field = jnp.zeros((20, 40))
    with pytest.raises(ValueError, match="does not match"):
        apply_conservative_regrid(bad_field, weights)


# ============================================================================
# AD compatibility
# ============================================================================

def test_apply_is_differentiable():
    """jax.grad through the regrid must produce finite gradients —
    the OMIP forcing pipeline is non-AD but this guards future
    differentiable-forcing use cases."""
    src_lat_e, src_lon_e = _regular_grid_edges(-90, 90, 18, 0, 360, 36)
    dst_lat_e, dst_lon_e = _regular_grid_edges(-90, 90, 9, 0, 360, 18)
    weights = compute_overlap_weights(src_lat_e, src_lon_e,
                                      dst_lat_e, dst_lon_e)

    def loss(field):
        out = apply_conservative_regrid(field, weights)
        return jnp.sum(out ** 2)

    field = jnp.ones((18, 36), dtype=jnp.float64)
    grads = jax.grad(loss)(field)
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert not bool(jnp.allclose(grads, 0.0))


def _row_sums(w):
    return np.bincount(np.asarray(w.dst_idx_flat),
                       weights=np.asarray(w.weights),
                       minlength=w.n_dst_cells)


def test_partial_lon_coverage_raises_when_required():
    # Source covers only the WESTERN half [0, 180deg]; destination spans the full
    # circle -> its eastern cells are UNCOVERED (sum(weights) 0 or < 1).
    # LONGITUDE is required complete, so relaxing the latitude reference must NOT
    # let this through.
    src_lat = np.deg2rad(np.linspace(-90, 90, 5))
    src_lon = np.deg2rad(np.linspace(0, 180, 5))   # half circle only
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))   # full circle
    # Default (require_attainable_coverage=False): silently reduced, documents
    # the hazard.
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    assert _row_sums(w).min() < 1.0 - 1e-6  # at least one dst cell under-covered
    # Required coverage: must RAISE loudly instead of emitting a reduced field.
    with pytest.raises(ValueError, match="does not fully cover"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_attainable_coverage=True)


def test_full_coverage_passes_when_required():
    # Source FINER than destination, both spanning the full sphere -> every dst
    # cell fully covered; the check must NOT raise and weights must sum to 1.
    src_lat = np.deg2rad(np.linspace(-90, 90, 9))
    src_lon = np.deg2rad(np.linspace(0, 360, 17))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_attainable_coverage=True)
    np.testing.assert_allclose(_row_sums(w), 1.0, atol=1e-12)


def test_default_floor_is_strict_and_reproduces_the_hard_unit_check():
    """``lat_shortfall_floor`` defaults to 1.0, so a caller that does not opt in
    (notably coupler/grid_remap.py, model->model) keeps the original behaviour:
    ANY latitude shortfall raises. Pins the regression where relaxing latitude
    unconditionally let a non-global Mercator source return coverage x field --
    a constant 290 K SST arriving as ~99 K -- instead of raising."""
    src_lon = np.deg2rad(np.linspace(0, 360, 17))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    # Source band +-80 deg into a global destination: outermost rows are partial.
    src_lat = np.deg2rad(np.linspace(-80, 80, 9))
    with pytest.raises(ValueError, match="is required"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_attainable_coverage=True)
    # Opting in to a floor the row clears lets exactly that case through.
    compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                            require_attainable_coverage=True,
                            lat_shortfall_floor=0.5)


def test_polar_taper_within_floor_passes_and_is_reduced_not_zero():
    """With the floor opted into, the physical polar taper is accepted and the
    outermost rows come back REDUCED (never fabricated zeros), while every
    interior row is exactly full."""
    src_lat = np.deg2rad(np.linspace(-80, 80, 9))
    src_lon = np.deg2rad(np.linspace(0, 360, 17))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_attainable_coverage=True,
                                lat_shortfall_floor=0.5)
    row = _row_sums(w).reshape(4, 8)
    np.testing.assert_allclose(row[1:-1, :], 1.0, atol=1e-12)
    expected_polar = ((np.sin(np.deg2rad(-45.0)) - np.sin(np.deg2rad(-80.0)))
                      / (np.sin(np.deg2rad(-45.0)) - np.sin(np.deg2rad(-90.0))))
    assert 0.5 < expected_polar < 0.99          # partial, and clears the floor
    np.testing.assert_allclose(row[[0, -1], :], expected_polar, atol=1e-12)


def test_row_below_the_floor_raises_instead_of_shipping_a_diluted_field():
    """A floor is a physical statement, not an epsilon: a row retaining only a few
    per-mille of real data would return a few K of air temperature. The old
    `> 1e-6` rule accepted exactly that (measured 2.4e-3 coverage for CORE-II at
    350 rows -> 0.61 K from 250 K); the floor must reject it."""
    # Source band +-50 deg: the outermost dst rows ([-90,-45], [45,90]) keep only
    # (sin 50 - sin 45)/(1 - sin 45) = 0.201 of their area -- small but NOT zero,
    # so this exercises the floor itself rather than the zero-coverage case.
    src_lat = np.deg2rad(np.linspace(-50, 50, 9))
    src_lon = np.deg2rad(np.linspace(0, 360, 17))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    with pytest.raises(ValueError, match="is required"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_attainable_coverage=True,
                                lat_shortfall_floor=0.5)
    # Anti-vacuity: the smallest ACCEPTED floor still lets this through, which is
    # why the floor has to be chosen as a policy rather than left near zero.
    compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                            require_attainable_coverage=True,
                            lat_shortfall_floor=1e-3)


def test_row_entirely_outside_source_raises_at_every_valid_floor():
    """A row the source cannot touch has coverage 0, so no floor in (0, 1] accepts
    it -- the deviation check alone would match 0 == 0 and emit ZEROS. A
    destination finer than the source's polar gap therefore raises; fixing that
    needs a renormalising/pole-filling remap, not a looser floor."""
    src_lat = np.deg2rad(np.linspace(-89.0, 89.0, 45))
    src_lon = np.deg2rad(np.linspace(0, 360, 33))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 181))   # rows [-90,-89], [89,90]
    dst_lon = np.deg2rad(np.linspace(0, 360, 33))
    # Anti-vacuity: with the check off those rows really are exact zeros.
    row_off = _row_sums(
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    ).reshape(180, 32)
    np.testing.assert_array_equal(row_off[0, :], 0.0)
    np.testing.assert_array_equal(row_off[-1, :], 0.0)
    for floor in (1.0, 0.5, 1e-3):     # 1e-3 is the smallest floor accepted
        with pytest.raises(ValueError, match="is required"):
            compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                    require_attainable_coverage=True,
                                    lat_shortfall_floor=floor)


def test_floor_outside_valid_range_is_rejected():
    """Dispatch hardening, and a pin on _MIN_LAT_FLOOR itself.

    The floor must stay strictly above _COVERAGE_TOL: the guard fires on
    ``lat_frac < floor - tol``, so a floor at or below the tolerance drives that
    threshold to <= 0 and an ALL-ZERO row passes -- the guard silently inverts.
    Rejecting only 0 / negatives / >1 would leave that unpinned, so 1e-6 and 1e-9
    are checked explicitly and the invariant is asserted directly.
    """
    assert _MIN_LAT_FLOOR > _COVERAGE_TOL
    lat = np.deg2rad(np.linspace(-90, 90, 5))
    lon = np.deg2rad(np.linspace(0, 360, 9))
    for bad in (0.0, -0.5, 1.5, _COVERAGE_TOL, 1e-9,
                _MIN_LAT_FLOOR - 1e-9):
        with pytest.raises(ValueError, match="lat_shortfall_floor"):
            compute_overlap_weights(lat, lon, lat, lon,
                                    lat_shortfall_floor=bad)


def test_lon_deficit_still_raises_with_a_relaxed_lat_floor():
    """Relaxing LATITUDE must not relax longitude.

    The source band is +-50 deg, NOT global, so the outermost destination rows
    really do have their latitude requirement relaxed to the floor (coverage
    0.201) -- otherwise ``required == 1.0`` everywhere and the floor plays no part,
    leaving the named interaction untested.
    """
    src_lat = np.deg2rad(np.linspace(-50, 50, 9))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    with pytest.raises(ValueError, match="does not fully cover"):
        compute_overlap_weights(src_lat, np.deg2rad(np.linspace(0, 180, 5)),
                                dst_lat, dst_lon,
                                require_attainable_coverage=True,
                                lat_shortfall_floor=_MIN_LAT_FLOOR)


def test_attainable_lat_coverage_is_a_fraction_in_unit_interval():
    """CONVENTION guard: coverage is a non-negative area fraction <= 1."""
    dst_lat = np.deg2rad(np.linspace(-90, 90, 7))
    dst_sin = np.sin(dst_lat)
    dst_area = dst_sin[1:] - dst_sin[:-1]
    for lo, hi in [(-90.0, 90.0), (-80.0, 80.0), (-10.0, 10.0), (30.0, 60.0)]:
        src_sin = np.sin(np.deg2rad(np.array([lo, hi])))
        frac = _attainable_lat_coverage(src_sin, dst_sin, dst_area)
        assert np.all(frac >= 0.0) and np.all(frac <= 1.0 + 1e-12), (lo, hi, frac)
    # Full-sphere source attains exactly 1 on every row (bit-exact: sin(+-pi/2)).
    src_sin = np.sin(np.deg2rad(np.array([-90.0, 90.0])))
    np.testing.assert_array_equal(
        _attainable_lat_coverage(src_sin, dst_sin, dst_area),
        np.ones(dst_lat.size - 1),
    )


def test_check_axis_span_accepts_globe_and_rejects_partial():
    """Anti-vacuity self-test for the precondition helper."""
    full = np.deg2rad(np.linspace(0.0, 360.0, 17))
    check_axis_span(full, 2.0 * np.pi, name="probe")          # must not raise
    check_axis_span(np.deg2rad(np.linspace(-90.0, 90.0, 5)), np.pi, name="probe")
    half = np.deg2rad(np.linspace(0.0, 180.0, 9))
    with pytest.raises(ValueError, match="axis spans"):
        check_axis_span(half, 2.0 * np.pi, name="probe")
    # One missing 22.5 deg column out of 360 must still be caught.
    gap = np.deg2rad(np.linspace(0.0, 337.5, 16))
    with pytest.raises(ValueError, match="axis spans"):
        check_axis_span(gap, 2.0 * np.pi, name="probe")
    # Too few edges to measure a span: clear message, not a numpy reduction error.
    with pytest.raises(ValueError, match="need >= 2 edges"):
        check_axis_span(np.array([0.0]), 2.0 * np.pi, name="probe")
