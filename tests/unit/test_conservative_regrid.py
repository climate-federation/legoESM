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


def _polar_gap_grids(n_dst_lat=720, n_dst_lon=8, src_lat_max=89.5, n_src_lat=60):
    """A source with a CORE-II-like polar gap and a destination finer than it.

    ``src_lat_max=89.5`` leaves a 0.5 deg gap (CORE-II's is 0.514); 0.25 deg
    destination rows mean the two outermost rows per pole get NO source overlap at
    all, which is the regime renormalisation alone cannot fix.
    """
    src_lat = np.deg2rad(np.linspace(-src_lat_max, src_lat_max, n_src_lat + 1))
    src_lon = np.deg2rad(np.linspace(0, 360, n_dst_lon + 1))
    dst_lat = np.deg2rad(np.linspace(-90, 90, n_dst_lat + 1))
    dst_lon = np.deg2rad(np.linspace(0, 360, n_dst_lon + 1))
    return src_lat, src_lon, dst_lat, dst_lon


def test_partial_lon_coverage_raises_when_required():
    # Source covers only the WESTERN half [0, 180deg]; destination spans the full
    # circle -> its eastern cells are UNCOVERED. A longitude deficit is never
    # "treated": neither fracarea nor polar_fill is allowed to paper over it.
    src_lat = np.deg2rad(np.linspace(-90, 90, 5))
    src_lon = np.deg2rad(np.linspace(0, 180, 5))   # half circle only
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))   # full circle
    # Default (require_full_coverage=False): silently reduced, documents the hazard.
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    assert _row_sums(w).min() < 1.0 - 1e-6
    with pytest.raises(ValueError, match="does not fully cover"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True)
    # polar_fill only ever touches rows beyond the source's LATITUDE band, so it
    # cannot rescue a longitude gap either.
    with pytest.raises(ValueError, match="does not fully cover"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True,
                                normalization="fracarea", polar_fill=True)


def test_fracarea_does_not_repair_a_partial_longitude_column():
    """REGRESSION: renormalising by the ROW SUM would silently fix a longitude
    seam/ghost deficit as well, destroying the guard this module exists for.

    A row sum is ``lat_frac[j] * lon_frac[i]``, so ``1/row_sum`` cannot tell a
    polar latitude gap from a seam gap and normalises BOTH to 1. Measured with
    that (wrong) scaling, a seam column at ``lon_frac = 0.625`` -- the historical
    single-ghost bug that left a forcing column HALVED and NaN'd the 10-m pressure
    iteration -- came back at 1.000 and PASSED the strict check. Scaling by
    ``lat_frac`` alone keeps the deficit visible.

    Unlike the fully-uncovered case above, this column is PARTIALLY covered, which
    is precisely the regime renormalisation acts on.
    """
    # Source lon tiles [0, 337.5]; destination has 8 cells of 45 deg, so its LAST
    # column [315, 360] is covered only over [315, 337.5] -> lon_frac = 0.5, while
    # every other column is complete. PARTIAL, not empty: ending the source at 315
    # would leave that column at exactly 0 and retest the uncovered case instead.
    # Latitude is global, so lat_frac == 1 throughout and any shortfall is longitude.
    src_lat = np.deg2rad(np.linspace(-90, 90, 9))
    src_lon = np.deg2rad(np.linspace(0.0, 337.5, 9))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    w_raw = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    row = _row_sums(w_raw).reshape(4, 8)
    # Anti-vacuity: the deficit is PARTIAL (strictly between 0 and 1), which is the
    # regime renormalisation acts on -- an empty column would prove nothing here.
    np.testing.assert_allclose(row[:, -1], 0.5, atol=1e-12)
    np.testing.assert_allclose(row[:, :-1], 1.0, atol=1e-12)
    for mode in ("dstarea", "fracarea"):
        with pytest.raises(ValueError, match="does not fully cover"):
            compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                    require_full_coverage=True,
                                    normalization=mode, polar_fill=True)


def test_full_coverage_passes_when_required():
    # Source FINER than destination, both spanning the full sphere -> every dst
    # cell fully covered; the check must NOT raise and weights must sum to 1.
    src_lat = np.deg2rad(np.linspace(-90, 90, 9))
    src_lon = np.deg2rad(np.linspace(0, 360, 17))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True)
    np.testing.assert_allclose(_row_sums(w), 1.0, atol=1e-12)


def test_unknown_normalization_raises():
    """Dispatch hardening: a typo must not silently select 'dstarea'."""
    lat = np.deg2rad(np.linspace(-90, 90, 5))
    lon = np.deg2rad(np.linspace(0, 360, 9))
    for bad in ("fracarea ", "FRACAREA", "frac_area", "dst_area", "none", ""):
        with pytest.raises(ValueError, match="Unknown normalization"):
            compute_overlap_weights(lat, lon, lat, lon, normalization=bad)
    # Anti-vacuity: both real modes are accepted.
    for good in ("dstarea", "fracarea"):
        compute_overlap_weights(lat, lon, lat, lon, normalization=good)


# A REALISTIC polar gap for the partial-row pair below: the source stops 0.5 deg
# short of each pole (CORE-II's real gap is 0.514, JRA55-do's 0.151), which is
# inside the 2 deg extrapolation budget. Onto 1 deg destination rows the polar row
# is 0.75 covered -- partial and visibly so, with no row left entirely empty, so
# these two isolate `fracarea` without `polar_fill`.
_GAP_SRC_LAT = np.deg2rad(np.linspace(-89.5, 89.5, 9))
_GAP_SRC_LON = np.deg2rad(np.linspace(0, 360, 17))
_GAP_DST_LAT = np.deg2rad(np.linspace(-90, 90, 181))
_GAP_DST_LON = np.deg2rad(np.linspace(0, 360, 9))
_GAP_POLAR_FRAC = 0.749995        # (sin -89 - sin -89.5) / (sin -89 - sin -90)


def test_dstarea_default_reduces_a_partly_covered_polar_row():
    """The hazard the treatment exists to fix, pinned as the DEFAULT behaviour:
    'dstarea' divides by the FULL destination cell area, so a partly covered polar
    row returns coverage x field -- 0.75 x the constant here, not the constant."""
    w = compute_overlap_weights(_GAP_SRC_LAT, _GAP_SRC_LON,
                                _GAP_DST_LAT, _GAP_DST_LON)
    out = np.asarray(apply_conservative_regrid(
        jnp.full((8, 16), 290.0, dtype=jnp.float64), w))
    np.testing.assert_allclose(out[1:-1, :], 290.0, atol=1e-10)
    np.testing.assert_allclose(out[[0, -1], :], 290.0 * _GAP_POLAR_FRAC,
                               rtol=1e-5)
    assert out[0, 0] < 220.0          # visibly wrong for an intensive field


def test_fracarea_preserves_a_constant_through_a_partly_covered_polar_row():
    """The fix for a PARTLY covered row: dividing by the covered LATITUDE fraction
    returns the area-weighted mean of the overlapping source, so the constant
    survives. No row is empty here, so `polar_fill` plays no part -- this isolates
    the renormalisation."""
    w = compute_overlap_weights(_GAP_SRC_LAT, _GAP_SRC_LON,
                                _GAP_DST_LAT, _GAP_DST_LON,
                                require_full_coverage=True,
                                normalization="fracarea")
    np.testing.assert_allclose(_row_sums(w), 1.0, atol=1e-12)
    out = np.asarray(apply_conservative_regrid(
        jnp.full((8, 16), 290.0, dtype=jnp.float64), w))
    np.testing.assert_allclose(out, 290.0, atol=1e-10)


def test_fracarea_is_a_noop_when_every_cell_is_fully_covered():
    """Renormalising must not perturb a fully covered remap: scale is exactly 1
    there, so the weights are bit-identical to 'dstarea'."""
    src_lat = np.deg2rad(np.linspace(-90, 90, 19))
    src_lon = np.deg2rad(np.linspace(0, 360, 37))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 7))
    dst_lon = np.deg2rad(np.linspace(0, 360, 13))
    a = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                normalization="dstarea")
    b = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                normalization="fracarea")
    np.testing.assert_array_equal(np.asarray(a.weights), np.asarray(b.weights))
    np.testing.assert_array_equal(np.asarray(a.dst_idx_flat),
                                  np.asarray(b.dst_idx_flat))


def test_fracarea_alone_cannot_fix_an_entirely_uncovered_row():
    """Renormalisation rescales; it cannot create. A row with NO overlap sums to
    exactly 0, there is nothing to scale, and the strict check must still raise --
    pointing at polar_fill."""
    src_lat, src_lon, dst_lat, dst_lon = _polar_gap_grids()
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                normalization="fracarea")
    row = _row_sums(w).reshape(720, 8)
    np.testing.assert_array_equal(row[:2, :], 0.0)     # still empty
    np.testing.assert_allclose(row[2:-2, :], 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="polar_fill=True"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True,
                                normalization="fracarea")


def test_polar_fill_gives_uncovered_rows_the_outermost_source_row_zonally():
    """polar_fill must reproduce the source's outermost row COLUMN BY COLUMN, not
    its zonal mean -- otherwise it would smear away the polar lon structure."""
    n_lon = 8
    src_lat, src_lon, dst_lat, dst_lon = _polar_gap_grids(n_dst_lon=n_lon)
    n_src_lat = src_lat.size - 1
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True,
                                normalization="fracarea", polar_fill=True)
    # The field MUST vary in latitude as well as longitude. With a lat-constant
    # field, swapping the two hemispheres (south <- src[-1], north <- src[0]) is
    # bit-for-bit invisible -- an orientation bug the whole suite would miss.
    profile = np.arange(n_lon, dtype=np.float64) * 3.0 + 250.0
    lat_ramp = 10.0 * np.arange(n_src_lat, dtype=np.float64)[:, None]
    src_field = profile[None, :] + lat_ramp
    out = np.asarray(apply_conservative_regrid(jnp.asarray(src_field), w))
    # South-polar rows take the source's SOUTHERNMOST row, north the NORTHERNMOST.
    for j in (0, 1):
        np.testing.assert_allclose(out[j, :], src_field[0, :], atol=1e-10)
    for j in (-1, -2):
        np.testing.assert_allclose(out[j, :], src_field[-1, :], atol=1e-10)
    # Anti-vacuity: the two ends differ by the full ramp, so a hemisphere swap
    # would move each filled row by 590 K.
    assert abs(src_field[-1, 0] - src_field[0, 0]) > 500.0
    assert out[0, :].std() > 1.0          # zonal structure kept, not a zonal mean


def test_fracarea_plus_polar_fill_preserves_a_constant_at_every_row():
    """Headline: a source with a polar gap onto a destination FINER than that gap
    now reproduces a constant field exactly on every row, and the STRICT coverage
    check passes. Previously this configuration either raised or emitted zeros."""
    src_lat, src_lon, dst_lat, dst_lon = _polar_gap_grids()
    n_src_lat = src_lat.size - 1
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True,
                                normalization="fracarea", polar_fill=True)
    np.testing.assert_allclose(_row_sums(w), 1.0, atol=1e-12)
    out = np.asarray(apply_conservative_regrid(
        jnp.full((n_src_lat, 8), 250.0, dtype=jnp.float64), w))
    assert out.shape == (720, 8)
    np.testing.assert_allclose(out, 250.0, atol=1e-10)
    # Anti-vacuity: with the treatment off, those rows are exactly zero.
    w_off = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    out_off = np.asarray(apply_conservative_regrid(
        jnp.full((n_src_lat, 8), 250.0, dtype=jnp.float64), w_off))
    np.testing.assert_array_equal(out_off[:2, :], 0.0)


def test_treatment_trades_strict_conservation_for_correct_magnitude():
    """The trade-off, quantified and pinned rather than left implicit.

    'dstarea' conserves the global integral of a constant field exactly (the
    uncovered polar caps contribute zero on both sides). fracarea + polar_fill fill
    those caps with real values, so the destination integral EXCEEDS the source's by
    the cap area -- that is the price of correct magnitude, and it is why
    coupler/grid_remap.py (the conservative flux direction) keeps 'dstarea'.
    """
    src_lat, src_lon, dst_lat, dst_lon = _polar_gap_grids()
    n_src_lat = src_lat.size - 1
    field = jnp.full((n_src_lat, 8), 250.0, dtype=jnp.float64)
    src_area = _spherical_areas(src_lat, src_lon)
    dst_area = _spherical_areas(dst_lat, dst_lon)
    src_total = float(np.sum(np.asarray(field) * src_area))

    w_cons = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    out_cons = np.asarray(apply_conservative_regrid(field, w_cons))
    assert abs(np.sum(out_cons * dst_area) - src_total) / src_total < 1e-12

    w_treat = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                      normalization="fracarea", polar_fill=True)
    out_treat = np.asarray(apply_conservative_regrid(field, w_treat))
    excess = np.sum(out_treat * dst_area) / src_total - 1.0
    # The treated destination carries the constant over the WHOLE sphere while the
    # source only ever covered the band, so the ratio is sphere/band:
    #   dst = C * 4pi,  src = C * 4pi * sin(89.5deg)  ->  excess = 1/sin - 1.
    # (Not 1 - sin: that would normalise by the sphere, not by the source.)
    # This closed form holds for THIS geometry only -- a band symmetric about the
    # equator carrying a CONSTANT field. For a band [a, b] it is
    # 2/(sin b - sin a) - 1, and for a non-constant field there is no closed form,
    # only the sign: the treated integral always exceeds the source's.
    band_frac = np.sin(np.deg2rad(89.5))
    assert excess > 0.0
    np.testing.assert_allclose(excess, 1.0 / band_frac - 1.0, rtol=1e-9)


def test_polar_fill_weights_stay_differentiable():
    """Weights are compile-time constants and the apply stays linear, so gradients
    must flow through a filled remap exactly as through an untreated one."""
    # 720 dst rows (0.25 deg) so rows really do fall BEYOND the source band --
    # at 180 rows (1 deg) the polar row merely clips it, polar_fill never engages,
    # and the comparison below would be vacuously equal.
    src_lat, src_lon, dst_lat, dst_lon = _polar_gap_grids()
    n_src_lat = src_lat.size - 1
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                normalization="fracarea", polar_fill=True)

    def loss(f):
        return jnp.sum(apply_conservative_regrid(f, w) ** 2)

    g = jax.grad(loss)(jnp.full((n_src_lat, 8), 250.0, dtype=jnp.float64))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert not bool(jnp.allclose(g, 0.0))
    # The source's EDGE rows feed the filled destination rows, so they must carry
    # strictly more sensitivity than they would without the fill.
    w_nofill = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                       normalization="fracarea")

    def loss_nofill(f):
        return jnp.sum(apply_conservative_regrid(f, w_nofill) ** 2)

    g_nofill = jax.grad(loss_nofill)(
        jnp.full((n_src_lat, 8), 250.0, dtype=jnp.float64))
    assert float(jnp.abs(g[0]).sum()) > float(jnp.abs(g_nofill[0]).sum())


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


def test_polar_fill_refuses_to_extrapolate_beyond_its_gap_budget():
    """polar_fill must fill a GAP, not spread a regional source over the globe.

    Without a budget the same code path happily extrapolates a +-10 deg band across
    all 180 rows of a 1 deg destination and passes the strict check. That is not
    hypothetical in this pipeline: a latitude axis silently read in the wrong units
    turns a global grid into a +-1.57 deg "band", which must stay LOUD.
    """
    src_lon = np.deg2rad(np.linspace(0, 360, 9))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 181))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    for band in (10.0, 50.0, 80.0, 1.57):
        with pytest.raises(ValueError, match="beyond the .* budget"):
            compute_overlap_weights(
                np.deg2rad(np.linspace(-band, band, 9)), src_lon,
                dst_lat, dst_lon,
                require_full_coverage=True,
                normalization="fracarea", polar_fill=True,
            )
    # A REAL polar gap (CORE-II's is 0.514 deg, JRA55-do's 0.151) is well inside.
    for band in (89.486, 89.849, 88.5):
        compute_overlap_weights(
            np.deg2rad(np.linspace(-band, band, 9)), src_lon, dst_lat, dst_lon,
            require_full_coverage=True,
            normalization="fracarea", polar_fill=True,
        )
    # Widening the budget deliberately is allowed -- the guard is a default, not a
    # prohibition.
    compute_overlap_weights(
        np.deg2rad(np.linspace(-80.0, 80.0, 9)), src_lon, dst_lat, dst_lon,
        require_full_coverage=True, normalization="fracarea", polar_fill=True,
        max_polar_gap_deg=15.0,
    )
