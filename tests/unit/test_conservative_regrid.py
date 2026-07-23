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


def test_partial_coverage_raises_when_required():
    # Source covers only the WESTERN half [0, 180deg]; destination spans the full
    # circle -> its eastern cells are UNCOVERED (sum(weights) 0 or < 1).
    src_lat = np.deg2rad(np.linspace(-90, 90, 5))
    src_lon = np.deg2rad(np.linspace(0, 180, 5))   # half circle only
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))   # full circle
    # Default (require_full_coverage=False): silently reduced, documents hazard.
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon)
    row = np.bincount(np.asarray(w.dst_idx_flat),
                      weights=np.asarray(w.weights),
                      minlength=w.n_dst_cells)
    assert row.min() < 1.0 - 1e-6          # at least one dst cell under-covered
    # Required coverage: must RAISE loudly instead of emitting a reduced field.
    with pytest.raises(ValueError, match="does not fully cover"):
        compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True)


def test_full_coverage_passes_when_required():
    # Source FINER than destination, both spanning the full sphere -> every dst
    # cell fully covered; require_full_coverage must NOT raise and weights sum 1.
    src_lat = np.deg2rad(np.linspace(-90, 90, 9))
    src_lon = np.deg2rad(np.linspace(0, 360, 17))
    dst_lat = np.deg2rad(np.linspace(-90, 90, 5))
    dst_lon = np.deg2rad(np.linspace(0, 360, 9))
    w = compute_overlap_weights(src_lat, src_lon, dst_lat, dst_lon,
                                require_full_coverage=True)
    row = np.bincount(np.asarray(w.dst_idx_flat),
                      weights=np.asarray(w.weights),
                      minlength=w.n_dst_cells)
    np.testing.assert_allclose(row, 1.0, atol=1e-12)
