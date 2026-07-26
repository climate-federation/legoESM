"""Conservative (area-weighted) regridding between regular lat-lon grids.

Used by the OMIP forcing pipeline to remap source data (e.g. JRA55-do
at TL319 / 0.5°) onto the model 1° lat-lon grid while preserving
global integrals — required for fluxes such as precipitation, runoff,
and wind stress, where bilinear or KD-tree inverse-distance
interpolation would silently violate conservation.

The existing ``grids/regridding.py`` provides KD-tree / bilinear /
gnomonic-bilinear paths between cubed-sphere and Gaussian grids — none
of those are flux-conservative.  This module fills that gap for the
restricted-but-common case of regular lat-lon → regular lat-lon.

Algorithm
---------
For each target cell with corners ``(lat_lo, lat_hi) x (lon_lo, lon_hi)``
the routine finds all source cells whose corner box overlaps it and
assigns weight equal to the **spherical-surface overlap area** divided
by the target cell area.  Spherical-surface area on a regular lat-lon
grid factorises:

    A(cell) = (sin(lat_hi) − sin(lat_lo)) * (lon_hi − lon_lo)

so the overlap of two cells is

    overlap = max(0, sin(min(lat_hi_s, lat_hi_d))
                     − sin(max(lat_lo_s, lat_lo_d)))
              * max(0, min(lon_hi_s, lon_hi_d)
                       − max(lon_lo_s, lon_lo_d))

and weight ``w = overlap / A(target)``.

Conservation
------------
For each target cell that is *fully covered* by the source grid, the
sum of weights equals 1 to machine precision.  This is verified by
the unit tests.  Global mass / energy / momentum integrals are
conserved because

    ∫_target field_target dA = Σ_target field_target * A_target
                             = Σ_target Σ_src w * A_target * field_src
                             = Σ_src field_src * (Σ_target w * A_target)

and the inner sum equals A_src for any source cell fully within the
union of target cells.

This holds for the default ``normalization='dstarea'`` with ``polar_fill=False``.
Both polar treatments deliberately BREAK it in exchange for correct magnitude on a
partly covered row — see ``compute_overlap_weights``' parameter docs and
``docs/dev-notes/regrid_polar_coverage_2026-07-24.md``.

Longitude convention
--------------------
Both grids must use the same longitude convention (e.g. both
``[0, 360)`` or both ``[-180, 180)``).  No automatic wrap-around — the
caller is responsible for normalising before calling
``compute_overlap_weights``.  The grids are assumed monotonically
increasing in longitude.  The whole-globe periodic wrap (i.e. last
source longitude edge meeting first source longitude edge) is **not**
treated specially because for the OMIP use case the grids are global
and aligned.

Use
---
    # Once at startup:
    weights = compute_overlap_weights(
        src_lat_edges, src_lon_edges,
        dst_lat_edges, dst_lon_edges,
    )
    # Per step (JIT-clean):
    field_target = apply_conservative_regrid(field_source, weights)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


class ConservativeRegridWeights(NamedTuple):
    """Sparse pre-computed conservative regridding weights.

    Attributes
    ----------
    src_idx_flat : jax.Array (int32, shape (N_overlaps,))
        Flat row-major index into the source ``(n_src_lat, n_src_lon)``
        grid for each non-zero overlap pair.
    dst_idx_flat : jax.Array (int32, shape (N_overlaps,))
        Flat row-major index into the target ``(n_dst_lat, n_dst_lon)``
        grid for each non-zero overlap pair.
    weights : jax.Array (float64, shape (N_overlaps,))
        Overlap area divided by target cell area.  Sum of weights
        per target cell is 1 (to ~1e-14) when the source grid fully
        covers that target cell.
    src_shape : tuple[int, int]
        ``(n_src_lat, n_src_lon)``.
    dst_shape : tuple[int, int]
        ``(n_dst_lat, n_dst_lon)``.
    n_dst_cells : int
        ``n_dst_lat * n_dst_lon`` — for ``segment_sum``.
    """
    src_idx_flat: jnp.ndarray
    dst_idx_flat: jnp.ndarray
    weights: jnp.ndarray
    src_shape: tuple
    dst_shape: tuple
    n_dst_cells: int


def _check_edges(edges: np.ndarray, name: str) -> None:
    if edges.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {edges.shape}")
    if edges.size < 2:
        raise ValueError(f"{name} must have at least 2 entries (1 cell)")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"{name} must be strictly monotonically increasing")


def cell_edges_1d(
    centers: np.ndarray,
    periodic_lon: bool = False,
) -> np.ndarray:
    """Cell-face edges (radians) from uniformly-spaced 1-D cell centres (radians).

    Canonical helper for building the edge arrays that
    :func:`compute_overlap_weights` consumes.  Assumes uniform spacing
    (``dc = centers[1] - centers[0]``); interior edges are the midpoints and
    the two boundary edges are half-cell extrapolations.

    Parameters
    ----------
    centers : 1-D array
        Cell-centre coordinates in **radians**.
    periodic_lon : bool
        If True, force the last edge to be exactly ``first_edge + 2*pi`` so the
        longitude axis spans the full circle.  This prevents the conservative
        regrid from under-weighting the seam column when the last centre is
        slightly less than ``2*pi - dc/2`` (the radian analogue of the
        +360 deg wrap used by the JRA55-do / OMIP forcing loaders).

    Notes
    -----
    Two legacy degree-based private copies still exist
    (``forcing/jra55_do.py:_grid_edges_from_centers`` and
    ``ocean/coupler/omip2_applicator.py:_edges_from_centers_deg``); the latter
    has no periodic mode at all and instead pads the source with ghost columns,
    so they are NOT drop-in replaceable by this helper without changing their
    behaviour — unify them in a dedicated, separately-validated PR rather
    than here.  For latitude prefer a grid's pole-clamped v-face coordinates
    (e.g. ``LatLonGrid.lat_v``) over this helper, which would otherwise
    extrapolate the first/last edge past +/-pi/2.
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.size < 2:
        raise ValueError("Need >= 2 cell centres to infer edges")
    dc = float(c[1] - c[0])
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[:-1] = c - 0.5 * dc
    edges[-1] = c[-1] + 0.5 * dc
    if periodic_lon:
        edges[-1] = edges[0] + 2.0 * np.pi
    return edges


_COVERAGE_TOL = 1e-6  # |sum(weights) - 1| tolerance for a remap that is
# REQUIRED to cover every destination cell completely.
# float64 area-ratio roundoff is ~1e-14 (~4 adds/cell), so 1e-6 sits ~1e8x above
# noise yet flags a single missing 0.25deg source cell in a 1deg destination or a
# ~12.5% seam deficit.  Deliberately NOT loosened to swallow the polar deficit
# (measured 6.6e-2 for CORE-II -> 2deg/90x180; 2.6e-1 at 1deg): a tolerance wide
# enough for that would also swallow a genuinely half-covered seam cell.  Fix the
# WEIGHTS instead -- normalization='fracarea' + polar_fill=True.


# Axis-span floor tolerance (radians).  float64 edge inference is exact to ~1e-15,
# but grid coordinates are STORED at the precision policy's dtype (float32 by
# default for LatLonGrid), which alone reaches ~2.2e-7 rad of span error at
# n_lon = 720 -- only ~4.5x under a flat 1e-6.  So the effective tolerance also
# scales with the cell width (see check_axis_span): a genuine missing column is a
# whole cell wide, so 1% of a cell separates the two by ~100x at any resolution.
_SPAN_TOL = 1e-6
_SPAN_CELL_FRAC = 0.01


def check_axis_span(edges, expected_span, *, name):
    """Raise unless ``edges`` span ``expected_span`` radians end to end.

    Use this to state a caller's geometric PRECONDITION explicitly, before any
    ghost padding hides it.  ``compute_overlap_weights`` cannot recover the
    precondition itself: a caller that wrap-pads its source longitude with
    +/-2pi shifts of its own end columns turns a partial-longitude source into
    one enormous ghost cell spanning the whole missing sector, which then reports
    complete longitude coverage.  Checking the RAW axis is the only place the
    difference is still visible.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        raise ValueError(
            f"{name}: need >= 2 edges to measure a span, got {edges.size}."
        )
    span = float(edges[-1] - edges[0])
    # Scale with the narrowest cell so float32-stored coordinates cannot trip the
    # check, while a whole missing cell still does (~100x margin).
    atol = max(_SPAN_TOL, _SPAN_CELL_FRAC * float(np.min(np.diff(edges))))
    if abs(span - float(expected_span)) > atol:
        raise ValueError(
            f"{name}: axis spans {span:.12g} rad ({np.degrees(span):.6g} deg) but "
            f"{float(expected_span):.12g} rad ({np.degrees(expected_span):.6g} deg) "
            f"is required (off by {span - float(expected_span):.3e} rad, "
            f"tol {atol:.3e})."
        )


#: Accepted ``normalization`` modes, named after the ESMF convention.
_NORMALIZATIONS = frozenset({"dstarea", "fracarea"})

# How far poleward `polar_fill` will extrapolate, in degrees.  Real polar DATA GAPS
# are small -- CORE-II NYF 0.514 deg, JRA55-do 0.151 deg -- and over such a distance
# even a strong (~1 K/deg) polar air-temperature gradient contributes well under
# 1 K, far below the tens of K the fill removes.  2 deg leaves ~4x margin on the
# widest real gap while keeping a REGIONAL source loud: without a budget, a
# +-10 deg band silently fills an entire global destination.
#
# Margin against real sources, measured through the callers' own edge inference
# (which pole-clamps, so uniform grids give a 0 deg gap): CORE-II 0.514, JRA55-do
# 0.151, ERA5 0.25 deg / 1 deg / 2 deg all 0.  Gaussian grids shrink toward the
# pole as they coarsen -- 256 rows 0.19, 128 rows 0.38, 64 rows 0.75, 32 rows 1.49
# (1.3x margin) -- and a 16-row Gaussian source (2.94) WOULD be rejected.  That is
# the intended behaviour at that coarseness, but a legitimate very coarse Gaussian
# source needs an explicit `max_polar_gap_deg`.
_MAX_POLAR_GAP_DEG = 2.0


def _row_sums(dst_idx, weights, n_dst_cells):
    """Summed weights per destination cell, length exactly ``n_dst_cells``."""
    return np.bincount(
        np.asarray(dst_idx).ravel(),
        weights=np.asarray(weights).ravel(),
        minlength=int(n_dst_cells),
    )


def _attainable_lat_coverage(src_sin, dst_sin, dst_lat_area):
    """Fraction of each destination lat band the source can supply, in [0, 1].

    CONVENTION: a non-negative AREA fraction in the ``sin(lat)`` coordinate where a
    spherical band's area is linear, both axes ascending (south -> north). Source
    cells are contiguous, so their union is the single band
    ``[src_sin[0], src_sin[-1]]``.
    """
    covered = np.clip(
        np.minimum(dst_sin[1:], src_sin[-1])
        - np.maximum(dst_sin[:-1], src_sin[0]), 0.0, None)
    frac = np.zeros_like(covered)
    ok = dst_lat_area > 0.0
    frac[ok] = covered[ok] / dst_lat_area[ok]
    return frac


def _renormalize_lat_shortfall(dst_idx, weights, lat_frac, n_dst_lon):
    """Rescale each PARTIALLY covered destination cell so its weights sum to 1.

    ESMF's ``FRACAREA`` normalisation. ``dstarea`` divides every overlap by the
    FULL destination cell area, so a cell the source only partly covers sums to
    ``f < 1`` and returns ``f * field`` -- a REDUCED value, which for an intensive
    field (air temperature, sea-level pressure, a flux DENSITY in W/m^2) is simply
    wrong: the uncovered part is a DATA GAP, not a region of zero flux. Dividing
    by the covered area instead returns the area-weighted MEAN of the source that
    does overlap, so a constant field is preserved exactly.

    TRADE-OFF, deliberate: this breaks strict global conservation. The remapped
    field's spherical integral no longer equals the source's, because the
    uncovered fraction is filled with the covered part's mean instead of zero.
    That is the right call for the forcing path (an intensive field must keep its
    magnitude) and the WRONG one for conservative flux coupling, which is why
    ``dstarea`` remains the default and ``coupler/grid_remap.py`` keeps it.

    Scaled by the LATITUDE factor only -- ``1 / lat_frac[j]``, NOT ``1 / row_sum``.
    That distinction is the whole safety property. The weights are separable
    (``row_sum[j, i] = lat_frac[j] * lon_frac[i]`` exactly), so dividing by the
    row sum would normalise a LONGITUDE deficit away too, silently repairing the
    seam/ghost gap this module's coverage check exists to catch -- the failure that
    once left a forcing column HALVED and NaN'd the 10-m pressure iteration.
    Measured with ``1 / row_sum``: a seam column at ``lon_frac = 0.625`` came back
    at 1.000 and passed the strict check. Dividing by ``lat_frac`` alone leaves any
    ``lon_frac < 1`` visible in the row sum, so it still raises.

    A row with NO latitude overlap (``lat_frac == 0``) is left untouched -- there is
    nothing to rescale, and ``polar_fill`` handles it. Rows already complete are
    skipped via ``_COVERAGE_TOL``: a full row can sum to 1 +/- an ulp, and dividing
    by that would perturb an already-correct weight in its last bit, so skipping
    makes "no-op on a fully covered remap" exactly true rather than nearly true.
    """
    scale = np.ones_like(lat_frac)
    live = (lat_frac > 0.0) & (np.abs(lat_frac - 1.0) > _COVERAGE_TOL)
    scale[live] = 1.0 / lat_frac[live]
    j_dst = np.asarray(dst_idx) // int(n_dst_lon)
    return np.asarray(weights) * scale[j_dst]


def _append_polar_fill(
    src_idx, dst_idx, weights, *,
    src_sin, dst_sin, lon_overlap, dst_lon_area,
    n_src_lat, n_src_lon, n_dst_lon,
):
    """Give destination rows beyond the source's latitude band its outermost row.

    A source that stops short of the pole leaves whole destination rows with NO
    overlap: their weights sum to exactly 0, so the remap returns 0 there --
    ``T_air = 0 K`` in forcing terms. Renormalisation cannot help (nothing to
    rescale), so those rows are instead filled from the source's own outermost
    latitude row, zonally resolved: each destination cell takes the area-weighted
    mean over longitude of ``src[0, :]`` (south) or ``src[-1, :]`` (north).

    This is a zeroth-order poleward extrapolation -- the standard treatment for a
    polar data gap, and far better than either zeros or a diluted value. It
    FABRICATES data outside the source's domain, so it is opt-in and, like
    ``fracarea``, not strictly conservative.

    CONVENTION: axes ascend south -> north, so ``src_sin[0]`` is the source's
    southern edge and ``src_sin[-1]`` its northern one. The two masks use
    non-strict inequalities against those edges, so a row that merely CLIPS the
    band (partially covered) is excluded here and handled by renormalisation --
    the two treatments cannot double-count a row.
    """
    entirely_south = dst_sin[1:] <= src_sin[0]
    entirely_north = dst_sin[:-1] >= src_sin[-1]

    i_dst, i_src = np.nonzero(lon_overlap > 0.0)
    # Weight within a filled cell: the source column's share of this destination
    # column, so the entries sum to 1 whenever the source spans the destination in
    # longitude (which check_axis_span asserts at the forcing callers).
    lon_w = lon_overlap[i_dst, i_src] / dst_lon_area[i_dst]

    add_src, add_dst, add_w = [], [], []
    for mask, j_src in ((entirely_south, 0), (entirely_north, n_src_lat - 1)):
        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            continue
        n_pair = i_dst.size
        add_dst.append(np.repeat(rows, n_pair) * n_dst_lon + np.tile(i_dst, rows.size))
        add_src.append(int(j_src) * n_src_lon + np.tile(i_src, rows.size))
        add_w.append(np.tile(lon_w, rows.size))

    if not add_dst:
        return src_idx, dst_idx, weights
    return (
        np.concatenate([np.asarray(src_idx)] + add_src),
        np.concatenate([np.asarray(dst_idx)] + add_dst),
        np.concatenate([np.asarray(weights)] + add_w),
    )


def _validate_full_coverage(
    dst_idx, weights, n_dst_lat, n_dst_lon, *, tol, name, src_sin,
):
    """Raise unless every destination cell's weights sum to 1.

    HOST-SIDE (numpy) check on the STATIC weights BEFORE they become a jnp
    constant -- zero autodiff / JIT / trace impact (the traced apply is an
    unchanged ``segment_sum`` over these compile-time weights). A cell summing to
    less than 1 returns a physically REDUCED field; one summing to 0 returns
    zeros. Fail LOUDLY rather than emit either.

    This is the strict invariant again, deliberately. An earlier iteration
    compared against per-row ATTAINABLE coverage so that a source with a polar gap
    could pass -- a workaround for weights that could not represent the polar row
    correctly. ``normalization='fracarea'`` and ``polar_fill=True`` fix the WEIGHTS
    instead, after which every cell genuinely sums to 1 and the honest invariant is
    checkable again. So a shortfall here now means one of: a longitude seam/ghost
    deficit, wrong edges, a regionally-limited source -- or a polar gap with the
    treatment left switched off.
    """
    n_dst_cells = int(n_dst_lat) * int(n_dst_lon)
    row_sum = _row_sums(dst_idx, weights, n_dst_cells)
    dev = np.abs(row_sum - 1.0)
    bad = np.nonzero(dev > tol)[0]
    if bad.size:
        worst = int(bad[int(dev[bad].argmax())])
        j, i = divmod(worst, int(n_dst_lon))
        empty = int(np.count_nonzero(row_sum <= tol))
        raise ValueError(
            f"{name}: {bad.size} destination cell(s) of {n_dst_cells} have summed "
            f"remap weights off 1 by > {tol:g} (worst |sum-1| = "
            f"{float(dev.max()):.3e} at dst lat_row {j}, lon_col {i}, sum = "
            f"{float(row_sum[worst]):.12g}; {empty} cell(s) have NO source overlap "
            "at all). The source does not fully cover the destination, so the "
            "remap would silently return a reduced or all-zero field. If the "
            "source has a polar gap (it spans "
            f"asin({float(src_sin[0]):.9g})..asin({float(src_sin[-1]):.9g}) rad), "
            "pass normalization='fracarea' for partly covered rows and "
            "polar_fill=True for rows it cannot reach; otherwise this is a "
            "longitude seam/ghost deficit, wrong edges, or a regionally-limited "
            "source."
        )


def compute_overlap_weights(
    src_lat_edges: np.ndarray,
    src_lon_edges: np.ndarray,
    dst_lat_edges: np.ndarray,
    dst_lon_edges: np.ndarray,
    *,
    require_full_coverage: bool = False,
    normalization: str = "dstarea",
    polar_fill: bool = False,
    max_polar_gap_deg: float = _MAX_POLAR_GAP_DEG,
) -> ConservativeRegridWeights:
    """Compute conservative overlap weights between two regular lat-lon grids.

    All edge arrays are 1-D, strictly monotonically increasing, in
    **radians**.  ``lat`` lies in ``[-π/2, π/2]``; ``lon`` may use
    any consistent origin (both grids must agree).

    Parameters
    ----------
    src_lat_edges : np.ndarray, shape (n_src_lat + 1,)
        Source latitude cell edges, radians, ascending.
    src_lon_edges : np.ndarray, shape (n_src_lon + 1,)
        Source longitude cell edges, radians, ascending.
    dst_lat_edges : np.ndarray, shape (n_dst_lat + 1,)
        Target latitude cell edges, radians, ascending.
    dst_lon_edges : np.ndarray, shape (n_dst_lon + 1,)
        Target longitude cell edges, radians, ascending.
    require_full_coverage : bool, default False
        Raise unless every destination cell's weights sum to 1, i.e. the remap
        cannot return a reduced or all-zero field. Applied AFTER ``normalization``
        and ``polar_fill``, so it validates the treated weights. Callers that
        wrap-pad their source longitude must ALSO assert the raw axis spans the
        globe (see :func:`check_axis_span`) -- padding makes a partial-longitude
        source look complete to this check. Host-side only; no AD/JIT impact.
    normalization : {'dstarea', 'fracarea'}, default 'dstarea'
        How a PARTIALLY covered destination cell is normalised (ESMF's names).
        ``'dstarea'`` divides each overlap by the full destination cell area:
        strictly conservative, but such a cell returns ``coverage * field``.
        ``'fracarea'`` divides by the COVERED LATITUDE FRACTION (never by the row
        sum, which would normalise a longitude deficit away too), returning the
        area-weighted mean of the source that overlaps -- so a constant field is preserved
        exactly. Use it whenever the field is INTENSIVE (temperature, pressure, a
        flux density) and the shortfall is a data gap rather than genuine zero.
        It is NOT strictly conservative; see :func:`_renormalize_lat_shortfall`.
    polar_fill : bool, default False
        Fill destination rows lying entirely beyond the source's latitude band
        from the source's own outermost row (zonally resolved). Renormalisation
        cannot reach those -- their weights sum to exactly 0 -- so without this a
        destination finer than the source's polar gap returns zeros there. A
        zeroth-order poleward extrapolation: it fabricates data outside the
        source's domain and is likewise not conservative. See
        :func:`_append_polar_fill`.
    max_polar_gap_deg : float, default 2.0
        How far poleward ``polar_fill`` will extrapolate. Filling is defensible
        across a genuine polar DATA GAP (CORE-II 0.51 deg, JRA55-do 0.15 deg) and
        not as a way to spread a regionally-limited source over the globe, so a
        source stopping farther than this from the pole raises instead.

    Returns
    -------
    ConservativeRegridWeights
        Sparse pre-computed weights ready for :func:`apply_conservative_regrid`.
    """
    # Validated at FUNCTION ENTRY on the static value, before the O(N) weight
    # build (CLAUDE.md dispatch rule): a typo must not silently select 'dstarea',
    # which returns coverage x field on a partly covered polar row.
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected one of "
            f"{sorted(_NORMALIZATIONS)}. 'dstarea' divides each overlap by the "
            "FULL destination cell area (strictly conservative, but a partly "
            "covered cell returns coverage x field); 'fracarea' divides by the "
            "COVERED latitude fraction instead (preserves the field value, not "
            "the integral)."
        )
    if not (np.isfinite(max_polar_gap_deg) and float(max_polar_gap_deg) > 0.0):
        raise ValueError(
            "max_polar_gap_deg must be finite and > 0 -- it bounds how far the "
            "polar treatments may extrapolate. NaN would silently disable the "
            f"guard (every comparison against it is False). Got "
            f"{max_polar_gap_deg!r}."
        )

    src_lat_edges = np.asarray(src_lat_edges, dtype=np.float64)
    src_lon_edges = np.asarray(src_lon_edges, dtype=np.float64)
    dst_lat_edges = np.asarray(dst_lat_edges, dtype=np.float64)
    dst_lon_edges = np.asarray(dst_lon_edges, dtype=np.float64)
    for name, e in [
        ("src_lat_edges", src_lat_edges),
        ("src_lon_edges", src_lon_edges),
        ("dst_lat_edges", dst_lat_edges),
        ("dst_lon_edges", dst_lon_edges),
    ]:
        _check_edges(e, name)

    n_src_lat = src_lat_edges.size - 1
    n_src_lon = src_lon_edges.size - 1
    n_dst_lat = dst_lat_edges.size - 1
    n_dst_lon = dst_lon_edges.size - 1

    # Pre-compute sin(edges) — the lat-band area factor on the sphere is
    # sin(top) - sin(bottom).  Working in this transformed coordinate
    # makes the overlap a literal 1-D segment intersection.
    src_sin = np.sin(src_lat_edges)        # (n_src_lat + 1,)
    dst_sin = np.sin(dst_lat_edges)        # (n_dst_lat + 1,)

    # Lat overlap matrix:  shape (n_dst_lat, n_src_lat)
    # entry [j_dst, j_src] = max(0, min(dst_sin[j+1], src_sin[k+1])
    #                              - max(dst_sin[j], src_sin[k]))
    src_sin_lo = src_sin[:-1][None, :]     # (1, n_src_lat)
    src_sin_hi = src_sin[1:][None, :]      # (1, n_src_lat)
    dst_sin_lo = dst_sin[:-1][:, None]     # (n_dst_lat, 1)
    dst_sin_hi = dst_sin[1:][:, None]      # (n_dst_lat, 1)
    lat_overlap = np.maximum(
        0.0,
        np.minimum(src_sin_hi, dst_sin_hi) - np.maximum(src_sin_lo, dst_sin_lo),
    )                                       # (n_dst_lat, n_src_lat)

    # Lon overlap matrix similarly: (n_dst_lon, n_src_lon)
    src_lon_lo = src_lon_edges[:-1][None, :]
    src_lon_hi = src_lon_edges[1:][None, :]
    dst_lon_lo = dst_lon_edges[:-1][:, None]
    dst_lon_hi = dst_lon_edges[1:][:, None]
    lon_overlap = np.maximum(
        0.0,
        np.minimum(src_lon_hi, dst_lon_hi) - np.maximum(src_lon_lo, dst_lon_lo),
    )                                       # (n_dst_lon, n_src_lon)

    # Target cell areas (per (j_dst, i_dst) — but areas separate into
    # lat and lon factors): A(j, i) = (dst_sin_hi - dst_sin_lo)[j]
    #                                * (dst_lon_hi - dst_lon_lo)[i]
    dst_lat_area = (dst_sin[1:] - dst_sin[:-1])           # (n_dst_lat,)
    dst_lon_area = (dst_lon_edges[1:] - dst_lon_edges[:-1])  # (n_dst_lon,)

    # Build the sparse triple (src_flat, dst_flat, weight).  Since the
    # overlap structure is separable, we can enumerate non-zero pairs:
    #   (j_dst, j_src) where lat_overlap[j_dst, j_src] > 0
    #   (i_dst, i_src) where lon_overlap[i_dst, i_src] > 0
    # and take their outer product.  For 0.5° -> 1°, that's ~2 lat and
    # ~2 lon contributions per target cell -> ~4 entries per cell.
    j_dst_lat, j_src_lat = np.nonzero(lat_overlap > 0.0)
    lat_w_pairs = lat_overlap[j_dst_lat, j_src_lat]       # (N_lat_pairs,)

    i_dst_lon, i_src_lon = np.nonzero(lon_overlap > 0.0)
    lon_w_pairs = lon_overlap[i_dst_lon, i_src_lon]       # (N_lon_pairs,)

    n_lat_pairs = j_dst_lat.size
    n_lon_pairs = i_dst_lon.size

    # Outer product: every (lat-pair, lon-pair) combination yields one
    # non-zero weight in the full overlap tensor.
    # Shape: (n_lat_pairs * n_lon_pairs,)
    j_dst_flat = np.repeat(j_dst_lat, n_lon_pairs)
    j_src_flat = np.repeat(j_src_lat, n_lon_pairs)
    lat_w_flat = np.repeat(lat_w_pairs, n_lon_pairs)
    i_dst_flat = np.tile(i_dst_lon, n_lat_pairs)
    i_src_flat = np.tile(i_src_lon, n_lat_pairs)
    lon_w_flat = np.tile(lon_w_pairs, n_lat_pairs)

    # Spherical overlap area for each pair:
    overlap_area = lat_w_flat * lon_w_flat                # (N_overlaps,)

    # Target cell area for normalisation:
    dst_cell_area = dst_lat_area[j_dst_flat] * dst_lon_area[i_dst_flat]
    # Guard against zero-area target cells (shouldn't happen for valid
    # edges, but float underflow at the pole might).
    weights = overlap_area / np.maximum(dst_cell_area, 1e-30)

    src_idx = j_src_flat * n_src_lon + i_src_flat
    dst_idx = j_dst_flat * n_dst_lon + i_dst_flat

    # BOUND the extrapolation, for BOTH treatments.  They differ only in degree:
    # `fracarea` spreads a partly covered row's real data over the part the source
    # never reached, `polar_fill` does the same for a row it never reached at all.
    # Either is defensible across a genuine polar DATA GAP -- 0.51 deg for CORE-II,
    # 0.15 deg for JRA55-do, over which even a 1 K/deg polar air-temperature
    # gradient contributes well under 1 K -- and neither is a licence to spread a
    # REGIONAL source over the globe.  Not hypothetical here: silent rad2deg bugs in
    # this pipeline (omip_pipeline_coordinate_bugs) turn a global latitude axis into
    # a +-1.57 deg "band", which must stay LOUD.  Checked on the source's geometry
    # alone, so `polar_fill=True` cannot pass where `polar_fill=False` fails.
    if normalization == "fracarea" or polar_fill:
        gap_deg = np.degrees(max(
            float(np.arcsin(np.clip(src_sin[0], -1.0, 1.0))) + np.pi / 2.0,
            np.pi / 2.0 - float(np.arcsin(np.clip(src_sin[-1], -1.0, 1.0))),
        ))
        if gap_deg > max_polar_gap_deg:
            raise ValueError(
                f"compute_overlap_weights: the source stops {gap_deg:.4g} deg from "
                f"the pole, beyond the {max_polar_gap_deg:g} deg polar-gap budget. "
                "'fracarea' and polar_fill fill a GAP; filling this far is "
                "extrapolation, and would silently spread a regionally limited "
                "source (or a latitude axis read in the wrong units) over the "
                "globe. Widen max_polar_gap_deg deliberately if the source really "
                "is meant to be extended this far."
            )

    # POLE FILL first: it ADDS entries for destination rows the source cannot
    # reach at all, which renormalisation cannot rescue (there is nothing to
    # rescale -- their latitude coverage is exactly 0).
    if polar_fill:
        src_idx, dst_idx, weights = _append_polar_fill(
            src_idx, dst_idx, weights,
            src_sin=src_sin, dst_sin=dst_sin,
            lon_overlap=lon_overlap, dst_lon_area=dst_lon_area,
            n_src_lat=n_src_lat, n_src_lon=n_src_lon, n_dst_lon=n_dst_lon,
        )

    # FRACAREA second: rescale each PARTIALLY covered cell so its weights sum to
    # 1, turning `coverage x field` into the area-weighted mean of the source
    # that does overlap.  Scaled by the LATITUDE factor only, so a LONGITUDE
    # seam/ghost deficit survives into the row sum and still raises below.
    # Rows already complete are untouched, so a fully covered remap is
    # bit-identical to 'dstarea'.
    if normalization == "fracarea":
        weights = _renormalize_lat_shortfall(
            dst_idx, weights,
            _attainable_lat_coverage(src_sin, dst_sin, dst_lat_area),
            n_dst_lon,
        )

    if require_full_coverage:
        _validate_full_coverage(
            dst_idx, weights, n_dst_lat, n_dst_lon,
            tol=_COVERAGE_TOL, name="compute_overlap_weights",
            src_sin=src_sin,
        )

    return ConservativeRegridWeights(
        src_idx_flat=jnp.asarray(src_idx, dtype=jnp.int32),
        dst_idx_flat=jnp.asarray(dst_idx, dtype=jnp.int32),
        weights=jnp.asarray(weights, dtype=jnp.float64),
        src_shape=(n_src_lat, n_src_lon),
        dst_shape=(n_dst_lat, n_dst_lon),
        n_dst_cells=n_dst_lat * n_dst_lon,
    )


def _apply_2d(field_2d: jnp.ndarray, weights: ConservativeRegridWeights) -> jnp.ndarray:
    """Apply weights to a single 2-D ``(n_src_lat, n_src_lon)`` field."""
    src_flat = field_2d.reshape(-1)
    contributions = src_flat[weights.src_idx_flat] * weights.weights
    dst_flat = jax.ops.segment_sum(
        contributions, weights.dst_idx_flat,
        num_segments=weights.n_dst_cells,
    )
    return dst_flat.reshape(weights.dst_shape)


def apply_conservative_regrid(
    field: jnp.ndarray,
    weights: ConservativeRegridWeights,
) -> jnp.ndarray:
    """Apply pre-computed conservative weights to a source field.

    Parameters
    ----------
    field : jax.Array
        Source field whose TRAILING axes equal ``weights.src_shape`` — rank 1
        ``(nCells,)`` (unstructured/MPAS), rank 2 ``(n_src_lat, n_src_lon)``
        (lat-lon), or rank 3 ``(6, n, n)`` (cubed-sphere).  Any remaining leading
        axes (levels, tracers, ensemble) are vmap-broadcast.
    weights : ConservativeRegridWeights
        From :func:`compute_overlap_weights` (lat-lon) or an unstructured weight
        generator (e.g. ``conservative_regrid_unstructured``).

    Returns
    -------
    jax.Array
        Regridded field with shape ``(*leading, *weights.dst_shape)``.
    """
    # ``src_shape`` may be rank 1 (unstructured/MPAS, ``(nCells,)``), rank 2
    # (regular lat-lon), or rank 3 (cubed-sphere, ``(6, n, n)``).  Match the
    # trailing axes against it and vmap over any remaining leading axes.
    nd = len(weights.src_shape)
    if tuple(field.shape[-nd:]) != tuple(weights.src_shape):
        raise ValueError(
            f"field trailing shape {field.shape[-nd:]} does not match "
            f"weights.src_shape {weights.src_shape}"
        )
    if field.ndim == nd:
        return _apply_2d(field, weights)
    # Generic ND: flatten leading axes, vmap, unflatten.
    leading = field.shape[:-nd]
    flat = field.reshape((-1,) + tuple(weights.src_shape))
    out = jax.vmap(_apply_2d, in_axes=(0, None))(flat, weights)
    return out.reshape(leading + tuple(weights.dst_shape))
