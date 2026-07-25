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


_COVERAGE_TOL = 1e-6  # |sum(weights) - attainable| tolerance for a remap that is
# REQUIRED to cover every destination cell as fully as its source allows.
# float64 area-ratio roundoff is ~1e-14 (~4 adds/cell), so 1e-6 sits ~1e8x above
# noise yet flags a single missing 0.25deg source cell in a 1deg destination or a
# ~12.5% seam deficit.  Deliberately NOT loosened to swallow the polar deficit
# (measured 6.6e-2 for CORE-II -> 2deg/90x180; 2.6e-1 at 1deg): a tolerance wide
# enough for that would also swallow a genuinely half-covered seam cell.  Relax
# the REFERENCE instead (see lat_shortfall_floor).


# Axis-span floor tolerance (radians).  float64 edge inference is exact to ~1e-15,
# but grid coordinates are STORED at the precision policy's dtype (float32 by
# default for LatLonGrid), which alone reaches ~2.2e-7 rad of span error at
# n_lon = 720 -- only ~4.5x under a flat 1e-6.  So the effective tolerance also
# scales with the cell width (see check_axis_span): a genuine missing column is a
# whole cell wide, so 1% of a cell separates the two by ~100x at any resolution.
_SPAN_TOL = 1e-6
_SPAN_CELL_FRAC = 0.01

# Smallest ENFORCEABLE lat_shortfall_floor.  Must stay comfortably above
# _COVERAGE_TOL: the guard fires on `lat_frac < floor - _COVERAGE_TOL`, so a floor
# at or below the tolerance drives that threshold to <= 0 and an all-zero row
# passes -- the guard would silently invert.  1e-3 is also the point below which a
# floor stops being a defensible policy: at 0.1% coverage a 250 K air temperature
# arrives as 0.25 K, which is not a field anyone means to ship.
_MIN_LAT_FLOOR = 1e-3


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


def _attainable_lat_coverage(src_sin, dst_sin, dst_lat_area):
    """Fraction of each destination lat band the source CAN supply, in [0, 1].

    CONVENTION: coverage is a non-negative AREA fraction, computed in the
    ``sin(lat)`` coordinate where a spherical band's area is linear, with both
    axes ascending (south -> north).  Source cells are contiguous, so their union
    is the single band ``[src_sin[0], src_sin[-1]]``; the attainable fraction is
    that band's overlap with the destination cell over the destination cell's own
    extent.  1.0 means the source reaches at least as far poleward as this
    destination row; < 1.0 means it physically cannot fill it.
    """
    covered = np.clip(
        np.minimum(dst_sin[1:], src_sin[-1])
        - np.maximum(dst_sin[:-1], src_sin[0]), 0.0, None)
    # No epsilon clamp on the denominator: it would have to differ from the
    # `max(dst_cell_area, 1e-30)` used for the weights themselves (that one is a
    # lat*lon PRODUCT and so binds at a different threshold), and a mismatched
    # pair produces a spurious raise blaming a longitude seam.  A zero-area
    # destination row cannot be covered at all, so report it as 0 and let the
    # zero-coverage guard reject it explicitly.
    frac = np.zeros_like(covered)
    ok = dst_lat_area > 0.0
    frac[ok] = covered[ok] / dst_lat_area[ok]
    return frac


def _required_lat_coverage(src_sin, dst_sin, *, floor):
    """Per-row minimum acceptable latitude coverage, in [floor, 1].

    A destination row that pokes outside the source's latitude band may be
    covered as little as ``floor``; every other row must be complete.

    ``floor = 1.0`` (the default everywhere) reproduces the original hard-1.0
    check exactly. Only a caller whose source is a real dataset with a physical
    polar gap should relax it, and only as far as it can defend: the weights are
    NOT renormalised (``weights = overlap_area / dst_cell_area``), so a row with
    coverage ``f`` returns ``f * field`` -- a REDUCED value, not an average. At
    ``f = 0.0024`` (CORE-II onto 350 lat rows) a 250 K air temperature arrives as
    0.61 K, which is the bulk-flux / 10-m-pressure NaN this guard exists to
    prevent. ``floor`` is
    therefore how much dilution the caller is willing to ship, not a numerical
    epsilon; 1e-6 would be no line at all.

    NOTE the geometry, which bounds how much this can ever buy: a contiguous
    source band's outer edge cuts EXACTLY ONE destination row, so at most one row
    per pole is partial and every row beyond it is exactly zero-covered. Relaxing
    the floor therefore cannot admit a destination finer than the polar gap -- the
    outermost rows there have no source data at all and no floor above 0 accepts
    them. Making those rows usable needs a renormalising or pole-filling remap,
    which is a change to the WEIGHTS, not to this check.
    """
    reaches_beyond = (dst_sin[:-1] < src_sin[0]) | (dst_sin[1:] > src_sin[-1])
    return np.where(reaches_beyond, float(floor), 1.0)


def _validate_attainable_coverage(
    dst_idx, weights, n_dst_lat, n_dst_lon, *, expected, tol, name,
):
    """Raise if any destination cell's weights depart from its ATTAINABLE coverage.

    HOST-SIDE (numpy) check on the STATIC weights BEFORE they become a jnp
    constant -- zero autodiff / JIT / trace impact (the traced apply is an
    unchanged ``segment_sum`` over these compile-time weights).  A destination
    cell covered less than the source can supply silently gets a physically
    REDUCED field (seam/ghost deficit, wrong edges, an interior hole); a
    fully-uncovered cell gets 0.  Fail LOUDLY rather than emit a reduced field.

    Compared against ``expected`` rather than a hard 1.0 because ``sum == 1`` is
    not attainable for every destination cell: a source whose outermost latitude
    row stops short of the pole (i.e. every real forcing dataset) legitimately
    under-covers the polar destination row, and demanding 1.0 there aborts valid
    preprocessing.  ``expected`` relaxes LATITUDE only -- longitude is required
    complete, so a source that does not span the destination in longitude still
    raises.
    """
    n_dst_cells = int(n_dst_lat) * int(n_dst_lon)
    row_sum = np.bincount(
        np.asarray(dst_idx).ravel(),
        weights=np.asarray(weights).ravel(),
        minlength=n_dst_cells,
    )
    dev = np.abs(row_sum - expected)
    bad = np.nonzero(dev > tol)[0]
    if bad.size:
        worst = int(bad[int(dev[bad].argmax())])
        j, i = divmod(worst, int(n_dst_lon))
        raise ValueError(
            f"{name}: {bad.size} destination cell(s) have summed remap weights "
            f"off their ATTAINABLE coverage by > {tol:g} (worst deviation "
            f"{float(dev.max()):.3e} at dst lat_row {j}, lon_col {i}: sum = "
            f"{float(row_sum[worst]):.12g}, attainable = "
            f"{float(expected[worst]):.12g}); the source does not fully cover the "
            "destination there. Latitude shortfall at the poles is already "
            "allowed for, so this is a real deficit -- a longitude seam/ghost "
            "gap, wrong edges, or an interior hole. It would silently produce a "
            "reduced field."
        )


def compute_overlap_weights(
    src_lat_edges: np.ndarray,
    src_lon_edges: np.ndarray,
    dst_lat_edges: np.ndarray,
    dst_lon_edges: np.ndarray,
    *,
    require_attainable_coverage: bool = False,
    lat_shortfall_floor: float = 1.0,
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
    require_attainable_coverage : bool, default False
        Raise if any destination cell's summed weights fall short of what the
        source can supply. LONGITUDE is always required complete, so a source that
        does not span the destination in longitude raises. Callers that wrap-pad
        their source longitude must ALSO assert the raw axis spans the globe (see
        :func:`check_axis_span`) -- padding makes a partial-longitude source look
        complete to this check. Host-side only; no AD/JIT impact.
    lat_shortfall_floor : float, default 1.0
        Minimum acceptable latitude coverage for a destination row that extends
        beyond the source's latitude band. The default 1.0 demands complete
        coverage everywhere, i.e. exactly the original hard-1.0 behaviour -- keep
        it for model-to-model remaps, where both grids are global and any
        shortfall is a bug. A caller whose source is a real dataset with a
        physical polar gap (CORE-II NYF, JRA55-do) relaxes it to the amount of
        dilution it is willing to ship: weights are NOT renormalised, so a row
        with coverage ``f`` returns ``f * field``. See
        :func:`_required_lat_coverage`.

    Returns
    -------
    ConservativeRegridWeights
        Sparse pre-computed weights ready for :func:`apply_conservative_regrid`.
    """
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

    if not _MIN_LAT_FLOOR <= float(lat_shortfall_floor) <= 1.0:
        raise ValueError(
            f"lat_shortfall_floor must lie in [{_MIN_LAT_FLOOR:g}, 1] -- 1.0 "
            "demands complete coverage, a smaller value is how much dilution the "
            "caller accepts on a destination row that extends past the source's "
            "latitude band. It is a physical policy, not a numerical epsilon: a "
            f"floor at or below the coverage tolerance ({_COVERAGE_TOL:g}) is not "
            "even enforceable, since the comparison threshold would go negative "
            "and an ALL-ZERO row would pass. Got "
            f"{lat_shortfall_floor!r}."
        )
    if require_attainable_coverage:
        # Relax LATITUDE to what the source can supply; keep LONGITUDE at 1.0 so
        # a lon seam/ghost gap still raises.  Separable weights make the product
        # exact: row_sum[j, i] = lat_frac[j] * lon_frac[i], and np.repeat matches
        # the dst_idx = j * n_dst_lon + i flattening used above.
        lat_frac = _attainable_lat_coverage(src_sin, dst_sin, dst_lat_area)
        # Qualify the reference BEFORE trusting it: a row matches its own
        # attainable coverage by construction, so without this a source that
        # cannot reach a row passes while emitting a reduced or all-zero field.
        required = _required_lat_coverage(
            src_sin, dst_sin, floor=lat_shortfall_floor)
        under = np.nonzero(lat_frac < required - _COVERAGE_TOL)[0]
        if under.size:
            worst = int(under[int(np.argmin(lat_frac[under] - required[under]))])
            raise ValueError(
                f"compute_overlap_weights: destination lat_row {worst} is only "
                f"{float(lat_frac[worst]):.6g} covered by the source but "
                f"{float(required[worst]):.6g} is required "
                f"({under.size} such row(s) of {n_dst_lat}). Weights are NOT "
                "renormalised, so that row would return coverage x field -- a "
                "REDUCED field (at forcing scale, an air temperature of a few K). "
                "The source spans "
                f"asin({float(src_sin[0]):.9g})..asin({float(src_sin[-1]):.9g}) "
                "rad. Either the destination is finer than the source's polar gap "
                "(coarsen it, or fill the source poleward), or the source is "
                "regionally limited where a global one was expected."
            )
        expected = np.repeat(lat_frac, n_dst_lon)
        _validate_attainable_coverage(
            dst_idx, weights, n_dst_lat, n_dst_lon,
            expected=expected, tol=_COVERAGE_TOL,
            name="compute_overlap_weights",
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
