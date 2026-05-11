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


def compute_overlap_weights(
    src_lat_edges: np.ndarray,
    src_lon_edges: np.ndarray,
    dst_lat_edges: np.ndarray,
    dst_lon_edges: np.ndarray,
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
        Source field with last two axes equal to ``weights.src_shape``,
        i.e. shape ``(*leading, n_src_lat, n_src_lon)``.  Leading axes
        (e.g. time) are vmap-broadcast.
    weights : ConservativeRegridWeights
        From :func:`compute_overlap_weights`.

    Returns
    -------
    jax.Array
        Regridded field with shape ``(*leading, n_dst_lat, n_dst_lon)``.
    """
    if field.shape[-2:] != weights.src_shape:
        raise ValueError(
            f"field last-2 shape {field.shape[-2:]} does not match "
            f"weights.src_shape {weights.src_shape}"
        )
    if field.ndim == 2:
        return _apply_2d(field, weights)
    # Generic ND: flatten leading axes, vmap, unflatten.
    leading = field.shape[:-2]
    flat = field.reshape((-1,) + weights.src_shape)
    out = jax.vmap(_apply_2d, in_axes=(0, None))(flat, weights)
    return out.reshape(leading + weights.dst_shape)
