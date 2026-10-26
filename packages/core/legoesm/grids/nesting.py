"""One-way regional grid NESTING on the latitude-longitude grid.

A *nested* configuration is a coarse PARENT grid plus a refined CHILD grid that
covers a contiguous rectangular sub-region of the parent, related by an integer
``refinement_ratio`` ``r``: every parent cell inside the child footprint is split
into ``r x r`` child cells, so the child cell *centres* are geometrically nested
within the parent (no offset drift, no overlap, no gaps).

This module owns only the GRID GEOMETRY and the parent->child interpolation
OPERATOR — both substrate-level (``legoesm.grids`` / ``legoesm.core``) concerns
with no dependency on any dynamical core.  No nested *time stepping* is
implemented (the former shallow-water nest driver was test-only and was
deleted); one would live with the dynamics that uses it, because the
substrate must not import a component (import-linter contract #4).

Design
------
* **One-way (parent -> child) only.**  The parent integrates as a standalone
  global model; the child's outermost ``n_halo`` boundary rows/columns are
  PRESCRIBED each step by interpolating the parent state, never fed back.
  Two-way feedback (child -> parent restriction) is intentionally NOT
  implemented here — see the module-level note at the bottom.
* **Bilinear, cached weights.**  The parent->child gather indices and weights
  are computed once at construction (host/NumPy, geometry only) and applied with
  a pure-JAX ``gather + weighted sum`` so the per-step boundary fill is
  JIT/``jax.grad`` compatible.  Longitude wraps periodically on the parent;
  latitude clamps at the parent's first/last cell centre (the child footprint
  must stay strictly inside the parent, validated at construction).
* **Conservation.**  Mass conservation is enforced and *measured* on the child
  INTERIOR (the cells excluded from the prescribed boundary band): with a
  boundary forcing equal to the exact steady solution (Williamson-2) the interior
  mass is conserved to round-off.  For a general flow the interior mass change
  equals the net boundary mass flux (a budget), which the stepping driver checks.

References
----------
* Davies, H. C. (1976). A lateral boundary formulation for multi-level
  prediction models. Q. J. R. Meteorol. Soc., 102, 405-418. (specified /
  relaxation lateral boundary forcing — the 1-way nest boundary condition).
* Harris, L. M., & Lin, S.-J. (2013). A two-way nested global-regional dynamical
  core on the cubed-sphere grid. Mon. Wea. Rev., 141, 283-306. (the FV3 nesting
  this is the lat-lon analogue of; integer concurrent refinement, specified
  boundary, mass budget at the nest edge).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.latlon import (
    LatLonGrid,
    build_uniform_latlon_grid_from_axes,
)


class BoundaryInterpWeights(NamedTuple):
    """Precomputed bilinear weights mapping a parent field to child targets.

    The operator gathers from the FLATTENED parent cell-centre field
    (``parent.n_lat * parent.n_lon``) at four corner indices per target point
    and forms a bilinear-weighted sum.  Indices/weights are static
    (geometry-only), so applying them is a pure-JAX gather — JIT and
    ``jax.grad`` safe.

    Attributes
    ----------
    src_indices : int32 array, shape (n_target, 4)
        Flat indices into the parent cell-centre field for the 4 bilinear
        corners (SW, SE, NW, NE) of each target point.
    weights : float array, shape (n_target, 4)
        Bilinear weights summing to 1 along axis -1.
    target_shape : tuple[int, int]
        The shape the gathered result reshapes to (child cell / face grid).
    parent_flat_size : int
        ``parent.n_lat * parent.n_lon`` (the flattened source size).
    """

    src_indices: jax.Array
    weights: jax.Array
    target_shape: tuple[int, int]
    parent_flat_size: int


class NestedLatLonGrid(NamedTuple):
    """A coarse parent + refined child lat-lon nest (one-way).

    A JAX pytree (NamedTuple of grids + arrays).  ``parent`` is a global uniform
    lat-lon grid; ``child`` is a uniform lat-lon grid refined by
    ``refinement_ratio`` covering a rectangular sub-region of the parent.  The
    interpolation operators are STAGGER-MATCHED: ``interp_centers`` maps the
    parent CELL-CENTRE field to the child cell centres, ``interp_uface`` maps the
    parent U-FACE field to the child u-faces, and ``interp_vface`` maps the parent
    V-FACE field to the child v-faces — each source location matches its target so
    the staggered C-grid boundary is prescribed component-by-component, face to
    face (a child face coincident with a parent face reproduces the parent value
    exactly, preserving the discrete divergence/geostrophic balance at the nest
    edge).

    ``n_halo`` is the width (in child cells) of the HARD-prescribed boundary band
    on each side (set exactly to the interpolated parent); the interior (cells at
    least ``n_halo`` from every edge) is the region over which conservation is
    measured.  ``n_relax`` is the width of an additional Davies (1976) RELAXATION
    zone just inside the hard band, over which the child is nudged toward the
    interpolated parent with a raised-cosine weight decaying ``1 -> 0`` inward —
    this absorbs outgoing gravity waves instead of reflecting them off a hard
    wall (which otherwise seeds grid-scale v-wind noise; visible in the W2
    diagnostic).
    """

    parent: LatLonGrid
    child: LatLonGrid
    refinement_ratio: int
    n_halo: int
    n_relax: int
    interp_centers: BoundaryInterpWeights
    interp_uface: BoundaryInterpWeights
    interp_vface: BoundaryInterpWeights

    @property
    def interior_mask(self) -> jax.Array:
        """``(child.n_lat, child.n_lon)`` 1.0 on interior cells, 0.0 on the band.

        "Interior" excludes only the HARD band (``n_halo``); the relaxation zone
        counts as interior for mass accounting (it is freely evolving + nudged,
        not prescribed), so the conservation budget is taken there.
        """
        return _interior_mask(self.child.n_lat, self.child.n_lon, self.n_halo)

    @property
    def boundary_mask(self) -> jax.Array:
        """``(child.n_lat, child.n_lon)`` 1.0 on the hard band, 0.0 inside."""
        return 1.0 - self.interior_mask

    @property
    def relax_weight(self) -> jax.Array:
        """``(n_lat, n_lon)`` Davies nudging weight on cell centres.

        1.0 on the hard band, raised-cosine taper to 0.0 across the next
        ``n_relax`` cells, 0.0 in the free interior.
        """
        return _relax_weight_profile(
            self.child.n_lat, self.child.n_lon, self.n_halo, self.n_relax,
        )

    @property
    def free_interior_mask(self) -> jax.Array:
        """``(n_lat, n_lon)`` 1.0 where the child is FREELY evolving.

        Excludes BOTH the hard ``n_halo`` band AND the ``n_relax`` relaxation
        zone — i.e. the cells where ``relax_weight == 0``.  This is the domain
        over which mass is conserved and the fixer acts: the band is prescribed
        and the relaxation zone is nudged (neither is a closed, conserved region).
        """
        return _interior_mask(
            self.child.n_lat, self.child.n_lon, self.n_halo + self.n_relax,
        )

    @property
    def relax_weight_uface(self) -> jax.Array:
        """``(n_lat, n_lon+1)`` relaxation weight at the u-faces (lon interfaces).

        A u-face's relaxation weight is the max of its two flanking cell-centre
        weights (the face is "in the zone" if either neighbouring cell is), so
        the zonal-momentum nudging tapers consistently with the cell-centre h
        nudging.  The periodic-seam wrap is irrelevant on a regional child (the
        west/east faces lie in the hard band regardless).
        """
        wc = self.relax_weight  # (n_lat, n_lon)
        left = jnp.concatenate([wc[:, :1], wc], axis=1)        # face k flank i-1
        right = jnp.concatenate([wc, wc[:, -1:]], axis=1)      # face k flank i
        return jnp.maximum(left, right)

    @property
    def relax_weight_vface(self) -> jax.Array:
        """``(n_lat+1, n_lon)`` relaxation weight at the v-faces (lat interfaces)."""
        wc = self.relax_weight  # (n_lat, n_lon)
        below = jnp.concatenate([wc[:1, :], wc], axis=0)
        above = jnp.concatenate([wc, wc[-1:, :]], axis=0)
        return jnp.maximum(below, above)


def _interior_mask(n_lat: int, n_lon: int, n_halo: int) -> jax.Array:
    m = jnp.zeros((n_lat, n_lon))
    m = m.at[n_halo : n_lat - n_halo, n_halo : n_lon - n_halo].set(1.0)
    return m


def _edge_distance(n: int, n_halo: int) -> np.ndarray:
    """For each index 0..n-1, distance (in cells) INTO the interior past the hard
    band: 0 on/within the hard band, 1 at the first relaxation cell, …."""
    idx = np.arange(n)
    d_lo = idx - n_halo + 1          # 1 at first cell inside the band edge
    d_hi = (n - 1 - idx) - n_halo + 1
    d = np.minimum(d_lo, d_hi)
    return d


def _relax_weight_profile(
    n_lat: int, n_lon: int, n_halo: int, n_relax: int,
) -> jax.Array:
    """Separable raised-cosine relaxation weight on cell centres (host->device).

    Weight ``w(d)`` as a function of the per-axis distance ``d`` into the interior
    past the hard band: ``w=1`` for ``d<=0`` (hard band), raised-cosine
    ``0.5*(1+cos(pi*d/(n_relax+1)))`` for ``0<d<=n_relax``, ``0`` beyond.  The 2-D
    weight is the MAX over the two axes so corners relax as strongly as edges.
    """
    if n_relax <= 0:
        return _interior_mask(n_lat, n_lon, n_halo) * 0.0 + (
            1.0 - _interior_mask(n_lat, n_lon, n_halo)
        )

    def axis_w(n: int) -> np.ndarray:
        d = _edge_distance(n, n_halo).astype(np.float64)
        w = np.where(
            d <= 0.0, 1.0,
            np.where(
                d <= n_relax,
                0.5 * (1.0 + np.cos(np.pi * d / (n_relax + 1.0))),
                0.0,
            ),
        )
        return w

    wlat = axis_w(n_lat)[:, None]
    wlon = axis_w(n_lon)[None, :]
    w2d = np.maximum(wlat, wlon)
    return jnp.asarray(w2d)


# ----------------------------------------------------------------------------
# Bilinear interpolation weights (host-side, geometry only)
# ----------------------------------------------------------------------------


def _bilinear_weights_general(
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    *,
    lat0: float,
    dlat_p: float,
    n_lat_src: int,
    lon0: float,
    dlon_p: float,
    n_lon_src: int,
    lon_periodic_period: int | None,
) -> BoundaryInterpWeights:
    """Bilinear gather weights from a uniform SOURCE node grid to targets.

    The source is a uniform lat-lon NODE grid laid out row-major as
    ``(n_lat_src, n_lon_src)`` whose node ``(j, i)`` sits at latitude
    ``lat0 + j*dlat_p`` and longitude ``lon0 + i*dlon_p``.  This is the single
    geometry kernel behind all three parent->child operators: the SAME bilinear
    gather serves parent CELL CENTRES, parent U-FACES and parent V-FACES — only
    the source ORIGIN, SPACING, COUNT and the longitude periodicity differ
    between the three staggered locations.  Centralising it (rather than three
    near-identical copies) is what the no-duplication rule requires.

    Crucially, bilinear interpolation is EXACT at the source nodes: when a target
    point coincides with a source node (e.g. a child face that lands exactly on a
    parent face under integer refinement) the gather puts unit weight on that
    node and REPRODUCES the parent value to round-off — the stagger-correct
    prolongation property the C-grid boundary forcing relies on.

    Parameters
    ----------
    tgt_lat, tgt_lon
        2-D target-point coordinate grids [rad] (same shape).
    lat0, dlat_p, n_lat_src
        Latitude of source row 0, row spacing, number of source rows.  Latitude
        is always CLAMPED to ``[0, n_lat_src-1]`` (the child footprint is
        validated to lie strictly inside the parent band, so no pole
        extrapolation; v-faces extend half a child cell past the outer cell
        centres but stay well inside the >=1-parent-row margin).
    lon0, dlon_p, n_lon_src
        Longitude of source column 0, column spacing, number of STORED source
        columns (the flattened row stride).
    lon_periodic_period
        If not ``None``, longitude is PERIODIC with this many DISTINCT columns
        (the global parent has ``n_lon_p`` distinct longitudes): the fractional
        column index wraps modulo this period and ``i1`` wraps too, so the seam
        between the last and first column interpolates correctly.  Pass ``None``
        for a NON-periodic / already-seam-padded longitude axis (e.g. the parent
        u-FACE field that stores ``n_lon_p+1`` columns with the duplicate seam at
        column ``n_lon_p``): then ``i0`` is clamped to ``[0, n_lon_src-2]`` and
        ``i1 = i0+1`` indexes the stored duplicate directly.
    """
    target_shape = tuple(int(s) for s in np.asarray(tgt_lat).shape)
    tlat = np.asarray(tgt_lat, dtype=np.float64).reshape(-1)
    tlon = np.asarray(tgt_lon, dtype=np.float64).reshape(-1)

    # --- Latitude: fractional index into source rows, clamped to interior. ---
    fj = (tlat - lat0) / dlat_p
    fj = np.clip(fj, 0.0, n_lat_src - 1.0 - 1e-12)
    j0 = np.floor(fj).astype(np.int64)
    j0 = np.clip(j0, 0, n_lat_src - 2)
    wj = fj - j0  # in [0, 1)
    j1 = j0 + 1

    # --- Longitude: fractional index into source columns. ---
    if lon_periodic_period is not None:
        # PERIODIC: bring (tlon - lon0) into [0, period*dlon) so the seam
        # between the last and first DISTINCT column interpolates across the
        # periodic boundary.
        period = int(lon_periodic_period)
        span = period * dlon_p
        dlon_rel = np.mod(tlon - lon0, span)
        fi = dlon_rel / dlon_p
        i0 = np.floor(fi).astype(np.int64) % period
        wi = fi - np.floor(fi)  # in [0, 1)
        i1 = (i0 + 1) % period
    else:
        # NON-periodic / seam-padded: clamp into the stored columns; the caller
        # has stored a duplicate seam column so i1 = i0+1 is always in range.
        fi = (tlon - lon0) / dlon_p
        fi = np.clip(fi, 0.0, n_lon_src - 1.0 - 1e-12)
        i0 = np.floor(fi).astype(np.int64)
        i0 = np.clip(i0, 0, n_lon_src - 2)
        wi = fi - i0  # in [0, 1)
        i1 = i0 + 1

    # Four corners: SW (j0,i0), SE (j0,i1), NW (j1,i0), NE (j1,i1).
    def flat(j, i):
        return (j * n_lon_src + i).astype(np.int32)

    src = np.stack(
        [flat(j0, i0), flat(j0, i1), flat(j1, i0), flat(j1, i1)], axis=-1,
    )  # (n_target, 4)

    w_sw = (1.0 - wi) * (1.0 - wj)
    w_se = wi * (1.0 - wj)
    w_nw = (1.0 - wi) * wj
    w_ne = wi * wj
    w = np.stack([w_sw, w_se, w_nw, w_ne], axis=-1)  # (n_target, 4)
    # Normalise defensively (analytically sums to 1).
    w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-30)

    return BoundaryInterpWeights(
        src_indices=jnp.asarray(src, dtype=jnp.int32),
        weights=jnp.asarray(w),
        target_shape=target_shape,
        parent_flat_size=int(n_lat_src * n_lon_src),
    )


def _bilinear_weights_to_targets(
    parent: LatLonGrid,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> BoundaryInterpWeights:
    """Bilinear gather weights from parent CELL CENTRES to arbitrary targets.

    Thin wrapper over :func:`_bilinear_weights_general` for the parent
    cell-centre source grid: rows at ``lat[0] + j*dlat_p``, columns at
    ``lon[0] + i*dlon_p`` periodic over the global ``n_lon_p`` columns, latitude
    clamped to the first/last cell-centre row (the child footprint is validated to
    lie strictly inside the parent latitude band, so no pole extrapolation).
    """
    plat = np.asarray(parent.lat, dtype=np.float64)  # (n_lat_p,)
    plon = np.asarray(parent.lon, dtype=np.float64)  # (n_lon_p,)
    n_lon_p = plon.shape[0]
    return _bilinear_weights_general(
        tgt_lat, tgt_lon,
        lat0=float(plat[0]), dlat_p=float(parent.dlat), n_lat_src=plat.shape[0],
        lon0=float(plon[0]), dlon_p=float(parent.dlon), n_lon_src=n_lon_p,
        lon_periodic_period=n_lon_p,
    )


def _bilinear_weights_from_parent_uface(
    parent: LatLonGrid,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> BoundaryInterpWeights:
    """Bilinear gather weights from the parent U-FACE field to child u-faces.

    The parent u (zonal) velocity lives on its lon INTERFACES: the stored field
    is ``(n_lat_p, n_lon_p+1)`` with face column ``k`` at longitude
    ``lon[0] - dlon_p/2 + k*dlon_p`` and the latitude axis at the parent CELL
    CENTRES ``lat[j]``.  Face column ``n_lon_p`` is the periodic DUPLICATE of
    column 0 (it stores the same value shifted by 2*pi), so we treat the lon axis
    as a seam-PADDED non-periodic axis of ``n_lon_p+1`` stored columns: a child
    u-face between parent faces ``k`` and ``k+1`` gathers stored columns ``k`` and
    ``k+1`` directly, and a child u-face exactly on parent face ``k`` reproduces
    ``parent.u[:, k]`` to round-off (face-to-face prolongation; no cell-centre
    round-trip that would smooth the geostrophic balance and seed checkerboard
    boundary noise).
    """
    plat = np.asarray(parent.lat, dtype=np.float64)  # (n_lat_p,) cell-centre lat
    n_lon_p = int(parent.n_lon)
    dlon_p = float(parent.dlon)
    lon_w0 = float(np.asarray(parent.lon, dtype=np.float64)[0]) - 0.5 * dlon_p
    return _bilinear_weights_general(
        tgt_lat, tgt_lon,
        lat0=float(plat[0]), dlat_p=float(parent.dlat), n_lat_src=plat.shape[0],
        lon0=lon_w0, dlon_p=dlon_p, n_lon_src=n_lon_p + 1,
        lon_periodic_period=None,
    )


def _bilinear_weights_from_parent_vface(
    parent: LatLonGrid,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> BoundaryInterpWeights:
    """Bilinear gather weights from the parent V-FACE field to child v-faces.

    The parent v (meridional) velocity lives on its lat INTERFACES: the stored
    field is ``(n_lat_p+1, n_lon_p)`` with face row ``j`` at latitude
    ``lat[0] - dlat_p/2 + j*dlat_p`` and the longitude axis at the parent CELL
    CENTRES ``lon[i]`` (periodic over ``n_lon_p`` distinct columns).  Latitude is
    clamped to the stored face rows.  A child v-face exactly on a parent v-face
    reproduces ``parent.v`` there to round-off (face-to-face prolongation).
    """
    plon = np.asarray(parent.lon, dtype=np.float64)  # (n_lon_p,) cell-centre lon
    n_lat_p = int(parent.n_lat)
    n_lon_p = plon.shape[0]
    dlat_p = float(parent.dlat)
    lat_s0 = float(np.asarray(parent.lat, dtype=np.float64)[0]) - 0.5 * dlat_p
    return _bilinear_weights_general(
        tgt_lat, tgt_lon,
        lat0=lat_s0, dlat_p=dlat_p, n_lat_src=n_lat_p + 1,
        lon0=float(plon[0]), dlon_p=float(parent.dlon), n_lon_src=n_lon_p,
        lon_periodic_period=n_lon_p,
    )


def apply_boundary_interp(
    parent_field: jax.Array,
    weights: BoundaryInterpWeights,
) -> jax.Array:
    """Apply precomputed bilinear weights to a parent CELL-CENTRE field.

    Pure JAX (gather + weighted sum); JIT and ``jax.grad`` safe.  ``parent_field``
    has shape ``(parent.n_lat, parent.n_lon)`` (matching ``parent_flat_size`` when
    flattened); the result has shape ``weights.target_shape``.
    """
    flat = parent_field.reshape(weights.parent_flat_size)
    gathered = flat[weights.src_indices]  # (n_target, 4)
    out = jnp.sum(gathered * weights.weights.astype(gathered.dtype), axis=-1)
    return out.reshape(weights.target_shape)


# ----------------------------------------------------------------------------
# Child grid construction (exact integer refinement of a parent sub-region)
# ----------------------------------------------------------------------------


def _refined_child_axes(
    parent: LatLonGrid,
    j_start: int,
    j_count: int,
    i_start: int,
    i_count: int,
    r: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Refined child cell-centre lat/lon axes for a parent index window.

    The child covers parent rows ``[j_start, j_start + j_count)`` and columns
    ``[i_start, i_start + i_count)``, each parent cell split into ``r`` child
    cells along each axis.  Child cell centres are placed so that the ``r`` child
    cells exactly tile the parent cell they refine (centre of parent cell ==
    mean of the ``r`` child-cell centres).
    """
    plat = np.asarray(parent.lat, dtype=np.float64)
    plon = np.asarray(parent.lon, dtype=np.float64)
    dlat_p = float(parent.dlat)
    dlon_p = float(parent.dlon)
    dlat_c = dlat_p / r
    dlon_c = dlon_p / r

    # First child cell-centre = first parent-cell south/west edge + dlat_c/2.
    lat_s_edge = plat[j_start] - 0.5 * dlat_p  # south edge of first parent row
    lon_w_edge = plon[i_start] - 0.5 * dlon_p  # west edge of first parent col
    n_lat_c = j_count * r
    n_lon_c = i_count * r
    lat_c = lat_s_edge + (np.arange(n_lat_c) + 0.5) * dlat_c
    lon_c = lon_w_edge + (np.arange(n_lon_c) + 0.5) * dlon_c
    return lat_c, lon_c, dlat_c, dlon_c


def create_nested_latlon_grid(
    parent_n_lat: int,
    refinement_ratio: int,
    *,
    lat_south_deg: float,
    lat_north_deg: float,
    lon_west_deg: float,
    lon_east_deg: float,
    n_halo: int = 2,
    n_relax: int = 4,
    parent_n_lon: int | None = None,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
) -> NestedLatLonGrid:
    """Build a one-way parent->child lat-lon nest.

    The parent is a GLOBAL uniform lat-lon grid (``create_latlon_grid`` axes);
    the child is its integer refinement over the requested lat/lon box, snapped
    OUTWARD to whole parent cells so the child cell centres nest exactly within
    the parent.

    Parameters
    ----------
    parent_n_lat
        Parent latitude resolution (``parent_n_lon`` defaults to ``2*parent_n_lat``).
    refinement_ratio
        Integer ``r >= 2``: each parent cell -> ``r x r`` child cells.
    lat_south_deg, lat_north_deg, lon_west_deg, lon_east_deg
        Requested child bounding box [deg].  Snapped to whole parent cells; the
        box must leave >= 1 parent row of margin north and south (no pole
        crossing — the bilinear boundary stencil reads parent cell centres only).
    n_halo
        Width (child cells) of the HARD-prescribed boundary band on each side.
    n_relax
        Width (child cells) of the Davies relaxation zone just inside the hard
        band (raised-cosine nudge toward the parent; absorbs outgoing waves).
        Set 0 to disable (hard wall only).
    radius, omega
        Sphere radius [m] and rotation rate [rad/s].

    Returns
    -------
    NestedLatLonGrid
    """
    if refinement_ratio < 2:
        raise ValueError(
            f"refinement_ratio must be >= 2, got {refinement_ratio}."
        )
    if n_halo < 1:
        raise ValueError(f"n_halo must be >= 1, got {n_halo}.")
    if n_relax < 0:
        raise ValueError(f"n_relax must be >= 0, got {n_relax}.")

    from legoesm.grids.latlon import create_latlon_grid

    parent = create_latlon_grid(
        n_lat=parent_n_lat, n_lon=parent_n_lon, radius=radius, omega=omega,
        dtype=dtype,
    )
    r = int(refinement_ratio)

    plat = np.asarray(parent.lat, dtype=np.float64)
    plon = np.asarray(parent.lon, dtype=np.float64)
    dlat_p = float(parent.dlat)
    dlon_p = float(parent.dlon)

    lat_s = np.deg2rad(lat_south_deg)
    lat_n = np.deg2rad(lat_north_deg)
    lon_w = np.deg2rad(lon_west_deg)
    lon_e = np.deg2rad(lon_east_deg)
    if lat_s >= lat_n:
        raise ValueError("lat_south_deg must be < lat_north_deg.")
    if lon_w >= lon_e:
        raise ValueError("lon_west_deg must be < lon_east_deg.")

    # Parent-cell index window covering the box (snap OUTWARD to whole cells).
    lat_edges = plat - 0.5 * dlat_p  # south edge of each parent row
    lon_edges = plon - 0.5 * dlon_p  # west edge of each parent col
    j_start = int(np.searchsorted(lat_edges, lat_s, side="right") - 1)
    j_end = int(np.searchsorted(lat_edges, lat_n, side="left"))
    i_start = int(np.searchsorted(lon_edges, lon_w, side="right") - 1)
    i_end = int(np.searchsorted(lon_edges, lon_e, side="left"))
    j_start = max(j_start, 0)
    j_end = min(max(j_end, j_start + 1), len(plat))
    i_start = max(i_start, 0)
    i_end = min(max(i_end, i_start + 1), len(plon))
    j_count = j_end - j_start
    i_count = i_end - i_start

    # The child footprint must stay strictly inside the parent latitude band so
    # the bilinear stencil never reaches past the first/last parent cell centre
    # (no pole extrapolation).  Require >= 1 parent row of margin N/S.
    if j_start < 1 or j_end > len(plat) - 1:
        raise ValueError(
            "child latitude box must leave >= 1 parent row of margin north and "
            "south of the domain (one-way nest boundary is bilinear-interpolated "
            "from parent cell centres; pole-adjacent nests are not supported)."
        )

    lat_c, lon_c, dlat_c, dlon_c = _refined_child_axes(
        parent, j_start, j_count, i_start, i_count, r,
    )

    child = build_uniform_latlon_grid_from_axes(
        lat=jnp.asarray(lat_c), lon=jnp.asarray(lon_c),
        dlat=dlat_c, dlon=dlon_c, radius=radius, omega=omega, dtype=dtype,
    )

    n_lat_c, n_lon_c = child.n_lat, child.n_lon
    if 2 * (n_halo + n_relax) >= min(n_lat_c, n_lon_c):
        raise ValueError(
            f"n_halo={n_halo} + n_relax={n_relax} too large for child "
            f"{n_lat_c}x{n_lon_c}: the prescribed band + relaxation zone would "
            f"leave no free interior."
        )

    # Cell-centre interpolation weights (parent -> child cell centres).
    child_lat2d = np.asarray(child.lat2d, dtype=np.float64)
    child_lon2d = np.asarray(child.lon2d, dtype=np.float64)
    interp_centers = _bilinear_weights_to_targets(parent, child_lat2d, child_lon2d)

    # u-face targets: lon interfaces, shape (n_lat_c, n_lon_c+1).  u-face k sits
    # at lon_c west-edge + k*dlon_c, latitude = cell-centre lat.  STAGGER-CORRECT
    # prolongation: source DIRECTLY from the parent U-FACE field (not from
    # cell-centre averages of u), so a child u-face coincident with a parent
    # u-face reproduces the parent face value exactly and the discrete
    # divergence/geostrophic balance is preserved at the boundary (no
    # checkerboard wind noise from a centre round-trip).
    lon_w_edge = float(lon_c[0]) - 0.5 * dlon_c
    uface_lon = lon_w_edge + np.arange(n_lon_c + 1) * dlon_c
    uface_lat2d, uface_lon2d = np.meshgrid(
        np.asarray(child.lat, dtype=np.float64), uface_lon, indexing="ij",
    )
    interp_uface = _bilinear_weights_from_parent_uface(
        parent, uface_lat2d, uface_lon2d,
    )

    # v-face targets: lat interfaces, shape (n_lat_c+1, n_lon_c).  v-face k sits
    # at lat_c south-edge + k*dlat_c, longitude = cell-centre lon.  The +/-
    # half-child-cell extension at the child N/S edges stays well inside the
    # >= 1-parent-row margin enforced above.  STAGGER-CORRECT prolongation:
    # source DIRECTLY from the parent V-FACE field (face-to-face).
    lat_s_edge = float(lat_c[0]) - 0.5 * dlat_c
    vface_lat = lat_s_edge + np.arange(n_lat_c + 1) * dlat_c
    vface_lat2d, vface_lon2d = np.meshgrid(
        vface_lat, np.asarray(child.lon, dtype=np.float64), indexing="ij",
    )
    interp_vface = _bilinear_weights_from_parent_vface(
        parent, vface_lat2d, vface_lon2d,
    )

    return NestedLatLonGrid(
        parent=parent,
        child=child,
        refinement_ratio=r,
        n_halo=int(n_halo),
        n_relax=int(n_relax),
        interp_centers=interp_centers,
        interp_uface=interp_uface,
        interp_vface=interp_vface,
    )


# ----------------------------------------------------------------------------
# Two-way nesting (child -> parent feedback): intentionally NOT implemented.
#
# A two-way nest additionally RESTRICTS the child interior back onto the parent
# cells it refines (area-weighted average of the r x r child cells, replacing the
# parent cell each parent step), and must reconcile the parent-edge mass flux
# with the child-edge flux to stay conservative (Harris & Lin 2013).  That couples
# the two integrations bidirectionally and needs an exact area-restriction
# operator (the inverse of the prolongation here).  This module delivers a
# CORRECT one-way nest; the two-way feedback is documented as remaining work
# rather than half-implemented.
# ----------------------------------------------------------------------------
