"""One-way regional grid NESTING on the latitude-longitude grid.

A *nested* configuration is a coarse PARENT grid plus a refined CHILD grid that
covers a contiguous rectangular sub-region of the parent, related by an integer
``refinement_ratio`` ``r``: every parent cell inside the child footprint is split
into ``r x r`` child cells, so the child cell *centres* are geometrically nested
within the parent (no offset drift, no overlap, no gaps).

This module owns only the GRID GEOMETRY and the parent->child interpolation
OPERATOR — both substrate-level (``legoesm.grids`` / ``legoesm.core``) concerns
with no dependency on any dynamical core.  The 1-way nested *time stepping*
(parent step, child step, boundary forcing) lives with the dynamics that uses it
(``legoesm.atmosphere.dynamics.shallow_water_nesting``), because the substrate
must not import a component (import-linter contract #4).

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
    _build_uniform_latlon_grid_from_axes,
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
    interpolation operators map the parent CELL-CENTRE field to the child cell
    centres (``interp_centers``), the child u-faces (``interp_uface``) and the
    child v-faces (``interp_vface``) so the staggered C-grid boundary can be
    prescribed component-by-component.

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


def _bilinear_weights_to_targets(
    parent: LatLonGrid,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> BoundaryInterpWeights:
    """Bilinear gather weights from parent cell centres to arbitrary targets.

    ``tgt_lat`` / ``tgt_lon`` are 2-D target-point coordinate grids [rad].  The
    parent is a GLOBAL uniform lat-lon grid: longitude is periodic (wraps
    modulo 2*pi at the seam between the last and first parent column), latitude
    is clamped to the parent's first/last cell-centre row (the child footprint
    is validated to lie strictly inside the parent latitude band, so no
    extrapolation past the poles occurs).
    """
    plat = np.asarray(parent.lat, dtype=np.float64)  # (n_lat_p,)
    plon = np.asarray(parent.lon, dtype=np.float64)  # (n_lon_p,)
    n_lat_p = plat.shape[0]
    n_lon_p = plon.shape[0]
    dlat_p = float(parent.dlat)
    dlon_p = float(parent.dlon)
    lon0 = float(plon[0])
    lat0 = float(plat[0])

    target_shape = tuple(int(s) for s in np.asarray(tgt_lat).shape)
    tlat = np.asarray(tgt_lat, dtype=np.float64).reshape(-1)
    tlon = np.asarray(tgt_lon, dtype=np.float64).reshape(-1)

    # --- Latitude: fractional index into parent rows, clamped to interior. ---
    # parent row j sits at lat0 + j*dlat_p (uniform global grid).
    fj = (tlat - lat0) / dlat_p
    fj = np.clip(fj, 0.0, n_lat_p - 1.0 - 1e-12)
    j0 = np.floor(fj).astype(np.int64)
    j0 = np.clip(j0, 0, n_lat_p - 2)
    wj = fj - j0  # in [0, 1)
    j1 = j0 + 1

    # --- Longitude: fractional index into parent columns, PERIODIC wrap. ---
    # Bring (tlon - lon0) into [0, 2*pi) so the seam between column n_lon_p-1
    # and column 0 interpolates across the periodic boundary.
    two_pi = 2.0 * np.pi
    dlon_rel = np.mod(tlon - lon0, two_pi)
    fi = dlon_rel / dlon_p
    i0 = np.floor(fi).astype(np.int64) % n_lon_p
    wi = fi - np.floor(fi)  # in [0, 1)
    i1 = (i0 + 1) % n_lon_p

    # Four corners: SW (j0,i0), SE (j0,i1), NW (j1,i0), NE (j1,i1).
    def flat(j, i):
        return (j * n_lon_p + i).astype(np.int32)

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
        parent_flat_size=int(n_lat_p * n_lon_p),
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

    child = _build_uniform_latlon_grid_from_axes(
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
    # at lon_c west-edge + k*dlon_c, latitude = cell-centre lat.
    lon_w_edge = float(lon_c[0]) - 0.5 * dlon_c
    uface_lon = lon_w_edge + np.arange(n_lon_c + 1) * dlon_c
    uface_lat2d, uface_lon2d = np.meshgrid(
        np.asarray(child.lat, dtype=np.float64), uface_lon, indexing="ij",
    )
    interp_uface = _bilinear_weights_to_targets(parent, uface_lat2d, uface_lon2d)

    # v-face targets: lat interfaces, shape (n_lat_c+1, n_lon_c).  v-face k sits
    # at lat_c south-edge + k*dlat_c, longitude = cell-centre lon.  The +/-
    # half-child-cell extension at the child N/S edges stays well inside the
    # >= 1-parent-row margin enforced above.
    lat_s_edge = float(lat_c[0]) - 0.5 * dlat_c
    vface_lat = lat_s_edge + np.arange(n_lat_c + 1) * dlat_c
    vface_lat2d, vface_lon2d = np.meshgrid(
        vface_lat, np.asarray(child.lon, dtype=np.float64), indexing="ij",
    )
    interp_vface = _bilinear_weights_to_targets(parent, vface_lat2d, vface_lon2d)

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
