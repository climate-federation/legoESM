"""Conservative, differentiable remap between a REGULAR lat-lon grid and a
CURVILINEAR structured C-grid (the eORCA *tripole* ocean, including its bipolar
Arctic cap), via fine-quadrature first-order conservative overlap weights.

This is the Phase-2 cross-grid coupling for a regular lat-lon ATMOSPHERE and a
tripole OCEAN.  The separable lat-lon overlap kernel
(``conservative_regrid.compute_overlap_weights``) factorises into 1-D lat-edge x
1-D lon-edge overlaps and so REQUIRES both grids to be rectilinear; a tripole
grid is rectilinear only below its bipolar cap, so that kernel cannot represent
the cap.  This module instead uses the SAME fine-quadrature first-order method
as the unstructured (MPAS) path (``conservative_regrid_unstructured``), which
handles arbitrary curvilinear geometry.

KEY SIMPLIFICATION — only the REGULAR grid is ever tiled.  In BOTH directions we
sub-divide the regular lat-lon grid into spherical sub-triangles (a lat-lon cell
is a sphere quad; trivial to tile) and locate each sub-triangle's centroid in
the tripole grid by NEAREST CELL CENTRE (a ``cKDTree`` on the tripole T-points).
We therefore never need the tripole cell CORNERS — only its centres (``lat_T`` /
``lon_T``) and the connectivity-free nearest-centre point location (first-order,
the same locator accuracy class as the Voronoi path).

* ``regular -> tripole`` (atm flux -> ocean): tile the regular SOURCE; each
  sub-area is assigned to the tripole destination cell owning its centroid.
  When the tripole ocean is FINER than the (coarse) atmosphere — the realistic
  coupled case — tiling the coarse source would miss many fine tripole cells, so
  a COVERAGE FALLBACK additionally samples every missed cell from the regular
  cell containing its centre (a first-order coarse->fine assignment); every
  tripole cell therefore receives flux.
* ``tripole -> regular`` (ocean SST/currents -> atm): tile the regular TARGET;
  each sub-area is assigned to the tripole source cell owning its centroid.  The
  regular target is always fully tiled, so every atm cell averages the (finer)
  tripole cells beneath it — no coverage gap in this direction.

Both directions normalise by the QUADRATURE destination-cell area (the sum of
the sub-areas assigned to it), so the weights sum to EXACTLY 1 over any covered
destination cell — constant fields are preserved exactly (no spurious flux
gradients), the right property for both a flux density and a state field.  The
GLOBAL INTEGRAL is conserved to FIRST ORDER: for the flux direction the residual
is the small, zero-mean difference between the nearest-centre Voronoi area and
the true ``area_T`` the ocean budget integrates over (exactly zero for a uniform
flux); for the state direction the regular target is tiled exactly so its
quadrature area IS its true area.  Machine-exact global conservation would
require the true tripole cell-corner polygons (exact spherical-polygon clip) and
is deferred; see ``make_regular_to_curvilinear_weights`` for why true-``area_T``
normalisation is NOT used (it blows up on the eORCA ``min_dx_m``-clamped cells).
The per-pair overlap is first-order in the sub-cell size, identical to the MPAS
path — convergent in ``n_sub``.

The output is the SAME :class:`~legoesm.grids.conservative_regrid.
ConservativeRegridWeights` triple as the lat-lon and MPAS paths, so the
differentiable apply kernel (``apply_conservative_regrid`` — a ``segment_sum``
linear in the field values) and the coupler dispatch are reused verbatim.  No
new traced numerics: the weights are static host-side constants and gradients
flow w.r.t. the field exactly as for lat-lon.

Vector (u, v) fields must be rotated to the SHARED geographic basis BEFORE being
remapped as two independent scalars (see
``coupler.grid_remap.rotate_tpoint_currents_to_geographic`` for the ocean->atm
current rotation; wind stress is rotated inside the ocean core).  This module is
scalar-only by construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.grids.conservative_regrid import (
    ConservativeRegridWeights,
    cell_edges_1d,
)
from legoesm.grids.conservative_regrid_unstructured import (
    triangle_subcells,
    unit_vector,
)
from scipy.spatial import cKDTree

# Default per-cell sub-division.  ``n_sub**2`` sub-triangles per fan triangle
# (2 fan triangles per lat-lon quad).  6 -> 72 sub-triangles/regular-cell, a
# good accuracy/cost balance for O(1deg-5deg) coupling (matches the MPAS path's
# n_sub=8 order of magnitude; tripole cells are ~1deg).
_DEFAULT_N_SUB = 6


def _lonlat_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) in radians -> unit-sphere xyz, last axis size 3."""
    cl = np.cos(lat)
    return unit_vector(np.stack(
        [cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1))


def _tripole_centres_xyz(grid) -> np.ndarray:
    """Flattened (n_cells, 3) unit-sphere xyz of the tripole T-point centres."""
    lat = np.asarray(grid.lat_T, dtype=np.float64)
    lon = np.asarray(grid.lon_T, dtype=np.float64)
    return _lonlat_to_xyz(lat, lon).reshape(-1, 3)


def _tile_regular_grid(lat_v, lon, n_sub):
    """Tile a regular lat-lon grid into spherical sub-triangles.

    Parameters
    ----------
    lat_v : (n_lat+1,) latitude edges (pole-clamped v-faces), radians.
    lon   : (n_lon,) cell-centre longitudes, radians.
    n_sub : per-triangle sub-division.

    Returns
    -------
    centroids : (M, 3) unit-sphere centroids of every sub-triangle.
    areas     : (M,) exact unit-sphere sub-triangle areas.
    cell_flat : (M,) flat row-major index ``j*n_lon + i`` of the regular cell
                each sub-triangle belongs to.
    shape     : (n_lat, n_lon).
    """
    lat_v = np.asarray(lat_v, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    n_lat = len(lat_v) - 1
    lon_e = cell_edges_1d(lon, periodic_lon=True)   # (n_lon+1,), last = first+2pi
    n_lon = len(lon_e) - 1

    cents_all = []
    areas_all = []
    cell_all = []
    for j in range(n_lat):
        la0, la1 = float(lat_v[j]), float(lat_v[j + 1])
        for i in range(n_lon):
            lo0, lo1 = float(lon_e[i]), float(lon_e[i + 1])
            # Four CCW corners of the lat-lon quad on the unit sphere.
            c00 = _lonlat_to_xyz(np.array(la0), np.array(lo0))
            c01 = _lonlat_to_xyz(np.array(la0), np.array(lo1))
            c11 = _lonlat_to_xyz(np.array(la1), np.array(lo1))
            c10 = _lonlat_to_xyz(np.array(la1), np.array(lo0))
            # Fan-triangulate the quad from c00: (c00,c01,c11) + (c00,c11,c10).
            flat = j * n_lon + i
            for (a, b, c) in ((c00, c01, c11), (c00, c11, c10)):
                cq, aq = triangle_subcells(a, b, c, n_sub)
                cents_all.append(cq)
                areas_all.append(aq)
                cell_all.append(np.full(len(aq), flat, dtype=np.int64))
    centroids = np.concatenate(cents_all, axis=0)
    areas = np.concatenate(areas_all, axis=0)
    cell_flat = np.concatenate(cell_all, axis=0)
    return centroids, areas, cell_flat, (n_lat, n_lon)


def _locate_in_regular(lat, lon, reg_grid):
    """Flat row-major index of the regular-grid cell containing each (lat, lon).

    Used for the coarse-source coverage fallback: when the tripole is FINER than
    the regular source, some tripole cells are missed by tiling the source, so
    each such cell SAMPLES the regular cell containing its centre (a first-order
    coarse->fine assignment that guarantees full coverage)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_v = np.asarray(reg_grid.lat_v, dtype=np.float64)            # (n_lat+1,)
    lon_e = cell_edges_1d(np.asarray(reg_grid.lon, dtype=np.float64),
                          periodic_lon=True)                        # (n_lon+1,)
    n_lat = len(lat_v) - 1
    n_lon = len(lon_e) - 1
    j = np.clip(np.searchsorted(lat_v, lat) - 1, 0, n_lat - 1)
    # Wrap longitude into the [lon_e[0], lon_e[0]+2pi) period before searching.
    lon_w = np.mod(lon - lon_e[0], 2.0 * np.pi) + lon_e[0]
    i = np.clip(np.searchsorted(lon_e, lon_w) - 1, 0, n_lon - 1)
    return (j * n_lon + i).astype(np.int64)


def _overlap_regular_tripole(reg_grid, trip_grid, n_sub):
    """Core overlap accumulation: tile ``reg_grid`` into sub-triangles and
    assign each to the nearest-centre tripole cell.

    Returns
    -------
    overlap : dict[(trip_cell, reg_cell)] -> overlap area (unit sphere).
    reg_shape : (n_lat, n_lon) of the regular grid.
    trip_shape : (n_lat, n_lon) of the tripole grid.
    """
    centroids, areas, reg_cell, reg_shape = _tile_regular_grid(
        reg_grid.lat_v, reg_grid.lon, n_sub)
    trip_xyz = _tripole_centres_xyz(trip_grid)
    trip_shape = (int(trip_grid.n_lat), int(trip_grid.n_lon))
    tree = cKDTree(trip_xyz)
    _, trip_cell = tree.query(centroids, k=1)
    trip_cell = np.asarray(trip_cell).astype(np.int64)

    overlap: dict[tuple[int, int], float] = {}
    for t, r, a in zip(trip_cell, reg_cell, areas):
        key = (int(t), int(r))
        overlap[key] = overlap.get(key, 0.0) + float(a)
    return overlap, reg_shape, trip_shape


def _assemble_weights(pairs, dst_area, src_shape, dst_shape) -> ConservativeRegridWeights:
    """Build a ConservativeRegridWeights from ``(src_flat, dst_flat, area)``
    pairs, normalising each weight by ``dst_area[dst_flat]``."""
    n_pairs = len(pairs)
    src_idx = np.empty(n_pairs, dtype=np.int32)
    dst_idx = np.empty(n_pairs, dtype=np.int32)
    wts = np.empty(n_pairs, dtype=np.float64)
    for p, (s, d, a) in enumerate(pairs):
        src_idx[p] = s
        dst_idx[p] = d
        wts[p] = a / max(dst_area[d], 1e-30)
    return ConservativeRegridWeights(
        src_idx_flat=jnp.asarray(src_idx, dtype=jnp.int32),
        dst_idx_flat=jnp.asarray(dst_idx, dtype=jnp.int32),
        weights=jnp.asarray(wts, dtype=jnp.float64),
        src_shape=src_shape,
        dst_shape=dst_shape,
        n_dst_cells=int(dst_shape[0] * dst_shape[1]),
    )


def make_regular_to_curvilinear_weights(
        reg_grid, trip_grid, *, n_sub: int = _DEFAULT_N_SUB) -> ConservativeRegridWeights:
    """Conservative remap weights ``regular -> tripole`` (atm FLUX -> ocean).

    Normalised by the nearest-centre QUADRATURE destination area (the sum of the
    sub-areas assigned to each tripole cell), so the weights sum to 1 per
    destination cell: a CONSTANT source flux maps to the same constant
    everywhere (no spurious flux gradients).  Global energy / freshwater
    conservation w.r.t. the ocean's own ``area_T`` then holds to FIRST ORDER —
    the residual is the small, zero-mean difference between the nearest-centre
    Voronoi area and the true cell area (it is exactly zero for a uniform flux,
    because the Voronoi areas and the cell areas both tile the whole sphere).

    Why NOT normalise by the true ``area_T`` (which would make the global
    integral machine-exact)?  On a real eORCA mesh ``area_T`` is floored by
    ``create_tripole_grid``'s ``min_dx_m`` clamp at the degenerate fold/halo
    cells; dividing a normal nearest-centre overlap by that tiny clamped area
    produces O(1e4) flux BLOW-UPS.  Partition-of-unity (quadrature-area)
    normalisation is robust to those degenerate cells.  Machine-exact global
    conservation AND exact constant-preservation simultaneously need the true
    tripole cell-corner polygons (exact spherical-polygon clip) — deferred.
    """
    overlap, reg_shape, trip_shape = _overlap_regular_tripole(
        reg_grid, trip_grid, n_sub)
    n_trip = trip_shape[0] * trip_shape[1]
    # COVERAGE FALLBACK (the realistic case: the tripole ocean is FINER than the
    # coarse atmosphere, so tiling the coarse source misses many fine tripole
    # cells).  Any missed cell SAMPLES the regular cell containing its centre (a
    # first-order coarse->fine assignment); the overlap value is arbitrary (a
    # single entry -> weight 1 after partition-of-unity normalisation), so every
    # tripole cell receives flux (no zero-flux cold-spots).  Where the source is
    # finer than the tripole this branch is empty and the quadrature stands.
    covered = np.zeros(n_trip, dtype=bool)
    for (t, _r) in overlap.keys():
        covered[t] = True
    uncovered = np.nonzero(~covered)[0]
    if uncovered.size:
        lat_c = np.asarray(trip_grid.lat_T, dtype=np.float64).reshape(-1)[uncovered]
        lon_c = np.asarray(trip_grid.lon_T, dtype=np.float64).reshape(-1)[uncovered]
        src_cell = _locate_in_regular(lat_c, lon_c, reg_grid)
        for d, s in zip(uncovered.tolist(), src_cell.tolist()):
            overlap[(int(d), int(s))] = 1.0   # single entry -> weight 1
    # Partition-of-unity destination area = sum of assigned sub-areas.
    dst_area = np.zeros(n_trip, dtype=np.float64)
    for (t, _r), a in overlap.items():
        dst_area[t] += a
    # src = regular (reg_cell), dst = tripole (trip_cell).
    pairs = [(r, t, a) for (t, r), a in overlap.items()]
    return _assemble_weights(pairs, dst_area, reg_shape, trip_shape)


def make_curvilinear_to_regular_weights(
        trip_grid, reg_grid, *, n_sub: int = _DEFAULT_N_SUB,
        src_wet=None) -> ConservativeRegridWeights:
    """Conservative remap weights ``tripole -> regular`` (ocean SST/currents ->
    atm).

    Tiles the regular TARGET; each sub-area is assigned to the tripole source
    cell owning its centroid.  Weights are normalised by the quadrature
    target-cell area (sum of the target cell's own sub-areas), so they sum to 1
    per target cell (constant preserved exactly).
    """
    overlap, reg_shape, trip_shape = _overlap_regular_tripole(
        reg_grid, trip_grid, n_sub)
    n_reg = reg_shape[0] * reg_shape[1]
    if src_wet is not None:
        # H4: EXCLUDE ocean-grid LAND source cells -- their fill SST / zero
        # currents must not bleed into coastal atm cells.  Drop every overlap
        # pair whose TRIPOLE SOURCE cell is land (wet <= 0.5); the per-target
        # partition-of-unity normalisation below then renormalises each atm cell
        # over its surviving WET sources, so a coastal atm cell averages ONLY real
        # ocean (a constant ocean field is still preserved exactly).  An atm cell
        # with NO wet overlap (continental interior) keeps ZERO pairs -> its
        # remapped value is 0 (finite, never the land fill); it is gated out
        # downstream by the atm ocean fraction f_ocean == 0, so the 0 is unused.
        # Baking the mask into the STATIC weights keeps the traced apply a pure
        # segment_sum (AD unchanged).
        n_trip = trip_shape[0] * trip_shape[1]
        wet = np.asarray(src_wet, dtype=np.float64).reshape(-1)
        if wet.size != n_trip:
            raise ValueError(
                f"src_wet size {wet.size} != tripole cells {n_trip} "
                f"(shape {trip_shape}); the wet mask must be on the tripole "
                "SOURCE grid, row-major.")
        overlap = {k: a for k, a in overlap.items() if wet[k[0]] > 0.5}
    # Quadrature target-cell (regular) area = sum over all tripole source cells.
    dst_area = np.zeros(n_reg, dtype=np.float64)
    for (_t, r), a in overlap.items():
        dst_area[r] += a
    # src = tripole (trip_cell), dst = regular (reg_cell).
    pairs = [(t, r, a) for (t, r), a in overlap.items()]
    return _assemble_weights(pairs, dst_area, trip_shape, reg_shape)


# Public promotions (CLAUDE.md cross-module private-import ratchet): the
# cubed-sphere conservative regridder reuses the lat-lon tiling + point-location
# quadrature core, so expose public aliases (definitions keep the underscore
# name for in-module callers).
tile_regular_grid = _tile_regular_grid
locate_in_regular = _locate_in_regular
