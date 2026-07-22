"""First-order conservative area-overlap remap between a regular lat-lon grid and
a cubed-sphere grid — the cross-family coupling the ESM energy/freshwater budgets
need (a cube/spectral atmosphere driving a lat-lon 3-D ocean).

Method (reuses the tripole/MPAS quadrature core, distribution-agnostic): tile the
REGULAR lat-lon grid into spherical sub-triangles with exact unit-sphere areas
(``_tile_regular_grid``) and assign each sub-triangle to the NEAREST cube cell
centre (KD-tree on the cube's stored unit-sphere centres — no assumption about
the cube's gnomonic distribution, and no cube-corner reconstruction).  The single
overlap-area matrix ``A[cube_cell, latlon_cell]`` this produces serves BOTH
directions with the appropriate destination-area normalisation:

* ``latlon -> cube``: ``w = A / area(cube)`` — partition of unity per cube cell
  (a missed cube cell, when the cube is finer than the source, samples the
  lat-lon cell containing its centre — the same coverage fallback the tripole
  builder uses, so no cold spots).
* ``cube -> latlon``: ``w = A / area(latlon)`` — EXACTLY partition of unity per
  lat-lon cell (tiling a lat-lon cell fully covers it), so a constant cube field
  maps to the same constant and the global area integral is preserved to
  quadrature order.

Both emit the SAME ``ConservativeRegridWeights`` triple ``apply_conservative_
regrid`` already consumes for cube ``(6, n, n)`` fields, so the coupler's
``remap_field`` / ``remap_surface_fields`` need no change.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.spatial import cKDTree

from legoesm.grids.conservative_regrid import ConservativeRegridWeights
from legoesm.grids.conservative_regrid_curvilinear import (
    tile_regular_grid as _tile_regular_grid,
    locate_in_regular as _locate_in_regular,
)

# Sub-division per lat-lon quad triangle (2 fan triangles/cell -> 2*n_sub^2
# sub-triangles).  Matches the tripole/MPAS default for a comparable quadrature.
_DEFAULT_N_SUB = 6


def _cube_centres_xyz(cube_grid) -> np.ndarray:
    """``(6*n*n, 3)`` unit-sphere cube cell centres, row-major over ``(6, n, n)``.

    Uses the grid's stored Cartesian centres when present (any gnomonic
    distribution), else reconstructs them from the cell lat/lon.
    """
    xc = getattr(cube_grid, "x_cart", None)
    if xc is not None:
        x = np.asarray(cube_grid.x_cart, dtype=np.float64).reshape(-1)
        y = np.asarray(cube_grid.y_cart, dtype=np.float64).reshape(-1)
        z = np.asarray(cube_grid.z_cart, dtype=np.float64).reshape(-1)
    else:
        lat = np.asarray(cube_grid.grid_lat, dtype=np.float64).reshape(-1)
        lon = np.asarray(cube_grid.grid_lon, dtype=np.float64).reshape(-1)
        cl = np.cos(lat)
        x, y, z = cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)
    out = np.stack([x, y, z], axis=-1)
    # Normalise to the unit sphere (guard against tiny metric drift).
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def _cube_shape(cube_grid) -> tuple[int, int, int]:
    s = tuple(int(v) for v in cube_grid.grid_shape_2d)
    if len(s) != 3 or s[0] != 6:
        raise ValueError(
            f"expected a cubed-sphere grid_shape_2d (6, n, n); got {s}")
    return s


def _overlap_latlon_cube(reg_grid, cube_grid, n_sub):
    """BASE overlap-area matrix by tiling the regular grid + nearest-cube
    assignment — NO coverage fallback (that belongs to the latlon->cube
    direction only; adding it to the shared matrix would pollute the
    cube->latlon destination areas with the fallback's synthetic entries).

    Returns ``(overlap, reg_shape, cube_shape)`` where ``overlap`` is
    ``dict[(cube_flat, reg_flat)] -> unit-sphere overlap area``.
    """
    centroids, areas, reg_cell, reg_shape = _tile_regular_grid(
        reg_grid.lat_v, reg_grid.lon, n_sub)
    cube_shape = _cube_shape(cube_grid)
    cube_xyz = _cube_centres_xyz(cube_grid)
    tree = cKDTree(cube_xyz)
    _, cube_cell = tree.query(centroids, k=1)
    cube_cell = np.asarray(cube_cell).astype(np.int64)

    overlap: dict[tuple[int, int], float] = {}
    for c, r, a in zip(cube_cell, reg_cell, areas):
        key = (int(c), int(r))
        overlap[key] = overlap.get(key, 0.0) + float(a)
    return overlap, reg_shape, cube_shape


def _assemble(pairs, dst_area, src_shape, dst_shape) -> ConservativeRegridWeights:
    """``(src_flat, dst_flat, area)`` -> ConservativeRegridWeights, weight =
    area / dst_area[dst_flat].  Handles an N-D ``dst_shape`` (cube ``(6, n, n)``)
    unlike the curvilinear module's 2-D-only assembler."""
    n = len(pairs)
    src_idx = np.empty(n, dtype=np.int32)
    dst_idx = np.empty(n, dtype=np.int32)
    wts = np.empty(n, dtype=np.float64)
    for p, (s, d, a) in enumerate(pairs):
        src_idx[p] = s
        dst_idx[p] = d
        wts[p] = a / max(dst_area[d], 1e-30)
    return ConservativeRegridWeights(
        src_idx_flat=jnp.asarray(src_idx, dtype=jnp.int32),
        dst_idx_flat=jnp.asarray(dst_idx, dtype=jnp.int32),
        weights=jnp.asarray(wts, dtype=jnp.float64),
        src_shape=tuple(int(v) for v in src_shape),
        dst_shape=tuple(int(v) for v in dst_shape),
        n_dst_cells=int(np.prod(dst_shape)),
    )


def make_latlon_to_cube_weights(
        reg_grid, cube_grid, *, n_sub: int = _DEFAULT_N_SUB
) -> ConservativeRegridWeights:
    """Conservative remap weights ``regular lat-lon -> cubed-sphere`` (atm FLUX ->
    ocean when the ATM is lat-lon and the ocean is a cube — or, transposed, the
    ocean->atm direction for a cube atm; see make_grid_remapper wiring)."""
    overlap, reg_shape, cube_shape = _overlap_latlon_cube(reg_grid, cube_grid, n_sub)
    overlap = dict(overlap)  # own copy — the fallback is direction-specific
    n_cube = cube_shape[0] * cube_shape[1] * cube_shape[2]
    # Coverage fallback (cube FINER than the source lat-lon): a cube cell that
    # received no lat-lon sub-triangle samples the lat-lon cell containing its
    # centre — a first-order coarse->fine assignment so every cube cell gets a
    # value (no zero-flux cold spots).  This is a latlon->cube-ONLY correction
    # (mirrors make_regular_to_curvilinear_weights); it must NOT enter the
    # cube->latlon normalisation, hence the separate dict.
    covered = np.zeros(n_cube, dtype=bool)
    for (c, _r) in overlap.keys():
        covered[c] = True
    missed = np.nonzero(~covered)[0]
    if missed.size:
        lat_c = np.asarray(cube_grid.grid_lat, dtype=np.float64).reshape(-1)[missed]
        lon_c = np.asarray(cube_grid.grid_lon, dtype=np.float64).reshape(-1)[missed]
        src_cell = _locate_in_regular(lat_c, lon_c, reg_grid)
        for c, r in zip(missed.tolist(), src_cell.tolist()):
            overlap[(int(c), int(r))] = 1.0  # single entry -> weight 1
    # Partition-of-unity destination (cube) area = sum of assigned sub-areas.
    dst_area = np.zeros(n_cube, dtype=np.float64)
    for (c, _r), a in overlap.items():
        dst_area[c] += a
    pairs = [(r, c, a) for (c, r), a in overlap.items()]  # src=reg, dst=cube
    return _assemble(pairs, dst_area, reg_shape, cube_shape)


def make_cube_to_latlon_weights(
        cube_grid, reg_grid, *, n_sub: int = _DEFAULT_N_SUB
) -> ConservativeRegridWeights:
    """Conservative remap weights ``cubed-sphere -> regular lat-lon``.

    Uses the SAME overlap matrix (tiling the lat-lon grid) normalised by the
    LAT-LON destination area — exactly partition-of-unity per lat-lon cell."""
    overlap, reg_shape, _cube_shape_ = _overlap_latlon_cube(reg_grid, cube_grid, n_sub)
    n_reg = reg_shape[0] * reg_shape[1]
    dst_area = np.zeros(n_reg, dtype=np.float64)
    for (_c, r), a in overlap.items():
        dst_area[r] += a
    pairs = [(c, r, a) for (c, r), a in overlap.items()]  # src=cube, dst=reg
    return _assemble(pairs, dst_area, _cube_shape_, reg_shape)
