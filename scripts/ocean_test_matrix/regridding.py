"""Regridding and interpolation utilities for the ocean test matrix.

Functions for interpolating unstructured-grid ocean model output onto
regular lat-lon grids for visualization and diagnostics, plus native
Voronoi polygon plotting for MPAS meshes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _build_latlon_weights(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    n_lat: int = 181,
    n_lon: int = 360,
    k: int = 6,
    max_dist: float | None = None,
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build KDTree interpolation weights from unstructured to lat-lon grid.

    Returns (idxs, weights) arrays of shape (n_lat*n_lon, K) for K-nearest-
    neighbor inverse-distance weighting in 3-D Cartesian coordinates.

    Parameters
    ----------
    max_dist : float, optional
        Maximum 3-D Cartesian distance (on the unit sphere) for a valid
        neighbour.  Target points whose nearest source cell is farther
        than this get zero weight and will produce NaN after
        ``_apply_weights``.  Prevents extrapolation artefacts in
        regional meshes.
    ocean_mask : array, optional
        Boolean-like mask where >0.5 means ocean.  When provided, only
        ocean cells are included in the KDTree so land values never
        contaminate interpolated ocean fields.  Returned ``idxs`` refer
        to the *original* (unmasked) array so ``_apply_weights`` works
        unchanged.
    """
    lon = np.asarray(lon_deg, dtype=np.float64).ravel() % 360
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64).ravel(), -90, 90)
    d2r = np.pi / 180.0
    src_all = np.column_stack([
        np.cos(lat * d2r) * np.cos(lon * d2r),
        np.cos(lat * d2r) * np.sin(lon * d2r),
        np.sin(lat * d2r)])

    # When an ocean mask is provided, build the tree from ocean cells only
    # but map indices back to the full array for _apply_weights.
    if ocean_mask is not None:
        omask = np.asarray(ocean_mask, dtype=np.float64).ravel() > 0.5
        ocean_idx = np.where(omask)[0]
        if ocean_idx.size == 0:
            # All land — return zero weights
            n_tgt = n_lat * n_lon
            return (np.zeros((n_tgt, 1), dtype=int),
                    np.zeros((n_tgt, 1), dtype=np.float64))
        src = src_all[ocean_idx]
    else:
        ocean_idx = None
        src = src_all

    # Use target grid if provided
    if target_lat is not None and target_lon is not None:
        lat_1d = target_lat
        lon_1d = target_lon
        n_lat, n_lon = len(lat_1d), len(lon_1d)
    else:
        lat_1d = np.linspace(-90.0, 90.0, n_lat)
        lon_1d = np.linspace(0.0, 360.0, n_lon)
    lo, la = np.meshgrid(lon_1d, lat_1d)
    tgt = np.column_stack([
        np.cos(la.ravel() * d2r) * np.cos(lo.ravel() * d2r),
        np.cos(la.ravel() * d2r) * np.sin(lo.ravel() * d2r),
        np.sin(la.ravel() * d2r)])
    tree = cKDTree(src)
    K = min(k, src.shape[0])
    dists, idxs = tree.query(tgt, k=K)
    if K == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # Map indices back to the full (unmasked) array
    if ocean_idx is not None:
        idxs = ocean_idx[idxs]

    w = 1.0 / np.maximum(dists, 1e-12)
    # Zero out weights for target points too far from any source cell.
    if max_dist is not None:
        too_far = dists[:, 0] > max_dist
        w[too_far] = 0.0
    w_sum = w.sum(axis=1, keepdims=True)
    w = np.where(w_sum > 0, w / np.maximum(w_sum, 1e-30), 0.0)
    return idxs, w


def _apply_weights(vals: np.ndarray, idxs: np.ndarray, w: np.ndarray,
                   n_lat: int, n_lon: int) -> np.ndarray:
    """Apply precomputed IDW weights, handling NaN source values."""
    v = vals[idxs]
    v_valid = np.isfinite(v)
    v_safe = np.where(v_valid, v, 0.0)
    wm = w * v_valid
    ws = wm.sum(axis=1, keepdims=True)
    wn = np.where(ws > 0, wm / np.maximum(ws, 1e-30), 0)
    result = np.sum(v_safe * wn, axis=1)
    return np.where(ws.ravel() > 0, result, np.nan).reshape(n_lat, n_lon)


def _build_voronoi_polygons(mesh) -> tuple[list, np.ndarray]:
    """Pre-compute Voronoi cell polygons from an MPAS VoronoiMesh.

    Returns
    -------
    polygons : list of (nv, 2) arrays
        Each polygon is an array of (lon_deg, lat_deg) vertices.
    cell_indices : int array, shape (n_polygons,)
        The cell index for each polygon (some cells may be skipped
        if vertex connectivity is invalid).
    """
    lat_v = np.degrees(np.asarray(mesh.latVertex, dtype=np.float64))
    lon_v = np.degrees(np.asarray(mesh.lonVertex, dtype=np.float64))
    verts_on_cell = np.asarray(mesh.verticesOnCell, dtype=int)  # (maxEdges, nCells)
    n_edges = np.asarray(mesh.nEdgesOnCell, dtype=int)          # (nCells,)

    polygons = []
    cell_indices = []
    for i in range(mesh.nCells):
        nv = int(n_edges[i])
        if nv < 3:
            continue
        vidx = verts_on_cell[:nv, i]
        if np.any(vidx < 0):
            continue
        poly = np.column_stack([lon_v[vidx], lat_v[vidx]])
        polygons.append(poly)
        cell_indices.append(i)
    return polygons, np.array(cell_indices, dtype=int)


def _plot_voronoi_field(ax, mesh, field: np.ndarray,
                        land_mask: np.ndarray | None = None,
                        cmap: str = "RdYlBu_r",
                        vmin: float | None = None,
                        vmax: float | None = None):
    """Plot a cell-centered field on the native Voronoi mesh.

    Uses matplotlib PolyCollection — no interpolation, so there are
    zero land-bleed artifacts.

    Parameters
    ----------
    ax : matplotlib Axes
    mesh : VoronoiMesh
    field : array, shape (nCells,)
    land_mask : array, shape (nCells,), optional
        >0.5 means ocean.  Land cells drawn in light gray.
    cmap, vmin, vmax : colormap parameters for ocean cells.

    Returns
    -------
    pc : PolyCollection for the ocean cells (for colorbar).
    """
    from matplotlib.collections import PolyCollection

    polygons, cell_idx = _build_voronoi_polygons(mesh)
    values = np.asarray(field, dtype=np.float64).ravel()

    if land_mask is not None:
        mask = np.asarray(land_mask, dtype=np.float64).ravel()
        ocean_polys, ocean_vals = [], []
        land_polys = []
        for poly, ci in zip(polygons, cell_idx):
            if mask[ci] > 0.5:
                ocean_polys.append(poly)
                ocean_vals.append(values[ci])
            else:
                land_polys.append(poly)
        # Draw land cells
        if land_polys:
            land_pc = PolyCollection(land_polys, facecolor="#d9d9d9",
                                     edgecolor="#bfbfbf", linewidth=0.3)
            ax.add_collection(land_pc)
        # Draw ocean cells
        pc = None
        if ocean_polys:
            pc = PolyCollection(ocean_polys, array=np.array(ocean_vals),
                                cmap=cmap, edgecolor="face", linewidth=0.1)
            if vmin is not None and vmax is not None:
                pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
    else:
        vals_arr = values[cell_idx]
        pc = PolyCollection(polygons, array=vals_arr, cmap=cmap,
                            edgecolor="face", linewidth=0.1)
        if vmin is not None and vmax is not None:
            pc.set_clim(vmin, vmax)
        ax.add_collection(pc)

    ax.autoscale_view()
    ax.set_aspect("equal")
    return pc


def _bin_to_latlon(
    values: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    n_lat: int = 181,
    n_lon: int = 360,
    max_dist: float | None = None,
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate unstructured points onto a regular lat-lon grid.

    Parameters
    ----------
    max_dist : float, optional
        If given, target grid points whose nearest source cell is farther
        than this (3-D Cartesian distance on unit sphere) are set to NaN.
        Automatically estimated for regional meshes when not provided.
    ocean_mask : array, optional
        Boolean-like mask (>0.5 = ocean).  When provided, land cells are
        excluded from the KDTree so they cannot contaminate interpolated
        ocean values.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    # Use target grid if provided, otherwise use default dimensions
    if target_lat is not None and target_lon is not None:
        n_lat, n_lon = len(target_lat), len(target_lon)
    if not np.any(np.isfinite(vals)):
        return np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    # Auto-detect regional mesh: if the source points span < 80% of
    # the globe in latitude, apply a distance cutoff to prevent
    # extrapolation artefacts outside the mesh.
    if max_dist is None:
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lat_span = lat.max() - lat.min()
        if lat_span < 0.8 * 180:
            # Regional: max_dist ≈ 3× median cell spacing (on unit sphere)
            lon = np.asarray(lon_deg, dtype=np.float64).ravel()
            d2r = np.pi / 180.0
            # Rough estimate: sqrt(4π / N) gives mean angular cell spacing
            n_pts = len(lat)
            mean_spacing = np.sqrt(
                d2r**2 * lat_span * min(360, lon.max() - lon.min()) / n_pts)
            # Convert angular spacing to 3-D chord distance
            max_dist = 2.0 * np.sin(0.5 * mean_spacing * 3.0)
    idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon,
                                     max_dist=max_dist,
                                     ocean_mask=ocean_mask)
    return _apply_weights(vals, idxs, w, n_lat, n_lon)


def _regrid_land_mask(mask_arr: np.ndarray, lon_deg: np.ndarray,
                     lat_deg: np.ndarray, coord_kind: str,
                     target_lat: np.ndarray | None = None,
                     target_lon: np.ndarray | None = None) -> np.ndarray:
    """Regrid a binary land mask using nearest neighbor interpolation."""
    if coord_kind in ("latlon", "gaussian"):
        return np.asarray(mask_arr, dtype=np.float64)

    # For unstructured grids, use nearest neighbor interpolation
    # (cKDTree already imported at module level)

    # Source points (unstructured)
    lon_src = np.asarray(lon_deg, dtype=np.float64).ravel() % 360
    lat_src = np.clip(np.asarray(lat_deg, dtype=np.float64).ravel(), -90, 90)
    mask_src = np.asarray(mask_arr, dtype=np.float64).ravel()

    # Target grid
    if target_lat is not None and target_lon is not None:
        n_lat, n_lon = len(target_lat), len(target_lon)
        lat_1d, lon_1d = target_lat, target_lon
    else:
        n_lat, n_lon = 181, 360
        lat_1d = np.linspace(-90.0, 90.0, n_lat)
        lon_1d = np.linspace(0.0, 360.0, n_lon)

    # Convert to 3D Cartesian coordinates for accurate distance calculation
    d2r = np.pi / 180.0

    # Source points in 3D
    src_3d = np.column_stack([
        np.cos(lat_src * d2r) * np.cos(lon_src * d2r),
        np.cos(lat_src * d2r) * np.sin(lon_src * d2r),
        np.sin(lat_src * d2r)
    ])

    # Target points in 3D
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    tgt_3d = np.column_stack([
        np.cos(lat_2d.ravel() * d2r) * np.cos(lon_2d.ravel() * d2r),
        np.cos(lat_2d.ravel() * d2r) * np.sin(lon_2d.ravel() * d2r),
        np.sin(lat_2d.ravel() * d2r)
    ])

    # Build KDTree and find nearest neighbors
    tree = cKDTree(src_3d)
    distances, indices = tree.query(tgt_3d, k=1)

    # Get mask values at nearest neighbors
    mask_interp = mask_src[indices]

    # For land mask, apply threshold to ensure binary values
    mask_interp = np.where(mask_interp > 0.5, 1.0, 0.0)

    return mask_interp.reshape(n_lat, n_lon)


def _regrid_2d(field_arr: np.ndarray, lon_deg: np.ndarray,
               lat_deg: np.ndarray, coord_kind: str,
               target_lat: np.ndarray | None = None,
               target_lon: np.ndarray | None = None,
               ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Regrid a 2D field to target lat-lon grid (default 181x360).

    Parameters
    ----------
    ocean_mask : array, optional
        Boolean-like mask (>0.5 = ocean).  For unstructured grids, land
        cells are excluded from the interpolation KDTree.
    """
    if coord_kind in ("latlon", "gaussian"):
        return np.asarray(field_arr, dtype=np.float64)
    # Cubed-sphere: use face-aware bilinear interpolation (no edge artifacts).
    if coord_kind == "cube":
        from legoesm.grids.regridding import (
            apply_cubedsphere_to_latlon, get_cubedsphere_to_latlon_weights)
        arr = np.asarray(field_arr, dtype=np.float64)
        if arr.ndim >= 3 and arr.shape[0] == 6:
            n = arr.shape[1]
        else:
            n = int(round(np.sqrt(arr.size / 6)))
            arr = arr.reshape(6, n, n)
        w = get_cubedsphere_to_latlon_weights(n)
        return apply_cubedsphere_to_latlon(arr, w)
    return _bin_to_latlon(field_arr.ravel(), lon_deg.ravel(), lat_deg.ravel(),
                          target_lat=target_lat, target_lon=target_lon,
                          ocean_mask=ocean_mask)


def _regrid_3d_level(field_3d: np.ndarray, lon_deg: np.ndarray,
                     lat_deg: np.ndarray, coord_kind: str,
                     target_lat: np.ndarray | None = None,
                     target_lon: np.ndarray | None = None,
                     ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Regrid a 3D field (*, nlev) to target lat-lon grid (default 181x360x nlev)."""
    arr = np.asarray(field_3d, dtype=np.float64)
    if coord_kind in ("latlon", "gaussian"):
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr
    # Cubed-sphere: use face-aware bilinear interpolation.
    if coord_kind == "cube":
        from legoesm.grids.regridding import (
            apply_cubedsphere_to_latlon_3d, get_cubedsphere_to_latlon_weights)
        if arr.ndim >= 3 and arr.shape[0] == 6:
            n = arr.shape[1]
        else:
            nlev = arr.shape[-1]
            n = int(round(np.sqrt(arr.size / (6 * nlev))))
            arr = arr.reshape(6, n, n, nlev)
        w = get_cubedsphere_to_latlon_weights(n)
        return apply_cubedsphere_to_latlon_3d(arr, w)
    if arr.ndim == 1:
        arr = arr[:, None]
    nlev = arr.shape[-1]
    if target_lat is not None and target_lon is not None:
        n_lat, n_lon = len(target_lat), len(target_lon)
    else:
        n_lat, n_lon = 181, 360
    flat = arr.reshape(-1, nlev)
    # Auto-detect regional mesh and apply distance cutoff
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    max_dist = None
    if lat.max() - lat.min() < 0.8 * 180:
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        d2r = np.pi / 180.0
        n_pts = len(lat)
        mean_spacing = np.sqrt(
            d2r**2 * (lat.max() - lat.min())
            * min(360, lon.max() - lon.min()) / n_pts)
        max_dist = 2.0 * np.sin(0.5 * mean_spacing * 3.0)
    idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon,
                                     max_dist=max_dist, target_lat=target_lat,
                                     target_lon=target_lon,
                                     ocean_mask=ocean_mask)
    out = np.full((n_lat, n_lon, nlev), np.nan, dtype=np.float64)
    for k in range(nlev):
        out[..., k] = _apply_weights(flat[:, k], idxs, w, n_lat, n_lon)
    return out


def _fill_nan_profile(profile: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1D profile by linear interpolation along index."""
    prof = np.asarray(profile, dtype=np.float64).copy()
    valid = np.isfinite(prof)
    if not np.any(valid):
        return prof
    if np.count_nonzero(valid) == 1:
        prof[:] = prof[valid][0]
        return prof
    x = np.arange(prof.size, dtype=np.float64)
    prof[:] = np.interp(x, x[valid], prof[valid])
    return prof


def _fill_nan_section(section: np.ndarray) -> np.ndarray:
    """Fill NaNs along the horizontal axis for each vertical level."""
    sec = np.asarray(section, dtype=np.float64).copy()
    if sec.ndim != 2:
        return sec
    for k in range(sec.shape[1]):
        sec[:, k] = _fill_nan_profile(sec[:, k])
    return sec
