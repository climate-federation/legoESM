"""Ocean bathymetry loading, coastline processing, and strait enforcement.

Provides a complete workflow for initializing realistic ocean bathymetry
on cubed-sphere, MPAS Voronoi, and Gaussian grids:

1. **Loading**: Reads depth data from ETOPO/GEBCO/custom NetCDF files and
   regrids to the model grid via bilinear interpolation.
2. **Land-sea mask**: Derives binary ocean mask and sub-grid ocean fraction
   from high-resolution bathymetric data.
3. **Strait enforcement**: Widens critical straits (Drake Passage, Gibraltar,
   Indonesian Throughflow, etc.) that would otherwise be closed at coarse
   resolution, using great-circle corridors with configurable minimum width.
4. **Smoothing**: Laplacian smoothing removes 2Δx noise that causes
   pressure-gradient errors in terrain-following coordinates.
5. **Minimum depth**: Shallow cells below a threshold are deepened or
   converted to land to prevent thin-layer instabilities.
6. **Fill values**: Isolated ocean basins disconnected from the main ocean
   are optionally filled to prevent stagnant pools.

References
----------
- ETOPO 2022 Global Relief Model: https://doi.org/10.25921/fd45-gt74
- GEBCO 2023: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
- Adcroft, A. (2013). Representation of topography by porous barriers
  and objective interpolation of topographic data. Ocean Modelling, 67.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


# ============================================================================
# Configuration
# ============================================================================


# Critical straits that must remain open at coarse resolution.
# Each entry: (name, lat, lon, min_width_km, min_depth_m)
CRITICAL_STRAITS = [
    ("Drake Passage",            -60.0, -67.0, 300.0, 3000.0),
    ("Gibraltar",                 36.0,  -5.5,  50.0,  300.0),
    ("Bab el Mandeb",             12.5,  43.3,  40.0,  150.0),
    ("Hormuz",                    26.5,  56.5,  40.0,  100.0),
    ("Malacca",                    2.5, 101.5,  40.0,   50.0),
    ("Indonesian Throughflow",    -3.0, 120.0, 200.0, 1500.0),
    ("Mozambique Channel",       -17.0,  41.0, 200.0, 2500.0),
    ("Denmark Strait",            66.0, -27.0, 150.0,  600.0),
    ("Faroe Bank Channel",        61.5,  -8.5, 100.0,  800.0),
    ("Bering Strait",             65.8,-169.0,  60.0,   40.0),
    ("Torres Strait",            -10.0, 142.0,  50.0,   15.0),
    ("English Channel",           50.5,   1.5,  60.0,   40.0),
    ("Taiwan Strait",             24.0, 119.5,  80.0,   60.0),
    ("Windward Passage",          20.0, -73.5,  60.0, 1500.0),
    ("Florida Strait",            25.5, -79.5, 100.0,  800.0),
    ("Luzon Strait",              20.5, 121.5, 100.0, 2000.0),
]


class BathymetryConfig(NamedTuple):
    """Configuration for ocean bathymetry initialization.

    Parameters
    ----------
    source : str
        Bathymetry source: ``"idealized"`` (flat bottom with latitude mask),
        ``"file"`` (load from NetCDF).
    path : str
        Path to bathymetry NetCDF file (for source="file").
    depth_var : str
        Depth/elevation variable name in NetCDF. Empty string = auto-detect.
    lat_var : str
        Latitude variable name. Empty string = auto-detect.
    lon_var : str
        Longitude variable name. Empty string = auto-detect.
    H_max : float
        Maximum ocean depth [m] for idealized, or depth clamp for realistic.
    H_min : float
        Minimum ocean depth [m]. Cells shallower than this become land.
    land_lat_threshold : float
        For idealized: latitude [deg] above which cells are land.
    smoothing_passes : int
        Number of Laplacian smoothing iterations.
    enforce_straits : bool
        Whether to enforce critical straits at coarse resolution.
    strait_width_factor : float
        Multiply strait minimum widths by this factor (>1 = wider opening).
    fill_isolated_basins : bool
        Whether to fill (make land) isolated ocean basins disconnected
        from the main ocean.
    edge_blend_strength : float
        Cubed-sphere face-edge blending strength [0, 1].
    edge_blend_width : int
        Number of cells from face edges to blend.
    depth_is_negative : bool
        If True, source data has negative values for ocean depth
        (e.g., ETOPO elevation convention). The loader will negate.
    """
    source: str = "idealized"
    path: str = ""
    depth_var: str = ""
    lat_var: str = ""
    lon_var: str = ""
    H_max: float = 5500.0
    H_min: float = 10.0
    land_lat_threshold: float = 80.0
    smoothing_passes: int = 2
    enforce_straits: bool = True
    strait_width_factor: float = 1.0
    fill_isolated_basins: bool = False
    edge_blend_strength: float = 0.2
    edge_blend_width: int = 2
    depth_is_negative: bool = True


# ============================================================================
# NetCDF loading and regridding
# ============================================================================


def _detect_depth_variable(ds) -> tuple[str, str, str]:
    """Auto-detect depth, latitude, and longitude variable names."""
    all_vars = set(ds.data_vars.keys()) | set(ds.coords.keys())

    # Depth/elevation candidates (ETOPO, GEBCO, generic)
    depth_candidates = [
        "z", "elevation", "depth", "topo", "Band1",
        "bedrock_topography", "surface_elevation",
    ]
    depth_var = None
    for c in depth_candidates:
        if c in all_vars:
            depth_var = c
            break
    if depth_var is None:
        data_vars = list(ds.data_vars.keys())
        if data_vars:
            depth_var = data_vars[0]
        else:
            raise KeyError(
                f"Cannot detect depth variable. Available: {sorted(all_vars)}"
            )

    # Latitude
    lat_candidates = ["lat", "latitude", "y", "Y"]
    lat_var = None
    for c in lat_candidates:
        if c in all_vars:
            lat_var = c
            break
    if lat_var is None:
        raise KeyError(
            f"Cannot detect latitude variable. Available: {sorted(all_vars)}"
        )

    # Longitude
    lon_candidates = ["lon", "longitude", "x", "X"]
    lon_var = None
    for c in lon_candidates:
        if c in all_vars:
            lon_var = c
            break
    if lon_var is None:
        raise KeyError(
            f"Cannot detect longitude variable. Available: {sorted(all_vars)}"
        )

    return depth_var, lat_var, lon_var


def _regrid_bathymetry(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    depth_data: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    """Regrid bathymetry from regular lat-lon to target grid points.

    Uses bilinear interpolation with longitude wrapping.

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitude in degrees, sorted ascending.
    lon_src : (nlon_src,)
        Source longitude in degrees, in [0, 360).
    depth_data : (nlat_src, nlon_src)
        Ocean depth [m], positive downward. Land ≤ 0.
    target_lat_deg : array
        Target latitude in degrees (any shape).
    target_lon_deg : array
        Target longitude in degrees (any shape).

    Returns
    -------
    np.ndarray
        Regridded depth, same shape as target_lat_deg.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Wrap longitude for periodic interpolation
    lon_wrapped = np.concatenate([
        lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0
    ])
    depth_wrapped = np.concatenate([
        depth_data[:, -1:], depth_data, depth_data[:, :1]
    ], axis=1)

    interp = RegularGridInterpolator(
        (lat_src, lon_wrapped), depth_wrapped,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    target_shape = target_lat_deg.shape
    points = np.stack([
        target_lat_deg.ravel(),
        target_lon_deg.ravel(),
    ], axis=-1)

    return interp(points).reshape(target_shape)


def _derive_ocean_fraction(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    depth_data: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
    grid_spacing_deg: float,
) -> np.ndarray:
    """Derive ocean fraction by sub-sampling the high-res depth data.

    For each target cell, samples sub-grid points and computes the
    fraction with depth > 0 (ocean).

    Parameters
    ----------
    lat_src, lon_src : 1D arrays
        Source grid in degrees.
    depth_data : (nlat_src, nlon_src)
        Ocean depth [m], positive downward. Land ≤ 0.
    target_lat_deg, target_lon_deg : arrays
        Target grid centers in degrees (any shape).
    grid_spacing_deg : float
        Approximate target grid spacing in degrees.

    Returns
    -------
    np.ndarray
        Ocean fraction [0, 1], same shape as target_lat_deg.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Wrap longitude
    lon_wrapped = np.concatenate([
        lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0
    ])
    depth_wrapped = np.concatenate([
        depth_data[:, -1:], depth_data, depth_data[:, :1]
    ], axis=1)

    interp = RegularGridInterpolator(
        (lat_src, lon_wrapped), depth_wrapped,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    n_sub = max(3, int(np.ceil(grid_spacing_deg / 0.5)))
    offsets = np.linspace(-0.5, 0.5, n_sub) * grid_spacing_deg

    flat_lat = target_lat_deg.ravel()
    flat_lon = target_lon_deg.ravel()
    n_pts = flat_lat.size

    ocean_frac = np.zeros(n_pts, dtype=np.float64)
    for dlat in offsets:
        for dlon in offsets:
            pts = np.stack([flat_lat + dlat, flat_lon + dlon], axis=-1)
            depth = interp(pts)
            ocean_frac += (depth > 0.0).astype(np.float64)

    ocean_frac /= n_sub * n_sub
    return ocean_frac.reshape(target_lat_deg.shape)


# ============================================================================
# Smoothing
# ============================================================================


def _laplacian_smooth_2d(arr: np.ndarray, passes: int, is_cubed: bool) -> np.ndarray:
    """Laplacian smoothing for cubed-sphere (6,n,n) or Gaussian (nlat,nlon) fields.

    Each pass replaces: result = 0.5*original + 0.5*neighbor_average.
    """
    if passes <= 0:
        return arr

    result = arr.copy()
    for _ in range(passes):
        smoothed = result.copy()
        if is_cubed:
            for face in range(6):
                n = result.shape[1]
                for i in range(n):
                    for j in range(n):
                        vals = [result[face, i, j]]
                        if i > 0:
                            vals.append(result[face, i - 1, j])
                        if i < n - 1:
                            vals.append(result[face, i + 1, j])
                        if j > 0:
                            vals.append(result[face, i, j - 1])
                        if j < n - 1:
                            vals.append(result[face, i, j + 1])
                        smoothed[face, i, j] = np.mean(vals)
        else:
            n_lat, n_lon = result.shape
            for i in range(n_lat):
                for j in range(n_lon):
                    vals = [result[i, j]]
                    if i > 0:
                        vals.append(result[i - 1, j])
                    if i < n_lat - 1:
                        vals.append(result[i + 1, j])
                    vals.append(result[i, (j - 1) % n_lon])
                    vals.append(result[i, (j + 1) % n_lon])
                    smoothed[i, j] = np.mean(vals)
        result = 0.5 * arr + 0.5 * smoothed
    return result


def _laplacian_smooth_voronoi(
    arr: np.ndarray,
    cells_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    passes: int,
) -> np.ndarray:
    """Laplacian smoothing on a Voronoi mesh.

    Parameters
    ----------
    arr : (nCells,)
        Field to smooth.
    cells_on_cell : (maxEdges, nCells)
        Neighbor connectivity (0-indexed, -1 for missing).
    n_edges_on_cell : (nCells,)
        Number of neighbors per cell.
    passes : int
        Number of smoothing iterations.
    """
    if passes <= 0:
        return arr

    result = arr.copy()
    nCells = arr.shape[0]
    maxEdges = cells_on_cell.shape[0]

    for _ in range(passes):
        smoothed = result.copy()
        for c in range(nCells):
            total = result[c]
            count = 1
            for e in range(int(n_edges_on_cell[c])):
                nb = int(cells_on_cell[e, c])
                if nb >= 0:
                    total += result[nb]
                    count += 1
            smoothed[c] = total / count
        result = 0.5 * arr + 0.5 * smoothed
    return result


# ============================================================================
# Strait enforcement
# ============================================================================


def _haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Great-circle distance in km between two points (degrees)."""
    R = 6371.0  # Earth radius [km]
    lat1, lon1 = np.radians(lat1_deg), np.radians(lon1_deg)
    lat2, lon2 = np.radians(lat2_deg), np.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def enforce_straits(
    depth: np.ndarray,
    ocean_mask: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    cfg: BathymetryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce that critical straits remain open at coarse resolution.

    For each strait, finds all grid cells within a corridor and ensures
    they are ocean with at least the strait's minimum depth.

    Parameters
    ----------
    depth : array
        Ocean depth [m], positive downward (any shape).
    ocean_mask : array
        1=ocean, 0=land (same shape as depth).
    lat_deg, lon_deg : arrays
        Grid point latitudes/longitudes in degrees (same shape as depth).
    cfg : BathymetryConfig
        Configuration (uses strait_width_factor).

    Returns
    -------
    depth : array
        Updated depth.
    ocean_mask : array
        Updated mask.
    """
    depth = depth.copy()
    ocean_mask = ocean_mask.copy()
    flat_lat = lat_deg.ravel()
    flat_lon = lon_deg.ravel()

    for name, s_lat, s_lon, min_width_km, min_depth_m in CRITICAL_STRAITS:
        radius_km = 0.5 * min_width_km * cfg.strait_width_factor
        dist_km = _haversine_km(flat_lat, flat_lon, s_lat, s_lon)
        in_corridor = dist_km < radius_km
        if not np.any(in_corridor):
            continue

        # Reshape mask back to original shape
        corridor = in_corridor.reshape(depth.shape)
        # Ensure ocean and minimum depth in corridor
        ocean_mask = np.where(corridor, 1.0, ocean_mask)
        depth = np.where(
            corridor & (depth < min_depth_m),
            min_depth_m,
            depth,
        )

    return depth, ocean_mask


# ============================================================================
# Isolated basin fill
# ============================================================================


def _fill_isolated_basins(ocean_mask: np.ndarray) -> np.ndarray:
    """Fill (set to land) isolated ocean basins not connected to the main ocean.

    Uses flood-fill from the largest connected ocean component.
    Works on flattened 1D mask with adjacency derived from the grid.

    For simplicity, this operates on the raw numpy mask and uses
    scipy's label function.

    Parameters
    ----------
    ocean_mask : array
        1=ocean, 0=land (any shape, but typically 2D or cubed-sphere).

    Returns
    -------
    ocean_mask : array
        Updated mask with isolated basins filled.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        return ocean_mask

    original_shape = ocean_mask.shape
    # For cubed-sphere (6, n, n), process each face separately then merge
    # This is approximate — cross-face connectivity is ignored — but
    # sufficient for removing small interior lakes
    if ocean_mask.ndim == 3 and ocean_mask.shape[0] == 6:
        result = ocean_mask.copy()
        for face in range(6):
            result[face] = _fill_isolated_basins_2d(ocean_mask[face])
        return result
    else:
        return _fill_isolated_basins_2d(ocean_mask)


def _fill_isolated_basins_2d(mask_2d: np.ndarray) -> np.ndarray:
    """Fill isolated basins in a 2D ocean mask."""
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask_2d

    labeled, n_features = label(mask_2d > 0.5)
    if n_features <= 1:
        return mask_2d

    # Find largest component
    sizes = np.array([
        np.sum(labeled == i) for i in range(1, n_features + 1)
    ])
    largest = np.argmax(sizes) + 1

    result = mask_2d.copy()
    result[labeled != largest] = 0.0
    return result


# ============================================================================
# Main loading functions
# ============================================================================


def load_bathymetry(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    cfg: BathymetryConfig,
    grid_spacing_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load bathymetry from NetCDF and process for the target grid.

    This is the grid-agnostic core function. It takes target grid
    coordinates in degrees and returns processed depth and ocean mask.

    Parameters
    ----------
    lat_deg : array
        Target grid latitudes in degrees (any shape).
    lon_deg : array
        Target grid longitudes in degrees (same shape as lat_deg).
    cfg : BathymetryConfig
        Bathymetry configuration.
    grid_spacing_deg : float, optional
        Approximate grid spacing in degrees (for sub-grid ocean fraction).
        If None, estimated from the grid.

    Returns
    -------
    depth : np.ndarray
        Ocean depth [m], positive downward. 0 on land. Same shape as lat_deg.
    ocean_mask : np.ndarray
        1=ocean, 0=land. Same shape as lat_deg.
    """
    import xarray as xr

    if not cfg.path:
        raise ValueError(
            "No bathymetry data path provided. Set cfg.path."
        )

    ds = xr.open_dataset(cfg.path)

    # Detect or use configured variable names
    if cfg.depth_var and cfg.lat_var and cfg.lon_var:
        depth_var, lat_var, lon_var = cfg.depth_var, cfg.lat_var, cfg.lon_var
    else:
        depth_var, lat_var, lon_var = _detect_depth_variable(ds)

    lat_src = ds[lat_var].values.astype(np.float64)
    lon_src = ds[lon_var].values.astype(np.float64)
    elev_data = ds[depth_var].values.astype(np.float64)

    # Squeeze extra dimensions
    while elev_data.ndim > 2:
        elev_data = elev_data[0]

    ds.close()

    # Ensure longitude in [0, 360)
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    elev_data = elev_data[:, lon_order]

    # Ensure latitude sorted ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        elev_data = elev_data[::-1, :]

    # Replace NaN with 0 (land)
    elev_data = np.where(np.isnan(elev_data), 0.0, elev_data)

    # Convert elevation to depth (positive downward for ocean)
    if cfg.depth_is_negative:
        # ETOPO convention: ocean floor is negative elevation
        depth_src = np.maximum(-elev_data, 0.0)
    else:
        # Already positive-downward depth
        depth_src = np.maximum(elev_data, 0.0)

    # Ensure target longitudes in [0, 360)
    target_lat = np.asarray(lat_deg, dtype=np.float64)
    target_lon = np.asarray(lon_deg, dtype=np.float64) % 360.0

    # Estimate grid spacing if not provided
    if grid_spacing_deg is None:
        # Rough estimate from total coverage and number of points
        n_pts = max(target_lat.size, 1)
        grid_spacing_deg = max(180.0 / np.sqrt(n_pts), 0.5)

    # Regrid depth to target
    depth = _regrid_bathymetry(lat_src, lon_src, depth_src,
                               target_lat, target_lon)

    # Derive ocean fraction from sub-grid sampling
    ocean_frac = _derive_ocean_fraction(lat_src, lon_src, depth_src,
                                        target_lat, target_lon,
                                        grid_spacing_deg)

    # Binary ocean mask: ocean if fraction > 0.5
    ocean_mask = (ocean_frac > 0.5).astype(np.float64)

    # Apply minimum depth: shallow cells become land
    too_shallow = depth < cfg.H_min
    ocean_mask = np.where(too_shallow, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # Clamp maximum depth
    depth = np.minimum(depth, cfg.H_max)

    # Enforce critical straits
    if cfg.enforce_straits:
        depth, ocean_mask = enforce_straits(
            depth, ocean_mask, target_lat, target_lon, cfg,
        )

    # Fill isolated basins
    if cfg.fill_isolated_basins:
        ocean_mask = _fill_isolated_basins(ocean_mask)
        depth = np.where(ocean_mask > 0.5, depth, 0.0)

    return depth, ocean_mask


# ============================================================================
# Grid-specific initialization
# ============================================================================


def load_bathymetry_cubed_sphere(
    grid,
    cfg: BathymetryConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load realistic bathymetry for cubed-sphere grid.

    Parameters
    ----------
    grid : CubedSphereGrid
        Target grid with lat/lon in radians, shape (6, n, n).
    cfg : BathymetryConfig
        Configuration.

    Returns
    -------
    H_bathy : jnp.ndarray
        Bathymetry depth [m], shape (6, n, n). Positive downward.
        Set to H_max on land for smooth z* Jacobian.
    ocean_mask : jnp.ndarray
        1=ocean, 0=land, shape (6, n, n).
    """
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    grid_spacing = 90.0 / grid.n

    depth, ocean_mask = load_bathymetry(
        lat_deg, lon_deg, cfg, grid_spacing_deg=grid_spacing,
    )

    # Smoothing on cubed-sphere topology
    depth = _laplacian_smooth_2d(depth, cfg.smoothing_passes, is_cubed=True)

    # Re-enforce minimum depth after smoothing
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # Edge blending for cubed-sphere face boundaries
    if cfg.edge_blend_strength > 0:
        try:
            from legoesm.grids.edge_blending import (
                blend_scalar_cube_edges_2d,
            )
            depth_jax = jnp.array(depth)
            depth_jax = blend_scalar_cube_edges_2d(
                depth_jax,
                strength=cfg.edge_blend_strength,
                width=cfg.edge_blend_width,
            )
            depth = np.asarray(depth_jax)
        except ImportError:
            pass

    # For z* Jacobian smoothness: set land depth to H_max
    # (the land_mask prevents actual flow on land cells)
    H_bathy = np.where(ocean_mask > 0.5, depth, cfg.H_max)

    return jnp.array(H_bathy, dtype=jnp.float32), jnp.array(ocean_mask, dtype=jnp.float32)


def load_bathymetry_mpas(
    mesh,
    cfg: BathymetryConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load realistic bathymetry for MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
        Target mesh with latCell/lonCell in radians.
    cfg : BathymetryConfig
        Configuration.

    Returns
    -------
    H_bathy : jnp.ndarray
        Bathymetry depth [m], shape (nCells,). Positive downward.
    ocean_mask : jnp.ndarray
        1=ocean, 0=land, shape (nCells,).
    """
    lat_deg = np.asarray(jnp.degrees(mesh.latCell))
    lon_deg = np.asarray(jnp.degrees(mesh.lonCell))

    # Estimate grid spacing from mean cell area
    mean_area = float(np.mean(np.asarray(mesh.areaCell)))
    R_earth = 6371.0e3  # [m]
    grid_spacing_m = np.sqrt(mean_area)
    grid_spacing_deg = grid_spacing_m / R_earth * 180.0 / np.pi

    depth, ocean_mask = load_bathymetry(
        lat_deg, lon_deg, cfg, grid_spacing_deg=grid_spacing_deg,
    )

    # Smoothing on Voronoi mesh topology
    cells_on_cell = np.asarray(mesh.cellsOnCell)
    n_edges_on_cell = np.asarray(mesh.nEdgesOnCell)
    depth = _laplacian_smooth_voronoi(
        depth, cells_on_cell, n_edges_on_cell, cfg.smoothing_passes,
    )

    # Re-enforce minimum depth after smoothing
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    H_bathy = np.where(ocean_mask > 0.5, depth, 0.0)

    return jnp.array(H_bathy), jnp.array(ocean_mask)


def load_bathymetry_gaussian(
    grid,
    cfg: BathymetryConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load realistic bathymetry for Gaussian (spectral) grid.

    Parameters
    ----------
    grid : GaussianGrid
        Target grid with lat/lon in radians.
    cfg : BathymetryConfig
        Configuration.

    Returns
    -------
    H_bathy : jnp.ndarray
        Bathymetry depth [m], shape (n_lat, n_lon). Positive downward.
        Set to H_max on land.
    ocean_mask : jnp.ndarray
        1=ocean, 0=land, shape (n_lat, n_lon).
    """
    target_lat = np.asarray(grid.lat) * 180.0 / np.pi
    target_lon = np.asarray(grid.lon) * 180.0 / np.pi
    target_lon = target_lon % 360.0
    target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    grid_spacing = 180.0 / grid.n_lat

    depth, ocean_mask = load_bathymetry(
        target_lat_2d, target_lon_2d, cfg, grid_spacing_deg=grid_spacing,
    )

    # Smoothing on Gaussian grid
    depth = _laplacian_smooth_2d(depth, cfg.smoothing_passes, is_cubed=False)

    # Re-enforce minimum depth after smoothing
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    H_bathy = np.where(ocean_mask > 0.5, depth, cfg.H_max)

    return jnp.array(H_bathy), jnp.array(ocean_mask)


# ============================================================================
# Updated initialization functions
# ============================================================================


def init_ocean_bathymetry(
    grid,
    cfg: BathymetryConfig | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize ocean bathymetry and land mask for any grid type.

    Dispatches to the appropriate loader based on grid type and
    configuration source.

    Parameters
    ----------
    grid : CubedSphereGrid, VoronoiMesh, or GaussianGrid
        Target grid.
    cfg : BathymetryConfig, optional
        Configuration. If None, uses idealized defaults.

    Returns
    -------
    H_bathy : jnp.ndarray
        Bathymetry depth [m]. Positive downward.
    ocean_mask : jnp.ndarray
        1=ocean, 0=land.
    """
    if cfg is None:
        cfg = BathymetryConfig()

    if cfg.source == "idealized":
        return _idealized_dispatch(grid, cfg)
    elif cfg.source == "file":
        return _file_dispatch(grid, cfg)
    else:
        raise ValueError(
            f"Unknown bathymetry source: {cfg.source!r}. "
            f"Use 'idealized' or 'file'."
        )


def _idealized_dispatch(grid, cfg: BathymetryConfig):
    """Dispatch idealized bathymetry to appropriate grid handler."""
    # CubedSphereGrid: has attribute 'n'
    if hasattr(grid, 'n') and hasattr(grid, 'lat') and not hasattr(grid, 'nCells'):
        from legoesm.ocean.init import idealized_bathymetry
        return idealized_bathymetry(grid, cfg.H_max, cfg.land_lat_threshold)

    # VoronoiMesh: has attribute 'nCells'
    if hasattr(grid, 'nCells') and hasattr(grid, 'latCell'):
        from legoesm.ocean.init_mpas import idealized_bathymetry_mpas
        return idealized_bathymetry_mpas(grid, cfg.H_max, cfg.land_lat_threshold)

    # GaussianGrid: has attribute 'n_lat'
    if hasattr(grid, 'n_lat') and not hasattr(grid, 'n'):
        lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
        ocean_mask = jnp.where(lat_deg < cfg.land_lat_threshold, 1.0, 0.0)
        H_bathy = jnp.where(ocean_mask > 0.5, cfg.H_max, 1.0)
        return H_bathy, ocean_mask

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def _file_dispatch(grid, cfg: BathymetryConfig):
    """Dispatch file-based bathymetry to appropriate grid handler."""
    # CubedSphereGrid
    if hasattr(grid, 'n') and hasattr(grid, 'lat') and not hasattr(grid, 'nCells'):
        return load_bathymetry_cubed_sphere(grid, cfg)

    # VoronoiMesh
    if hasattr(grid, 'nCells') and hasattr(grid, 'latCell'):
        return load_bathymetry_mpas(grid, cfg)

    # GaussianGrid
    if hasattr(grid, 'n_lat') and not hasattr(grid, 'n'):
        return load_bathymetry_gaussian(grid, cfg)

    raise TypeError(f"Unsupported grid type: {type(grid)}")


# ============================================================================
# Full ocean state initialization with realistic bathymetry
# ============================================================================


def rest_state_ocean_realistic(
    grid,
    z_coord,
    cfg: BathymetryConfig,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
):
    """Create rest-state ocean initial condition with realistic bathymetry.

    Works for CubedSphereGrid, VoronoiMesh, and GaussianGrid.
    Dispatches to the appropriate state constructor.

    Parameters
    ----------
    grid : CubedSphereGrid, VoronoiMesh, or GaussianGrid
    z_coord : OceanZStarCoordinate
    cfg : BathymetryConfig
    T_surface : float
        Surface temperature [degC].
    T_deep : float
        Deep ocean temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].

    Returns
    -------
    State : OceanState, MPASOceanState, or SpectralOceanState
    """
    H_bathy, ocean_mask = init_ocean_bathymetry(grid, cfg)

    # CubedSphereGrid
    if hasattr(grid, 'n') and hasattr(grid, 'lat') and not hasattr(grid, 'nCells'):
        return _rest_state_cubed(grid, z_coord, H_bathy, ocean_mask,
                                 T_surface, T_deep, S_uniform)

    # VoronoiMesh
    if hasattr(grid, 'nCells') and hasattr(grid, 'latCell'):
        return _rest_state_mpas(grid, z_coord, H_bathy, ocean_mask,
                                T_surface, T_deep, S_uniform)

    # GaussianGrid
    if hasattr(grid, 'n_lat') and not hasattr(grid, 'n'):
        return _rest_state_spectral(grid, z_coord, H_bathy, ocean_mask,
                                    T_surface, T_deep, S_uniform)

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def _rest_state_cubed(grid, z_coord, H_bathy, ocean_mask,
                      T_surface, T_deep, S_uniform):
    """Create cubed-sphere ocean rest state with given bathymetry."""
    from legoesm.core.field import Field
    from legoesm.ocean.state import OceanState

    n = grid.n
    nlev = z_coord.n_levels

    scale_depth = 1000.0
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
        (6, n, n, nlev),
    ).astype(jnp.float32)

    S_3d = jnp.full((6, n, n, nlev), S_uniform, dtype=jnp.float32)
    zeros_3d = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    zeros_2d = jnp.zeros((6, n, n), dtype=jnp.float32)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return OceanState(
        u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=ocean_mask, name="land_mask", dims=dims_2d, units=""),
    )


def _rest_state_mpas(mesh, z_coord, H_bathy, ocean_mask,
                     T_surface, T_deep, S_uniform):
    """Create MPAS ocean rest state with given bathymetry."""
    from legoesm.core.field import Field
    from legoesm.core.state import MPASOceanState

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = z_coord.n_levels

    scale_depth = 1000.0
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
    T_data = jnp.broadcast_to(T_profile[jnp.newaxis, :], (nCells, nlev))

    S_data = jnp.full((nCells, nlev), S_uniform)
    u_data = jnp.zeros((nEdges, nlev))
    eta_data = jnp.zeros(nCells)

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def _rest_state_spectral(grid, z_coord, H_bathy, ocean_mask,
                         T_surface, T_deep, S_uniform):
    """Create spectral ocean rest state with given bathymetry."""
    from legoesm.ocean.dynamics.spectral_ocean_pe import (
        rest_state_spectral_ocean,
    )
    # For spectral ocean, the standard rest_state function handles
    # the spectral transform. We provide bathymetry via a custom path.
    # For now, delegate to the existing function and override H_bathy/mask.
    # The existing function uses lat threshold — we just need to ensure
    # the state gets the correct bathymetry.
    #
    # TODO: extend rest_state_spectral_ocean to accept external H_bathy/mask
    return rest_state_spectral_ocean(
        grid, z_coord,
        T_surface=T_surface,
        T_deep=T_deep,
        S_uniform=S_uniform,
        H_max=float(jnp.max(H_bathy)),
    )
