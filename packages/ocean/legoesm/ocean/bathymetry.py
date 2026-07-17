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

from legoesm import constants
from legoesm.core.precision import get_policy


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
    r_factor_max: float | None = None
    """Mellor-Ezer-Oey r-factor cap.  When set, iteratively deepens
    shallower ocean cells so that for every ocean-ocean neighbour pair
    ``r = |H_i - H_j| / max(H_i, H_j) <= r_factor_max``.  Typical
    target: 0.2 (the Beckmann-Haidvogel literature bound).  When None,
    no MEO smoothing is applied — the model relies on Laplacian
    smoothing alone, which is known to be insufficient for sharp
    bathymetric features (Phase 3a finding)."""
    meo_max_iter: int = 200
    north_cap_lat: float | None = None
    """Northern polar cap latitude [deg].  When set, all ocean cells
    with ``lat > north_cap_lat`` are converted to land.  Matches the
    ``polar_cap_lat`` convention of the idealized GO config (default
    80°).  Standard production fix for lat-lon ocean models that the
    Arctic singular-point + tiny-dx high-latitude regime is hard to
    keep stable + damped simultaneously, especially with cos²(lat)
    A_h scaling that *reduces* damping at high latitudes.  See
    ``docs/ocean/experiments/realistic_geometry_topology_fixes.md``.
    When None, no cap is applied (preserves bit-exact regression)."""
    south_cap_lat: float | None = None
    """Southern polar cap latitude [deg].  When set, all ocean cells
    with ``lat < south_cap_lat`` are converted to land.  Antarctica is
    already mostly land in ETOPO so this is rarely needed in practice;
    provided for symmetry with ``north_cap_lat``.  When None, no cap."""


# ============================================================================
# NetCDF loading and regridding
# ============================================================================


def _detect_depth_variable(ds) -> tuple[str, str, str]:
    """Auto-detect depth, latitude, and longitude variable names."""
    all_vars = set(ds.data_vars.keys()) | set(ds.coords.keys())

    # Depth/elevation candidates (ETOPO, GEBCO, NOAA ERDDAP, generic).
    # "altitude" is the NOAA ERDDAP convention for etopo180.
    depth_candidates = [
        "z", "elevation", "altitude", "depth", "topo", "Band1",
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

    # Drop a duplicated periodic endpoint if the source covers the full
    # span twice (e.g. ETOPO with lon ∈ [−180, 180] — both endpoints
    # represent the same physical line).  Without this the periodic
    # padding below produces a back-to-back duplicate that scipy's
    # RegularGridInterpolator rejects with "points must be strictly
    # ascending or descending".
    if (
        lon_src.size >= 2
        and np.isclose(lon_src[-1] - lon_src[0], 360.0, atol=1e-6)
    ):
        lon_src = lon_src[:-1]
        depth_data = depth_data[:, :-1]

    # Wrap longitude for periodic interpolation.
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


def laplacian_smooth_2d(arr: np.ndarray, passes: int, is_cubed: bool) -> np.ndarray:
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


def laplacian_smooth_voronoi(
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
    cells_on_cell.shape[0]

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
# Mellor-Ezer-Oey r-factor cap
# ============================================================================


def compute_max_r_factor(H_bathy, ocean_mask):
    """Maximum r-factor over ocean-ocean neighbour pairs (4-connected).

    r = |H_i - H_j| / max(H_i, H_j).  Periodic in longitude (axis=1)
    via np.roll; latitude (axis=0) is bounded — we still roll, but
    only count pairs where both cells are ocean, which excludes any
    polar-row wrap-around since polar rows are always land in our
    setup.
    """
    H = np.asarray(H_bathy)
    ocean = np.asarray(ocean_mask) > 0.5
    r_max = 0.0
    for shift, axis in [(-1, 1), (+1, 1), (-1, 0), (+1, 0)]:
        H_n = np.roll(H, shift, axis=axis)
        ocean_n = np.roll(ocean, shift, axis=axis)
        valid = ocean & ocean_n
        if not valid.any():
            continue
        diff = np.abs(H - H_n)
        denom = np.fmax(H, H_n)
        # Avoid division-by-zero (only occurs at land neighbours, masked out)
        denom_safe = np.where(denom > 0.0, denom, 1.0)
        r = np.where(valid, diff / denom_safe, 0.0)
        r_max = max(r_max, float(r.max()))
    return r_max


def _r_factor_max_voronoi(H_bathy, ocean_mask, cells_on_edge):
    """Maximum r-factor over ocean-ocean edges on a Voronoi mesh.

    r_e = |H[c1] - H[c2]| / max(H[c1], H[c2])  for each edge (c1, c2)
    where both adjacent cells are ocean.

    Parameters
    ----------
    H_bathy : (nCells,)
    ocean_mask : (nCells,)  1=ocean, 0=land
    cells_on_edge : (2, nEdges)  c1, c2 indices per edge
    """
    H = np.asarray(H_bathy, dtype=np.float64)
    ocean = np.asarray(ocean_mask) > 0.5
    c1 = np.asarray(cells_on_edge[0], dtype=np.int64)
    c2 = np.asarray(cells_on_edge[1], dtype=np.int64)
    valid = ocean[c1] & ocean[c2]
    if not valid.any():
        return 0.0
    H1 = H[c1]
    H2 = H[c2]
    diff = np.abs(H1 - H2)
    denom = np.fmax(H1, H2)
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    r = np.where(valid, diff / denom_safe, 0.0)
    return float(r.max())


def apply_meo_r_factor_cap_voronoi(
    H_bathy,
    ocean_mask,
    mesh,
    r_factor_max,
    *,
    max_iter: int = 200,
    tol: float = 1.0e-6,
):
    """Mellor-Ezer-Oey r-factor cap on a Voronoi/MPAS mesh.

    Edge-list analog of :func:`apply_meo_r_factor_cap`.  For each
    ocean-ocean edge with ``r_e = |H[c1]-H[c2]| / max(H[c1],H[c2])``
    above ``r_factor_max``, deepens the shallower cell to
    ``H_deep * (1 - r_factor_max)``.  Jacobi iteration: each pass
    uses the previous pass's H as the source.  Cells only get
    deeper, so ``max_r`` is non-increasing.

    Sikiric et al. 2009 LSC2 / FESOM2 / MPAS-O Hoch 2020 use this
    edge-based selective deepening — not a 2D Cartesian Laplacian
    ``np.roll`` rule, which is meaningless on an unstructured mesh.

    Parameters
    ----------
    H_bathy : (nCells,)
        Ocean depth [m], positive downward.
    ocean_mask : (nCells,)
        1 = ocean, 0 = land.
    mesh : VoronoiMesh
        Provides ``cellsOnEdge`` (shape (2, nEdges)).
    r_factor_max : float
        Target cap (typically 0.2 for sigma-like; 0.3 acceptable
        for z-star).
    max_iter, tol : convergence controls.

    Returns
    -------
    H_new : np.ndarray  (nCells,)
    info : dict  same keys as the Cartesian variant.
    """
    if r_factor_max <= 0.0 or r_factor_max >= 1.0:
        raise ValueError(
            f"r_factor_max must be in (0, 1), got {r_factor_max!r}",
        )
    H = np.asarray(H_bathy, dtype=np.float64).copy()
    ocean = np.asarray(ocean_mask) > 0.5
    c1 = np.asarray(mesh.cellsOnEdge[0], dtype=np.int64)
    c2 = np.asarray(mesh.cellsOnEdge[1], dtype=np.int64)
    edge_active = ocean[c1] & ocean[c2]
    factor = 1.0 - r_factor_max

    H_initial = H.copy()
    initial_r = _r_factor_max_voronoi(H, ocean, mesh.cellsOnEdge)

    iterations = 0
    for it in range(max_iter):
        H_old = H.copy()
        H1 = H_old[c1]
        H2 = H_old[c2]
        # For each active edge, the shallower cell must be at least
        # ``factor * deeper``.  Compute the required floor and scatter
        # to both endpoints (np.maximum.at supports unbuffered update).
        deeper = np.fmax(H1, H2)
        required = deeper * factor
        H_new = H_old.copy()
        # Active edges only.
        active_idx = np.where(edge_active)[0]
        # Scatter the floor to both endpoints; the deeper cell's
        # required <= its current depth, so its update is a no-op.
        np.maximum.at(H_new, c1[active_idx], required[active_idx])
        np.maximum.at(H_new, c2[active_idx], required[active_idx])
        # Land cells unchanged.
        H_new = np.where(ocean, H_new, H_initial)
        delta_max = float(np.max(np.abs(H_new - H_old)))
        H = H_new
        iterations = it + 1
        if delta_max < tol:
            break

    final_r = _r_factor_max_voronoi(H, ocean, mesh.cellsOnEdge)
    vol_change = float(np.sum(np.where(ocean, H - H_initial, 0.0)))
    vol_initial = float(np.sum(np.where(ocean, H_initial, 0.0)))
    vol_change_frac = vol_change / vol_initial if vol_initial > 0 else 0.0
    cells_modified = int(
        np.sum(np.where(ocean, np.abs(H - H_initial) > tol, 0))
    )
    max_change = float(
        np.max(np.where(ocean, np.abs(H - H_initial), 0.0))
    )

    info = {
        "iterations": iterations,
        "initial_r_max": initial_r,
        "final_r_max": final_r,
        "volume_change_frac": vol_change_frac,
        "max_depth_change_m": max_change,
        "cells_modified": cells_modified,
    }
    return H, info


def apply_meo_r_factor_cap(
    H_bathy,
    ocean_mask,
    r_factor_max,
    *,
    max_iter: int = 200,
    tol: float = 1.0e-6,
):
    """Mellor-Ezer-Oey iterative r-factor cap.

    Iteratively deepens shallower ocean cells so that for every
    ocean-ocean neighbour pair, ``r = |H_i - H_j| / max(H_i, H_j) <=
    r_factor_max``.  Uses Jacobi-style parallel updates (each
    iteration updates all cells based on the previous iteration's
    state).  Convergence is monotone: cells only get deeper, so
    max(r) is non-increasing.

    The classic Mellor-Ezer-Oey 1994 prescription: when r > target,
    deepen the shallower cell to ``H_deep * (1 - r_factor_max)``.
    Volume change is added (small, typically <2% at 1° on real
    bathymetry).

    Parameters
    ----------
    H_bathy : array, shape (...)
        Ocean depth [m], positive downward.  Land cells should have
        H_bathy = 0 or any value, but ``ocean_mask`` must mark them.
    ocean_mask : array, shape (...)
        1 = ocean, 0 = land.  Same shape as H_bathy.
    r_factor_max : float
        Target r-factor cap (typically 0.2).
    max_iter : int
        Maximum Jacobi iterations.
    tol : float
        Convergence tolerance — stop when no cell changed by more
        than tol [m] in an iteration.

    Returns
    -------
    H_new : np.ndarray
        New bathymetry with r <= r_factor_max for every ocean pair.
    info : dict
        - ``iterations``: number of iterations performed
        - ``initial_r_max``: r before MEO
        - ``final_r_max``: r after MEO
        - ``volume_change_frac``: fractional ocean volume change
          (positive means added water)
        - ``max_depth_change_m``: max single-cell depth change
        - ``cells_modified``: count of cells whose depth changed
    """
    if r_factor_max <= 0.0 or r_factor_max >= 1.0:
        raise ValueError(
            f"r_factor_max must be in (0, 1), got {r_factor_max!r}",
        )
    H = np.asarray(H_bathy, dtype=np.float64).copy()
    ocean = np.asarray(ocean_mask) > 0.5
    factor = 1.0 - r_factor_max  # H_shallow >= H_deep * factor

    H_initial = H.copy()
    initial_r = compute_max_r_factor(H, ocean)

    iterations = 0
    for it in range(max_iter):
        H_old = H.copy()
        # Walk the four neighbours, deepening this cell to satisfy
        # the constraint with each.  Jacobi-style: use H_old's neighbour
        # values within the same iteration.
        H_new = H.copy()
        for shift, axis in [(-1, 1), (+1, 1), (-1, 0), (+1, 0)]:
            H_n = np.roll(H_old, shift, axis=axis)
            ocean_n = np.roll(ocean, shift, axis=axis)
            min_required = H_n * factor
            # Only constrain ocean-ocean pairs.
            valid = ocean & ocean_n
            H_new = np.where(
                valid, np.maximum(H_new, min_required), H_new,
            )
        # Land cells unchanged.
        H_new = np.where(ocean, H_new, H)
        delta_max = float(np.max(np.abs(H_new - H_old)))
        H = H_new
        iterations = it + 1
        if delta_max < tol:
            break

    final_r = compute_max_r_factor(H, ocean)
    # Volume change (per unit area; this is in units of m, i.e. mean depth change)
    vol_change = float(np.sum(np.where(ocean, H - H_initial, 0.0)))
    vol_initial = float(np.sum(np.where(ocean, H_initial, 0.0)))
    vol_change_frac = vol_change / vol_initial if vol_initial > 0 else 0.0
    cells_modified = int(np.sum(np.where(ocean, np.abs(H - H_initial) > tol, 0)))
    max_change = float(np.max(np.where(ocean, np.abs(H - H_initial), 0.0)))

    info = {
        "iterations": iterations,
        "initial_r_max": initial_r,
        "final_r_max": final_r,
        "volume_change_frac": vol_change_frac,
        "max_depth_change_m": max_change,
        "cells_modified": cells_modified,
    }
    return H, info


# ============================================================================
# Strait enforcement
# ============================================================================


def haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Great-circle distance in km between two points (degrees).

    Broadcasts NumPy-style, so a scalar point against an array of points
    returns per-point distances -- used for nearest-cell assignment on both
    structured (lat-lon) and unstructured (MPAS Voronoi) meshes. Promoted from
    ``_haversine_km`` so ocean.forcing can share the one implementation rather
    than re-derive the formula (CLAUDE.md: no re-derivation, no private
    cross-module imports).
    """
    R = constants.R_earth * 1e-3  # Earth radius [km]
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
        dist_km = haversine_km(flat_lat, flat_lon, s_lat, s_lon)
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


def fill_isolated_basins(ocean_mask: np.ndarray, grid=None) -> np.ndarray:
    """Remove isolated ocean basins not connected to the main ocean.

    Flood-fills connected ocean components and keeps only the largest
    (the global ocean).  All other components — semi-enclosed seas
    whose connecting straits are narrower than the grid scale — are
    set to land.

    Works for any grid type:
    - **Lat-lon** (2D): 4-connected with periodic longitude wrapping.
    - **MPAS / Voronoi** (1D): uses ``mesh.cellsOnCell`` connectivity.
    - **Cubed-sphere** (6, n, n): per-face 4-connected (approximate —
      cross-face connectivity is ignored, sufficient for lakes).

    Parameters
    ----------
    ocean_mask : array
        1=ocean, 0=land.  Shape ``(n_lat, n_lon)`` for lat-lon,
        ``(nCells,)`` for MPAS, or ``(6, n, n)`` for cubed-sphere.
    grid : VoronoiMesh, optional
        Required for MPAS (provides ``cellsOnCell`` adjacency).
        Ignored for structured grids.

    Returns
    -------
    array
        Updated mask with isolated basins filled to land.
    """
    from collections import deque

    mask = np.asarray(ocean_mask, dtype=np.float64).copy()
    is_ocean = mask > 0.5

    # --- MPAS (unstructured): use cellsOnCell adjacency ---
    if mask.ndim == 1 and grid is not None and hasattr(grid, "cellsOnCell"):
        n_cells = mask.shape[0]
        adj_raw = np.asarray(grid.cellsOnCell)
        # cellsOnCell may be (maxEdges, nCells) or (nCells, maxEdges)
        if adj_raw.shape[0] < adj_raw.shape[1]:
            adj = adj_raw.T  # → (nCells, maxEdges)
        else:
            adj = adj_raw

        labels = np.zeros(n_cells, dtype=np.int32)
        component_id = 0
        sizes = []

        for seed in range(n_cells):
            if not is_ocean[seed] or labels[seed] > 0:
                continue
            component_id += 1
            q = deque([seed])
            labels[seed] = component_id
            count = 0
            while q:
                c = q.popleft()
                count += 1
                for nb in adj[c]:
                    nb = int(nb)
                    if 0 <= nb < n_cells and is_ocean[nb] and labels[nb] == 0:
                        labels[nb] = component_id
                        q.append(nb)
            sizes.append((component_id, count))

        if not sizes:
            return mask
        largest_id = max(sizes, key=lambda x: x[1])[0]
        n_removed = int(np.sum(is_ocean & (labels != largest_id)))
        mask[is_ocean & (labels != largest_id)] = 0.0
        if n_removed > 0:
            basins = [(cid, cnt) for cid, cnt in sizes if cid != largest_id]
            print(f"  Removed {n_removed} isolated-basin cells "
                  f"({len(basins)} basins, sizes: "
                  f"{sorted([c for _, c in basins], reverse=True)[:10]})")
        return mask

    # --- Cubed-sphere (6, n, n): per-face, approximate ---
    if mask.ndim == 3 and mask.shape[0] == 6:
        result = mask.copy()
        for face in range(6):
            result[face] = fill_isolated_basins(mask[face])
        return result

    # --- Lat-lon (2D): 4-connected with periodic longitude wrapping ---
    if mask.ndim != 2:
        return mask

    n_lat, n_lon = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    component_id = 0
    sizes = []

    for j in range(n_lat):
        for i in range(n_lon):
            if not is_ocean[j, i] or labels[j, i] > 0:
                continue
            component_id += 1
            q = deque([(j, i)])
            labels[j, i] = component_id
            count = 0
            while q:
                cj, ci = q.popleft()
                count += 1
                for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nj = cj + dj
                    ni = (ci + di) % n_lon  # periodic in longitude
                    if 0 <= nj < n_lat and is_ocean[nj, ni] and labels[nj, ni] == 0:
                        labels[nj, ni] = component_id
                        q.append((nj, ni))
            sizes.append((component_id, count))

    if not sizes:
        return mask
    largest_id = max(sizes, key=lambda x: x[1])[0]
    n_removed = int(np.sum(is_ocean & (labels != largest_id)))
    mask[is_ocean & (labels != largest_id)] = 0.0
    if n_removed > 0:
        basins = [(cid, cnt) for cid, cnt in sizes if cid != largest_id]
        print(f"  Removed {n_removed} isolated-basin cells "
              f"({len(basins)} basins, sizes: "
              f"{sorted([c for _, c in basins], reverse=True)[:10]})")
    return mask


# Keep old names as aliases for back-compat with tests
_fill_isolated_basins = fill_isolated_basins
_fill_isolated_basins_2d = fill_isolated_basins


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

    # Ensure longitude in [0, 360).  ETOPO/GEBCO often store longitude in
    # [-180, +180] inclusive, which after modulo produces duplicate values
    # (both -180 and +180 → 180).  RegularGridInterpolator below rejects
    # non-strictly-monotonic axes, so we deduplicate after sorting.
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    elev_data = elev_data[:, lon_order]
    if lon_src.size > 1:
        keep_lon = np.concatenate([[True], np.diff(lon_src) > 0.0])
        if not keep_lon.all():
            lon_src = lon_src[keep_lon]
            elev_data = elev_data[:, keep_lon]

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


# Last MEO run summary (read by diagnostic scripts).  Per-call info also
# returned by ``apply_meo_r_factor_cap`` directly.
_LAST_MEO_INFO: dict = {}


def _maybe_apply_meo(depth, ocean_mask, cfg):
    """Helper used by per-grid loaders to apply MEO after smoothing.

    MEO must run after Laplacian smoothing because the smoothing
    blends land cells (depth=0) into adjacent ocean cells, which
    reduces ocean-cell depth at coastlines and re-introduces
    r-factor violations.  Running MEO last guarantees the final
    bathymetry satisfies r <= cfg.r_factor_max.
    """
    if cfg.r_factor_max is None:
        return depth
    depth, meo_info = apply_meo_r_factor_cap(
        np.asarray(depth), np.asarray(ocean_mask), cfg.r_factor_max,
        max_iter=cfg.meo_max_iter,
    )
    _LAST_MEO_INFO.update(meo_info)
    return depth


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
    depth = laplacian_smooth_2d(depth, cfg.smoothing_passes, is_cubed=True)

    # Re-enforce minimum depth after smoothing
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # MEO r-factor cap (after smoothing).
    depth = _maybe_apply_meo(depth, ocean_mask, cfg)

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

    dtype = get_policy().storage
    return jnp.array(H_bathy, dtype=dtype), jnp.array(ocean_mask, dtype=dtype)


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
    grid_spacing_m = np.sqrt(mean_area)
    grid_spacing_deg = grid_spacing_m / constants.R_earth * 180.0 / np.pi

    depth, ocean_mask = load_bathymetry(
        lat_deg, lon_deg, cfg, grid_spacing_deg=grid_spacing_deg,
    )

    # Smoothing on Voronoi mesh topology
    cells_on_cell = np.asarray(mesh.cellsOnCell)
    n_edges_on_cell = np.asarray(mesh.nEdgesOnCell)
    depth = laplacian_smooth_voronoi(
        depth, cells_on_cell, n_edges_on_cell, cfg.smoothing_passes,
    )

    # Re-enforce minimum depth after smoothing
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # MEO r-factor cap on the Voronoi mesh (edge-based, Sikiric 2009 LSC2 /
    # MPAS-O Hoch 2020 / FESOM2 Danilov 2017).  The 2D Cartesian
    # ``apply_meo_r_factor_cap`` is meaningless on a 1D unstructured
    # array — uses ``np.roll`` on the cell-index axis.
    if cfg.r_factor_max is not None:
        depth, meo_info = apply_meo_r_factor_cap_voronoi(
            depth, ocean_mask, mesh, cfg.r_factor_max,
            max_iter=cfg.meo_max_iter,
        )
        _LAST_MEO_INFO.update(meo_info)

    # Remove isolated basins on the Voronoi mesh connectivity.
    # The generic load_bathymetry call above may have attempted this
    # via _fill_isolated_basins, but that version doesn't have mesh
    # connectivity for 1D arrays.  Redo properly here.
    if cfg.fill_isolated_basins:
        ocean_mask = fill_isolated_basins(ocean_mask, grid=mesh)
        depth = np.where(ocean_mask > 0.5, depth, 0.0)

    H_bathy = np.where(ocean_mask > 0.5, depth, 0.0)

    dtype = get_policy().storage
    return jnp.array(H_bathy, dtype=dtype), jnp.array(ocean_mask, dtype=dtype)


def load_bathymetry_latlon_cgrid(
    grid,
    cfg: BathymetryConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load realistic bathymetry for the lat-lon C-grid.

    Parameters
    ----------
    grid : LatLonGrid
        Target grid with lat2d/lon2d in radians, shape (n_lat, n_lon).
    cfg : BathymetryConfig
        Configuration.

    Returns
    -------
    H_bathy : jnp.ndarray
        Bathymetry depth [m], shape (n_lat, n_lon). Positive downward.
        Set to H_max on land for smooth z* Jacobian (the land_mask
        prevents actual flow on land cells).
    ocean_mask : jnp.ndarray
        1=ocean, 0=land, shape (n_lat, n_lon).
    """
    target_lat = np.asarray(grid.lat2d) * 180.0 / np.pi
    target_lon = np.asarray(grid.lon2d) * 180.0 / np.pi
    target_lon = target_lon % 360.0
    grid_spacing = 180.0 / grid.n_lat

    depth, ocean_mask = load_bathymetry(
        target_lat, target_lon, cfg, grid_spacing_deg=grid_spacing,
    )

    # Polar caps — close off the high-lat regions where the lat-lon grid
    # singularity + small dx make the cos²(lat) A_h scaling regime hard
    # to keep both stable and damped.  Applied BEFORE smoothing so the
    # cap edge is also smoothed into a neat coastline.
    if cfg.north_cap_lat is not None:
        ocean_mask = np.where(target_lat > cfg.north_cap_lat, 0.0, ocean_mask)
    if cfg.south_cap_lat is not None:
        ocean_mask = np.where(target_lat < cfg.south_cap_lat, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # Smoothing on regular lat-lon (periodic in longitude, walls at poles).
    depth = laplacian_smooth_2d(depth, cfg.smoothing_passes, is_cubed=False)

    # Re-enforce minimum depth after smoothing.
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # MEO r-factor cap — applied last, after smoothing has finished
    # blending coastal cells.  This guarantees the final bathymetry
    # satisfies r <= cfg.r_factor_max for every ocean-ocean pair.
    depth = _maybe_apply_meo(depth, ocean_mask, cfg)

    # For z* Jacobian smoothness: set land depth to H_max
    # (the land_mask prevents actual flow on land cells).
    H_bathy = np.where(ocean_mask > 0.5, depth, cfg.H_max)

    dtype = get_policy().storage
    return jnp.array(H_bathy, dtype=dtype), jnp.array(ocean_mask, dtype=dtype)


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

    # Polar caps (see load_bathymetry_latlon_cgrid for rationale).
    if cfg.north_cap_lat is not None:
        ocean_mask = np.where(target_lat_2d > cfg.north_cap_lat, 0.0, ocean_mask)
    if cfg.south_cap_lat is not None:
        ocean_mask = np.where(target_lat_2d < cfg.south_cap_lat, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # Smoothing on Gaussian grid
    depth = laplacian_smooth_2d(depth, cfg.smoothing_passes, is_cubed=False)

    # Re-enforce minimum depth after smoothing
    ocean_mask = np.where(depth < cfg.H_min, 0.0, ocean_mask)
    depth = np.where(ocean_mask > 0.5, depth, 0.0)

    # MEO r-factor cap (after smoothing).
    depth = _maybe_apply_meo(depth, ocean_mask, cfg)

    H_bathy = np.where(ocean_mask > 0.5, depth, cfg.H_max)

    dtype = get_policy().storage
    return jnp.array(H_bathy, dtype=dtype), jnp.array(ocean_mask, dtype=dtype)


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


def _is_latlon_cgrid(grid) -> bool:
    """LatLonGrid has dlon/dlat (regular spacing) — distinguishes from
    CubedSphereGrid (which has the same ``n`` property), Voronoi (nCells),
    and Gaussian (irregular Gauss latitudes, no dlat)."""
    return (
        hasattr(grid, 'dlon')
        and hasattr(grid, 'dlat')
        and hasattr(grid, 'lat2d')
        and not hasattr(grid, 'nCells')
        and not hasattr(grid, 'Pnm')
    )


def _idealized_dispatch(grid, cfg: BathymetryConfig):
    """Dispatch idealized bathymetry to appropriate grid handler."""
    # LatLonGrid: check FIRST since it also exposes ``n`` (= n_lat) via property
    if _is_latlon_cgrid(grid):
        from legoesm.ocean.init_latlon_cgrid import (
            idealized_bathymetry_latlon_cgrid,
        )
        return idealized_bathymetry_latlon_cgrid(
            grid, cfg.H_max, cfg.land_lat_threshold,
        )

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
        lat_deg = jnp.abs(grid.grid_lat) * (180.0 / jnp.pi)
        ocean_mask = jnp.where(lat_deg < cfg.land_lat_threshold, 1.0, 0.0)
        H_bathy = jnp.where(ocean_mask > 0.5, cfg.H_max, 1.0)
        return H_bathy, ocean_mask

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def _file_dispatch(grid, cfg: BathymetryConfig):
    """Dispatch file-based bathymetry to appropriate grid handler."""
    # LatLonGrid: check first
    if _is_latlon_cgrid(grid):
        return load_bathymetry_latlon_cgrid(grid, cfg)

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
    T_water_init_C: float = 20.0,
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
    T_water_init_C : float
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

    # LatLonGrid: check first
    if _is_latlon_cgrid(grid):
        return _rest_state_latlon_cgrid(grid, z_coord, H_bathy, ocean_mask,
                                          T_water_init_C, T_deep, S_uniform)

    # CubedSphereGrid
    if hasattr(grid, 'n') and hasattr(grid, 'lat') and not hasattr(grid, 'nCells'):
        return _rest_state_cubed(grid, z_coord, H_bathy, ocean_mask,
                                 T_water_init_C, T_deep, S_uniform)

    # VoronoiMesh
    if hasattr(grid, 'nCells') and hasattr(grid, 'latCell'):
        return _rest_state_mpas(grid, z_coord, H_bathy, ocean_mask,
                                T_water_init_C, T_deep, S_uniform)

    # GaussianGrid
    if hasattr(grid, 'n_lat') and not hasattr(grid, 'n'):
        return _rest_state_spectral(grid, z_coord, H_bathy, ocean_mask,
                                    T_water_init_C, T_deep, S_uniform)

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def _rest_state_latlon_cgrid(grid, z_coord, H_bathy, ocean_mask,
                              T_water_init_C, T_deep, S_uniform):
    """Create lat-lon C-grid ocean rest state with given bathymetry.

    Delegates to ``rest_state_latlon_cgrid_ocean`` (which handles the
    full LatLonCGridOceanState construction including face masks and
    z* metadata) by passing the loaded bathymetry + mask via the
    *_override* parameters.
    """
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=T_water_init_C,
        T_deep=T_deep,
        S_uniform=S_uniform,
        land_mask_override=ocean_mask,
        H_bathy_override=H_bathy,
    )


def _rest_state_cubed(grid, z_coord, H_bathy, ocean_mask,
                      T_water_init_C, T_deep, S_uniform):
    """Create cubed-sphere ocean rest state with given bathymetry."""
    from legoesm.core.field import Field
    from legoesm.ocean.state import OceanState

    n = grid.n
    nlev = z_coord.n_levels

    scale_depth = 1000.0
    T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
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
                     T_water_init_C, T_deep, S_uniform):
    """Create MPAS ocean rest state with given bathymetry."""
    from legoesm.core.field import Field
    from legoesm.core.state import MPASOceanState

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = z_coord.n_levels

    scale_depth = 1000.0
    T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
    T_data = jnp.broadcast_to(T_profile[jnp.newaxis, :], (nCells, nlev))

    S_data = jnp.full((nCells, nlev), S_uniform)
    u_data = jnp.zeros((nEdges, nlev))
    eta_data = jnp.zeros(nCells)

    w_data = jnp.zeros((nCells, nlev + 1))

    return MPASOceanState(
        u=Field(data=u_data, name="u", dims=("nEdges", "nlev"), units="m/s",
                staggering="edge"),
        T=Field(data=T_data, name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data, name="S", dims=("nCells", "nlev"), units="PSU"),
        eta=Field(data=eta_data, name="eta", dims=("nCells",), units="m"),
        w=Field(data=w_data, name="w", dims=("nCells", "nlev+1"), units="m/s"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask, name="land_mask", dims=("nCells",), units="1"),
    )


def _rest_state_spectral(grid, z_coord, H_bathy, ocean_mask,
                         T_water_init_C, T_deep, S_uniform):
    """Create spectral ocean rest state with given bathymetry.

    Spectral ocean is soft-retired (#99); custom bathymetry and land masks
    cannot be threaded through `rest_state_spectral_ocean` reliably (Gibbs
    ringing at coastlines). This wrapper accepts only flat-bottom, all-ocean
    inputs and raises otherwise so the caller sees the failure instead of
    silently getting an inconsistent state.
    """
    from legoesm.ocean.dynamics.spectral_ocean_pe import (
        rest_state_spectral_ocean,
    )
    H_np = np.asarray(H_bathy)
    mask_np = np.asarray(ocean_mask)
    if not np.allclose(H_np, H_np.flat[0]) or not np.all(mask_np == 1):
        raise NotImplementedError(
            "spectral ocean does not support custom bathymetry or land mask "
            "(see #99 for retirement context). Use the cubed-sphere, "
            "lat-lon C-grid, or MPAS ocean model instead."
        )
    return rest_state_spectral_ocean(
        grid, z_coord,
        T_water_init_C=T_water_init_C,
        T_deep=T_deep,
        S_uniform=S_uniform,
        H_max=float(H_np.flat[0]),
    )
