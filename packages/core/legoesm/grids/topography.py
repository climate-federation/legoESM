"""Topography generators, real topography loading, and land-mask utilities.

Provides functions to generate idealized topography fields on a
cubed-sphere grid (6, n, n) or Gaussian grid (n_lat, n_lon),
load real topography from NetCDF files (ETOPO, GEBCO, etc.),
and derive surface geopotential and land/ocean masks.

All topography generators return surface elevation z_s in meters.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.edge_blending import blend_scalar_cube_edges_2d
from legoesm.grids.halo import pad_halo_local

# Elevation-magnitude sanity threshold [m] for the geopotential-vs-meters
# "double-g" trap: no Earth surface elevation exceeds ~8850 m (Everest) and no
# ocean depth exceeds ~11000 m (Mariana), but the GEOPOTENTIAL of Everest is
# ~86800 m²/s².  An "elevation" field whose magnitude exceeds this threshold is
# almost certainly geopotential [m²/s²] mistakenly passed as elevation [m] —
# multiplying it by g again in load_real_topography would silently inflate
# phis ~9.8×.
_MAX_PLAUSIBLE_ELEV_M = 12000.0


def _grid_lat_lon_2d(grid):
    """Per-cell (lat, lon) in the grid's native horizontal layout [rad].

    Cubed-sphere exposes 2-D ``(6, n, n)`` ``lat``/``lon`` directly; the lat-lon
    grid stores 1-D axes (``lat`` shape ``(n_lat,)``, ``lon`` shape ``(n_lon,)``)
    plus 2-D meshes ``lat2d``/``lon2d``.  The analytic-mountain generators below
    need per-cell 2-D fields, so prefer ``lat2d``/``lon2d`` when present and fall
    back to ``lat``/``lon`` (already 2-D on the cubed sphere, 1-D per-cell on
    unstructured meshes).  Without this, ``grid.lat - grid.lon`` style broadcasts
    fail on lat-lon as ``(n_lat,) + (n_lon,)``.
    """
    lat = getattr(grid, "lat2d", None)
    lon = getattr(grid, "lon2d", None)
    return (grid.lat if lat is None else lat,
            grid.lon if lon is None else lon)


def gaussian_mountain(
    grid: CubedSphereGrid,
    h0: float = 2500.0,
    lat0: float = 40.0 * jnp.pi / 180.0,
    lon0: float = 255.0 * jnp.pi / 180.0,
    sigma_lat: float = 15.0 * jnp.pi / 180.0,
    sigma_lon: float = 15.0 * jnp.pi / 180.0,
) -> jnp.ndarray:
    """Generate a single Gaussian mountain.

    z_s = h0 * exp(-((lat-lat0)^2/(2*sigma_lat^2) + (lon-lon0)^2/(2*sigma_lon^2)))

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m]. Default 2500 (Rocky Mountains scale).
    lat0 : float
        Center latitude [rad]. Default 40 deg N.
    lon0 : float
        Center longitude [rad]. Default 255 deg E.
    sigma_lat : float
        Latitudinal half-width [rad]. Default 15 deg.
    sigma_lon : float
        Longitudinal half-width [rad]. Default 15 deg.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat, lon = _grid_lat_lon_2d(grid)

    dlat = lat - lat0

    # Periodic longitude wrapping to [-pi, pi]
    dlon = jnp.mod(lon - lon0 + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    z_s = h0 * jnp.exp(
        -(dlat ** 2 / (2.0 * sigma_lat ** 2) + dlon ** 2 / (2.0 * sigma_lon ** 2))
    )
    return z_s


def zonal_ridge(
    grid: CubedSphereGrid,
    h0: float = 2500.0,
    lat0: float = 45.0 * jnp.pi / 180.0,
    sigma_lat: float = 10.0 * jnp.pi / 180.0,
) -> jnp.ndarray:
    """Generate a zonally symmetric ridge.

    z_s = h0 * exp(-(lat-lat0)^2/(2*sigma_lat^2))

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m].
    lat0 : float
        Ridge center latitude [rad]. Default 45 deg N.
    sigma_lat : float
        Latitudinal half-width [rad]. Default 10 deg.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat, _ = _grid_lat_lon_2d(grid)
    dlat = lat - lat0
    return h0 * jnp.exp(-(dlat ** 2) / (2.0 * sigma_lat ** 2))


def schaer_mountain(
    grid: CubedSphereGrid,
    h0: float = 2000.0,
    lat0: float = 0.0,
    lon0: float = jnp.pi,
    half_width: float = 72.0e3,
) -> jnp.ndarray:
    """Generate Schaer-type cosine-bell mountain topography.

    Uses great-circle distance: z_s = h0/2 * (1 + cos(pi*r/a)) for r < a,
    where r is the distance from the mountain center and a is the half-width.

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m].
    lat0 : float
        Mountain center latitude [rad].
    lon0 : float
        Mountain center longitude [rad].
    half_width : float
        Mountain half-width [m].

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat, lon = _grid_lat_lon_2d(grid)

    # Great-circle distance via haversine
    dlat = lat - lat0
    dlon = lon - lon0
    a = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(lat) * jnp.cos(lat0) * jnp.sin(dlon / 2.0) ** 2
    )
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))
    dist = angular_dist * grid.radius

    # Cosine bell: h0/2 * (1 + cos(pi*r/a)) for r < a, 0 otherwise
    z_s = jnp.where(
        dist < half_width,
        0.5 * h0 * (1.0 + jnp.cos(jnp.pi * dist / half_width)),
        0.0,
    )
    return z_s


def land_mask_from_topography(
    z_s: jnp.ndarray,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """Derive a land/ocean mask from surface elevation.

    Parameters
    ----------
    z_s : jnp.ndarray
        Surface elevation [m], shape (6, n, n).
    threshold : float
        Elevation threshold [m]. Grid cells with z_s > threshold are land.

    Returns
    -------
    jnp.ndarray
        Land fraction f_land (1.0 = land, 0.0 = ocean), shape (6, n, n).
    """
    return jnp.where(z_s > threshold, 1.0, 0.0)


def phis_from_topography(z_s: jnp.ndarray) -> jnp.ndarray:
    """Convert surface elevation to surface geopotential.

    phis = g * z_s

    Parameters
    ----------
    z_s : jnp.ndarray
        Surface elevation [m].

    Returns
    -------
    jnp.ndarray
        Surface geopotential [m^2/s^2], same shape as z_s.
    """
    return constants.g * z_s


# ==============================================================================
# Topography configuration
# ==============================================================================


class TopographyConfig(NamedTuple):
    """Configuration for topography loading.

    Parameters
    ----------
    source : str
        Topography source: "flat", "gaussian", "file".
    path : str
        Path to topography NetCDF file (for source="file").
    elev_var : str
        Elevation variable name in NetCDF.
    lat_var : str
        Latitude variable name in NetCDF.
    lon_var : str
        Longitude variable name in NetCDF.
    smoothing_passes : int
        Number of Laplacian smoothing passes to apply (removes 2Δx noise).
    edge_blend_strength : float
        Edge blending strength at cubed-sphere face boundaries [0, 1].
    edge_blend_width : int
        Number of grid cells from face edge to blend.
    clip_negative : bool
        If True, set negative elevations (ocean floor) to 0.
    land_mask_path : str
        Optional path to a separate land-sea-mask NetCDF file (e.g. CMIP6
        ``sftlf`` or an ERA5 ``lsm`` invariant).  When set, the land
        fraction is taken from this file instead of being derived from
        ``elevation > 0`` — which avoids misclassifying below-sea-level
        land (Caspian/Dead Sea/Netherlands) as ocean.
    land_mask_var : str
        Variable name in ``land_mask_path``.  Empty → auto-detect among
        ``sftlf``, ``lsm``, ``land_sea_mask``, ``land``,
        ``land_area_fraction``.
    """
    source: str = "flat"
    path: str = ""
    elev_var: str = ""
    lat_var: str = ""
    lon_var: str = ""
    smoothing_passes: int = 4
    edge_blend_strength: float = 0.3
    edge_blend_width: int = 2
    clip_negative: bool = True
    land_mask_path: str = ""
    land_mask_var: str = ""


# ==============================================================================
# Real topography loading
# ==============================================================================


def _detect_variables(ds) -> tuple[str, str, str]:
    """Auto-detect elevation, latitude, and longitude variable names."""
    all_vars = set(ds.data_vars.keys()) | set(ds.coords.keys())

    # Elevation: try common names
    elev_candidates = ["z", "elevation", "topo", "Band1", "ROSE",
                       "bedrock_topography", "surface_elevation"]
    elev_var = None
    for c in elev_candidates:
        if c in all_vars:
            elev_var = c
            break
    if elev_var is None:
        # Use first data variable
        data_vars = list(ds.data_vars.keys())
        if data_vars:
            elev_var = data_vars[0]
        else:
            raise KeyError(
                f"Cannot detect elevation variable. "
                f"Available: {sorted(all_vars)}"
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

    return elev_var, lat_var, lon_var


def _build_latlon_interpolator(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    data: np.ndarray,
):
    """Bilinear interpolator with longitudinal wrap AND polar edge padding.

    Shared by the elevation (:func:`_regrid_to_target`) and land-fraction
    (:func:`_derive_land_fraction`, :func:`_load_land_fraction_file` via
    ``_regrid_to_target``) regrid paths so both get identical boundary
    handling:

    * Longitude is wrapped by one column on each side for periodicity.
    * The latitude axis is padded to exactly ±90° by EDGE REPLICATION
      (prepend/append a copy of the first/last latitude row).  Without this,
      ``bounds_error=False, fill_value=0.0`` silently assigns 0 (ocean /
      sea level) to target cells POLEWARD of the source grid's outermost
      latitude CENTER — e.g. a 1° sftlf spanning ±89.5° left cubed-sphere or
      Gaussian cells near ±90° misclassified as f_land=0 open ocean over
      Antarctica.  With the padding, pole-adjacent targets get the nearest
      real source value instead.
    """
    from scipy.interpolate import RegularGridInterpolator

    lat_src = np.asarray(lat_src)
    # Longitudinal wrap (unchanged behavior).
    lon_wrapped = np.concatenate([
        lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0
    ])
    data_wrapped = np.concatenate([
        data[:, -1:], data, data[:, :1]
    ], axis=1)

    # Polar edge replication (lat_src is ascending by the callers' contract).
    if lat_src[0] > -90.0:
        lat_src = np.concatenate([[-90.0], lat_src])
        data_wrapped = np.concatenate([data_wrapped[:1], data_wrapped], axis=0)
    if lat_src[-1] < 90.0:
        lat_src = np.concatenate([lat_src, [90.0]])
        data_wrapped = np.concatenate([data_wrapped, data_wrapped[-1:]], axis=0)

    return RegularGridInterpolator(
        (lat_src, lon_wrapped), data_wrapped,
        method="linear", bounds_error=False, fill_value=0.0,
    )


def _regrid_to_target(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    elev_data: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    """Regrid elevation data from regular lat-lon to target grid points.

    Uses bilinear interpolation with longitude wrapping for periodicity
    and polar edge replication (see ``_build_latlon_interpolator``).

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitude in degrees.
    lon_src : (nlon_src,)
        Source longitude in degrees, in [0, 360).
    elev_data : (nlat_src, nlon_src)
        Source elevation in meters.
    target_lat_deg : array
        Target latitude in degrees (any shape).
    target_lon_deg : array
        Target longitude in degrees (any shape).

    Returns
    -------
    np.ndarray
        Regridded elevation, same shape as target_lat_deg.
    """
    interp = _build_latlon_interpolator(lat_src, lon_src, elev_data)

    target_shape = target_lat_deg.shape
    points = np.stack([
        target_lat_deg.ravel(),
        target_lon_deg.ravel(),
    ], axis=-1)

    result = interp(points).reshape(target_shape)
    return result


def _derive_land_fraction(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    elev_data: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
    grid_spacing_deg: float,
) -> np.ndarray:
    """Derive land fraction by sub-sampling the high-res elevation data.

    For each target grid cell, samples the source elevation at sub-grid
    points and computes the fraction with elevation > 0.

    Parameters
    ----------
    lat_src, lon_src : (nlat_src,), (nlon_src,)
        Source grid in degrees.
    elev_data : (nlat_src, nlon_src)
        Source elevation in meters.
    target_lat_deg, target_lon_deg : arrays
        Target grid centers in degrees (any shape).
    grid_spacing_deg : float
        Approximate target grid spacing in degrees.

    Returns
    -------
    np.ndarray
        Land fraction [0, 1], same shape as target_lat_deg.
    """
    interp = _build_latlon_interpolator(lat_src, lon_src, elev_data)

    # Sample at a sub-grid of points around each target point
    n_sub = max(3, int(np.ceil(grid_spacing_deg / 0.5)))
    offsets = np.linspace(-0.5, 0.5, n_sub) * grid_spacing_deg

    target_shape = target_lat_deg.shape
    flat_lat = target_lat_deg.ravel()
    flat_lon = target_lon_deg.ravel()
    n_pts = flat_lat.size

    land_frac = np.zeros(n_pts, dtype=np.float64)

    for dlat in offsets:
        for dlon in offsets:
            pts = np.stack([flat_lat + dlat, flat_lon + dlon], axis=-1)
            elev = interp(pts)
            land_frac += (elev > 0.0).astype(np.float64)

    land_frac /= n_sub * n_sub
    return land_frac.reshape(target_shape)


def _load_land_fraction_file(
    path: str,
    var_name: str,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    """Load a land-sea-mask file and regrid (bilinear) to the target grid.

    Accepts any regular lat-lon NetCDF land-fraction field — CMIP6
    ``sftlf`` (percent), ERA5 ``lsm`` (fraction), etc.  Percent fields
    are auto-detected (max value > 1.5) and rescaled to [0, 1].

    Parameters
    ----------
    path : str
        NetCDF file containing the land-sea mask.
    var_name : str
        Mask variable name; empty → auto-detect.
    target_lat_deg, target_lon_deg : np.ndarray
        Target grid centers in degrees (any shape).

    Returns
    -------
    np.ndarray
        Land fraction in [0, 1], same shape as ``target_lat_deg``.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        if var_name:
            mask_var = var_name
        else:
            mask_var = None
            for c in ("sftlf", "lsm", "land_sea_mask", "land",
                      "land_area_fraction", "LSM"):
                if c in all_names:
                    mask_var = c
                    break
            if mask_var is None:
                data_vars = list(ds.data_vars)
                if not data_vars:
                    raise KeyError(
                        f"No data variables in land-mask file {path}"
                    )
                mask_var = data_vars[0]

        lat_var = None
        for c in ("lat", "latitude", "y", "Y"):
            if c in all_names:
                lat_var = c
                break
        lon_var = None
        for c in ("lon", "longitude", "x", "X"):
            if c in all_names:
                lon_var = c
                break
        if lat_var is None or lon_var is None:
            raise KeyError(
                f"Cannot detect lat/lon in land-mask file {path}; "
                f"available: {sorted(all_names)}"
            )

        lat_src = ds[lat_var].values.astype(np.float64)
        lon_src = ds[lon_var].values.astype(np.float64)
        mask_data = ds[mask_var].values.astype(np.float64)
    finally:
        ds.close()

    while mask_data.ndim > 2:
        mask_data = mask_data[0]

    # Longitude in [0, 360), ascending
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    mask_data = mask_data[:, lon_order]

    # Latitude ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        mask_data = mask_data[::-1, :]

    mask_data = np.where(np.isnan(mask_data), 0.0, mask_data)

    # Percent → fraction (CMIP6 sftlf is 0-100; ERA5 lsm is 0-1)
    if np.nanmax(mask_data) > 1.5:
        mask_data = mask_data / 100.0

    f_land = _regrid_to_target(
        lat_src, lon_src, mask_data, target_lat_deg, target_lon_deg
    )
    return np.clip(f_land, 0.0, 1.0)


def _laplacian_smooth_cubed_sphere(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    """Laplacian smoothing on a cubed-sphere field (6, n, n).

    Each cell becomes ``0.5*original + 0.5*smoothed`` where ``smoothed`` is the
    5-point mean ``(self + 4 neighbours)/5``.  The neighbours at face boundaries
    come from the cross-face HALO (``pad_halo_local``, which handles the axis
    swaps and reversals), so the smoothing is CONTINUOUS across cube edges.

    The previous implementation used one-sided boundary CLAMPING (edge cells
    averaged only their in-face neighbours), which smoothed each face in
    isolation and left a per-face discontinuity at the shared edges — the
    "cube imprint" artifact.  Using the real cross-face halo removes it at the
    source (the downstream ``blend_scalar_cube_edges_2d`` step is then a light
    final touch, not a band-aid for a seam this function created).

    NOTE: cube-imprint artifacts are confirmed VISUALLY (CLAUDE.md visual-verify
    rule); the unit test asserts the necessary cross-face-leakage property, but
    the nightly cube-SW visual-regression gate is the authoritative check.
    """
    if passes <= 0:
        return arr
    field = jnp.asarray(arr)
    orig = field
    for _ in range(passes):
        # (6, n+2, n+2) with REAL neighbour data from adjacent faces in the halo.
        # Use the LOCAL halo=1 fill directly (not the backend-dispatching
        # ``pad_halo``): this is host-side topography preprocessing on the FULL
        # global field, so it must stay deterministic and never enter the
        # MPI/SPMD exchange path even if a distributed halo backend is active.
        p = pad_halo_local(field, None)
        neighbour_sum = (
            p[:, :-2, 1:-1] + p[:, 2:, 1:-1]    # i-1, i+1
            + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]  # j-1, j+1
        )
        smoothed = (field + neighbour_sum) / 5.0   # self + 4 cross-face neighbours
        field = 0.5 * orig + 0.5 * smoothed
    return np.asarray(field)


def _laplacian_smooth_gaussian(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    """Simple Laplacian smoothing on a Gaussian grid (n_lat, n_lon).

    Uses periodic boundary in longitude, clamped at poles.
    """
    if passes <= 0:
        return arr
    result = arr.copy()
    n_lat, n_lon = arr.shape
    for _ in range(passes):
        smoothed = result.copy()
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


def smooth_phis_cubed_sphere(
    phis: jnp.ndarray,
    smoothing_passes: int = 4,
    edge_blend_strength: float = 0.3,
) -> jnp.ndarray:
    """Apply standard cubed-sphere topography smoothing to a phis field.

    Applies the same Laplacian + edge-blend pipeline used by
    :func:`load_real_topography` so that ERA5-derived or other externally
    regridded phis fields receive equivalent gradient reduction at face
    boundaries before being used as model initial conditions.

    Parameters
    ----------
    phis : (6, n, n) surface geopotential [m^2/s^2]
    smoothing_passes : int
        Number of Laplacian smoothing passes.  Default matches
        ``TopographyConfig.smoothing_passes = 4``.
    edge_blend_strength : float
        Face-edge blend strength.  Default matches
        ``TopographyConfig.edge_blend_strength = 0.3``.

    Returns
    -------
    (6, n, n) smoothed surface geopotential [m^2/s^2]
    """
    phis_np = np.asarray(phis)
    phis_np = _laplacian_smooth_cubed_sphere(phis_np, passes=smoothing_passes)
    return blend_scalar_cube_edges_2d(jnp.asarray(phis_np), strength=edge_blend_strength)


def smooth_phis_gaussian(
    phis: jnp.ndarray,
    smoothing_passes: int = 4,
) -> jnp.ndarray:
    """Apply lat-lon (Gaussian-grid) topography smoothing to a phis field.

    Lat-lon analogue of :func:`smooth_phis_cubed_sphere`: applies the same
    Laplacian smoothing used by :func:`load_real_topography` on a regular
    lat-lon grid (periodic in longitude, clamped at the poles) so that
    ERA5-derived or other externally regridded ``phis`` fields receive
    equivalent gradient reduction before being used as model initial
    conditions.  Without it, the raw regridded ERA5 orography (peaks
    ~5.6e4 m^2/s^2) drives an unbalanced pressure-gradient force that blows
    up the coarse lat-lon dycore at step ~0.

    Parameters
    ----------
    phis : (n_lat, n_lon) surface geopotential [m^2/s^2]
    smoothing_passes : int
        Number of Laplacian smoothing passes.  Default matches
        ``TopographyConfig.smoothing_passes = 4`` (== the cube default).

    Returns
    -------
    (n_lat, n_lon) smoothed surface geopotential [m^2/s^2]
    """
    phis_np = np.asarray(phis)
    return _laplacian_smooth_gaussian(phis_np, passes=smoothing_passes)


def _laplacian_smooth_voronoi(
    arr: np.ndarray,
    cells_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    passes: int = 1,
) -> np.ndarray:
    """Laplacian smoothing of a cell-centred field on an SCVT/Voronoi mesh.

    The unstructured-mesh analogue of :func:`_laplacian_smooth_cubed_sphere`:
    each pass replaces a cell with the mean of itself and its edge-neighbours,
    then blends the result halfway back toward the *original* field
    (``0.5*arr + 0.5*smoothed``) so the smoothing stays anchored and cannot
    drift far from the input.

    Parameters
    ----------
    arr : (nCells,) cell-centred field to smooth.
    cells_on_cell : (maxEdges, nCells) int — 0-based neighbour-cell indices per
        edge of each cell, with ``-1`` in unused slots (rows
        ``i >= n_edges_on_cell[c]``), matching the ``VoronoiMesh.cellsOnCell``
        construction.
    n_edges_on_cell : (nCells,) int — number of edges (= neighbours) per cell.
    passes : int — number of smoothing passes (``<= 0`` is a no-op).

    Returns
    -------
    (nCells,) smoothed field (same dtype as ``arr``).
    """
    if passes <= 0:
        return arr
    coc = np.asarray(cells_on_cell)
    if coc.ndim != 2:
        raise ValueError(
            "cells_on_cell must be 2-D (maxEdges, nCells); got shape "
            f"{coc.shape}"
        )
    n_edges = np.asarray(n_edges_on_cell)
    if n_edges.shape[0] != coc.shape[1] or arr.shape[0] != coc.shape[1]:
        raise ValueError(
            "arr, n_edges_on_cell, and cells_on_cell must agree on nCells: "
            f"arr={arr.shape}, n_edges_on_cell={n_edges.shape}, "
            f"cells_on_cell={coc.shape}"
        )
    valid = coc >= 0                       # (maxEdges, nCells)
    safe_idx = np.where(valid, coc, 0)     # clamp -1 -> 0; masked out below
    # self + valid neighbours; +1.0 counts the cell itself.
    count = n_edges.astype(np.float64) + 1.0
    arr0 = arr.astype(np.float64, copy=True)
    result = arr0.copy()
    for _ in range(passes):
        neigh = result[safe_idx]                              # (maxEdges, nCells)
        neigh_sum = np.where(valid, neigh, 0.0).sum(axis=0)   # (nCells,)
        smoothed = (result + neigh_sum) / count
        result = 0.5 * arr0 + 0.5 * smoothed
    return result.astype(arr.dtype)


def smooth_phis_voronoi(
    phis: jnp.ndarray,
    cells_on_cell,
    n_edges_on_cell,
    smoothing_passes: int = 4,
) -> jnp.ndarray:
    """Laplacian-smooth an ERA5-derived phis field on an SCVT/Voronoi mesh.

    The MPAS analogue of :func:`smooth_phis_cubed_sphere`.  Raw regridded ERA5
    surface geopotential retains grid-scale roughness over steep terrain
    (Himalaya/Andes/Antarctica) on a coarse Voronoi mesh; the TRiSK
    pressure-gradient amplifies those cell-to-cell gradients to O(dx^-1)
    spurious force, which drives a localized wind runaway / blowup within days
    from an ERA5 initial condition.  Smoothing phis before it is used as an IC
    reduces those gradients (the caller still applies a barometric ``p_s``
    correction for hydrostatic consistency).  There is no cube-edge blend — a
    Voronoi mesh has no face boundaries.

    Parameters
    ----------
    phis : (nCells,) surface geopotential [m^2/s^2].
    cells_on_cell : (maxEdges, nCells) ``VoronoiMesh.cellsOnCell``.
    n_edges_on_cell : (nCells,) ``VoronoiMesh.nEdgesOnCell``.
    smoothing_passes : int
        Number of Laplacian passes.  Default matches
        ``TopographyConfig.smoothing_passes = 4`` and the cubed-sphere path.

    Returns
    -------
    (nCells,) smoothed surface geopotential [m^2/s^2].
    """
    smoothed = _laplacian_smooth_voronoi(
        np.asarray(phis),
        np.asarray(cells_on_cell),
        np.asarray(n_edges_on_cell),
        passes=smoothing_passes,
    )
    return jnp.asarray(smoothed)


def _target_grid_degrees(grid):
    """Return target grid centers in degrees and grid metadata.

    Returns ``(target_lat_2d, target_lon_2d, is_gaussian, grid_spacing)``
    where the lat/lon arrays are in degrees and longitude is in [0, 360).

    Unstructured Voronoi/MPAS meshes expose ``grid_lat``/``grid_lon`` as
    rank-1 ``(nCells,)`` cell centres (``latCell``/``lonCell``); they have
    no structured ``n``/``n_lat``, so the cell-count is used to estimate a
    mean angular cell spacing.  ``is_gaussian`` is False for them (they are
    not a regular lat-lon mesh); callers that smooth must additionally guard
    on the rank-1 target shape (no structured/cube smoother applies).
    """
    grid_lat = np.asarray(grid.grid_lat)
    grid_lon = np.asarray(grid.grid_lon)
    # Classify by coordinate rank, not attribute presence.  Cubed-sphere stores
    # grid_lat as (6, n, n) (ndim 3); the structured lat-lon meshes — Gaussian
    # AND the regular lat-lon grid — store it as (n_lat, n_lon) (ndim 2).  The
    # old ``hasattr(grid, 'n_lat') and not hasattr(grid, 'n')`` heuristic
    # mis-classified the lat-lon grid (which also exposes ``n``) as
    # cubed-sphere, so its 2-D field was routed through the cubed-sphere
    # smoother + cube-edge blend instead of the structured lat-lon (periodic-lon,
    # pole-clamped) smoother.  ``is_structured_latlon`` keeps the variable's
    # downstream meaning (1-D lat/lon meshgrid + gaussian smoother, no cube-edge
    # blend) — Gaussian and regular lat-lon share that path.
    is_gaussian = grid_lat.ndim == 2

    if is_gaussian:
        target_lat = np.asarray(grid.lat) * 180.0 / np.pi
        target_lon = (np.asarray(grid.lon) * 180.0 / np.pi) % 360.0
        target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
        grid_spacing = 180.0 / grid.n_lat
    else:
        target_lat_2d = grid_lat * 180.0 / np.pi
        target_lon_2d = (grid_lon * 180.0 / np.pi) % 360.0
        if hasattr(grid, "n"):
            grid_spacing = 90.0 / grid.n
        else:
            # Unstructured Voronoi/MPAS: no structured ``n``.  Estimate the
            # mean angular cell size from the cell count — the whole sphere
            # (4π sr) split over nCells cells gives a linear angular extent
            # ~sqrt(4π/nCells) rad per cell (used only as the sub-grid
            # land-fraction sampling box width).
            ncols = int(np.asarray(grid_lat).size)
            grid_spacing = float(
                np.sqrt(4.0 * np.pi / max(ncols, 1)) * 180.0 / np.pi
            )

    return target_lat_2d, target_lon_2d, is_gaussian, grid_spacing


def load_land_fraction(
    grid,
    path: str,
    var_name: str = "",
) -> jnp.ndarray:
    """Load a land-sea mask from a NetCDF file, regridded to the model grid.

    A standalone entry point for the land fraction alone (independent of
    topography), so a real land-sea mask can be used with any
    ``topography`` setting — including ``flat``.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        Land-sea-mask NetCDF (CMIP6 ``sftlf`` percent, ERA5 ``lsm``
        fraction, etc.).
    var_name : str, optional
        Mask variable name; empty → auto-detect.

    Returns
    -------
    jax.Array
        Land fraction in [0, 1], shape ``(6, n, n)`` for cubed-sphere or
        ``(n_lat, n_lon)`` for Gaussian.
    """
    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    f_land = _load_land_fraction_file(
        path, var_name, target_lat_2d, target_lon_2d
    )
    return jnp.asarray(f_land)


def load_land_albedo(
    grid,
    path: str,
    var_name: str = "",
    month: int | None = None,
) -> jnp.ndarray:
    """Load a land surface albedo field, regridded to the model grid.

    Accepts a regular lat-lon NetCDF albedo (e.g. the ICON-extpar ALB
    broadband albedo remapped to 0.25 deg lat-lon at
    ``/work/bd1083/b309178/diffESM/land_data/
    extpar_albedo_latlon_0p25deg.nc``).  When the field carries a time
    axis (monthly climatology), ``month`` (1-12) selects a single month;
    otherwise the annual mean is used.

    Auto-detects percent vs. fraction (max > 1.5 → divide by 100).

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        NetCDF file containing land albedo.
    var_name : str, optional
        Albedo variable name; empty → auto-detect (ALB, al, alb, albedo,
        surface_albedo, fal).
    month : int, optional
        1-12 to pick a single month from a monthly file; None → annual
        mean (or the only field if the file is already time-collapsed).

    Returns
    -------
    jax.Array
        Land albedo (fraction in [0, 1]), shape ``(6, n, n)`` for
        cubed-sphere or ``(n_lat, n_lon)`` for Gaussian.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        if var_name:
            alb_var = var_name
        else:
            alb_var = None
            for c in ("ALB", "al", "alb", "albedo", "surface_albedo",
                      "fal", "Albedo"):
                if c in all_names:
                    alb_var = c
                    break
            if alb_var is None:
                data_vars = [
                    v for v in ds.data_vars
                    if ds[v].ndim >= 2
                    and not str(v).endswith(("_bnds", "_bounds"))
                ]
                if not data_vars:
                    raise KeyError(
                        f"No data variables in albedo file {path}"
                    )
                alb_var = data_vars[0]

        lat_var = None
        for c in ("lat", "latitude", "y", "Y"):
            if c in all_names:
                lat_var = c
                break
        lon_var = None
        for c in ("lon", "longitude", "x", "X"):
            if c in all_names:
                lon_var = c
                break
        if lat_var is None or lon_var is None:
            raise KeyError(
                f"Cannot detect lat/lon in albedo file {path}; "
                f"available: {sorted(all_names)}"
            )

        lat_src = ds[lat_var].values.astype(np.float64)
        lon_src = ds[lon_var].values.astype(np.float64)
        alb_data = ds[alb_var].values.astype(np.float64)
    finally:
        ds.close()

    # Reduce time / level dims to 2-D
    if alb_data.ndim >= 3 and month is not None:
        if not 1 <= month <= alb_data.shape[0]:
            raise ValueError(
                f"month={month} out of range [1, {alb_data.shape[0]}]"
            )
        alb_data = alb_data[month - 1]
    while alb_data.ndim > 2:
        # Time-mean over leading axis; squeeze any singleton level.
        if alb_data.shape[0] == 1:
            alb_data = alb_data[0]
        else:
            alb_data = np.nanmean(alb_data, axis=0)

    # Longitude in [0, 360), ascending
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    alb_data = alb_data[:, lon_order]

    # Latitude ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        alb_data = alb_data[::-1, :]

    alb_data = np.where(np.isnan(alb_data), 0.0, alb_data)

    # Percent → fraction (extpar ALB is 0-100; ERA5 fal is 0-1)
    if np.nanmax(alb_data) > 1.5:
        alb_data = alb_data / 100.0

    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    alb_grid = _regrid_to_target(
        lat_src, lon_src, alb_data, target_lat_2d, target_lon_2d
    )
    alb_grid = np.clip(alb_grid, 0.0, 1.0)
    return jnp.asarray(alb_grid)


def load_subgrid_orography(
    grid,
    path: str,
    var_name: str = "SSO_STDH",
) -> jnp.ndarray:
    """Load the subgrid orographic standard deviation, regridded to the grid.

    The orographic gravity-wave-drag schemes (Lindzen, McFarlane) launch a
    stress ``tau_0 ∝ h_topo²`` where ``h_topo`` is the standard deviation of
    the unresolved (subgrid) orography in each model cell.  Without a real
    per-column field the schemes fall back to a single global
    ``config.h_topo = 500 m`` everywhere — i.e. a 500 m mountain over the open
    ocean too — which is unphysical.  This loader reads a real subgrid
    orographic stddev (ICON-extpar ``SSO_STDH`` remapped to a regular lat-lon
    grid, e.g. ``extpar_sso_latlon_0p25deg.nc``) so the drag is localized to
    real mountains and is ~0 over flat ocean.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        NetCDF file containing the subgrid orographic stddev [m].
    var_name : str, optional
        Variable name (default ``SSO_STDH``; ICON-extpar convention).

    Returns
    -------
    jax.Array
        Subgrid orographic stddev [m], clipped to >= 0, shape ``(6, n, n)``
        for cubed-sphere or ``(n_lat, n_lon)`` for Gaussian.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        sso_var = var_name if var_name in all_names else None
        if sso_var is None:
            for c in ("SSO_STDH", "sso_stdh", "stddev_orography", "sgh"):
                if c in all_names:
                    sso_var = c
                    break
        if sso_var is None:
            raise KeyError(
                f"Cannot find subgrid orography variable in {path}; "
                f"available: {sorted(ds.data_vars)}"
            )

        lat_var = None
        for c in ("lat", "latitude", "y", "Y"):
            if c in all_names:
                lat_var = c
                break
        lon_var = None
        for c in ("lon", "longitude", "x", "X"):
            if c in all_names:
                lon_var = c
                break
        if lat_var is None or lon_var is None:
            raise KeyError(
                f"Cannot detect lat/lon in SSO file {path}; "
                f"available: {sorted(all_names)}"
            )

        lat_src = ds[lat_var].values.astype(np.float64)
        lon_src = ds[lon_var].values.astype(np.float64)
        sso_data = ds[sso_var].values.astype(np.float64)
    finally:
        ds.close()

    while sso_data.ndim > 2:
        sso_data = sso_data[0]

    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    sso_data = sso_data[:, lon_order]

    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        sso_data = sso_data[::-1, :]

    sso_data = np.where(np.isnan(sso_data), 0.0, sso_data)

    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    sso_grid = _regrid_to_target(
        lat_src, lon_src, sso_data, target_lat_2d, target_lon_2d
    )
    sso_grid = np.maximum(sso_grid, 0.0)
    return jnp.asarray(sso_grid)


def load_real_topography(
    grid,
    config: TopographyConfig | None = None,
    data_path: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load real topography from a NetCDF file and regrid to model grid.

    Supports any regular lat-lon elevation dataset (ETOPO, GEBCO,
    GMTED2010, etc.). The elevation field is bilinearly interpolated
    to model grid centers, smoothed to remove 2Δx noise, and
    (for cubed-sphere grids) edge-blended at face boundaries.

    Land fraction is taken from ``config.land_mask_path`` when set (a
    true land-sea mask such as CMIP6 ``sftlf`` or ERA5 ``lsm``);
    otherwise it is derived from sub-grid sampling of the elevation
    field (the fraction of sub-grid points with elevation > 0).

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    config : TopographyConfig, optional
        Configuration. If None, uses defaults.
    data_path : str, optional
        Override path to NetCDF file (takes precedence over config.path).

    Returns
    -------
    phis : jax.Array
        Surface geopotential [m^2/s^2].
        Shape (6, n, n) for cubed-sphere or (n_lat, n_lon) for Gaussian.
    f_land : jax.Array
        Land fraction [0, 1], same shape.
    """
    import xarray as xr

    if config is None:
        config = TopographyConfig(source="file")

    path = data_path or config.path
    if not path:
        raise ValueError(
            "No topography data path provided. "
            "Set config.path or pass data_path."
        )

    # Open dataset
    ds = xr.open_dataset(path)

    # Detect or use configured variable names
    if config.elev_var and config.lat_var and config.lon_var:
        elev_var, lat_var, lon_var = config.elev_var, config.lat_var, config.lon_var
    else:
        elev_var, lat_var, lon_var = _detect_variables(ds)

    # Extract source data
    lat_src = ds[lat_var].values.astype(np.float64)
    lon_src = ds[lon_var].values.astype(np.float64)
    elev_data = ds[elev_var].values.astype(np.float64)
    elev_units = str(ds[elev_var].attrs.get("units", ""))

    # Handle multi-dimensional data (squeeze extra dims)
    while elev_data.ndim > 2:
        elev_data = elev_data[0]

    # Ensure longitude in [0, 360)
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    elev_data = elev_data[:, lon_order]

    # Ensure latitude is sorted ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        elev_data = elev_data[::-1, :]

    # Replace NaN with 0 (ocean)
    elev_data = np.where(np.isnan(elev_data), 0.0, elev_data)

    ds.close()

    # --- geopotential-vs-meters "double-g" guard -----------------------------
    # This loader multiplies elevation [m] by constants.g below; an ERA5-style
    # invariant where 'z' is GEOPOTENTIAL [m²/s²] would be silently inflated
    # ~9.8×.  Detect it by the units attribute (when present) and by magnitude
    # (see _MAX_PLAUSIBLE_ELEV_M).
    _double_g_msg = (
        f"Topography variable {elev_var!r} in {path!r} looks like surface "
        f"GEOPOTENTIAL [m**2 s**-2], not elevation [m] "
        f"(units={elev_units!r}, max |value| = "
        f"{float(np.max(np.abs(elev_data))):.0f}; no Earth elevation exceeds "
        f"~8850 m, threshold {_MAX_PLAUSIBLE_ELEV_M:.0f} m). Multiplying it "
        "by g again (the double-g trap) would inflate phis ~9.8x. Divide the "
        "field by g (legoesm.constants.g) first, or pass the correct "
        "elevation variable via config.elev_var."
    )
    _units_norm = elev_units.replace(" ", "").replace("**", "^").lower()
    if _units_norm in ("m^2s^-2", "m^2/s^2", "m2s-2", "m2/s2", "m^2s-2"):
        raise ValueError(_double_g_msg)
    if float(np.max(np.abs(elev_data))) > _MAX_PLAUSIBLE_ELEV_M:
        raise ValueError(_double_g_msg)

    # Use protocol for grid detection (classification fixed in
    # _target_grid_degrees: by coordinate rank, so the lat-lon grid is no longer
    # mis-routed into the cubed-sphere smoother).
    target_lat_2d, target_lon_2d, is_gaussian, grid_spacing = (
        _target_grid_degrees(grid)
    )

    # Regrid elevation
    z_s = _regrid_to_target(lat_src, lon_src, elev_data,
                            target_lat_2d, target_lon_2d)

    # Land fraction: prefer an explicit land-sea-mask file (true land
    # fraction, including below-sea-level land); otherwise derive it
    # from sub-grid elevation sampling (elevation > 0).
    if config.land_mask_path:
        f_land = _load_land_fraction_file(
            config.land_mask_path, config.land_mask_var,
            target_lat_2d, target_lon_2d,
        )
    else:
        f_land = _derive_land_fraction(lat_src, lon_src, elev_data,
                                       target_lat_2d, target_lon_2d,
                                       grid_spacing)

    # Clip negative elevations if requested
    if config.clip_negative:
        z_s = np.maximum(z_s, 0.0)

    # Smoothing.  Unstructured Voronoi/MPAS fields are rank-1 ``(nCells,)``
    # with no structured neighbour stencil, so neither the gaussian (2-D)
    # nor the cubed-sphere (6, n, n) smoother applies — the point-sampled
    # field is used as-is (a mesh-native Laplacian smoother is a follow-up).
    is_unstructured = np.asarray(z_s).ndim == 1
    if is_unstructured:
        pass
    elif is_gaussian:
        z_s = _laplacian_smooth_gaussian(z_s, passes=config.smoothing_passes)
    else:
        z_s = _laplacian_smooth_cubed_sphere(z_s, passes=config.smoothing_passes)

    # Edge blending for cubed-sphere
    if not is_gaussian and not is_unstructured and config.edge_blend_strength > 0:
        z_s_jax = jnp.array(z_s)
        z_s_jax = blend_scalar_cube_edges_2d(
            z_s_jax,
            strength=config.edge_blend_strength,
            width=config.edge_blend_width,
        )
        z_s = np.asarray(z_s_jax)

    # Convert to JAX arrays
    phis = jnp.array(constants.g * z_s)
    f_land = jnp.array(np.clip(f_land, 0.0, 1.0))

    return phis, f_land
