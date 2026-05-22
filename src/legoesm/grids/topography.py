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
    lat = grid.lat
    lon = grid.lon

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
    dlat = grid.lat - lat0
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
    lat = grid.lat
    lon = grid.lon

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
        Optional path to a separate land-sea-mask NetCDF file — a regular
        lat-lon mask (CMIP6 ``sftlf``, ERA5 ``lsm``) or an ICON
        unstructured ``extpar`` file (``FR_LAND``); the grid type is
        auto-detected.  When set, the land fraction is taken from this
        file instead of being derived from ``elevation > 0`` — which
        avoids misclassifying below-sea-level land (Caspian/Dead
        Sea/Netherlands) as ocean.
    land_mask_var : str
        Variable name in ``land_mask_path``.  Empty → auto-detect among
        ``FR_LAND``, ``sftlf``, ``lsm``, ``land_sea_mask``, ``land``,
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


def _regrid_to_target(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    elev_data: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    """Regrid elevation data from regular lat-lon to target grid points.

    Uses bilinear interpolation with longitude wrapping for periodicity.

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
    from scipy.interpolate import RegularGridInterpolator

    # Wrap longitude for interpolation continuity
    lon_wrapped = np.concatenate([
        lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0
    ])
    elev_wrapped = np.concatenate([
        elev_data[:, -1:], elev_data, elev_data[:, :1]
    ], axis=1)

    interp = RegularGridInterpolator(
        (lat_src, lon_wrapped), elev_wrapped,
        method="linear", bounds_error=False, fill_value=0.0,
    )

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
    from scipy.interpolate import RegularGridInterpolator

    # Wrap longitude
    lon_wrapped = np.concatenate([
        lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0
    ])
    elev_wrapped = np.concatenate([
        elev_data[:, -1:], elev_data, elev_data[:, :1]
    ], axis=1)

    interp = RegularGridInterpolator(
        (lat_src, lon_wrapped), elev_wrapped,
        method="linear", bounds_error=False, fill_value=0.0,
    )

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
    """Simple Laplacian smoothing on a cubed-sphere field (6, n, n).

    Replaces each interior cell with (1-w)*self + w*avg_neighbors where
    w = 0.5. Boundary cells are handled by wrapping via halo padding.
    """
    if passes <= 0:
        return arr
    result = arr.copy()
    for _ in range(passes):
        smoothed = result.copy()
        for face in range(6):
            n = result.shape[1]
            # Interior 4-point average
            for i in range(n):
                for j in range(n):
                    # Use available neighbors with boundary clamping
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
        result = 0.5 * arr + 0.5 * smoothed  # blend toward smoothed
    return result


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


def _target_grid_degrees(grid):
    """Return target grid centers in degrees and grid metadata.

    Returns ``(target_lat_2d, target_lon_2d, is_gaussian, grid_spacing)``
    where the lat/lon arrays are in degrees and longitude is in [0, 360).
    """
    grid_lat = np.asarray(grid.grid_lat)
    grid_lon = np.asarray(grid.grid_lon)
    is_gaussian = hasattr(grid, 'n_lat') and not hasattr(grid, 'n')

    if is_gaussian:
        target_lat = np.asarray(grid.lat) * 180.0 / np.pi
        target_lon = (np.asarray(grid.lon) * 180.0 / np.pi) % 360.0
        target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
        grid_spacing = 180.0 / grid.n_lat
    else:
        target_lat_2d = grid_lat * 180.0 / np.pi
        target_lon_2d = (grid_lon * 180.0 / np.pi) % 360.0
        grid_spacing = 90.0 / grid.n

    return target_lat_2d, target_lon_2d, is_gaussian, grid_spacing


def _load_land_fraction_icon(
    grid,
    path: str,
    var_name: str = "",
) -> jnp.ndarray:
    """Regrid an ICON unstructured land-sea mask to the model grid.

    ICON ``extpar`` files store the land fraction (``FR_LAND``) on the
    unstructured icosahedral grid: a 1-D ``(cell,)`` field with per-cell
    centroid coordinates ``clon``/``clat`` in radians.  Regridding uses a
    KD-tree nearest-neighbour search on the unit sphere — the same
    approach as the ICON SST/SIC forcing in ``forcing/amip.py``.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid (``grid_lat``/``grid_lon`` in radians).
    path : str
        ICON unstructured NetCDF (e.g. ``icon_extpar_*.nc``).
    var_name : str, optional
        Mask variable name; empty → auto-detect (``FR_LAND`` first).

    Returns
    -------
    jax.Array
        Land fraction in [0, 1], same shape as ``grid.grid_lat``.
    """
    import xarray as xr
    from scipy.spatial import cKDTree

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        if var_name:
            mask_var = var_name
        else:
            mask_var = None
            for c in ("FR_LAND", "fr_land", "sftlf", "lsm",
                      "land_sea_mask", "land", "land_area_fraction"):
                if c in all_names:
                    mask_var = c
                    break
            if mask_var is None:
                raise KeyError(
                    f"No land-fraction variable in ICON file {path}; "
                    f"available: {sorted(ds.data_vars)}"
                )
        clon = ds["clon"].values.astype(np.float64)   # radians
        clat = ds["clat"].values.astype(np.float64)
        mask = ds[mask_var].values.astype(np.float64)
    finally:
        ds.close()

    # Land fraction is static — collapse any leading (e.g. time) axis.
    while mask.ndim > 1:
        mask = mask[0]
    mask = np.where(np.isnan(mask), 0.0, mask)
    if np.nanmax(mask) > 1.5:        # percent → fraction
        mask = mask / 100.0
    mask = np.clip(mask, 0.0, 1.0)

    # KD-tree nearest-neighbour from ICON centroids (3-D unit sphere).
    x_s = np.cos(clat) * np.cos(clon)
    y_s = np.cos(clat) * np.sin(clon)
    z_s = np.sin(clat)
    tree = cKDTree(np.stack([x_s, y_s, z_s], axis=-1))

    grid_lat = np.asarray(grid.grid_lat)   # radians
    grid_lon = np.asarray(grid.grid_lon)   # radians
    target_shape = grid_lat.shape
    x_t = np.cos(grid_lat.ravel()) * np.cos(grid_lon.ravel())
    y_t = np.cos(grid_lat.ravel()) * np.sin(grid_lon.ravel())
    z_t = np.sin(grid_lat.ravel())
    _, idx = tree.query(np.stack([x_t, y_t, z_t], axis=-1))

    return jnp.asarray(mask[idx].reshape(target_shape))


def load_land_fraction(
    grid,
    path: str,
    var_name: str = "",
) -> jnp.ndarray:
    """Load a land-sea mask from a NetCDF file, regridded to the model grid.

    A standalone entry point for the land fraction alone (independent of
    topography), so a real land-sea mask can be used with any
    ``topography`` setting — including ``flat``.  Both regular lat-lon
    masks (CMIP6 ``sftlf``, ERA5 ``lsm``) and ICON unstructured masks
    (``icon_extpar_*.nc`` ``FR_LAND``) are supported; the grid type is
    auto-detected.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        Land-sea-mask NetCDF.
    var_name : str, optional
        Mask variable name; empty → auto-detect.

    Returns
    -------
    jax.Array
        Land fraction in [0, 1], shape ``(6, n, n)`` for cubed-sphere or
        ``(n_lat, n_lon)`` for Gaussian.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    is_icon = (
        "cell" in ds.dims and "clon" in ds.coords and "clat" in ds.coords
    )
    ds.close()

    if is_icon:
        return _load_land_fraction_icon(grid, path, var_name)

    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    f_land = _load_land_fraction_file(
        path, var_name, target_lat_2d, target_lon_2d
    )
    return jnp.asarray(f_land)


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

    # Use protocol for grid detection
    target_lat_2d, target_lon_2d, is_gaussian, grid_spacing = (
        _target_grid_degrees(grid)
    )

    # Regrid elevation
    z_s = _regrid_to_target(lat_src, lon_src, elev_data,
                            target_lat_2d, target_lon_2d)

    # Land fraction: prefer an explicit land-sea-mask file (true land
    # fraction, including below-sea-level land; regular lat-lon or ICON
    # unstructured); otherwise derive it from sub-grid elevation
    # sampling (elevation > 0).
    if config.land_mask_path:
        f_land = np.asarray(load_land_fraction(
            grid, config.land_mask_path, config.land_mask_var,
        ))
    else:
        f_land = _derive_land_fraction(lat_src, lon_src, elev_data,
                                       target_lat_2d, target_lon_2d,
                                       grid_spacing)

    # Clip negative elevations if requested
    if config.clip_negative:
        z_s = np.maximum(z_s, 0.0)

    # Smoothing
    if is_gaussian:
        z_s = _laplacian_smooth_gaussian(z_s, passes=config.smoothing_passes)
    else:
        z_s = _laplacian_smooth_cubed_sphere(z_s, passes=config.smoothing_passes)

    # Edge blending for cubed-sphere
    if not is_gaussian and config.edge_blend_strength > 0:
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
