"""WOA18 (World Ocean Atlas 2018) ocean initialization.

Provides realistic temperature and salinity initial conditions by
interpolating WOA18 annual climatology to the model grid and vertical
coordinate.  Supports cubed-sphere, lat-lon, MPAS, and spectral grids.

WOA18 data are expected as Zarr stores or NetCDF files on a 1-degree
lat-lon grid with standard depth levels.  If files are not available, a
latitude-dependent analytical fallback is provided.

References
----------
Locarnini, R. A., et al. (2019): World Ocean Atlas 2018, Volume 1:
    Temperature. NOAA Atlas NESDIS 81, 52 pp.
Zweng, M. M., et al. (2019): World Ocean Atlas 2018, Volume 2:
    Salinity. NOAA Atlas NESDIS 82, 50 pp.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.vertical import OceanZStarCoordinate


# ==============================================================================
# WOA18 standard depth levels [m] (57 levels, surface to 5500 m)
# ==============================================================================
WOA_DEPTHS = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
    80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300,
    325, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750,
    800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300,
    1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800,
    1850, 1900, 1950, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
    2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
    3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600,
    4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500,
], dtype=np.float64)


def _interp_profile_to_z_coord(
    profile: np.ndarray,
    woa_depths: np.ndarray,
    z_coord: OceanZStarCoordinate,
) -> np.ndarray:
    """Interpolate a 1-D WOA profile onto model z-star full levels.

    Parameters
    ----------
    profile : array, shape (n_woa,)
        WOA profile values (T or S).
    woa_depths : array, shape (n_woa,)
        WOA depth levels [m] (positive downward).
    z_coord : OceanZStarCoordinate

    Returns
    -------
    array, shape (nlev,)
        Profile on model full levels.
    """
    # Model full-level depths (positive), z_full_ref is negative
    model_depths = np.abs(np.asarray(z_coord.z_full_ref))

    # Remove NaN entries from WOA profile
    valid = ~np.isnan(profile)
    if valid.sum() < 2:
        # Not enough valid data — return surface value everywhere
        return np.full(z_coord.n_levels, profile[valid][0] if valid.any() else 0.0)

    return np.interp(model_depths, woa_depths[valid], profile[valid])


def _analytical_woa_profiles(
    lat_deg: np.ndarray,
    z_coord: OceanZStarCoordinate,
) -> tuple[np.ndarray, np.ndarray]:
    """Latitude-dependent analytical T/S profiles mimicking WOA18.

    Provides a reasonable approximation when WOA18 NetCDF files are not
    available.  Temperature uses a latitude-dependent surface value with
    exponential decay.  Salinity uses a latitude-dependent profile with
    a subsurface maximum (subtropical halocline).

    Parameters
    ----------
    lat_deg : array, shape (...)
        Latitude in degrees.
    z_coord : OceanZStarCoordinate

    Returns
    -------
    T : array, shape (..., nlev) — potential temperature [degC]
    S : array, shape (..., nlev) — salinity [PSU]
    """
    z_full = np.asarray(z_coord.z_full_ref)  # negative, (nlev,)
    lat = np.asarray(lat_deg)

    # --- Temperature ---
    # SST: ~28 degC at equator, ~0 degC at poles
    T_water_init_C = 28.0 * np.cos(np.radians(lat)) ** 2
    T_deep = 1.5  # Bottom water ~1.5 degC globally
    scale = 500.0  # e-folding depth [m]

    # z_full is negative; exp(z/scale) = exp(-depth/scale)
    T = T_deep + (T_water_init_C[..., np.newaxis] - T_deep) * np.exp(z_full / scale)

    # --- Salinity ---
    # Surface: fresher near equator (ITCZ precip) and poles; saltier subtropics
    S_surface = 34.5 + 1.0 * np.cos(np.radians(2 * lat)) ** 2
    S_deep = 34.7  # Deep water ~34.7 PSU

    # Subsurface salinity maximum at ~100m in subtropics
    S_scale = 1000.0
    S = S_deep + (S_surface[..., np.newaxis] - S_deep) * np.exp(z_full / S_scale)

    return T, S


def _open_woa_dataset(path: str | Path):
    """Open a WOA18 file as xarray Dataset (Zarr or NetCDF).

    Uses ``decode_times=False`` because the NCEI WOA18 distribution
    encodes the time axis as ``"months since 1955-01-01"``, which
    xarray cannot decode without ``cftime`` and which is irrelevant
    for our use (we only read the annual-mean climatology, which is
    a single time slice we treat as static).
    """
    import os
    import xarray as xr
    path_str = str(path)
    if os.path.isdir(path_str) or path_str.endswith(".zarr"):
        return xr.open_zarr(path_str, decode_times=False)
    return xr.open_dataset(path_str, decode_times=False)


def load_woa18(
    T_path: str | Path | None = None,
    S_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load WOA18 annual climatology from Zarr stores or NetCDF files.

    Parameters
    ----------
    T_path : path or None
        Path to WOA18 temperature file/store.
    S_path : path or None
        Path to WOA18 salinity file/store.

    Returns
    -------
    T_woa : array, shape (n_lat, n_lon, n_depth)
    S_woa : array, shape (n_lat, n_lon, n_depth)
    lat_woa : array, shape (n_lat,)  — degrees
    lon_woa : array, shape (n_lon,)  — degrees [0, 360)
    """
    ds_T = _open_woa_dataset(T_path)
    # Support both raw WOA ("t_an") and ESMValTool-formatted ("thetao") files
    _t_var = "t_an" if "t_an" in ds_T else "thetao"
    T_woa = np.array(ds_T[_t_var].values[0, :, :, :])  # (depth, lat, lon)
    lat_woa = np.array(ds_T["lat"].values)
    lon_woa = np.array(ds_T["lon"].values)
    ds_T.close()

    ds_S = _open_woa_dataset(S_path)
    _s_var = "s_an" if "s_an" in ds_S else "so"
    S_woa = np.array(ds_S[_s_var].values[0, :, :, :])
    ds_S.close()

    # Transpose to (lat, lon, depth) for easier interpolation
    T_woa = np.transpose(T_woa, (1, 2, 0))
    S_woa = np.transpose(S_woa, (1, 2, 0))

    return T_woa, S_woa, lat_woa, lon_woa


def _nearest_neighbor_2d(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor interpolation from regular lat-lon to target points.

    Parameters
    ----------
    target_lat, target_lon : array, shape (...)
        Target coordinates in degrees.
    src_lat : array, shape (n_lat,)
    src_lon : array, shape (n_lon,)
    field : array, shape (n_lat, n_lon, ...)
        Source field.

    Returns
    -------
    array, shape (..., remaining_dims)
    """
    # Find nearest lat/lon indices
    lat_idx = np.argmin(np.abs(src_lat[:, np.newaxis] - target_lat.ravel()[np.newaxis, :]),
                        axis=0)
    # Normalize longitude to source range
    tlon = target_lon.ravel() % 360.0
    slon = src_lon % 360.0
    lon_idx = np.argmin(np.abs(slon[:, np.newaxis] - tlon[np.newaxis, :]),
                        axis=0)

    result = field[lat_idx, lon_idx]  # (n_points, ...)
    out_shape = target_lat.shape + field.shape[2:]
    return result.reshape(out_shape)


def init_ocean_from_woa(
    grid,
    z_coord: OceanZStarCoordinate,
    T_path: str | Path | None = None,
    S_path: str | Path | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize ocean T and S from WOA18 climatology.

    If ``T_path`` and ``S_path`` are provided, loads WOA18 NetCDF files
    and interpolates to the model grid.  Otherwise, uses analytical
    latitude-dependent profiles mimicking WOA18.

    Parameters
    ----------
    grid : CubedSphereGrid, LatLonGrid, VoronoiMesh, or GaussianGrid
    z_coord : OceanZStarCoordinate
    T_path, S_path : path or None
        Paths to WOA18 NetCDF annual climatology files.

    Returns
    -------
    T : jax.Array, shape (..., nlev) — potential temperature [degC]
    S : jax.Array, shape (..., nlev) — salinity [PSU]
    """
    # Extract latitude in degrees from the grid.
    # Dispatch order matters: GaussianGrid has both 'lat' and 'lat2d',
    # while CubedSphereGrid has 'lat' (3D) but no 'lat2d'.
    if hasattr(grid, 'lat2d'):
        # GaussianGrid: lat2d in radians, shape (n_lat, n_lon)
        lat_deg = np.asarray(grid.lat2d) * (180.0 / np.pi)
        lon_2d = np.asarray(grid.lon) * (180.0 / np.pi)
        lon_deg = np.broadcast_to(lon_2d[np.newaxis, :], lat_deg.shape)
    elif hasattr(grid, 'latCell'):
        # VoronoiMesh: latCell in radians, shape (nCells,)
        lat_deg = np.asarray(grid.latCell) * (180.0 / np.pi)
        lon_deg = np.asarray(grid.lonCell) * (180.0 / np.pi)
    elif hasattr(grid, 'lat') and hasattr(grid, 'lon'):
        # CubedSphereGrid or LatLonGrid: lat/lon in radians
        lat_deg = np.asarray(grid.lat) * (180.0 / np.pi)
        lon_deg = np.asarray(grid.lon) * (180.0 / np.pi)
    else:
        raise TypeError(f"Unsupported grid type: {type(grid)}")

    if T_path is not None and S_path is not None:
        # Load from WOA18 NetCDF files
        T_woa, S_woa, lat_woa, lon_woa = load_woa18(T_path, S_path)

        # Nearest-neighbor horizontal interpolation
        T_woa_horiz = _nearest_neighbor_2d(
            lat_deg, lon_deg, lat_woa, lon_woa, T_woa,
        )  # (..., n_woa_depth)
        S_woa_horiz = _nearest_neighbor_2d(
            lat_deg, lon_deg, lat_woa, lon_woa, S_woa,
        )

        # Vertical interpolation to model levels (vectorized over columns)
        flat_shape = (-1, T_woa_horiz.shape[-1])
        T_flat = T_woa_horiz.reshape(flat_shape)
        S_flat = S_woa_horiz.reshape(flat_shape)

        woa_depths = WOA_DEPTHS[:T_woa.shape[-1]]

        T_out = np.stack([
            _interp_profile_to_z_coord(T_flat[i], woa_depths, z_coord)
            for i in range(T_flat.shape[0])
        ]).reshape(lat_deg.shape + (z_coord.n_levels,))

        S_out = np.stack([
            _interp_profile_to_z_coord(S_flat[i], woa_depths, z_coord)
            for i in range(S_flat.shape[0])
        ]).reshape(lat_deg.shape + (z_coord.n_levels,))

        # Fill remaining NaNs with nearest valid value
        T_out = np.nan_to_num(T_out, nan=1.5)
        S_out = np.nan_to_num(S_out, nan=34.7)
        # Replace spurious zeros (land cells missed by nan_to_num)
        T_out = np.where(T_out == 0.0, 1.5,  T_out)
        S_out = np.where(S_out == 0.0, 34.7, S_out)

    else:
        # Analytical fallback
        T_out, S_out = _analytical_woa_profiles(lat_deg, z_coord)

    return jnp.array(T_out), jnp.array(S_out)


# ==============================================================================
# WOA18 BGC nutrient initialization
# ==============================================================================

def init_bgc_from_woa(
    grid,
    z_coord: OceanZStarCoordinate,
    no3_path: str | Path | None = None,
    po4_path: str | Path | None = None,
    si_path:  str | Path | None = None,
) -> dict[str, jnp.ndarray]:
    """Initialize BGC nutrient tracers from WOA18 climatology.

    Loads WOA18 NO3, PO4, and Si annual climatology files and
    interpolates to the model grid using the same nearest-neighbour +
    vertical interpolation scheme as ``init_ocean_from_woa``.

    Parameters
    ----------
    grid : any legoESM grid
    z_coord : OceanZStarCoordinate
    no3_path, po4_path, si_path : path or None
        Paths to WOA18 NetCDF files.  If None, analytical fallback
        is used (linear depth profile from surface to deep values).

    Returns
    -------
    dict with keys "NO3", "PO4", "Si" — each shape (..., nlev)
    Units: mol/m^3 (WOA data are in umol/kg; converted by /1022*1e-3).
    """
    import xarray as xr

    # Extract lat/lon from grid (same dispatch as init_ocean_from_woa)
    if hasattr(grid, "lat2d"):
        lat_deg = np.asarray(grid.lat2d) * (180.0 / np.pi)
        lon_2d  = np.asarray(grid.lon)   * (180.0 / np.pi)
        lon_deg = np.broadcast_to(lon_2d[np.newaxis, :], lat_deg.shape)
    elif hasattr(grid, "latCell"):
        lat_deg = np.asarray(grid.latCell) * (180.0 / np.pi)
        lon_deg = np.asarray(grid.lonCell) * (180.0 / np.pi)
    elif hasattr(grid, "lat") and hasattr(grid, "lon"):
        lat_deg = np.asarray(grid.lat) * (180.0 / np.pi)
        lon_deg = np.asarray(grid.lon) * (180.0 / np.pi)
    else:
        raise TypeError(f"Unsupported grid type: {type(grid)}")

    # ESMValTool-formatted WOA files are already in mol/m^3
    _umol_kg_to_mol_m3 = 1.0

    def _load_woa_tracer(path, varname):
        """Load a single WOA NetCDF tracer and return (field, lat, lon)."""
        ds  = xr.open_dataset(path, decode_times=False)
        # Find the variable — WOA files use short names (no3, po4, si, o_an)
        var = ds[varname] if varname in ds else ds[list(ds.data_vars)[0]]
        # Remove time dimension if present (annual climatology)
        if "time" in var.dims:
            var = var.isel(time=0)
        # Depth coordinate — try common names
        # After squeezing time, remaining dims are (lev/depth, lat, lon)
        # Find each axis by name
        depth_coord = None
        for name in ("lev", "depth", "depth_std", "zlev"):
            if name in var.dims:
                depth_coord = name
                break
        if depth_coord is None:
            # fall back: first dim that is not lat/lon
            for d in var.dims:
                if d not in ("lat", "lon", "latitude", "longitude"):
                    depth_coord = d
                    break
        lat_name = None
        for name in ("lat", "latitude"):
            if name in var.dims:
                lat_name = name
                break
        lon_name = None
        for name in ("lon", "longitude"):
            if name in var.dims:
                lon_name = name
                break
        # Reorder to (lat, lon, depth)
        var = var.transpose(lat_name, lon_name, depth_coord)
        field  = np.asarray(var.values, dtype=np.float64)
        lat_1d = np.asarray(ds[lat_name].values, dtype=np.float64)
        lon_1d = np.asarray(ds[lon_name].values, dtype=np.float64)
        # Standardise lon to [0, 360)
        lon_1d = lon_1d % 360.0
        return field, lat_1d, lon_1d

    def _interp_tracer(path, varname, fallback_surf, fallback_deep):
        """Interpolate one tracer to model grid, or use analytical fallback."""
        if path is None:
            # Analytical: linear increase from surface to deep
            z_norm = np.clip(
                np.abs(np.asarray(z_coord.z_full_ref)) / 1000.0, 0.0, 1.0)
            profile = fallback_surf + (fallback_deep - fallback_surf) * z_norm
            field = np.broadcast_to(
                profile.reshape((1,) * lat_deg.ndim + (z_coord.n_levels,)),
                lat_deg.shape + (z_coord.n_levels,),
            ).copy()
            return jnp.array(field * _umol_kg_to_mol_m3)

        raw, lat_woa, lon_woa = _load_woa_tracer(path, varname)
        woa_d = WOA_DEPTHS[:raw.shape[-1]]

        # Horizontal nearest-neighbour
        horiz = _nearest_neighbor_2d(lat_deg, lon_deg, lat_woa, lon_woa, raw)

        # Vertical interpolation
        flat = horiz.reshape(-1, horiz.shape[-1])
        out_flat = np.stack([
            _interp_profile_to_z_coord(flat[i], woa_d, z_coord)
            for i in range(flat.shape[0])
        ])
        out = out_flat.reshape(lat_deg.shape + (z_coord.n_levels,))
        out = np.nan_to_num(out, nan=fallback_surf)

        # Convert umol/kg -> mol/m^3
        return jnp.array(out * _umol_kg_to_mol_m3)

    # Fallback values in mol/m³ (same units as ESMValTool WOA files)
    # NO3/PO4/Si: umol/kg * 1e-3 * 1025 ~ mol/m3; Fe: nmol/kg * 1e-6 * 1025
    NO3 = _interp_tracer(no3_path, "no3",  fallback_surf=1.0e-3,  fallback_deep=30.0e-3)
    PO4 = _interp_tracer(po4_path, "po4",  fallback_surf=0.1e-3,  fallback_deep=2.0e-3)
    Si  = _interp_tracer(si_path,  "si",   fallback_surf=2.0e-3,  fallback_deep=100.0e-3)

    # Fe: no WOA file — start from very low value so scavenging
    # equilibrates quickly (within ~1 year at k_scav=5/day)
    _Fe_surf = 50.0e-12   # mol/m³  (50 pM — config default)
    _Fe_deep = 200.0e-12  # mol/m³  (200 pM deep)
    _Fe_p = _Fe_surf + (_Fe_deep - _Fe_surf) * np.clip(
        np.abs(np.asarray(z_coord.z_full_ref)) / 1000.0, 0.0, 1.0)
    _Fe_arr = np.broadcast_to(
        _Fe_p.reshape((1,) * lat_deg.ndim + (z_coord.n_levels,)),
        lat_deg.shape + (z_coord.n_levels,),
    ).copy()
    Fe = jnp.array(_Fe_arr.astype(np.float64))
    return {"NO3": NO3, "PO4": PO4, "Si": Si, "Fe": Fe}
