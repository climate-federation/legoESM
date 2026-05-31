"""WOA18 (World Ocean Atlas 2018) ocean initialization.

Provides realistic temperature and salinity initial conditions by
interpolating WOA18 climatology (annual or monthly) to the model grid
and vertical coordinate.  Supports cubed-sphere, lat-lon, MPAS, and
spectral grids.

WOA18 data are expected as Zarr stores or NetCDF files on a 1-degree
lat-lon grid with standard depth levels.  If files are not available, a
latitude-dependent analytical fallback is provided.

OMIP-2 protocol notes
---------------------
The OMIP-2 (CMIP6-OMIP) experimental design (Griffies et al. 2016, GMD)
specifies that ocean simulations start from observed temperature and
salinity climatology to suppress spin-up artefacts.  This module
implements that recipe:

    * Annual or monthly WOA18 climatology (``month=None`` or
      ``month in 1..12``).
    * Bilinear horizontal interpolation (OMIP-2 default).  Nearest-
      neighbour is retained for column-grid debugging.
    * Bathymetry-aware vertical mask so model cells below the local
      water column receive the deep-ocean fill rather than spurious
      surface-extended values.

References
----------
Locarnini, R. A., et al. (2019): World Ocean Atlas 2018, Volume 1:
    Temperature. NOAA Atlas NESDIS 81, 52 pp.
Zweng, M. M., et al. (2019): World Ocean Atlas 2018, Volume 2:
    Salinity. NOAA Atlas NESDIS 82, 50 pp.
Griffies, S. M., et al. (2016): OMIP contribution to CMIP6: Experimental
    and diagnostic protocol for the physical component of the Ocean
    Model Intercomparison Project. Geosci. Model Dev., 9, 3231-3296.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm import constants
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
    month: int | None = None,
    T_var: str | None = None,
    S_var: str | None = None,
    monthly_layout: str = "concatenated",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load WOA18 climatology (annual or monthly) from Zarr/NetCDF.

    Parameters
    ----------
    T_path : path or None
        Path to WOA18 temperature file/store.
    S_path : path or None
        Path to WOA18 salinity file/store.
    month : int or None
        Month index 1..12 to extract from the WOA dataset's time axis.
        ``None`` returns the annual mean (time index 0, which is the
        WOA18 convention for ``*_an`` annual fields).  Monthly fields
        are stored along the ``time`` dimension with length 12 in the
        WOA18 monthly distribution (Vols. 1/2 NetCDF format
        ``woa18_decav_t<MM>_01.nc`` concatenated, or the single-file
        ``woa18_decav_t01_01.nc`` monthly products).
    T_var, S_var : str or None
        Override the dataset variable names.  Defaults: ``"t_an"`` and
        ``"s_an"`` for annual mean, ``"t_an"`` / ``"s_an"`` again for
        the monthly products (they reuse the name with a length-12
        time axis).  Set to the actual variable name (e.g. ``"t_mn"``)
        for non-standard files.
    monthly_layout : {"concatenated", "single_file_per_month"}
        File layout convention:
            * ``"concatenated"`` (default): one file with a length-12
              time axis.  ``month=N`` reads index ``N - 1``.
            * ``"single_file_per_month"``: caller has already selected
              the file matching the desired month.  ``month`` must be
              ``None`` (the loader reads the single time slice);
              passing ``month is not None`` raises ``ValueError`` to
              prevent silent month mismatch where ``month=2`` is
              requested but the file is a January climatology.

    Returns
    -------
    T_woa : array, shape (n_lat, n_lon, n_depth)
    S_woa : array, shape (n_lat, n_lon, n_depth)
    lat_woa : array, shape (n_lat,)  — degrees
    lon_woa : array, shape (n_lon,)  — degrees [0, 360)
    """
    if monthly_layout not in ("concatenated", "single_file_per_month"):
        raise ValueError(
            f"monthly_layout must be 'concatenated' or "
            f"'single_file_per_month'; got {monthly_layout!r}."
        )
    if monthly_layout == "single_file_per_month" and month is not None:
        raise ValueError(
            "monthly_layout='single_file_per_month' expects the caller "
            "to have pre-selected the file matching the desired month; "
            "pass month=None and the loader will read the single time "
            "slice.  Got month="
            f"{month}."
        )
    if month is not None and not (1 <= month <= 12):
        raise ValueError(
            f"month must be 1..12 or None (annual); got {month}."
        )
    requested_idx = 0 if month is None else (month - 1)
    t_name = T_var if T_var is not None else "t_an"
    s_name = S_var if S_var is not None else "s_an"

    def _read(ds, var_name, label, path):
        if var_name not in ds.data_vars:
            avail = list(ds.data_vars)
            ds.close()
            raise ValueError(
                f"WOA {label} file {path!r} has no variable "
                f"{var_name!r}; available data vars: {avail}.  Pass "
                f"the correct {label}_var kwarg."
            )
        n_time = ds[var_name].shape[0]
        # Convention dispatch:
        #   * ``month is None`` → read index 0 (annual mean, or the
        #     single time slice in per-month files).
        #   * ``monthly_layout='concatenated'`` + ``month=N`` →
        #     index N-1; raise if the time axis is too short.  No
        #     silent fallback to index 0, since that would return
        #     the wrong month with no error.
        if month is None:
            idx = 0
        else:
            # Per-month-file callers go through ``monthly_layout=
            # 'single_file_per_month'`` which forces ``month is None``
            # before reaching this point (validated above).
            idx = requested_idx
        if idx >= n_time:
            ds.close()
            raise IndexError(
                f"WOA {label} file {path!r} variable {var_name!r} has "
                f"time length {n_time}; cannot extract month={month} "
                f"(index {idx}).  Pass a concatenated length-12 "
                f"monthly file, or use monthly_layout='single_file_"
                f"per_month' with the matching per-month file and "
                f"month=None."
            )
        return np.array(ds[var_name].values[idx, :, :, :])

    ds_T = _open_woa_dataset(T_path)
    T_woa = _read(ds_T, t_name, "T", T_path)
    lat_woa = np.array(ds_T["lat"].values)
    lon_woa = np.array(ds_T["lon"].values)
    ds_T.close()

    ds_S = _open_woa_dataset(S_path)
    S_woa = _read(ds_S, s_name, "S", S_path)
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


def _bilinear_2d(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation from regular lat-lon to target points.

    NaN-aware: if any of the four surrounding source cells is NaN, the
    weighted average is renormalised over the valid cells.  When all
    four are NaN the output is NaN (the caller fills with the deep-
    ocean climatology).  This is the OMIP-2 default horizontal
    interpolation and is required for smooth surface forcing — the
    nearest-neighbour path leaves grid-scale steps that show up as
    spurious gradients in coarse models.

    Parameters
    ----------
    target_lat, target_lon : array, shape (...)
        Target coordinates in degrees.  Longitude is wrapped to
        ``[0, 360)``.
    src_lat : array, shape (n_lat,)
        Source latitudes in degrees, **monotonically increasing**.
    src_lon : array, shape (n_lon,)
        Source longitudes in degrees, **monotonically increasing**,
        assumed periodic with period 360°.
    field : array, shape (n_lat, n_lon, ...)
        Source field.  Trailing dims are passed through.

    Returns
    -------
    array, shape (target_lat.shape + field.shape[2:])
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    if not np.all(np.diff(src_lat) > 0):
        raise ValueError("src_lat must be monotonically increasing.")

    # Longitude bookkeeping: accept any source convention
    # (``[0, 360)``, ``[-180, 180)``, ``[-540, -180)``, etc.).  Wrap
    # into ``[0, 360)`` and sort the field along the lon axis so the
    # subsequent searchsorted / bracket logic sees a strictly
    # increasing axis.  The post-mod values are checked to ensure no
    # duplicate longitudes survived (which would indicate aliasing).
    src_lon_raw = np.asarray(src_lon, dtype=np.float64)
    src_lon_mod = src_lon_raw % 360.0
    sort_idx = np.argsort(src_lon_mod, kind="stable")
    src_lon = src_lon_mod[sort_idx]
    field = field[:, sort_idx]
    if not np.all(np.diff(src_lon) > 0):
        # Two source longitudes mapped to the same modulo bucket.
        raise ValueError(
            "src_lon contains duplicate longitudes modulo 360 — "
            "cannot build a strictly increasing axis."
        )

    tlat = np.asarray(target_lat, dtype=np.float64).ravel()
    tlon = np.asarray(target_lon, dtype=np.float64).ravel() % 360.0

    # Clip latitude to source bounds (no extrapolation past poles).
    tlat = np.clip(tlat, src_lat[0], src_lat[-1])

    # Latitude bracketing
    j_hi = np.searchsorted(src_lat, tlat, side="left")
    j_hi = np.clip(j_hi, 1, src_lat.size - 1)
    j_lo = j_hi - 1
    dlat = src_lat[j_hi] - src_lat[j_lo]
    wj = np.where(dlat > 0, (tlat - src_lat[j_lo]) / dlat, 0.0)  # 0..1

    # Longitude bracketing — periodic.  ``i_hi`` may wrap from
    # n_lon - 1 to 0.
    nlon = src_lon.size
    # Insert a virtual point src_lon[-1] - 360 ... src_lon[0] handled
    # by treating any tlon < src_lon[0] as bracketed by (n_lon-1, 0)
    # with the upper point shifted by +360.
    i_lo = np.searchsorted(src_lon, tlon, side="right") - 1
    i_lo = i_lo % nlon
    i_hi = (i_lo + 1) % nlon
    # Wrapped neighbour location: when i_hi wraps to 0, treat src_lon[0]
    # as src_lon[0] + 360 so the bilinear interval stays monotonic.
    src_lon_hi = np.where(
        i_hi == 0, src_lon[0] + 360.0, src_lon[i_hi],
    )
    src_lon_lo = src_lon[i_lo]
    dlon = src_lon_hi - src_lon_lo
    # tlon may be on the wrap interval (tlon < src_lon[0]).  Shift up
    # so the formula uses the same continuous interval.
    tlon_shifted = np.where(tlon < src_lon_lo, tlon + 360.0, tlon)
    wi = np.where(dlon > 0, (tlon_shifted - src_lon_lo) / dlon, 0.0)
    wi = np.clip(wi, 0.0, 1.0)

    # Gather the four corners
    f00 = field[j_lo, i_lo]
    f01 = field[j_lo, i_hi]
    f10 = field[j_hi, i_lo]
    f11 = field[j_hi, i_hi]

    # Broadcast scalar weights to trailing dims
    wj_b = wj.reshape(wj.shape + (1,) * (f00.ndim - 1))
    wi_b = wi.reshape(wi.shape + (1,) * (f00.ndim - 1))
    w00 = (1.0 - wj_b) * (1.0 - wi_b)
    w01 = (1.0 - wj_b) * wi_b
    w10 = wj_b * (1.0 - wi_b)
    w11 = wj_b * wi_b

    # NaN-aware reweighting
    m00 = (~np.isnan(f00)).astype(np.float64)
    m01 = (~np.isnan(f01)).astype(np.float64)
    m10 = (~np.isnan(f10)).astype(np.float64)
    m11 = (~np.isnan(f11)).astype(np.float64)
    w00 = w00 * m00
    w01 = w01 * m01
    w10 = w10 * m10
    w11 = w11 * m11
    wsum = w00 + w01 + w10 + w11
    # Replace NaN inputs with zero for the weighted sum, then divide
    # by the active weight sum.  Where all four are NaN, wsum == 0
    # and we restore NaN so the caller knows to apply the fill.
    f00 = np.where(np.isnan(f00), 0.0, f00)
    f01 = np.where(np.isnan(f01), 0.0, f01)
    f10 = np.where(np.isnan(f10), 0.0, f10)
    f11 = np.where(np.isnan(f11), 0.0, f11)
    num = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    result = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)

    out_shape = np.asarray(target_lat).shape + field.shape[2:]
    return result.reshape(out_shape)


def init_ocean_from_woa(
    grid,
    z_coord: OceanZStarCoordinate,
    T_path: str | Path | None = None,
    S_path: str | Path | None = None,
    *,
    month: int | None = None,
    interp: str = "bilinear",
    bathymetry_depth: np.ndarray | None = None,
    T_fill_C: float | None = None,
    S_fill_psu: float | None = None,
    T_var: str | None = None,
    S_var: str | None = None,
    monthly_layout: str = "concatenated",
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
        Paths to WOA18 NetCDF annual or monthly climatology files.
    month : int or None
        Month 1..12 for monthly climatology; ``None`` (default) uses
        the annual mean.  Requires the input files to be monthly
        products (time axis of length 12).
    interp : {"bilinear", "nearest"}
        Horizontal interpolation method.  ``"bilinear"`` is the
        OMIP-2 default and is required for smooth surface forcing on
        coarse grids; ``"nearest"`` preserves legacy behaviour and is
        useful for debugging column-by-column.
    bathymetry_depth : array or None
        Positive water-column depth [m] at each horizontal target
        point, broadcastable to the grid's spatial shape.  When
        supplied, every model level below the local bathymetry is
        replaced with the fill value (``T_fill_C`` / ``S_fill_psu``)
        instead of carrying surface-extended WOA data into the
        seafloor — this avoids spurious density inversions when the
        model's bottom cell sits below the WOA-resolved water column.
    T_fill_C : float or None
        Replacement value [°C] for ``NaN`` profile entries and for
        levels below ``bathymetry_depth``.  Defaults to
        ``constants.T_deep_ocean_ref_C`` (1.5°C — WOA18 abyssal mean).
    S_fill_psu : float or None
        Replacement value [PSU] for ``NaN`` / below-bathymetry levels.
        Defaults to ``constants.S_deep_ocean_ref_psu`` (34.7 PSU).
    T_var, S_var : str or None
        Override the WOA dataset variable names.  Pass-through to
        :func:`load_woa18`.  Defaults: ``"t_an"`` / ``"s_an"``.  Use
        when the input dataset uses a non-standard name (e.g.
        ``"t_mn"`` for the monthly-mean object in newer WOA bundles).
    monthly_layout : {"concatenated", "single_file_per_month"}
        Pass-through to :func:`load_woa18`.  Defaults to
        ``"concatenated"`` (one 12-month file).  Use
        ``"single_file_per_month"`` when the caller has pre-selected
        a length-1 monthly file for the desired ``month`` — the
        loader then reads index 0 with ``month=None`` internally,
        avoiding silent month mismatch.

    Returns
    -------
    T : jax.Array, shape (..., nlev) — potential temperature [degC]
    S : jax.Array, shape (..., nlev) — salinity [PSU]
    """
    if interp not in ("bilinear", "nearest"):
        raise ValueError(
            f"interp must be 'bilinear' or 'nearest'; got {interp!r}."
        )
    if monthly_layout not in ("concatenated", "single_file_per_month"):
        raise ValueError(
            f"monthly_layout must be 'concatenated' or "
            f"'single_file_per_month'; got {monthly_layout!r}."
        )
    if monthly_layout == "single_file_per_month" and month is not None:
        # Same contract as load_woa18: the caller must pre-select the
        # file matching the desired month and pass month=None.
        # Silently dropping ``month`` here would mask file/month
        # mismatch and return the wrong climatology.
        raise ValueError(
            "init_ocean_from_woa(monthly_layout='single_file_per_month') "
            "requires month=None — pass the per-month file already "
            "selected for the target month.  Got month=" f"{month}."
        )
    T_fill = (
        float(constants.T_deep_ocean_ref_C)
        if T_fill_C is None else float(T_fill_C)
    )
    S_fill = (
        float(constants.S_deep_ocean_ref_psu)
        if S_fill_psu is None else float(S_fill_psu)
    )
    if not np.isfinite(T_fill):
        raise ValueError(
            f"T_fill_C must be finite; got {T_fill}."
        )
    if not np.isfinite(S_fill):
        raise ValueError(
            f"S_fill_psu must be finite; got {S_fill}."
        )

    # Extract latitude in degrees from the grid.
    # Dispatch order matters: GaussianGrid has both 'lat' and 'lat2d',
    # while CubedSphereGrid has 'lat' (3D) but no 'lat2d'.
    if hasattr(grid, 'lat2d'):
        # Grids exposing a 2-D lat2d (GaussianGrid, LatLonGrid, and the
        # orthogonal-curvilinear tripole LatLonCGridGeometry). Use the grid's
        # TRUE 2-D longitude (lon2d) -- on the tripole grid longitude varies
        # down every i-column, and grid.lon is only the 1-D SOUTHERNMOST row
        # (lon_T[0, :]); pairing it with the 2-D lat_T mis-sampled WOA by a
        # median ~9 deg (up to ~155 deg in the bipolar cap) at every cell. For
        # regular lat-lon / Gaussian grids lon2d is the meshgrid of the 1-D lon,
        # so this is identical there (no regression); fall back to the broadcast
        # 1-D lon only for a grid that lacks lon2d.
        lat_deg = np.asarray(grid.lat2d) * (180.0 / np.pi)
        lon2d = getattr(grid, 'lon2d', None)
        if lon2d is not None:
            lon_deg = np.asarray(lon2d) * (180.0 / np.pi)
        else:
            lon_1d = np.asarray(grid.lon) * (180.0 / np.pi)
            lon_deg = np.broadcast_to(lon_1d[np.newaxis, :], lat_deg.shape)
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
        # Load from WOA18 NetCDF files.  The guard above ensures that
        # ``monthly_layout='single_file_per_month'`` implies
        # ``month is None``, so this single call covers both layouts.
        T_woa, S_woa, lat_woa, lon_woa = load_woa18(
            T_path, S_path, month=month,
            T_var=T_var, S_var=S_var,
            monthly_layout=monthly_layout,
        )

        if interp == "bilinear":
            T_woa_horiz = _bilinear_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, T_woa,
            )
            S_woa_horiz = _bilinear_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, S_woa,
            )
        else:
            T_woa_horiz = _nearest_neighbor_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, T_woa,
            )
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

        # Bathymetry-aware mask: blank out levels below the local
        # water-column depth so the fill (rather than the
        # surface-extended profile) drives the deep cells.  Model
        # full-level depths are ``|z_full_ref|`` in metres.
        if bathymetry_depth is not None:
            bath_arr = np.asarray(bathymetry_depth, dtype=np.float64)
            if not np.all(np.isfinite(bath_arr)):
                raise ValueError(
                    "bathymetry_depth contains non-finite values; "
                    "expected positive water-column depth in metres."
                )
            if np.any(bath_arr <= 0.0):
                raise ValueError(
                    "bathymetry_depth must be strictly positive "
                    "(land/dry cells should be excluded by the caller "
                    "or set to a small wet-cell depth); got values "
                    f"with min={float(bath_arr.min())}."
                )
            model_depths = np.abs(np.asarray(z_coord.z_full_ref))
            nlev = T_out.shape[-1]
            # Only 1-D z_full_ref of length nlev is supported here.
            # Shaped model depths (e.g. spatially varying interfaces
            # under z-tilde) need a separate code path and an
            # explicit shape match.
            if model_depths.shape == (nlev,):
                depth_grid = np.broadcast_to(
                    model_depths.reshape((1,) * (T_out.ndim - 1) + (-1,)),
                    T_out.shape,
                )
            elif model_depths.shape == T_out.shape:
                depth_grid = model_depths
            else:
                raise ValueError(
                    f"z_coord.z_full_ref has shape {model_depths.shape}; "
                    f"expected ({nlev},) for a column-uniform z* "
                    f"coordinate, or {T_out.shape} for a spatially-"
                    f"varying coordinate."
                )
            bath = np.broadcast_to(bath_arr[..., None], T_out.shape)
            below_floor = depth_grid > bath
            T_out = np.where(below_floor, np.nan, T_out)
            S_out = np.where(below_floor, np.nan, S_out)

        # Fill remaining NaNs (whole-column failures + below-floor mask)
        T_out = np.where(np.isnan(T_out), T_fill, T_out)
        S_out = np.where(np.isnan(S_out), S_fill, S_out)

    else:
        # Analytical fallback
        T_out, S_out = _analytical_woa_profiles(lat_deg, z_coord)

    return jnp.array(T_out), jnp.array(S_out)
