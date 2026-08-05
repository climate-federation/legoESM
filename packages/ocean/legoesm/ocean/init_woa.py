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


def interp_column_to_depths(
    col_src: np.ndarray,
    src_depths: np.ndarray,
    dst_depths: np.ndarray,
) -> np.ndarray:
    """Interpolate one T/S column from ``src_depths`` onto ``dst_depths``.

    NaN-safe: non-finite source entries (land, below-seafloor fill) are
    dropped before interpolating, and a column with fewer than two valid
    entries returns all-NaN so the caller's fill logic — not a fabricated
    value — decides what goes there.  Below the deepest valid source level
    the deepest valid value is held (``np.interp``'s edge behaviour).

    Shared helper for the IC-preparation scripts that re-level an external
    climatology onto :data:`WOA_DEPTHS`.  NOTE this is no longer *required*:
    :func:`init_ocean_from_woa` now reads the source depth axis from the FILE
    and only falls back to :data:`WOA_DEPTHS` when the file has none.
    Re-levelling first remains useful when a caller wants one common axis for
    several sources.

    Parameters
    ----------
    col_src : 1-D array of values at ``src_depths``.
    src_depths : 1-D array, positive downward, ascending.
    dst_depths : 1-D array, positive downward, ascending.
    """
    good = np.isfinite(col_src)
    if good.sum() < 2:
        return np.full(np.shape(dst_depths), np.nan)
    return np.interp(dst_depths, np.asarray(src_depths)[good],
                     np.asarray(col_src)[good])


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
        # Not enough valid data.  ONE valid entry is propagated down the
        # column; NONE returns NaN so the caller's fill (T_fill_C / S_fill_psu)
        # applies.
        #
        # Returning 0.0 here instead of NaN was a real defect: 0.0 is not NaN,
        # so ``np.where(np.isnan(T_out), T_fill, T_out)`` below never replaced
        # it, and a wet cell whose source column had no data entered the model
        # as T = 0 degC, S = 0 PSU.  Fresh water at 0 degC is rho = 999.8
        # against ~1027 for real sea water, so such a cell sat next to normal
        # ocean with a density jump up to 30 kg/m^3 -- larger than the entire
        # ocean's density range.  On the FESOM2-matched CORE2 config that gave
        # a hydrostatic pressure-gradient acceleration of 2e-2 m/s^2 and ~42 m/s
        # in a single 2400 s step, i.e. an immediate blowup (1569 wet columns
        # affected; see docs/ocean/fidelity/fesom2_gap_analysis.md 3.7).
        return np.full(
            z_coord.n_levels,
            profile[valid][0] if valid.any() else np.nan,
        )

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


#: Coordinate names a climatology may use for its vertical axis.  Matched
#: case-insensitively; the variable's OWN dimension order is consulted first,
#: so an unusual name is still found as long as it is the field's 2nd axis.
_DEPTH_AXIS_NAMES: tuple[str, ...] = (
    "depth", "z", "lev", "level", "deptht", "depth_bnds_mid", "olevel",
)


def _depth_axis_of(ds, var_name: str):
    """Positive depths [m] for ``ds[var_name]``'s vertical axis, or ``None``.

    Resolution order, so an unusual coordinate name is not silently missed:
    (1) the variable's OWN second dimension (time, DEPTH, lat, lon) when that
    dimension has coordinate values; (2) any recognised name from
    :data:`_DEPTH_AXIS_NAMES`, case-insensitively.
    """
    dims = getattr(ds[var_name], "dims", ())
    if len(dims) >= 2:
        cand = dims[1]
        if cand in ds.coords or cand in ds.variables:
            return np.abs(np.asarray(ds[cand].values, dtype=np.float64))
    lowered = {str(k).lower(): k for k in
               list(ds.coords) + list(ds.variables)}
    for name in _DEPTH_AXIS_NAMES:
        if name in lowered:
            return np.abs(
                np.asarray(ds[lowered[name]].values, dtype=np.float64))
    return None


def load_woa18(
    T_path: str | Path | None = None,
    S_path: str | Path | None = None,
    month: int | None = None,
    T_var: str | None = None,
    S_var: str | None = None,
    monthly_layout: str = "concatenated",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray | None]:
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
    depth_woa : array, shape (n_depth,) — POSITIVE depths [m] read from the
        FILE, or ``None`` if it carries no recognisable depth coordinate
        (``depth``/``z``/``lev``/``level``/``deptht``).  Callers must prefer
        this over :data:`WOA_DEPTHS`: a non-WOA climatology on its own levels
        (PHC3's 33, WOCE's 75) read against the hardcoded WOA axis puts deep
        water in the thermocline without raising anything.
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
    # The FILE's own depth axis, when it has one.  Reading it here rather than
    # assuming ``WOA_DEPTHS`` is what lets a non-WOA climatology (PHC3, WOCE,
    # EN4) be used without its deep values being read as if they sat in the
    # top few hundred metres -- a silently wrong ocean, not an error.  A file
    # with no recognisable depth coordinate returns ``None`` and the caller
    # falls back to ``WOA_DEPTHS``.
    depth_woa = _depth_axis_of(ds_T, t_name)
    ds_T.close()

    ds_S = _open_woa_dataset(S_path)
    S_woa = _read(ds_S, s_name, "S", S_path)
    depth_S = _depth_axis_of(ds_S, s_name)
    ds_S.close()

    # T and S are interpolated onto the model levels with ONE axis, so they
    # must agree.  Equal level COUNTS with different depths would otherwise
    # place salinity at the temperature file's depths without a word.
    if depth_woa is not None and depth_S is not None:
        # Tolerance is PRACTICAL, not exact: a legitimate pair can differ by
        # float32-vs-float64 storage of the same nominal levels (5500.0 m is
        # 5500.0005 in float32), and rejecting that would be a false alarm.
        # 1 mm absolute + 1e-6 relative is far below any real level spacing
        # and far above storage roundoff.
        if depth_woa.shape != depth_S.shape or not np.allclose(
                depth_woa, depth_S, rtol=1e-6, atol=1e-3):
            raise ValueError(
                f"T file {T_path!r} and S file {S_path!r} have DIFFERENT "
                f"depth axes ({depth_woa[:3]}... vs {depth_S[:3]}...). "
                "They are interpolated onto the model levels with a single "
                "axis, so salinity would be placed at the temperature file's "
                "depths. Re-level them onto a common axis first."
            )
    elif depth_S is not None and depth_woa is None:
        # Only S carries an axis: use it rather than falling back to WOA's.
        depth_woa = depth_S

    # Transpose to (lat, lon, depth) for easier interpolation
    T_woa = np.transpose(T_woa, (1, 2, 0))
    S_woa = np.transpose(S_woa, (1, 2, 0))

    for _axis, _field, _name, _path, _label in (
        (depth_woa, T_woa, t_name, T_path, "T"),
        (depth_S, S_woa, s_name, S_path, "S"),
    ):
        if _axis is not None and _axis.size != _field.shape[-1]:
            raise ValueError(
                f"WOA {_label} file {_path!r}: depth coordinate has "
                f"{_axis.size} entries but {_name!r} has "
                f"{_field.shape[-1]} levels."
            )
    if T_woa.shape[-1] != S_woa.shape[-1]:
        raise ValueError(
            f"T file {T_path!r} has {T_woa.shape[-1]} levels but S file "
            f"{S_path!r} has {S_woa.shape[-1]}; they are interpolated onto "
            "the model levels together."
        )

    return T_woa, S_woa, lat_woa, lon_woa, depth_woa


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
        T_woa, S_woa, lat_woa, lon_woa, depth_woa = load_woa18(
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

        # Source depth axis: the FILE's own when it has one, otherwise the
        # WOA18 standard levels.  Reading it from the file is what lets a
        # non-WOA climatology (PHC3's 33 levels, WOCE's 75) be used directly;
        # the ``WOA_DEPTHS[:n]`` fallback silently mis-levels such a file, so
        # it is only reached when the file genuinely has no depth coordinate.
        if depth_woa is not None:
            woa_depths = np.asarray(depth_woa, dtype=np.float64)
            if not np.all(np.diff(woa_depths) > 0):
                raise ValueError(
                    "WOA/climatology depth coordinate is not strictly "
                    f"ascending after abs(); got {woa_depths[:5]}..."
                    f"{woa_depths[-3:]}. np.interp requires monotonic source "
                    "depths."
                )
        else:
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


def woa_ocean_mask(
    grid,
    T_path: str | Path,
    *,
    T_var: str | None = None,
    monthly_layout: str = "concatenated",
    ocean_fraction_threshold: float = 0.5,
) -> np.ndarray:
    """Derive a realistic ``(1=ocean, 0=land)`` wet mask from WOA18 surface T.

    A grid cell is OCEAN where the area-weighted bilinear regrid of the WOA18
    surface-temperature *valid-data fraction* (1 over ocean, 0 over WOA's land
    fill) reaches ``ocean_fraction_threshold``, and LAND otherwise.  Using the
    regridded fraction (rather than the NaN-aware T itself) gives a crisp,
    threshold-controlled coastline on coarse target grids instead of flooding
    every coastal cell to ocean.

    This is the single source of truth for the realistic coupled run's wet
    domain: the same mask initialises the ocean ``land_mask`` AND the
    atmosphere ``f_land`` (``= 1 - ocean_mask``) on the shared lat-lon grid, so
    the surface/ocean wet masks agree EXACTLY — no atmosphere flux leaks onto
    an ocean-masked cell (codex Phase-1 HIGH).  Only the lat-lon grid is
    supported because the prognostic 3D ocean is lat-lon-only.

    Parameters
    ----------
    grid : LatLonGrid
        Must expose ``lat2d`` / ``lon2d`` (radians).
    T_path : path
        WOA18 temperature NetCDF/Zarr (the ``t_an`` annual field, or a
        monthly product — only the surface level, time index 0, is read).
    T_var : str or None
        Override the temperature variable name (default ``"t_an"``).
    monthly_layout : str
        Accepted for signature parity with :func:`load_woa18`; the surface
        mask only ever reads time index 0, so the layout does not change the
        result.
    ocean_fraction_threshold : float, default 0.5
        Regridded ocean-fraction cutoff for classifying a cell as ocean.

    Returns
    -------
    ocean_mask : np.ndarray, shape (n_lat, n_lon), float64
        ``1.0`` over ocean, ``0.0`` over land.  Caller casts to the active
        precision policy (e.g. via ``rest_state_latlon_cgrid_ocean``).

    Notes
    -----
    Marginal/enclosed seas present in WOA (Caspian, Black, etc.) are classified
    as ocean here; for a coupled run that has previously destabilised the
    barotropic solver (see the OMIP ``--mask-marginal-seas`` note), so a
    follow-up may want to prune them.  Left in for a first realistic run.
    """
    if monthly_layout not in ("concatenated", "single_file_per_month"):
        raise ValueError(
            f"monthly_layout must be 'concatenated' or "
            f"'single_file_per_month'; got {monthly_layout!r}."
        )
    if not (0.0 < ocean_fraction_threshold <= 1.0):
        raise ValueError(
            "ocean_fraction_threshold must be in (0, 1]; got "
            f"{ocean_fraction_threshold}."
        )
    lat2d = getattr(grid, "lat2d", None)
    lon2d = getattr(grid, "lon2d", None)
    if lat2d is None or lon2d is None:
        raise TypeError(
            "woa_ocean_mask requires a lat-lon grid exposing lat2d/lon2d "
            f"(radians); got {type(grid).__name__}."
        )
    lat_deg = np.asarray(lat2d) * (180.0 / np.pi)
    lon_deg = np.asarray(lon2d) * (180.0 / np.pi)

    t_name = T_var if T_var is not None else "t_an"
    ds = _open_woa_dataset(T_path)
    if t_name not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"WOA T file {str(T_path)!r} has no variable {t_name!r}; "
            f"available data vars: {avail}.  Pass the correct T_var."
        )
    # Select the surface level of the first time slice by DIMENSION NAME
    # (robust to transposed / preprocessed files): time->0, the depth axis
    # (the dim that is not lat/lon/time)->0, then order as (lat, lon).
    da = ds[t_name]
    isel = {}
    if "time" in da.dims:
        isel["time"] = 0
    depth_dim = next(
        (d for d in da.dims if d not in ("lat", "lon", "time")), None)
    if depth_dim is not None:
        isel[depth_dim] = 0
    da_surf = da.isel(**isel)
    if {"lat", "lon"}.issubset(da_surf.dims):
        da_surf = da_surf.transpose("lat", "lon")
    T_surf = np.asarray(da_surf.values)  # (n_lat_woa, n_lon_woa)
    lat_woa = np.array(ds["lat"].values)
    lon_woa = np.array(ds["lon"].values)
    ds.close()
    if T_surf.ndim != 2:
        raise ValueError(
            f"WOA surface T from {str(T_path)!r} is {T_surf.ndim}-D after "
            f"selecting time/depth (var dims {da.dims}); expected 2-D "
            "(lat, lon).")

    # Ocean-fraction source field: 1 where WOA has valid surface T (ocean),
    # 0 where NaN (land fill).  Regrid with the (NaN-free) bilinear operator
    # and threshold.  Trailing singleton dim so _bilinear_2d's trailing-dim
    # broadcasting applies.
    ocean_src = (~np.isnan(T_surf)).astype(np.float64)[..., None]
    frac = _bilinear_2d(lat_deg, lon_deg, lat_woa, lon_woa, ocean_src)[..., 0]
    # Cells with no valid bracketing source (frac is NaN) are land.
    frac = np.where(np.isnan(frac), 0.0, frac)
    ocean_mask = (frac >= ocean_fraction_threshold).astype(np.float64)
    return ocean_mask


def woa_ocean_bathymetry(
    grid,
    T_path: str | Path,
    *,
    H_max: float = 5500.0,
    min_depth_m: float = 200.0,
    T_var: str | None = None,
    ocean_fraction_threshold: float = 0.5,
) -> np.ndarray:
    """Derive a realistic per-column bathymetry [m] from WOA18 temperature.

    For each model column the depth is the DEEPEST WOA standard level that is
    still ocean (valid T) after the NaN-aware bilinear regrid of the per-level
    valid-data fraction, clamped to ``[min_depth_m, H_max]``.  Land columns
    (surface fraction below threshold) get ``0``.

    A realistic (varying) bathymetry — instead of a flat ``H_max`` everywhere —
    is REQUIRED for a stable WOA cold start: the flat bottom extends the WOA
    abyssal-fill density into a spurious deep column and (with the plain z-star
    coord) gates off the smc03 partial-cell pressure-gradient correction.  Feed
    the result through :func:`make_partial_cell_latlon` (smoothing + thin-cell
    snap) before building the ocean coordinate.

    Parameters
    ----------
    grid : LatLonGrid (lat2d/lon2d in radians).
    T_path : WOA18 temperature file.
    H_max : maximum depth clamp [m].
    min_depth_m : minimum wet-column depth [m] (shallow shelves are floored
        here; ``make_partial_cell_latlon(min_levels=...)`` later masks columns
        with too few active levels).
    T_var : override temperature variable name (default ``"t_an"``).
    ocean_fraction_threshold : ocean cutoff for the per-level regrid.

    Returns
    -------
    H_bathy : np.ndarray (n_lat, n_lon), float64 — depth [m], 0 over land.
    """
    if not (0.0 < ocean_fraction_threshold <= 1.0):
        raise ValueError(
            f"ocean_fraction_threshold must be in (0, 1]; got "
            f"{ocean_fraction_threshold}.")
    if not (0.0 < min_depth_m < H_max):
        raise ValueError(
            f"min_depth_m must be in (0, H_max={H_max}); got {min_depth_m}.")
    lat2d = getattr(grid, "lat2d", None)
    lon2d = getattr(grid, "lon2d", None)
    if lat2d is None or lon2d is None:
        raise TypeError(
            "woa_ocean_bathymetry requires a lat-lon grid exposing lat2d/lon2d.")
    lat_deg = np.asarray(lat2d) * (180.0 / np.pi)
    lon_deg = np.asarray(lon2d) * (180.0 / np.pi)

    t_name = T_var if T_var is not None else "t_an"
    ds = _open_woa_dataset(T_path)
    if t_name not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"WOA T file {str(T_path)!r} has no variable {t_name!r}; "
            f"available: {avail}.")
    da = ds[t_name]
    isel = {}
    if "time" in da.dims:
        isel["time"] = 0
    depth_dim = next(
        (d for d in da.dims if d not in ("lat", "lon", "time")), None)
    if depth_dim is None:
        ds.close()
        raise ValueError(
            f"WOA T variable has no depth dim (dims {da.dims}).")
    da3 = da.isel(**isel).transpose(depth_dim, "lat", "lon")
    T3 = np.asarray(da3.values)                       # (n_depth, lat, lon)
    lat_woa = np.array(ds["lat"].values)
    lon_woa = np.array(ds["lon"].values)
    ds.close()

    n_depth = T3.shape[0]
    depths = WOA_DEPTHS[:n_depth]
    # Ocean fraction per WOA level, regridded to the model grid.
    valid = (~np.isnan(T3)).astype(np.float64)        # (depth, lat, lon)
    valid_lld = np.moveaxis(valid, 0, -1)             # (lat, lon, depth)
    fr = _bilinear_2d(lat_deg, lon_deg, lat_woa, lon_woa, valid_lld)
    fr = np.where(np.isnan(fr), 0.0, fr)
    ocean_at_depth = fr >= ocean_fraction_threshold   # (n_lat, n_lon, n_depth)
    # Deepest ocean depth per column (0 if no ocean at any level).
    H = np.where(ocean_at_depth, depths[None, None, :], 0.0).max(axis=-1)
    wet = H > 0.0
    H = np.where(wet, np.clip(H, min_depth_m, H_max), 0.0)
    return H.astype(np.float64)
