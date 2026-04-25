"""External forcing interfaces for AMIP/CMIP-style experiments.

Provides configuration and interpolation for time-varying external
forcings beyond SST/SIC:

- Greenhouse gas concentrations (CO2, CH4, N2O, CFCs)
- Ozone climatology / forcing
- Aerosol optical depth climatology
- Solar irradiance variations

Status
------
- GHG: config + constant-value active; time-varying from file supported
- Ozone: zonal-mean climatology with lat/vertical interpolation + reference fallback
- Aerosol: zonal-mean AOD climatology with optional volcanic contribution
- Solar: constant/file TSI and full spectral (per g-point) forcing
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Data loading helpers (Zarr-first, NetCDF fallback via xarray)
# ==============================================================================

def _open_forcing_dataset(path: str):
    """Open a forcing file as xarray Dataset (Zarr or NetCDF).

    Always opens with ``decode_times=False`` so that time coordinates are
    returned as raw numeric values (float/int) that can be safely cast to
    ``np.float64``.  Files whose dates extend beyond numpy's datetime64[ns]
    range (~2262) would otherwise cause xarray to fall back to
    ``cftime.datetime`` objects, which cannot be cast to float64.
    """
    import os
    import xarray as xr
    if os.path.isdir(path) or path.endswith(".zarr"):
        return xr.open_zarr(path, decode_times=False)
    return xr.open_dataset(path, decode_times=False)


def _to_days_float(time_values, *, units: str | None = None,
                   calendar: str | None = None) -> np.ndarray:
    """Convert xarray time values to float days since the first record.

    Handles four cases that arise in practice:

    * ``numpy.datetime64`` arrays — standard CF time within the
      ``datetime64[ns]`` representable range (~year 100–2262).
    * ``cftime.datetime`` object arrays — xarray falls back to cftime
      when dates exceed the ``datetime64[ns]`` range (e.g. 1850–2299
      CMIP6 solar files).  Converted via :class:`datetime.timedelta`
      arithmetic.
    * Numeric arrays with a CF ``units`` attribute of the form
      ``"<unit> since <ref>"`` (e.g. ``"months since 1850-01-01"``,
      ``"hours since 2000-01-01"``): decoded through ``cftime.num2date``
      and converted to days since the first record, so e.g. CMIP6 ozone
      files with a 600-month time axis produce a 600-element array of
      real day offsets (≈0 … 18262), not of month indices.
    * Already-numeric arrays with no / unrecognised units — returned
      as-is after a simple ``float64`` cast (e.g. ``"year as %Y.%f"``
      GHG files which the GHG loader already handles separately, and
      test fixtures that write raw day offsets).

    Parameters
    ----------
    units : str, optional
        Value of the ``time`` variable's ``units`` attribute.
    calendar : str, optional
        Value of the ``time`` variable's ``calendar`` attribute.
        Defaults to ``"standard"`` when CF units are detected.
    """
    arr = np.asarray(time_values)
    if arr.dtype == object and arr.size > 0:
        try:
            import cftime  # noqa: F401
            ref = arr.flat[0]
            days = np.array(
                [(t - ref).days + (t - ref).seconds / 86400.0 for t in arr.ravel()],
                dtype=np.float64,
            ).reshape(arr.shape)
            return days
        except (ImportError, AttributeError):
            pass
    if np.issubdtype(arr.dtype, np.datetime64):
        t0 = arr.flat[0]
        return ((arr - t0) / np.timedelta64(1, "D")).astype(np.float64)
    numeric = arr.astype(np.float64)
    if units and "since" in units.lower():
        ul = units.lower()
        # "months since" / "years since" are only supported by cftime on
        # the ``360_day`` calendar.  For the much more common CMIP6
        # "months since 1850-01-01" on a standard/gregorian calendar we
        # convert with the 30.4375-day average month (year/12).  For
        # ``360_day`` we use the exact 30 d/month so multi-decade files
        # don't drift by whole months (Codex review 2026-04-24).
        cal = (calendar or "standard").lower()
        if "months since" in ul:
            scale = 30.0 if cal == "360_day" else 365.25 / 12.0
            return (numeric - numeric.flat[0]) * scale
        if "years since" in ul:
            scale = 360.0 if cal == "360_day" else 365.25
            return (numeric - numeric.flat[0]) * scale
        try:
            import cftime
            dates = cftime.num2date(numeric, units,
                                     calendar=calendar or "standard")
            ref = dates.flat[0] if hasattr(dates, "flat") else dates
            days = np.array(
                [(t - ref).days + (t - ref).seconds / 86400.0
                 for t in np.asarray(dates).ravel()],
                dtype=np.float64,
            ).reshape(numeric.shape)
            return days
        except Exception:
            # Fall through: treat raw numeric as days if cftime parse fails.
            pass
    return numeric


def _read_time_days(ds) -> np.ndarray:
    """Return ``time`` as float days since the first record, CF-unit-aware.

    Convenience wrapper that forwards ``ds['time'].attrs['units']`` and
    ``['calendar']`` to :func:`_to_days_float` so every call site picks up
    proper CF-time decoding without repeating the attribute lookup.
    """
    tv = ds["time"]
    attrs = getattr(tv, "attrs", {}) or {}
    return _to_days_float(
        tv.values,
        units=attrs.get("units"),
        calendar=attrs.get("calendar"),
    )


@lru_cache(maxsize=16)
def _load_time_anchor(path: str):
    """Return the first-record ``cftime.datetime`` / ``datetime64`` for a
    forcing file's ``time`` axis, or ``None`` when no CF ``units`` are
    available.  Cached; used by :func:`get_solar_forcing_at_time` and
    :func:`get_ozone_at_time` to map a simulation day onto the file's
    absolute time axis without re-opening the dataset each call.
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds:
        ds.close()
        return None
    tv = ds["time"]
    attrs = getattr(tv, "attrs", {}) or {}
    anchor = _extract_first_date(
        tv.values, attrs.get("units"), attrs.get("calendar"),
    )
    ds.close()
    return anchor


def _read_time_axis(ds):
    """Return ``(days_since_first_record, first_record_date_or_none)``.

    ``first_record_date`` is a :class:`cftime.datetime` (or ``numpy.datetime64``)
    best-effort parsed from either the actual ``time`` values (object /
    ``datetime64`` arrays) or the CF ``units`` attribute when present.
    Callers use this anchor to map a simulation day (days since
    ``start_year-01-01``) onto the file's absolute time axis.
    """
    tv = ds["time"]
    attrs = getattr(tv, "attrs", {}) or {}
    units = attrs.get("units")
    calendar = attrs.get("calendar")
    days = _to_days_float(tv.values, units=units, calendar=calendar)
    first_date = _extract_first_date(tv.values, units, calendar)
    return days, first_date


def _extract_first_date(time_values, units, calendar):
    """Best-effort first-record extraction for a CF time axis.

    Handles the four shapes that arise in practice:

    * object arrays of ``cftime`` datetimes: take ``[0]``.
    * ``numpy.datetime64`` arrays: take ``[0]``.
    * Numeric arrays with CF ``units`` of the form ``"<unit> since <ref>"``
      on a calendar ``cftime`` understands natively: use
      :func:`cftime.num2date`.
    * Numeric arrays with CF units of ``months since`` / ``years since``
      on a non-``360_day`` calendar (which ``cftime.num2date`` refuses):
      parse ``<ref>`` manually and add the integer month / year offset
      using :class:`cftime.DatetimeGregorian`.

    Returns ``None`` when none of those paths succeeds — the caller then
    falls back to the 1850 CMIP6 reference year in
    :func:`_simday_to_file_day`.
    """
    arr = np.asarray(time_values)
    if arr.dtype == object and arr.size > 0:
        return arr.flat[0]
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.flat[0]
    if not units or "since" not in units.lower():
        return None
    numeric = arr.astype(np.float64)
    ul = units.lower()
    cal = (calendar or "standard").lower()
    try:
        import cftime
        import datetime as _dt
        if "months since" in ul or "years since" in ul:
            anchor_str = units.split("since", 1)[1].strip().split(" ")[0]
            y_str, m_str, d_str = anchor_str.split("-")
            anchor = cftime.DatetimeGregorian(
                int(y_str), int(m_str), int(d_str),
            )
            v0 = float(numeric.flat[0])
            # Add whole months / years, then fractional part as days so
            # files with mid-month sample points (v0 = 0.5) anchor to
            # the actual first-record date, not the units reference
            # (fix per Codex 2026-04-24).
            if "months since" in ul:
                whole = int(np.floor(v0))
                frac = v0 - whole
                total_m = anchor.month - 1 + whole
                ny = anchor.year + total_m // 12
                nm = (total_m % 12) + 1
                base = cftime.DatetimeGregorian(ny, nm, anchor.day)
                frac_days = frac * (30.0 if cal == "360_day" else 365.25 / 12.0)
            else:  # years since
                whole = int(np.floor(v0))
                frac = v0 - whole
                base = cftime.DatetimeGregorian(
                    anchor.year + whole, anchor.month, anchor.day,
                )
                frac_days = frac * (360.0 if cal == "360_day" else 365.25)
            if frac_days:
                return base + _dt.timedelta(days=frac_days)
            return base
        dates = cftime.num2date(
            numeric[:1] if numeric.ndim else numeric,
            units, calendar=calendar or "standard",
        )
        return dates.flat[0] if hasattr(dates, "flat") else dates
    except Exception:
        return None


@lru_cache(maxsize=16)
def _load_timeseries(path: str, varnames: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load 1-D time series variables from a Zarr store or NetCDF file.

    Parameters
    ----------
    path : str
        Path to a Zarr store or NetCDF file with a ``time`` dimension
        (in fractional days).
    varnames : tuple of str
        Variable names to load (must be 1-D along time).

    Returns
    -------
    (times, data) where times is shape (N,) in days and data maps
    each varname to a 1-D numpy array of length N.
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds.dims:
        ds.close()
        raise ValueError(f"Forcing file {path!r} has no 'time' dimension")
    times = _read_time_days(ds)
    # Build a case-insensitive lookup for variable names to handle files where
    # conventions differ (e.g., "TSI" in CMIP6 solar files vs. "tsi" default).
    varname_map = {name.lower(): name for name in ds.data_vars}
    data = {}
    for v in varnames:
        actual = v if v in ds.data_vars else varname_map.get(v.lower())
        if actual is None:
            ds.close()
            raise ValueError(f"Variable {v!r} not found in {path!r}")
        arr = np.asarray(ds[actual].values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != times.shape[0]:
            ds.close()
            raise ValueError(
                f"Variable {actual!r} must be 1-D with length matching 'time' "
                f"(got shape {arr.shape}, expected ({times.shape[0]},))"
            )
        data[v] = arr
    ds.close()
    return times, data


def _interp_1d(times: np.ndarray, values: np.ndarray, day: float) -> float:
    """Linearly interpolate a 1-D time series at *day*, clamping at edges."""
    return float(np.interp(day, times, values))


def _interp_2d_time(times: np.ndarray, values: np.ndarray, day: float) -> np.ndarray:
    """Linearly interpolate a 2-D time series (time, nfeat) at *day*."""
    if values.ndim != 2 or values.shape[0] != times.shape[0]:
        raise ValueError(
            "Expected values with shape (ntime, nfeat) matching times length; "
            f"got {values.shape} with times {times.shape}",
        )
    out = np.zeros((values.shape[1],), dtype=np.float64)
    for j in range(values.shape[1]):
        out[j] = np.interp(day, times, values[:, j])
    return out


@lru_cache(maxsize=16)
def _load_monthly_zonal(path: str, varname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a monthly zonal-mean field from a Zarr store or NetCDF file.

    Handles files with full lat/lon grids (e.g. Kinne aerosol files with
    dims ``(time, band, lat, lon)``) by averaging over lon and any extra
    non-(time, lat) dimensions to produce a ``(ntime, nlat)`` array.

    Returns
    -------
    (mid_days, lat, data) where mid_days is shape (ntime,), lat is shape
    (nlat,), and data is shape (ntime, nlat).
    """
    ds = _open_forcing_dataset(path)
    if varname not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable {varname!r} not found in {path!r}")
    var = ds[varname]
    data = np.asarray(var.values, dtype=np.float64)
    dims = list(var.dims)

    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' variable in {path!r}")
    if "time" in ds:
        mid_days = _read_time_days(ds)
    else:
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
    ds.close()

    # Average over longitude to produce a zonal mean.
    for lname in ("lon", "longitude"):
        if lname in dims:
            ax = dims.index(lname)
            data = np.nanmean(data, axis=ax)
            dims.pop(ax)
            break

    # Average over any remaining non-(time, lat) dimensions (e.g. spectral
    # bands in Kinne aerosol files: (time, lnwl, lat) → (time, lat)).
    lat_name = "lat" if "lat" in dims else "latitude"
    time_name = "time" if "time" in dims else None
    keep = {lat_name}
    if time_name:
        keep.add(time_name)
    extra_axes = [i for i, d in enumerate(dims) if d not in keep]
    for ax in sorted(extra_axes, reverse=True):
        data = np.nanmean(data, axis=ax)
        dims.pop(ax)

    return mid_days, lat, data


@lru_cache(maxsize=16)
def _load_volcanic_cmip6(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CMIP6 / MPI-M stratospheric volcanic aerosol file.

    These files (``bc_aeropt_cmip6_volc_lw_b16_sw_b14_<year>.nc``) do not
    carry a flat ``aod(time, lat)`` variable; instead they store per-band
    SW extinction ``ext_sun(solar_bands, lat, altitude, month)`` in
    [1/km] on an altitude grid in km.  A scalar AOD compatible with the
    rest of the aerosol pipeline is recovered in two steps:

    1. Trapezoidal integration over altitude: ``AOD_band = ∫ ext · dz``
       with ``dz`` in km, yielding a per-band AOD per (lat, month).
    2. Mean over the spectral-band axis to produce a representative
       single-band broadband AOD per (lat, month).  Mean — not sum —
       matches the Kinne aerosol convention (``_load_monthly_zonal``
       averages non-(time, lat) axes via ``np.nanmean``) and avoids the
       n-bands-dependent inflation that a raw sum over 14 SW bands
       would introduce.

    Returns ``(mid_days, lat, aod)`` with ``aod`` shape
    ``(ntime, nlat)`` — a drop-in replacement for
    :func:`_load_monthly_zonal` output.
    """
    ds = _open_forcing_dataset(path)
    # Only ``ext_sun`` (SW stratospheric extinction) is valid here: the SW
    # aerosol branch multiplies this into broadband AOD, so reading the LW
    # ``ext_earth`` variable as a SW quantity would silently inject
    # terrestrial-IR extinction into the shortwave forcing budget
    # (Codex review 2026-04-24).  A dedicated LW-aware path can use
    # ``ext_earth`` later when the radiation scheme supports per-band LW
    # aerosol input.
    if "ext_sun" in ds.data_vars:
        ext_name = "ext_sun"
    else:
        ds.close()
        raise ValueError(
            f"No CMIP6 volcanic SW extinction variable 'ext_sun' in "
            f"{path!r}. (CMIP6 files typically also carry 'ext_earth' "
            f"for LW; that variable is not usable as a SW AOD source.)",
        )
    var = ds[ext_name]
    ext = np.asarray(var.values, dtype=np.float64)
    dims = list(var.dims)

    alt_name = next(
        (v for v in ("altitude", "alt", "height", "lev") if v in ds),
        None,
    )
    if alt_name is None:
        ds.close()
        raise ValueError(
            f"No altitude coordinate in CMIP6 volcanic file {path!r}",
        )
    alt = np.asarray(ds[alt_name].values, dtype=np.float64)

    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    else:
        ds.close()
        raise ValueError(
            f"No 'lat'/'latitude' variable in CMIP6 volcanic file {path!r}",
        )

    # Month / time axis — CMIP6 volcanic may use ``month`` or ``time``.
    if "time" in dims:
        time_name = "time"
        mid_days = _read_time_days(ds)
    elif "month" in dims:
        time_name = "month"
        nm = ext.shape[dims.index("month")]
        mid_days = np.array([15.5 + 30.4375 * m for m in range(nm)])
    else:
        ds.close()
        raise ValueError(
            f"No time/month dimension in CMIP6 volcanic file {path!r}",
        )
    ds.close()

    # Trapezoidal integral over altitude: ext [1/km] * dz [km] → AOD.
    # ``np.trapz`` was removed in NumPy 2.x; use the renamed ``trapezoid``.
    alt_axis = dims.index(alt_name)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    aod_with_bands = _trapz(ext, alt, axis=alt_axis)
    dims_after = [d for i, d in enumerate(dims) if i != alt_axis]

    # Collapse the spectral-band axis to a single broadband AOD BEFORE
    # returning.  We take the mean over bands (not the sum) to match the
    # implicit Kinne aerosol convention — ``_load_monthly_zonal`` averages
    # any non-(time, lat) axis via ``np.nanmean`` — and to avoid the
    # n_bands factor Codex flagged: summing 14 SW bands would inflate
    # the per-wavelength optical depth into a 14×-multiple that is not
    # physically a broadband AOD.  The value returned is therefore a
    # representative single-band AOD, consistent with the scalar AOD
    # modifier used by the rest of the aerosol pipeline.
    band_dim_name = next(
        (d for d in dims_after if d not in (time_name, "lat", "latitude")),
        None,
    )
    if band_dim_name is not None:
        band_axis = dims_after.index(band_dim_name)
        aod_with_bands = np.nanmean(aod_with_bands, axis=band_axis)
        dims_after.pop(band_axis)

    lat_dim_name = "lat" if "lat" in dims_after else (
        "latitude" if "latitude" in dims_after else None
    )
    order: list[int] = []
    for dim_name in (time_name, lat_dim_name):
        if dim_name in dims_after:
            order.append(dims_after.index(dim_name))
    order += [i for i in range(len(dims_after)) if i not in order]
    aod = np.transpose(aod_with_bands, order)
    return mid_days, lat, aod


@lru_cache(maxsize=16)
def _load_volcanic_auto(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch ``path`` to the right volcanic loader.

    Returns the same ``(mid_days, lat, data)`` tuple as
    :func:`_load_monthly_zonal` regardless of which on-disk schema the
    file follows.  Criterion: presence of the CMIP6 ``ext_sun`` variable
    triggers :func:`_load_volcanic_cmip6`; otherwise fall back to the
    legacy ``aod(time, lat)`` schema.  Cached so that repeated calls in
    an AMIP time-step loop don't re-open the dataset just to dispatch
    (Codex review 2026-04-24).
    """
    ds = _open_forcing_dataset(path)
    dvars = set(ds.data_vars)
    ds.close()
    if "ext_sun" in dvars:
        return _load_volcanic_cmip6(path)
    if "ext_earth" in dvars and "aod" not in dvars:
        # CMIP6 file with only LW extinction — not a valid SW aerosol
        # source.  Fail with a specific error rather than the generic
        # "aod not found" from the legacy-path fallback.
        raise ValueError(
            f"Volcanic file {path!r} has only LW extinction "
            "('ext_earth') and no SW extinction ('ext_sun'); cannot be "
            "used as a SW AOD source.",
        )
    return _load_monthly_zonal(path, "aod")


@lru_cache(maxsize=16)
def _load_monthly_zonal_with_levels(path: str, varname: str):
    """Like ``_load_monthly_zonal`` but also returns pressure levels and a
    CF-time anchor for multi-year files.

    Returns
    -------
    (mid_days, first_date, lat, plev, data)

    * ``mid_days``: shape (ntime,), days since the first record.
    * ``first_date``: :class:`cftime.datetime` of the first record, or
      ``None`` when the time axis carries no CF ``units`` attribute.
      Used by :func:`get_ozone_at_time` to map the simulation day onto
      the file's absolute time axis for multi-year (``ntime > 12``)
      ozone files.
    * ``plev``: shape (nlev,) in [Pa] or ``None`` if no vertical dim.
    """
    ds = _open_forcing_dataset(path)
    if varname not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable {varname!r} not found in {path!r}")
    var = ds[varname]
    data = np.asarray(var.values, dtype=np.float64)
    dims = list(var.dims)
    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' variable in {path!r}")
    if "time" in ds:
        mid_days, first_date = _read_time_axis(ds)
    else:
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
        first_date = None

    plev = None
    for vname in ("plev", "level", "lev"):
        if vname in ds:
            plev = np.asarray(ds[vname].values, dtype=np.float64)
            # Convert hPa → Pa if needed (detect via units attr or magnitude)
            units = ds[vname].attrs.get("units", "")
            if units in ("hPa", "millibar", "mbar", "mb"):
                plev = plev * 100.0
            elif not units and plev.size > 0 and np.max(plev) < 1500.0:
                # Heuristic: surface pressure ~1013 hPa; if max < 1500 assume hPa
                plev = plev * 100.0
            break

    # Average over longitude if present → zonal mean
    lon_names = ("lon", "longitude")
    for lname in lon_names:
        if lname in dims:
            lon_ax = dims.index(lname)
            data = np.nanmean(data, axis=lon_ax)
            dims.pop(lon_ax)
            break

    # Ensure dimension order is (time, lat, plev) for downstream code.
    # CMIP6 files often have (time, plev, lat); swap if needed.
    if plev is not None and data.ndim == 3:
        lat_name = "lat" if "lat" in dims else "latitude"
        plev_name = next((v for v in ("plev", "level", "lev") if v in dims), None)
        if plev_name and lat_name in dims and plev_name in dims:
            lat_ax = dims.index(lat_name)
            plev_ax = dims.index(plev_name)
            if plev_ax < lat_ax:
                # (time, plev, lat) → (time, lat, plev)
                data = np.swapaxes(data, plev_ax, lat_ax)

    ds.close()
    return mid_days, first_date, lat, plev, data


@lru_cache(maxsize=16)
def _load_time_gpt(
    path: str,
    tsi_var: str,
    spectral_var: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load time-varying solar spectral forcing (Zarr or NetCDF).

    Expected variables
    ------------------
    - ``time`` (CF-decoded to float days)
    - optional TSI (1-D) — e.g. ``tsi`` in legoESM-native files or
      ``TSI`` in canonical CMIP6 / MPI-M spectral-solar files
    - spectral weights (2-D ``(time, ngpt)`` or ``(time, nband)``) —
      e.g. ``solar_fraction_by_gpt`` or the CMIP6 ``SSI_frac``

    Variable name lookup is case-insensitive so that a caller using
    legoESM defaults (``"tsi"`` / ``"solar_fraction_by_gpt"``) can still
    read MPI-M files where the stored names are ``"TSI"`` / ``"SSI_frac"``.
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds:
        ds.close()
        raise ValueError(f"No 'time' variable in spectral solar file {path!r}")
    times = _read_time_days(ds)

    # Case-insensitive data-var lookup — mirrors ``_load_timeseries``.  The
    # MPI-M CMIP6 spectral-solar file stores ``TSI`` / ``SSI_frac`` while
    # the in-tree default is ``tsi`` / ``solar_fraction_by_gpt``.
    varname_map = {name.lower(): name for name in ds.data_vars}

    def _resolve(name: str) -> str | None:
        return name if name in ds.data_vars else varname_map.get(name.lower())

    spectral_actual = _resolve(spectral_var)
    if spectral_actual is None:
        ds.close()
        raise ValueError(
            f"Spectral variable {spectral_var!r} not found in {path!r}",
        )
    spec = np.asarray(ds[spectral_actual].values, dtype=np.float64)
    if spec.ndim != 2 or spec.shape[0] != times.shape[0]:
        ds.close()
        raise ValueError(
            f"Spectral variable {spectral_actual!r} must have shape "
            f"(time, ngpt); got {spec.shape}",
        )
    tsi = None
    tsi_actual = _resolve(tsi_var)
    if tsi_actual is not None:
        tsi = np.asarray(ds[tsi_actual].values, dtype=np.float64)
        if tsi.ndim != 1 or tsi.shape[0] != times.shape[0]:
            ds.close()
            raise ValueError(
                f"TSI variable {tsi_actual!r} must have shape (time,); "
                f"got {tsi.shape}",
            )
    ds.close()
    return times, tsi, spec


def _interp_monthly_cyclic(mid_days: np.ndarray, data: np.ndarray, day: float) -> np.ndarray:
    """Interpolate a monthly-cyclic field (12, ...) to a day of year.

    Uses cyclic linear interpolation with period 365.25 days.
    """
    period = 365.25
    day_mod = day % period
    n = len(mid_days)
    idx_right = np.searchsorted(mid_days % period, day_mod)
    if idx_right >= n:
        idx_right = 0
    idx_left = (idx_right - 1) % n
    d_left = mid_days[idx_left] % period
    d_right = mid_days[idx_right] % period
    span = (d_right - d_left) % period
    if span == 0:
        return data[idx_left]
    w = ((day_mod - d_left) % period) / span
    return (1 - w) * data[idx_left] + w * data[idx_right]


def _interp_monthly_noncyclic(mid_days: np.ndarray, data: np.ndarray,
                               day: float) -> np.ndarray:
    """Linear interpolation along a non-cyclic multi-year monthly axis.

    For multi-year forcing files (e.g. CMIP6 ozone 1850–2014, 1980
    months), ``_interp_monthly_cyclic`` wraps with period 365.25 d and
    throws away interannual evolution.  This helper treats ``mid_days``
    as a monotonic absolute time axis and uses :func:`numpy.interp`
    semantics per trailing column (clamps at the endpoints when ``day``
    falls outside the file range).

    Parameters
    ----------
    mid_days : (ntime,) float, ascending.
    data : (ntime, ...) float, any trailing shape.
    day : float — requested day on the same axis as ``mid_days``.
    """
    if data.ndim == 1:
        return np.array(np.interp(day, mid_days, data), dtype=np.float64)
    trailing = data.shape[1:]
    flat = data.reshape(data.shape[0], -1)
    out = np.empty(flat.shape[1], dtype=np.float64)
    for k in range(flat.shape[1]):
        out[k] = np.interp(day, mid_days, flat[:, k])
    return out.reshape(trailing)


def _simday_to_file_day(sim_day: float, start_year: int,
                        first_date) -> float:
    """Map a simulation day (days since ``start_year-01-01``) onto a
    forcing file's absolute time axis anchored at ``first_date``.

    * ``first_date`` is a :class:`cftime.datetime` / ``numpy.datetime64``
      returned by :func:`_read_time_axis` — the calendar date of the
      file's first record.  ``file_day = (sim_epoch - first_date)·days
      + sim_day``.
    * ``first_date`` is ``None`` when the file's time axis has no CF
      ``units`` attribute (the test-fixture case).  In that case we
      fall through to ``file_day = sim_day``: the file's raw numeric
      time is assumed to share the simulation reference, which is the
      only consistent interpretation for a units-less axis.
    """
    if first_date is None:
        return sim_day
    try:
        import cftime
        sim_epoch = cftime.DatetimeGregorian(int(start_year), 1, 1)
        if hasattr(first_date, "year") and hasattr(first_date, "month"):
            delta = sim_epoch - first_date
            delta_days = delta.days + delta.seconds / 86400.0
            return delta_days + sim_day
    except Exception:
        pass
    # numpy.datetime64 path: take scalar difference in days.
    try:
        anchor = np.datetime64(f"{int(start_year):04d}-01-01")
        delta_days = float(
            (anchor - np.datetime64(first_date)) / np.timedelta64(1, "D"),
        )
        return delta_days + sim_day
    except Exception:
        return sim_day


def _interp_zonal_to_grid(lat_src: np.ndarray, field: np.ndarray,
                           lat_grid: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a zonal-mean field to model grid latitudes.

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitudes [degrees].  May be ascending (``-90 → +90``)
        or descending (``+89.5 → -89.5``, as in Kinne aerosol files);
        this helper detects a descending axis and flips both ``lat_src``
        and ``field`` along the leading axis before calling
        :func:`numpy.interp` (which requires strictly ascending ``xp``).
        Without this guard ``np.interp`` would silently extrapolate to
        the endpoint and produce a near-constant field at every model
        latitude, not a NaN / error (issue #207 bug 6).
    field : (nlat_src,) or (nlat_src, nlev) or (nlat_src, nlev, nband)
        Zonal-mean field at source latitudes.  Handles arbitrary trailing
        dimensions (e.g., 3D Kinne aerosol files with a band axis, #178).
    lat_grid : jax array, any shape
        Model grid latitudes [radians]. Will be converted to degrees.

    Returns
    -------
    jax array with shape (*lat_grid.shape,) or (*lat_grid.shape, *trailing)
    """
    lat_deg = np.asarray(jnp.degrees(lat_grid)).ravel()

    lat_src = np.asarray(lat_src)
    if lat_src.size > 1 and lat_src[0] > lat_src[-1]:
        # Descending source axis (e.g. Kinne aerosol files): flip so
        # ``np.interp``'s strictly-ascending-xp contract is satisfied.
        lat_src = lat_src[::-1]
        field = field[::-1]

    if field.ndim == 1:
        result = np.interp(lat_deg, lat_src, field)
        return jnp.array(result).reshape(lat_grid.shape)
    else:
        # Handle 2D, 3D, ... by flattening trailing dims, interpolating
        # each column independently, and reshaping back (issue #178).
        trailing_shape = field.shape[1:]
        n_cols = int(np.prod(trailing_shape))
        field_flat = field.reshape(field.shape[0], n_cols)
        result = np.zeros((len(lat_deg), n_cols))
        for k in range(n_cols):
            result[:, k] = np.interp(lat_deg, lat_src, field_flat[:, k])
        return jnp.array(result).reshape((*lat_grid.shape, *trailing_shape))


def _interp_vertical(field_plev: jnp.ndarray, plev_src: np.ndarray,
                      p_target: jnp.ndarray) -> jnp.ndarray:
    """Interpolate vertically in log-pressure space.

    Parameters
    ----------
    field_plev : (..., nlev_src)
        Field on source pressure levels.
    plev_src : (nlev_src,)
        Source pressure levels [Pa].
    p_target : (..., nlev_target)
        Target pressure levels [Pa].

    Returns
    -------
    (..., nlev_target) interpolated field.
    """
    log_p_src = np.log(np.maximum(plev_src, 1e-10))
    log_p_tgt = jnp.log(jnp.maximum(p_target, 1e-10))

    # Use numpy interp on flattened arrays
    shape_prefix = field_plev.shape[:-1]
    nlev_tgt = p_target.shape[-1]
    field_flat = np.asarray(field_plev).reshape(-1, len(plev_src))
    log_p_tgt_flat = np.asarray(log_p_tgt).reshape(-1, nlev_tgt)

    result = np.zeros((field_flat.shape[0], nlev_tgt))
    for i in range(field_flat.shape[0]):
        result[i] = np.interp(log_p_tgt_flat[i], log_p_src, field_flat[i])

    return jnp.array(result).reshape((*shape_prefix, nlev_tgt))


# ==============================================================================
# Greenhouse gases
# ==============================================================================

class GHGConfig(NamedTuple):
    """Greenhouse gas configuration.

    When source="constant", the specified concentrations are used at
    all times. When source="file", values are linearly interpolated
    from a NetCDF time series containing co2_ppmv, ch4_ppbv, n2o_ppbv.
    When source="annual_file", values are loaded from a CMIP6-style
    annual global-mean file (e.g. greenhouse_historical_plus.nc) with
    variables CO2, CH4, N2O, CFC_11, CFC_12 dimensioned (time, lat, lon)
    where lat=1, lon=1 and time is in fractional years.

    The concentrations are consumed by RRTMGP radiation (not gray).

    Fields
    ------
    co2_ppmv : float
        CO2 concentration [ppmv]. Default: 348 (AMIP II reference ~1979-1996).
    ch4_ppbv : float
        CH4 concentration [ppbv].
    n2o_ppbv : float
        N2O concentration [ppbv].
    cfc11_pptv : float
        CFC-11 concentration [pptv].
    cfc12_pptv : float
        CFC-12 concentration [pptv].
    source : str
        "constant", "file" (1-D time series), or "annual_file" (CMIP6
        annual global-mean file with year-indexed time axis).
    path : str
        Path to time-varying GHG file (only used if source="file" or
        "annual_file").
    start_year : int
        Simulation start year, used to convert simulation day → calendar
        year when source="annual_file".
    """
    co2_ppmv: float = 348.0
    ch4_ppbv: float = 1650.0
    n2o_ppbv: float = 306.0
    cfc11_pptv: float = 240.0
    cfc12_pptv: float = 530.0
    source: str = "constant"
    path: str = ""
    start_year: int = 1979


@lru_cache(maxsize=4)
def _load_ghg_annual_file(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load CMIP6-style annual global-mean GHG file (Zarr or NetCDF).

    Expected format: variables CO2, CH4, N2O, CFC_11, CFC_12 with
    dimensions (time, lat, lon) where lat=1, lon=1.  The time axis
    uses units ``"year as %Y.%f"`` giving fractional year values
    (e.g. 1850.0, 1851.0, ...).

    Returns
    -------
    (years, data) where years is shape (N,) and data maps variable
    names to 1-D numpy arrays of length N.  Values are in the file's
    native units (CO2 in 1e-6, CH4/N2O in 1e-9, CFCs in 1e-12).
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds.dims:
        ds.close()
        raise ValueError(f"GHG file {path!r} has no 'time' dimension")
    years = np.asarray(ds["time"].values, dtype=np.float64)

    data = {}
    for varname in ("CO2", "CH4", "N2O", "CFC_11", "CFC_12"):
        if varname not in ds.data_vars:
            ds.close()
            raise ValueError(f"Variable {varname!r} not found in {path!r}")
        arr = np.asarray(ds[varname].values, dtype=np.float64)
        # Squeeze spatial dimensions (lat=1, lon=1) → 1-D time series
        arr = arr.squeeze()
        if arr.ndim != 1 or arr.shape[0] != years.shape[0]:
            ds.close()
            raise ValueError(
                f"Variable {varname!r} must reduce to 1-D after "
                f"squeezing (got shape {arr.shape}, expected "
                f"({years.shape[0]},))"
            )
        data[varname] = arr

    ds.close()
    return years, data


def get_ghg_at_time(config: GHGConfig, day: float) -> dict:
    """Return GHG concentrations at a given simulation day.

    Parameters
    ----------
    config : GHGConfig
    day : float
        Simulation day (unused for constant source).

    Returns
    -------
    dict with keys "co2_ppmv", "ch4_ppbv", "n2o_ppbv", "cfc11_pptv",
    "cfc12_pptv".
    """
    if config.source == "constant":
        return {
            "co2_ppmv": config.co2_ppmv,
            "ch4_ppbv": config.ch4_ppbv,
            "n2o_ppbv": config.n2o_ppbv,
            "cfc11_pptv": config.cfc11_pptv,
            "cfc12_pptv": config.cfc12_pptv,
        }
    elif config.source == "file":
        if not config.path:
            raise ValueError("GHGConfig.path must be set when source='file'")
        varnames = ("co2_ppmv", "ch4_ppbv", "n2o_ppbv")
        times, data = _load_timeseries(config.path, varnames)
        result = {v: _interp_1d(times, data[v], day) for v in varnames}
        result["cfc11_pptv"] = config.cfc11_pptv
        result["cfc12_pptv"] = config.cfc12_pptv
        return result
    elif config.source == "annual_file":
        if not config.path:
            raise ValueError(
                "GHGConfig.path must be set when source='annual_file'"
            )
        years, data = _load_ghg_annual_file(config.path)
        # Convert simulation day → fractional year
        year = config.start_year + day / 365.25
        # File stores mole fractions scaled by unit metadata:
        # CO2 in 1e-6 (ppmv), CH4/N2O in 1e-9 (ppbv), CFCs in 1e-12 (pptv)
        return {
            "co2_ppmv": _interp_1d(years, data["CO2"], year),
            "ch4_ppbv": _interp_1d(years, data["CH4"], year),
            "n2o_ppbv": _interp_1d(years, data["N2O"], year),
            "cfc11_pptv": _interp_1d(years, data["CFC_11"], year),
            "cfc12_pptv": _interp_1d(years, data["CFC_12"], year),
        }
    else:
        raise ValueError(f"Unknown GHG source: {config.source!r}")


def ghg_concentrations_to_vmr(ghg: dict) -> dict:
    """Convert GHG concentrations dict to volume mixing ratios.

    Parameters
    ----------
    ghg : dict
        As returned by ``get_ghg_at_time`` (ppmv/ppbv/pptv units).

    Returns
    -------
    dict mapping RRTMGP gas names to VMR (dimensionless mole fractions).
    """
    vmr = {
        "co2": ghg["co2_ppmv"] * 1.0e-6,
        "ch4": ghg["ch4_ppbv"] * 1.0e-9,
        "n2o": ghg["n2o_ppbv"] * 1.0e-9,
    }
    if "cfc11_pptv" in ghg:
        vmr["cfc11"] = ghg["cfc11_pptv"] * 1.0e-12
    if "cfc12_pptv" in ghg:
        vmr["cfc12"] = ghg["cfc12_pptv"] * 1.0e-12
    return vmr


# ==============================================================================
# Ozone
# ==============================================================================

class OzoneConfig(NamedTuple):
    """Ozone forcing configuration.

    When enabled, provides column ozone or 3D ozone mixing ratio
    for radiation. Supports zonal-mean monthly climatology from
    NetCDF with latitude and optional vertical interpolation to the
    model grid.

    Expected NetCDF schema for climatology:
    - Variable: 'ozone', 'vmro3', 'o3', 'O3', or 'tro3' (auto-detected)
      with dims (time=12, lat[, plev])
    - Units: volume mixing ratio [mol/mol]
    - 'lat' in degrees, 'plev' in Pa (if 3D)

    Fields
    ------
    enabled : bool
        Whether ozone forcing is active.
    source : str
        "climatology" (monthly zonal-mean) or "file" (full 3D).
    path : str
        Path to ozone NetCDF file.
    use_reference_if_missing : bool
        If True and path is empty, use a built-in reference ozone profile.
    reference_p_peak_hPa : float
        Peak pressure for the reference profile [hPa].
    reference_o3_max_vmr : float
        Peak ozone VMR [mol/mol] for the reference profile.
    reference_sigma_logp : float
        Width of reference ozone in log-pressure space.
    reference_lat_dependence : bool
        If True, reference ozone increases toward poles.
    start_year : int
        Simulation start year.  Used for the non-cyclic dispatch: when
        the ozone file has ``ntime > 12`` records (e.g. CMIP6
        ``vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_185001-201412.nc``
        with 1980 months), ``get_ozone_at_time`` treats ``mid_days`` as a
        monotonic absolute-time axis and maps the simulation day onto the
        file's timeline via :func:`_simday_to_file_day`, preserving
        interannual ozone evolution instead of collapsing to a fixed
        12-month climatology.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""
    use_reference_if_missing: bool = False
    reference_p_peak_hPa: float = 30.0
    reference_o3_max_vmr: float = 8.0e-6
    reference_sigma_logp: float = 1.5
    reference_lat_dependence: bool = True
    start_year: int = 1850


def _reference_ozone_profile(
    lat_grid: jnp.ndarray,
    p_grid: jnp.ndarray | None,
    config: OzoneConfig,
) -> jnp.ndarray:
    """Reference ozone profile used when external ozone is active without a file."""
    if p_grid is None:
        base = jnp.full(lat_grid.shape, config.reference_o3_max_vmr)
        if config.reference_lat_dependence:
            base = base * (1.0 + 0.5 * jnp.sin(lat_grid) ** 2)
        return jnp.clip(base, 1.0e-10, None)

    p_hPa = p_grid / 100.0
    sigma = config.reference_sigma_logp
    p_peak = config.reference_p_peak_hPa
    o3 = config.reference_o3_max_vmr * jnp.exp(
        -0.5 * ((jnp.log(jnp.maximum(p_hPa, 1.0e-12)) - jnp.log(p_peak)) / sigma) ** 2,
    )
    if config.reference_lat_dependence:
        o3 = o3 * (1.0 + 0.5 * jnp.sin(lat_grid)[..., None] ** 2)
    return jnp.clip(o3, 1.0e-10, None)


_OZONE_VARNAMES = ("ozone", "vmro3", "o3", "O3", "tro3")


@lru_cache(maxsize=16)
def _detect_ozone_varname(path: str) -> str:
    """Auto-detect the ozone variable name in a Zarr store or NetCDF file.

    Tries common names: 'ozone', 'vmro3' (CMIP6), 'o3', 'O3', 'tro3'.
    """
    ds = _open_forcing_dataset(path)
    for name in _OZONE_VARNAMES:
        if name in ds.data_vars:
            ds.close()
            return name
    ds.close()
    raise ValueError(
        f"No ozone variable found in {path!r}. "
        f"Expected one of {_OZONE_VARNAMES}"
    )


def get_ozone_at_time(config: OzoneConfig, day: float,
                       lat_grid: jnp.ndarray | None = None,
                       p_grid: jnp.ndarray | None = None):
    """Return ozone field at a given simulation day.

    Parameters
    ----------
    config : OzoneConfig
    day : float
        Day of year (fractional).
    lat_grid : jax array or None
        Model grid latitudes [radians]. If provided, interpolates
        the zonal-mean climatology to these latitudes.
    p_grid : jax array or None
        Model pressure levels [Pa]. If provided and the ozone file
        contains a vertical dimension, interpolates vertically.

    Returns
    -------
    None if ozone is disabled.
    jax array interpolated to (lat_grid.shape,) or
    (lat_grid.shape, nlev) if 3D, or dict with "lat"/"ozone" keys
    if lat_grid is not provided.
    """
    if not config.enabled:
        return None
    if (not config.path) and config.use_reference_if_missing:
        if lat_grid is None:
            lat_ref = np.linspace(-90.0, 90.0, 181)
            lat_ref_rad = jnp.radians(jnp.array(lat_ref))
            return {"lat": lat_ref, "ozone": np.asarray(_reference_ozone_profile(lat_ref_rad, None, config))}
        return _reference_ozone_profile(lat_grid, p_grid, config)
    if not config.path:
        raise ValueError(
            "OzoneConfig.path must be set when enabled=True unless "
            "use_reference_if_missing=True",
        )

    varname = _detect_ozone_varname(config.path)
    mid_days, first_date, lat, plev, data = _load_monthly_zonal_with_levels(
        config.path, varname
    )
    # Dispatch cyclic (12-month climatology) vs non-cyclic (multi-year
    # interannually varying file, e.g. CMIP6 ozone 1850–2014 @ 1980 months).
    # Keyed on ``len(mid_days) > 12`` AND ``config.start_year`` is required
    # to map simulation day → file absolute time.
    if len(mid_days) > 12:
        file_day = _simday_to_file_day(day, config.start_year, first_date)
        ozone_interp = _interp_monthly_noncyclic(mid_days, data, file_day)
    else:
        ozone_interp = _interp_monthly_cyclic(mid_days, data, day)

    if lat_grid is not None:
        # Interpolate to model grid latitudes
        ozone_on_grid = _interp_zonal_to_grid(lat, ozone_interp, lat_grid)
        # Vertical interpolation if 3D data and pressure grid provided
        if plev is not None and p_grid is not None and ozone_on_grid.ndim > len(lat_grid.shape):
            ozone_on_grid = _interp_vertical(ozone_on_grid, plev, p_grid)
        return ozone_on_grid
    else:
        return {"lat": lat, "ozone": ozone_interp}


# ==============================================================================
# Aerosols
# ==============================================================================

class AerosolConfig(NamedTuple):
    """Aerosol forcing configuration.

    When enabled, provides prescribed aerosol optical depth from a
    zonal-mean monthly climatology. The AOD modifies SW radiation.

    Expected NetCDF schema for climatology:
    - Variable: 'aod' with dims (time=12, lat)
    - Units: dimensionless optical depth at 550 nm
    - 'lat' in degrees

    Fields
    ------
    enabled : bool
        Whether aerosol forcing is active.
    source : str
        "climatology" or "file".
    path : str
        Path to aerosol NetCDF file.
    use_reference_if_missing : bool
        If True and path is empty, use a built-in reference aerosol field.
    reference_aod_550 : float
        Reference background AOD at 550 nm.
    reference_lat_factor : float
        Strength of low-latitude enhancement in the reference AOD profile.
    volcanic_enabled : bool
        If True, add volcanic aerosol forcing on top of baseline aerosol.
    volcanic_path : str
        NetCDF path for volcanic AOD climatology/time-series (same schema as aerosol).
    volcanic_scale : float
        Multiplicative scaling applied to volcanic AOD.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""
    use_reference_if_missing: bool = False
    reference_aod_550: float = 0.03
    reference_lat_factor: float = 0.35
    volcanic_enabled: bool = False
    volcanic_path: str = ""
    volcanic_scale: float = 1.0


def _reference_aerosol_profile(lat_grid: jnp.ndarray, config: AerosolConfig) -> jnp.ndarray:
    """Reference zonal aerosol optical depth profile."""
    aod = config.reference_aod_550 * (
        1.0 + config.reference_lat_factor * jnp.cos(lat_grid) ** 2
    )
    return jnp.clip(aod, 0.0, None)


def get_aerosol_at_time(config: AerosolConfig, day: float,
                         lat_grid: jnp.ndarray | None = None):
    """Return aerosol optical depth at a given simulation day.

    Parameters
    ----------
    config : AerosolConfig
    day : float
        Day of year (fractional).
    lat_grid : jax array or None
        If provided, interpolates AOD to model grid latitudes.

    Returns
    -------
    None if aerosol is disabled.
    jax array if lat_grid provided, else dict with "lat"/"aod".
    """
    if not config.enabled:
        return None
    if config.path:
        mid_days, lat, data = _load_monthly_zonal(config.path, "aod")
        aod_interp = _interp_monthly_cyclic(mid_days, data, day)
        if lat_grid is not None:
            base_aod = _interp_zonal_to_grid(lat, aod_interp, lat_grid)
            # Kinne aerosol files may have extra dimensions (level, band).
            # Sum over all trailing dims to get total column AOD (issue #178).
            while base_aod.ndim > lat_grid.ndim:
                base_aod = jnp.sum(base_aod, axis=-1)
        else:
            # Sum trailing dims for dict-mode too
            aod_flat = aod_interp
            while aod_flat.ndim > 1:
                aod_flat = np.sum(aod_flat, axis=-1)
            base_aod = {"lat": lat, "aod": aod_flat}
    elif config.use_reference_if_missing:
        if lat_grid is None:
            lat = np.linspace(-90.0, 90.0, 181)
            lat_rad = jnp.radians(jnp.array(lat))
            base_aod = {"lat": lat, "aod": np.asarray(_reference_aerosol_profile(lat_rad, config))}
        else:
            base_aod = _reference_aerosol_profile(lat_grid, config)
    else:
        raise ValueError(
            "AerosolConfig.path must be set when enabled=True unless "
            "use_reference_if_missing=True",
        )

    # Optional volcanic contribution.
    if config.volcanic_enabled:
        if config.volcanic_path:
            # ``_load_volcanic_auto`` handles both the legacy
            # ``aod(time, lat)`` schema and the CMIP6 / MPI-M
            # ``ext_sun(solar_bands, lat, altitude, month)`` schema
            # (issue #207 bug 3) by integrating ext * dz over altitude
            # and returning a drop-in ``(mid_days, lat, aod)`` tuple.
            mid_days_v, lat_v, data_v = _load_volcanic_auto(config.volcanic_path)
            aod_v = _interp_monthly_cyclic(mid_days_v, data_v, day) * config.volcanic_scale
            if lat_grid is not None:
                volc = _interp_zonal_to_grid(lat_v, aod_v, lat_grid)
                # Sum trailing dims for multi-dimensional volcanic files
                while volc.ndim > lat_grid.ndim:
                    volc = jnp.sum(volc, axis=-1)
            else:
                aod_v_flat = aod_v
                while aod_v_flat.ndim > 1:
                    aod_v_flat = np.sum(aod_v_flat, axis=-1)
                volc = {"lat": lat_v, "aod": aod_v_flat}
        elif config.use_reference_if_missing:
            volc = (
                jnp.zeros_like(base_aod)
                if lat_grid is not None
                else {"lat": base_aod["lat"], "aod": np.zeros_like(base_aod["aod"])}
            )
        else:
            raise ValueError(
                "AerosolConfig.volcanic_path must be set when volcanic_enabled=True "
                "unless use_reference_if_missing=True",
            )

        if lat_grid is not None:
            return jnp.clip(base_aod + volc, 0.0, None)
        volc_aod = volc["aod"]
        if not np.array_equal(base_aod["lat"], volc["lat"]):
            volc_aod = np.interp(base_aod["lat"], volc["lat"], volc_aod)
        return {"lat": base_aod["lat"], "aod": np.clip(base_aod["aod"] + volc_aod, 0.0, None)}

    return base_aod


# ==============================================================================
# Solar irradiance
# ==============================================================================

class SolarConfig(NamedTuple):
    """Solar irradiance configuration.

    The seasonal cycle is handled by the radiation scheme's insolation
    calculation. This config adds support for TSI variations (e.g.,
    solar cycle, volcanic dimming).

    Constant TSI is active by default. Time-varying TSI from file is
    supported via source="file".

    Fields
    ------
    S_0 : float
        Total solar irradiance [W/m2]. Default: 1360.
    source : str
        "constant", "file", or "spectral_file".
    path : str
        Path to forcing NetCDF file.
    tsi_var : str
        Variable name for TSI time series.
    spectral_var : str
        Variable name for per-g-point solar fractions when source="spectral_file".
    normalize_spectral : bool
        If True, normalize interpolated spectral fractions to sum to one.
    start_year : int
        Simulation start year.  Used to convert simulation day (days since
        start) to the absolute day count used in file-based solar records
        whose time axis is expressed as "days since 1850-01-01".
    """
    S_0: float = 1360.0
    source: str = "constant"
    path: str = ""
    tsi_var: str = "tsi"
    spectral_var: str = "solar_fraction_by_gpt"
    normalize_spectral: bool = True
    start_year: int = 1979


def get_solar_forcing_at_time(config: SolarConfig, day: float) -> dict:
    """Return solar forcing dict with broadband TSI and optional spectral weights.

    Returns
    -------
    dict
        Keys:
        - ``"tsi"``: float [W/m^2]
        - ``"solar_fraction_by_gpt"``: jax array (ngpt,) or None
    """
    if config.source == "constant":
        return {"tsi": float(config.S_0), "solar_fraction_by_gpt": None}

    if config.source not in ("file", "spectral_file"):
        raise ValueError(f"Unknown solar source: {config.source!r}")

    if not config.path:
        raise ValueError(
            f"SolarConfig.path must be set when source={config.source!r}",
        )

    # Map simulation day (days since ``start_year-01-01``) onto the
    # file's absolute time axis.  CF ``units`` on the time variable
    # (e.g. CMIP6 spectral-solar "days since 1850-01-01") anchor the
    # conversion; units-less test fixtures fall through to
    # ``file_day = sim_day`` (file times interpreted in simulation
    # reference).  This replaces the earlier hardcoded
    # ``_REF_YEAR = 1850`` assumption, which broke units-less files.
    anchor = _load_time_anchor(config.path)
    abs_day = _simday_to_file_day(day, config.start_year, anchor)

    if config.source == "file":
        times, data = _load_timeseries(config.path, (config.tsi_var,))
        return {"tsi": _interp_1d(times, data[config.tsi_var], abs_day), "solar_fraction_by_gpt": None}

    if config.source == "spectral_file":
        times, tsi_series, spec_series = _load_time_gpt(
            config.path,
            config.tsi_var,
            config.spectral_var,
        )
        tsi_val = _interp_1d(times, tsi_series, abs_day) if tsi_series is not None else float(config.S_0)
        spec = _interp_2d_time(times, spec_series, abs_day)
        spec = np.clip(spec, 0.0, None)
        # CMIP6 solar files store one fraction per RRTMG-SW band (14 bands).
        # The RRTMG solver expects one fraction per g-point (112 g-points).
        # Expand only when the loaded spectral length matches the table's
        # ``n_bands`` EXACTLY.  Loose gates like ``shape[1] < 50`` catch
        # unit fixtures with 2 gpoints and crash the expansion because
        # ``bnd_limits_gpt`` indexes beyond the input array (Codex
        # review 2026-04-24); the strict match makes the heuristic
        # robust to arbitrary-sized inputs.
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            _DEFAULT_SW_GAS,
        )
        _n_bands, _n_gpt = _rrtmg_sw_band_counts(_DEFAULT_SW_GAS)
        if spec_series.shape[1] == _n_bands and _n_bands != _n_gpt:
            spec = _expand_bands_to_gpoints(spec, _DEFAULT_SW_GAS)
        if config.normalize_spectral:
            denom = float(np.sum(spec))
            if denom <= 0.0:
                raise ValueError(
                    f"Interpolated spectral forcing has non-positive sum at day={day}",
                )
            spec = spec / denom
        return {"tsi": tsi_val, "solar_fraction_by_gpt": jnp.array(spec)}

    # Unreachable — the ``source not in (..., ...)`` guard above already
    # raised for unknown sources before the file-I/O block.
    raise AssertionError("unreachable")


@lru_cache(maxsize=4)
def _rrtmg_sw_band_counts(rrtmg_sw_path: str) -> tuple[int, int]:
    """Return ``(n_bands, n_gpt)`` for the RRTMG-SW lookup table.

    Used by :func:`get_solar_forcing_at_time` to decide whether an
    input spectral axis needs the band→g-point expansion.  Cached so
    that repeated calls in a time-step loop don't reopen the table.
    """
    import xarray as xr
    ds = (xr.open_dataset(rrtmg_sw_path) if not rrtmg_sw_path.endswith(".zarr")
          else xr.open_zarr(rrtmg_sw_path))
    bnd_lims = ds["bnd_limits_gpt"].values.astype(int)
    ds.close()
    return int(bnd_lims.shape[0]), int(bnd_lims[:, 1].max())


def _expand_bands_to_gpoints(spec_bands: np.ndarray, rrtmg_sw_path: str) -> np.ndarray:
    """Expand per-band spectral fractions to per-g-point fractions.

    CMIP6 solar files (e.g. ``SSI_frac``) store one value per RRTMG-SW
    band (14 bands), but the RRTMG solver expects one value per g-point
    (112 g-points for the standard g112 table).  Each band's fraction is
    repeated uniformly across all g-points that belong to that band.

    Parameters
    ----------
    spec_bands : np.ndarray, shape (n_bands,)
        Per-band solar fractions.
    rrtmg_sw_path : str
        Path to the RRTMG-SW lookup table NetCDF/Zarr (for ``bnd_limits_gpt``).

    Returns
    -------
    np.ndarray, shape (n_gpt,)
        Per-g-point fractions.
    """
    import xarray as xr
    ds = xr.open_dataset(rrtmg_sw_path) if not rrtmg_sw_path.endswith(".zarr") \
        else xr.open_zarr(rrtmg_sw_path)
    # bnd_limits_gpt: (n_bands, 2) with 1-based [start, end] gpt indices
    bnd_lims = ds["bnd_limits_gpt"].values.astype(int)  # 1-indexed
    ds.close()
    n_gpt = int(bnd_lims[:, 1].max())
    out = np.zeros(n_gpt, dtype=np.float64)
    for i, (lo, hi) in enumerate(bnd_lims):
        out[lo - 1 : hi] = spec_bands[i]   # convert to 0-indexed slice
    return out


def get_tsi_at_time(config: SolarConfig, day: float) -> float:
    """Return total solar irradiance at a given simulation day.

    Parameters
    ----------
    config : SolarConfig
    day : float

    Returns
    -------
    float : TSI [W/m2]
    """
    return float(get_solar_forcing_at_time(config, day)["tsi"])


# ==============================================================================
# Combined external forcing config
# ==============================================================================

class ExternalForcingConfig(NamedTuple):
    """Combined configuration for all external forcings.

    Pass this to the AMIP experiment to enable/configure external forcings.
    All sub-configs default to inactive/constant, so the baseline AMIP
    experiment works without any external forcing files.
    """
    ghg: GHGConfig = GHGConfig()
    ozone: OzoneConfig = OzoneConfig()
    aerosol: AerosolConfig = AerosolConfig()
    solar: SolarConfig = SolarConfig()
