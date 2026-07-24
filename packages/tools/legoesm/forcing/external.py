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

import logging
from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.forcing.time_utils import (
    NOLEAP_DAYS_PER_MONTH,
    NOLEAP_DAYS_PER_YEAR,
    NOLEAP_MONTH_STARTS,
)

from legoesm import constants

logger = logging.getLogger(__name__)

# CF calendars whose year length differs from the model's noleap 365-day
# clock — files dated on these need the by-calendar-date sim-day mapping
# (see _simday_to_file_day).  ``julian`` and ``all_leap``/``366_day`` carry
# leap days too (an all-leap axis would drift +1 d/yr under the linear add).
# ``noleap``/``365_day`` stay linear (already exact); ``360_day`` keeps the
# legacy linear form (its pinned 360-day epoch arithmetic predates this fix).
_BY_DATE_MAPPED_CALENDARS = frozenset(
    {"gregorian", "standard", "proleptic_gregorian", "julian",
     "all_leap", "366_day"}
)


# --- Volcanic stratospheric LW aerosol placement / band collapse ---------------
# The CMIP6 ``bc_aeropt_cmip6_volc_lw_b16_sw_b14`` file is HEIGHT-RESOLVED
# (altitude 5-40 km) and carries per-band single-scattering albedo, so the LW
# volcanic aerosol is fed to RRTMGP as an ABSORPTION optical depth
# (ext*(1-omega)) placed at its true stratospheric pressure.
#
# US Standard Atmosphere 1976 base-layer table (NOAA-S/T 76-1562): geopotential
# base altitude [m], base temperature [K], lapse rate [K/m].  Used ONLY to map
# the file's GEOMETRIC altitude axis to pressure for placement.  The file
# altitude is geometric; ``_ussa1976_pressure`` converts to geopotential first
# (the height difference is ~0.5 % at 30 km but the PRESSURE effect is ~2 % at
# 30 km / ~3 % at 40 km, so the conversion IS applied).  The 5-40 km band the
# volcanic file occupies lies in the 11-47 km stratospheric layers.
_USSA1976_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
)
# US Std Atm 1976 sea-level reference pressure [Pa] (definition of the profile).
_USSA1976_P_SEALEVEL_PA = 101325.0
# Representative lower-stratosphere temperature [K] for the Planck-emission-
# weighted gray collapse of the 16 terrestrial bands (B3).  Volcanic sulfate
# sits at ~16-30 km where US Std Atm 1976 T ~ 217-227 K.  This is a documented
# approximation, NOT the exact gray equivalent (which would fold in the full
# spectral radiative kernel): on the real 1979 file a +/-10 K change moves the
# Planck-weighted gray column OD by ~+/-4.7 % via the band-varying OD (Codex
# review iter-1).
_VOLC_LW_PLANCK_TEMP_K = 220.0
# Sub-samples per band for the Planck-weight quadrature (numerics only).
_PLANCK_QUAD_NSUB = 16


def _ussa1976_pressure(z_m: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 pressure [Pa] at GEOMETRIC altitude ``z_m`` [m].

    Piecewise-analytic hydrostatic integral of :data:`_USSA1976_LAYERS`
    (constant-lapse power law / isothermal exponential), anchored at
    :data:`_USSA1976_P_SEALEVEL_PA`.  Monotone decreasing in height.  Used only
    to map the CMIP6 volcanic file's altitude axis onto pressure for
    stratospheric aerosol placement.

    The CMIP6 file altitude is GEOMETRIC, but the USSA base heights are
    GEOPOTENTIAL; the input is converted ``h = R_e z / (R_e + z)`` before the
    table lookup.  Skipping this leaves the mapped pressure low by ~1 % at
    20 km and ~3 % at 40 km (Codex review iter-1).
    """
    z = np.asarray(z_m, dtype=np.float64)
    r_e = float(constants.R_earth)
    z = r_e * z / (r_e + z)          # geometric -> geopotential height [m]
    g = float(constants.g)
    r_d = float(constants.R_d)

    def _layer_p(p_b, z_b, t_b, lapse, z_eval):
        # Hydrostatic pressure within one constant-lapse layer.
        if lapse == 0.0:
            return p_b * np.exp(-g * (z_eval - z_b) / (r_d * t_b))
        return p_b * (t_b / (t_b + lapse * (z_eval - z_b))) ** (g / (r_d * lapse))

    # Base pressures at each layer boundary (integrate upward from sea level).
    p_base = [_USSA1976_P_SEALEVEL_PA]
    for (z_b, t_b, lapse), (z_t, _, _) in zip(
        _USSA1976_LAYERS, _USSA1976_LAYERS[1:]
    ):
        p_base.append(_layer_p(p_base[-1], z_b, t_b, lapse, z_t))

    n = len(_USSA1976_LAYERS)
    p = np.full_like(z, np.nan)
    for i, (z_b, t_b, lapse) in enumerate(_USSA1976_LAYERS):
        z_t = _USSA1976_LAYERS[i + 1][0] if i + 1 < n else np.inf
        z_lo = -np.inf if i == 0 else z_b  # first layer also covers z < 0
        in_layer = (z >= z_lo) & (z < z_t)
        p = np.where(in_layer, _layer_p(p_base[i], z_b, t_b, lapse, z), p)
    return p


def _planck_band_weights(
    wl1_um: np.ndarray, wl2_um: np.ndarray, temp_k: float,
) -> np.ndarray:
    """Un-normalised Planck-emission weight per spectral band at ``temp_k``.

    Integrates the Planck function ``B_nu(T)`` over each band's wavenumber
    interval (band bounds given as wavelengths [um]).  The emission-weighted
    mean ``sum_b W_b*tau_b / sum_b W_b`` is a physically-motivated HEURISTIC
    gray value (better-motivated than a flat band mean, which overweights the
    ~zero-flux near-IR bands) when a per-band optical depth is collapsed to one
    value applied to every g-point (B3) -- NOT the exact gray equivalent, which
    would fold in the full spectral radiative kernel.  Only relative weights
    matter, so the result is un-normalised.
    """
    h = float(constants.h_planck)
    c = float(constants.c_light)
    k_b = float(constants.k_B)
    t = float(temp_k)
    w1 = np.asarray(wl1_um, dtype=np.float64) * 1.0e-6  # -> m
    w2 = np.asarray(wl2_um, dtype=np.float64) * 1.0e-6
    nu_lo = 1.0 / np.maximum(w1, w2)  # smaller wavenumber [1/m]
    nu_hi = 1.0 / np.minimum(w1, w2)  # larger wavenumber [1/m]
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    weights = np.empty(nu_lo.shape[0], dtype=np.float64)
    for b in range(nu_lo.shape[0]):
        nu = np.linspace(nu_lo[b], nu_hi[b], _PLANCK_QUAD_NSUB)
        x = h * c * nu / (k_b * t)
        b_nu = 2.0 * h * c**2 * nu**3 / np.expm1(x)
        weights[b] = _trapz(b_nu, nu)
    return weights


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


def _cftime_constructor_for_calendar(cal: str | None):
    """Return the cftime datetime constructor that matches a CF
    ``calendar`` string.

    Used in two places:
    * :func:`_extract_first_date` — so the file's first-record anchor
      stays in the file's native calendar (a noleap CF file produces
      a ``DatetimeNoLeap`` anchor, not Gregorian).
    * :func:`_simday_to_file_day` — so the simulation epoch is built
      in the same calendar before the cftime subtraction.

    The mapping covers every calendar legoESM forcing files use in
    practice (CMIP6 input4MIPs ozone / aerosol / GHG annual default
    to ``noleap``; some CMIP-like archives use ``360_day``).
    Unknown calendars fall back to ``DatetimeGregorian`` so the
    function never raises.
    """
    import cftime
    cal_norm = (cal or "standard").lower()
    return {
        "noleap": cftime.DatetimeNoLeap,
        "365_day": cftime.DatetimeNoLeap,
        "all_leap": cftime.DatetimeAllLeap,
        "366_day": cftime.DatetimeAllLeap,
        "360_day": cftime.Datetime360Day,
        "julian": cftime.DatetimeJulian,
        "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
        "gregorian": cftime.DatetimeGregorian,
        "standard": cftime.DatetimeGregorian,
    }.get(cal_norm, cftime.DatetimeGregorian)


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
      using a calendar-aware ``cftime.Datetime*`` constructor.

    The returned anchor stays in the file's native calendar
    (``noleap`` / ``360_day`` / ``gregorian`` / …) so downstream
    callers like :func:`_simday_to_file_day` can build a matching
    simulation epoch and avoid Gregorian-vs-noleap leap-day drift
    (Codex iter-6 review).

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
        ctor = _cftime_constructor_for_calendar(cal)
        if "months since" in ul or "years since" in ul:
            anchor_str = units.split("since", 1)[1].strip().split(" ")[0]
            y_str, m_str, d_str = anchor_str.split("-")
            # Build the anchor in the file's native calendar so the
            # offset arithmetic below stays calendar-consistent.
            anchor = ctor(int(y_str), int(m_str), int(d_str))
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
                base = ctor(ny, nm, anchor.day)
                frac_days = frac * (30.0 if cal == "360_day" else 365.25 / 12.0)
            else:  # years since
                whole = int(np.floor(v0))
                frac = v0 - whole
                base = ctor(
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
    """Linearly interpolate a 2-D time series (time, nfeat) at *day*.

    Vectorised search + broadcast — the previous per-feature
    ``for j in range(values.shape[1]): out[j] = np.interp(...)`` loop
    fired per simulated step with up to 112 g-points (CMIP6 spectral
    SSI), forcing 112 Python calls into ``np.interp`` per step.
    """
    if values.ndim != 2 or values.shape[0] != times.shape[0]:
        raise ValueError(
            "Expected values with shape (ntime, nfeat) matching times length; "
            f"got {values.shape} with times {times.shape}",
        )
    n = times.shape[0]
    if n == 0:
        return np.zeros((values.shape[1],), dtype=np.float64)
    if n == 1 or day <= times[0]:
        return values[0].astype(np.float64, copy=True)
    if day >= times[-1]:
        return values[-1].astype(np.float64, copy=True)
    # Locate the bracketing interval once, then linear-interp in NumPy.
    i = int(np.clip(np.searchsorted(times, day, side="right") - 1, 0, n - 2))
    dx = times[i + 1] - times[i]
    w = 0.0 if dx <= 0.0 else float((day - times[i]) / dx)
    return ((1.0 - w) * values[i] + w * values[i + 1]).astype(np.float64, copy=False)


@lru_cache(maxsize=16)
def _load_monthly_zonal_anchored(
    path: str, varname: str,
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """Load a monthly zonal-mean field from a Zarr store or NetCDF file,
    also returning the first record's CF-time anchor (``first_date``) so
    callers can map a simulation day onto the file's absolute time axis
    when the file spans multiple years.

    Handles files with full lat/lon grids (e.g. Kinne aerosol files with
    dims ``(time, band, lat, lon)``) by averaging over lon and any extra
    non-(time, lat) dimensions to produce a ``(ntime, nlat)`` array.

    This matches the API of :func:`_load_monthly_zonal_with_levels`
    used by the ozone loader.

    Returns
    -------
    (mid_days, first_date, lat, data) where ``first_date`` is a
    :class:`cftime.datetime` / :class:`numpy.datetime64` matching
    :func:`_read_time_axis`'s second return, or ``None`` when the
    file's time axis lacks a CF ``units`` attribute.
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

    return mid_days, first_date, lat, data


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
       matches the Kinne aerosol convention (``_load_monthly_zonal_anchored``
       averages non-(time, lat) axes via ``np.nanmean``) and avoids the
       n-bands-dependent inflation that a raw sum over 14 SW bands
       would introduce.

    Returns ``(mid_days, lat, aod)`` with ``aod`` shape
    ``(ntime, nlat)`` — a drop-in replacement for
    :func:`_load_monthly_zonal_anchored` output.

    For a CF-anchored variant that also returns the calendar anchor
    of the first record (needed by callers that map sim-day → file-day
    for non-cyclic multi-year volcanic time-series), see
    :func:`_load_volcanic_cmip6_anchored`.
    """
    mid_days, _first_date, lat, aod = _load_volcanic_cmip6_anchored(path)
    return mid_days, lat, aod


@lru_cache(maxsize=16)
def _load_volcanic_extinction_anchored(
    path: str,
    ext_var: str,
    kind_label: str,
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """Load a CMIP6 volcanic stratospheric-extinction variable (anchored).

    Shared body for the SW (``ext_sun``) and LW (``ext_earth``) loaders.
    Reads ``ext_var`` [1/km] on an altitude grid, integrates over altitude
    (trapezoid → AOD), collapses the spectral-band axis to a representative
    single-band broadband AOD via the band-MEAN (Kinne convention; summing
    bands would inflate it by the band count, not a broadband AOD), and
    returns ``(mid_days, first_date, lat, aod)`` with ``aod`` shape
    ``(ntime, nlat)``.  ``first_date`` is the CF-time anchor so callers can
    map a simulation day onto a multi-year file's absolute axis (1982 El
    Chichón / 1991 Pinatubo in their real months); ``None`` on a bare
    ``month`` axis (caller falls back to cyclic dispatch).

    The SW/LW bands are NOT interchangeable: reading ``ext_earth`` as a SW
    quantity (or ``ext_sun`` as LW) would inject the wrong band into the
    forcing budget, so each caller pins ``ext_var`` and a missing variable
    raises (Codex review 2026-04-24).  ``kind_label`` ('SW'/'LW') only
    labels the error.
    """
    ds = _open_forcing_dataset(path)
    if ext_var in ds.data_vars:
        ext_name = ext_var
    else:
        ds.close()
        raise ValueError(
            f"No CMIP6 volcanic {kind_label} extinction variable "
            f"{ext_var!r} in {path!r}.",
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
        mid_days, first_date = _read_time_axis(ds)
    elif "month" in dims:
        time_name = "month"
        nm = ext.shape[dims.index("month")]
        mid_days = np.array([15.5 + 30.4375 * m for m in range(nm)])
        first_date = None
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

    # Collapse the spectral-band axis to a single broadband AOD via the
    # band-mean (see docstring for the Kinne-convention rationale).
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
    return mid_days, first_date, lat, aod


def _load_volcanic_cmip6_anchored(
    path: str,
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """SHORTWAVE volcanic extinction (``ext_sun``); see
    :func:`_load_volcanic_extinction_anchored` for the shared body."""
    return _load_volcanic_extinction_anchored(path, "ext_sun", "SW")


@lru_cache(maxsize=8)
def _load_volcanic_lw_absorption_profile(
    path: str,
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray, np.ndarray]:
    """CMIP6 volcanic LW aerosol as a PRESSURE-RESOLVED absorption-OD profile.

    Fixes three defects in feeding ``bc_aeropt_cmip6_volc_lw_b16`` to the
    RRTMGP longwave ABSORPTION optical-depth slot:

    * **B1 extinction -> absorption.**  ``ext_earth`` is EXTINCTION; the LW slot
      wants ABSORPTION (single-scattering albedo 0).  Scaled per band by the
      co-located ``omega_earth``: ``abs = ext*(1-omega)``.  ``omega_earth``
      reaches ~0.4 in the window bands, so feeding raw extinction overstates
      the absorption OD by up to that factor.  If the file lacks
      ``omega_earth`` the sulfate LW ``omega ~ 0`` approximation is used
      (``abs == ext``) and documented.
    * **B2 stratospheric placement.**  The file is HEIGHT-RESOLVED (altitude
      5-40 km).  The per-layer profile is retained and each file layer tagged
      with a US-Std-Atm-1976 pressure edge, so the driver places the aerosol
      at its true stratospheric pressure instead of spreading a column AOD by
      tropospheric pressure mass (which lands ~90 % of it below the
      tropopause).
    * **B3 spectral collapse.**  The 16 terrestrial bands ARE collapsed to one
      gray value (the LW slot is applied to every g-point), but via a
      Planck-emission-weighted mean at :data:`_VOLC_LW_PLANCK_TEMP_K` rather
      than a flat band mean.  This is better-motivated than a flat mean (which
      overweights the near-IR bands that carry ~no LW flux at stratospheric
      temperature) but is a HEURISTIC, not the exact gray equivalent -- the
      exact single-tau response would fold in the full spectral radiative
      kernel (surface/gas/cloud transmission + local emission), not Planck
      emission alone.  Residual per-band spread is a documented approximation;
      threading true per-band tau through the g-point loop is deferred as
      disproportionate for this background-magnitude forcing.

    Returns ``(mid_days, first_date, lat, aod_profile, p_edges)``:

    * ``aod_profile`` ``(ntime, nlat, nlayer)`` per-layer absorption OD [-];
    * ``p_edges`` ``(nlayer+1,)`` [Pa] layer pressure edges, ASCENDING (index 0
      = top / lowest pressure), aligned so ``aod_profile[..., j]`` occupies
      ``[p_edges[j], p_edges[j+1]]``.

    Raises ``ValueError`` if the file lacks ``ext_earth`` (caller treats as no
    LW source and stays byte-identical to no volcanic LW aerosol).
    """
    ds = _open_forcing_dataset(path)
    if "ext_earth" not in ds.data_vars:
        ds.close()
        raise ValueError(
            f"No CMIP6 volcanic LW extinction 'ext_earth' in {path!r}.",
        )
    var = ds["ext_earth"]
    ext = np.asarray(var.values, dtype=np.float64)  # [1/km]
    dims = list(var.dims)

    # B1: single-scattering albedo -> absorption fraction (1-omega) per band.
    if "omega_earth" in ds.data_vars:
        omega = np.clip(
            np.asarray(ds["omega_earth"].values, dtype=np.float64), 0.0, 1.0,
        )
    else:
        omega = np.zeros_like(ext)  # sulfate LW omega ~ 0 approximation
    abs_ext = ext * (1.0 - omega)  # absorption extinction [1/km]

    alt_name = next(
        (v for v in ("altitude", "alt", "height", "lev") if v in ds), None,
    )
    if alt_name is None:
        ds.close()
        raise ValueError(f"No altitude coordinate in volcanic file {path!r}")
    alt_km = np.asarray(ds[alt_name].values, dtype=np.float64)

    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lat_name = "lat"
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lat_name = "latitude"
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' in volcanic file {path!r}")

    if "time" in dims:
        time_name = "time"
        mid_days, first_date = _read_time_axis(ds)
    elif "month" in dims:
        time_name = "month"
        nm = ext.shape[dims.index("month")]
        mid_days = np.array([15.5 + 30.4375 * m for m in range(nm)])
        first_date = None
    else:
        ds.close()
        raise ValueError(f"No time/month dimension in volcanic file {path!r}")

    # B3: Planck-emission-weighted collapse of the terrestrial-band axis.
    band_name = next(
        (d for d in dims
         if d not in (time_name, lat_name, alt_name)),
        None,
    )
    if band_name is not None:
        band_axis = dims.index(band_name)
        if "wl1_earth" in ds and "wl2_earth" in ds:
            weights = _planck_band_weights(
                np.asarray(ds["wl1_earth"].values),
                np.asarray(ds["wl2_earth"].values),
                _VOLC_LW_PLANCK_TEMP_K,
            )
        else:
            weights = np.ones(abs_ext.shape[band_axis], dtype=np.float64)
        # Weighted mean over bands (band axis -> gray).
        abs_ext = np.tensordot(
            weights, np.moveaxis(abs_ext, band_axis, 0), axes=(0, 0),
        ) / weights.sum()
        dims = [d for d in dims if d != band_name]
    ds.close()

    # Canonical (time, lat, altitude) order.
    order = [dims.index(time_name), dims.index(lat_name), dims.index(alt_name)]
    prof = np.transpose(abs_ext, order)  # (ntime, nlat, nalt) [1/km]

    # Altitude layer edges (midpoints, outer half-steps) and thickness [km].
    edges_km = np.empty(alt_km.size + 1, dtype=np.float64)
    edges_km[1:-1] = 0.5 * (alt_km[:-1] + alt_km[1:])
    edges_km[0] = alt_km[0] - 0.5 * (alt_km[1] - alt_km[0])
    edges_km[-1] = alt_km[-1] + 0.5 * (alt_km[-1] - alt_km[-2])
    dz_km = np.abs(np.diff(edges_km))  # (nalt,)

    # Per-layer AOD [-] = absorption extinction [1/km] * dz [km].
    aod_profile = prof * dz_km[None, None, :]
    p_edges = _ussa1976_pressure(edges_km * 1000.0)  # [Pa]

    # Return ASCENDING pressure (top first): altitude ascending -> pressure
    # descending, so flip the layer axis and the edges together.
    aod_profile = aod_profile[..., ::-1]
    p_edges = p_edges[::-1]
    return mid_days, first_date, lat, aod_profile, p_edges


@lru_cache(maxsize=16)
def _load_volcanic_auto(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch ``path`` to the right volcanic loader.

    Returns a ``(mid_days, lat, data)`` tuple (the anchor-free form of
    :func:`_load_monthly_zonal_anchored`) regardless of which on-disk
    schema the file follows.  See :func:`_load_volcanic_auto_anchored` for a
    variant that also returns the file's CF anchor (used by the
    multi-year non-cyclic dispatch in :func:`get_aerosol_at_time`).
    """
    mid_days, _first_date, lat, data = _load_volcanic_auto_anchored(path)
    return mid_days, lat, data


@lru_cache(maxsize=16)
def _load_volcanic_auto_anchored(
    path: str,
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """Like :func:`_load_volcanic_auto` but also returns ``first_date``.

    Criterion: presence of the CMIP6 ``ext_sun`` variable triggers
    :func:`_load_volcanic_cmip6_anchored`; otherwise fall back to the
    legacy ``aod(time, lat)`` schema via
    :func:`_load_monthly_zonal_anchored`.  Cached so that repeated
    calls in an AMIP time-step loop don't re-open the dataset just to
    dispatch.
    """
    ds = _open_forcing_dataset(path)
    dvars = set(ds.data_vars)
    ds.close()
    if "ext_sun" in dvars:
        return _load_volcanic_cmip6_anchored(path)
    if "ext_earth" in dvars and "aod" not in dvars:
        raise ValueError(
            f"Volcanic file {path!r} has only LW extinction "
            "('ext_earth') and no SW extinction ('ext_sun'); cannot be "
            "used as a SW AOD source.",
        )
    return _load_monthly_zonal_anchored(path, "aod")


def _ozone_unit_factor(units: str, varname: str) -> float:
    """Return the multiplicative factor that converts ozone from the
    file's units into volume mixing ratio (mol/mol).

    The legoESM radiation kernel (RRTMG / RRTMGP) consumes ozone as a
    dimensionless mole fraction.  CMIP6 input4MIPs ozone is supplied
    as ``vmro3`` in mol/mol, but older / non-CMIP6 datasets expose
    ozone as ``tro3`` in kg/kg (mass mixing ratio).  Without unit
    detection a kg/kg file would be silently consumed at the wrong
    magnitude (off by ``M_dry / M_o3 ≈ 0.6035``), producing a quiet
    ~40% radiative bias in the stratospheric SW heating.

    Detection rules
    ---------------
    1. ``mol mol-1`` / ``mole mole-1`` / ``vmr`` / ``"1"`` /
       ``dimensionless`` → return 1.0 (already vmr).
    2. ``kg kg-1`` / ``kg/kg`` / ``g/g`` → return ``M_dry / M_o3``
       (convert mass-mixing-ratio to vmr).
    3. ``ppmv`` → return 1e-6 (ppmv to mol/mol).
    4. ``ppbv`` → return 1e-9.
    5. Empty / unrecognised units: fall back on the **variable name**:
       - ``vmro3`` / ``o3`` / ``O3`` / ``ozone`` → 1.0 (vmr — CMIP6 default).
       - ``tro3`` → ``M_dry / M_o3`` (CMIP convention for mass mixing ratio).
    6. Anything else: raise ``ValueError`` with the offending units
       string so the user can either rename the variable or specify
       a known unit.

    Parameters
    ----------
    units : str
        Value of the variable's ``units`` attribute (lower-cased
        before comparison).
    varname : str
        Variable name, used as a fallback when ``units`` is empty.

    Returns
    -------
    float
        Multiplicative factor: ``vmr_data = file_data * factor``.
    """
    from legoesm import constants
    u = (units or "").strip().lower().replace(" ", "")
    # Canonicalise: "mol mol-1" → "molmol-1"; "kg kg-1" → "kgkg-1".
    vmr_strings = {
        "mol/mol", "molmol-1", "molemole-1", "mole/mole",
        "vmr", "1", "dimensionless", "fraction", "",
    }
    mmr_strings = {
        "kg/kg", "kgkg-1", "kg.kg-1", "g/g", "gg-1",
        "mass_mixing_ratio", "massmixingratio",
    }
    if u in vmr_strings or u in {"mol mol-1"}:  # extra guard
        # When units is empty fall through to the varname heuristic
        # below (CMIP6 default is vmr).
        if u == "":
            if varname in {"vmro3", "ozone", "o3", "O3"}:
                return 1.0
            if varname == "tro3":
                # CMIP convention: ``tro3`` is mass mixing ratio.
                return constants.M_dry / constants.M_o3
            return 1.0  # last-ditch default: assume vmr
        return 1.0
    if u in mmr_strings:
        return constants.M_dry / constants.M_o3
    if u in {"ppmv", "ppm"}:
        return 1.0e-6
    if u in {"ppbv", "ppb"}:
        return 1.0e-9
    if u in {"pptv", "ppt"}:
        return 1.0e-12
    raise ValueError(
        f"Unrecognised ozone units {units!r} on variable {varname!r}. "
        f"Supported: 'mol mol-1' (vmr, CMIP6 default), 'kg kg-1' "
        f"(mass mixing ratio), 'ppmv', 'ppbv', 'pptv', or empty."
    )


@lru_cache(maxsize=16)
def _load_monthly_zonal_with_levels(path: str, varname: str):
    """Like ``_load_monthly_zonal_anchored`` but also returns pressure levels
    for multi-year files.

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
    * ``data`` is **always returned in volume mixing ratio (mol/mol)**.
      When the file's ozone variable carries ``units = "kg kg-1"`` (or
      the variable is named ``tro3`` per CMIP convention), the data is
      multiplied by ``M_dry / M_o3 ≈ 0.6035`` so the downstream
      radiation kernel — which consumes vmr — sees physically
      consistent values regardless of the file's native unit.  The
      conversion is logged at info-level when applied.
    """
    ds = _open_forcing_dataset(path)
    if varname not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable {varname!r} not found in {path!r}")
    var = ds[varname]
    data = np.asarray(var.values, dtype=np.float64)
    var_units = var.attrs.get("units", "")
    # Apply the vmr conversion (no-op for CMIP6 vmro3, multiplicative
    # for tro3 / kg-kg-1 / ppm-style files).
    factor = _ozone_unit_factor(var_units, varname)
    if factor != 1.0:
        logger.info(
            f"[ozone-loader] Converting {varname!r} from units={var_units!r} "
            f"to vmr (mol/mol) with factor {factor:.5g}; "
            f"file: {path!r}"
        )
        data = data * factor
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

    Uses cyclic linear interpolation with period 365.0 days — the model
    clock is a strict noleap calendar (``time_utils.day_to_calendar``) and
    the AMIP SST/SIC climatology wrap (``amip.get_forcing_at_time``) also
    uses 365.0; the previous 365.25 period drifted the seasonal phase by
    0.25 d/yr against both (audit F5).
    """
    period = 365.0
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
    months), ``_interp_monthly_cyclic`` wraps with period 365 d and
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
    # Vectorised lerp instead of the previous per-trailing-column
    # ``for k in range(...): np.interp(...)`` loop.  For CMIP6 ozone
    # (~64 lat × 80 plev) this is ~5k Python calls per AMIP forcing
    # update — replaced with a single ``np.searchsorted`` + broadcast.
    n = mid_days.shape[0]
    if n == 0:
        return np.zeros(trailing, dtype=np.float64)
    if n == 1 or day <= mid_days[0]:
        return data[0].astype(np.float64, copy=True)
    if day >= mid_days[-1]:
        return data[-1].astype(np.float64, copy=True)
    i = int(np.clip(np.searchsorted(mid_days, day, side="right") - 1, 0, n - 2))
    dx = mid_days[i + 1] - mid_days[i]
    w = 0.0 if dx <= 0.0 else float((day - mid_days[i]) / dx)
    return ((1.0 - w) * data[i] + w * data[i + 1]).astype(np.float64, copy=False)


def _calendar_date_fields(date) -> tuple[int, int, int, float]:
    """Return ``(year, month, day_of_month, day_frac)`` for a calendar-aware
    date object (``cftime.datetime`` / ``datetime.datetime``) or a
    ``numpy.datetime64``.  ``day_frac`` is the fraction of day in [0, 1)."""
    if hasattr(date, "year") and hasattr(date, "month"):
        day_frac = (
            date.hour * 3600.0
            + date.minute * 60.0
            + date.second
            + getattr(date, "microsecond", 0) / 1e6
        ) / 86400.0
        return int(date.year), int(date.month), int(date.day), day_frac
    d = np.datetime64(date)
    months_int = int(d.astype("datetime64[M]").astype(np.int64))
    year = months_int // 12 + 1970
    month = months_int % 12 + 1
    days_d = d.astype("datetime64[D]")
    dom = int(
        (days_d - d.astype("datetime64[M]").astype("datetime64[D]"))
        .astype(np.int64)
    ) + 1
    day_frac = float((d - days_d) / np.timedelta64(1, "D"))
    return year, month, dom, day_frac


def _model_noleap_date(sim_day: float, start_year: int) -> tuple[int, int, int, float]:
    """Calendar date ``(year, month, day_of_month, day_frac)`` of a model
    day count (days since ``start_year-01-01`` on the model's noleap
    365-day clock).  Handles negative ``sim_day`` (dates before the sim
    epoch) via floor division."""
    year_off = int(np.floor(sim_day / NOLEAP_DAYS_PER_YEAR))
    rem = sim_day - year_off * NOLEAP_DAYS_PER_YEAR
    doy0 = min(int(np.floor(rem)), NOLEAP_DAYS_PER_YEAR - 1)
    day_frac = rem - doy0
    # NOLEAP_MONTH_STARTS is length 13 ([12] == 365), so side="right" gives
    # the 1-based month directly.
    month = int(np.searchsorted(
        np.asarray(NOLEAP_MONTH_STARTS), doy0, side="right",
    ))
    dom = doy0 - NOLEAP_MONTH_STARTS[month - 1] + 1
    return start_year + year_off, month, dom, day_frac


def _cyclic_phase_anchor(
    mid_days: np.ndarray, data: np.ndarray, first_date,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-anchor a <=12-record climatology at the first record's noleap
    day-of-year (audit F4), returning ``(mid_days, data)`` co-sorted.

    ``_read_time_axis`` returns days since the FIRST RECORD, so a CF-dated
    monthly climatology has ``mid_days[0] == 0`` and cyclic interpolation
    would place a mid-January-stamped record at Jan 1 — a ~15-day forward
    phase shift.  Offsetting by the first record's noleap day-of-year
    (Feb 29 collapsed onto Feb 28) restores the stamped phase (a mid-Jan
    record sits at day ~14.5-15.5).  The anchored axis is wrapped into
    [0, 365) and co-sorted with ``data`` so that a climatology starting
    mid-year (e.g. a July-first file) still satisfies the ascending-axis
    contract of :func:`_interp_monthly_cyclic`'s ``searchsorted``.  A
    units-less axis (``first_date is None``) already carries day-of-year
    values and is returned unchanged (e.g. the ``15.5 + 30.4375*m`` fallback).

    Residual (audit FL4, ACCEPTED not fixed): only the phase ORIGIN is
    re-anchored; within-year spacing keeps the file's native elapsed days, so a
    leap-year-dated climatology's post-February records sit <=1 day late.
    Exact for the noleap-dated CMIP6 convention (the production case). A
    per-record date reconstruction was rejected: ``_to_days_float`` collapses
    ``months since`` / ``years since`` axes to average-length elapsed days, so
    reconstructing dates from ``first_date + elapsed`` is itself ~1-day wrong
    for those axes (codex review) — no net improvement, and threading the raw
    per-record dates through three loaders is disproportionate to a <=1-day
    residual on a rare leap-year-dated file.
    """
    if first_date is None:
        return mid_days, data
    try:
        _, month, dom, day_frac = _calendar_date_fields(first_date)
    except Exception:
        return mid_days, data
    dom = min(dom, NOLEAP_DAYS_PER_MONTH[month - 1])  # Feb 29 -> Feb 28
    offset = NOLEAP_MONTH_STARTS[month - 1] + (dom - 1) + day_frac
    anchored = (np.asarray(mid_days, dtype=np.float64) + offset) % 365.0
    order = np.argsort(anchored, kind="stable")
    if np.array_equal(order, np.arange(order.size)):
        return anchored, data
    return anchored[order], np.asarray(data)[order]


def _simday_to_file_day(sim_day: float, start_year: int,
                        first_date) -> float:
    """Map a simulation day (days since ``start_year-01-01`` on the model's
    noleap clock) onto a forcing file's absolute time axis anchored at
    ``first_date``.

    * ``first_date`` is a :class:`cftime.datetime` / ``numpy.datetime64``
      returned by :func:`_read_time_axis` — the calendar date of the
      file's first record.
    * ``first_date`` is ``None`` when the file's time axis has no CF
      ``units`` attribute (the test-fixture case).  In that case we
      fall through to ``file_day = sim_day``: the file's raw numeric
      time is assumed to share the simulation reference, which is the
      only consistent interpretation for a units-less axis.

    **Calendar-aware mapping**:

    * Noleap-family (and ``360_day``) anchors keep the linear form
      ``file_day = (sim_epoch - first_date)·days + sim_day`` with the
      sim epoch built in the file's calendar (Codex iter-5 review) — exact
      for a noleap file, whose axis advances in step with the model clock.
    * Leap-bearing anchors (``gregorian`` / ``standard`` /
      ``proleptic_gregorian`` / ``julian`` / ``all_leap`` cftime, or
      ``datetime64``) use a BY-CALENDAR-DATE mapping (audit F1): adding the
      noleap ``sim_day`` linearly onto a Gregorian axis drifts ~1 day per
      4 years (the file gains leap days the model clock never lives
      through — ~9 days over 1979-2014; an all-leap axis drifts 1 d/yr).
      Instead the model day is converted to its noleap calendar date
      (:func:`_model_noleap_date`) and THAT date is located on the file
      axis, so the model reads the file at its simulated calendar date
      exactly.
    """
    if first_date is None:
        return sim_day
    cal = str(getattr(first_date, "calendar", None) or "standard").lower()
    if not hasattr(first_date, "calendar") or cal in _BY_DATE_MAPPED_CALENDARS:
        # Gregorian-dated file: by-calendar-date mapping (see docstring).
        try:
            y, m, d, day_frac = _model_noleap_date(sim_day, int(start_year))
            if hasattr(first_date, "calendar"):
                ctor = _cftime_constructor_for_calendar(first_date.calendar)
                delta = ctor(y, m, d) - first_date
                return delta.days + delta.seconds / 86400.0 + day_frac
            target = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}")
            delta_days = float(
                (target - np.datetime64(first_date)) / np.timedelta64(1, "D"),
            )
            return delta_days + day_frac
        except Exception:
            pass
    try:
        import cftime
        # Calendar-aware epoch: pick the cftime constructor that
        # matches ``first_date.calendar`` so the subtraction stays
        # calendar-consistent.  Falls back to Gregorian when the
        # anchor is not a calendar-aware cftime datetime.
        if hasattr(first_date, "calendar"):
            ctor = _cftime_constructor_for_calendar(first_date.calendar)
            sim_epoch = ctor(int(start_year), 1, 1)
        else:
            sim_epoch = cftime.DatetimeGregorian(int(start_year), 1, 1)
        if hasattr(first_date, "year") and hasattr(first_date, "month"):
            delta = sim_epoch - first_date
            delta_days = delta.days + delta.seconds / 86400.0
            return delta_days + sim_day
    except Exception:
        pass
    # numpy.datetime64 fallback path: take scalar difference in days
    # (reached only if the by-calendar-date branch above raised).
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

    # Vectorised lerp: locate the bracketing source latitude once, then
    # gather both endpoints and combine.  Replaces the previous
    # ``for k in range(n_cols): np.interp(...)`` loop, which scaled
    # quadratically on T170+ (n_cols ~ 10⁴) at every external-forcing
    # update.
    nsrc = lat_src.shape[0]
    if nsrc == 1:
        result_flat = np.broadcast_to(
            field.reshape(1, -1), (lat_deg.size, np.prod(field.shape[1:])),
        ).copy()
    else:
        idx = np.clip(np.searchsorted(lat_src, lat_deg) - 1, 0, nsrc - 2)
        x0 = lat_src[idx]
        x1 = lat_src[idx + 1]
        # Avoid division by zero on duplicated source points.
        dx = np.where(x1 > x0, x1 - x0, 1.0)
        w = np.where(x1 > x0, (lat_deg - x0) / dx, 0.0)
        # Clamp at the endpoints (matches np.interp's flat extrapolation).
        w = np.clip(w, 0.0, 1.0)
        # Reshape ``field`` to (nsrc, n_cols) and gather along leading.
        trailing_shape = field.shape[1:]
        n_cols = int(np.prod(trailing_shape))
        field_flat = np.asarray(field).reshape(nsrc, n_cols)
        f0 = field_flat[idx]                    # (n_lat, n_cols)
        f1 = field_flat[idx + 1]                # (n_lat, n_cols)
        result_flat = (1.0 - w[:, None]) * f0 + w[:, None] * f1
    return jnp.array(result_flat).reshape((*lat_grid.shape, *field.shape[1:]))


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

    shape_prefix = field_plev.shape[:-1]
    nlev_tgt = p_target.shape[-1]
    nsrc = log_p_src.shape[0]
    field_flat = np.asarray(field_plev).reshape(-1, nsrc)
    log_p_tgt_flat = np.asarray(log_p_tgt).reshape(-1, nlev_tgt)

    # Canonicalise to ascending log_p_src.  CMIP6 ozone files (e.g.
    # ``vmro3_input4MIPs_ozone_*.nc``) ship plev descending (1000→0.1 hPa);
    # ``np.searchsorted`` requires an ascending xp, otherwise the lerp
    # silently lands on the wrong bracketing pair (bug surfaced as zero/
    # near-zero ozone in the synthetic AMIP deck unit test).
    if nsrc > 1 and log_p_src[0] > log_p_src[-1]:
        log_p_src = log_p_src[::-1]
        field_flat = field_flat[:, ::-1]

    # Vectorised lerp in log-pressure space.  Replaces the
    # ``for i in range(ncol): np.interp(...)`` loop that scaled
    # ``ncol × n_lev`` Python-level for every forcing update — at
    # T170 with 14 000 columns the loop dominated wall time.
    if nsrc == 1:
        result = np.broadcast_to(field_flat[:, :1], (field_flat.shape[0], nlev_tgt)).copy()
    else:
        # Find bracketing index per (col, target-level).  ``np.searchsorted``
        # is vectorised so this is one C call.
        idx = np.clip(np.searchsorted(log_p_src, log_p_tgt_flat) - 1, 0, nsrc - 2)
        x0 = log_p_src[idx]
        x1 = log_p_src[idx + 1]
        dx = np.where(x1 > x0, x1 - x0, 1.0)
        w = np.where(x1 > x0, (log_p_tgt_flat - x0) / dx, 0.0)
        w = np.clip(w, 0.0, 1.0)
        # Gather endpoints from ``field_flat`` along last axis.
        col_idx = np.arange(field_flat.shape[0])[:, None]
        f0 = field_flat[col_idx, idx]             # (ncol, nlev_tgt)
        f1 = field_flat[col_idx, idx + 1]
        result = (1.0 - w) * f0 + w * f1

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


# CF/CMIP6 ``units`` strings → multiplicative factor that converts the
# stored value to a DIMENSIONLESS mole fraction (mol/mol).  Covers the
# input4MIPs convention (``"1"`` / ``"mole_fraction"`` already mol/mol)
# and the common pre-scaled forms (ppm/ppb/ppt and their ``1e-6`` etc.
# spellings).  Used by ``_load_ghg_annual_file`` so the rest of the
# pipeline reasons in a single, unambiguous unit.
_GHG_UNIT_TO_MOLE_FRACTION = {
    # Already dimensionless mole fraction.  NOTE: empty units ("") is
    # deliberately NOT here — a file with no units attribute is
    # AMBIGUOUS (mol/mol vs ppm) and must go through the magnitude
    # heuristic + warning, never be assumed mol/mol (codex review).
    "1": 1.0, "mol/mol": 1.0, "mol mol-1": 1.0, "mol mol^-1": 1.0,
    "mole_fraction": 1.0, "mole fraction": 1.0, "dimensionless": 1.0,
    # ppm family → 1e-6.
    "1e-6": 1.0e-6, "ppm": 1.0e-6, "ppmv": 1.0e-6,
    "umol/mol": 1.0e-6, "µmol/mol": 1.0e-6, "umol mol-1": 1.0e-6,
    "micromol/mol": 1.0e-6,
    # ppb family → 1e-9.
    "1e-9": 1.0e-9, "ppb": 1.0e-9, "ppbv": 1.0e-9,
    "nmol/mol": 1.0e-9, "nmol mol-1": 1.0e-9, "nanomol/mol": 1.0e-9,
    # ppt family → 1e-12.
    "1e-12": 1.0e-12, "ppt": 1.0e-12, "pptv": 1.0e-12,
    "pmol/mol": 1.0e-12, "pmol mol-1": 1.0e-12, "picomol/mol": 1.0e-12,
}


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
    names to 1-D numpy arrays of length N.  Values are normalized to
    DIMENSIONLESS mole fractions (mol/mol) using each variable's
    ``units`` attribute (audit 2026-06-11) so the caller can convert to
    ppmv/ppbv/pptv with fixed factors regardless of the file's storage
    unit.  An unrecognised ``units`` string falls back to a magnitude
    heuristic (CO2-scale value < 1 ⇒ already mole fraction) and logs a
    warning rather than silently feeding a 1e6-wrong concentration to
    radiation.
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
        var = ds[varname]
        arr = np.asarray(var.values, dtype=np.float64)
        # Squeeze spatial dimensions (lat=1, lon=1) → 1-D time series
        arr = arr.squeeze()
        if arr.ndim != 1 or arr.shape[0] != years.shape[0]:
            ds.close()
            raise ValueError(
                f"Variable {varname!r} must reduce to 1-D after "
                f"squeezing (got shape {arr.shape}, expected "
                f"({years.shape[0]},))"
            )
        # Normalize to mole fraction via the units attribute.
        units = str(var.attrs.get("units", "")).strip().lower()
        if units in _GHG_UNIT_TO_MOLE_FRACTION:
            arr = arr * _GHG_UNIT_TO_MOLE_FRACTION[units]
        else:
            # Unknown units: fall back to a magnitude heuristic on the
            # CO2-scale (mole fraction ~3e-4 vs ppmv ~300) and warn.
            ref = float(np.nanmax(np.abs(arr)))
            scale = 1.0
            if varname == "CO2" and ref > 1.0:
                scale = 1.0e-6
            elif varname in ("CH4", "N2O") and ref > 1.0e-3:
                scale = 1.0e-9
            elif varname.startswith("CFC") and ref > 1.0e-6:
                scale = 1.0e-12
            logger.warning(
                "GHG file %s: variable %s has unrecognised/empty units "
                "%r; assuming scale %g to mole fraction (max=%g). Set a "
                "CF units attribute (e.g. '1', 'ppm', 'nmol/mol') to "
                "remove this guess.",
                path, varname, var.attrs.get("units", ""), scale, ref,
            )
            arr = arr * scale
        # ``@lru_cache`` returns the SAME object on every call; mark the
        # arrays read-only so a caller's in-place mutation cannot poison
        # subsequent cached reads (codex review).
        arr.flags.writeable = False
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
        # Convert simulation day → fractional year on the model's noleap
        # (365-day) clock — matching the experiment-transient override in
        # model_driver (day/365.0) and the calendar-date mapping the other
        # forcing channels use.  365.25 here drifted ~9 days over a 35-year
        # run relative to the rest of the forcing suite (audit 2026-07-21).
        year = config.start_year + day / 365.0
        # ``_load_ghg_annual_file`` returns DIMENSIONLESS mole fractions
        # (mol/mol), normalized from the file's units attribute.  This
        # function's contract is ppmv/ppbv/pptv, so convert with the
        # fixed factors — the inverse of ``ghg_concentrations_to_vmr``.
        # (audit 2026-06-11: the previous code returned the raw mole
        # fraction labelled "co2_ppmv", so radiation saw CO2 ≈ 3.4e-10
        # vmr — effectively zero CO2, a ~1 K/day spurious LW cooling.)
        return {
            "co2_ppmv": _interp_1d(years, data["CO2"], year) * 1.0e6,
            "ch4_ppbv": _interp_1d(years, data["CH4"], year) * 1.0e9,
            "n2o_ppbv": _interp_1d(years, data["N2O"], year) * 1.0e9,
            "cfc11_pptv": _interp_1d(years, data["CFC_11"], year) * 1.0e12,
            "cfc12_pptv": _interp_1d(years, data["CFC_12"], year) * 1.0e12,
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
        # Phase-anchor a CF-dated climatology at its first record's
        # day-of-year (audit F4: days-since-first-record put mid-Jan at Jan 1).
        mid_anchored, data_anchored = _cyclic_phase_anchor(
            mid_days, data, first_date,
        )
        ozone_interp = _interp_monthly_cyclic(mid_anchored, data_anchored, day)

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
    volcanic_lw_enabled : bool
        If True, ALSO load the LONGWAVE volcanic stratospheric extinction
        (``ext_earth`` in CMIP6 ``bc_aeropt_cmip6_volc_*`` files) as a
        column LW absorption optical depth, exposed via
        :func:`get_aerosol_lw_at_time`.  Default OFF — when disabled (or
        when the volcanic file has no LW band) the LW aerosol path returns
        zeros / None and the run is byte-identical (zeros LW od is a no-op
        in the RRTMGP solver).  Independent of the SHORTWAVE
        ``volcanic_enabled`` path, which is unaffected.  (gap #9)
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
    volcanic_lw_enabled: bool = False
    # Calendar year of simulation day 0.  Required for the non-cyclic
    # dispatch in :func:`get_aerosol_at_time` so multi-year volcanic
    # files (e.g. 1850–2014 CMIP6 ``bc_aeropt_cmip6_volc_*`` with the
    # 1982 El Chichón and 1991 Pinatubo eruptions) are sampled at their
    # actual calendar months instead of being collapsed onto a 12-month
    # cyclic axis.  Mirrors :class:`OzoneConfig.start_year`.
    start_year: int = 1979


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
        # Dispatch cyclic (12-month climatology) vs non-cyclic
        # (multi-year, e.g. real CMIP6 input4MIPs aerosol).  Keyed on
        # ``len(mid_days) > 12``; ``config.start_year`` maps the
        # simulation day onto the file's absolute time axis when a CF
        # anchor is present.  Without this dispatch, a 36-year aerosol
        # file would be sampled cyclically (mod 365.25), throwing away
        # interannual evolution.
        mid_days, first_date, lat, data = _load_monthly_zonal_anchored(
            config.path, "aod",
        )
        if len(mid_days) > 12:
            file_day = _simday_to_file_day(day, config.start_year, first_date)
            aod_interp = _interp_monthly_noncyclic(mid_days, data, file_day)
        else:
            # Phase-anchor a CF-dated climatology at its first record's
            # day-of-year (audit F4).
            mid_anchored, data_anchored = _cyclic_phase_anchor(
                mid_days, data, first_date,
            )
            aod_interp = _interp_monthly_cyclic(mid_anchored, data_anchored, day)
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
            # ``_load_volcanic_auto_anchored`` handles both the legacy
            # ``aod(time, lat)`` schema and the CMIP6 / MPI-M
            # ``ext_sun(solar_bands, lat, altitude, month)`` schema
            # (issue #207 bug 3), and additionally returns the file's
            # CF time anchor so multi-year volcanic time series with
            # real eruption calendars (1982 El Chichón, 1991 Pinatubo,
            # …) are sampled non-cyclically — the previous cyclic
            # path mod-365.25-wrapped a multi-year file onto 12 months
            # and lost the eruption calendars entirely (Codex iter-3
            # review).
            mid_days_v, first_date_v, lat_v, data_v = _load_volcanic_auto_anchored(
                config.volcanic_path
            )
            if len(mid_days_v) > 12:
                file_day_v = _simday_to_file_day(
                    day, config.start_year, first_date_v,
                )
                aod_v = (
                    _interp_monthly_noncyclic(mid_days_v, data_v, file_day_v)
                    * config.volcanic_scale
                )
            else:
                mid_anchored_v, data_anchored_v = _cyclic_phase_anchor(
                    mid_days_v, data_v, first_date_v,
                )
                aod_v = (
                    _interp_monthly_cyclic(mid_anchored_v, data_anchored_v, day)
                    * config.volcanic_scale
                )
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


def get_aerosol_lw_at_time(config: AerosolConfig, day: float,
                           lat_grid: jnp.ndarray | None = None):
    """Return the LONGWAVE volcanic aerosol as a pressure-resolved profile.

    Mirrors :func:`get_aerosol_at_time` but returns the volcanic
    stratospheric LW (terrestrial-IR) ABSORPTION optical-depth profile,
    loaded from the ``ext_earth`` band of the CMIP6
    ``bc_aeropt_cmip6_volc_*`` file via
    :func:`_load_volcanic_lw_absorption_profile` (extinction -> absorption via
    ``1-omega``, Planck-weighted gray band collapse, and the file's native
    altitude axis mapped to pressure).  There is no background LW aerosol
    climatology, so the baseline is zero and the result is the volcanic LW
    absorption alone.

    Parameters
    ----------
    config : AerosolConfig
    day : float
        Day of year (fractional).
    lat_grid : jax array or None
        If provided, interpolates the profile to model grid latitudes.

    Returns
    -------
    None
        If LW aerosol is disabled (``volcanic_lw_enabled`` False), or enabled
        with no ``volcanic_path``, or the file carries no ``ext_earth`` LW
        band.  Callers default a None to zeros, so the run stays
        byte-identical to no volcanic LW aerosol.
    tuple ``(aod_profile, p_edges)``
        If ``lat_grid`` provided: ``aod_profile`` ``(ncol, nlayer)`` per-file-
        layer absorption OD (>= 0) and ``p_edges`` ``(nlayer+1,)`` [Pa]
        ascending layer pressure edges, ready for
        :func:`legoesm.forcing.surface_utils.place_stratospheric_aod_profile_to_layers`.
    dict
        Otherwise ``{"lat": ..., "aod_profile": ..., "p_edges": ...}``.
    """
    # Default OFF: only the explicit LW switch + a real volcanic file with
    # an ``ext_earth`` band produces nonzero LW aerosol.  Anything else
    # returns None ⇒ the driver fills zeros ⇒ no-op in RRTMGP.
    if not config.volcanic_lw_enabled or not config.volcanic_path:
        return None

    try:
        mid_days_v, first_date_v, lat_v, prof_v, p_edges = (
            _load_volcanic_lw_absorption_profile(config.volcanic_path)
        )
    except ValueError:
        # File has no LW (``ext_earth``) band — treat as "no LW source".
        return None

    # Interpolate the (ntime, nlat, nlayer) profile in time -> (nlat, nlayer).
    if len(mid_days_v) > 12:
        file_day_v = _simday_to_file_day(day, config.start_year, first_date_v)
        prof_t = _interp_monthly_noncyclic(mid_days_v, prof_v, file_day_v)
    else:
        mid_anchored_v, prof_anchored_v = _cyclic_phase_anchor(
            mid_days_v, prof_v, first_date_v,
        )
        prof_t = _interp_monthly_cyclic(mid_anchored_v, prof_anchored_v, day)
    prof_t = prof_t * config.volcanic_scale  # (nlat, nlayer)

    if lat_grid is not None:
        # (nlat, nlayer) -> (*lat_grid.shape, nlayer); the driver hands a
        # raveled column-latitude grid, so flatten leading dims to
        # (ncol, nlayer).  The nlayer axis is the vertical PROFILE and is
        # intentionally NOT summed away (unlike the gray SW/column path).
        prof_col = _interp_zonal_to_grid(lat_v, prof_t, lat_grid)
        prof_col = jnp.reshape(
            jnp.clip(prof_col, 0.0, None), (-1, prof_t.shape[-1]),
        )
        return prof_col, jnp.asarray(p_edges)

    return {
        "lat": lat_v,
        "aod_profile": np.clip(prof_t, 0.0, None),
        "p_edges": np.asarray(p_edges),
    }


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
    spectral_band_order : str
        Band ordering of a *per-band* (14-band) ``spectral_file`` input,
        used to align it to the RRTMGP-SW g-point table before
        expansion (issue #322).  Ignored for per-g-point inputs.

        - ``"auto"`` (default): rotate to RRTMGP order IFF the input
          carries the unambiguous MPI-M CMIP6 signature (variable
          ``SSI_frac`` or ``swflux_14band`` filename); otherwise leave
          untouched.  Never silently mis-expands the canonical CMIP6
          file and never rotates a generic file lacking the signature.
        - ``"rrtmg_sw"``: always rotate (caller asserts the file is in
          RRTMG-SW / CMIP order, 820-2680 cm^-1 band last).
        - ``"as_is"``: never rotate (caller asserts the file is already
          in RRTMGP band order).
    normalize_spectral : bool
        If True, normalize interpolated spectral fractions to sum to one.
    start_year : int
        Simulation start year.  Used to convert simulation day (days since
        start) to the absolute day count used in file-based solar records
        whose time axis is expressed as "days since 1850-01-01".
    """
    S_0: float = constants.S_0
    source: str = "constant"
    path: str = ""
    tsi_var: str = "tsi"
    spectral_var: str = "solar_fraction_by_gpt"
    spectral_band_order: str = "auto"
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
            DEFAULT_SW_GAS,
        )
        if config.spectral_band_order not in ("auto", "as_is", "rrtmg_sw"):
            raise ValueError(
                "SolarConfig.spectral_band_order must be 'auto', 'as_is' "
                f"or 'rrtmg_sw'; got {config.spectral_band_order!r}",
            )
        _n_bands, _n_gpt = _rrtmg_sw_band_counts(DEFAULT_SW_GAS)
        if spec_series.shape[1] == _n_bands and _n_bands != _n_gpt:
            # A per-band (14-band) file may need reordering to the
            # RRTMGP-SW g-point table order before index-based expansion.
            # The MPI-M CMIP6 ``swflux_14band_cmip6`` file is in RRTMG-SW
            # order (820-2680 cm^-1 overlap band LAST), which differs
            # from RRTMGP order (that band FIRST) by a one-band cyclic
            # rotation.  Expanding by index without reordering shifts the
            # whole spectrum one band toward longer wavelengths, dumping
            # UV flux into the near-IR water-vapour band (~2x clear-sky
            # SW absorption, issue #322).
            #
            #   "rrtmg_sw" -> always rotate (caller asserts CMIP order).
            #   "as_is"    -> never rotate (caller asserts RRTMGP order).
            #   "auto"     -> rotate IFF the input carries the unambiguous
            #                 MPI-M CMIP6 signature (variable ``SSI_frac``
            #                 or ``swflux_14band`` filename), else leave
            #                 untouched.  This never silently mis-expands
            #                 the canonical CMIP6 file, and never rotates a
            #                 generic file that lacks the signature.
            _looks_cmip = (
                "ssi_frac" in (config.spectral_var or "").lower()
                or "swflux_14band" in (config.path or "").lower()
            )
            if config.spectral_band_order == "rrtmg_sw" or (
                config.spectral_band_order == "auto" and _looks_cmip
            ):
                if config.spectral_band_order == "auto":
                    logger.info(
                        "Auto-detected MPI-M CMIP6 14-band solar file %r "
                        "(spectral_var=%r); applying 'rrtmg_sw' band order "
                        "(RRTMG-SW -> RRTMGP one-band rotation, issue "
                        "#322). Set spectral_band_order explicitly to "
                        "override.", config.path, config.spectral_var,
                    )
                spec = _cmip_sw_band_order_to_rrtmgp(spec)
            spec = _expand_bands_to_gpoints(spec, DEFAULT_SW_GAS)
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
    """Expand per-band spectral fractions to per-g-point fractions
    such that the band-integrated flux is preserved.

    CMIP6 solar files (e.g. ``SSI_frac``) store one *band-integrated*
    fraction per RRTMG-SW band (14 bands).  The RRTMG solver expects
    one fraction per g-point (112 g-points for the g112 table).  When
    each band's value is **copied** to every g-point in that band, the
    per-g-point integral over the band becomes ``f_b * n_gpt_b``
    instead of ``f_b`` — different g-point counts across bands then
    distort the per-band ratio after a downstream sum-to-1
    normalization (Codex iter-5 review).

    Fix: distribute each band's fraction over its g-points PROPORTIONAL
    to the table's own per-g-point solar source (``solar_source_quiet``),
    rescaled to the band total ``f_b``.  RRTMGP g-points within a band
    carry very unequal solar weight — the source concentrates flux in the
    window (weakly-absorbing) g-points and assigns almost none to the
    strongly-absorbing ones.  The earlier UNIFORM split (``f_b / n_gpt_b``)
    smeared each band's flux evenly across its g-points, dumping solar
    energy into the strong-absorption g-points and ~doubling clear-sky
    atmospheric SW absorption (24.5% → 48% in a single-column test) — a
    within-band variant of the band-order bug #322 (the band rotation was
    correct; the within-band shape was not).  Using the source shape gives
    ``Σ_{g ∈ band b} f_g = f_b`` AND the physically correct spectral
    distribution.  Falls back to the uniform split for any band whose
    source sums to zero, or if the table lacks ``solar_source_quiet``.

    Parameters
    ----------
    spec_bands : np.ndarray, shape ``(..., n_bands)``
        Per-band solar fractions.  Trailing axis must equal ``n_bands``.
        (Time-major arrays of shape ``(ntime, n_bands)`` are also
        supported and broadcast over the leading axis.)
    rrtmg_sw_path : str
        Path to the RRTMG-SW lookup table NetCDF/Zarr (for
        ``bnd_limits_gpt`` and ``solar_source_quiet``).

    Returns
    -------
    np.ndarray, shape ``(..., n_gpt)``
        Per-g-point fractions, with band-integrated flux preserved.
    """
    import xarray as xr
    ds = xr.open_dataset(rrtmg_sw_path) if not rrtmg_sw_path.endswith(".zarr") \
        else xr.open_zarr(rrtmg_sw_path)
    # bnd_limits_gpt: (n_bands, 2) with 1-based [start, end] gpt indices
    bnd_lims = ds["bnd_limits_gpt"].values.astype(int)  # 1-indexed
    # Per-g-point solar source = the physical within-band spectral shape.
    if "solar_source_quiet" in ds:
        solar_src = np.asarray(ds["solar_source_quiet"].values, dtype=np.float64)
    else:
        solar_src = None
    ds.close()
    n_gpt = int(bnd_lims[:, 1].max())
    spec_bands = np.asarray(spec_bands, dtype=np.float64)

    # Per-band within-band weights summing to 1 (source shape, or uniform).
    band_weights = []  # list of (lo0, hi, w[g]) with 0-based lo0
    for (lo, hi) in bnd_lims:
        lo0, n_in = lo - 1, int(hi - lo + 1)
        if solar_src is not None:
            w = solar_src[lo0:hi]
            s = w.sum()
            w = w / s if s > 0 else np.full(n_in, 1.0 / n_in)
        else:
            w = np.full(n_in, 1.0 / n_in)
        band_weights.append((lo0, hi, w))

    if spec_bands.ndim == 1:
        out = np.zeros(n_gpt, dtype=np.float64)
        for i, (lo0, hi, w) in enumerate(band_weights):
            out[lo0:hi] = spec_bands[i] * w
        return out
    # Trailing-axis case: (..., n_bands) → (..., n_gpt).
    leading_shape = spec_bands.shape[:-1]
    out = np.zeros(leading_shape + (n_gpt,), dtype=np.float64)
    for i, (lo0, hi, w) in enumerate(band_weights):
        out[..., lo0:hi] = spec_bands[..., i:i + 1] * w
    return out


def _cmip_sw_band_order_to_rrtmgp(spec_bands: np.ndarray) -> np.ndarray:
    """Re-order per-band SW fractions from CMIP6 / RRTMG-SW band order
    to the RRTMGP-SW gas-optics-table band order.

    The MPI-M CMIP6 spectral-solar file (``swflux_14band_cmip6``) stores
    its 14 fractions in **RRTMG-SW** band order, where the long-wave
    overlap band (820-2680 cm^-1) is the LAST band.  The RRTMGP-SW
    gas-optics table (``rrtmgp-gas-sw-g112.nc``, ``bnd_limits_wavenumber``)
    lists the SAME 14 wavenumber intervals but with that band FIRST, so
    the two orderings differ by exactly a one-band cyclic rotation
    (issue #322):

        RRTMGP band ``i``  ==  CMIP band ``(i - 1) mod 14``

    hence ``np.roll(spec, +1, axis=-1)`` converts CMIP order -> RRTMGP
    order.  Without it, ``_expand_bands_to_gpoints`` (which maps by array
    index) assigns each CMIP band's flux to the wrong RRTMGP band,
    shifting the whole spectrum one band toward longer wavelengths — e.g.
    UV flux lands in the near-IR water-vapour band, roughly doubling
    clear-sky shortwave absorption.

    This is applied ONLY on the ``spectral_file`` ingest path (per-band
    14-element MPI-M input).  ``_expand_bands_to_gpoints`` itself stays a
    pure index-preserving expansion so its band-integral contract is
    unchanged.

    Parameters
    ----------
    spec_bands : np.ndarray, shape ``(..., 14)``
        Per-band fractions in CMIP / RRTMG-SW order (trailing axis).

    Returns
    -------
    np.ndarray
        Same shape, re-ordered to RRTMGP-SW band order.
    """
    spec_bands = np.asarray(spec_bands, dtype=np.float64)
    return np.roll(spec_bands, 1, axis=-1)


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
