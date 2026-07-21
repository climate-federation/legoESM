"""CRU-JRA (CLM datm format) reanalysis loader for TRENDY / LMIP land runs.

Reference
---------
Sitch, S., et al. (2024). "Trends and Drivers of Terrestrial Sources and Sinks
of Carbon Dioxide: An Overview of the TRENDY Project." Global Biogeochemical
Cycles, 38, e2024GB008102. doi:10.1029/2024GB008102.

Spec
----
CRU-JRA v2.5 provides **6-hourly 0.5-degree** atmospheric forcing (1901-present)
on a 365-day (noleap) calendar.  Distributed in CLM / CTSM ``datm`` format as
three annual NetCDF streams (``clmforc.CRUJRAv2.5_0.5x0.5.<stream>.<year>.nc``):

* ``Solr``  : ``FSDS``     [W/m^2]          downward shortwave
* ``Prec``  : ``PRECTmms`` [mm/s = kg/m^2/s] total precipitation
* ``TPQWL`` : ``TBOT`` [K], ``PSRF`` [Pa], ``QBOT`` [kg/kg], ``WIND`` [m/s],
              ``FLDS`` [W/m^2]

**Time-stamp convention (verified in the example 2023 files, important for the
M2 disaggregation):** ``Solr`` is stamped at the **start** of each 6-h interval
(00, 06, 12, 18 UTC) and is an interval-mean flux to be redistributed by
``cos(zenith)``; ``TPQWL`` and ``Prec`` are stamped at the interval **midpoint**
(03, 09, 15, 21 UTC) — ``TPQWL`` is treated as instantaneous (linear interp) and
``Prec`` as an interval-mean rate (constant hold).  This loader therefore carries
TWO time axes (``time_s`` for TPQWL/Prec, ``time_s_solar`` for FSDS).

Design
------
Mirrors the OMIP-2 JRA55-do loader (``ocean/forcing/jra55_do.py``): a
:class:`LandForcing` ``NamedTuple`` container on the native grid, a deterministic
:func:`synthetic_land_forcing` fallback (so smoke runs need no external data), and
a :func:`load_cru_jra` entry point.  Spatial regridding reuses the shared
KD-tree / inverse-distance machinery (``legoesm.grids.regridding``) into the land
model's column space, and the variable map produces the standard
:class:`legoesm.core.coupling_fields.AtmToSurface`.

This module is M1 of the LMIP forcing workplan (``docs/land/lmip_s3_scope.md``):
read + regrid + variable map with **nearest-time** selection.  The 6h->dt
temporal disaggregation (zenith-weighted SW, constant-hold precip, linear interp
for the rest) is M2 and builds on the two time axes carried here.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import moist_air_density
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.grids.regridding import (
    RegridWeights,
    compute_latlon_to_voronoi_weights,
    regrid_scalar_nan_aware,
)
from legoesm.land.forcing.solar import cos_solar_zenith

# --- CRU-JRA native grid / cadence ---
CRUJRA_NLON: int = 720           # 0.5 deg longitude
CRUJRA_NLAT: int = 360           # 0.5 deg latitude
CRUJRA_FREQ_HOURS: int = 6       # 6-hourly cadence
CRUJRA_FILE_PREFIX: str = "clmforc.CRUJRAv2.5_0.5x0.5"

# --- CLM datm stream / variable names ---
_STREAM_SOLAR = "Solr"
_STREAM_PRECIP = "Prec"
_STREAM_TPQWL = "TPQWL"
_VAR_FSDS = "FSDS"
_VAR_PRECT = "PRECTmms"
_VAR_TBOT = "TBOT"
_VAR_PSRF = "PSRF"
_VAR_QBOT = "QBOT"
_VAR_WIND = "WIND"
_VAR_FLDS = "FLDS"

# --- time / unit conversions (exact) ---
_SEC_PER_DAY = 86400.0
_SEC_PER_HOUR = 3600.0
_HOURS_PER_DAY = 24.0
_DAYS_PER_YEAR_NOLEAP = 365.0    # CRU-JRA calendar: fixed 365-day (noleap) year
_CRUJRA_STEPS_PER_YEAR = 1460    # 365 noleap days x 4 (6-hourly)

# --- variable-map parameters ---
_DEFAULT_CO2_PPMV = 412.0        # constant CO2 this push (transient series = carbon WS)
SNOW_RAIN_RAMP_K = 2.0           # rain/snow partition ramp width above T_freeze (CLM convention)

# --- synthetic fallback climatology (no external data needed for smoke/tests) ---
_SYN_T_EQUATOR_K = 300.0         # equatorial air temperature
_SYN_T_POLE_K = 263.0            # polar air temperature
_SYN_RH = 0.8                    # uniform relative humidity for q
_SYN_SW_PEAK = 300.0             # noon-equator downward shortwave [W/m^2]
_SYN_LW = 300.0                  # uniform downward longwave [W/m^2]
_SYN_PRECIP = 2.0e-5             # tropical precipitation rate [kg/m^2/s]
_SYN_WIND = 3.0                  # uniform wind speed [m/s]
_SYN_PSRF = constants.p_ref      # uniform surface pressure [Pa]
_LAT_SOUTH_DEG = -90.0           # synthetic grid latitude bounds (pole to pole)
_LAT_NORTH_DEG = 90.0


class LandForcing(NamedTuple):
    """One year of CRU-JRA atmospheric forcing on the native lat-lon grid.

    Spatial arrays have shape ``(n_time, n_lat, n_lon)``.  Two time axes are
    carried because the CLM datm streams are stamped differently (see module
    docstring): ``time_s`` applies to ``tbot/psrf/qbot/wind/flds/prectmms``
    (midpoint-stamped) and ``time_s_solar`` to ``fsds`` (start-stamped).  Stored
    in float32 (observational forcing; cast to the model dtype at the variable
    map) to keep a native-resolution year tractable in host memory.
    """
    lon: np.ndarray            # (n_lon,) deg E in [0, 360)
    lat: np.ndarray            # (n_lat,) deg N in [-90, 90]
    time_s: np.ndarray         # (n_time,) seconds since year start (TPQWL/Prec)
    time_s_solar: np.ndarray   # (n_time,) seconds since year start (FSDS)
    fsds: np.ndarray           # W/m^2   downward shortwave
    flds: np.ndarray           # W/m^2   downward longwave
    prectmms: np.ndarray       # kg/m^2/s total precipitation
    tbot: np.ndarray           # K       lowest-level temperature
    psrf: np.ndarray           # Pa      surface pressure
    qbot: np.ndarray           # kg/kg   specific humidity
    wind: np.ndarray           # m/s     SCALAR wind SPEED (CRU-JRA has no u/v)
    year: int = 0


class LandForcingColumns(NamedTuple):
    """CRU-JRA forcing regridded to the land model's column space.

    Spatial arrays have shape ``(n_time, ncol)`` — the native ``(n_lat, n_lon)``
    grid collapsed onto the unstructured model columns by :func:`regrid_forcing`.
    Time axes are unchanged from :class:`LandForcing`.
    """
    time_s: np.ndarray         # (n_time,)
    time_s_solar: np.ndarray   # (n_time,)
    fsds: np.ndarray           # (n_time, ncol)
    flds: np.ndarray
    prectmms: np.ndarray
    tbot: np.ndarray
    psrf: np.ndarray
    qbot: np.ndarray
    wind: np.ndarray
    year: int = 0


def _stream_path(data_dir: Path, stream: str, year: int, prefix: str,
                 suffix: str = "") -> Path:
    """CLM datm stream path ``<prefix>.<stream>.<year><suffix>.nc``.

    ``suffix`` covers naming variants after the year, e.g. the TRENDY glade files
    ``clmforc.TRENDY.c2023_0.5x0.5.Solr.2022_cdf5.nc`` (``suffix="_cdf5"``); the
    UEA-style example files use ``suffix=""``.
    """
    return data_dir / f"{prefix}.{stream}.{year}{suffix}.nc"


def _time_seconds(ds, *, allow_units: str) -> np.ndarray:
    """Return the time axis in seconds since the year start.

    CRU-JRA CLM files use ``days since <year>-01-01`` (noleap); ``allow_units``
    is the expected ``units`` substring asserted for safety.
    """
    t = ds["time"]
    units = str(t.attrs.get("units", ""))
    if allow_units not in units:
        raise ValueError(
            f"unexpected CRU-JRA time units {units!r}; expected to contain "
            f"{allow_units!r} (days-since the file year)"
        )
    return np.asarray(t.values, dtype=np.float64) * _SEC_PER_DAY


def read_crujra_year(
    data_dir,
    year: int,
    *,
    prefix: str = CRUJRA_FILE_PREFIX,
    suffix: str = "",
    time_indices: Optional[Sequence[int]] = None,
) -> LandForcing:
    """Read one CRU-JRA year from the three CLM datm NetCDF streams.

    Parameters
    ----------
    data_dir : path-like
        Directory holding ``<prefix>.{Solr,Prec,TPQWL}.<year>.nc``.
    year : int
        Calendar year (used for the filename and stored on the result).
    prefix : str
        Filename prefix (default :data:`CRUJRA_FILE_PREFIX`).
    time_indices : sequence of int, optional
        Subset of the 1460 6-hourly steps to read.  Strongly recommended for
        local development — a full native-resolution year is ~3 GB/variable.
        ``None`` reads the whole year (production / Derecho).

    Returns
    -------
    LandForcing
        Native-grid forcing (float32 spatial fields).
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "CRU-JRA real-data load requires xarray; install with "
            "``pip install xarray netcdf4``"
        ) from exc

    data_dir = Path(data_dir)
    sel = None if time_indices is None else np.asarray(time_indices, dtype=int)

    def _open(stream):
        return xr.open_dataset(
            _stream_path(data_dir, stream, year, prefix, suffix), decode_times=False
        )

    def _read(ds, var):
        da = ds[var] if sel is None else ds[var].isel(time=sel)
        return np.asarray(da.values, dtype=np.float32)

    with _open(_STREAM_SOLAR) as solr, _open(_STREAM_PRECIP) as prec, \
            _open(_STREAM_TPQWL) as tpqwl:
        lat = np.asarray(tpqwl["lat"].values, dtype=np.float64)
        lon = np.asarray(tpqwl["lon"].values, dtype=np.float64)
        t_solar = _time_seconds(solr, allow_units="days since")
        t_state = _time_seconds(tpqwl, allow_units="days since")
        if sel is not None:
            t_solar = t_solar[sel]
            t_state = t_state[sel]
        out = LandForcing(
            lon=lon,
            lat=lat,
            time_s=t_state,
            time_s_solar=t_solar,
            fsds=_read(solr, _VAR_FSDS),
            flds=_read(tpqwl, _VAR_FLDS),
            prectmms=_read(prec, _VAR_PRECT),
            tbot=_read(tpqwl, _VAR_TBOT),
            psrf=_read(tpqwl, _VAR_PSRF),
            qbot=_read(tpqwl, _VAR_QBOT),
            wind=_read(tpqwl, _VAR_WIND),
            year=int(year),
        )
    return out


def synthetic_land_forcing(
    year: int = 0,
    *,
    n_time: int = 4,
    nlon: int = CRUJRA_NLON,
    nlat: int = CRUJRA_NLAT,
) -> LandForcing:
    """Deterministic CRU-JRA-shaped climatology fallback (no external data).

    Physically-reasonable, smooth fields used by smoke runs and unit tests when
    the real CLM streams are unavailable.  Mirrors the role of
    ``ocean/forcing/jra55_do.synthetic_ocean_forcing``.
    """
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    lat = np.linspace(_LAT_SOUTH_DEG, _LAT_NORTH_DEG, nlat, dtype=np.float64)
    # Solar stamped at interval start; TPQWL/Prec at interval midpoint.
    step_s = _SEC_PER_DAY * float(CRUJRA_FREQ_HOURS) / _HOURS_PER_DAY
    t_solar = np.arange(n_time, dtype=np.float64) * step_s
    t_state = t_solar + 0.5 * step_s

    lat_r = np.radians(lat)[None, :, None]
    cos_lat = np.cos(lat_r)
    shape = (n_time, nlat, nlon)

    tbot = np.broadcast_to(
        _SYN_T_POLE_K + (_SYN_T_EQUATOR_K - _SYN_T_POLE_K) * cos_lat, shape
    ).astype(np.float32)
    psrf = np.full(shape, _SYN_PSRF, dtype=np.float32)
    # q = RH * q_sat(T, p); canonical SPECIFIC-humidity saturation (legoesm.thermo,
    # no re-derivation).  qbot is consumed as specific humidity (the moist-air
    # density uses virtual_temperature(t, q)) and the real QBOT stream is
    # specific humidity, so the
    # synthetic field must be q_sat, not the mixing ratio r_sat.
    from legoesm.thermo import saturation_specific_humidity
    qsat = np.asarray(
        saturation_specific_humidity(jnp.asarray(tbot, dtype=jnp.float64),
                                     jnp.asarray(psrf, dtype=jnp.float64))
    )
    qbot = (_SYN_RH * qsat).astype(np.float32)
    fsds = np.broadcast_to(
        np.maximum(_SYN_SW_PEAK * cos_lat, 0.0), shape
    ).astype(np.float32)
    flds = np.full(shape, _SYN_LW, dtype=np.float32)
    prectmms = np.broadcast_to(
        _SYN_PRECIP * cos_lat ** 2, shape
    ).astype(np.float32)
    wind = np.full(shape, _SYN_WIND, dtype=np.float32)

    return LandForcing(
        lon=lon, lat=lat, time_s=t_state, time_s_solar=t_solar,
        fsds=fsds, flds=flds, prectmms=prectmms, tbot=tbot,
        psrf=psrf, qbot=qbot, wind=wind, year=int(year),
    )


def load_cru_jra(
    year: int,
    *,
    data_dir=None,
    prefix: str = CRUJRA_FILE_PREFIX,
    suffix: str = "",
    time_indices: Optional[Sequence[int]] = None,
    allow_synthetic: bool = True,
) -> LandForcing:
    """Load one CRU-JRA year, falling back to a synthetic climatology.

    Reads the CLM streams from ``data_dir`` when present; otherwise (and only
    if ``allow_synthetic``) returns :func:`synthetic_land_forcing`.
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        solr = _stream_path(data_dir, _STREAM_SOLAR, year, prefix, suffix)
        if solr.exists():
            return read_crujra_year(
                data_dir, year, prefix=prefix, suffix=suffix,
                time_indices=time_indices,
            )
    if not allow_synthetic:
        raise FileNotFoundError(
            f"CRU-JRA streams not found in {data_dir!r} for year {year}; "
            f"expected e.g. {prefix}.{_STREAM_SOLAR}.{year}{suffix}.nc"
        )
    n_time = 4 if time_indices is None else len(time_indices)
    return synthetic_land_forcing(year, n_time=n_time)


def build_forcing_weights(
    forcing: LandForcing,
    tgt_lat_rad: np.ndarray,
    tgt_lon_rad: np.ndarray,
    *,
    k_neighbors: int = 4,
) -> RegridWeights:
    """KD-tree / inverse-distance weights: CRU-JRA native grid -> model columns.

    Reuses the shared regridding machinery (the same path surfdata uses), so the
    coarse-target regrid is consistent with the boundary-data regrid.
    """
    return compute_latlon_to_voronoi_weights(
        np.radians(np.asarray(forcing.lat, dtype=np.float64)),
        np.radians(np.asarray(forcing.lon, dtype=np.float64)),
        np.asarray(tgt_lat_rad, dtype=np.float64),
        np.asarray(tgt_lon_rad, dtype=np.float64),
        k_neighbors=k_neighbors,
    )


def _regrid_series(field_txy: np.ndarray, weights: RegridWeights) -> np.ndarray:
    """Regrid ``(n_time, n_lat, n_lon)`` -> ``(n_time, ncol)``.

    Uses the **NaN-aware** IDW regrid: CRU-JRA is land-only (ocean = NaN, ~57 %
    of cells), so plain IDW would bleed ocean NaN into every target with a wet
    neighbour.  NaN-aware drops missing neighbours and renormalises; a target is
    NaN only when ALL its neighbours are ocean (open sea), which the land mask
    discards downstream.  ``regrid_scalar*`` flattens the LEADING spatial dims and
    preserves trailing dims, so move time to the trailing axis, regrid, move back.
    """
    src = np.moveaxis(np.asarray(field_txy), 0, -1)        # (nlat, nlon, n_time)
    out = np.asarray(regrid_scalar_nan_aware(jnp.asarray(src), weights))  # (ncol, n_time)
    return np.moveaxis(out, -1, 0)                          # (n_time, ncol)


def regrid_forcing(
    forcing: LandForcing, weights: RegridWeights
) -> LandForcingColumns:
    """Regrid every CRU-JRA channel onto the model columns."""
    return LandForcingColumns(
        time_s=forcing.time_s,
        time_s_solar=forcing.time_s_solar,
        fsds=_regrid_series(forcing.fsds, weights),
        flds=_regrid_series(forcing.flds, weights),
        prectmms=_regrid_series(forcing.prectmms, weights),
        tbot=_regrid_series(forcing.tbot, weights),
        psrf=_regrid_series(forcing.psrf, weights),
        qbot=_regrid_series(forcing.qbot, weights),
        wind=_regrid_series(forcing.wind, weights),
        year=forcing.year,
    )


def snow_fraction(t_air, *, ramp_k: float):
    """Liquid/solid precipitation split: all snow <= T_freeze, all rain at
    ``T_freeze + ramp_k``, linear between (CLM convention).

    The one land snow/rain partition, shared by the gridded CRU-JRA path and the
    eddy-covariance single-site loader so both use the identical CLM ramp.
    """
    frac = (constants.T_freeze + ramp_k - t_air) / ramp_k
    return jnp.clip(frac, 0.0, 1.0)


def _time_index(time_s: np.ndarray, model_time_s: float) -> int:
    """Nearest forcing index to ``model_time_s`` (M1: nearest, not interp)."""
    return int(np.argmin(np.abs(np.asarray(time_s) - float(model_time_s))))


def _assemble_atm_surface(
    *, sw, lw, precip, t_air, p_sfc, q_air, wind, cos_z,
    co2_ppmv: float, snow_ramp_k: float, dtype,
) -> AtmToSurface:
    """Pack the standard surface fields + derived (rho, snow split, CO2) from
    already-time-selected channels.  Shared by the nearest-time var map and the
    M2 disaggregation so the derived-field logic lives in one place.  Inputs may
    be ``(ncol,)`` (single time) or ``(n_steps, ncol)`` (stacked) — the ops
    broadcast either way.

    ``wind`` is the CRU-JRA SCALAR wind SPEED.  The land step only uses
    ``sqrt(u^2 + v^2 + U_min^2)``, so the speed is carried in ``u_lowest`` with
    ``v_lowest = 0``; this is a representational choice (the data fixes no
    direction), and momentum-stress direction is irrelevant for a land-only run.
    """
    sw = jnp.asarray(sw, dtype=dtype)
    lw = jnp.asarray(lw, dtype=dtype)
    precip = jnp.asarray(precip, dtype=dtype)
    t_air = jnp.asarray(t_air, dtype=dtype)
    p_sfc = jnp.asarray(p_sfc, dtype=dtype)
    q_air = jnp.asarray(q_air, dtype=dtype)
    wind = jnp.asarray(wind, dtype=dtype)
    cos_z = jnp.asarray(cos_z, dtype=dtype)

    # Moist-air density rho = p / (R_d * T_v) via the shared thermo helper.
    rho = moist_air_density(t_air, p_sfc, q_air)
    snow_frac = snow_fraction(t_air, ramp_k=snow_ramp_k).astype(dtype)
    one = jnp.ones_like(t_air)
    return AtmToSurface(
        sw_down=sw,
        lw_down=lw,
        precip_total=precip,
        precip_snow=precip * snow_frac,
        T_lowest=t_air,
        q_lowest=q_air,
        u_lowest=wind,                       # scalar SPEED carried in u; v=0
        v_lowest=jnp.zeros_like(wind),
        p_lowest=p_sfc,
        p_surface=p_sfc,
        rho_lowest=rho.astype(dtype),
        cos_zenith=cos_z,
        co2_ppmv=jnp.asarray(co2_ppmv, dtype=dtype) * one,
        # Flags carry the data's leading shape (the land step ignores their value)
        # so a stacked AtmToSurface is directly lax.scan-able as a forcing input.
        has_radiation=one,
        has_precipitation=one,
    )


def forcing_to_atm_surface(
    cols: LandForcingColumns,
    lat_rad,
    lon_rad,
    model_time_s: float,
    *,
    co2_ppmv: float = _DEFAULT_CO2_PPMV,
    snow_ramp_k: float = SNOW_RAIN_RAMP_K,
    dtype=jnp.float64,
) -> AtmToSurface:
    """Build :class:`AtmToSurface` from regridded forcing at ``model_time_s``.

    M1 uses **nearest-time** selection on each stream's own clock (the TPQWL/Prec
    midpoint axis and the FSDS start axis); M2 replaces this with the per-variable
    6h->dt disaggregation.  Derived fields: ``cos_zenith`` (shared solar helper),
    ``rho_lowest`` (moist-air density from p, T, q), ``precip_snow`` (T-ramp
    partition).  ``co2_ppmv`` is a constant this push.
    """
    it = _time_index(cols.time_s, model_time_s)
    isol = _time_index(cols.time_s_solar, model_time_s)

    lat = jnp.asarray(lat_rad, dtype=dtype)
    lon = jnp.asarray(lon_rad, dtype=dtype)
    doy = float(cols.time_s_solar[isol]) / _SEC_PER_DAY
    hour = (float(cols.time_s_solar[isol]) % _SEC_PER_DAY) / _SEC_PER_HOUR
    cos_z = cos_solar_zenith(lat, lon, doy, hour)

    return _assemble_atm_surface(
        sw=cols.fsds[isol], lw=cols.flds[it], precip=cols.prectmms[it],
        t_air=cols.tbot[it], p_sfc=cols.psrf[it], q_air=cols.qbot[it],
        wind=cols.wind[it], cos_z=cos_z,
        co2_ppmv=co2_ppmv, snow_ramp_k=snow_ramp_k, dtype=dtype,
    )


# ===========================================================================
# M2 — 6-hourly -> model-dt temporal disaggregation
# ===========================================================================
_COSZ_EPS = 1.0e-6               # daylight threshold for the SW night-window guard


def _linterp_series(time_src, values_tc, t_query):
    """Linear interpolation of ``(n_time, ncol)`` values onto ``t_query``
    ``(n_steps,)``, clamped at the endpoints (no extrapolation)."""
    ts = np.asarray(time_src, dtype=np.float64)
    vals = np.asarray(values_tc, dtype=np.float64)
    tq = np.asarray(t_query, dtype=np.float64)
    n = ts.shape[0]
    if n == 1:
        return np.broadcast_to(vals[0], (tq.shape[0], vals.shape[1])).copy()
    i1 = np.clip(np.searchsorted(ts, tq, side="right"), 1, n - 1)
    i0 = i1 - 1
    denom = ts[i1] - ts[i0]
    w = np.where(denom > 0.0, (tq - ts[i0]) / denom, 0.0)
    w = np.clip(w, 0.0, 1.0)                          # clamp extrapolation
    return (1.0 - w)[:, None] * vals[i0] + w[:, None] * vals[i1]


def disaggregate_forcing(
    cols: LandForcingColumns,
    lat_rad,
    lon_rad,
    model_times_s,
    *,
    co2_ppmv: float = _DEFAULT_CO2_PPMV,
    snow_ramp_k: float = SNOW_RAIN_RAMP_K,
    freq_hours: int = CRUJRA_FREQ_HOURS,
    dtype=jnp.float64,
) -> AtmToSurface:
    """Disaggregate 6-hourly forcing onto the model step times (M2).

    Per-variable rules (see the module docstring):

    * ``TBOT``/``PSRF``/``QBOT``/``WIND``/``FLDS`` (midpoint-stamped) ->
      **linear interpolation** on the midpoint clock.
    * ``PRECTmms`` (midpoint-stamped interval-mean rate) -> **constant hold**
      over the containing 6-h interval (mass-conserving).
    * ``FSDS`` (start-stamped interval-mean flux) -> **solar-zenith-weighted**
      split that **conserves the 6-h interval mean** (energy-conserving), with a
      night-window guard (uniform fallback when the interval is fully dark).

    Parameters
    ----------
    cols : LandForcingColumns
        CRU-JRA already regridded to the model columns.
    lat_rad, lon_rad : array, shape (ncol,)
        Column latitude / longitude in radians.
    model_times_s : array, shape (n_steps,)
        Model step times in seconds since the forcing-year start (e.g.
        ``t0 + dt * arange(n_steps)``).

    Returns
    -------
    AtmToSurface
        Each field has shape ``(n_steps, ncol)`` — the explicit per-step
        ``lax.scan`` input (SegmentForcing doctrine), built on the host.
    """
    lat = np.asarray(lat_rad, dtype=np.float64)
    lon = np.asarray(lon_rad, dtype=np.float64)
    tq = np.asarray(model_times_s, dtype=np.float64)
    step_s = float(freq_hours) * _SEC_PER_HOUR

    # instantaneous channels -> linear interpolation on the midpoint clock
    t_air = _linterp_series(cols.time_s, cols.tbot, tq)
    p_sfc = _linterp_series(cols.time_s, cols.psrf, tq)
    q_air = _linterp_series(cols.time_s, cols.qbot, tq)
    wind = _linterp_series(cols.time_s, cols.wind, tq)
    lw = _linterp_series(cols.time_s, cols.flds, tq)

    # precip -> constant hold over the containing 6-h interval (round to the
    # nearest midpoint stamp, which IS the interval the model time falls in)
    k_pr = np.clip(
        np.round((tq - cols.time_s[0]) / step_s).astype(int),
        0, cols.prectmms.shape[0] - 1,
    )
    precip = np.asarray(cols.prectmms)[k_pr]                  # (n_steps, ncol)

    # per-step solar zenith (also the AtmToSurface cos_zenith)
    doy = tq / _SEC_PER_DAY
    hour = (tq % _SEC_PER_DAY) / _SEC_PER_HOUR
    cosz = np.asarray(
        cos_solar_zenith(lat[None, :], lon[None, :], doy[:, None], hour[:, None])
    )                                                        # (n_steps, ncol)

    # shortwave -> zenith weighting that conserves each 6-h interval mean.
    # Interval index from the START-stamped solar clock; the per-interval mean
    # of cos(zenith) is taken over EXACTLY the model substeps that tile it, so
    # mean_substeps( fsds * cosz / mean_cosz ) == fsds.
    ts_sol = np.asarray(cols.time_s_solar, dtype=np.float64)
    n_int = ts_sol.shape[0]
    k_sw = np.clip(np.floor((tq - ts_sol[0]) / step_s).astype(int), 0, n_int - 1)
    sum_cosz = np.zeros((n_int, cosz.shape[1]))
    cnt = np.zeros(n_int)
    np.add.at(sum_cosz, k_sw, cosz)
    np.add.at(cnt, k_sw, 1.0)
    mean_cosz = (sum_cosz / np.maximum(cnt, 1.0)[:, None])[k_sw]   # (n_steps, ncol)
    fsds_step = np.asarray(cols.fsds)[k_sw]
    daylit = mean_cosz > _COSZ_EPS
    sw = np.where(
        daylit,
        fsds_step * cosz / np.where(daylit, mean_cosz, 1.0),
        fsds_step,                                           # uniform fallback (dark window)
    )

    return _assemble_atm_surface(
        sw=sw, lw=lw, precip=precip, t_air=t_air, p_sfc=p_sfc, q_air=q_air,
        wind=wind, cos_z=cosz, co2_ppmv=co2_ppmv, snow_ramp_k=snow_ramp_k,
        dtype=dtype,
    )


def stage_forcing(
    lat_rad,
    lon_rad,
    model_times_s,
    *,
    year: int,
    data_dir=None,
    prefix: str = CRUJRA_FILE_PREFIX,
    suffix: str = "",
    k_neighbors: int = 4,
    co2_ppmv: float = _DEFAULT_CO2_PPMV,
    snow_ramp_k: float = SNOW_RAIN_RAMP_K,
    freq_hours: int = CRUJRA_FREQ_HOURS,
    allow_synthetic: bool = True,
    dtype=jnp.float64,
) -> AtmToSurface:
    """High-level entry: load -> regrid -> disaggregate into scan-ready forcing.

    Loads ONLY the 6-hourly CRU-JRA slices bracketing ``model_times_s`` (cheap
    for a short run — avoids holding a full native-resolution year), regrids them
    to the model columns, and disaggregates to the model step times.  Returns the
    stacked :class:`AtmToSurface` ``(n_steps, ncol)`` used directly as the
    ``lax.scan`` forcing input.

    Synthetic fallback (no ``data_dir``) starts its clock at 0, so callers using
    synthetic forcing should set ``model_times_s`` to start at 0.
    """
    tq = np.asarray(model_times_s, dtype=np.float64)
    step_s = float(freq_hours) * _SEC_PER_HOUR
    i_lo = max(0, int(np.floor(tq.min() / step_s)) - 1)
    i_hi = min(_CRUJRA_STEPS_PER_YEAR - 1, int(np.floor(tq.max() / step_s)) + 2)
    idx = np.arange(i_lo, i_hi + 1)
    forcing = load_cru_jra(
        year, data_dir=data_dir, prefix=prefix, suffix=suffix,
        time_indices=idx, allow_synthetic=allow_synthetic,
    )
    weights = build_forcing_weights(forcing, lat_rad, lon_rad, k_neighbors=k_neighbors)
    cols = regrid_forcing(forcing, weights)
    return disaggregate_forcing(
        cols, lat_rad, lon_rad, tq, co2_ppmv=co2_ppmv,
        snow_ramp_k=snow_ramp_k, freq_hours=freq_hours, dtype=dtype,
    )


def stage_forcing_years(
    lat_rad,
    lon_rad,
    model_times_s,
    *,
    year_start: int,
    year_end: int,
    data_dir=None,
    prefix: str = CRUJRA_FILE_PREFIX,
    suffix: str = "",
    k_neighbors: int = 4,
    co2_ppmv: float = _DEFAULT_CO2_PPMV,
    snow_ramp_k: float = SNOW_RAIN_RAMP_K,
    freq_hours: int = CRUJRA_FREQ_HOURS,
    allow_synthetic: bool = True,
    dtype=jnp.float64,
) -> AtmToSurface:
    """Contiguous multi-year forcing (year_start .. year_end, inclusive).

    ``model_times_s`` is the model's continuous time axis, expressed as seconds
    since Jan 1, 00:00 of ``year_start`` (a noleap 365-day calendar — matches
    CRU-JRA / CLM datm).  Each calendar year Y in the range is loaded, regridded,
    and disaggregated INDEPENDENTLY via :func:`stage_forcing`, then the per-year
    slices are concatenated along the leading (time) axis into one scan-ready
    :class:`AtmToSurface` ``(n_steps_total, ncol)``.

    Because each year's disaggregation resets at 6-h stamps, a tiny sub-6-hourly
    discontinuity is possible at the 12/31 → 1/1 boundary for the linearly-
    interpolated channels (T, q, p, wind, LW).  Precipitation (constant-hold)
    and SW (zenith-weighted, energy-conserving) each respect their own 6-h
    intervals on either side of the boundary.  Fine for spin-up runs; for
    scientific analysis of year-boundary weather, use single-year runs.

    **Memory:** the returned pytree is fully materialised.  At 2° latlon with
    hourly steps, one year is ~1.5 GB; 5 years is ~7.5 GB.  For longer spans
    the driver should switch to per-year restart-based looping (Phase C).
    """
    if year_end < year_start:
        raise ValueError(f"year_end={year_end} < year_start={year_start}")
    tq = np.asarray(model_times_s, dtype=np.float64)
    sec_per_year = _SEC_PER_DAY * _DAYS_PER_YEAR_NOLEAP       # noleap

    per_year_atm = []
    for k, year in enumerate(range(year_start, year_end + 1)):
        t_lo = k * sec_per_year
        t_hi = (k + 1) * sec_per_year
        mask = (tq >= t_lo) & (tq < t_hi)
        if not mask.any():
            continue
        # local time within this year (each stage_forcing call expects times in
        # its own [0, sec_per_year) frame).
        tq_local = tq[mask] - t_lo
        atm_local = stage_forcing(
            lat_rad, lon_rad, tq_local,
            year=year, data_dir=data_dir, prefix=prefix, suffix=suffix,
            k_neighbors=k_neighbors, co2_ppmv=co2_ppmv,
            snow_ramp_k=snow_ramp_k, freq_hours=freq_hours,
            allow_synthetic=allow_synthetic, dtype=dtype,
        )
        per_year_atm.append(atm_local)

    if not per_year_atm:
        raise ValueError(
            f"no model times fell within years [{year_start}, {year_end}] "
            f"(model_times_s spans [{tq.min():.1f}, {tq.max():.1f}] s)"
        )
    if len(per_year_atm) == 1:
        return per_year_atm[0]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *per_year_atm)


__all__ = [
    "CRUJRA_NLON",
    "CRUJRA_NLAT",
    "CRUJRA_FREQ_HOURS",
    "CRUJRA_FILE_PREFIX",
    "LandForcing",
    "LandForcingColumns",
    "read_crujra_year",
    "synthetic_land_forcing",
    "load_cru_jra",
    "build_forcing_weights",
    "regrid_forcing",
    "forcing_to_atm_surface",
    "disaggregate_forcing",
    "stage_forcing",
    "stage_forcing_years",
]
