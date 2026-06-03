"""PLUMBER2 fluxtower meteorology reader for land-only legoESM runs.

Reads per-site PLUMBER2 NetCDF (gap-filled, half-hourly tower meteorology)
and exposes it as a clean in-memory container.  A companion builder (added in
the next increment) maps each timestep onto the model's ``AtmToSurface``
coupling fields, deriving the three quantities PLUMBER2 does not store directly
(air density, rain/snow partition, solar zenith).

This module is the *forcing* half of the AmeriFlux land runner
(see ``docs/run_fluxnet_plan.md``).  The tower-measured turbulent and carbon
fluxes (``Qh``, ``Qle``, ``NEE``, ``GPP``) are loaded as **evaluation targets
only** — they are never fed back into the model.

PLUMBER2 schema notes
---------------------
- Per-site files come as ``<SITE>_<years>_FLUXNET2015_Met.nc`` and ``..._Flux.nc``.
- Dimensions are ``(time, y, x)`` with ``y == x == 1``; variables are squeezed
  to 1-D ``(time,)`` here.
- Met variables (already in SI, no unit conversion needed for most):
  ``Tair`` [K], ``SWdown`` [W/m2], ``LWdown`` [W/m2], ``Qair`` [kg/kg specific
  humidity], ``Wind`` [m/s], ``PSurf`` [Pa], ``Precip`` [kg/m2/s], ``CO2air``
  [ppm], ``LAI`` [-].
- Site metadata (``latitude``, ``longitude``, ``elevation``,
  ``reference_height``, ``IGBP_veg_short``) appears as **NetCDF variables** in
  most vintages and as **global attributes** in others; we read both.
- **Time is local standard time** in PLUMBER2 site files (not UTC).  This is
  recorded on the container so the zenith derivation (next increment) can treat
  the timestamp as local solar clock without a longitude shift.

References
----------
- Ukkola, A. M., et al. (2022). Land surface models systematically
  overestimate the intensity, duration and magnitude of seasonal-scale
  evaporative droughts. *Environ. Res. Lett.* (PLUMBER2 protocol).
- Abramowitz, G., et al. (2024). The PLUMBER2 experiment dataset.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import compute_rho
from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
from legoesm.coupler.coupling_fields import AtmToSurface


# ===========================================================================
# Configuration: variable names + unit knobs (mirrors forcing/amip.py style)
# ===========================================================================

class Plumber2ForcingConfig(NamedTuple):
    """Variable-name and unit configuration for a PLUMBER2 site file.

    Defaults match the standard PLUMBER2 ``FLUXNET2015`` NetCDF layout.  Unit
    knobs (``*_offset`` / ``*_scale``) cover the common vintage differences
    (e.g. air temperature in degrees C, precip in mm/half-hour).

    Fields
    ------
    tair_var, swdown_var, lwdown_var, qair_var, wind_var, psurf_var,
    precip_var, co2_var, lai_var : str
        Met variable names in the ``*_Met.nc`` file.  ``co2_var`` and
        ``lai_var`` are optional — missing ones fall back to ``co2_fallback_ppmv``
        / no-LAI.
    tair_offset : float
        Additive offset applied to ``Tair`` [K] (set to ``constants.T_freeze``
        if the file stores degrees C).
    precip_scale : float
        Multiplicative scale applied to ``Precip`` to reach [kg/m2/s]
        (1.0 if already a rate; e.g. ``1/1800`` if mm per 30-min step).
    co2_fallback_ppmv : float
        CO2 used when ``co2_var`` is absent from the file.
    qh_var, qle_var, nee_var, gpp_var, qg_var : str
        Evaluation-target variable names in the ``*_Flux.nc`` file.
    time_is_local : bool
        Whether the file's time axis is local standard time (PLUMBER2: True).
    """
    tair_var: str = "Tair"
    swdown_var: str = "SWdown"
    lwdown_var: str = "LWdown"
    qair_var: str = "Qair"
    wind_var: str = "Wind"
    psurf_var: str = "PSurf"
    precip_var: str = "Precip"
    co2_var: str = "CO2air"
    lai_var: str = "LAI"

    tair_offset: float = 0.0
    precip_scale: float = 1.0
    co2_fallback_ppmv: float = 412.0

    qh_var: str = "Qh"
    qle_var: str = "Qle"
    nee_var: str = "NEE"
    gpp_var: str = "GPP"
    qg_var: str = "Qg"

    time_is_local: bool = True


# ===========================================================================
# Site metadata
# ===========================================================================

class Plumber2Site(NamedTuple):
    """Static site metadata read from the PLUMBER2 file.

    All angles in degrees; heights in metres.
    """
    site_id: str
    lat_deg: float
    lon_deg: float
    elevation_m: float
    reference_height_m: float
    igbp_veg: str


# ===========================================================================
# Forcing container
# ===========================================================================

class Plumber2Forcing(NamedTuple):
    """Full-record tower meteorology + evaluation targets for one site.

    All series are float64 ``numpy`` arrays of shape ``(n_time,)`` on the
    file's native (half-hourly) time axis.  ``eval_targets`` holds whichever
    flux variables were present (``Qh``/``Qle``/``NEE``/``GPP``/``Qg``); it is
    empty when no flux file is supplied.
    """
    site: Plumber2Site

    # --- Time axis ---
    time: np.ndarray          # datetime64[ns], shape (n_time,)
    seconds: np.ndarray       # seconds since record start [s]
    doy: np.ndarray           # fractional day-of-year [0, 366)
    hour: np.ndarray          # hour-of-day [0, 24)
    dt_s: float               # inferred timestep [s] (median of diffs)
    time_is_local: bool       # True if `time` is local standard time

    # --- Raw met series (SI units after config offset/scale) ---
    tair: np.ndarray          # [K]
    swdown: np.ndarray        # [W/m2]
    lwdown: np.ndarray        # [W/m2]
    qair: np.ndarray          # [kg/kg] specific humidity
    wind: np.ndarray          # [m/s] scalar wind speed
    psurf: np.ndarray         # [Pa]
    precip: np.ndarray        # [kg/m2/s] total precipitation rate
    co2: np.ndarray           # [ppmv]
    lai: np.ndarray | None    # [-] leaf area index, or None if absent

    # --- Evaluation targets (scoring only; never forced) ---
    eval_targets: dict        # {name: np.ndarray [W/m2 or umol/m2/s]}

    @property
    def n_time(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_years(self) -> int:
        """Number of distinct calendar years spanned (for spin-up cycling)."""
        years = self.time.astype("datetime64[Y]").astype(int) + 1970
        return int(years.max() - years.min() + 1)


# ===========================================================================
# Loader
# ===========================================================================

def _get_scalar(ds, name: str, default: float | None = None) -> float:
    """Read a scalar site attribute, trying data variables then global attrs.

    PLUMBER2 stores ``latitude``/``longitude``/``elevation``/
    ``reference_height`` as (y, x) variables in most vintages and as global
    attributes in others.
    """
    if name in ds.variables:
        return float(np.asarray(ds[name].values).ravel()[0])
    if name in ds.attrs:
        return float(ds.attrs[name])
    if default is not None:
        return float(default)
    raise KeyError(
        f"PLUMBER2 file has no variable or attribute {name!r}; "
        f"available variables: {sorted(ds.variables)}"
    )


def _get_string(ds, name: str, default: str = "") -> str:
    """Read a string site attribute (variable or global attr)."""
    if name in ds.variables:
        val = np.asarray(ds[name].values).ravel()[0]
        return val.decode() if isinstance(val, bytes) else str(val)
    if name in ds.attrs:
        return str(ds.attrs[name])
    return default


def _series(ds, name: str) -> np.ndarray:
    """Extract a 1-D float64 met series, squeezing the (y, x) singleton dims."""
    if name not in ds.variables:
        raise KeyError(
            f"PLUMBER2 file is missing required variable {name!r}; "
            f"available variables: {sorted(ds.variables)}"
        )
    arr = np.asarray(ds[name].values, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)[:, 0]  # (time, y*x) -> (time,)


def load_plumber2(
    met_path: str,
    flux_path: str | None = None,
    config: Plumber2ForcingConfig = Plumber2ForcingConfig(),
) -> Plumber2Forcing:
    """Load a PLUMBER2 site's meteorology (and optional flux targets).

    Parameters
    ----------
    met_path : str
        Path to the ``*_Met.nc`` file.
    flux_path : str or None
        Path to the ``*_Flux.nc`` file.  When given, the flux variables named
        in ``config`` are loaded as evaluation targets.
    config : Plumber2ForcingConfig
        Variable names and unit knobs.

    Returns
    -------
    Plumber2Forcing
        Full-record container; raw met series only (no derived fields yet).
    """
    import xarray as xr

    ds = xr.open_dataset(met_path, decode_times=True)
    try:
        # --- Site metadata ---
        site = Plumber2Site(
            site_id=_get_string(ds, "site_id") or _get_string(ds, "name"),
            lat_deg=_get_scalar(ds, "latitude"),
            lon_deg=_get_scalar(ds, "longitude"),
            elevation_m=_get_scalar(ds, "elevation", default=0.0),
            reference_height_m=_get_scalar(ds, "reference_height", default=2.0),
            igbp_veg=_get_string(ds, "IGBP_veg_short")
            or _get_string(ds, "IGBP_veg_long"),
        )

        # --- Time axis ---
        time = np.asarray(ds["time"].values)  # datetime64[ns]
        seconds = (time - time[0]) / np.timedelta64(1, "s")
        seconds = seconds.astype(np.float64)
        # Fractional day-of-year and hour-of-day from the decoded calendar.
        year_start = time.astype("datetime64[Y]")
        doy = ((time - year_start) / np.timedelta64(1, "D")).astype(np.float64)
        day_start = time.astype("datetime64[D]")
        hour = ((time - day_start) / np.timedelta64(1, "h")).astype(np.float64)
        diffs = np.diff(seconds)
        dt_s = float(np.median(diffs)) if diffs.size else 0.0

        # --- Met series (apply config offset/scale) ---
        tair = _series(ds, config.tair_var) + config.tair_offset
        swdown = _series(ds, config.swdown_var)
        lwdown = _series(ds, config.lwdown_var)
        qair = _series(ds, config.qair_var)
        wind = _series(ds, config.wind_var)
        psurf = _series(ds, config.psurf_var)
        precip = _series(ds, config.precip_var) * config.precip_scale

        if config.co2_var in ds.variables:
            co2 = _series(ds, config.co2_var)
        else:
            co2 = np.full_like(tair, config.co2_fallback_ppmv)

        lai = _series(ds, config.lai_var) if config.lai_var in ds.variables else None
    finally:
        ds.close()

    # --- Evaluation targets ---
    eval_targets: dict = {}
    if flux_path is not None:
        fds = xr.open_dataset(flux_path, decode_times=True)
        try:
            for key, var in [
                ("Qh", config.qh_var), ("Qle", config.qle_var),
                ("NEE", config.nee_var), ("GPP", config.gpp_var),
                ("Qg", config.qg_var),
            ]:
                if var in fds.variables:
                    eval_targets[key] = _series(fds, var)
        finally:
            fds.close()

    return Plumber2Forcing(
        site=site,
        time=time, seconds=seconds, doy=doy, hour=hour, dt_s=dt_s,
        time_is_local=config.time_is_local,
        tair=tair, swdown=swdown, lwdown=lwdown, qair=qair, wind=wind,
        psurf=psurf, precip=precip, co2=co2, lai=lai,
        eval_targets=eval_targets,
    )


# ===========================================================================
# Source-agnostic derivations (plain arrays in → jnp arrays out)
#
# These three functions are the *reusable physics* that turns forcing
# primitives into the quantities PLUMBER2 does not store.  They take plain
# arrays — not Plumber2Forcing — so an ERA5 or coupler-history reader can
# call them through a thin adapter without re-deriving anything.  When a
# second source lands they should move to ``forcing/`` (see
# docs/run_fluxnet_plan.md §10).
# ===========================================================================

def partition_precip_snow(precip, tair, t_freeze: float = constants.T_freeze):
    """Split total precipitation into snowfall by a temperature threshold.

    Snowfall is all-or-nothing per timestep (``precip`` below ``t_freeze``,
    else 0), so it is mass-conserving by construction: ``precip_snow ≤ precip``.

    Parameters
    ----------
    precip : array
        Total precipitation rate [kg/m2/s].
    tair : array
        Air temperature [K].
    t_freeze : float
        Rain/snow threshold [K] (default ``constants.T_freeze``).
    """
    return jnp.where(jnp.asarray(tair) < t_freeze, jnp.asarray(precip), 0.0)


def air_density(tair, psurf, qair):
    """Moist air density [kg/m3] via the shared ideal-gas helper.

    Delegates to ``_shared.compute_rho`` (virtual-temperature form); no
    independent ideal-gas law is implemented here.

    Parameters
    ----------
    tair : array
        Air temperature [K].
    psurf : array
        Air pressure [Pa].
    qair : array
        Specific humidity [kg/kg] (used as ``q_v`` for the virtual-T form).
    """
    return compute_rho(jnp.asarray(tair), jnp.asarray(psurf), jnp.asarray(qair))


def solar_zenith(lat_deg, lon_deg, doy, hour, time_is_local: bool = True):
    """Cosine of the solar zenith angle, clamped to ``[0, 1]`` (night → 0).

    Reuses ``radiation.solar.cos_zenith_angle``.  For **local** standard time
    (PLUMBER2) the longitude term is dropped (the local clock already encodes
    it), so we pass ``lon = 0`` and the local hour.  For UTC sources pass the
    real longitude.  ``doy`` is shifted by +1 to match the function's 1-based
    day-of-year convention.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Site latitude / longitude [degrees].
    doy : array
        Fractional day-of-year, 0-based [0, 366).
    hour : array
        Hour-of-day [0, 24); local standard time when ``time_is_local``.
    time_is_local : bool
        Whether ``hour`` is local standard time (True) or UTC (False).
    """
    lat_rad = jnp.asarray(lat_deg) * constants.DEG_TO_RAD
    lon_rad = 0.0 if time_is_local else jnp.asarray(lon_deg) * constants.DEG_TO_RAD
    cos_z = cos_zenith_angle(lat_rad, lon_rad, jnp.asarray(doy) + 1.0, jnp.asarray(hour))
    return jnp.maximum(cos_z, 0.0)


# ===========================================================================
# AtmToSurface builder (source-agnostic) + PLUMBER2 adapter
# ===========================================================================

def build_atm_to_surface_series(
    *,
    tair, qair, swdown, lwdown, precip, wind, psurf, co2,
    lat_deg: float, lon_deg: float, doy, hour,
    time_is_local: bool = True,
    t_freeze: float = constants.T_freeze,
) -> AtmToSurface:
    """Pack forcing primitives into an ``AtmToSurface`` time series.

    Every field has shape ``(n_time, 1)`` — a leading time axis over a single
    column (``ncol = 1``).  Slice timestep ``i`` with :func:`forcing_at_index`
    to get the ``(1,)``-shaped fields the land step expects, or feed the whole
    series to ``lax.scan`` (which slices the leading axis).

    Source-agnostic: takes plain arrays, derives ``precip_snow``,
    ``rho_lowest`` and ``cos_zenith`` internally, and assumes scalar wind
    (``v_lowest = 0``) and fully-valid radiation/precip flags (PLUMBER2 is
    gap-filled).
    """
    tair = jnp.asarray(tair)
    n = tair.shape[0]
    dtype = tair.dtype

    precip_snow = partition_precip_snow(precip, tair, t_freeze)
    rho = air_density(tair, psurf, qair)
    cos_z = solar_zenith(lat_deg, lon_deg, doy, hour, time_is_local)

    def col(a):
        return jnp.asarray(a, dtype=dtype).reshape(-1, 1)

    return AtmToSurface(
        sw_down=col(swdown),
        lw_down=col(lwdown),
        precip_total=col(precip),
        precip_snow=col(precip_snow),
        T_lowest=col(tair),
        q_lowest=col(qair),
        u_lowest=col(wind),
        v_lowest=jnp.zeros((n, 1), dtype=dtype),
        p_lowest=col(psurf),
        p_surface=col(psurf),
        rho_lowest=col(rho),
        cos_zenith=col(cos_z),
        co2_ppmv=col(co2),
        has_radiation=jnp.ones((n, 1), dtype=dtype),
        has_precipitation=jnp.ones((n, 1), dtype=dtype),
    )


def plumber2_to_atm_series(data: Plumber2Forcing) -> AtmToSurface:
    """Adapt a loaded :class:`Plumber2Forcing` record to an ``AtmToSurface``
    series — the thin PLUMBER2-specific shim over the source-agnostic builder.
    """
    return build_atm_to_surface_series(
        tair=data.tair, qair=data.qair, swdown=data.swdown, lwdown=data.lwdown,
        precip=data.precip, wind=data.wind, psurf=data.psurf, co2=data.co2,
        lat_deg=data.site.lat_deg, lon_deg=data.site.lon_deg,
        doy=data.doy, hour=data.hour, time_is_local=data.time_is_local,
    )


def forcing_at_index(series: AtmToSurface, i) -> AtmToSurface:
    """Slice timestep ``i`` from an ``(n_time, 1)`` series → ``(1,)`` fields.

    For spin-up cycling, call with ``i = step % n_time`` (the data is loaded
    once and re-indexed; no copies).
    """
    return jax.tree_util.tree_map(lambda x: x[i], series)


__all__ = [
    "Plumber2ForcingConfig",
    "Plumber2Site",
    "Plumber2Forcing",
    "load_plumber2",
    "partition_precip_snow",
    "air_density",
    "solar_zenith",
    "build_atm_to_surface_series",
    "plumber2_to_atm_series",
    "forcing_at_index",
]
