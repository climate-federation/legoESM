"""Slim ONEFlux FLUXNET FULLSET CSV -> per-timestep ``AtmToSurface`` forcing.

Loads a FLUXNET FP-standard FULLSET CSV (already filtered to a target
window) for one of the three MLC pilot sites (US-MMS, FI-Hyy, US-Ton)
and produces a :class:`FluxnetForcing` container of per-timestep forcing
arrays plus observed turbulent fluxes for validation.

Conventions
-----------
* FLUXNET timestamps are in **local standard time** (no DST); we convert
  to UTC with a simple ``-lon/15`` hour offset for the solar-geometry
  calculation.
* Missing values in the ONEFlux FULLSET CSV are the sentinel ``-9999``
  (and any ``_QC`` column value of ``-9999``); we convert these to
  ``NaN`` before doing any derivation math.
* All physical constants come from :mod:`legoesm.constants`; saturation
  vapour pressure / mixing ratio come from :mod:`legoesm.thermo`.  No
  Tetens/Magnus is re-derived here (per CLAUDE.md guardrails).
* The cosine-of-solar-zenith uses the Spencer (1971) polynomial from
  :mod:`legoesm.land.boundary_data.solar_geometry`.

Per-timestep ``AtmToSurface`` construction
------------------------------------------
:func:`build_atm2sfc_at` returns a **1-column** :class:`AtmToSurface`
for step index ``i`` — the runner loop calls this once per timestep.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd

from legoesm import constants
from legoesm.land.boundary_data.solar_geometry import spencer_cos_zen_array
from legoesm.thermo import saturation_vapor_pressure

if TYPE_CHECKING:  # forward ref only; keep imports lightweight at load
    from legoesm.core.coupling_fields import AtmToSurface


# ---------------------------------------------------------------------------
# Static site metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FluxnetSite:
    """Static per-site metadata (lat/lon/PFT/CLM PFT code etc.)."""

    code: str            # "US-MMS", "FI-Hyy", "US-Ton"
    lat_deg: float
    lon_deg: float
    z_ref_m: float       # tower measurement height above ground [m]
    pft_clm: int         # CLM PFT index for CLMMLCanopyConfig.pft_clm
    htop_m: float        # canopy top height [m]
    lai_peak: float      # peak LAI (for prescribed-LAI runs) [m2/m2]
    sai: float           # stem area index [m2/m2]


# CLM PFT codes (subset):
#   2  = needleleaf_evergreen_boreal_tree
#   7  = broadleaf_deciduous_temperate_tree
SITES: dict[str, FluxnetSite] = {
    "US-MMS": FluxnetSite("US-MMS", 39.3232, -86.4131, 46.0, 7, 27.0, 4.7, 0.5),  # DBF
    "FI-Hyy": FluxnetSite("FI-Hyy", 61.8474, 24.2948, 23.0, 2, 14.0, 3.0, 0.5),   # ENF
    "US-Ton": FluxnetSite("US-Ton", 38.4316, -120.9660, 23.0, 7, 7.1, 0.8, 0.5),  # SAV oak
}


# Sentinel value used by ONEFlux FLUXNET FULLSET for "missing".
_MISSING = -9999.0

# Columns required to build the forcing.  All must be finite for a step
# to count as ``valid``; a missing LW_IN_F falls back to LW_IN_JSB (see
# below).
_REQUIRED_FORCING = (
    "TA_F", "SW_IN_F", "WS_F", "PA_F", "P_F", "CO2_F_MDS", "RH",
)


# ---------------------------------------------------------------------------
# Forcing container
# ---------------------------------------------------------------------------


@dataclass
class FluxnetForcing:
    """Per-timestep FLUXNET forcing for a single tower site.

    Attributes
    ----------
    time
        Local-standard-time :class:`pandas.DatetimeIndex` at
        ``TIMESTAMP_START`` (interval-start).
    dt_s
        Timestep length in seconds (3600 for HR, 1800 for HH).
    site
        Static site metadata (:class:`FluxnetSite`).
    T_bot_K
        Air temperature [K] (from ``TA_F`` °C + ``T_freeze``).
    sw_down
        Downward shortwave [W/m²] (``SW_IN_F``).
    lw_down
        Downward longwave [W/m²] (``LW_IN_F`` if present, else
        ``LW_IN_JSB``).
    ws
        Wind speed [m/s] (``WS_F``).
    p_surface_Pa
        Surface pressure [Pa] (``PA_F`` kPa × 1000).
    precip_kg_m2_s
        Precipitation rate [kg/m²/s] = ``P_F`` [mm/dt] / ``dt_s``.
    co2_ppmv
        CO2 mixing ratio [µmol/mol] (``CO2_F_MDS``).
    q_bot
        Air specific humidity [kg/kg], derived from ``RH`` * ``q_sat``
        where possible; falls back to VPD-based derivation when RH is
        missing but VPD_F is present.
    cos_zen
        Cosine of solar zenith angle at ``TIMESTAMP_START`` (converted
        to UTC via ``-lon/15`` offset).
    LE_obs, H_obs, NETRAD_obs, ustar_obs
        Observed fluxes for downstream validation (not fed into model).
    valid
        Boolean mask: True where every required input is finite and its
        ONEFlux QC flag (if present) is 0.
    """

    time: pd.DatetimeIndex
    dt_s: float
    site: FluxnetSite

    T_bot_K: np.ndarray
    sw_down: np.ndarray
    lw_down: np.ndarray
    ws: np.ndarray
    p_surface_Pa: np.ndarray
    precip_kg_m2_s: np.ndarray
    co2_ppmv: np.ndarray
    q_bot: np.ndarray
    cos_zen: np.ndarray

    LE_obs: np.ndarray
    H_obs: np.ndarray
    NETRAD_obs: np.ndarray
    ustar_obs: np.ndarray

    valid: np.ndarray


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _to_nan(a: np.ndarray) -> np.ndarray:
    """Convert ONEFlux ``-9999`` sentinel to ``NaN`` in a float array."""
    out = np.asarray(a, dtype=np.float64).copy()
    out[out == _MISSING] = np.nan
    return out


def _get_or_nan(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return ``df[col]`` as ``float64`` with ``-9999`` -> ``NaN``.

    If ``col`` is not present in ``df``, return an all-NaN array of
    length ``len(df)``.  This is deliberate: FLUXNET FULLSET files
    differ slightly per site (e.g. some hourly files omit ``LW_IN_JSB``
    or ``USTAR``) and downstream code treats "not present" the same as
    "present but sentinel".
    """
    if col in df.columns:
        return _to_nan(df[col].to_numpy())
    return np.full(len(df), np.nan)


def _timestamps_to_index(ts_start: np.ndarray) -> pd.DatetimeIndex:
    """Parse ``YYYYMMDDHHMM`` integer column to a naive ``DatetimeIndex``.

    FLUXNET timestamps are in **local standard time** (no DST); we keep
    the index tz-naive to reflect that.  UTC conversion for the solar
    geometry is done downstream via ``-lon/15``.
    """
    # ``format="%Y%m%d%H%M"`` handles the compact ONEFlux integer form.
    return pd.to_datetime(ts_start.astype(np.int64).astype(str),
                          format="%Y%m%d%H%M")


def _derive_q_from_rh(
    T_K: np.ndarray, p_Pa: np.ndarray, rh_pct: np.ndarray,
) -> np.ndarray:
    """Specific humidity ``q`` [kg/kg] from RH [%], T [K], p [Pa].

    ``RH = e / e_sat``  =>  ``e = (RH/100) * e_sat``, then the SAME
    specific-humidity form as the VPD branch:
    ``q = epsilon * e / (p - (1 - epsilon) * e)``.  Uses the shared
    :func:`legoesm.thermo.saturation_vapor_pressure` (no re-derived
    Tetens/Magnus, CLAUDE.md guardrail).  Previously this returned a
    *mixing ratio* (``epsilon e / (p - e)`` via ``saturation_mixing_ratio``)
    labelled as specific humidity, which is inconsistent with the VPD
    branch and overestimates ``q``.
    """
    # ``saturation_vapor_pressure`` is JAX-based; we accept the small
    # host-side jax call because this is an offline loader (once per
    # site per run, not inside a hot loop).
    e_sat_Pa = np.asarray(saturation_vapor_pressure(jnp.asarray(T_K)))
    e = np.clip((rh_pct / 100.0) * e_sat_Pa, 0.0, None)
    denom = p_Pa - (1.0 - constants.epsilon) * e
    return constants.epsilon * e / denom


def _derive_q_from_vpd(
    T_K: np.ndarray, p_Pa: np.ndarray, vpd_hPa: np.ndarray,
) -> np.ndarray:
    """Specific humidity ``q`` [kg/kg] from VPD [hPa], T [K], p [Pa].

    ``VPD = e_sat - e``  =>  ``e = e_sat - VPD``.
    ``q = epsilon * e / (p - (1 - epsilon) * e)``  (mixing ratio form).
    Uses :func:`legoesm.thermo.saturation_vapor_pressure` and
    :attr:`legoesm.constants.epsilon`.
    """
    e_sat_Pa = np.asarray(saturation_vapor_pressure(jnp.asarray(T_K)))
    vpd_Pa = vpd_hPa * 100.0  # coeff-ok: exact hPa -> Pa
    e = np.clip(e_sat_Pa - vpd_Pa, 0.0, None)
    denom = p_Pa - (1.0 - constants.epsilon) * e
    return constants.epsilon * e / denom


def load_fluxnet_forcing(
    csv_path: pathlib.Path, site_code: str,
) -> FluxnetForcing:
    """Load a slim ONEFlux FLUXNET FULLSET CSV as :class:`FluxnetForcing`.

    Parameters
    ----------
    csv_path
        Path to the slim FULLSET CSV (already filtered to the target
        window; we do NOT re-filter).
    site_code
        One of the keys of :data:`SITES` (``"US-MMS"``, ``"FI-Hyy"``,
        ``"US-Ton"``).  Determines the static site metadata used for
        the solar-geometry calculation and CLM PFT code.
    """
    if site_code not in SITES:
        raise ValueError(
            f"Unknown FLUXNET site_code={site_code!r}; "
            f"expected one of {sorted(SITES)}"
        )
    site = SITES[site_code]

    df = pd.read_csv(csv_path)

    # --- timestamps + dt ---------------------------------------------------
    if "TIMESTAMP_START" not in df.columns:
        raise ValueError(f"{csv_path}: missing required column TIMESTAMP_START")
    time = _timestamps_to_index(df["TIMESTAMP_START"].to_numpy())

    if "TIMESTAMP_END" in df.columns:
        t_end = _timestamps_to_index(df["TIMESTAMP_END"].to_numpy())
        # First-row delta is representative; the slim files are strictly
        # regular (either 60 or 30 min).
        dt_s = float((t_end[0] - time[0]).total_seconds())
    else:
        # Fall back on inter-step delta.
        dt_s = float((time[1] - time[0]).total_seconds())
    if dt_s not in (1800.0, 3600.0):  # coeff-ok: HH=1800 s / HR=3600 s timesteps
        raise ValueError(
            f"{csv_path}: unexpected timestep dt_s={dt_s} "
            "(expected 1800 s HH or 3600 s HR)"
        )

    # --- raw forcing columns (sentinel -> NaN) ----------------------------
    ta_C = _get_or_nan(df, "TA_F")
    sw_in = _get_or_nan(df, "SW_IN_F")
    lw_in = _get_or_nan(df, "LW_IN_F")
    lw_in_jsb = _get_or_nan(df, "LW_IN_JSB")
    vpd_hPa = _get_or_nan(df, "VPD_F")
    pa_kPa = _get_or_nan(df, "PA_F")
    ws = _get_or_nan(df, "WS_F")
    p_mm = _get_or_nan(df, "P_F")
    co2 = _get_or_nan(df, "CO2_F_MDS")
    rh_pct = _get_or_nan(df, "RH")

    # LW fallback: use LW_IN_JSB where LW_IN_F is missing.
    lw_down = np.where(np.isnan(lw_in), lw_in_jsb, lw_in)

    # --- unit conversions --------------------------------------------------
    T_bot_K = ta_C + constants.T_freeze
    p_surface_Pa = pa_kPa * 1000.0  # coeff-ok: exact kPa -> Pa
    precip_kg_m2_s = p_mm / dt_s     # mm accumulated over dt -> kg/m2/s

    # --- specific humidity -------------------------------------------------
    # Prefer RH-based derivation (direct); fall back to VPD where RH is
    # missing but VPD_F is present.
    q_from_rh = _derive_q_from_rh(T_bot_K, p_surface_Pa, rh_pct)
    q_from_vpd = _derive_q_from_vpd(T_bot_K, p_surface_Pa, vpd_hPa)
    q_bot = np.where(np.isnan(rh_pct), q_from_vpd, q_from_rh)

    # --- solar geometry ---------------------------------------------------
    # Approximate LST -> UTC by lon/15: LST = UTC + lon/15  =>
    # UTC = LST - lon/15.  ONEFlux TIMESTAMP_START is interval-start LST.
    utc_hour_offset = site.lon_deg / 15.0  # coeff-ok: exact 360°/24h = 15°/h
    # Cast time to seconds-since-epoch (naive), add the offset, then
    # compute the UTC fractional day of year.
    lst_dt = time.to_numpy().astype("datetime64[ns]")
    utc_dt = lst_dt - np.timedelta64(
        int(round(utc_hour_offset * 3600e9)), "ns"  # coeff-ok: exact hours->ns
    )
    utc_pd = pd.DatetimeIndex(utc_dt)
    doy_frac_utc = (
        (utc_pd.dayofyear - 1).to_numpy(dtype=np.float64)
        + utc_pd.hour.to_numpy(dtype=np.float64) / 24.0  # coeff-ok: hours/day
        + utc_pd.minute.to_numpy(dtype=np.float64) / (24.0 * 60.0)  # coeff-ok: min/day
    )
    cos_zen = spencer_cos_zen_array(site.lat_deg, site.lon_deg, doy_frac_utc)

    # --- observed fluxes (validation only) --------------------------------
    LE_obs = _get_or_nan(df, "LE_F_MDS")
    H_obs = _get_or_nan(df, "H_F_MDS")
    NETRAD_obs = _get_or_nan(df, "NETRAD")
    ustar_obs = _get_or_nan(df, "USTAR")

    # --- valid mask -------------------------------------------------------
    # A step counts as valid iff every required forcing input is finite
    # AND the ONEFlux QC flag (where present) is 0.  QC == 0 means
    # "measured, not gap-filled".
    valid = np.isfinite(T_bot_K)
    valid &= np.isfinite(sw_in)
    valid &= np.isfinite(lw_down)
    valid &= np.isfinite(ws)
    valid &= np.isfinite(p_surface_Pa)
    valid &= np.isfinite(precip_kg_m2_s)
    valid &= np.isfinite(co2)
    # Humidity validity keys off the DERIVED q, not RH specifically, so the
    # documented VPD fallback (RH missing but VPD_F present) is reachable.
    valid &= np.isfinite(q_bot)

    for qc_col in ("TA_F_QC", "SW_IN_F_QC", "WS_F_QC", "PA_F_QC",
                   "P_F_QC", "CO2_F_MDS_QC"):
        if qc_col in df.columns:
            qc = df[qc_col].to_numpy()
            valid &= (qc == 0)

    return FluxnetForcing(
        time=time,
        dt_s=dt_s,
        site=site,
        T_bot_K=T_bot_K,
        sw_down=sw_in,
        lw_down=lw_down,
        ws=ws,
        p_surface_Pa=p_surface_Pa,
        precip_kg_m2_s=precip_kg_m2_s,
        co2_ppmv=co2,
        q_bot=q_bot,
        cos_zen=cos_zen,
        LE_obs=LE_obs,
        H_obs=H_obs,
        NETRAD_obs=NETRAD_obs,
        ustar_obs=ustar_obs,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Per-timestep AtmToSurface builder
# ---------------------------------------------------------------------------


def build_atm2sfc_at(forcing: FluxnetForcing, i: int) -> "AtmToSurface":
    """Build a 1-column :class:`AtmToSurface` for step index ``i``.

    The tower is a single "column" from the model's perspective, so
    every field is shape ``(1,)``.  ``rho_lowest`` is computed from the
    ideal-gas law with a virtual-temperature correction; the meridional
    wind is set to zero (the tower reports scalar wind speed only).
    """
    from legoesm.core.coupling_fields import AtmToSurface

    T = float(forcing.T_bot_K[i])
    p = float(forcing.p_surface_Pa[i])
    q = float(forcing.q_bot[i])
    # Virtual temperature: T_v = T * (1 + (1/eps - 1) * q).  Using the
    # shared constants.epsilon keeps the density consistent with the
    # rest of the model (do not hardcode 0.608).
    inv_eps_minus_one = 1.0 / constants.epsilon - 1.0
    T_v = T * (1.0 + inv_eps_minus_one * q)
    rho = p / (constants.R_d * T_v)

    return AtmToSurface(
        sw_down=jnp.full((1,), float(forcing.sw_down[i])),
        lw_down=jnp.full((1,), float(forcing.lw_down[i])),
        precip_total=jnp.full((1,), float(forcing.precip_kg_m2_s[i])),
        precip_snow=jnp.zeros((1,)),
        T_lowest=jnp.full((1,), T),
        q_lowest=jnp.full((1,), q),
        u_lowest=jnp.full((1,), float(forcing.ws[i])),
        v_lowest=jnp.zeros((1,)),
        p_lowest=jnp.full((1,), p),
        p_surface=jnp.full((1,), p),
        rho_lowest=jnp.full((1,), rho),
        cos_zenith=jnp.full((1,), float(forcing.cos_zen[i])),
        co2_ppmv=jnp.full((1,), float(forcing.co2_ppmv[i])),
        has_radiation=jnp.ones((1,)),
        has_precipitation=jnp.ones((1,)),
    )


__all__ = [
    "FluxnetSite",
    "SITES",
    "FluxnetForcing",
    "load_fluxnet_forcing",
    "build_atm2sfc_at",
]
