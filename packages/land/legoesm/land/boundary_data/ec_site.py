"""Read a DifferBESS per-site driver NetCDF into legoESM offline-land inputs.

The DifferBESS project ships per-site driver NetCDFs
(``data/sitelevel/nc/<SITE>_driver_v2.nc``) that bundle FLUXNET2015 half-hourly
meteorology + MODIS LAI/albedo + BESSRad PAR + site parameters + observed tower
fluxes.  This module is the legoESM-side analogue of DifferBESS
``util/io.py:load_driver``: it maps one v2 driver onto

  * a per-timestep :class:`~legoesm.core.coupling_fields.AtmToSurface` forcing
    time series (shape ``(n_time, 1)`` per field),
  * a per-timestep :class:`~legoesm.land.canopy.config.CanopyLandParams`,
  * prescribed soil skin temperature + root-zone moisture stress (for the
    diagnostic canopy run),
  * a per-timestep ``valid`` mask (gap handling), and
  * the observed tower fluxes (GPP/LE/H/NEE) for comparison.

Faithfulness choices (match production DifferBESS):
  * **LAI** = ``ds.LAI`` clipped to ``[0, 7]`` (NOT the MODIS_LAI_* / HiQ-LAI
    variants); ``kn = a ln(mean LAI) + b`` and ``FNonVeg = exp(-0.5 CI LAI)``
    follow from that same LAI.
  * Specific humidity from VPD uses :func:`legoesm.thermo` AERK saturation +
    :mod:`legoesm.constants` (NO re-derived saturation curve / constants).

Robustness: FLUXNET meteorology has gaps.  Each driver variable is
physical-range-masked (sentinel/out-of-range -> NaN, like DifferBESS
``_mask_out_of_range``), the genuine NaNs are time-interpolated to finite values
so the (NaN-intolerant) JAX canopy can ingest them, and the ``valid`` mask is the
AND of the pre-fill observation masks of EVERY required driving input — so a step
is "valid" only if all its forcing/canopy inputs were genuinely observed (an
entirely-missing required series makes every step invalid).

Host-side only (xarray + numpy IO); returns ``jnp`` arrays for the model inputs.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xarray as xr
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure_aerk
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import (
    CanopyLandParams,
    PFT_AERO_PARAMS,
    PFT_LEAF_WIDTH,
    lookup_vcmax25,
)

# --- DifferBESS empirical fits (provenance-tagged; util/io.py) ----------------
_KN_LAI_SLOPE = -0.62        # k_n = slope*ln(mean LAI) + intercept  (DifferBESS io.py:317)
_KN_LAI_INTERCEPT = 0.98
# --- Canopy stomatal / emissivity defaults (Ball-Berry; CLM/DifferBESS).
#     Mirror legoesm.land.boundary_data._internals; defined locally to avoid a
#     private cross-module import (test_no_private_cross_imports).
M_C3 = 9.0                  # Ball-Berry slope, C3 [-]
M_C4 = 4.0                  # Ball-Berry slope, C4 [-]
B0_C3 = 0.01               # Ball-Berry intercept, C3 [mol m-2 s-1]
B0_C4 = 0.04               # Ball-Berry intercept, C4 [mol m-2 s-1]
ALF_DEFAULT = 0.3          # quantum yield [mol CO2 / mol photon]
EMISS_VEG = 0.97           # vegetated-surface emissivity [-]
# --- Diagnostic moisture-stress proxy defaults (texture-independent) -----------
_THETA_WP_DEFAULT = 0.10     # wilting-point volumetric water content [m3/m3]
_THETA_FC_DEFAULT = 0.35     # field-capacity volumetric water content [m3/m3]
_LAI_MAX = 7.0               # DifferBESS LAI cap [m2/m2]
_LAI_MEAN_FLOOR = 1.0e-3     # guard for log(mean LAI) in the kn fit
_D_LEAF_DEFAULT = 0.025      # characteristic leaf width fallback [m] (Schuepp 1993)
# Vcmax provenance rules (DifferBESS io.py:426-432): cropland always uses the
# PFT/climate lookup; other PFTs use the driver optimality value only when its
# pre-fill coverage exceeds this fraction, else the lookup.
_CROPLAND_IGBP = (10, 12)
_VCMAX_COVERAGE_MIN = 0.5
# --- Exact unit conversions ---------------------------------------------------
_KPA_TO_PA = 1000.0
_HPA_TO_PA = 100.0
_SECONDS_PER_DAY = 86400.0
# Forest-floor litter persistence timescale [days].  The litter cover in the
# soil-evaporation resistance is driven by a running maximum of LAI that relaxes
# back over this timescale, so a deciduous forest keeps its floor litter through
# the leaf-off season (litter decomposes over months, not with the live canopy).
_LITTER_TAU_DAYS = 180.0


def _persistent_litter_lai(lai, dt_s):
    """Running maximum of LAI with slow exponential relaxation (litter persistence).

    Rises immediately to the live LAI and decays back over ``_LITTER_TAU_DAYS``,
    giving the structural LAI that drives the forest-floor litter cover.  Pure
    NumPy (reader-side, not traced); the recurrence is inherently sequential.
    """
    decay = float(np.exp(-(dt_s / _SECONDS_PER_DAY) / _LITTER_TAU_DAYS))
    out = np.empty(lai.shape[0], dtype=float)
    acc = float(lai[0]) if np.isfinite(lai[0]) else 0.0
    for i in range(lai.shape[0]):
        li = float(lai[i]) if np.isfinite(lai[i]) else acc
        acc = max(li, acc * decay)
        out[i] = acc
    return out
# --- Time-axis validation ------------------------------------------------------
_DT_UNIFORM_TOL_S = 1.0      # max allowed spread in the timestep [s] (uniform grid)
_DT_FALLBACK_S = 1800.0      # safe positive dt for malformed-time arrays (gated invalid)

# --- Physical-range guards [driver units] (sentinel/fill -> NaN) --------------
# Mirrors DifferBESS util/io.py:_mask_out_of_range; generous bounds that reject
# obvious fill values, not real extremes.  SZA spans the full 0-180 deg domain so
# nighttime geometry (>90) is kept (cos clipped to 0), never NaN-interpolated.
_RANGE: dict[str, tuple[float, float]] = {
    "SW_IN": (-20.0, 1500.0), "LW_IN": (40.0, 750.0), "TA": (-70.0, 60.0),
    "VPD": (0.0, 250.0), "PA": (30.0, 110.0), "WS": (0.0, 75.0),
    "P": (0.0, 500.0), "CO2": (200.0, 2000.0), "SWC": (0.0, 75.0),
    "TS": (-70.0, 80.0), "SZA": (0.0, 180.0), "CI": (0.05, 1.0),
    "LAI": (0.0, _LAI_MAX), "T_GROWTH": (-50.0, 50.0), "EMISSIVITY": (0.85, 1.0),
    # Vcmax lower bound 0 (not 1): production counts every non-NaN Vcmax, and
    # real drivers contain valid zero samples (e.g. CA-Gro dormant season).
    "Vcmax25_C3Leaf": (0.0, 250.0),
    "Albedo_BSA_vis": (0.0, 1.0), "Albedo_WSA_vis": (0.0, 1.0),
    "Albedo_BSA_nir": (0.0, 1.0), "Albedo_WSA_nir": (0.0, 1.0),
    "BESS_PAR_DIFF_PAR_RATIO": (0.0, 1.0),
}
# SW_IN nighttime sensor noise in [-20, 0) is clamped to 0 but kept observed;
# below -20 is masked NaN (DifferBESS util/io.py).
_SW_NOISE_FLOOR = -20.0


def is_observed(name: str, values) -> np.ndarray:
    """Boolean mask of GENUINELY-OBSERVED samples of a driver field: finite AND
    within the same physical range the reader uses to mask fills/sentinels.

    Public so producers (e.g. ``scripts/data/build_ec_gapfree_driver.py``) define
    fill provenance the SAME way the reader decides validity — a finite but
    out-of-range value (sentinel / rejected extreme) is NOT an observation.
    """
    a = np.asarray(values, dtype=np.float64)
    lo, hi = _RANGE.get(name, (-np.inf, np.inf))
    return np.isfinite(a) & (a >= lo) & (a <= hi)

# Observed tower-flux physical ranges (FLUXNET -9999 / fill sentinels -> NaN, so
# the comparison ignores them).  Generous bounds that reject sentinels only.
_OBS_RANGE: dict[str, tuple[float, float]] = {
    "GPP_DT": (-50.0, 200.0),    # umolCO2/m2/s
    "NEE": (-100.0, 100.0),      # umolCO2/m2/s
    "ET": (-50.0, 100.0),        # mm/day
    "H": (-1000.0, 1500.0),      # W/m2
    "USTAR": (0.0, 10.0),        # friction velocity [m/s]
}

# Required per-timestep driving inputs (forcing + canopy params); a step is
# `valid` only if all of these were genuinely observed (pre gap-fill).
_REQUIRED = (
    "SW_IN", "LW_IN", "TA", "VPD", "PA", "WS", "P", "CO2", "SZA",
    "SWC", "TS", "LAI", "CI", "T_GROWTH",
    "Albedo_BSA_vis", "Albedo_WSA_vis", "Albedo_BSA_nir", "Albedo_WSA_nir",
    "BESS_PAR_DIFF_PAR_RATIO",   # drives the BSA/WSA albedo blend
)
# Vcmax25_C3Leaf and EMISSIVITY are NOT unconditionally required: production
# DifferBESS gives them deliberate fallbacks (Vcmax -> PFT/climate lookup for
# cropland or <=50% coverage; emissivity -> site-mean then 0.97), so a missing
# sample still receives the intended production forcing and stays valid.  Vcmax
# gates `valid` only when the driver series IS used (see read_ec_site_driver).

# --- IGBP code (DifferBESS util/constants.py, 0-indexed) -> legoESM PFT name.
#     Category lookup (CSH/OSH -> SHR; WSA/SAV -> SAV; CRO/CNM -> CRO).
_IGBP_TO_PFT: dict[int, str] = {
    0: "ENF", 1: "EBF", 2: "DNF", 3: "DBF", 4: "MF",
    5: "SHR", 6: "SHR", 7: "SAV", 8: "SAV", 9: "GRA", 10: "CRO", 12: "CRO",
}
# DifferBESS util/io.py:_climate_to_clm (Koppen group 0-4 -> CLM climate zone).
_CLIMATE_TO_NAME: dict[int, str] = {0: "tropical", 1: "temperate", 2: "temperate",
                                    3: "boreal", 4: "boreal"}


class ECSiteDriver(NamedTuple):
    """Offline-land inputs read from one DifferBESS v2 site driver NetCDF."""
    forcing: AtmToSurface          # each field (n_time, 1) — finite (gap-filled)
    canopy_params: CanopyLandParams  # each field (n_time, 1) — finite
    T_soil_top: jnp.ndarray        # (n_time, 1) prescribed soil skin T [K] (diagnostic)
    w_frac_rz: jnp.ndarray         # (n_time, 1) root-zone moisture stress beta [0, 1]
    theta_soil: jnp.ndarray        # (n_time, 1) observed volumetric soil moisture [m3/m3]
                                   #   (SWC/100; for PROGNOSTIC soil initialisation)
    wind_speed: jnp.ndarray        # (n_time, 1) [m/s]
    valid: np.ndarray              # (n_time,) bool — all required inputs observed
    obs: dict                      # {'gpp_umol','le_wm2','h_wm2','nee_umol'} (n_time,) w/ NaN
    lat_rad: float
    lon_deg: float
    doy: np.ndarray                # (n_time,) day-of-year [1, 366]
    dt_s: float                    # timestep [s]
    igbp: int
    climate: str
    pft: str
    site_id: str
    # (n_time,) 1 where any ATMOSPHERIC forcing field (TA/VPD/SW_IN/LW_IN/PA) was
    # gap-filled by the gap-free producer (build_ec_gapfree_driver.py); None for a
    # standard driver_v2 whose met is already measured-only.  Per-step forcing in
    # both modes -> lets the consumer score skill on genuinely-observed forcing.
    met_filled: np.ndarray | None = None
    # (n_time,) 1 where a SOIL field (TS/SWC) was gap-filled.  Prescribed per-step
    # in diagnostic mode; only the initial state in prognostic mode.
    soil_filled: np.ndarray | None = None


def _masked(ds: xr.Dataset, name: str) -> np.ndarray:
    """Read a 1-D time variable, replacing out-of-physical-range with NaN."""
    a = np.asarray(ds[name].values, dtype=np.float64).ravel()
    lo, hi = _RANGE.get(name, (-np.inf, np.inf))
    return np.where((a >= lo) & (a <= hi), a, np.nan)


def _gapfill(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time-interpolate NaNs to finite values; return (filled, was_observed).

    Linear interpolation over the finite samples (flat extrapolation at the
    ends).  ``was_observed`` is the pre-fill finite mask, so the caller can flag
    gap-filled timesteps.  An all-NaN series fills with 0.0 and is never valid.
    """
    n = a.shape[0]
    good = np.isfinite(a)
    if not good.any():
        return np.zeros(n, dtype=np.float64), good
    idx = np.arange(n)
    return np.interp(idx, idx[good], a[good]), good


def _obs_masked(ds: xr.Dataset, name: str, n: int) -> np.ndarray:
    """Read an observed flux, mapping fill sentinels / out-of-range to NaN.

    Observations keep NaN (no gap-fill) so the comparison ignores missing tower
    data.  Absent variable -> all-NaN.
    """
    if name not in ds:
        return np.full(n, np.nan)
    a = np.asarray(ds[name].values, dtype=np.float64).ravel()
    lo, hi = _OBS_RANGE[name]
    return np.where((a >= lo) & (a <= hi), a, np.nan)


def _scalar(ds: xr.Dataset, name: str) -> float:
    """Site-scalar attribute (broadcast over time): first finite value.

    Returns NaN if the variable is ABSENT from the dataset (schema omission) so
    that ``site_ok`` can invalidate the site rather than crashing.
    """
    if name not in ds:
        return float("nan")
    a = np.asarray(ds[name].values, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    return float(finite[0]) if finite.size else float("nan")


def _specific_humidity_from_vpd(T_K: np.ndarray, vpd_pa: np.ndarray,
                                p_pa: np.ndarray) -> np.ndarray:
    """Specific humidity [kg/kg] from VPD, via legoESM AERK saturation.

    ``es`` from :func:`saturation_vapor_pressure_aerk`; actual vapour pressure
    ``e = clip(es - VPD, 0, es)``; ``q = eps e / (p - (1-eps) e)`` with
    ``eps = constants.epsilon`` (matches DifferBESS ``Meteorology``).
    """
    es = np.asarray(saturation_vapor_pressure_aerk(jnp.asarray(T_K)))  # Pa
    e = np.clip(es - vpd_pa, 0.0, es)
    eps = constants.epsilon
    return eps * e / np.maximum(p_pa - (1.0 - eps) * e, 1.0)


def read_ec_site_driver(
    path: str,
    *,
    theta_wp: float = _THETA_WP_DEFAULT,
    theta_fc: float = _THETA_FC_DEFAULT,
) -> ECSiteDriver:
    """Read a DifferBESS v2 site driver NetCDF into :class:`ECSiteDriver`.

    Parameters
    ----------
    path : str
        Path to ``<SITE>_driver_v2.nc``.
    theta_wp, theta_fc : float
        Wilting-point / field-capacity volumetric water content [m3/m3] used to
        map the prescribed ``SWC`` onto a root-zone moisture-stress factor
        ``w_frac_rz = clip((theta - wp)/(fc - wp), 0, 1)`` for the DIAGNOSTIC
        run (first-pass proxy; the prognostic Richards path derives the stress
        from the soil state instead).
    """
    ds = xr.open_dataset(path)
    n = ds.sizes["time"]
    site_id = str(ds.attrs.get("site", ds.attrs.get("SITE", path.split("/")[-1])))

    # --- timestep + day-of-year from the time coordinate ---
    t = ds["time"].values
    if n < 2:
        raise ValueError(f"EC-site driver {path!r} has < 2 timesteps (n={n})")
    dt_arr = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
    dt_pos = dt_arr[np.isfinite(dt_arr) & (dt_arr > 0.0)]
    # Uniform, strictly-increasing time required; a duplicate/reversed/irregular
    # axis (which would make dt_s zero/negative -> NaN/Inf precip, or apply the
    # wrong mm/step->kg/m2/s conversion) invalidates every step.  A safe positive
    # dt keeps the arrays finite meanwhile.
    time_ok = bool(np.all(np.isfinite(dt_arr)) and np.all(dt_arr > 0.0)
                   and float(np.ptp(dt_arr)) <= _DT_UNIFORM_TOL_S)
    dt_s = float(np.median(dt_pos)) if dt_pos.size else _DT_FALLBACK_S
    doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(int) + 1.0

    # --- read + range-mask + gap-fill every driving series; keep pre-fill masks
    fil, obs = {}, {}
    for v in _RANGE:
        if v in ds:
            fil[v], obs[v] = _gapfill(_masked(ds, v))
        else:                       # absent optional series -> NaN -> invalid
            fil[v], obs[v] = np.zeros(n), np.zeros(n, dtype=bool)

    # SW_IN: clamp nighttime sensor noise in [_SW_NOISE_FLOOR, 0) to 0 (kept
    # observed); below the floor was already masked NaN by _RANGE (DifferBESS).
    fil["SW_IN"] = np.maximum(fil["SW_IN"], 0.0)

    # --- LAI: production DifferBESS masks ds.LAI outside [0, _LAI_MAX] -> missing
    #     (_mask_out_of_range, io.py:315), NOT clipped.  It is range-masked in the
    #     loop above, so LAI > 7 / sentinels are NaN -> obs['LAI'] False (those
    #     steps are invalid via _REQUIRED); gap-fill keeps the model finite.
    LAI = fil["LAI"]
    lai_obs = np.where(obs["LAI"], LAI, np.nan)                 # observed-only
    lai_obs_mean = np.nanmean(lai_obs) if obs["LAI"].any() else 1.0
    kn_site = _KN_LAI_SLOPE * np.log(max(lai_obs_mean, _LAI_MEAN_FLOOR)) + _KN_LAI_INTERCEPT

    # --- a step is valid only if ALL required inputs were genuinely observed ---
    valid = np.logical_and.reduce([obs[v] for v in _REQUIRED])

    # --- AtmToSurface forcing (mapping table; units -> SI) ---
    T_K = fil["TA"] + constants.T_freeze
    p_pa = fil["PA"] * _KPA_TO_PA                   # kPa -> Pa
    vpd_pa = fil["VPD"] * _HPA_TO_PA                # hPa -> Pa
    q = _specific_humidity_from_vpd(T_K, vpd_pa, p_pa)
    T_v = T_K * (1.0 + (1.0 / constants.epsilon - 1.0) * q)   # virtual temperature
    rho = p_pa / (constants.R_d * T_v)
    cos_zen = np.clip(np.cos(np.deg2rad(fil["SZA"])), 0.0, 1.0)   # night SZA>90 -> 0
    precip = fil["P"] / dt_s                       # mm/step (=kg/m2/step) -> kg/m2/s
    snow_frac = np.clip((constants.T_freeze + 2.0 - T_K) / 4.0, 0.0, 1.0)

    def col(a):  # (n,) -> (n, 1) jnp
        return jnp.asarray(a, dtype=jnp.float64)[:, None]

    forcing = AtmToSurface(
        sw_down=col(fil["SW_IN"]), lw_down=col(fil["LW_IN"]),
        precip_total=col(precip), precip_snow=col(precip * snow_frac),
        T_lowest=col(T_K), q_lowest=col(q),
        u_lowest=col(fil["WS"]), v_lowest=col(np.zeros(n)),
        p_lowest=col(p_pa), p_surface=col(p_pa), rho_lowest=col(rho),
        cos_zenith=col(cos_zen), co2_ppmv=col(fil["CO2"]),
        has_radiation=col(np.ones(n)), has_precipitation=col(np.ones(n)),
    )

    # --- site metadata + PFT mapping (all validated before use) ---
    # IGBP/CLIMATE parsed NaN-safely (absent -> NaN -> sentinel -1, never crashes
    # int(round(...))); a finite fallback PFT/climate keeps the arrays buildable.
    igbp_raw, climate_raw = _scalar(ds, "IGBP"), _scalar(ds, "CLIMATE")
    igbp = int(round(igbp_raw)) if np.isfinite(igbp_raw) else -1
    climate_code = int(round(climate_raw)) if np.isfinite(climate_raw) else -1
    pft = _IGBP_TO_PFT.get(igbp, "DBF")
    climate = _CLIMATE_TO_NAME.get(climate_code, "temperate")
    aero = PFT_AERO_PARAMS.get(pft, PFT_AERO_PARAMS["DBF"])
    # Required site metadata: a SUPPORTED IGBP & CLIMATE code, C4 fraction in
    # [0,1], canopy height > 0, finite lat/lon.  Missing/invalid metadata is NOT
    # silently defaulted into a valid comparison (e.g. an absent C4 -> pure C3, or
    # an unknown IGBP -> DBF physiology); it invalidates every step instead, while
    # the finite fallbacks above keep the arrays well-formed.
    lat_deg, lon_deg = _scalar(ds, "LAT"), _scalar(ds, "LONG")
    fC4_raw, hc_raw = _scalar(ds, "C4"), _scalar(ds, "CANOPY_HEIGHT")
    site_ok = bool(igbp in _IGBP_TO_PFT and climate_code in _CLIMATE_TO_NAME
                   and np.isfinite(lat_deg) and np.isfinite(lon_deg)
                   and np.isfinite(fC4_raw) and 0.0 <= fC4_raw <= 1.0
                   and np.isfinite(hc_raw) and hc_raw > 0.0)
    valid = valid & site_ok & time_ok
    fC4 = float(np.clip(fC4_raw, 0.0, 1.0)) if np.isfinite(fC4_raw) else 0.0
    hc = hc_raw if (np.isfinite(hc_raw) and hc_raw > 0.0) else 1.0
    # Finite lat/lon fallbacks (kept invalid via site_ok) so a caller that runs
    # the model before applying `valid` never ingests NaN geometry.
    lat_rad = float(np.deg2rad(lat_deg)) if np.isfinite(lat_deg) else 0.0
    lon_out = float(lon_deg) if np.isfinite(lon_deg) else 0.0

    CI = fil["CI"]
    FNonVeg = np.exp(-0.5 * CI * LAI)                                # io.py:358

    # Vcmax (DifferBESS io.py:426-432): cropland -> always PFT/climate lookup;
    # other PFTs -> driver optimality value only when its PRE-fill coverage
    # exceeds 50% (else lookup).  Avoids extrapolating a sparse driver series.
    vcmax_coverage = float(obs["Vcmax25_C3Leaf"].mean())
    use_driver_vcmax = (igbp not in _CROPLAND_IGBP) \
        and (vcmax_coverage > _VCMAX_COVERAGE_MIN)
    vc3 = fil["Vcmax25_C3Leaf"] if use_driver_vcmax \
        else np.full(n, lookup_vcmax25(pft, climate, c4=False))
    vc4 = np.full(n, lookup_vcmax25(pft, climate, c4=True))
    # Only when the per-step DRIVER Vcmax is used does its observation gate
    # validity; the lookup fallback (cropland / sparse coverage) is the intended
    # production forcing and is valid at every step.
    if use_driver_vcmax:
        valid = valid & obs["Vcmax25_C3Leaf"]

    # MODIS BSA(direct)/WSA(diffuse) blended by the BESSRad diffuse-PAR fraction
    # (matches DifferBESS).  An entirely-missing ratio uses a 0.5 production
    # default for the blend, but is in _REQUIRED -> those steps are flagged
    # invalid (the blend was not genuinely observed), never silently compared.
    fdiff = fil["BESS_PAR_DIFF_PAR_RATIO"] if obs["BESS_PAR_DIFF_PAR_RATIO"].any() \
        else np.full(n, 0.5)
    alb_vis = (1.0 - fdiff) * fil["Albedo_BSA_vis"] + fdiff * fil["Albedo_WSA_vis"]
    alb_nir = (1.0 - fdiff) * fil["Albedo_BSA_nir"] + fdiff * fil["Albedo_WSA_nir"]

    # Emissivity production fallback: observed where present, else the site-mean
    # of observed values, else EMISS_VEG (0.97).  Gaps do NOT invalidate the
    # step (EMISSIVITY is excluded from _REQUIRED), matching DifferBESS.
    emiss_mean = (np.nanmean(np.where(obs["EMISSIVITY"], fil["EMISSIVITY"], np.nan))
                  if obs["EMISSIVITY"].any() else EMISS_VEG)
    emissivity_arr = np.where(obs["EMISSIVITY"], fil["EMISSIVITY"], emiss_mean)

    def colf(a):  # scalar -> (n, 1) jnp, broadcasting over time
        return jnp.broadcast_to(jnp.asarray(a, dtype=jnp.float64), (n,))[:, None]

    # Persistent structural LAI for the forest-floor litter cover (see
    # _persistent_litter_lai): keeps a deciduous forest's litter suppression active
    # through the leaf-off season instead of collapsing with the bare canopy.
    litter_LAI = _persistent_litter_lai(LAI, dt_s)

    canopy_params = CanopyLandParams(
        LAI=col(LAI), hc=colf(hc), litter_LAI=col(litter_LAI),
        fC4=colf(fC4), FNonVeg=col(FNonVeg), CI=col(CI), kn=colf(kn_site),
        Vcmax25_C3_leaf=col(vc3), Vcmax25_C4_leaf=col(vc4),
        m_C3=colf(M_C3), m_C4=colf(M_C4), b0_C3=colf(B0_C3), b0_C4=colf(B0_C4),
        alf=colf(ALF_DEFAULT), TgC=col(fil["T_GROWTH"]),
        ALB_VIS=col(np.clip(alb_vis, 0.0, 1.0)), ALB_NIR=col(np.clip(alb_nir, 0.0, 1.0)),
        emissivity=col(emissivity_arr),
        rz0m=colf(aero["rz0m"]), rd=colf(aero["rd"]),
        # PFT-specific leaf width (DifferBESS): ENF/DNF 0.01, EBF 0.04, GRA/SHR
        # 0.02, DBF/MF/SAV/CRO 0.025 — directly sets leaf boundary-layer resistance.
        d_leaf=colf(PFT_LEAF_WIDTH.get(pft, _D_LEAF_DEFAULT)),
    )

    # --- diagnostic soil prescription ---
    T_soil_top = col(fil["TS"] + constants.T_freeze)
    theta = fil["SWC"] / 100.0                        # % -> volumetric fraction
    w_frac_rz = col(np.clip((theta - theta_wp) / max(theta_fc - theta_wp, 1e-6),
                            0.0, 1.0))

    # --- observed tower fluxes (sentinels -> NaN; NaN preserved for masked
    #     comparison so missing tower data is never compared) ---
    et = _obs_masked(ds, "ET", n)                                    # mm/day
    le_obs = et * constants.L_v / _SECONDS_PER_DAY                    # -> W/m2 (NaN stays)
    obs_flux = {
        "gpp_umol": _obs_masked(ds, "GPP_DT", n),
        "le_wm2": le_obs,
        "h_wm2": _obs_masked(ds, "H", n),
        "nee_umol": _obs_masked(ds, "NEE", n),
        "ustar": _obs_masked(ds, "USTAR", n),     # friction velocity [m/s]
    }

    # Gap-free producer provenance (absent on a standard driver_v2).  Prefer the
    # atmospheric-only flag; fall back to the aggregate for older gap-free files.
    def _flag(name):
        return (np.asarray(ds[name].values).ravel().astype(np.int8)
                if name in ds.data_vars else None)
    met_filled = _flag("met_atm_filled")
    if met_filled is None:
        met_filled = _flag("met_any_filled")
    soil_filled = _flag("soil_filled")

    return ECSiteDriver(
        forcing=forcing, canopy_params=canopy_params, T_soil_top=T_soil_top,
        w_frac_rz=w_frac_rz, theta_soil=col(theta),
        wind_speed=col(fil["WS"]), valid=valid, obs=obs_flux,
        lat_rad=lat_rad, lon_deg=lon_out,
        doy=doy, dt_s=dt_s, igbp=igbp, climate=climate, pft=pft, site_id=site_id,
        met_filled=met_filled, soil_filled=soil_filled,
    )
