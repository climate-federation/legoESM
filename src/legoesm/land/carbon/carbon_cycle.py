"""Land carbon cycle: DifferLand prognostic model and seasonal cycle.

DifferLand (Fang & Gentine, Columbia) — DALEC990-based:
    6 carbon pools (labile, foliage, root, wood, litter, SOM) driven by a
    light-use-efficiency GPP with temperature/moisture responses.  Phenology
    follows DALEC990 Gaussian seasonal forcing.

Seasonal cycle:
    Prescribed sinusoidal NEE with latitude-dependent amplitude and phase.
    No prognostic pools — useful for testing carbon transport.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.land.carbon.config import CarbonConfig, CarbonState

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
_GC_TO_KG_CO2 = (44.0 / 12.0) * 1e-3   # gC -> kgCO2
_SPD = 86400.0                            # seconds per day

# ---------------------------------------------------------------------------
# Phenology offset polynomial coefficients (DALEC990)
# ---------------------------------------------------------------------------
_OFFSET_COEFFS = (
    0.000023599784710,
    0.000332730053021,
    0.000901865258885,
    -0.005437736864888,
    -0.020836027517787,
    0.126972018064287,
    -0.188459767342504,
)


# ===================================================================
# GPP — Light-Use Efficiency model
# ===================================================================

def compute_gpp(
    sw_down: jnp.ndarray,
    T: jnp.ndarray,
    LAI: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
    beta: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Gross primary production via light-use efficiency.

    Parameters
    ----------
    sw_down : Downward shortwave radiation [W/m2].
    T       : Surface temperature [K].
    LAI     : Leaf area index [m2/m2].
    co2_ppmv: Atmospheric CO2 [ppmv].
    beta    : Moisture availability [0-1].
    config  : CarbonConfig.

    Returns
    -------
    gpp : Instantaneous GPP [gC/m2/s].
    """
    # PAR = 48 % of SW, convert W/m2 -> MJ/m2/s
    PAR_MJ = 0.48 * sw_down * 1e-6

    # Absorbed fraction (Beer's law)
    fAPAR = 1.0 - jnp.exp(-config.k_ext * LAI)
    APAR = fAPAR * PAR_MJ

    T_C = T - constants.T_freeze
    f_T = jnp.exp(-0.5 * ((T_C - config.T_opt) / config.T_width) ** 2)

    # CO2 fertilization — Michaelis-Menten.  Use ``jnp.full`` (single
    # ``Broadcast`` HLO op) instead of
    # ``broadcast_to(jnp.asarray(scalar, dtype), shape)`` which
    # additionally forces a ``ConvertElementType``.
    if hasattr(co2_ppmv, "shape"):
        co2 = jnp.broadcast_to(co2_ppmv.astype(T.dtype), T.shape)
    else:
        co2 = jnp.full(T.shape, co2_ppmv, dtype=T.dtype)
    f_CO2 = co2 / (co2 + config.K_CO2)

    # GPP = epsilon * APAR * f_T * f_CO2 * beta  [gC/m2/s]
    return jnp.maximum(config.epsilon * APAR * f_T * f_CO2 * beta, 0.0)


# ===================================================================
# Phenology (DALEC990 Gaussian seasonal forcing)
# ===================================================================

def _phenology_offset(lifespan: float, width: float) -> jnp.ndarray:
    """Empirical DALEC990 offset correction (polynomial in ln(L-1))."""
    x = jnp.log(jnp.maximum(lifespan - 1.0, 1e-6))
    # Horner evaluation of degree-6 polynomial
    val = jnp.asarray(_OFFSET_COEFFS[0])
    for c in _OFFSET_COEFFS[1:]:
        val = val * x + c
    return width * val


def compute_phenology(
    doy: jnp.ndarray,
    lat: jnp.ndarray,
    config: CarbonConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Labile-release and leaf-fall fractions [day^-1].

    Parameters
    ----------
    doy  : Day of year [0-365].
    lat  : Latitude [radians].
    config : CarbonConfig.

    Returns
    -------
    lrf : Labile release fraction [day^-1].
    lff : Leaf fall fraction [day^-1].
    """
    sf = 365.25 / jnp.pi
    inv_sqrt_pi = 2.0 / jnp.sqrt(jnp.pi)
    sqrt2_half = jnp.sqrt(2.0) / 2.0

    # Labile release parameters
    fl = 0.5 * (
        jnp.log(config.lab_lifespan)
        - jnp.log(jnp.maximum(config.lab_lifespan - 1.0, 1e-6))
    )
    wl = config.clab_release_period * sqrt2_half
    osl = _phenology_offset(config.lab_lifespan, wl)

    # Leaf fall parameters
    ff = 0.5 * (
        jnp.log(config.leaf_lifespan)
        - jnp.log(jnp.maximum(config.leaf_lifespan - 1.0, 1e-6))
    )
    wf = config.leaf_fall_period * sqrt2_half
    osf = _phenology_offset(config.leaf_lifespan, wf)

    # Hemisphere-aware effective doy (shift by half year for SH).
    # ``jnp.full`` is one ``Broadcast`` HLO op vs the
    # ``broadcast_to(jnp.asarray(...))`` form's
    # ``ConvertElementType + Broadcast`` pair.
    if hasattr(doy, "shape"):
        doy_arr = jnp.broadcast_to(doy.astype(lat.dtype), lat.shape)
    else:
        doy_arr = jnp.full(lat.shape, doy, dtype=lat.dtype)
    if config.hemisphere_aware:
        doy_eff = jnp.where(
            lat < 0,
            jnp.mod(doy_arr + 182.5, 365.25),
            doy_arr,
        )
    else:
        doy_eff = doy_arr

    # Labile release factor — Gaussian pulse around Bday
    arg_l = jnp.sin((doy_eff - config.Bday + osl) / sf) * sf / wl
    lrf = inv_sqrt_pi * (fl / wl) * jnp.exp(-arg_l ** 2)

    # Leaf fall factor — Gaussian pulse around Fday
    arg_f = jnp.sin((doy_eff - config.Fday + osf) / sf) * sf / wf
    lff = inv_sqrt_pi * (ff / wf) * jnp.exp(-arg_f ** 2)

    return lrf, lff


# ===================================================================
# Decomposition modifier
# ===================================================================

def _temperate_modifier(
    T: jnp.ndarray,
    precip: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Temperature-moisture modifier for heterotrophic respiration."""
    temp_factor = jnp.exp(config.Q10_exp * (T - config.T_ref))
    precip_ratio = precip / jnp.maximum(config.precip_ref, 1e-10)
    moist = (precip_ratio - 1.0) * config.moisture_factor + 1.0
    moist = jnp.clip(moist, config.moist_modifier_min, config.moist_modifier_max)
    return temp_factor * moist


# ===================================================================
# Effective turnover for finite timestep
# ===================================================================

def _effective_rate(rate: jnp.ndarray, dt_days: jnp.ndarray) -> jnp.ndarray:
    """Convert daily rate *r* to effective flux fraction per day.

    Returns (1 - (1-r)^dt) / dt  so that  C * eff * dt = C * (1-(1-r)^dt),
    the exact exponential-decay loss over the timestep.
    """
    rate = jnp.clip(rate, 0.0, 1.0 - 1e-10)
    return (1.0 - (1.0 - rate) ** dt_days) / jnp.maximum(dt_days, 1e-10)


# ===================================================================
# DifferLand prognostic step
# ===================================================================

def step_carbon_differland(
    state: CarbonState,
    sw_down: jnp.ndarray,
    T: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
    beta: jnp.ndarray,
    lat: jnp.ndarray,
    doy: float,
    precip: jnp.ndarray,
    config: CarbonConfig,
    dt: float,
    gpp_override: jnp.ndarray | None = None,
) -> tuple[CarbonState, jnp.ndarray]:
    """Advance all six carbon pools by *dt* seconds.

    Parameters
    ----------
    state    : Current carbon pools [gC/m2].
    sw_down  : Downward SW radiation [W/m2].
    T        : Surface temperature [K].
    co2_ppmv : Atmospheric CO2 [ppmv].
    beta     : Moisture availability [0-1].
    lat      : Latitude [radians].
    doy      : Day of year.
    precip   : Total precipitation [kg/m2/s].
    config   : CarbonConfig.
    dt       : Timestep [s].

    Returns
    -------
    new_state : Updated CarbonState.
    co2_flux  : Net CO2 flux [kgCO2/m2/s], positive up.
    """
    dt_days = dt / _SPD

    # LAI from foliar carbon
    LAI = state.C_fol / config.LCMA

    # --- GPP ---------------------------------------------------------------
    if gpp_override is not None:
        gpp = gpp_override
    else:
        gpp = compute_gpp(sw_down, T, LAI, co2_ppmv, beta, config)  # gC/m2/s
    gpp_day = gpp * _SPD  # gC/m2/day rate

    # --- Autotrophic respiration & NPP -------------------------------------
    # Maintenance respiration: biomass-proportional, temperature-dependent,
    # always active (including nighttime and dormant seasons).
    temp_factor_ra = jnp.exp(config.Q10_exp * (T - config.T_ref))
    R_maint_day = (
        config.r_maint_fol * state.C_fol
        + config.r_maint_root * state.C_root
        + config.r_maint_wood * state.C_wood
    ) * temp_factor_ra  # gC/m2/day

    # Growth respiration: fraction of net assimilation (GPP minus maintenance)
    R_growth_day = config.f_auto * jnp.maximum(gpp_day - R_maint_day, 0.0)
    R_auto_day = R_maint_day + R_growth_day
    NPP_day = gpp_day - R_auto_day

    # --- NPP allocation (sequential partition) -----------------------------
    # Allocation only occurs when NPP > 0 (growth); maintenance losses are
    # already accounted for in R_auto_day and flow directly to atmosphere.
    NPP_pos = jnp.maximum(NPP_day, 0.0)
    A_fol = NPP_pos * config.f_fol
    A_lab = (NPP_pos - A_fol) * config.f_lab
    A_root = (NPP_pos - A_fol - A_lab) * config.f_root
    A_wood = jnp.maximum(NPP_pos - A_fol - A_lab - A_root, 0.0)

    # NPP deficit (NPP_day < 0, GPP cannot cover R_auto).  Cascade the
    # draw C_lab → C_fol → C_root → C_wood so the atmosphere gain via
    # NEE is matched exactly by biomass loss, even if any single pool
    # is exhausted (codex iter-62 stop-time review: "labile-reserve cap
    # still leaves ghost carbon when C_lab is exhausted").  Standard
    # CASA/DALEC carbon-starvation convention: labile reserves first,
    # then foliage, then root, then wood (slowest-turnover pool last).
    # Each draw is capped at the pool contents per timestep so no pool
    # goes negative within one step; the remaining deficit (if all four
    # pools are simultaneously exhausted — extremely rare) is the only
    # residual carbon-imbalance source and bounded by
    # ``(sum_pools)/dt``.
    _inv_dt_days = 1.0 / jnp.maximum(dt_days, 1e-10)
    npp_deficit_day = jnp.maximum(-NPP_day, 0.0)
    lab_deficit_draw = jnp.minimum(
        npp_deficit_day, state.C_lab * _inv_dt_days,
    )
    remaining_after_lab = npp_deficit_day - lab_deficit_draw
    fol_deficit_draw = jnp.minimum(
        remaining_after_lab, state.C_fol * _inv_dt_days,
    )
    remaining_after_fol = remaining_after_lab - fol_deficit_draw
    root_deficit_draw = jnp.minimum(
        remaining_after_fol, state.C_root * _inv_dt_days,
    )
    remaining_after_root = remaining_after_fol - root_deficit_draw
    wood_deficit_draw = jnp.minimum(
        remaining_after_root, state.C_wood * _inv_dt_days,
    )

    # --- Phenology ---------------------------------------------------------
    lrf, lff = compute_phenology(jnp.asarray(doy), lat, config)

    lab_release = state.C_lab * _effective_rate(lrf, dt_days)   # gC/m2/day
    leaf_litter = state.C_fol * _effective_rate(lff, dt_days)

    # --- Structural turnover -----------------------------------------------
    wood_litter = state.C_wood * _effective_rate(
        jnp.asarray(config.tor_wood), dt_days,
    )
    root_litter = state.C_root * _effective_rate(
        jnp.asarray(config.tor_root), dt_days,
    )

    # --- Heterotrophic respiration & decomposition -------------------------
    tempmod = _temperate_modifier(T, precip, config)

    R_het_lit = state.C_lit * _effective_rate(
        tempmod * config.tor_litter, dt_days,
    )
    R_het_som = state.C_som * _effective_rate(
        tempmod * config.tor_som, dt_days,
    )
    lit_to_som = state.C_lit * _effective_rate(
        tempmod * config.decomp_rate, dt_days,
    )

    # --- Pool updates (Euler, gC/m2/day rates * dt_days) -------------------
    # Smooth non-negativity (softplus): preserves AD gradients and allows
    # pools to approach zero without the artificial 1 gC/m2 hard floor.
    _alpha = 0.01  # smoothing scale [gC/m2]
    def _soft_pos(x):
        return _alpha * jnp.logaddexp(x / _alpha, 0.0)

    new_state = CarbonState(
        C_lab=_soft_pos(
            state.C_lab + (A_lab - lab_release - lab_deficit_draw) * dt_days),
        C_fol=_soft_pos(
            state.C_fol + (A_fol + lab_release - leaf_litter
                           - fol_deficit_draw) * dt_days),
        C_root=_soft_pos(
            state.C_root + (A_root - root_litter
                            - root_deficit_draw) * dt_days),
        C_wood=_soft_pos(
            state.C_wood + (A_wood - wood_litter
                            - wood_deficit_draw) * dt_days),
        C_lit=_soft_pos(
            state.C_lit + (leaf_litter + root_litter
                           - R_het_lit - lit_to_som) * dt_days),
        C_som=_soft_pos(
            state.C_som + (lit_to_som + wood_litter
                           - R_het_som) * dt_days),
    )

    # --- NEE: positive = source to atmosphere ------------------------------
    NEE_day = R_auto_day + R_het_lit + R_het_som - gpp_day
    co2_flux = NEE_day / _SPD * _GC_TO_KG_CO2

    return new_state, co2_flux


# ===================================================================
# Seasonal prescribed cycle
# ===================================================================

def seasonal_co2_flux(
    doy: float,
    lat: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Prescribed repeating seasonal NEE cycle.

    NH: peak uptake at *nee_peak_day* (~July).  SH: opposite phase.
    Amplitude peaks at mid-latitudes (~45 deg), falls toward equator/poles.

    Returns
    -------
    co2_flux : kgCO2/m2/s, positive up (source to atmosphere).
    """
    phase = 2.0 * jnp.pi * (doy - config.nee_peak_day) / 365.0

    # -cos peaks at phase=0  -->  uptake maximum at nee_peak_day
    seasonal = -jnp.cos(phase)

    # Flip sign for Southern Hemisphere
    seasonal = jnp.where(lat >= 0, seasonal, -seasonal)

    # Latitude weighting: sin(2|lat|) peaks at 45 deg
    lat_weight = jnp.sin(jnp.clip(2.0 * jnp.abs(lat), 0.0, jnp.pi))

    return config.nee_amplitude * seasonal * lat_weight


# ===================================================================
# Top-level dispatch
# ===================================================================

def step_carbon(
    carbon_state: CarbonState | None,
    sw_down: jnp.ndarray,
    T: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
    beta: jnp.ndarray,
    lat: jnp.ndarray,
    doy: float,
    precip: jnp.ndarray,
    config: CarbonConfig,
    dt: float,
    gpp_override: jnp.ndarray | None = None,
) -> tuple[CarbonState | None, jnp.ndarray]:
    """Dispatch carbon step based on *config.scheme*.

    Parameters
    ----------
    gpp_override : jnp.ndarray, optional
        When provided (e.g. from Farquhar photosynthesis), replaces the
        internal LUE-based GPP computation.

    Returns
    -------
    carbon_state : Updated state (None for "none"/"seasonal").
    co2_flux     : kgCO2/m2/s, positive up.
    """
    if config.scheme == "differland":
        if carbon_state is None:
            raise ValueError("differland scheme requires a CarbonState")
        return step_carbon_differland(
            carbon_state, sw_down, T, co2_ppmv, beta, lat, doy,
            precip, config, dt, gpp_override=gpp_override,
        )
    elif config.scheme == "seasonal":
        return carbon_state, seasonal_co2_flux(doy, lat, config)
    else:
        return carbon_state, jnp.zeros_like(T)


# ===================================================================
# Initialization
# ===================================================================

def init_carbon_state(
    shape: tuple[int, ...],
    config: CarbonConfig,
) -> CarbonState:
    """Create initial carbon pool state with uniform values from config."""
    return CarbonState(
        C_lab=jnp.full(shape, config.C_lab_init),
        C_fol=jnp.full(shape, config.C_fol_init),
        C_root=jnp.full(shape, config.C_root_init),
        C_wood=jnp.full(shape, config.C_wood_init),
        C_lit=jnp.full(shape, config.C_lit_init),
        C_som=jnp.full(shape, config.C_som_init),
    )
