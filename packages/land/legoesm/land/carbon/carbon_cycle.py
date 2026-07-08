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
from legoesm.land.carbon.config import (
    CarbonConfig,
    CarbonDiagnostics,
    CarbonState,
)

# Fixed calendar / radiation constants (not tunable).
_PAR_FRACTION_OF_SW = 0.48     # photosynthetically-active fraction of shortwave
_DAYS_PER_YEAR = 365.25        # Julian year length [days]
_HALF_YEAR_OFFSET_DAYS = 182.5  # half-year phenology phase offset [days]
_SEASONAL_YEAR_DAYS = 365.0    # year length for the NEE seasonal phase [days]

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
    PAR_MJ = _PAR_FRACTION_OF_SW * sw_down * 1e-6

    # Absorbed fraction (Beer's law)
    fAPAR = 1.0 - jnp.exp(-config.k_ext * LAI)
    APAR = fAPAR * PAR_MJ

    T_C = T - constants.T_freeze
    f_T = jnp.exp(-0.5 * ((T_C - config.T_opt_C) / config.T_width_C) ** 2)

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
    sf = _DAYS_PER_YEAR / jnp.pi
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
            jnp.mod(doy_arr + _HALF_YEAR_OFFSET_DAYS, _DAYS_PER_YEAR),
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
    """Temperature-moisture modifier for heterotrophic (soil) respiration.

    Uses ``Q10_het_exp`` (soil-decomposition Q10 ~2.5), decoupled from the
    autotrophic ``Q10_exp``: warm soils turn SOM/litter over fast (low
    equilibrium SOM) while cold soils retain carbon.
    """
    temp_factor = jnp.exp(config.Q10_het_exp * (T - config.T_ref))
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
    return_diagnostics: bool = False,
) -> (
    tuple[CarbonState, jnp.ndarray]
    | tuple[CarbonState, jnp.ndarray, CarbonDiagnostics]
):
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
    return_diagnostics : When True, also return a :class:`CarbonDiagnostics`
        with the full GPP / NPP / respiration / allocation / turnover
        breakdown (per-day rates [gC/m2/day]).  Purely diagnostic — the
        state update is byte-identical either way.

    Returns
    -------
    new_state : Updated CarbonState.
    co2_flux  : Net CO2 flux [kgCO2/m2/s], positive up.
    diag      : (only when ``return_diagnostics``) CarbonDiagnostics.
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
    A_root_base = (NPP_pos - A_fol - A_lab) * config.f_root
    A_wood_raw = jnp.maximum(NPP_pos - A_fol - A_lab - A_root_base, 0.0)
    # Woodiness is a static per-PFT flag -> Python branch (feature-gating
    # exception, not a data-dependent jnp.where).  Herbaceous PFTs (grass,
    # crop, tundra) have no wood: the structural fraction that would form wood
    # is invested belowground (roots) instead, so a grassland does not silently
    # grow a phantom multi-kgC tree pool.  Allocation still closes exactly
    # (A_fol + A_lab + A_root + A_wood == NPP_pos) either way.
    if config.woody:
        A_root = A_root_base
        A_wood = A_wood_raw
    else:
        A_root = A_root_base + A_wood_raw
        A_wood = jnp.zeros_like(A_wood_raw)

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

    # NPP deficit (NPP_day < 0, GPP cannot cover R_auto).  Cascade the draw
    # C_lab → C_wood → C_root → C_fol so the atmosphere gain via NEE is matched
    # exactly by biomass loss, even if any single pool is exhausted.
    #
    # DRAW ORDER = mobile reserve first, PHOTOSYNTHETIC ORGANS last.  A
    # DALEC-style daily respiration model run at the land model's sub-daily dt
    # sees GPP=0 every night while the biomass-proportional maintenance
    # respiration stays on, so every night NPP_day<0 and this cascade fires to
    # pay maintenance from stored carbon.  Which pool absorbs that recurring
    # nighttime draw matters:
    #   * Draining C_fol (the leaves) collapses LAI → GPP → the whole column
    #     (a carbon death-spiral once the labile reserve empties at leaf-fall).
    #   * Draining C_root (a functional, relatively small pool) empties it in
    #     weeks — a grassland root system cannot buffer a forest's respiration.
    # So after the labile (NSC) reserve, the draw goes to WOOD — the largest,
    # most allocation-fed pool (also the plant's structural NSC store) — which
    # absorbs the diurnal deficit yet still net-accumulates because daytime
    # allocation (the wood share of NPP) exceeds the nighttime draw whenever
    # daily NPP>0.  Roots and leaves are touched only in genuine starvation
    # (labile+wood both exhausted, e.g. a leafless herbaceous column in polar
    # winter), and the leaves strictly last.  This keeps every functional pool
    # at a physical steady state instead of one pool cannibalising to zero.
    #
    # Each draw is capped at the **net** pool contents after natural turnover
    # flows (lab_release / leaf_litter / etc.) so the pool cannot go below 0
    # once both the natural drain AND the deficit are applied (codex iter-63:
    # the cap must subtract the natural turnover drain, not use raw state.C_x).
    _inv_dt_days = 1.0 / jnp.maximum(dt_days, 1e-10)
    npp_deficit_day = jnp.maximum(-NPP_day, 0.0)
    # Net pool capacity per day after subtracting natural turnover drain
    # (and adding natural turnover gain, e.g. ``lab_release → C_fol``).
    # ``A_x`` contributions are zero in deficit regime (NPP_pos = 0).
    lab_net_avail = jnp.maximum(
        state.C_lab + (A_lab - lab_release) * dt_days, 0.0,
    ) * _inv_dt_days
    fol_net_avail = jnp.maximum(
        state.C_fol + (A_fol + lab_release - leaf_litter) * dt_days, 0.0,
    ) * _inv_dt_days
    root_net_avail = jnp.maximum(
        state.C_root + (A_root - root_litter) * dt_days, 0.0,
    ) * _inv_dt_days
    wood_net_avail = jnp.maximum(
        state.C_wood + (A_wood - wood_litter) * dt_days, 0.0,
    ) * _inv_dt_days

    # Draw order: labile (mobile NSC) → wood (structural store/buffer)
    # → root → foliage (photosynthetic organ, strictly last).
    lab_deficit_draw = jnp.minimum(npp_deficit_day, lab_net_avail)
    remaining_after_lab = npp_deficit_day - lab_deficit_draw
    wood_deficit_draw = jnp.minimum(remaining_after_lab, wood_net_avail)
    remaining_after_wood = remaining_after_lab - wood_deficit_draw
    root_deficit_draw = jnp.minimum(remaining_after_wood, root_net_avail)
    remaining_after_root = remaining_after_wood - root_deficit_draw
    fol_deficit_draw = jnp.minimum(remaining_after_root, fol_net_avail)
    # Unmet NPP deficit: the part of the maintenance-respiration demand that
    # NO pool could supply (all biomass exhausted).  R_auto is charged to
    # the atmosphere via NEE below, so the flux MUST be reduced by this
    # undrawn remainder or NEE reports carbon that left no pool (atmosphere
    # gain > biomass loss).  See finding #5.
    unmet_npp_deficit_day = jnp.maximum(
        remaining_after_root - fol_deficit_draw, 0.0,
    )

    # --- Heterotrophic respiration & decomposition -------------------------
    tempmod = _temperate_modifier(T, precip, config)

    # The litter pool is drained by TWO independent first-order sinks:
    # heterotrophic respiration (-> atmosphere) and decomposition to SOM
    # (-> C_som).  Each ``_effective_rate`` is the exact single-sink decay,
    # but their SUM can exceed the litter net availability for large dt,
    # which ``_soft_pos`` would clip while NEE still reported the full,
    # uncapped R_het_lit -> atmosphere gain > litter loss.  Apply a joint
    # net-availability cap (cf. the biomass deficit cascade): scale both
    # demanded sinks by the same factor so their realised sum never exceeds
    # ``C_lit + (leaf_litter + root_litter)*dt`` (the litter content after
    # this step's litter inputs).  The atm/SOM partition ratio is preserved.
    # See finding #4.
    R_het_lit_demand = state.C_lit * _effective_rate(
        tempmod * config.tor_litter, dt_days,
    )
    lit_to_som_demand = state.C_lit * _effective_rate(
        tempmod * config.decomp_rate, dt_days,
    )
    R_het_som = state.C_som * _effective_rate(
        tempmod * config.tor_som, dt_days,
    )
    lit_total_demand = R_het_lit_demand + lit_to_som_demand  # gC/m2/day
    lit_net_avail = jnp.maximum(
        state.C_lit + (leaf_litter + root_litter) * dt_days, 0.0,
    ) * _inv_dt_days  # gC/m2/day available to drain over the step
    # scale in [0, 1]; == 1 (no-op) whenever demand <= availability.
    lit_scale = jnp.where(
        lit_total_demand > lit_net_avail,
        lit_net_avail / jnp.maximum(lit_total_demand, 1e-30),
        1.0,
    )
    R_het_lit = lit_scale * R_het_lit_demand
    lit_to_som = lit_scale * lit_to_som_demand

    # Coarse woody debris: wood turnover does NOT humify 100 % into the
    # millennial SOM pool.  Split it like the litter pathway — a fraction
    # ``cwd_humification_eff`` becomes stable SOM, the rest respires to the
    # atmosphere (CWD heterotrophic respiration).  Routing all of wood_litter
    # to C_som (the previous behaviour) drove an unphysically large soil-carbon
    # stock (SOM_eq ~ input × 550-yr turnover).  Conserves exactly:
    # wood_litter = wood_to_som + R_het_cwd, so C_wood loses wood_litter,
    # C_som gains wood_to_som, and the atmosphere gains R_het_cwd (added to NEE).
    wood_to_som = config.cwd_humification_eff * wood_litter
    R_het_cwd = wood_litter - wood_to_som

    # --- Pool updates (Euler, gC/m2/day rates * dt_days) -------------------
    # Hard non-negativity ``jnp.maximum(x, 0)``.  The iter-64 cap
    # (``lab_net_avail`` etc., computed against the pool AFTER natural
    # turnover) guarantees ``state.C + (A − drain − deficit) · dt ≥ 0``
    # exactly, so no smoothing is required.  Previously a softplus
    # (``_alpha · logaddexp(x/_alpha, 0)`` with ``_alpha = 0.01``) was
    # used; even when ``raw_new_C == 0`` exactly, ``softplus(0) =
    # _alpha · log(2) ≈ 0.007 gC/m2`` of phantom carbon was added per
    # pool per step (codex iter-63 stop-time review).  ``jnp.maximum``
    # has subgradient 0 below 0 and 1 above — well-defined for both
    # forward simulation and AD when crossing zero is unphysical
    # anyway.
    def _soft_pos(x):
        return jnp.maximum(x, 0.0)

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
            state.C_som + (lit_to_som + wood_to_som
                           - R_het_som) * dt_days),
    )

    # --- NEE: positive = source to atmosphere ------------------------------
    # R_het_lit is the litter-availability-CAPPED respiration (finding #4);
    # ``unmet_npp_deficit_day`` removes the maintenance respiration that no
    # biomass pool could supply (finding #5); ``R_het_cwd`` is the coarse
    # woody-debris respiration split off from wood turnover.  With these
    # corrections the column closes exactly: sum(dC_pools) == -NEE_day*dt
    # (verified by test_carbon_*_conservation).
    NEE_day = (
        R_auto_day - unmet_npp_deficit_day
        + R_het_lit + R_het_som + R_het_cwd - gpp_day
    )
    co2_flux = NEE_day / _SPD * _GC_TO_KG_CO2

    if not return_diagnostics:
        return new_state, co2_flux

    diag = CarbonDiagnostics(
        gpp=gpp_day,
        npp=NPP_day,
        r_maint=R_maint_day,
        r_growth=R_growth_day,
        r_auto=R_auto_day,
        r_het_lit=R_het_lit,
        r_het_som=R_het_som,
        r_het_cwd=R_het_cwd,
        r_het=R_het_lit + R_het_som + R_het_cwd,
        nee=NEE_day,
        unmet_npp_deficit=unmet_npp_deficit_day,
        a_fol=A_fol,
        a_lab=A_lab,
        a_root=A_root,
        a_wood=A_wood,
        lab_release=lab_release,
        leaf_litter=leaf_litter,
        root_litter=root_litter,
        wood_litter=wood_litter,
        wood_to_som=wood_to_som,
        lit_to_som=lit_to_som,
        lai=LAI,
    )
    return new_state, co2_flux, diag


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
    phase = 2.0 * jnp.pi * (doy - config.nee_peak_day) / _SEASONAL_YEAR_DAYS

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
    return_diagnostics: bool = False,
) -> tuple[CarbonState | None, jnp.ndarray] | tuple[
    CarbonState | None, jnp.ndarray, CarbonDiagnostics
]:
    """Dispatch carbon step based on *config.scheme*.

    Parameters
    ----------
    gpp_override : jnp.ndarray, optional
        When provided (e.g. from Farquhar photosynthesis), replaces the
        internal LUE-based GPP computation.
    return_diagnostics : bool, optional
        Return the CarbonDiagnostics breakdown as a third element.  Only
        the ``differland`` scheme produces diagnostics; requesting them for
        another scheme raises ``ValueError`` (there is no NPP/allocation
        breakdown for a prescribed or disabled carbon cycle).

    Returns
    -------
    carbon_state : Updated state (None for "none"/"seasonal").
    co2_flux     : kgCO2/m2/s, positive up.
    diag         : (only when ``return_diagnostics``) CarbonDiagnostics.
    """
    if config.scheme == "differland":
        if carbon_state is None:
            raise ValueError("differland scheme requires a CarbonState")
        return step_carbon_differland(
            carbon_state, sw_down, T, co2_ppmv, beta, lat, doy,
            precip, config, dt, gpp_override=gpp_override,
            return_diagnostics=return_diagnostics,
        )
    if return_diagnostics:
        raise ValueError(
            f"return_diagnostics is only supported for the 'differland' "
            f"carbon scheme, not {config.scheme!r}."
        )
    if config.scheme == "seasonal":
        return carbon_state, seasonal_co2_flux(doy, lat, config)
    elif config.scheme == "none":
        return carbon_state, jnp.zeros_like(T)
    else:
        raise ValueError(
            f"Unknown carbon scheme {config.scheme!r}; expected one of "
            "'none', 'differland', 'seasonal'."
        )


# ===================================================================
# Initialization
# ===================================================================

def init_carbon_state(
    shape: tuple[int, ...],
    config: CarbonConfig,
) -> CarbonState:
    """Create initial carbon pool state with uniform values from config.

    An herbaceous config (``woody=False``) starts with NO wood pool: the flag
    disables both wood allocation AND the initial wood stock, so a grassland
    never carries a phantom tree that would otherwise respire, turn over, and
    feed CWD/SOM for decades from the ``C_wood_init`` default.  Woody configs
    use ``C_wood_init`` unchanged.
    """
    C_wood_init = config.C_wood_init if config.woody else 0.0
    return CarbonState(
        C_lab=jnp.full(shape, config.C_lab_init),
        C_fol=jnp.full(shape, config.C_fol_init),
        C_root=jnp.full(shape, config.C_root_init),
        C_wood=jnp.full(shape, C_wood_init),
        C_lit=jnp.full(shape, config.C_lit_init),
        C_som=jnp.full(shape, config.C_som_init),
    )
