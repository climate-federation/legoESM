"""Documented tuning guide as executable code.

Provides a registry of key tunable parameters with valid ranges,
sensitivities, and descriptions, plus helpers that validate an
experiment configuration and suggest resolution-appropriate defaults.
"""

from __future__ import annotations

from typing import NamedTuple


from legoesm import constants

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class TuningParameter(NamedTuple):
    """Metadata for a single tunable model parameter."""

    name: str
    default: float
    min_val: float
    max_val: float
    units: str
    description: str
    category: str       # "dynamics", "radiation", "convection", "diffusion", "surface"
    sensitivity: str    # "high", "medium", "low"
    notes: str


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

TUNING_PARAMETERS: dict[str, TuningParameter] = {
    # -- Dynamics --------------------------------------------------------
    "dt": TuningParameter(
        name="dt",
        default=600.0,
        min_val=300.0,
        max_val=1200.0,
        units="s",
        description="Integration time step",
        category="dynamics",
        sensitivity="high",
        notes=(
            "Must satisfy CFL: dt < 0.8 * dx_min / (u_max + sqrt(gH)). "
            "Halve when doubling resolution."
        ),
    ),
    "hyperdiff_scale": TuningParameter(
        name="hyperdiff_scale",
        default=5e16,
        min_val=1e15,
        max_val=1e18,
        units="m^4/s",
        description="Hyper-diffusion coefficient scale",
        category="dynamics",
        sensitivity="high",
        notes=(
            "Controls small-scale noise removal. Use physical e-folding "
            "time: nu4 = dx^4 / tau_efold. Too large causes overdamping; "
            "too small lets grid-scale noise grow."
        ),
    ),

    # -- Radiation -------------------------------------------------------
    "tau_equator": TuningParameter(
        name="tau_equator",
        default=7.2,
        min_val=5.0,
        max_val=10.0,
        units="1",
        description="Gray radiation optical depth at equator",
        category="radiation",
        sensitivity="medium",
        notes="Controls tropical longwave cooling. Higher = warmer tropics.",
    ),
    "tau_pole": TuningParameter(
        name="tau_pole",
        default=1.8,
        min_val=1.0,
        max_val=3.0,
        units="1",
        description="Gray radiation optical depth at poles",
        category="radiation",
        sensitivity="medium",
        notes="Controls polar longwave cooling. Lower = colder poles.",
    ),
    "tau_moist_coeff": TuningParameter(
        name="tau_moist_coeff",
        default=0.0115,
        min_val=0.005,
        max_val=0.030,
        units="m^2/kg",
        description="Gray radiation moisture LW optical-depth coefficient",
        category="radiation",
        sensitivity="high",
        notes=(
            "Strength of the water-vapour longwave greenhouse in the gray "
            "scheme: dtau_moist = tau_moist_coeff * column_water. Higher = "
            "stronger greenhouse, lower OLR. Frierson (2006) default 0.0115."
        ),
    ),
    "linear_frac": TuningParameter(
        name="linear_frac",
        default=0.2,
        min_val=0.05,
        max_val=0.6,
        units="1",
        description="Gray radiation linear vs sigma^4 LW weighting fraction",
        category="radiation",
        sensitivity="medium",
        notes=(
            "Fraction f_l of the linear-in-sigma LW optical-depth profile "
            "vs the sigma^4 profile. Shifts where the gray greenhouse acts "
            "vertically. Frierson (2006) default 0.2."
        ),
    ),
    "lw_diff_factor": TuningParameter(
        name="lw_diff_factor",
        default=1.66,
        min_val=1.4,
        max_val=2.0,
        units="1",
        description="Gray radiation LW diffusivity factor D",
        category="radiation",
        sensitivity="medium",
        notes=(
            "Hemispheric-mean two-stream diffusivity factor. Scales the "
            "effective LW path length; ~5/3 (1.66) is the standard value."
        ),
    ),
    "sfc_emissivity": TuningParameter(
        name="sfc_emissivity",
        default=0.97,
        min_val=0.90,
        max_val=1.0,
        units="1",
        description="Surface longwave emissivity",
        category="radiation",
        sensitivity="medium",
        notes=(
            "Surface LW emissivity for the gray scheme. Lower emissivity "
            "reduces upward surface LW, warming the surface."
        ),
    ),
    "sw_tau_0": TuningParameter(
        name="sw_tau_0",
        default=0.22,
        min_val=0.0,
        max_val=0.6,
        units="1",
        description="Gray radiation SW optical-depth scale",
        category="radiation",
        sensitivity="medium",
        notes=(
            "Shortwave optical-depth scale: tau_sw(sigma) = sw_tau_0 * "
            "sigma^sw_exponent. Higher = more atmospheric SW absorption, "
            "less SW reaching the surface. 0.0 = surface-absorbing limit."
        ),
    ),
    "sw_exponent": TuningParameter(
        name="sw_exponent",
        default=2.0,
        min_val=1.0,
        max_val=4.0,
        units="1",
        description="Gray radiation SW optical-depth vertical exponent",
        category="radiation",
        sensitivity="low",
        notes=(
            "Exponent of the SW optical-depth profile sigma^sw_exponent; "
            "controls how SW absorption is distributed in the vertical."
        ),
    ),
    "S_0": TuningParameter(
        name="S_0",
        default=constants.S_0,
        min_val=1340.0,
        max_val=1380.0,
        units="W/m^2",
        description="Solar constant",
        category="radiation",
        sensitivity="low",
        notes="Total solar irradiance. Standard value ~1361 W/m^2.",
    ),
    "rad_update_steps": TuningParameter(
        name="rad_update_steps",
        default=1,
        min_val=1.0,
        max_val=12.0,
        units="steps",
        description="Radiation call cadence (every N time steps)",
        category="radiation",
        sensitivity="medium",
        notes=(
            "Radiation is expensive; calling less often saves compute "
            "but may miss fast-evolving cloud forcing."
        ),
    ),

    # -- Convection ------------------------------------------------------
    "sbm_tau_c": TuningParameter(
        name="sbm_tau_c",
        default=7200.0,
        min_val=3600.0,
        max_val=14400.0,  # validated training range (trainable_params.py);
                          # re-widen tuning + trainable together if intended
        units="s",
        description="Convective relaxation timescale (SBM)",
        category="convection",
        sensitivity="high",
        notes=(
            "Shorter = more aggressive convective adjustment. "
            "Typical range 2-4 hours. Upper bound widened to 1e6 s "
            "(~12 days) after 2026-05-26 diagnostic showed SBM "
            "over-heats the troposphere by 5-9 K at every level; "
            "calibration may want to weaken SBM toward off."
        ),
    ),
    "sbm_RH_ref": TuningParameter(
        name="sbm_RH_ref",
        default=0.7,
        min_val=0.6,
        max_val=0.9,
        units="1",
        description="Reference relative humidity for SBM convection",
        category="convection",
        sensitivity="high",
        notes="Column moistened toward this RH. Higher = wetter atmosphere.",
    ),
    "sundqvist_auto_rate": TuningParameter(
        name="sundqvist_auto_rate",
        default=1e-3,
        min_val=2e-4,
        max_val=5e-3,
        units="1/s",
        description="Sundqvist autoconversion rate (cloud water -> rain)",
        category="convection",
        sensitivity="high",
        notes=(
            "Rate at which cloud water converts to (diagnostic, instantly "
            "falling) rain: P_auto = auto_rate * q_c. Higher = more "
            "efficient precipitation and lower cloud water (LWP); lower = "
            "more cloud water retained. Sundqvist et al. (1989)."
        ),
    ),
    "cloud_rh_crit": TuningParameter(
        name="cloud_rh_crit",
        default=0.7,
        min_val=0.5,
        max_val=0.95,
        units="1",
        description="Critical relative humidity for cloud onset",
        category="radiation",
        sensitivity="high",
        notes=(
            "Sundqvist cloud-fraction threshold: cloud fraction = "
            "clamp((RH - rh_crit)/(1 - rh_crit), 0, 1). Higher = less cloud "
            "cover, higher OSR/OLR. The primary cloud-amount knob."
        ),
    ),
    "cloud_r_eff_liq": TuningParameter(
        name="cloud_r_eff_liq",
        default=10.0e-6,
        min_val=4.0e-6,
        max_val=20.0e-6,
        units="m",
        description="Cloud liquid droplet effective radius (single value, no land/ocean split)",
        category="radiation",
        sensitivity="high",
        notes=(
            "Sets cloud shortwave optical thickness (tau ~ LWP / r_eff). "
            "Smaller droplets => optically thicker, brighter clouds => "
            "higher OSR. Genuinely uncertain (depends on CCN / aerosol)."
        ),
    ),
    "sundqvist_evap_coeff": TuningParameter(
        name="sundqvist_evap_coeff",
        default=5e-4,
        min_val=1e-4,
        max_val=2e-3,
        units="1",
        description="Sundqvist sub-cloud rain evaporation coefficient",
        category="convection",
        sensitivity="high",
        notes=(
            "Rate at which falling rain re-evaporates in subsaturated "
            "layers: evap = evap_coeff * subsaturation * P_total. Higher "
            "= more rain lost to evaporation, drier surface precip and "
            "moister mid-troposphere; lower = more rain reaches the "
            "surface. Sundqvist et al. (1989)."
        ),
    ),
    "sbm_cape_threshold": TuningParameter(
        name="sbm_cape_threshold",
        default=70.0,
        min_val=0.0,
        max_val=200.0,
        units="J/kg",
        description="Minimum CAPE to trigger SBM convection",
        category="convection",
        sensitivity="high",
        notes=(
            "Columns with CAPE below this are gated off via a smooth "
            "sigmoid trigger. Higher = convection fires less readily, "
            "fewer/weaker convective columns; lower = more widespread "
            "convection."
        ),
    ),

    # -- Scheme knobs added for calibration (AIMIP commit 0c747d4) -------
    # Threaded to the schemes via the calibration's physics_cfg_overrides
    # dict-of-dicts (keys micro / conv / turb / gwd).
    "sundqvist_sigmoid_sharpness": TuningParameter(
        name="sundqvist_sigmoid_sharpness",
        default=20.0,
        min_val=5.0,
        max_val=60.0,
        units="1",
        description="Sundqvist cloud-fraction smooth-activation sharpness",
        category="convection",
        sensitivity="medium",
        notes=(
            "Sharpness of the sigmoid that ramps cloud fraction across the "
            "critical RH. Higher = sharper (more step-like) cloud onset."
        ),
    ),
    "louis_l_mix_max": TuningParameter(
        name="louis_l_mix_max",
        default=100.0,
        min_val=20.0,
        max_val=400.0,
        units="m",
        description="Louis turbulence asymptotic mixing length",
        category="turbulence",
        sensitivity="high",
        notes=(
            "Asymptotic (free-troposphere) mixing length for the Louis "
            "boundary-layer scheme. Higher = stronger vertical mixing."
        ),
    ),
    "louis_Ri_crit": TuningParameter(
        name="louis_Ri_crit",
        default=0.25,
        min_val=0.1,
        max_val=0.6,
        units="1",
        description="Louis turbulence critical Richardson number",
        category="turbulence",
        sensitivity="medium",
        notes=(
            "Richardson-number scale in the Louis stability functions; "
            "sets how readily stable layers suppress turbulence."
        ),
    ),
    "louis_b_louis": TuningParameter(
        name="louis_b_louis",
        default=5.0,
        min_val=2.0,
        max_val=10.0,
        units="1",
        description="Louis turbulence stability-function coefficient b",
        category="turbulence",
        sensitivity="low",
        notes="Coefficient b in the Louis (1979/1982) stability functions.",
    ),
    "louis_c_louis": TuningParameter(
        name="louis_c_louis",
        default=16.6,
        min_val=5.0,
        max_val=30.0,
        units="1",
        description="Louis turbulence stability-function coefficient c",
        category="turbulence",
        sensitivity="low",
        notes="Coefficient c in the Louis stability functions (1979: 5, updated 16.6).",
    ),
    "louis_d_louis": TuningParameter(
        name="louis_d_louis",
        default=5.0,
        min_val=2.0,
        max_val=15.0,
        units="1",
        description="Louis turbulence stability-function coefficient d",
        category="turbulence",
        sensitivity="low",
        notes="Coefficient d in the Louis stability functions.",
    ),
    "mcfarlane_k_wave": TuningParameter(
        name="mcfarlane_k_wave",
        default=6.283185307e-5,
        min_val=1.0e-5,
        max_val=2.0e-4,
        units="1/m",
        description="McFarlane orographic GWD horizontal wavenumber",
        category="gwd",
        sensitivity="medium",
        notes=(
            "Horizontal wavenumber of the launched orographic gravity "
            "waves; scales the launch stress tau_0 ~ G_0*rho*N*k*h^2*U."
        ),
    ),
    # NOTE: ``mcfarlane_N_ref`` was REMOVED from this catalog 2026-07-24.  It
    # advertised a "reference stratification used in the orographic
    # launch-stress closure" that does not exist: McFarlaneConfig has no
    # ``N_ref`` field and mcfarlane_gwd derives N from the column state, so
    # tuning it changed nothing; the ExperimentConfig scalar was deleted
    # 2026-09-26.  Re-add only if a
    # real reference-stratification closure parameter is introduced.
    "mcfarlane_directional_spread": TuningParameter(
        name="mcfarlane_directional_spread",
        default=1.0,
        min_val=0.5,
        max_val=2.0,
        units="1",
        description="McFarlane GWD multi-directional spreading factor",
        category="gwd",
        sensitivity="low",
        notes="Spreads the launched wave stress over multiple directions.",
    ),
    "mcfarlane_tau_max": TuningParameter(
        name="mcfarlane_tau_max",
        default=10.0,
        min_val=1.0,
        max_val=30.0,
        units="Pa",
        description="McFarlane GWD upper clip on launch stress",
        category="gwd",
        sensitivity="low",
        notes=(
            "Upper bound on orographic launch stress; protects against "
            "runaway drag in pathological columns."
        ),
    ),


    # -- Diffusion / turbulence -----------------------------------------
    "k_free_per_day": TuningParameter(
        name="k_free_per_day",
        default=0.1,
        min_val=0.01,
        max_val=1.0,
        units="1/day",
        description="Free-atmosphere vertical mixing rate",
        category="diffusion",
        sensitivity="medium",
        notes="Background mixing above the BL. Too large smears out jets.",
    ),
    "k_BL_max_per_day": TuningParameter(
        name="k_BL_max_per_day",
        default=1.0,
        min_val=0.5,
        max_val=5.0,
        units="1/day",
        description="Maximum boundary layer mixing rate",
        category="diffusion",
        sensitivity="high",
        notes=(
            "Peak mixing in the BL. Controls depth and intensity of "
            "surface coupling."
        ),
    ),
    "sigma_b": TuningParameter(
        name="sigma_b",
        default=0.7,
        min_val=0.6,
        max_val=0.85,
        units="1",
        description="Boundary layer top (sigma level)",
        category="diffusion",
        sensitivity="medium",
        notes="Sigma level above which BL mixing tapers off.",
    ),

    # -- Surface ---------------------------------------------------------
    "C_H": TuningParameter(
        name="C_H",
        default=1.5e-3,
        min_val=1e-3,
        max_val=5e-3,
        units="1",
        description="Sensible heat bulk transfer coefficient",
        category="surface",
        sensitivity="high",
        notes="Larger = stronger surface sensible heat flux.",
    ),
    "C_E": TuningParameter(
        name="C_E",
        default=1.5e-3,
        min_val=1e-3,
        max_val=5e-3,
        units="1",
        description="Latent heat bulk transfer coefficient",
        category="surface",
        sensitivity="high",
        notes="Larger = stronger surface evaporation.",
    ),
    "albedo_land": TuningParameter(
        name="albedo_land",
        default=0.3,
        min_val=0.1,
        max_val=0.4,
        units="1",
        description="Land surface albedo",
        category="surface",
        sensitivity="medium",
        notes="Affects net surface SW absorption over land.",
    ),
    "C_land": TuningParameter(
        name="C_land",
        default=2.0e5,
        min_val=5.0e4,
        max_val=1.0e6,
        units="J/m^2/K",
        description="Slab-land effective heat capacity",
        category="surface",
        sensitivity="medium",
        notes=(
            "Sets the land skin-temperature thermal inertia: smaller "
            "C_land → larger diurnal range, sharper response to "
            "radiative forcing; larger C_land → damped, ocean-like "
            "behaviour.  ~2e5 corresponds to a ~0.15 m active layer "
            "of moist soil."
        ),
    ),
    "emissivity_land": TuningParameter(
        name="emissivity_land",
        default=constants.emissivity_land,
        min_val=0.85,
        max_val=1.00,
        units="1",
        description="Land surface LW emissivity",
        category="surface",
        sensitivity="medium",
        notes=(
            "Multiplies both LW absorption and LW emission at the "
            "land surface.  Tunes the OLR contribution from land "
            "and the radiative damping rate of T_land."
        ),
    ),
    "albedo_ice": TuningParameter(
        name="albedo_ice",
        default=0.65,
        min_val=0.4,
        max_val=0.8,
        units="1",
        description="Sea ice albedo",
        category="surface",
        sensitivity="high",
        notes=(
            "Strong ice-albedo feedback. Higher = more SW reflected "
            "by ice, colder poles."
        ),
    ),
    "albedo_ocean": TuningParameter(
        name="albedo_ocean",
        default=0.06,
        min_val=0.03,  # matches trainable_params.py validated range
        max_val=0.10,
        units="1",
        description="Open-ocean surface shortwave albedo",
        category="surface",
        sensitivity="medium",
        notes=(
            "Genuine ocean surface reflectance (~0.06) under RRTMGP, where "
            "clouds carry the planetary SW reflection. Tuned only within a "
            "tight physical range — it is a surface property, not a free "
            "planetary-albedo knob."
        ),
    ),
    # ---- Morrison ice-microphysics knobs (sub-stepped Morrison only) ----
    # Active only when microphysics='morrison'; under Sundqvist the
    # MicrophysicsConfig hasattr filter skips them silently.
    "morrison_bergeron_rate": TuningParameter(
        name="morrison_bergeron_rate",
        default=1e-3,
        min_val=1e-4,
        max_val=1e-2,
        units="1/s",
        description="Morrison Bergeron-Findeisen rate (supercooled liquid -> ice)",
        category="convection",
        sensitivity="high",
        notes=(
            "Primary lever for moving mass from q_c to q_i in the "
            "mixed-phase zone. Higher = lower LWP, higher IWP, stronger "
            "LW_CRE. Active only with microphysics='morrison'."
        ),
    ),
    "morrison_rime_coeff": TuningParameter(
        name="morrison_rime_coeff",
        default=1.0,
        min_val=0.1,
        max_val=2.0,
        units="1",
        description="Morrison riming collection efficiency (ice/snow capture q_c)",
        category="convection",
        sensitivity="high",
        notes=(
            "Controls how aggressively ice and snow collect cloud water. "
            "Higher = converts liquid to ice/snow faster, reducing LWP and "
            "raising IWP. Pairs with morrison_bergeron_rate."
        ),
    ),
    "morrison_dep_coeff": TuningParameter(
        name="morrison_dep_coeff",
        # Was default=1e-8, range 3e-9..3e-8 — FIVE ORDERS below the real
        # MorrisonConfig.dep_coeff default (1e-3) that every run uses; the
        # scalar was unwired so nothing ever noticed (flag-reachability audit
        # 2026-07-25).  Range re-centred on the true default, same ~3x span.
        default=1e-3,
        min_val=3e-4,
        max_val=3e-3,
        units="1",
        description="Morrison depositional ice growth coefficient (vapor -> q_i)",
        category="convection",
        sensitivity="high",
        notes=(
            "Morrison-2005 capacitance form: dq_i/dt = dep_coeff * S_i * "
            "q_i^(1/3) * N_i^(2/3) * f_ice.  Higher = q_i grows faster in "
            "cold supersaturated layers (cirrus mass).  Direct lever on "
            "LW_CRE / IWP without affecting LWP.  Range sized so a typical "
            "cirrus layer (T=220K, N_i=1e5/m^3, q_i=1e-7 kg/kg, S_i=0.1) "
            "gives dq_i/dt ~ 1e-8 kg/kg/s at the midpoint."
        ),
    ),
    "morrison_agg_coeff": TuningParameter(
        name="morrison_agg_coeff",
        default=1e-3,
        min_val=1e-4,
        max_val=5e-3,
        units="1/s",
        description="Morrison ice-to-snow aggregation rate",
        category="convection",
        sensitivity="medium",
        notes=(
            "Loss of q_i to q_s (snow). Lower = ice persists longer in "
            "the column, higher IWP. Snow itself precipitates faster than "
            "ice so once q_i is aggregated the column loses it."
        ),
    ),
    "morrison_k_au": TuningParameter(
        name="morrison_k_au",
        default=6e2,
        min_val=2e2,
        max_val=2e3,
        units="1/(kg*s)",
        description="Morrison warm-rain autoconversion rate (Seifert-Beheng k_au)",
        category="convection",
        sensitivity="high",
        notes=(
            "Drains q_c to q_r in warm/mixed columns. Higher = lower LWP "
            "via warm-rain channel (independent of the ice path). Useful "
            "when LWP is too high (Morrison runs liquid-rich)."
        ),
    ),
    # (cloud_rh_ice_crit / cloud_rh_ice_sat catalog entries DELETED
    # 2026-07-26: the ExperimentConfig fields documented a CloudConfig
    # RH_i cirrus ramp that was never implemented — no
    # CloudConfig.rh_ice_crit/rh_ice_sat exists (flag-reachability audit
    # cause 3, codex-verified).  Advertising tuning ranges for
    # nonexistent physics invited wasted calibration campaigns.)
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _extract_tuning_fields(config):
    """Extract tuning-relevant fields from either config type.

    Returns a namespace-like object with attributes: resolution, dt,
    hyperdiff_scale, co2_ppmv, sbm_tau_c, sbm_RH_ref, rad_update_steps,
    radiation, cloud_scheme, microphysics, C_H, C_E, albedo_ice,
    albedo_ocean, nlev.

    Accepts ``ExperimentConfig`` (nested sub-configs) or
    ``AMIPExperimentConfig`` (flat fields).
    """
    from legoesm.driver.config import ExperimentConfig

    if isinstance(config, ExperimentConfig):
        class _Ns:
            pass
        ns = _Ns()
        ns.resolution = config.grid.resolution
        ns.dt = config.dycore.dt
        ns.hyperdiff_scale = config.dycore.hyperdiff_scale
        ns.co2_ppmv = config.co2_ppmv
        ns.sbm_tau_c = config.sbm_tau_c
        ns.sbm_RH_ref = config.sbm_RH_ref
        ns.rad_update_steps = config.rad_update_steps
        ns.radiation = config.radiation
        ns.cloud_scheme = config.cloud_scheme
        ns.microphysics = config.microphysics
        ns.C_H = config.C_H
        ns.C_E = config.C_E
        ns.albedo_ice = config.albedo_ice
        ns.albedo_ocean = config.albedo_ocean
        ns.nlev = config.grid.nlev
        return ns
    # Legacy AMIPExperimentConfig — fields are flat
    return config


def validate_tuning(config) -> list[str]:
    """Return a list of warning messages for potentially problematic settings.

    Accepts either ``ExperimentConfig`` (canonical) or
    ``AMIPExperimentConfig`` (legacy).

    Checks parameter ranges, CFL heuristics, and conflicting options.
    """
    cfg = _extract_tuning_fields(config)
    warnings_list: list[str] = []

    # -- dt vs. resolution heuristic ------------------------------------
    _dt_limits = {16: 900.0, 32: 600.0, 48: 600.0, 64: 300.0, 96: 150.0}
    dt_max = _dt_limits.get(cfg.resolution)
    if dt_max is not None and cfg.dt > dt_max:
        warnings_list.append(
            f"dt={cfg.dt:.0f}s may violate CFL at C{cfg.resolution} "
            f"resolution (recommend dt<={dt_max:.0f}s)"
        )

    # -- Hyper-diffusion vs. resolution ---------------------------------
    _hd_ranges = {
        16: (1e16, 1e18),
        48: (1e15, 5e16),
        96: (1e14, 1e16),
    }
    hd_range = _hd_ranges.get(cfg.resolution)
    if hd_range is not None:
        lo, hi = hd_range
        if cfg.hyperdiff_scale < lo:
            warnings_list.append(
                f"hyperdiff_scale={cfg.hyperdiff_scale:.1e} may be too "
                f"weak for C{cfg.resolution} (recommended >= {lo:.0e})"
            )
        if cfg.hyperdiff_scale > hi:
            warnings_list.append(
                f"hyperdiff_scale={cfg.hyperdiff_scale:.1e} may be too "
                f"strong for C{cfg.resolution} (recommended <= {hi:.0e})"
            )

    # -- Gas concentrations ---------------------------------------------
    if cfg.co2_ppmv <= 0.0:
        warnings_list.append(
            f"co2_ppmv={cfg.co2_ppmv} — zero or negative CO2 will "
            "cause radiation issues"
        )
    if cfg.co2_ppmv > 2000.0:
        warnings_list.append(
            f"co2_ppmv={cfg.co2_ppmv} — unusually high CO2 "
            "(pre-industrial ~280, current ~420)"
        )

    # -- SBM parameters -------------------------------------------------
    if cfg.sbm_tau_c < 1800.0:
        warnings_list.append(
            f"sbm_tau_c={cfg.sbm_tau_c:.0f}s is very short — may "
            "cause excessive convective adjustment"
        )
    if cfg.sbm_RH_ref > 0.95:
        warnings_list.append(
            f"sbm_RH_ref={cfg.sbm_RH_ref:.2f} is very high — may "
            "cause near-saturated atmosphere everywhere"
        )
    if cfg.sbm_RH_ref < 0.4:
        warnings_list.append(
            f"sbm_RH_ref={cfg.sbm_RH_ref:.2f} is very low — may "
            "produce an unrealistically dry atmosphere"
        )

    # -- Radiation cadence ----------------------------------------------
    if cfg.rad_update_steps > 12:
        warnings_list.append(
            f"rad_update_steps={cfg.rad_update_steps} is very large — "
            "radiation will be very stale between calls"
        )

    # -- Conflicting options -------------------------------------------
    if cfg.radiation == "gray" and cfg.cloud_scheme != "none":
        warnings_list.append(
            "Cloud scheme is active but radiation='gray' — clouds will "
            "have no effect on radiation"
        )
    if cfg.microphysics != "none" and cfg.cloud_scheme == "none":
        warnings_list.append(
            "Microphysics is active but cloud_scheme='none' — cloud "
            "water will not affect radiation"
        )

    # -- Surface bulk coefficients --------------------------------------
    for name in ("C_H", "C_E"):
        val = getattr(cfg, name)
        if val > 0.01:
            warnings_list.append(
                f"{name}={val:.4f} is unrealistically large "
                "(typical range 1e-3 to 5e-3)"
            )
        if val < 1e-4:
            warnings_list.append(
                f"{name}={val:.1e} is very small — surface fluxes will "
                "be negligible"
            )

    # -- Albedo range checks -------------------------------------------
    if not (0.0 <= cfg.albedo_ice <= 1.0):
        warnings_list.append(
            f"albedo_ice={cfg.albedo_ice} is outside [0, 1]"
        )
    if not (0.0 <= cfg.albedo_ocean <= 1.0):
        warnings_list.append(
            f"albedo_ocean={cfg.albedo_ocean} is outside [0, 1]"
        )

    # -- Vertical levels ------------------------------------------------
    if cfg.nlev < 10:
        warnings_list.append(
            f"nlev={cfg.nlev} — very few vertical levels, may not "
            "resolve the boundary layer or tropopause"
        )

    return warnings_list


# ---------------------------------------------------------------------------
# Recommended parameters
# ---------------------------------------------------------------------------

def recommended_params(resolution: int, nlev: int) -> dict:
    """Return recommended parameter values for a given resolution and nlev.

    Parameters
    ----------
    resolution : int
        Cubed-sphere face resolution (e.g. 16, 48, 96).
    nlev : int
        Number of vertical levels (informational; may influence dt slightly
        in the future).

    Returns
    -------
    dict
        Suggested parameter values keyed by parameter name.
    """
    presets = {
        16: dict(dt=600.0, hyperdiff_scale=5e16, rad_update_steps=1),
        48: dict(dt=300.0, hyperdiff_scale=5e15, rad_update_steps=3),
        96: dict(dt=150.0, hyperdiff_scale=5e14, rad_update_steps=6),
    }

    if resolution in presets:
        return presets[resolution]

    # Interpolate / extrapolate for arbitrary resolutions
    # dt scales roughly as 1/resolution (CFL), hyperdiff as 1/resolution^4
    ref_res = 48
    ref = presets[ref_res]
    scale = ref_res / resolution
    return dict(
        dt=round(ref["dt"] * scale / 10) * 10,      # round to nearest 10s
        hyperdiff_scale=ref["hyperdiff_scale"] * scale**4,
        rad_update_steps=max(1, round(ref["rad_update_steps"] / scale)),
    )


# ---------------------------------------------------------------------------
# Pretty-printed guide
# ---------------------------------------------------------------------------

def print_tuning_guide() -> None:
    """Print a formatted table of all tuning parameters."""
    categories = sorted({p.category for p in TUNING_PARAMETERS.values()})

    header = (
        f"{'Name':<22s} {'Default':>10s}  {'Range':>22s}  "
        f"{'Units':<8s} {'Sens.':<6s} Description"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  legoESM Tuning Guide")
    print("=" * len(header))

    for cat in categories:
        params = [
            p for p in TUNING_PARAMETERS.values() if p.category == cat
        ]
        if not params:
            continue

        print()
        print(f"  [{cat.upper()}]")
        print(sep)
        print(header)
        print(sep)

        for p in params:
            # Format default and range compactly
            def _fmt(v: float) -> str:
                if v >= 1e6 or (0 < v < 1e-2):
                    return f"{v:.0e}"
                if v == int(v):
                    return f"{int(v)}"
                return f"{v:g}"

            default_s = _fmt(p.default)
            range_s = f"[{_fmt(p.min_val)}, {_fmt(p.max_val)}]"

            print(
                f"  {p.name:<20s} {default_s:>10s}  {range_s:>22s}  "
                f"{p.units:<8s} {p.sensitivity:<6s} {p.description}"
            )
            if p.notes:
                # Wrap notes below the entry
                print(f"{'':>24s}  -> {p.notes}")

        print(sep)

    print()
