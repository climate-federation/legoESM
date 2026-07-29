"""Configuration for diagnostic cloud fraction schemes.

Provides CloudConfig for controlling cloud fraction diagnosis and
cloud optical property computation for radiation coupling.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants


__param_spec__ = {
    "CloudConfig": {
        "scheme_key": "atm.clouds.CloudConfig",
        "excluded": {
            # Lower clip on the Martin gamma-PSD shape (1/PGAM^2 - 1 clipped to
            # [pgam_min, pgam_max]); a numerics regulariser/cap on the droplet
            # spectral-width, not a tunable closure coefficient. Paired with
            # the pgam_max cap (Morrison module_mp_mg.F90).
            "pgam_min": "numerics: lower clip/cap on the gamma-PSD shape parameter (regulariser, paired with pgam_max)",
            "cloud_optics_asymmetry_g": "numerics: scattering asymmetry g of the two-region inhomogeneity two-stream reflectance (shapes the reduction; the real per-band g lives in RRTMGP, not trained here)",
            "clubb_cf_override_p_min_pa": "structural: BL-top pressure [Pa] above which the diagnostic-CLUBB cloud-fraction override applies (a level/regime gate, not a trained closure coefficient); 0 => full-column override",
            "clubb_cf_override_ramp_pa": "numerics: smoothing width [Pa] of the override level gate (linear blend over [p_min-ramp, p_min] to avoid a cloud/heating discontinuity); a regulariser, not a trained coefficient; 0 => sharp step",
            # Defaults sit on a HARD physical bound (1.0 = full / neutral, 0.0 =
            # off): a sigmoid maps to the OPEN (lo, hi) so an on-bound default is
            # unseedable, and the bound cannot widen past the physical limit
            # (strength/inhomogeneity>1 or a negative floor are unphysical). So
            # these opt-in levers are fixed (tier 0) at their off/neutral default;
            # a scenario that trains one gives it a scenario-specific interior
            # default. Same class as land.canopy.interception_fraction.
            "clubb_cf_override_strength": "opt-in marine-Sc lever, default 1.0 (full) = the physical ceiling; not a well-posed sigmoid tunable (default on the bound)",
            "clubb_cf_override_floor": "opt-in marine-Sc cloud-collapse floor, default 0.0 (off) = the physical floor; not a well-posed sigmoid tunable (default on the bound)",
            "cloud_inhomogeneity_factor": "Cahalan plane-parallel-bias reduction, default 1.0 (homogeneous, no reduction) = the physical ceiling; not a well-posed sigmoid tunable (default on the bound)",
        },
        "params": {
            # --- critical_rh: primary cloud-onset RH (Sundqvist + Xu-Randall lower bound) ---
            "rh_crit": {"units": "1", "bounds": (0.5, 0.99), "tunable_tier": 1, "transform": "sigmoid", "category": "critical_rh", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            # --- cloud_fraction: Xu-Randall (1996) cf = RH^p_xr * (1 - exp(-alpha*q_c/((1-RH)q_sat)^gamma)) ---
            "alpha_xr": {"units": "1", "bounds": (10.0, 1000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "cloud_fraction", "reference": "Xu & Randall (1996)", "shape": None},
            "p_xr": {"units": "1", "bounds": (0.05, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cloud_fraction", "reference": "Xu & Randall (1996)", "shape": None},
            "gamma_xr": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cloud_fraction", "reference": "Xu & Randall (1996)", "shape": None},
            # (clubb_cf_override_strength / clubb_cf_override_floor: excluded —
            #  opt-in marine-Sc levers whose defaults sit on a hard bound.)
            # --- condensate: diagnostic in-cloud water + resolved-cf condensate scale [kg/kg] ---
            "q_c_diagnostic": {"units": "kg/kg", "bounds": (5.0e-5, 1.5e-3), "tunable_tier": 1, "transform": "sigmoid", "category": "condensate", "reference": "diagnostic-cloud scheme default", "shape": None},
            "q_cloud_resolved_ref": {"units": "kg/kg", "bounds": (1.0e-7, 1.0e-5), "tunable_tier": 2, "transform": "sigmoid", "category": "condensate", "reference": "resolved (CRM/SAM) cloud-fraction scheme default", "shape": None},
            # --- ice_fraction: temperature below which all condensate is ice [K] ---
            "T_ice_only": {"units": "K", "bounds": (220.0, 268.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_fraction", "reference": "linear ice-fraction ramp scheme default", "shape": None},
            # --- optical_radius: fixed-fallback effective radii [m] for RRTMGP cloud optics ---
            "r_eff_liq": {"units": "m", "bounds": (4.0e-6, 30.0e-6), "tunable_tier": 2, "transform": "sigmoid", "category": "optical_radius", "reference": "cloud-optics fallback default", "shape": None},
            "r_eff_ice": {"units": "m", "bounds": (10.0e-6, 90.0e-6), "tunable_tier": 2, "transform": "sigmoid", "category": "optical_radius", "reference": "cloud-optics fallback default", "shape": None},
            # (cloud_inhomogeneity_factor: excluded — default 1.0 on the bound.)
            "cloud_fsd": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "optical_radius", "reference": "Shonk & Hogan (2008, 2010) fractional standard deviation of in-cloud water", "shape": None},
            # --- droplet_psd: Morrison M2005 liquid effective-radius PSD (gamma-shape from Nc) ---
            "Nc_default": {"units": "1/m^3", "bounds": (1.0e7, 1.0e9), "tunable_tier": 2, "transform": "sigmoid", "category": "droplet_psd", "reference": "Morrison et al. (2005) M2005 (SAM Nc_0)", "shape": None},
            "martin_pgam_slope": {"units": "cm^3", "bounds": (1.0e-4, 2.0e-3), "tunable_tier": 3, "transform": "sigmoid", "category": "droplet_psd", "reference": "Martin et al. (1994)", "shape": None},
            "martin_pgam_intercept": {"units": "1", "bounds": (0.1, 0.8), "tunable_tier": 3, "transform": "sigmoid", "category": "droplet_psd", "reference": "Martin et al. (1994)", "shape": None},
            "pgam_max": {"units": "1", "bounds": (4.0, 30.0), "tunable_tier": 3, "transform": "sigmoid", "category": "droplet_psd", "reference": "Morrison module_mp_mg.F90 (gamma-PSD shape cap)", "shape": None},
            # --- microphysics_density: M2005 cloud-ice bulk density [kg/m^3] for ice r_eff PSD ---
            "rho_cloud_ice": {"units": "kg/m^3", "bounds": (100.0, 917.0), "tunable_tier": 3, "transform": "sigmoid", "category": "microphysics_density", "reference": "Morrison et al. (2005) M2005 (RHOI)", "shape": None},
            # --- convective_cloud: Slingo(1987)-style cumulus cloud-fraction from convective precip (opt-in) ---
            "conv_cloud_coeff": {"units": "1", "bounds": (0.0, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "convective_cloud", "reference": "Slingo (1987) convective cloud-amount vs ln(precip)", "shape": None},
            "conv_cloud_max": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "convective_cloud", "reference": "Slingo (1987) convective cloud-amount cap", "shape": None},
            "conv_precip_scale": {"units": "kg/m^2/s", "bounds": (1.0e-6, 1.0e-4), "tunable_tier": 2, "transform": "sigmoid", "category": "convective_cloud", "reference": "convective-cloud reference precip rate (~1 mm/day)", "shape": None},
            "conv_cloud_sigma_top": {"units": "1", "bounds": (0.05, 0.4), "tunable_tier": 0, "transform": "sigmoid", "category": "convective_cloud", "reference": "convective cloud-deck top (sigma); numerics layer-bound", "shape": None},
            "conv_cloud_sigma_base": {"units": "1", "bounds": (0.35, 0.98), "tunable_tier": 0, "transform": "sigmoid", "category": "convective_cloud", "reference": "convective anvil-deck base (sigma); numerics layer-bound", "shape": None},
            "conv_cloud_condensate": {"units": "kg/kg", "bounds": (1.0e-5, 1.0e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "condensate", "reference": "thin anvil-cirrus in-cloud condensate", "shape": None},
            # --- condensate (adiabatic in-cloud LWC growth rate, opt-in vertical structure) ---
            "adiabatic_lwc_rate": {"units": "kg/kg/m", "bounds": (5.0e-7, 3.0e-6), "tunable_tier": 2, "transform": "sigmoid", "category": "condensate", "reference": "adiabatic cloud LWC gradient ~1-2 g/kg per km (Brenguier et al. 2000)", "shape": None},
        },
    },
}


class CloudConfig(NamedTuple):
    """Configuration for diagnostic cloud fraction and cloud-radiation coupling.

    Fields
    ------
    scheme : str
        Cloud fraction scheme:
        - ``"sundqvist"``: RH-based (Sundqvist 1988). Simple, no condensate needed.
        - ``"xu_randall"``: RH + condensate-based (Xu & Randall 1996).
          Requires explicit q_cloud/q_ice from microphysics.
        - ``"resolved"``: cloud-resolving (CRM) cloud fraction. A grid
          cell is fully cloudy where it holds condensate — SAM's
          convention at CRM resolution. SAM uses a HARD binary
          ``cf = (qn > 0)``; this is a smooth surrogate
          ``cf = q_cond/(q_cond + q_cloud_resolved_ref)`` → 1 for
          ``q_cond ≫ ref``.  ``ref`` is a tunable cloud-PRESENCE scale,
          not a SAM constant.  ``cf`` is smooth in condensate, but (a)
          the gradient ``∂cf/∂q_cond → 1/ref`` is large as ``q_cond→0``
          (a caveat for differentiable training through radiation, not
          for forward RCE), and (b) full radiation-operator AD-smoothness
          also depends on the RRTMGP cloud-overlap/McICA treatment.
          Requires explicit q_cloud/q_ice.
        - ``"none"``: No clouds (clear-sky radiation).
    rh_crit : float
        Critical relative humidity for cloud onset (default 0.7).
        Used by Sundqvist scheme and as lower bound in Xu-Randall.
    alpha_xr : float
        Condensate scaling in Xu-Randall formula (default 100.0).
    p_xr : float
        RH exponent in Xu-Randall formula (default 0.25).
    gamma_xr : float
        Saturation-deficit exponent in the Xu-Randall denominator
        ``((1−RH)·q_sat)^γ`` (default 0.49, the Xu & Randall 1996
        best-fit value; the earlier code omitted it ⇒ γ=1).
    r_eff_liq : float
        Effective radius for liquid cloud droplets [m] (default 10e-6 = 10 um).
    r_eff_ice : float
        Effective radius for ice cloud particles [m] (default 30e-6 = 30 um).
    q_c_diagnostic : float
        Typical in-cloud liquid water content [kg/kg] used when explicit
        cloud condensate is not available (default 0.2e-3 = 0.2 g/kg).
    T_freeze : float
        Temperature [K] at which condensate begins transitioning to ice
        (default 273.15).
    T_ice_only : float
        Temperature [K] below which all condensate is ice (default 233.15).
    q_cloud_resolved_ref : float
        Condensate scale [kg/kg] for the ``"resolved"`` (CRM) cloud
        fraction ``cf = q_cond/(q_cond + ref)`` (default 1e-6 = 1 mg/kg,
        so any resolved cloud with ``q_cond ≳ 0.1 g/kg`` gives cf ≈ 1).
    """
    scheme: str = "none"
    # 0.8 (standard Sundqvist/ECHAM) matches the microphysics
    # SundqvistConfig.rh_crit=0.8.  At 0.7 the cloud-FRACTION diagnostic (used
    # for the RRTMGP cloud-radiative effect) onset 0.1 RH BELOW where the
    # microphysics condenses, so the 0.70-0.80 RH band produced radiative cloud
    # with no matching condensate.  Because cf = 1 - sqrt((1-RH)/(1-rh_crit))
    # multiplies the cloud optical depth (both LW + SW), that mismatch fed a
    # cooling -> RH-up -> cf-up -> OLR-down/albedo-up positive feedback that
    # cold-drifted the coupled rrtmgp run to a ~277 K overcast plateau.
    rh_crit: float = 0.77
    alpha_xr: float = 100.0
    p_xr: float = 0.25
    gamma_xr: float = 0.49
    r_eff_liq: float = 10.0e-6
    r_eff_ice: float = 30.0e-6
    # Cahalan et al. (1994) horizontal-inhomogeneity factor on the radiative
    # in-cloud water path: real clouds are horizontally PATCHY, so a plane-
    # parallel HOMOGENEOUS layer carrying the same mean water is systematically
    # too reflective (the plane-parallel albedo bias).  Operational GCMs scale
    # LWP/IWP by chi ~ 0.7 to correct it.  1.0 = homogeneous (legacy, no change);
    # < 1 reduces the effective optical depth (both SW + LW).
    cloud_inhomogeneity_factor: float = 1.0
    Nc_default: float = 1.0e8        # fallback cloud-droplet number [1/m³] for the
                                     # gamma-PSD liquid effective radius when the
                                     # passed n_cloud is 0/garbage — e.g. SAM
                                     # specified-Nc Morrison (dopredictNc=.false.,
                                     # MorrisonConfig.predict_Nc=False) where the
                                     # prognostic Nc slot stays 0. Matches Morrison
                                     # Nc_0=1e8 so RRTMGP r_eff_liq is SAM-faithful.
    q_c_diagnostic: float = 1.0e-3   # in-cloud condensate for the radiative
    # cloud-opacity floor [kg/kg].  Raised 0.2e-3 -> 1.0e-3 per the CAM
    # surface-energy audit: thin clouds gave planetary albedo ~14% (vs ~30%)
    # AND weak LW_down (~270 vs ~340) => surface LW loss -110 (vs -55) =>
    # energy-starved evaporation => cold/dry feedback.  More in-cloud condensate
    # raises BOTH cloud albedo (SW) and cloud LW emissivity (LW_down), running
    # the cold/dry feedback in reverse.  Upper bound of the __param_spec__ range.
    T_freeze: float = constants.T_freeze
    T_ice_only: float = 233.15
    q_cloud_resolved_ref: float = 1.0e-6
    # M2005 cloud-ice bulk density [kg/m³] (RHOI) for the PSD ice effective
    # radius EFFI=1.5/LAMI, LAMI=(ρ_ci·π·N_i/q_i)^(1/3) (RAD-1-ice). Only
    # used when ``compute_cloud_properties`` is given explicit ``n_ice``.
    rho_cloud_ice: float = 500.0
    # Martin et al. (1994) gamma-PSD spectral-shape (pgam) fit used by the
    # M2005 liquid effective radius: PGAM = slope·Nc[cm⁻³] + intercept, then
    # 1/PGAM² − 1 clipped to [pgam_min, pgam_max] (Morrison module_mp_mg.F90).
    martin_pgam_slope: float = 0.0005714
    martin_pgam_intercept: float = 0.2714
    pgam_min: float = 2.0
    pgam_max: float = 10.0
    # CLUBB cloud-fraction override LEVEL GATE (marine-Sc albedo lever): when a
    # ``cloud_fraction_override`` (diagnostic CLUBB's PDF cf) is supplied, apply it
    # ONLY where ``p_full >= clubb_cf_override_p_min_pa`` — the boundary layer /
    # low cloud, the marine-Sc target — keeping the RH grid-scale fraction ALOFT.
    # The real-SST A/B showed a FULL-COLUMN override over-clouds at altitude (OLR
    # collapse to ~160 W/m² + albedo RISE 0.53->0.65) because CLUBB's PDF
    # over-diagnoses high/mid cloud; restricting it to the BL removes that backfire
    # while keeping the intended low-cloud reduction.  ``0.0`` => apply at ALL
    # levels (the original full-column override).
    clubb_cf_override_p_min_pa: float = 70000.0
    # SMOOTH ramp width [Pa] for the override level gate: the CLUBB fraction is
    # blended in linearly over ``[p_min - ramp, p_min]`` (full override at
    # p_full >= p_min; RH cf below p_min - ramp) instead of a sharp step at p_min.
    # A step gate created a cloud/heating discontinuity at ~700 hPa that seeded a
    # late (day-20) numerical blowup in the real-SST A/B; the ramp removes it.
    # ``0.0`` => sharp step (the original level gate).
    clubb_cf_override_ramp_pa: float = 10000.0
    # Override STRENGTH [0,1]: the effective BL cloud fraction is
    # ``strength*CLUBB_cf + (1-strength)*RH_cf`` — a partial blend toward CLUBB
    # rather than a full replacement.  Full replacement (1.0) removed enough low
    # cloud under real forcing to drive a surface-heating runaway (day-15..20
    # blowup, NOT fixed by halving dt); a gentler blend still lowers albedo but
    # keeps the column stable.  1.0 => full replacement (the original override).
    clubb_cf_override_strength: float = 1.0
    # Minimum BL cloud fraction the override may leave [0,1].  The blowup diagnosis
    # (blowup_state day 20) showed BL cloud collapsing to ~0 globally under the
    # lever -> near-clear-sky -> a cloud-temperature positive-feedback RUNAWAY
    # (less cloud -> warmer/drier -> less RH cloud -> warmer ...; mid-trop hit
    # 400 K).  Flooring the overridden cf breaks the runaway at its source while
    # still allowing a (bounded) low-cloud reduction, so a LARGER albedo fix can
    # run STABLY than the strength knob alone allows.  0.0 => no floor (original).
    clubb_cf_override_floor: float = 0.0
    # --- Convective cloud fraction (Slingo 1987), OPT-IN (default OFF) ---
    # The RH-based stratiform schemes (sundqvist/xu_randall) give cloud only
    # near saturation, so an adjustment convection scheme (sbm) that holds the
    # tropical column at RH~0.7 produces NO radiative cloud => the convecting
    # tropics radiate surface LW straight to space (measured LW_net_sfc ~-137
    # W/m^2, precip ~1 mm/day, ~4.5 K cold bias).  When ``convective_cloud`` is
    # True, ``convective_cloud_fraction`` adds a bounded cumulus cloud cover
    # ``cf_conv = clip(coeff * ln(1 + P_conv/P0), 0, cf_max)`` over the
    # free-tropospheric deck [sigma_top, sigma_base], combined with the
    # stratiform fraction by maximum overlap.  Default False => byte-identical
    # to the validated stratiform-only path (no production change).
    convective_cloud: bool = False
    # Tuned DOWN from coeff=0.15/cap=0.6/deck[0.15,0.90] (15 levels), which
    # over-produced: applying cf_conv across the whole free troposphere, each
    # level then carrying the cf*q_c_diagnostic radiative-condensate floor,
    # stacked column LWP to OVERCAST (validation 8534361: albedo 78.9%, R_TOA
    # -100 W/m2 — the mass_flux failure mode).  Concentrate the cover in a thin
    # upper-tropospheric ANVIL deck [0.15,0.45] (~3-4 levels) with a small
    # coeff/cap so the convective cloud NUDGES the tropical cloud-radiative
    # effect instead of dominating it.
    # v2 (coeff0.04/cap0.2/deck[0.15,0.45]) closed the LW deficit (LW_net_sfc
    # -110 -> -66, CWV 14 -> 25) but albedo was still 44% (Earth ~30%): the
    # anvil reflected too much SW, so the LW warming was offset (R_TOA -15, SST
    # drift unchanged).  v3 RAISES + THINS the anvil to make it LW-DOMINANT —
    # colder/higher cloud tops trap LW efficiently while a thinner, higher deck
    # reflects less SW.
    conv_cloud_coeff: float = 0.04      # cloud-amount per e-fold of P_conv
    conv_cloud_max: float = 0.15        # cap on convective cloud cover
    conv_precip_scale: float = 1.1574e-5  # ~1 mm/day in kg/m^2/s (P0)
    conv_cloud_sigma_top: float = 0.10   # convective anvil deck top (sigma)
    conv_cloud_sigma_base: float = 0.35  # convective anvil deck base (sigma)
    # In-cloud condensate [kg/kg] for the convective EXCESS fraction — an
    # optically-THIN anvil cirrus (~7x less than the thick stratiform
    # q_c_diagnostic=1e-3) so the high cloud traps LW without over-reflecting SW
    # (the v3 albedo~42% overshoot; high cold tops keep the LW benefit).
    conv_cloud_condensate: float = 1.5e-4
    # --- Diagnostic in-cloud condensate vertical structure (opt-in) ---
    # The stratiform radiative floor ``q_total_diag = cf * q_c_diagnostic`` uses
    # a CONSTANT in-cloud water (1 g/kg) at every cloudy level.  Measured against
    # AMIP checkpoints this over-brightens THIN warm marine stratocumulus: the
    # radiative q_c there is the floor (11.5x the prognostic, dominating 77% of
    # BL cells), and a shallow Sc's real in-cloud LWC (~0.2-0.5 g/kg) is set by
    # its DEPTH, not the deep-cloud calibration value.  ``"adiabatic"`` replaces
    # the constant with a capped adiabatic LWC that grows with cloudy depth above
    # cloud base (``q_ad = adiabatic_lwc_rate * D``, ``D`` the cloudy GEOMETRIC
    # depth from cloud base to the level MIDPOINT — reset at clear gaps — capped
    # at ``q_c_diagnostic``, and only for warm/liquid cells): thin low
    # clouds dim while a deep cloud is ~unchanged ABOVE the cap depth (~667 m;
    # its near-base layers still dim, but q_ad can only DIM, never exceed
    # q_c_diagnostic — preserving the LW_down / anti-too-dark calibration that
    # raised q_c_diagnostic to 1e-3, and leaving CRM 'resolved' + SBM alone).
    # Physics of the decoupling: SW cloud albedo is UNSATURATED in optical depth
    # so it drops with the water path; LW emissivity SATURATES above ~20 g/m^2
    # LWP so LW_down is ~untouched.  Default ``"constant"`` = byte-identical to
    # the validated floor.
    diagnostic_condensate_scheme: str = "constant"
    adiabatic_lwc_rate: float = 1.5e-6   # in-cloud LWC growth per metre of cloudy
    # depth [kg/kg/m] ~ 1.5 g/kg per km (adiabatic marine-Sc gradient); only read
    # when diagnostic_condensate_scheme="adiabatic".
    # --- Sub-grid cloud-optics inhomogeneity (appended at the END of the
    # NamedTuple so positional / checkpoint callers keep their field order) ---
    # Scheme: "constant" (Cahalan scalar cloud_inhomogeneity_factor above;
    # legacy, byte-identical default) or "two_region" (tau-DEPENDENT
    # Shonk & Hogan 2008 factor chi_eff = 1 - fsd^2 tau/(gamma0+tau) that reduces
    # a THICK cloud more than a thin one; asymptote 1-fsd^2).  Unknown => raise.
    cloud_optics_inhomogeneity: str = "constant"
    # Fractional standard deviation of in-cloud water for two_region (Shonk &
    # Hogan 2010 global mean ~0.75; broken marine Sc -> ~1).  fsd -> 0 is
    # homogeneous (chi_eff -> 1); higher fsd => larger reduction (floor 1-fsd^2).
    cloud_fsd: float = 0.75
    # Scattering asymmetry g of the two_region conservative two-stream
    # reflectance R(t) = t/(t + 2/(1-g)).  ~0.85 for liquid clouds (Mie, SW).
    # A numerics constant of the optic (the real per-band g lives in RRTMGP).
    cloud_optics_asymmetry_g: float = 0.85


def build_cloud_config(
    scheme: str,
    *,
    convective_cloud: bool = False,
    rh_crit: float | None = None,
    q_c_diagnostic: float | None = None,
    conv_cloud_max: float | None = None,
    conv_cloud_condensate: float | None = None,
    cloud_inhomogeneity_factor: float | None = None,
    cloud_optics_inhomogeneity: str | None = None,
    cloud_fsd: float | None = None,
    p_xr: float | None = None,
    alpha_xr: float | None = None,
    diagnostic_condensate_scheme: str | None = None,
    adiabatic_lwc_rate: float | None = None,
    clubb_cf_override_strength: float | None = None,
    clubb_cf_override_floor: float | None = None,
) -> "CloudConfig":
    """Assemble a ``CloudConfig`` from the ``ExperimentConfig``-level cloud
    fields (``cloud_scheme`` + the optional ``cloud_rh_crit`` /
    ``cloud_q_c_diagnostic`` / ``cloud_conv_cloud_max`` overrides).

    Single source of truth so the radiation path (physics pipeline) and the
    ``clt`` diagnostic (diagnostics collector) build the SAME cloud fraction
    and can never drift as override knobs are added (issue #689 codex review).
    A ``None`` override falls back to the ``CloudConfig`` default (so an
    all-``None`` call is byte-identical to the defaults).
    """
    overrides: dict[str, float | str] = {}
    if rh_crit is not None:
        overrides["rh_crit"] = rh_crit
    if q_c_diagnostic is not None:
        overrides["q_c_diagnostic"] = q_c_diagnostic
    if conv_cloud_max is not None:
        overrides["conv_cloud_max"] = conv_cloud_max
    if conv_cloud_condensate is not None:
        overrides["conv_cloud_condensate"] = conv_cloud_condensate
    if cloud_inhomogeneity_factor is not None:
        overrides["cloud_inhomogeneity_factor"] = cloud_inhomogeneity_factor
    if cloud_optics_inhomogeneity is not None:
        overrides["cloud_optics_inhomogeneity"] = cloud_optics_inhomogeneity
    if cloud_fsd is not None:
        overrides["cloud_fsd"] = cloud_fsd
    if p_xr is not None:
        overrides["p_xr"] = p_xr
    if alpha_xr is not None:
        overrides["alpha_xr"] = alpha_xr
    if diagnostic_condensate_scheme is not None:
        overrides["diagnostic_condensate_scheme"] = diagnostic_condensate_scheme
    if adiabatic_lwc_rate is not None:
        overrides["adiabatic_lwc_rate"] = adiabatic_lwc_rate
    if clubb_cf_override_strength is not None:
        overrides["clubb_cf_override_strength"] = clubb_cf_override_strength
    if clubb_cf_override_floor is not None:
        overrides["clubb_cf_override_floor"] = clubb_cf_override_floor
    return CloudConfig(
        scheme=scheme, convective_cloud=convective_cloud, **overrides
    )
