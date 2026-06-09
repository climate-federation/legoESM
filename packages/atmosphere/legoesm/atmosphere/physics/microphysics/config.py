"""Configuration for atmospheric microphysics schemes.

Provides configuration NamedTuples for:
1. Kessler — warm-rain one-moment (refactored from physics/kessler.py)
2. Sundqvist — large-scale diagnostic condensation
3. Seifert-Beheng — two-moment warm rain
4. Morrison — double-moment ice+liquid
5. Thompson — hybrid moment with graupel
6. ML Emulator — Equinox MLP surrogate
7. P3 — Predicted Particle Properties single-category ice
8. Top-level MicrophysicsConfig that selects the active scheme.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water Substance.
- Sundqvist et al. (1989): Condensation and cloud parameterization studies.
- Seifert & Beheng (2001): A two-moment cloud microphysics scheme.
- Morrison et al. (2005): A new double-moment microphysics scheme.
- Thompson et al. (2008): Explicit forecasts of winter precipitation.
- Morrison & Milbrandt (2015): Parameterization of cloud microphysics
  based on the prediction of bulk ice particle properties. Part I.
  J. Atmos. Sci., 72, 287-311.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants


class KesslerConfig(NamedTuple):
    """Configuration for Kessler warm-rain microphysics."""
    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    accretion_coeff: float = 2.2                # Collection coefficient
    evaporation_coeff: float = 1.0              # Evaporation coefficient
    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    saturation_sharpness: float = 100.0         # Smooth switch sharpness


class SundqvistConfig(NamedTuple):
    """Configuration for Sundqvist large-scale condensation."""
    rh_crit: float = 0.8              # Critical relative humidity
    sigmoid_sharpness: float = 20.0   # Sharpness for smooth activation
    auto_rate: float = 1e-3           # Autoconversion rate c_0 [1/s]
    # Critical cloud water for autoconversion: P_auto = c_0·q_c·
    # (1 − exp(−(q_c/q_c,crit)²)) (Sundqvist 1989).  Suppresses
    # autoconversion below q_c,crit (drizzle forms only once cloud water
    # is large enough).  0 → the linear no-threshold limit.
    qc_crit: float = 5e-4             # [kg/kg]
    evap_coeff: float = 5e-4          # Sub-cloud evaporation coefficient


class SeifertBehengConfig(NamedTuple):
    """Configuration for Seifert-Beheng two-moment warm rain."""
    k_au: float = 6e2                # Autoconversion rate [1/(kg*s)]
    x_star: float = 2.6e-10          # Separation mass [kg]
    # ``N_c`` per-volume: see ``_warm_rain.effective_Nc`` Notes.
    Nc_0: float = 1e8                # Initial cloud droplet number [1/m³]
    k_ac: float = 5.25               # Accretion rate [m^3/(kg*s)]
    k_sc: float = 1e-3               # Self-collection rate [m^3/(kg*s)]
    D_eq: float = 1.1e-3             # Equilibrium breakup diameter [m]
    breakup_sharpness: float = 1e4   # Sigmoid sharpness for breakup [1/m]
    a_v_r: float = 130.0             # Rain fall speed coefficient a [m^(1-b)/s]
    b_v_r: float = 0.5               # Rain fall speed exponent b
    evap_coeff: float = 1.0          # Evaporation coefficient
    saturation_sharpness: float = 100.0  # Sigmoid sharpness for saturation [1/(kg/kg)]
    # Autoconversion onset sharpness — dimensionless steepness on the
    # normalised argument ``x_c / x_star − 1`` (iter-97 fix).  Default
    # of 10 gives transition over ~10 % of x_star around the threshold;
    # set higher for a sharper SB-style step, lower for a smoother
    # Kessler-like onset.  Distinct from ``saturation_sharpness`` which
    # is in units of kg/kg and would be 100× too steep here.
    autoconversion_sharpness: float = 10.0


class MorrisonConfig(NamedTuple):
    """Configuration for Morrison double-moment (ice+liquid)."""
    # Warm-rain autoconversion + accretion scheme:
    #   "kk2000" (default) = Khairoutdinov-Kogan 2000, the SAM M2005
    #     DEFAULT (IRAIN=0): PRC=1350·qc^2.47·(Nc[#/cm³])^-1.79,
    #     PRA=67·(qc·qr)^1.15. Faithful to the gSAM oracle.
    #   "seifert_beheng" = the legacy SB onset (k_au/x_star/onset sigmoid).
    warm_rain_scheme: str = "kk2000"
    predict_Nc: bool = False         # SAM M2005 dopredictNc. False (SAM DEFAULT) =
                                     # SPECIFIED constant droplet number Nc_0: the
                                     # size distribution uses Nc_0 and cloud number is
                                     # NOT evolved (dN_c/dt=0). True = prognostic Nc,
                                     # which needs SAM's droplet-activation source
                                     # (not yet ported) — leaving it False matches SAM
                                     # and avoids the sink-only Nc<0 drift.
    # Warm rain (Seifert-Beheng knobs; consumed only when
    # warm_rain_scheme="seifert_beheng")
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    # Rain self-collection + breakup:
    #   "sb2001" (default, faithful) = SAM NRAGG (Seifert-Beheng 2001,
    #     module_mp_graupel.f90:1980): NRAGG=−5.78·dum·q_r·N_r·ρ with dum=1 for
    #     mean drops < 300µm (self-collection) and dum=2−exp(2300·(1/LAMR−300µm))
    #     for larger drops (→ active breakup as dum<0). Uses the prognostic N_r.
    #     (legoESM's legacy k_sc=1e-3 self-collected ~5580× too WEAKLY.)
    #   "legacy" = the sigmoid-breakup form (k_sc, breakup_sharpness, D_eq).
    rain_selfcoll_scheme: str = "sb2001"
    rain_selfcoll_k: float = 5.78        # SAM/SB2001 self-collection coeff
    rain_breakup_d0: float = 300.0e-6    # SB2001 breakup onset mean size [m]
    rain_breakup_steepness: float = 2300.0  # SB2001 breakup exponent [1/m]
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    # Rain evaporation:
    #   "m2005" (default, faithful) = SAM PRE diffusion+ventilation evaporation
    #     of the rain PSD (module_mp_graupel.f90:1994): EPSR=2π·N0R·DV·[F1R/LAMR
    #     + F2R·CONS9·(ARN·ρ/μ)^½·SC^⅓·LAMR^(1−CONS34)], PRE=EPSR·(q_sat−q_v)/AB.
    #     Uses the prognostic N_r (double-moment rain). Ventilation accelerates
    #     evaporation of falling drops → faithful cold-pool cooling.
    #   "bulk" = the legacy evap_coeff·subsat·q_r^0.525 (number-blind).
    rain_evap_scheme: str = "m2005"
    rain_vent_f1: float = 0.78       # SAM F1R ventilation coefficient
    rain_vent_f2: float = 0.308      # SAM F2R ventilation coefficient
    saturation_sharpness: float = 100.0
    autoconversion_sharpness: float = 10.0  # See SB config — iter-97/99
    # Ice nucleation (Cooper 1986, SAM M2005 INUC=0)
    # Canonical Cooper curve: N_i = 0.005·exp(0.304·(T_f−T)) per LITRE
    # = 5·exp(...) per m³.  The base coefficient is therefore 5 /m³ (NOT
    # 5e3 = 5 /L, which over-nucleated by 1000× before iter-9).  Divided
    # by ρ to the per-mass ``N_i`` stored in HydrometeorState, capped at
    # ``N_i_nuc_max`` (SAM "limit to 500 L⁻¹").
    N_i0: float = 5.0                # Cooper base ice crystal number [1/m³]
    N_i_nuc_max: float = 5.0e5       # Cooper cap [1/m³] (= 500 /L)
    cooper_a: float = 0.304          # Cooper exponent
    cooper_T_act: float = 265.0      # Activation temperature [K]
    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    # Smooth supersaturation gate for nucleation (AD-regularised surrogate
    # for SAM's HARD gate (RH_liq≥0.999 & T≤265.15) OR RH_ice≥1.08).
    # Sharpnesses chosen so leakage is negligible: at 1 % RH below the
    # threshold σ(1000·−0.01)=σ(−10)≈4.5e-5; at 2 K above 265.15 K
    # σ(3·−2)=σ(−6)≈2.5e-3 (Codex iter-9: 200/1 leaked ~0.1–0.27).
    nuc_rh_sharpness: float = 1000.0    # 1/(RH units) — ramp ~0.1 % RH
    nuc_T_sharpness: float = 3.0        # 1/K — ramp ~0.3 K at 265.15 K
    ice_nuc_radius: float = 10.0e-6     # Initial nucleated crystal radius [m]
    #                                     (SAM MI0 = 4/3·π·ρ_ci·r³)
    # Homogeneous ice nucleation (Koop 2000 / Kärcher-Lohmann 2002) — OPT-IN.
    # SAM M2005 Cooper (INUC=0) carries ONLY primary/heterogeneous ice (≤500/L),
    # so NOTHING caps cirrus ice-supersaturation: the diffusional-growth rate
    # EPSI ∝ N_i^⅔·q_i^⅓ bootstraps slowly when fresh convective outflow floods
    # an upper-tropospheric level with vapour, leaving RH_ice far above 100%
    # (transiently >1000% in a violent small-domain RCE spin-up — the deposition
    # sink lags the convective source). Real cirrus homogeneous freezing of
    # aqueous haze bursts a HIGH crystal number once RH_ice exceeds the
    # homogeneous threshold S_hom(T); those crystals deposit the excess vapour
    # and pin RH_ice near S_hom (~1.5-1.6). This adds that missing process as a
    # supersaturation-gated ice-NUMBER (+seed-mass) source that feeds the
    # EXISTING M2005 deposition (so the vapour/ice/number/heat budgets + donor
    # clamps stay consistent automatically). Default OFF ⇒ byte-identical to the
    # SAM-faithful Cooper-only path; enable for RCE cirrus realism. AD-safe.
    #   S_hom(T) = koop_s_hom_a − koop_s_hom_b·T  [K-units], clipped to
    #   [koop_s_hom_min, koop_s_hom_max]: ≈1.64 at 185 K, ≈1.58 at 200 K,
    #   ≈1.44 at 235 K. This is the Ren & MacKenzie (2005, QJRMS 131:1585)
    #   ANALYTICAL LINEAR FIT to the Koop et al. (2000, Nature 406) water-
    #   activity homogeneous-freezing threshold for a representative critical
    #   nucleation rate; NOT Koop's full J(Δa_w) integral. Only valid in the
    #   cold cirrus regime, so activation is additionally gated on T below
    #   ``hom_freeze_T_max`` (homogeneous freezing of aqueous haze needs
    #   T ≲ −38 °C). The ``koop_s_hom_min`` floor only bites for T the cold
    #   gate has already switched off.
    homogeneous_ice_nucleation: bool = False
    koop_s_hom_a: float = 2.349         # Ren-MacKenzie 2005 intercept [-]
    koop_s_hom_b: float = 1.0 / 259.0   # Ren-MacKenzie 2005 slope [1/K]
    koop_s_hom_min: float = 1.4         # floor on S_hom [-]
    koop_s_hom_max: float = 1.7         # cap on S_hom [-]
    hom_freeze_T_max: float = 235.0     # max T for homogeneous freezing [K]
    hom_freeze_T_sharpness: float = 1.0  # cold-gate ramp [1/K] (~few-K width)
    hom_ice_nuc_sharpness: float = 200.0  # smooth RH_ice gate ramp [1/(RH unit)]
    hom_ice_nuc_N: float = 1.0e6        # homogeneous crystal number [1/m³]
    #                                     (~1 cm⁻³; Kärcher-Lohmann 2002 cirrus
    #                                     range 1e4-1e7 /m³); stored per-mass via
    #                                     /ρ. STABILITY: the boosted diffusional-
    #                                     growth EPSI·dt/ABI stays <1 (no explicit
    #                                     overshoot) for N up to ~1e7 /m³ at
    #                                     dt≤20 s; an ON-only cap backstops it.
    # Depositional growth + sublimation.
    #   "m2005" (default) = faithful SAM M2005 diffusional growth
    #     PRD = EPSI·(q_v−q_sat_i)/ABI with EPSI ∝ ρ·DV·N_i^⅔·q_i^⅓ and the
    #     ABI psychrometric (latent-heat) correction; SUBLIMATION when
    #     q_v < q_sat_i (donor-clamped). Tuned by the DIMENSIONLESS
    #     ``ice_deposition_efficiency`` (NOT ``dep_coeff`` — they have
    #     different units; keeping them separate avoids the iter-8 codex
    #     "two-meanings" config trap).
    #   "heuristic" = the legacy form dep_coeff·max(S_i,0)·q_i·N_i^⅓
    #     (deposition only, no sublimation; dep_coeff has units 1/s-ish).
    ice_deposition_scheme: str = "m2005"
    ice_deposition_efficiency: float = 1.0   # m2005 dimensionless multiplier
    rho_cloud_ice: float = 500.0     # SAM M2005 cloud-ice bulk density
    #                                  [kg/m³] (RHOI); ≠ solid-ice 917.
    dep_coeff: float = 1e-3          # legacy "heuristic" deposition coeff
    # Floor on q_i used inside the diffusional-growth term so freshly
    # nucleated particles (N_i > 0, q_i ≈ 0) can grow.  ~10 µm-sized
    # crystals at N_i ~ 5e3 m^-3 correspond to q_i ~ 1e-9 kg/kg.
    q_i_min_growth: float = 1e-9     # Minimum effective q_i for deposition [kg/kg]
    # Wegener-Bergeron-Findeisen (cloud water → ice in mixed phase):
    #   "emergent" (default) = SAM M2005 behaviour. Verified in the gSAM
    #     source: NO explicit Bergeron rate — the vapour budget
    #     (module_mp_graupel.f90:3690) is deposition/sublimation/nucleation
    #     ONLY; line 3924 "EQUILIBRIUM SS INCLUDING BERGERON EFFECT" shows
    #     the effect is carried by the deposition's equilibrium
    #     supersaturation, not a rate. WBF here emerges from ice deposition
    #     (PRD draws q_v toward ice saturation) + saturation adjustment
    #     (evaporates cloud water as q_v falls below liquid saturation),
    #     so glaciation is DEPOSITION-RATE-LIMITED (supercooled liquid
    #     persists). DEPENDS on ice_deposition_scheme="m2005": pairing
    #     "emergent" with the "heuristic" deposition is not faithful (the
    #     heuristic does not diffuse vapour to ice saturation properly).
    #     As of iter-14 the m2005 deposition GROWTH branch is no longer
    #     f_ice-gated (EPSI∝N_i^⅔ self-gates on ice), so WBF now fires across
    #     the full mixed phase incl. the WARM tail (265–273 K), matching SAM.
    #   "bergeron_heuristic" = the legacy explicit rate; forces glaciation
    #     far faster than the deposition physics supports (~50× at 260 K in
    #     a liquid-saturated test — magnitude is regime-dependent).
    # NOTE: the default is "emergent", a SCIENTIFICALLY MEANINGFUL change
    # from the legacy explicit Bergeron (more supercooled liquid, slower
    # mixed-phase glaciation) — not a neutral refactor.
    wbf_scheme: str = "emergent"
    bergeron_rate: float = 1e-3      # heuristic Bergeron conversion rate [1/s]
    T_center: float = 258.0          # Bergeron T window center [K]
    T_width: float = 10.0            # Bergeron T window width [K]
    # Riming
    rime_coeff: float = 1.0          # Riming collection efficiency
    # Rain freezing (Bigg 1953 immersion freezing of supercooled rain):
    #   "bigg" (default, faithful) = SAM MNUCCR/NNUCCR (module_mp_graupel.f90:
    #     3251-3257): rate ∝ (exp(AIMM·(T₀−T))−1)·N_r/LAMR^{6,3}. Supercooled
    #     rain freezes to snow (legoESM has no graupel — SAM freezes to
    #     graupel), releasing the latent heat of fusion. Critical for the
    #     phase + buoyancy of cold convective updrafts.
    #   "none" = no rain freezing (legacy — rain stayed liquid below 0°C).
    rain_freeze_scheme: str = "bigg"
    bigg_aimm: float = 0.66          # SAM AIMM [1/K] (Bigg immersion freezing)
    bigg_bimm: float = 100.0         # SAM BIMM (Bigg immersion freezing)
    # Homogeneous freezing of cloud water (SAM module_mp_graupel.f90:4661):
    # below ≈−40 °C all supercooled cloud water freezes instantly to cloud ice
    # (droplet number → ice number), releasing L_f. Smooth sigmoid threshold
    # for AD; the rain analog is handled by the steep Bigg rate.
    do_homogeneous_freezing: bool = True
    homogeneous_freeze_T: float = 233.15      # −40 °C threshold [K]
    # Sharp sigmoid (~0.5 K transition) to approximate SAM's hard T≤233.15
    # switch closely while staying AD-smooth (codex iter-25: 5/K was too soft —
    # half-froze at exactly −40 °C and leaked above). The huge N_i this can
    # inject (full droplet number) is bounded downstream by the LAMI clamp.
    homogeneous_freeze_sharpness: float = 20.0  # sigmoid sharpness [1/K]
    # Ice → snow autoconversion:
    #   "m2005_autoconv" (default, faithful) = SAM PRCI: depositional growth
    #     of the cloud-ice PSD across the snow-size threshold DCS converts ice
    #     to snow (module_mp_graupel.f90:3322-3326). Deposition-driven (only
    #     when ice-supersaturated); computed in the m2005 deposition block, so
    #     it REQUIRES ice_deposition_scheme="m2005" (else it is 0). The snow
    #     NUMBER source NPRCI is dropped (legoESM single-moment snow).
    #   "heuristic" = the legacy constant-rate agg_coeff·q_i·f_ice.
    ice_to_snow_scheme: str = "m2005_autoconv"
    ice_snow_d_auto: float = 250.0e-6   # SAM DCS [m] (clice_snow_Dauto)
    # Aggregation (heuristic ice→snow rate)
    agg_coeff: float = 1e-3          # Ice-to-snow aggregation rate [1/s]
    # Melting
    melt_rate: float = 5e-3          # Melting rate [1/s]
    melt_sharpness: float = 2.0      # Sigmoid sharpness near T_freeze
    # Ice sedimentation
    a_v_i: float = 50.0              # Ice fall speed coefficient [m^(1-b)/s]
    b_v_i: float = 0.25              # Ice fall speed exponent
    # Snow sedimentation
    a_v_s: float = 30.0              # Snow fall speed coefficient
    b_v_s: float = 0.3               # Snow fall speed exponent
    # === M2005 PSD mass-weighted fall speeds (iter-15 M5) ===
    # "m2005_psd" (default, faithful): rain + cloud ice fall speeds are the
    #   SAM mass-weighted moments of the PSD using the PROGNOSTIC N_r, N_i
    #   (module_mp_graupel.f90:1854-1865, 4230-4248):
    #     LAMR = (π·ρ_w·N_r/(ρ·q_r))^¼   (clamped lamr_min..lamr_max)
    #     UMR  = AR·Γ(4+BR)/6 · LAMR^−BR · (ρ_su/ρ)^0.54   (cap 9.1·dum)
    #     LAMI = (ρ_ci·π·N_i/q_i)^⅓       (clamped lami_min..lami_max)
    #     UMI  = AI·Γ(4+BI)/6 · LAMI^−BI · (ρ_su/ρ)^0.54   (cap 1.2·(ρ_su/ρ)^0.35)
    #   Snow stays single-moment bulk (no prognostic N_s — see the double-
    #   moment-snow gap, M2/M4), so snow uses the legacy q-power V_t below.
    # "bulk_qpower" = the legacy V_t = a_v·(q·ρ/ρ_sfc)^b_v for ALL species.
    fall_speed_scheme: str = "m2005_psd"
    fall_a_r: float = 841.99667      # SAM AR rain fall-speed coeff [m^(1-BR)/s]
    fall_b_r: float = 0.8            # SAM BR rain fall-speed exponent
    fall_a_i: float = 700.0          # SAM AI cloud-ice fall-speed coeff
    fall_b_i: float = 0.865          # gSAM BI = clice_fall_b (MK tune,
                                     # micro_params.f90:62; the M2005-ORIGINAL
                                     # 1.0 fell ~3x too slow → anvil cloud-ice
                                     # over-accumulated. gSAM v=0.03/0.24/1.78
                                     # vs M2005-orig 0.01/0.07/0.7 m/s).
    # Standard air density at 850 mb (SAM RHOSU = 85000/(R_d·273.15)); the
    # (ρ_su/ρ)^0.54 factor accelerates fall speeds in thin upper-trop air.
    rho_su: float = 8.5e4 / (constants.R_d * constants.T_freeze)
    lamr_min: float = 1.0 / 2800.0e-6   # SAM LAMMINR (slope floor) [1/m]
    lamr_max: float = 1.0 / 20.0e-6     # SAM LAMMAXR (slope cap) [1/m]
    lami_min: float = 1.0 / 600.0e-6    # SAM LAMMINI=1/(2·DCS+100µm), DCS=250µm
    lami_max: float = 1.0 / 1.0e-6      # SAM LAMMAXI [1/m]
    # SNOW PSD (double-moment, used only when N_s is prognostic). LAMS=
    # (π·ρ_sn·N_s/q_s)^⅓ (per-mass N_s), UMS=AS·Γ(4+BS)/6/LAMS^BS·(ρ_su/ρ)^0.54,
    # UNS=AS·Γ(1+BS)/LAMS^BS·… (SAM AS=11.72, BS=0.41, RHOSN=100).
    rho_snow: float = 100.0             # SAM RHOSN snow bulk density [kg/m³]
    fall_a_s: float = 11.72             # SAM AS snow fall-speed coeff
    fall_b_s: float = 0.41              # SAM BS snow fall-speed exponent
    lams_min: float = 1.0 / 2000.0e-6   # SAM LAMMINS [1/m]
    lams_max: float = 1.0 / 10.0e-6     # SAM LAMMAXS [1/m]
    # Snow vapor deposition/sublimation PRDS (double-moment snow only): EPSS=
    # 2π·N0S·ρ·DV·[F1S/LAMS²+F2S·CONS10·(ASN·ρ/μ)^½·SC^⅓·LAMS^(1−CONS35)],
    # PRDS=EPSS·(q_v−q_sat_i)/ABI (SAM F1S=0.86, F2S=0.28). Grows snow by vapor
    # deposition (anvil) and sublimates it in dry downdrafts (cooling).
    do_snow_deposition: bool = True
    snow_vent_f1: float = 0.86          # SAM F1S snow ventilation coefficient
    snow_vent_f2: float = 0.28          # SAM F2S snow ventilation coefficient
    # Snow riming of cloud water PSACWS (double-moment snow only): PSACWS=
    # Γ(BS+3)·π/4·ECI·ASN·q_c·ρ·N0S/LAMS^(BS+3) — PSD collection of supercooled
    # droplets by falling snow, freezing onto it (riming, +L_f). Replaces the
    # crude rime_coeff·q_s·q_c·f_ice.
    do_snow_riming: bool = True
    snow_collect_eff: float = 0.7       # SAM ECI snow-droplet collection eff.
    # Snow melting PSMLT (double-moment snow only): heat-balance-limited melting
    # melt=2π·N0S·KAP·(T−T0)₊/L_f·[F1S/LAMS²+F2S·ventilation], KAP=1.414e3·μ
    # (air thermal conductivity). Replaces the crude bulk melt_rate·q_s.
    do_snow_melting: bool = True
    # Snow self-aggregation NSAGG (double-moment snow only): Passarelli-1978 /
    # Reisner-1998. Falling flakes collide+merge ⇒ snow NUMBER sink (mass
    # conserved). NSAGG=CONS15·ASN·ρ^((2+BS)/3)·q_s^((2+BS)/3)·(N_s·ρ)^((4−BS)/3)/ρ
    # with CONS15=−1108·EII·π^((1−BS)/3)·ρ_sn^((−2−BS)/3)/(4·720).
    do_snow_aggregation: bool = True
    snow_aggregation_eii: float = 0.1   # SAM EII snow-snow collection eff.
    # Graupel (M4, iter-34): SAM freezes supercooled rain to GRAUPEL (dense
    # frozen drops), not snow. Single-moment graupel in slot [5] with a fixed
    # intercept N0G; SAM module_mp_graupel.f90: AG=19.3, BG=0.37, RHOG=400,
    # LAMMING/LAMMAXG slope limits. do_graupel=False ⇒ legacy frozen-rain→snow.
    do_graupel: bool = True
    rho_graupel: float = 400.0          # SAM RHOG graupel bulk density [kg/m³]
    fall_a_g: float = 19.3              # SAM AG graupel fall-speed coeff
    fall_b_g: float = 0.37              # SAM BG graupel fall-speed exponent
    n0_graupel: float = 4.0e6           # fixed intercept N0G [1/m⁴] (single-mom.)
    lamg_min: float = 1.0 / 2000.0e-6   # SAM LAMMING [1/m] = 500
    lamg_max: float = 1.0 / 20.0e-6     # SAM LAMMAXG [1/m] = 50000
    graupel_vent_f1: float = 0.86       # SAM F1S (graupel shares snow vent params)
    graupel_vent_f2: float = 0.28       # SAM F2S
    do_graupel_melting: bool = True     # PGMLT graupel→rain (heat-balance)
    # Graupel riming PSACWG (iter-35): graupel collects supercooled cloud water
    # (q_c→q_g + L_f). SAM PSACWG=CONS14·AGN·q_c·ρ·N0G/LAMG^(BG+3),
    # CONS14=Γ(BG+3)·π/4·ECI. The primary graupel GROWTH mechanism in updrafts.
    do_graupel_riming: bool = True
    graupel_collect_eff: float = 0.7    # SAM ECI graupel-droplet collection eff.
    # Graupel vapor deposition/sublimation PRDG (iter-36): the graupel analog of
    # snow PRDS. PRDG=EPSG·(q_v−q_sat_i)/ABI, EPSG=2π·N0G·DV·[F1S/LAMG²+F2S·vent].
    # Deposition grows graupel (+L_s); sublimation cools dry downdrafts (−L_s).
    do_graupel_deposition: bool = True
    # Graupel rain accretion PRACG (iter-37): graupel collects rain (two-PSD
    # gravitational collection). Cold branch (T<0): rain→graupel + L_f. SAM
    # PRACG=CONS41·VDIFF·ρ·N0RR·N0G/LAMR³·[5/(LAMR³·LAMG)+2/(LAMR²·LAMG²)+
    # 0.5/(LAMR·LAMG³)], CONS41=π²·ECR·ρ_w. The warm-branch graupel→rain shedding
    # is deferred (partially captured by PGMLT melting).
    do_graupel_rain_accretion: bool = True
    graupel_rain_collect_eff: float = 1.0   # SAM ECR rain-graupel collection eff.
    # Snow→graupel conversion PGSACW (iter-38): heavily-rimed snow densifies to
    # graupel (Rutledge-Hobbs 1984 / Reisner 1998). PGSACW=min(PSACWS, CONS17·dt·
    # N0S·q_c²·ASN²/(ρ·LAMS^(2BS+2))), CONS17=3·ρ_su·π·ECI²·Γ(2BS+2)/(ρ_g−ρ_sn).
    # Gated on q_s≥0.1 g/kg AND q_c≥0.5 g/kg. Snow number sink NSCNG via MG0.
    do_snow_to_graupel: bool = True
    graupel_embryo_mass: float = 1.6e-10    # SAM MG0 graupel embryo mass [kg]


class ThompsonConfig(NamedTuple):
    """Configuration for Thompson hybrid-moment microphysics."""
    # All Morrison params
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    saturation_sharpness: float = 100.0
    autoconversion_sharpness: float = 10.0  # See SB config — iter-97/99
    N_i0: float = 5e3
    cooper_a: float = 0.304
    cooper_T_act: float = 265.0
    ice_sigmoid_sharpness: float = 5.0
    dep_coeff: float = 1e-3
    q_i_min_growth: float = 1e-9
    bergeron_rate: float = 1e-3
    T_center: float = 258.0
    T_width: float = 10.0
    rime_coeff: float = 1.0
    agg_coeff: float = 1e-3
    melt_rate: float = 5e-3
    melt_sharpness: float = 2.0
    a_v_i: float = 50.0
    b_v_i: float = 0.25
    a_v_s: float = 30.0
    b_v_s: float = 0.3
    # Graupel
    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
    graupel_sharpness: float = 1e4           # Sigmoid sharpness
    a_v_g: float = 80.0                      # Graupel fall speed coefficient
    b_v_g: float = 0.4                       # Graupel fall speed exponent
    # Gamma distribution shape
    mu_c: float = 3.0                        # Cloud droplet shape parameter
    mu_r: float = 1.0                        # Rain drop shape parameter


class P3Config(NamedTuple):
    """Configuration for P3 (Predicted Particle Properties) microphysics.

    Single ice category with predicted rime mass (q_rim) and rime volume
    (B_rim). Liquid phase uses Seifert-Beheng warm-rain helpers.

    Ice particle properties (fall speed, density) are diagnosed from
    (q_i, N_i, q_rim, B_rim) rather than assumed from a fixed habit.

    Slot reuse in HydrometeorState
    --------------------------------
    q_s → q_rim  [kg/kg]        rime mass mixing ratio
    q_g → B_rim  [m³/kg_air]    rime volume per unit air mass

    The rime density rho_rim = q_rim / B_rim [kg/m³] spans the full
    range from unrimed aggregates (~50 kg/m³) to dense graupel
    (~900 kg/m³).
    """
    # --- Warm rain (Seifert-Beheng liquid phase) ---
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    saturation_sharpness: float = 100.0
    autoconversion_sharpness: float = 10.0
    # --- Ice nucleation (Cooper 1986) ---
    N_i0: float = 5e3               # Cooper base ice crystal number [1/m³]
    cooper_a: float = 0.304         # Cooper exponent
    cooper_T_act: float = 265.0     # Activation temperature [K]
    ice_sigmoid_sharpness: float = 5.0
    # --- Ice depositional growth ---
    dep_coeff: float = 1e-3
    q_i_min_growth: float = 1e-9    # Minimum effective q_i for deposition [kg/kg]
    # --- Cloud riming (ice collects cloud droplets) ---
    rime_coeff: float = 0.5         # Collection efficiency E_ri [-]
    # --- Rain riming (ice collects rain drops, freezes) ---
    rain_rime_coeff: float = 0.1    # Collection efficiency E_rr [-]
    # --- Self-collection / aggregation (N_i reduction) ---
    agg_coeff: float = 1e-3         # Aggregation rate [1/s]
    # --- Melting ---
    melt_rate: float = 5e-3         # Melting rate [1/s]
    melt_sharpness: float = 2.0     # Sigmoid sharpness near T_freeze [1/K]
    # --- P3 fall speed: V_t = a_v_i * (q_i*rho_ratio)^b_v_i * density_factor ---
    a_v_i: float = 40.0             # Base fall speed coefficient
    b_v_i: float = 0.3              # Fall speed exponent
    # --- Rime density limits [kg/m³] ---
    rho_rim_min: float = 50.0       # Minimum rime density (unrimed aggregates)
    rho_rim_max: float = 900.0      # Maximum rime density (dense graupel)
    # --- Fall speed density enhancement ---
    # V_t *= (rho_rim / rho_ice_ref)^c_rim_fallspeed
    c_rim_fallspeed: float = 0.4    # Density enhancement exponent [-]
    rho_ice_ref: float = 500.0      # Reference rime density for scaling [kg/m³]
    # --- Accreted rime density (for dB_rim/dt from riming) ---
    rho_rim_accrete: float = 400.0  # Density of newly accreted rime [kg/m³]
    # --- Bulk ice density (for B_rim from nucleation) ---
    rho_ice: float = constants.rho_ice  # Solid ice density [kg/m³]


class MicrophysicsMLEmulatorConfig(NamedTuple):
    """Configuration for ML microphysics emulator (Equinox MLP)."""
    n_input: int = 9
    n_hidden: int = 128
    n_layers: int = 3
    n_output: int = 7
    seed: int = 0
    use_residual: bool = True
    norm_T: float = 300.0       # Temperature scale [K] for input normalization
    norm_q_factor: float = 1e3  # q_v / q_c / q_r / q_i scale
    norm_rho: float = 1.2       # Air density scale [kg/m^3]
    norm_dz: float = 1000.0     # Layer thickness scale [m]
    norm_dt: float = 3600.0     # Time-step scale [s]


class MicrophysicsConfig(NamedTuple):
    """Top-level microphysics configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active scheme: "kessler", "sundqvist", "seifert_beheng",
        "morrison", "thompson", "p3", "ml_emulator", or "none".
    kessler : KesslerConfig
    sundqvist : SundqvistConfig
    seifert_beheng : SeifertBehengConfig
    morrison : MorrisonConfig
    thompson : ThompsonConfig
    p3 : P3Config
    ml_emulator : MicrophysicsMLEmulatorConfig
    """
    scheme: str = "none"
    kessler: KesslerConfig = KesslerConfig()
    sundqvist: SundqvistConfig = SundqvistConfig()
    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
    morrison: MorrisonConfig = MorrisonConfig()
    thompson: ThompsonConfig = ThompsonConfig()
    p3: P3Config = P3Config()
    ml_emulator: MicrophysicsMLEmulatorConfig = MicrophysicsMLEmulatorConfig()
