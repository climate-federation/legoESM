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
    RH_crit: float = 0.8              # Critical relative humidity
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
    # Warm rain (same as SB)
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
    # Ice nucleation (Cooper 1986)
    # Cooper (1986) per-volume base number; nucleation divides by rho
    # to produce the per-mass ``N_i`` stored in HydrometeorState.
    N_i0: float = 5e3               # Cooper base ice crystal number [1/m³]
    cooper_a: float = 0.304          # Cooper exponent
    cooper_T_act: float = 265.0      # Activation temperature [K]
    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    # Depositional growth
    dep_coeff: float = 1e-3          # Deposition growth coefficient
    # Floor on q_i used inside the diffusional-growth term so freshly
    # nucleated particles (N_i > 0, q_i ≈ 0) can grow.  ~10 µm-sized
    # crystals at N_i ~ 5e3 m^-3 correspond to q_i ~ 1e-9 kg/kg.
    q_i_min_growth: float = 1e-9     # Minimum effective q_i for deposition [kg/kg]
    # Bergeron
    bergeron_rate: float = 1e-3      # Bergeron conversion rate [1/s]
    T_center: float = 258.0          # Bergeron T window center [K]
    T_width: float = 10.0            # Bergeron T window width [K]
    # Riming
    rime_coeff: float = 1.0          # Riming collection efficiency
    # Aggregation
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


class MLEmulatorConfig(NamedTuple):
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
    ml_emulator : MLEmulatorConfig
    """
    scheme: str = "none"
    kessler: KesslerConfig = KesslerConfig()
    sundqvist: SundqvistConfig = SundqvistConfig()
    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
    morrison: MorrisonConfig = MorrisonConfig()
    thompson: ThompsonConfig = ThompsonConfig()
    p3: P3Config = P3Config()
    ml_emulator: MLEmulatorConfig = MLEmulatorConfig()
