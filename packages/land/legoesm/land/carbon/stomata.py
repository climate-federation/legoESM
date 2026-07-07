"""Plant physiology: Farquhar photosynthesis and stomatal conductance.

Models implemented:
- Farquhar et al. (1980): C3 biochemical photosynthesis.
- Ball, Woodrow & Berry (1987): empirical stomatal conductance.
- Medlyn et al. (2011): optimal stomatal conductance (USO).
- Jarvis (1976): multiplicative stomatal conductance (CO2-independent).

When the carbon cycle is active, the Farquhar model replaces the
light-use-efficiency GPP and is coupled to Ball-Berry or Medlyn stomatal
conductance.  When the carbon cycle is off, the Jarvis model provides
stomatal control on evapotranspiration without requiring CO2 information.

Limitation: C3 biochemistry only (Farquhar 1980).  The C4 PFTs in the
CLM5 table (``c4_grass``, ``crop_c4``) are approximated with C3 kinetics --
no Collatz (1992) C4 biochemical path is implemented.  Consequently the
distinctive C4 CO2 sensitivity and near-zero CO2 compensation point are
not represented for those PFTs.

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Farquhar, G. D., von Caemmerer, S. & Berry, J. A. (1980): A biochemical
  model of photosynthetic CO2 assimilation in leaves of C3 species.
  Planta, 149, 78-90.
- Ball, J. T., Woodrow, I. E. & Berry, J. A. (1987): A model predicting
  stomatal conductance and its contribution to the control of photosynthesis
  under different environmental conditions. Progress in Photosynthesis
  Research, 4, 221-224.
- Medlyn, B. E. et al. (2011): Reconciling the optimal and empirical
  approaches to modelling stomatal conductance. Global Change Biology,
  17, 2134-2144.
- Jarvis, P. G. (1976): The interpretation of the variations in leaf water
  potential and stomatal conductance found in canopies in the field.
  Phil. Trans. Roy. Soc. London B, 273, 593-610.
- Bernacchi, C. J. et al. (2001): Improved temperature response functions
  for models of Rubisco-limited photosynthesis. Plant, Cell & Environment,
  24, 253-259.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.land.canopy.sif import SIFConfig
from legoesm.thermo import saturation_vapor_pressure

# Fixed Farquhar / gas-exchange constants (not tunable).
_FARQUHAR_WJ_GAMMA_COEFF = 8.0      # 4*Ci + 8*Gamma* electron-transport denominator
_DIFFUSIVITY_RATIO_H2O_CO2 = 1.6    # H2O:CO2 stomatal diffusivity ratio
_CI_CA_INIT_RATIO = 0.7             # initial intercellular:ambient CO2 guess


# =====================================================================
# Constants
# =====================================================================

# Universal gas constant lives in legoesm.constants (CODATA 2018, exact).
_R_GAS = constants.R_universal     # [J/(mol·K)]
_T_REF = 298.15    # Reference temperature 25 deg C [K]
_PAR_FRAC = 0.48   # Fraction of shortwave that is PAR
_PAR_CONV = 4.6    # umol photons per J of PAR
_MC = 12.0e-6      # g C per umol CO2

# A-gs coupling fixed-point iteration COUNT. A loop count is never a config /
# trainable leaf (loop-counts-never-trainable doctrine) — it stays a module
# constant so it can never reach the trainable collector.
_N_AGS_ITER_DEFAULT = 5


# =====================================================================
# Configuration
# =====================================================================

__param_spec__ = {
    "StomataConfig": {
        "scheme_key": "land.stomata",
        "excluded": {
            "Gamma_star25": "CO2 compensation point at 25C (Bernacchi 2001 fixed) [umol/mol]",
            "Ha_Gamma": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Ha_J": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Ha_Kc": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Ha_Ko": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Ha_Rd": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Ha_Vc": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Hd_J": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "Kc25": "kinetic constant at 25C (Bernacchi 2001 fixed) [umol/mol]",
            "Ko25": "kinetic constant at 25C (Bernacchi 2001 fixed) [umol/mol]",
            "O2_conc": "atmospheric O2 (environmental constant) [umol/mol]",
            "S_J": "Arrhenius activation/entropy energy (Bernacchi 2001 fixed) [J/mol]",
            "co_limitation_eps": "numerics: smooth-min co-limitation width",
        },
        "params": {
            "beta_soil_min": {"units": "1", "bounds": (0.001, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CLM soil-water stress floor", "shape": None},
            "f_VPD_min": {"units": "1", "bounds": (0.001, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CLM VPD stress floor", "shape": None},
            "J_max25": {"units": "1", "bounds": (39.6, 360.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "K_PAR": {"units": "1", "bounds": (66.0, 600.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "Rd25": {"units": "1", "bounds": (0.495, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "T_opt_jarvis_C": {"units": "degC", "bounds": (8.25, 75.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None, "legacy_name": "T_opt_jarvis"},
            "T_range_jarvis_C": {"units": "degC", "bounds": (6.6, 60.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None, "legacy_name": "T_range_jarvis"},
            "Vc_max25": {"units": "1", "bounds": (19.8, 180.0), "tunable_tier": 1, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "a_vpd": {"units": "1", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "alpha_q": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "g0": {"units": "1", "bounds": (0.0033, 0.03), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "g1_bb": {"units": "1", "bounds": (2.97, 27.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "g1_med": {"units": "1", "bounds": (1.32, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "gs_max": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "gs_ref": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "k_ext": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
            "theta_j": {"units": "1", "bounds": (0.297, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Farquhar 1980 / Medlyn 2011 / Bernacchi 2001", "shape": None},
        },
    },
}


class StomataConfig(NamedTuple):
    """Stomatal conductance and plant physiology configuration.

    enabled=False (default): use simple bucket beta (backward compatible).
    enabled=True + carbon active: Farquhar + Ball-Berry/Medlyn.
    enabled=True + carbon off: Jarvis multiplicative model.
    """
    enabled: bool = False

    # --- Coupled stomatal model selection ---
    stomata_model: str = "ball_berry"   # "ball_berry" or "medlyn"

    # --- Farquhar C3 photosynthesis (Farquhar et al. 1980) ---
    Vc_max25: float = 60.0       # Max carboxylation at 25 C [umol/m2/s]
    J_max25: float = 120.0       # Max electron transport at 25 C [umol/m2/s]
    Rd25: float = 1.5            # Dark respiration at 25 C [umol/m2/s]
    alpha_q: float = 0.3         # Quantum yield [mol e-/mol photon]
    theta_j: float = 0.9         # J light-response curvature [-]

    # --- Arrhenius activation energies [J/mol] ---
    Ha_Vc: float = 65330.0
    Ha_J: float = 43540.0
    Hd_J: float = 152040.0       # Deactivation energy for J_max
    S_J: float = 495.0           # Entropy term for J_max [J/mol/K]
    Ha_Rd: float = 46390.0

    # --- Kinetic constants at 25 C (Bernacchi et al. 2001) ---
    Kc25: float = 404.9          # Michaelis for CO2 [umol/mol]
    Ko25: float = 278400.0       # Michaelis for O2 [umol/mol]
    Gamma_star25: float = 42.75  # CO2 compensation point [umol/mol]
    Ha_Kc: float = 79430.0
    Ha_Ko: float = 36380.0
    Ha_Gamma: float = 37830.0

    # --- Ball-Berry (Ball et al. 1987) ---
    g0: float = 0.01             # Residual conductance [mol/m2/s]
    g1_bb: float = 9.0           # Ball-Berry slope [-]

    # --- Medlyn (Medlyn et al. 2011) ---
    g1_med: float = 4.0          # Medlyn slope [kPa^0.5]

    # --- Jarvis (Jarvis 1976) ---
    gs_max: float = 0.3          # Maximum conductance [mol/m2/s]
    K_PAR: float = 200.0         # PAR half-saturation [W/m2]
    T_opt_jarvis_C: float = 25.0   # Optimal temperature [deg C]
    T_range_jarvis_C: float = 20.0 # Temperature range [deg C]
    a_vpd: float = 0.05          # VPD sensitivity [1/hPa]

    # --- Solver ---
    # A-gs fixed-point iteration count. An int field, so it is never spec-
    # eligible / trainable (the loop-count-never-trainable guarantee holds
    # structurally); the default references the module constant. Kept as a
    # config field for backward compatibility + per-call convergence tuning.
    n_iter_ags: int = _N_AGS_ITER_DEFAULT

    # --- Other ---
    O2_conc: float = 209000.0    # Atmospheric O2 [umol/mol]
    k_ext: float = 0.5           # Beer-law extinction coefficient [-]
    gs_ref: float = 0.3          # Reference max gs for beta [mol/m2/s]

    # --- Soil moisture / VPD stress floors (CLM-style; tunable) ---
    beta_soil_min: float = 0.01  # Lower bound on soil water stress factor
    f_VPD_min: float = 0.01      # Lower bound on VPD stress factor

    # --- Numerics ---
    co_limitation_eps: float = 0.1  # Smooth-min width for Wc/Wj co-limitation

    # --- Optional solar-induced fluorescence (SIF) diagnostic ---
    # None (default) disables it; a SIFConfig enables the passive big-leaf SIF
    # output on SurfaceFluxOutput.sif (coupled Farquhar path only — the Jarvis
    # fallback has no Ci/An to invert). Static config leaf, never traced.
    sif: SIFConfig | None = None


# =====================================================================
# Temperature response functions
# =====================================================================

def arrhenius(
    param25: float | jnp.ndarray,
    Ha: float,
    T: jnp.ndarray,
) -> jnp.ndarray:
    """Arrhenius temperature dependence relative to 25 deg C."""
    return param25 * jnp.exp(Ha * (T - _T_REF) / (_T_REF * _R_GAS * T))


def peaked_arrhenius(
    param25: float | jnp.ndarray,
    Ha: float,
    Hd: float,
    S: float,
    T: jnp.ndarray,
) -> jnp.ndarray:
    """Peaked Arrhenius for parameters that decline at high T."""
    f_T = jnp.exp(Ha * (T - _T_REF) / (_T_REF * _R_GAS * T))
    num = 1.0 + jnp.exp((S * _T_REF - Hd) / (_R_GAS * _T_REF))
    den = 1.0 + jnp.exp((S * T - Hd) / (_R_GAS * T))
    return param25 * f_T * num / den


# =====================================================================
# Farquhar C3 photosynthesis
# =====================================================================

def farquhar_photosynthesis(
    Ci: jnp.ndarray,
    APAR_umol: jnp.ndarray,
    T_leaf: jnp.ndarray,
    config: StomataConfig,
    beta_soil: jnp.ndarray | None = None,
    canopy_scaling: jnp.ndarray | float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Farquhar et al. (1980) C3 photosynthesis model.

    C3 biochemistry only.  C4 PFTs (``c4_grass``, ``crop_c4``) are run
    through this same C3 kinetics as a documented approximation -- there is
    no Collatz C4 path, so their CO2 sensitivity / compensation point are
    not represented.

    Parameters
    ----------
    Ci : Intercellular CO2 concentration [umol/mol].
    APAR_umol : Absorbed PAR [umol photons/m2/s].
    T_leaf : Leaf temperature [K].
    config : StomataConfig.
    beta_soil : Soil moisture stress factor [0-1], optional.
        When provided, scales Vc_max (CLM approach).
    canopy_scaling : Big-leaf canopy integral ``L_c = (1-exp(-k·LAI))/k``
        [m2 leaf / m2 ground], default 1.0.  Multiplies the photosynthetic
        *capacity* terms (Vc_max, J_max, Rd) to convert leaf-level rates to
        canopy-level rates PER UNIT GROUND AREA.  ``APAR_umol`` is already the
        canopy-absorbed PAR (``fAPAR·PAR``, per ground area), so scaling the
        capacities keeps the Rubisco-limited (Wc) and light-limited (Wj)
        branches consistently canopy-scale — without it, Wc caps at a single
        leaf's rate while Wj is driven by the whole canopy's light, so a dense
        canopy (LAI≫1) is silently limited to leaf-level assimilation and GPP
        is under-estimated ~L_c-fold (Sellers 1992 big-leaf; Bonan 2011).
        At LAI→0, ``L_c→LAI→0`` so a leafless column assimilates nothing.

    Returns
    -------
    A_net : Net assimilation rate [umol CO2/m2/s] (per unit ground area).
    A_gross : Gross assimilation (before dark respiration) [umol CO2/m2/s].
    """
    # Temperature-dependent parameters (leaf-level capacities/kinetics).
    Vc_max = arrhenius(config.Vc_max25, config.Ha_Vc, T_leaf)
    J_max = peaked_arrhenius(
        config.J_max25, config.Ha_J, config.Hd_J, config.S_J, T_leaf)
    Rd = arrhenius(config.Rd25, config.Ha_Rd, T_leaf)
    Kc = arrhenius(config.Kc25, config.Ha_Kc, T_leaf)
    Ko = arrhenius(config.Ko25, config.Ha_Ko, T_leaf)
    Gamma_star = arrhenius(config.Gamma_star25, config.Ha_Gamma, T_leaf)

    # Big-leaf canopy scaling: convert the leaf-level CAPACITY terms to
    # canopy-level per-ground-area rates.  Kc/Ko/Gamma_star are intensive
    # (concentrations), so they are NOT scaled.
    Vc_max = Vc_max * canopy_scaling
    J_max = J_max * canopy_scaling
    Rd = Rd * canopy_scaling

    # Soil moisture stress on Vc_max (CLM / Bonan et al. 2011)
    if beta_soil is not None:
        Vc_max = Vc_max * jnp.clip(beta_soil, config.beta_soil_min, 1.0)

    # Rubisco-limited rate
    Ci_safe = jnp.maximum(Ci, Gamma_star + 1.0)
    Wc = Vc_max * (Ci_safe - Gamma_star) / (
        Ci_safe + Kc * (1.0 + config.O2_conc / Ko))

    # Electron transport rate J (quadratic light response)
    a = config.theta_j
    b = -(config.alpha_q * APAR_umol + J_max)
    c_coeff = config.alpha_q * APAR_umol * J_max
    # Floor the discriminant at a tiny positive rather than 0: with curvature
    # theta_j -> 1 (some Farquhar formulations use 1.0) the co-limitation
    # discriminant b^2 - 4ac can reach 0, where sqrt'(0) = inf gives a NaN
    # gradient (d J / d APAR).  For the default theta_j = 0.9 the discriminant
    # is O(1e3), so the 1e-12 floor never changes the forward value.
    disc = jnp.maximum(b ** 2 - 4.0 * a * c_coeff, 1e-12)
    J = (-b - jnp.sqrt(disc)) / (2.0 * a + 1e-20)

    # RuBP-regeneration-limited rate
    Wj = J * (Ci_safe - Gamma_star) / (4.0 * Ci_safe + _FARQUHAR_WJ_GAMMA_COEFF * Gamma_star)

    # Smooth minimum (differentiable)
    _eps = config.co_limitation_eps
    A_gross = 0.5 * (Wc + Wj - jnp.sqrt((Wc - Wj) ** 2 + _eps ** 2))
    A_gross = jnp.maximum(A_gross, 0.0)

    A_net = A_gross - Rd
    return A_net, A_gross


# =====================================================================
# Stomatal conductance models
# =====================================================================

def ball_berry_gs(
    A: jnp.ndarray,
    RH: jnp.ndarray,
    Cs: jnp.ndarray,
    config: StomataConfig,
) -> jnp.ndarray:
    """Ball-Berry (1987) stomatal conductance [mol H2O/m2/s].

    gs = g0 + g1 * max(A, 0) * RH / Cs
    """
    A_pos = jnp.maximum(A, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(
        config.g0 + config.g1_bb * A_pos * RH / Cs_safe,
        config.g0,
    )


def medlyn_gs(
    A: jnp.ndarray,
    VPD_kPa: jnp.ndarray,
    Cs: jnp.ndarray,
    config: StomataConfig,
) -> jnp.ndarray:
    """Medlyn et al. (2011) optimal stomatal conductance [mol H2O/m2/s].

    gs = g0 + 1.6 * (1 + g1 / sqrt(VPD)) * max(A, 0) / Cs
    """
    A_pos = jnp.maximum(A, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    VPD_safe = jnp.maximum(VPD_kPa, 0.05)  # coeff-ok: VPD floor [kPa]
    return jnp.maximum(
        config.g0 + _DIFFUSIVITY_RATIO_H2O_CO2 * (1.0 + config.g1_med / jnp.sqrt(VPD_safe))
        * A_pos / Cs_safe,
        config.g0,
    )


def jarvis_gs(
    T: jnp.ndarray,
    sw_down: jnp.ndarray,
    q_air: jnp.ndarray,
    p_surface: jnp.ndarray,
    beta_soil: jnp.ndarray,
    config: StomataConfig,
) -> jnp.ndarray:
    """Jarvis (1976) multiplicative stomatal conductance [mol H2O/m2/s].

    gs = gs_max * f(PAR) * f(T) * f(VPD) * f(soil moisture)

    Independent of CO2 — used as fallback when the carbon cycle is off.
    """
    # PAR response (hyperbolic saturation)
    PAR = _PAR_FRAC * sw_down
    f_PAR = PAR / (PAR + config.K_PAR + 1e-10)

    T_C = T - constants.T_freeze
    dT = (T_C - config.T_opt_jarvis_C) / config.T_range_jarvis_C
    f_T = jnp.maximum(1.0 - dT ** 2, 0.0)

    # VPD response (linear decrease).
    # ``q_air`` is specific humidity (kg vapour / kg moist air), so
    # ``e_air = q · p / (ε + (1 − ε) · q)`` is the correct formula.
    # The earlier ``q · p / (ε + q)`` is the *mixing-ratio* form
    # (kg vapour / kg dry air) and biases e_air by ~1% for typical
    # tropical q ≈ 0.02 — small but propagates into VPD-driven
    # stomatal closure.  Audit cycle 2026-05-05 finding #24.
    e_sat = saturation_vapor_pressure(T)
    e_air = q_air * p_surface / (constants.epsilon + (1.0 - constants.epsilon) * q_air)
    VPD_hPa = jnp.maximum(e_sat - e_air, 0.0) / 100.0
    f_VPD = jnp.clip(1.0 - config.a_vpd * VPD_hPa, config.f_VPD_min, 1.0)

    # Soil moisture
    f_soil = jnp.clip(beta_soil, 0.0, 1.0)

    return config.gs_max * f_PAR * f_T * f_VPD * f_soil


# =====================================================================
# Coupled Farquhar-stomata solver
# =====================================================================

class CoupledLeafState(NamedTuple):
    """Converged leaf state from the coupled Farquhar-stomata solve.

    Carries everything the SIF diagnostic needs (``A_net``, ``Ci``,
    ``APAR_umol``, ``gamma_star``) alongside the ``(gs, gpp)`` the beta / ET
    coupling consumes.  All fields share the ``T_leaf`` shape.
    """

    gs: jnp.ndarray          # stomatal conductance [mol H2O/m2/s]
    gpp: jnp.ndarray         # gross primary production [gC/m2/s]
    A_net: jnp.ndarray       # net assimilation [umol CO2/m2/s]
    Ci: jnp.ndarray          # intercellular CO2 [umol/mol]
    APAR_umol: jnp.ndarray   # absorbed PAR [umol photons/m2/s]
    gamma_star: jnp.ndarray  # CO2 compensation point Gamma* [umol/mol]


def solve_coupled_farquhar_ci(
    T_leaf: jnp.ndarray,
    sw_down: jnp.ndarray,
    co2_ppmv: jnp.ndarray | float,
    q_air: jnp.ndarray,
    p_surface: jnp.ndarray,
    LAI: jnp.ndarray,
    beta_soil: jnp.ndarray,
    config: StomataConfig,
) -> CoupledLeafState:
    """Solve the coupled Farquhar-stomata system; return the converged leaf state.

    Shared core of :func:`coupled_farquhar_stomata` (which returns just
    ``(gs, gpp)``) and the SIF diagnostic (``canopy/sif.py``, which also needs
    ``A_net``, ``Ci``, ``APAR`` and ``Gamma*``).  Numerics are identical to the
    historical ``coupled_farquhar_stomata`` body — see that function for the
    parameter documentation.
    """
    if config.stomata_model not in ("ball_berry", "medlyn"):
        raise ValueError(
            f"Unknown stomata_model {config.stomata_model!r}; "
            "expected one of: 'ball_berry', 'medlyn'."
        )

    # PAR in umol photons/m2/s
    PAR_umol = _PAR_FRAC * sw_down * _PAR_CONV

    # Absorbed PAR (Beer's law, per unit ground area)
    fAPAR = 1.0 - jnp.exp(-config.k_ext * LAI)
    APAR_umol = fAPAR * PAR_umol

    # Big-leaf canopy scaling for the photosynthetic capacities:
    #   L_c = (1 - exp(-k·LAI)) / k = fAPAR / k   [m2 leaf / m2 ground].
    # Converts leaf-level Vc_max/J_max/Rd to canopy-level per-ground-area
    # rates so the Rubisco- and light-limited branches are both canopy-scale
    # (see farquhar_photosynthesis docstring).  L_c -> LAI as LAI -> 0.
    canopy_scaling = fAPAR / jnp.maximum(config.k_ext, 1e-6)

    # Atmospheric CO2
    Ca = jnp.broadcast_to(
        jnp.asarray(co2_ppmv, dtype=T_leaf.dtype), T_leaf.shape)

    # Vapour pressure deficit and relative humidity.  ``q_air`` is
    # specific humidity, so use ``e = q · p / (ε + (1 − ε) · q)``;
    # see VPD comment in jarvis_gs above.  Audit finding #24.
    e_sat = saturation_vapor_pressure(T_leaf)
    e_air = q_air * p_surface / (constants.epsilon + (1.0 - constants.epsilon) * q_air)
    VPD_kPa = jnp.maximum(e_sat - e_air, 0.0) / 1000.0
    RH = jnp.clip(e_air / jnp.maximum(e_sat, 1.0), 0.0, 1.0)

    # Initial guess for Ci (typical C3 ratio)
    Ci = _CI_CA_INIT_RATIO * Ca

    # Fixed-point iteration (unrolled for JIT compatibility)
    for _ in range(config.n_iter_ags):
        A_net, _ = farquhar_photosynthesis(
            Ci, APAR_umol, T_leaf, config, beta_soil, canopy_scaling)

        if config.stomata_model == "medlyn":
            gs = medlyn_gs(A_net, VPD_kPa, Ca, config)
        else:
            gs = ball_berry_gs(A_net, RH, Ca, config)

        # Update Ci via stomatal diffusion (1.6 = H2O/CO2 ratio)
        gs_safe = jnp.maximum(gs, config.g0)
        A_pos = jnp.maximum(A_net, 0.0)
        Ci = Ca - _DIFFUSIVITY_RATIO_H2O_CO2 * A_pos / gs_safe
        Ci = jnp.clip(Ci, 1.0, Ca)

    # Final evaluation
    A_net, A_gross = farquhar_photosynthesis(
        Ci, APAR_umol, T_leaf, config, beta_soil, canopy_scaling)

    if config.stomata_model == "medlyn":
        gs = medlyn_gs(A_net, VPD_kPa, Ca, config)
    else:
        gs = ball_berry_gs(A_net, RH, Ca, config)

    # GPP: A_gross [umol CO2/m2/s] -> gC/m2/s
    gpp = jnp.maximum(A_gross, 0.0) * _MC

    # Gamma* for the SIF electron-transport inversion — same Arrhenius the
    # Farquhar rates use (Bernacchi 2001), computed here so the two never drift.
    gamma_star = arrhenius(config.Gamma_star25, config.Ha_Gamma, T_leaf)

    return CoupledLeafState(
        gs=gs, gpp=gpp, A_net=A_net, Ci=Ci,
        APAR_umol=APAR_umol, gamma_star=gamma_star)


def coupled_farquhar_stomata(
    T_leaf: jnp.ndarray,
    sw_down: jnp.ndarray,
    co2_ppmv: jnp.ndarray | float,
    q_air: jnp.ndarray,
    p_surface: jnp.ndarray,
    LAI: jnp.ndarray,
    beta_soil: jnp.ndarray,
    config: StomataConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the coupled Farquhar-stomata system iteratively.

    Iteratively finds the intercellular CO2 (Ci) consistent with both
    the Farquhar assimilation rate and the stomatal conductance model
    (Ball-Berry or Medlyn).

    Soil moisture stress is applied as a down-regulation of Vc_max
    following CLM (Bonan et al. 2011).

    Parameters
    ----------
    T_leaf : Surface/leaf temperature [K].
    sw_down : Downward shortwave radiation [W/m2].
    co2_ppmv : Atmospheric CO2 concentration [umol/mol].
    q_air : Specific humidity of near-surface air [kg/kg].
    p_surface : Surface pressure [Pa].
    LAI : Leaf area index [m2/m2].
    beta_soil : Soil moisture availability factor [0-1].
    config : StomataConfig.

    Returns
    -------
    gs : Canopy stomatal conductance [mol H2O/m2/s].
    gpp : Gross primary production [gC/m2/s].

    Thin wrapper over :func:`solve_coupled_farquhar_ci` (which also exposes
    the ``A_net`` / ``Ci`` / ``APAR`` / ``Gamma*`` the SIF diagnostic needs).
    """
    st = solve_coupled_farquhar_ci(
        T_leaf, sw_down, co2_ppmv, q_air, p_surface, LAI, beta_soil, config)
    return st.gs, st.gpp


# =====================================================================
# Effective beta for ET coupling
# =====================================================================

def compute_stomatal_beta(
    gs: jnp.ndarray,
    LAI: jnp.ndarray | None,
    beta_soil: jnp.ndarray,
    config: StomataConfig,
) -> jnp.ndarray:
    """Convert stomatal conductance to effective beta for latent heat.

    Blends bare-soil evaporation (limited by beta_soil) with canopy
    transpiration (limited by gs).  When LAI is available (carbon active),
    a Beer-law canopy fraction weights the two components.

    Parameters
    ----------
    gs : Stomatal conductance [mol H2O/m2/s].
    LAI : Leaf area index (None when carbon state unavailable).
    beta_soil : Bucket moisture beta [0-1].
    config : StomataConfig.

    Returns
    -------
    beta_eff : Effective moisture factor [0-1].
    """
    beta_canopy = jnp.clip(gs / config.gs_ref, 0.0, 1.0)

    if LAI is not None:
        f_canopy = 1.0 - jnp.exp(-config.k_ext * LAI)
        beta_eff = (1.0 - f_canopy) * beta_soil + f_canopy * beta_canopy
    else:
        # Without LAI, take the more limiting of soil and stomata
        beta_eff = jnp.minimum(beta_soil, beta_canopy)

    return jnp.clip(beta_eff, 0.0, 1.0)
