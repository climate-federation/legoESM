"""Leaf stomatal conductance + big-leaf photosynthesis (neutral single source).

The one home for the land stomatal / plant-physiology roster, shared by BOTH
surface stacks (the two-leaf canopy in ``land/canopy/`` and the big-leaf
SimpleSEB path).  Deliberately at the neutral ``land`` top level — NOT under
``carbon/`` or ``canopy/`` — so neither subpackage owns it and there is exactly
one copy of the stomatal-conductance numerics.

Models implemented:
- Canonical FvCB photosynthesis (``canopy.photosynthesis``): the big-leaf
  coupled A-gs solver delegates to the SAME Farquhar-von-Caemmerer-Berry C3 +
  Collatz C4 kernels the two-leaf canopy uses (shared ``c3_assimilation`` /
  ``c4_assimilation``), so the two land stacks cannot drift.
- Ball, Woodrow & Berry (1987): empirical stomatal conductance.
- Medlyn et al. (2011): optimal stomatal conductance (USO).
- Jarvis (1976): multiplicative stomatal conductance (CO2-independent).

The Ball-Berry / Medlyn kernels take the slope/intercept as explicit arguments
(so the two-leaf canopy can pass per-leaf-class C3/C4 values); the big-leaf
coupled solver unpacks its scalar ``StomataConfig`` at the call site.

When the carbon cycle is active, canonical FvCB replaces the light-use-efficiency
GPP and is coupled to Ball-Berry or Medlyn stomatal conductance; a ``fC4``
fraction blends the Collatz C4 branch (``fC4=0`` = pure C3).  When the carbon
cycle is off, the Jarvis model provides stomatal control on evapotranspiration
without requiring CO2 information.

Big-leaf acclimation note: the SimpleSEB path carries no prognostic growth-
temperature state (unlike the two-leaf canopy's 30-day TgC EMA), so its Kattge &
Knorr acclimation uses a fixed reference growth temperature
(``_TGC_REF_BIGLEAF_C``).  Wiring a prognostic TgC EMA into SimpleSEB is a
future enhancement.

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
from legoesm.land.canopy.photosynthesis import (
    c3_assimilation,
    c4_assimilation,
    co2_compensation_point,
)
from legoesm.land.canopy.sif import SIFConfig
from legoesm.land.leaf_biophysics import DIFFUSIVITY_RATIO_H2O_CO2
from legoesm.thermo import saturation_vapor_pressure

# Fixed gas-exchange constants (not tunable).
_CI_CA_INIT_RATIO = 0.7             # initial intercellular:ambient CO2 guess
# H2O:CO2 diffusivity ratio (Medlyn USO prefactor + A -> Ci back-calc) is the
# single-source constant DIFFUSIVITY_RATIO_H2O_CO2 imported from leaf_biophysics.
# Floor on the vapour-pressure deficit [kPa] guarding 1/sqrt(VPD) in Medlyn.
_VPD_FLOOR_KPA = 0.05


# =====================================================================
# Constants
# =====================================================================

# Gas constant, T reference, and Bernacchi kinetics now come from
# legoesm.land.leaf_biophysics (shared with the two-leaf FvCB path) so the
# two photosynthesis paths cannot drift.
_PAR_FRAC = 0.48   # Fraction of shortwave that is PAR
_PAR_CONV = 4.6    # umol photons per J of PAR
_MC = 12.0e-6      # g C per umol CO2

# Reference growth temperature [degC] for the big-leaf FvCB Kattge & Knorr
# acclimation.  The SimpleSEB big-leaf path carries no prognostic 30-day growth-
# temperature state (unlike the two-leaf canopy's TgC EMA), so it acclimates to a
# FIXED reference rather than a per-column running mean.  25 degC is the centre of
# the K&K [11,35] calibration range — a neutral "no seasonal acclimation" choice,
# not a tunable knob (a measurement/convention constant, per the fixed/tunable
# split).  Wiring a prognostic TgC EMA into SimpleSEB is a future enhancement.
_TGC_REF_BIGLEAF_C = 25.0

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
        # Photosynthesis kinetics (Vcmax T-response, Jmax, Rd, Bernacchi
        # constants, co-limitation) are no longer StomataConfig fields: the
        # coupled solver delegates to the canonical FvCB kernels in
        # canopy.photosynthesis, which carry those as module constants +
        # Kattge & Knorr acclimation. The one surviving photosynthesis knob is
        # Vc_max25 (the canonical Vcmax25 capacity). No excluded fields remain.
        "excluded": {},
        "params": {
            "beta_soil_min": {"units": "1", "bounds": (0.001, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CLM soil-water stress floor", "shape": None},
            "f_VPD_min": {"units": "1", "bounds": (0.001, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CLM VPD stress floor", "shape": None},
            "K_PAR": {"units": "1", "bounds": (66.0, 600.0), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Jarvis 1976", "shape": None},
            "T_opt_jarvis_C": {"units": "degC", "bounds": (8.25, 75.0), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Jarvis 1976", "shape": None, "legacy_name": "T_opt_jarvis"},
            "T_range_jarvis_C": {"units": "degC", "bounds": (6.6, 60.0), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Jarvis 1976", "shape": None, "legacy_name": "T_range_jarvis"},
            "Vc_max25": {"units": "umol/m2/s", "bounds": (19.8, 180.0), "tunable_tier": 1, "transform": "sigmoid", "category": "photosynthesis", "reference": "canonical FvCB Vcmax25 capacity (Bonan 2019 ch. 11)", "shape": None},
            "a_vpd": {"units": "1", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Jarvis 1976", "shape": None},
            "g0": {"units": "1", "bounds": (0.0033, 0.03), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Ball-Berry 1987 / Medlyn 2011", "shape": None},
            "g1_bb": {"units": "1", "bounds": (2.97, 27.0), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Ball-Berry 1987", "shape": None},
            "g1_med": {"units": "1", "bounds": (1.32, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Medlyn 2011", "shape": None},
            "gs_max": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "Jarvis 1976", "shape": None},
            "gs_ref": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "stomata", "reference": "beta-coupling reference gs", "shape": None},
            "k_ext": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "photosynthesis", "reference": "Beer-law canopy extinction (Sellers 1992)", "shape": None},
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

    # --- Canonical FvCB photosynthesis capacity ---
    # Vcmax25 is the one photosynthesis knob still owned here; every other FvCB
    # kinetic (Jmax, Rd, Kc/Ko/Gamma*, curvatures, acclimation) lives as module
    # constants in canopy.photosynthesis, which the coupled solver delegates to.
    Vc_max25: float = 60.0       # Max carboxylation at 25 C [umol/m2/s]

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

    # --- Canopy / beta coupling ---
    k_ext: float = 0.5           # Beer-law extinction coefficient [-]
    gs_ref: float = 0.3          # Reference max gs for beta [mol/m2/s]

    # --- Soil moisture / VPD stress floors (CLM-style; tunable) ---
    beta_soil_min: float = 0.01  # Lower bound on soil water stress factor
    f_VPD_min: float = 0.01      # Lower bound on VPD stress factor

    # --- Optional solar-induced fluorescence (SIF) diagnostic ---
    # None (default) disables it; a SIFConfig enables the passive big-leaf SIF
    # output on SurfaceFluxOutput.sif (coupled Farquhar path only — the Jarvis
    # fallback has no Ci/An to invert). Static config leaf, never traced.
    sif: SIFConfig | None = None


# =====================================================================
# Stomatal conductance models
# =====================================================================

def ball_berry_gs(
    A: jnp.ndarray,
    RH: jnp.ndarray,
    Cs: jnp.ndarray,
    slope: jnp.ndarray | float,
    intercept: jnp.ndarray | float,
) -> jnp.ndarray:
    """Ball-Berry (1987) stomatal conductance [mol H2O/m2/s].

    Single source of the gs numerics for BOTH land surface stacks — the
    slope/intercept are injected explicitly so the two-leaf canopy can pass
    per-leaf-class ``(m_C3, b0_C3)`` / ``(m_C4, b0_C4)`` and the big-leaf
    SimpleSEB solver can pass ``(config.g1_bb, config.g0)``.

    ``gs = intercept + slope * max(A, 0) * RH / Cs``, lower-bounded by
    ``intercept``.

    Parameters
    ----------
    A : net assimilation rate [umol CO2/m2/s]
    RH : relative humidity at the leaf surface [-]
    Cs : CO2 concentration at the leaf surface [umol/mol]
    slope : Ball-Berry slope ``m`` / ``g1_bb`` [-]
    intercept : residual conductance ``b0`` / ``g0`` [mol/m2/s]
    """
    A_pos = jnp.maximum(A, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(intercept + slope * A_pos * RH / Cs_safe, intercept)


def medlyn_gs(
    A: jnp.ndarray,
    VPD_kPa: jnp.ndarray,
    Cs: jnp.ndarray,
    g1: jnp.ndarray | float,
    g0: jnp.ndarray | float,
) -> jnp.ndarray:
    """Medlyn et al. (2011) optimal (USO) stomatal conductance [mol H2O/m2/s].

    Single source of the gs numerics (explicit-arg, see :func:`ball_berry_gs`):
    ``gs = g0 + 1.6 * (1 + g1 / sqrt(VPD)) * max(A, 0) / Cs``, lower-bounded by
    ``g0``.  ``g1`` [kPa^0.5]; ``VPD_kPa`` floored to guard 1/sqrt(VPD).
    """
    A_pos = jnp.maximum(A, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    VPD = jnp.maximum(VPD_kPa, _VPD_FLOOR_KPA)
    return jnp.maximum(
        g0 + DIFFUSIVITY_RATIO_H2O_CO2 * (1.0 + g1 / jnp.sqrt(VPD)) * A_pos / Cs_safe,
        g0,
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


def _bigleaf_assimilation(
    Ci: jnp.ndarray,
    APAR_umol: jnp.ndarray,
    T_leaf: jnp.ndarray,
    Vcmax25_eff: jnp.ndarray,
    TgC_C: jnp.ndarray | float,
    fC4: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Big-leaf GROSS + net assimilation via the canonical FvCB kernels.

    Single source of the photosynthesis biochemistry: delegates to the shared
    :func:`~legoesm.land.canopy.photosynthesis.c3_assimilation` /
    :func:`~legoesm.land.canopy.photosynthesis.c4_assimilation` kernels (the same
    ones the two-leaf canopy uses), so the big-leaf and two-leaf paths cannot
    drift.  ``Vcmax25_eff`` and ``APAR_umol`` are already canopy-scaled (per unit
    ground area), so the C3/C4 rates are per ground area; ``fC4`` area-weights them
    (differentiable, matching ``canopy.photosynthesis``).

    Both branches share the single column capacity ``Vcmax25_eff``.  In production
    ``fC4`` is the DOMINANT-PFT C4 flag (0 or 1; see the boundary-data providers),
    so a column is pure C3 or pure C4 with its own Vcmax — the natural big-leaf
    interpretation (a single leaf is one pathway).  Intermediate ``fC4`` is
    supported for testing / smooth gradients but then shares one Vcmax across both
    branches (an approximation); genuine sub-grid C3/C4 mixing with separate C3/C4
    capacities is the two-leaf canopy's role, not this simple scheme's.

    Returns
    -------
    (A_net, A_gross) : net (gross - Rd) and floored gross assimilation
        [umol CO2/m2/s].  GPP uses the gross rate; the stomata coupling uses net.
    """
    c3 = c3_assimilation(T_leaf, Ci, APAR_umol, Vcmax25_eff, TgC_C)
    c4 = c4_assimilation(T_leaf, Ci, APAR_umol, Vcmax25_eff)
    a_gross = (1.0 - fC4) * c3.a_gross + fC4 * c4.a_gross
    rd = (1.0 - fC4) * c3.rd + fC4 * c4.rd
    return a_gross - rd, jnp.maximum(a_gross, 0.0)


def solve_coupled_farquhar_ci(
    T_leaf: jnp.ndarray,
    sw_down: jnp.ndarray,
    co2_ppmv: jnp.ndarray | float,
    q_air: jnp.ndarray,
    p_surface: jnp.ndarray,
    LAI: jnp.ndarray,
    beta_soil: jnp.ndarray,
    config: StomataConfig,
    fC4: jnp.ndarray | float = 0.0,
) -> CoupledLeafState:
    """Solve the coupled FvCB-stomata system; return the converged leaf state.

    Big-leaf photosynthesis now routes through the CANONICAL FvCB biochemistry
    (``canopy.photosynthesis.c3_assimilation`` / ``c4_assimilation``) rather than a
    C3-only Farquhar kernel: it gains Jmax-bounded electron transport, Kattge &
    Knorr (2007) acclimation (at the fixed reference growth temperature
    ``_TGC_REF_BIGLEAF_C``; SimpleSEB has no prognostic TgC state), Tjoelker/Atkin
    dark respiration, and — via ``fC4`` — a Collatz C4 branch (``fC4=0`` = pure C3).

    Soil-water stress (``beta_soil``) and the big-leaf canopy integral
    (``L_c``) are folded into an effective ``Vcmax25`` so they scale the Rubisco-,
    light- and respiration-capacity terms coherently through the canonical
    derivation (Jmax25 = ratio·Vcmax25, Rd0 = 0.015·Vcmax25).

    Shared core of :func:`coupled_farquhar_stomata` (which returns just
    ``(gs, gpp)``) and the SIF diagnostic (``canopy/sif.py``, which also needs
    ``A_net``, ``Ci``, ``APAR`` and ``Gamma*``).
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
    # Converts leaf-level Vcmax25 to a canopy-level per-ground-area capacity so
    # the Rubisco- and light-limited branches are both canopy-scale (APAR_umol is
    # already canopy-absorbed).  L_c -> LAI as LAI -> 0.
    canopy_scaling = fAPAR / jnp.maximum(config.k_ext, 1e-6)

    # Effective Vcmax25 folds BOTH the canopy integral and the soil-water stress
    # (CLM btran) into a single capacity passed to the canonical FvCB kernels.
    # Because the kernels derive Jmax25 (= ratio·Vcmax25) and Rd0 (= 0.015·Vcmax25)
    # from Vcmax25, scaling it here scales all three capacities coherently.  NOTE:
    # unlike the retired C3-only kernel (which stressed Vcmax ONLY), beta now also
    # down-regulates Jmax and Rd — the CLM5-consistent behaviour.
    Vcmax25_eff = (config.Vc_max25 * canopy_scaling
                   * jnp.clip(beta_soil, config.beta_soil_min, 1.0))

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
        A_net, _ = _bigleaf_assimilation(
            Ci, APAR_umol, T_leaf, Vcmax25_eff, _TGC_REF_BIGLEAF_C, fC4)

        if config.stomata_model == "medlyn":
            gs = medlyn_gs(A_net, VPD_kPa, Ca, config.g1_med, config.g0)
        else:
            gs = ball_berry_gs(A_net, RH, Ca, config.g1_bb, config.g0)

        # Update Ci via stomatal diffusion (1.6 = H2O/CO2 ratio)
        gs_safe = jnp.maximum(gs, config.g0)
        A_pos = jnp.maximum(A_net, 0.0)
        Ci = Ca - DIFFUSIVITY_RATIO_H2O_CO2 * A_pos / gs_safe
        Ci = jnp.clip(Ci, 1.0, Ca)

    # Final evaluation
    A_net, A_gross = _bigleaf_assimilation(
        Ci, APAR_umol, T_leaf, Vcmax25_eff, _TGC_REF_BIGLEAF_C, fC4)

    if config.stomata_model == "medlyn":
        gs = medlyn_gs(A_net, VPD_kPa, Ca, config.g1_med, config.g0)
    else:
        gs = ball_berry_gs(A_net, RH, Ca, config.g1_bb, config.g0)

    # GPP: A_gross [umol CO2/m2/s] -> gC/m2/s
    gpp = jnp.maximum(A_gross, 0.0) * _MC

    # Gamma* for the SIF electron-transport inversion — the SAME canonical
    # Bernacchi (2001) compensation point the FvCB Aj branch uses, so the SIF
    # inversion and the photosynthesis kernel can never drift.
    gamma_star = co2_compensation_point(T_leaf)

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
    fC4: jnp.ndarray | float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the coupled FvCB-stomata system iteratively.

    Iteratively finds the intercellular CO2 (Ci) consistent with both the
    canonical FvCB assimilation rate and the stomatal conductance model
    (Ball-Berry or Medlyn).  Soil-water stress down-regulates the effective
    Vcmax25 (and, through the canonical derivation, Jmax/Rd); see
    :func:`solve_coupled_farquhar_ci`.

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
    fC4 : C4 area fraction [0-1] (0 = pure C3; blends the Collatz C4 branch).

    Returns
    -------
    gs : Canopy stomatal conductance [mol H2O/m2/s].
    gpp : Gross primary production [gC/m2/s].

    Thin wrapper over :func:`solve_coupled_farquhar_ci` (which also exposes
    the ``A_net`` / ``Ci`` / ``APAR`` / ``Gamma*`` the SIF diagnostic needs).
    """
    st = solve_coupled_farquhar_ci(
        T_leaf, sw_down, co2_ppmv, q_air, p_surface, LAI, beta_soil, config, fC4)
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
