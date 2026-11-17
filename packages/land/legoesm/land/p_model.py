"""P model: optimality-based acclimated photosynthesis parameters (Stocker 2020).

Least-cost + coordination optimality (Prentice et al. 2014; Wang et al. 2017;
Stocker et al. 2020 GMD "rpmodel"; Mengoli et al. 2022 dual-timescale) supplying
ACCLIMATED PARAMETERS to the existing FvCB/Medlyn machinery — it does not
compute fluxes itself:

- leaf-top ``Vcmax25`` and the ratio ``rjv25 = Jmax25/Vcmax25`` for the canopy
  FvCB kernel (``canopy/photosynthesis.py``), and
- a predicted Medlyn slope ``g1 = xi`` for ``stomata.medlyn_gs``.

The instantaneous ("fast") temperature response stays in the FvCB kernel; this
module handles only the slow (acclimated) optimum, evaluated at whole-day
running means of the drivers (plain exponential moving averages, see
:func:`advance_pmodel_acclim`).  Everything is closed form — no ci iteration —
and smooth, so the whole path is differentiable.

Faithfulness
------------
Transcribed from Stocker et al. (2020) GMD 13, 1545-1581 and the rpmodel R
source (geco-bern/rpmodel, R/rpmodel.R + R/subroutines.R), pinned by
``tests/land/unit/test_p_model_faithful.py`` against an independent scalar
oracle.  Chi: rpmodel ``calc_optimal_chi`` (Stocker eq. 8-9); Jmax limitation:
``wang17`` method, ``mprime = mj*sqrt(1-(c*/mj)^(2/3))`` with c* = 0.41 (Wang
et al. 2017); ``Vcmax = phi0(T)*Iabs*mprime/mc``; ``Jmax = 4*phi0(T)*Iabs /
sqrt((1/fact_jmaxlim)^2 - 1)``, algebraically rearranged here to
``4*phi0*Iabs*sqrt((1-k)/k)`` with ``k = (c*/mj)^(2/3)`` (identical, since
``fact_jmaxlim = sqrt(1-k)``); ``phi0(T) = phi0_peak/q_max*(0.352 + 0.022 Tc
- 3.4e-4 Tc^2)`` (Bernacchi 2003 fit, rpmodel ``ftemp_kphio``, peak-normalised
so ``phi0_peak`` is the curve maximum; default 0.0579 = rpmodel's kphio
0.081785 x q_max, its temperature-dependent, no-soil-moisture calibration).

DELIBERATE DEPARTURES from rpmodel, for internal consistency with the host
kernel (each documented where it happens):

1. **Leaf-level area basis.** rpmodel returns capacities per unit GROUND area
   using ``Iabs = fAPAR * PPFD``; here the output is the LEAF-TOP Vcmax25 the
   existing kn nitrogen-profile integral (``canopy/radiative_transfer.py``)
   expects, so ``Iabs`` is light per unit LEAF area.  Under the two-leaf
   scheme the caller passes the PAR a sunlit leaf ABSORBS,
   ``APAR_Sun / (fSun * LAI)`` from the scheme's own radiative transfer
   (``two_leaf_canopy.sunlit_leaf_apar``) — the same absorbed light the host
   kernel later runs the capacities with.  Lanes without that sunlit/shaded
   transfer (CLM-ML, simple SEB) still pass the incident top-of-canopy PPFD,
   which over-sizes the capacities by roughly the incident-to-absorbed ratio.
2. **Kinetics from the shared repo constants.** Kc/Ko/Gamma* mole fractions
   from ``leaf_biophysics`` (Bernacchi 2001, umol/mol), converted to Pa with
   the acclimated-mean surface pressure — not rpmodel's Pa-basis literals
   (~1% different for Kc).
3. **25 degC normalisation with the host kernel's Kattge & Knorr response**
   (``vcmax_temperature_response`` / ``jmax_temperature_response``), not
   rpmodel's constants, so the growth-T optimum survives the kernel's own
   temperature response on the way back up.
4. **Smooth mj floor.** Where rpmodel hard-zeroes capacities when
   ``mj <= c*`` (Jmax investment infeasible), we keep ``mj`` smoothly above
   ``c*`` (softplus floor) so capacities decay smoothly to ~0 with finite
   gradients (no NaN/dead adjoints).

References
----------
- Stocker et al. (2020) GMD 13, 1545-1581 (rpmodel).
- Wang et al. (2017) Nature Plants 3, 734-741 (least cost + Jmax limitation).
- Prentice et al. (2014) Ecol. Lett. 17, 82-91 (least-cost hypothesis).
- Mengoli et al. (2022) JAMES 14, e2021MS002767 (dual-timescale acclimation).
- Bernacchi et al. (2003) Plant Cell Environ. 26, 1419-1430 (phi0(T) fit).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.leaf_biophysics import (
    DIFFUSIVITY_RATIO_H2O_CO2,
    GAMMA_STAR25_UMOL_MOL,
    HA_GAMMA,
    HA_KC,
    HA_KO,
    KC25_UMOL_MOL,
    KO25_UMOL_MOL,
    O2_UMOL_MOL,
    arrhenius_factor,
    water_viscosity_ratio,
)
# NB: the Kattge & Knorr 25C-normalisation responses are imported at FUNCTION
# scope inside acclimated_capacities: a top-level import closes the cycle
# land.config -> p_model -> canopy.photosynthesis -> canopy.__init__ ->
# canopy.config -> p_model (CLAUDE.md: function-scope deferred imports for
# legitimate cross-module cycles).

__physics_contract__ = {
    "summary": (
        "P-model optimality (Stocker et al. 2020 rpmodel): closed-form "
        "least-cost chi = Gamma*/ca + (1-Gamma*/ca)*xi/(xi+sqrt(D)) with "
        "xi = sqrt(beta*(K+Gamma*)/(1.6*eta*)), Wang-2017 Jmax limitation, "
        "and coordination Vcmax = phi0(T)*Iabs*mj'/mc — evaluated at "
        "whole-day running means (plain EMA) of the drivers and "
        "normalised to 25 degC with the host kernel's Kattge & Knorr "
        "response. Supplies leaf-top Vcmax25, rjv25 = Jmax25/Vcmax25 and a "
        "predicted Medlyn slope g1 = xi to the existing FvCB/Medlyn path."
    ),
    "inputs": {
        "t_mean_K": "K", "vpd_mean_pa": "Pa", "co2_mean_ppm": "umol/mol",
        "ps_ema": "Pa", "iabs_mean": "umol/m^2/s", "dt": "s",
    },
    "outputs": {
        "vcmax25_leaf": "umol/m^2/s", "rjv25": "1", "g1_kpa": "kPa^0.5",
        "chi": "1",
    },
    "sign_convention": (
        "All capacities >= 0; chi in (0, 1); g1 > 0. Acclimation-state "
        "update is a convex combination (gain in [0, 1)), so means stay "
        "inside the historical driver range."
    ),
    "conserves": [],  # parameter prediction, not a flux
    "differentiable": True,
    "reference": (
        "Stocker et al. (2020) GMD 13 1545-1581; Wang et al. (2017) Nature "
        "Plants 3 734-741; Prentice et al. (2014) Ecol. Lett. 17 82-91; "
        "Mengoli et al. (2022) JAMES 14 e2021MS002767; Bernacchi et al. "
        "(2003) Plant Cell Environ. 26 1419-1430."
    ),
    "idealized_test": (
        "tests/land/unit/test_p_model.py: chi in (0,1) and decreasing in "
        "VPD; g1 ~ 2-6 kPa^0.5 at 25 degC / 1 kPa / 400 ppm / sea level; "
        "capacities -> 0 smoothly as Iabs -> 0 with finite rjv25 and finite "
        "gradients; rjv25 independent of phi0_peak and Iabs; the running means "
        "converge to the 24-hour average and stay finite through polar "
        "night. tests/land/unit/"
        "test_p_model_faithful.py: rel-1e-9 scalar-oracle pin."
    ),
}

__param_spec__ = {
    "PModelConfig": {
        "scheme_key": "land.p_model",
        "excluded": {
            "vpd_min_pa": "numerics: smooth floor under sqrt(VPD)",
            "mj_floor_eps": "numerics: softplus floor offset above c*",
            "mj_floor_width": "numerics: softplus floor width (AD safety)",
            "init_ppfd": "initialisation constant, not a physical closure",
            "init_vpd_pa": "initialisation constant, not a physical closure",
            "init_co2_ppm": "initialisation constant, not a physical closure",
            "window_start_h": (
                "structural: start of the subdaily noon acclimation window "
                "[local solar hour] (user decision 2026-11-06: 11:00-13:00)"),
            "window_end_h": (
                "structural: end of the subdaily noon acclimation window "
                "[local solar hour]"),
        },
        "params": {
            "beta_cost": {
                "units": "1", "bounds": (30.0, 500.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": "Stocker et al. 2020 GMD sect. 2.4.1 (beta = 146)",
                "shape": None,
            },
            "phi0_peak": {
                "units": "mol/mol", "bounds": (0.014, 0.125), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": (
                    "peak of phi0(T): Stocker et al. 2020 GMD Table 1 kphio "
                    "0.081785 x q_max = 0.0579 (temperature option); Long et "
                    "al. 1993 1/9 (sandoval option)"
                ),
                "shape": None,
                "legacy_name": "kphio",
            },
            "beta_cost_c4": {
                "units": "1", "bounds": (3.0, 60.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": "rpmodel c4 beta = 146/9 (Cai & Prentice 2020)",
                "shape": None,
            },
            "kphio_c4": {
                "units": "mol/mol", "bounds": (0.2, 2.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "photosynthesis",
                "reference": (
                    "rpmodel c4 base kphio = 1.0 (the Cai & Prentice 2020 "
                    "temperature quadratic carries the magnitude)"
                ),
                "shape": None,
            },
            "acclimation_days": {
                "units": "day", "bounds": (1.0, 120.0),
                "tunable_tier": 2, "transform": "sigmoid",
                "category": "photosynthesis",
                "reference": (
                    "Mengoli et al. 2022 / pyrealm AcclimationModel (~15 d "
                    "memory): the e-folding time of the half-hourly driver "
                    "EMA (han, phydro) and 1/alpha of the daily trait memory "
                    "(mengoli), one parameter for every scheme"),
                "shape": None, "legacy_name": "tau_acclim_s",
            },
        },
    },
}

# --- Wang et al. (2017) Jmax-limitation cost (rpmodel wang17) ---
C_STAR = 0.41  # [-] unit cost of Jmax maintenance; k = (c*/mj)^(2/3)

# --- Bernacchi (2003) quantum-yield temperature fit (rpmodel ftemp_kphio, C3) ---
# q(Tc) = a + b*Tc + c*Tc^2, ascending powers of leaf T [degC].
_KPHIO_QUAD_C3 = (0.352, 0.022, -3.4e-4)
# PEAK-NORMALISED FORM (user decision 2026-11-17; do NOT revert to the rpmodel
# ``kphio * q(Tc)`` form): phi0(Tc) = phi0_peak * q(Tc) / q_max, with q_max the
# quadratic's maximum, q(-b/2c) = a - b^2/(4c) = 0.7079 at Tc = 32.35 degC.
# ``phi0_peak`` is then the CURVE MAXIMUM [mol/mol], the same quantity the
# Sandoval option's peak is, so one tunable means one thing under both
# ``intrinsic_quantum_yield`` options and the two arms can be compared and
# tuned on one slot.  rpmodel's kphio = 0.081785 corresponds exactly to
# phi0_peak = 0.081785 * q_max = 0.0579 (the default below), so the
# temperature option is numerically unchanged at its default.
_KPHIO_QUAD_C3_MAX = _KPHIO_QUAD_C3[0] - _KPHIO_QUAD_C3[1] ** 2 / (4.0 * _KPHIO_QUAD_C3[2])
_RPMODEL_KPHIO = 0.081785  # [mol/mol] rpmodel / Stocker 2020 Table 1 kphio (pre-normalisation scale)
# Published peak per intrinsic_quantum_yield option (user decision 2026-11-17):
# the drivers set PModelConfig.phi0_peak from this table when the config does
# not give it, and the resolved config records the value.
PHI0_PEAK_DEFAULT = {
    "temperature": _RPMODEL_KPHIO * _KPHIO_QUAD_C3_MAX,  # 0.0579 = rpmodel kphio x q_max
    "sandoval": 1.0 / 9.0,                               # Long et al. 1993
}

# --- C4 quantum-yield temperature fit (rpmodel ftemp_kphio, c4 branch) ---
# "Cai & Prentice (2020), corrected by David Orme" (rpmodel source comment);
# the C4 base kphio is 1.0 in rpmodel — this quadratic carries the magnitude
# (peaks ~0.42 near 32 degC), floored at 0 like the C3 fit.
_KPHIO_QUAD_C4 = (-0.064, 0.03, -0.000464)
# rpmodel evaluates the C4 quadratic at a FIXED 15 degC when the temperature
# dependence is switched off (do_ftemp_kphio = FALSE branch).
_KPHIO_C4_REF_TC = 15.0

# --- numerics (smooth guards) ---
_PHI0_FLOOR_WIDTH = 0.01  # coeff-ok: softplus width keeping phi0 >= 0 smooth

# --- time and mean-solar-time conversions (exact; subdaily acclimation) ---
_S_PER_HOUR = 3600.0
_HOURS_PER_DAY = 24.0
_DEG_LON_PER_HOUR = 15.0  # 360 deg of longitude per 24 h of mean solar time

# --- Sandoval et al. (2026, New Phytol., doi:10.1111/nph.71303) quantum yield ---
# Published constants as in pyrealm 3.0.0rc5 PModelConst.sandoval_* and the
# JULES vn7.6_pmodel_unified_kphio pmodel_kphio_mod.F90 (identical):
# phi_peak = phi_max / (1 + AI^m)^n; dS = s0 * mGDD0^s1; Hd = hd_fac * dS.
# The peak itself (1/9 = 0.1111, Long et al. 1993) is NOT a constant here: it
# is ``PModelConfig.phi0_peak``, the tunable shared with the temperature
# option (set phi0_peak = 1/9 for the published Sandoval curve).
_SANDOVAL_PEAK_MN = (4.090556, 0.121122)  # aridity curvature (m, n)
_SANDOVAL_ENTROPY = (3468.185, -0.6680158)  # dS = s0 * mGDD0^s1 [J/mol/K]
_SANDOVAL_HA = 70885.39                   # [J/mol] activation energy
_SANDOVAL_HD_FACTOR = 295.0               # [K] Hd = factor * dS
# mGDD0 range over which Topt is defined (dS -> inf at 0; Hd < Ha above ~54 C);
# JULES init_pmodel_kphio clamps to the same range.
_MGDD0_RANGE_C = (1.0, 50.0)

# --- LUNA nitrogen use efficiencies (CTSM LunaMod.F90:76-77, verbatim) ---
# Diagnostic-only (PR5): the model carries NO nitrogen cycle; these convert
# C3 capacities into implied N pools for output tapes and NOTHING reads them
# back (no-feedback guarantee).  CTSM pins the ROUNDED values (comments there
# give the products 6.22*47.3 = 294.206 and 8.06*156 = 1257.36; the code uses
# 294.2 / 1257.0) — we copy the code values exactly.  C3 ONLY: the constants
# do not apply to the C4 pathway.  Provenance is secondhand in CTSM (Rogers
# 2014; Coste 2005 via Xu et al 2012) and 47.3 umol/gRubisco/s sits at the
# high end of specific-activity estimates — carry ~+/-50% structural
# uncertainty when interpreting the pools.
_F_C25_UMOL_GN_S = 294.2   # umol CO2 /s /gN in Rubisco (= 6.22 gRub/gN * 47.3)
_F_J25_UMOL_GN_S = 1257.0  # umol e- /s /gN in electron transport (= 8.06 * 156)


class PModelConfig(NamedTuple):
    """Configuration for the P-model optimality parameter source.

    ``beta_cost``/``phi0_peak``/``acclimation_days`` are the tunables (see
    ``__param_spec__``); the rest are numeric guards and recorded
    initialisation constants (excluded from calibration).
    """

    # --- optimality (Stocker et al. 2020 GMD) ---
    beta_cost: float = 146.0     # [-] carboxylation:transpiration cost ratio
    # Peak C3 quantum yield [mol/mol] = the maximum of phi0(T) under BOTH
    # intrinsic_quantum_yield options (see the _KPHIO_QUAD_C3_MAX note):
    # temperature -> phi0_peak * q(Tc)/q_max; sandoval -> phi0_peak /
    # (1 + AI^m)^n * f(T).  Default = rpmodel's kphio 0.081785 x q_max
    # (0.0579), i.e. the temperature option at its published calibration;
    # the published Sandoval peak is 1/9 (set explicitly on that arm).
    phi0_peak: float = _RPMODEL_KPHIO * _KPHIO_QUAD_C3_MAX  # PHI0_PEAK_DEFAULT["temperature"]
    kphio_temp: bool = True      # Bernacchi 2003 phi0(T) quadratic on (compile-time)
    # --- C4 optimality (rpmodel c4; Cai & Prentice 2020) ---
    beta_cost_c4: float = 146.0 / 9.0  # [-] C4 cost ratio (rpmodel: beta/9)
    kphio_c4: float = 1.0        # [mol/mol] C4 base quantum yield (rpmodel 1.0)
    # --- acclimation memory (Mengoli et al. 2022), one parameter for every
    #     scheme: e-folding time [days] of the driver EMA (han, phydro) and
    #     1/alpha of the daily trait memory (mengoli) ---
    acclimation_days: float = 15.0   # [day]
    # --- numerics (smooth guards; excluded from calibration) ---
    vpd_min_pa: float = 50.0     # [Pa] smooth floor under sqrt(D)
    mj_floor_eps: float = 0.01   # [-] mj kept above C_STAR + eps
    mj_floor_width: float = 0.01  # [-] softplus width of that floor
    # --- initialisation constants (recorded here, no hidden choice) ---
    init_ppfd: float = 400.0     # [umol/m^2/s] light-mean cold-start
    init_vpd_pa: float = 800.0   # [Pa] VPD-mean cold-start
    init_co2_ppm: float = 400.0  # [umol/mol] CO2 cold-start (driver may override)
    # --- acclimation scheme (appended last; positional configs stay aligned) ---
    # "han": the optimum solved on half-hourly exponential running means of the
    # drivers (e-folding ``acclimation_days``; Stocker 2020 order: average the
    # drivers, then optimise).  "mengoli": Mengoli et al. (2022) / pyrealm
    # SubdailyPModel — the daily optimum at the window means, then the memory
    # (alpha = 1/acclimation_days per day) on the optimal TRAITS xi, Vcmax25,
    # Jmax25 themselves.  Compile-time strings.
    acclimation_scheme: str = "han"
    # Which steps feed the acclimation: "24h" (every step), "daytime" (steps
    # with PAR > 0) or "noon" (the window_start_h..window_end_h local-solar
    # window).  han (and the P-hydro acclimation, which shares this EMA) keep
    # the half-hourly EMA on window-only options, weighting each in-window
    # step by the time elapsed since the previous one, so the e-folding stays
    # ``acclimation_days`` of calendar time; mengoli takes the window means.
    acclimation_hours: str = "24h"
    window_start_h: float = 11.0  # [h, local solar] first noon-window step
    window_end_h: float = 13.0    # [h, local solar] last noon-window step
    # --- C3 intrinsic quantum yield phi0 (appended last) ---
    # "temperature": phi0_peak x Bernacchi (2003) quadratic / its maximum.
    # "sandoval": Sandoval et al. (2026) — peak 1/9 reduced by the
    # climatological aridity index, peaked temperature response whose optimum
    # follows mGDD0 (JULES form, with the T/Topt factor).  Needs per-column
    # aridity index and mGDD0 (init_pmodel_acclim ``sandoval_inputs``); the
    # peak is ``phi0_peak`` (published value 1/9).
    intrinsic_quantum_yield: str = "temperature"


VALID_ACCLIMATION_SCHEMES = ("han", "mengoli")
VALID_ACCLIMATION_HOURS = ("24h", "daytime", "noon")
VALID_INTRINSIC_QUANTUM_YIELD = ("temperature", "sandoval")
# Retired names, refused with the new one named (no silent aliases).
_RETIRED_SCHEMES = {"ema_24h": "han", "subdaily_noon": "mengoli"}
_S_PER_DAY = 86400.0


def _retired(kind: str, value, table: dict) -> str:
    return (f" ({kind} {value!r} was renamed {table[value]!r})"
            if value in table else "")


def _check_quantum_yield(cfg: "PModelConfig") -> str:
    """Return ``cfg.intrinsic_quantum_yield`` after validating it."""
    if cfg.intrinsic_quantum_yield not in VALID_INTRINSIC_QUANTUM_YIELD:
        raise ValueError(
            f"unknown PModelConfig.intrinsic_quantum_yield "
            f"{cfg.intrinsic_quantum_yield!r}; must be one of "
            f"{VALID_INTRINSIC_QUANTUM_YIELD}")
    return cfg.intrinsic_quantum_yield


def _check_acclimation(cfg: "PModelConfig") -> str:
    """Return ``cfg.acclimation_scheme`` after validating the scheme, the
    hours, the memory and the noon window (raise on anything unknown)."""
    if cfg.acclimation_scheme not in VALID_ACCLIMATION_SCHEMES:
        raise ValueError(
            f"unknown PModelConfig.acclimation_scheme "
            f"{cfg.acclimation_scheme!r}; must be one of "
            f"{VALID_ACCLIMATION_SCHEMES}"
            + _retired("acclimation", cfg.acclimation_scheme, _RETIRED_SCHEMES))
    if cfg.acclimation_hours not in VALID_ACCLIMATION_HOURS:
        raise ValueError(
            f"unknown PModelConfig.acclimation_hours "
            f"{cfg.acclimation_hours!r}; must be one of "
            f"{VALID_ACCLIMATION_HOURS}")
    days = cfg.acclimation_days
    if isinstance(days, (int, float)) and not days > 0.0:
        # (a traced tunable is not checked here; its spec bounds apply)
        raise ValueError(
            f"PModelConfig.acclimation_days={days} must be positive")
    if (cfg.acclimation_scheme == "mengoli" and isinstance(days, (int, float))
            and days < 1.0):
        # the daily trait memory is alpha = 1/days per day; alpha > 1 overshoots
        raise ValueError(
            f"PModelConfig.acclimation_days={days} must be >= 1 under the "
            "mengoli scheme (alpha = 1/days per day)")
    if not (0.0 <= cfg.window_start_h < cfg.window_end_h < _HOURS_PER_DAY):
        raise ValueError(
            "PModelConfig noon window must satisfy 0 <= window_start_h < "
            f"window_end_h < 24; got ({cfg.window_start_h}, "
            f"{cfg.window_end_h})")
    return cfg.acclimation_scheme


def acclimation_tau_s(cfg: "PModelConfig"):
    """E-folding time [s] of the driver EMA: ``acclimation_days`` in seconds."""
    return cfg.acclimation_days * _S_PER_DAY


def needs_solar_hour(cfg: "PModelConfig") -> bool:
    """Whether the acclimation update reads the local solar hour: the noon
    window, or the mengoli scheme's day boundary (24h / noon windows).
    ``daytime`` is decided by PAR > 0 and needs no clock."""
    scheme = _check_acclimation(cfg)
    return cfg.acclimation_hours == "noon" or (
        scheme == "mengoli" and cfg.acclimation_hours != "daytime")


# Valid selector values for the CanopyConfig switches wired in
# surface_scheme/two_leaf_canopy.py; imported by canopy/config.py.
# capacity_scheme: "prescribed" (table) | "pmodel" (least-cost acclimated
# optimum) | "phydro" (P-hydro acclimated optimum, two-leaf canopy only).
VALID_CAPACITY_SCHEMES = ("prescribed", "pmodel", "phydro")
OPTIMALITY_CAPACITY_SCHEMES = ("pmodel", "phydro")
VALID_G1_SOURCES = ("table", "pmodel")


class PModelAcclimState(NamedTuple):
    """Whole-day running-mean acclimation drivers, all shape ``(ncol,)``.

    Every field is a plain exponential moving average over day AND night
    with e-folding time ``PModelConfig.acclimation_days`` (han / P-hydro;
    on window-only ``acclimation_hours`` each in-window step is weighted by
    the time elapsed since the previous one, ``dt_since_update_s``)
    (:func:`advance_pmodel_acclim`); ``iabs_mean`` is the mean of the light
    the caller passes (sunlit-leaf absorbed PAR under the two-leaf scheme),
    night zeros included.
    """

    iabs_mean: jax.Array
    t_mean_K: jax.Array
    vpd_mean_pa: jax.Array
    co2_mean_ppm: jax.Array
    ps_ema: jax.Array
    # Acclimated soil state seen by the roots (P-hydro only; ``None`` when
    # transpiration_stress != 'phydro'): the same running mean of the
    # conductance-weighted soil water potential [MPa] and of the soil-to-root
    # resistance per leaf area [MPa m2 s/mol] (land/phydro.py PhydroSupply).
    psi_s_mean_mpa: jax.Array | None = None
    rsoil_mean: jax.Array | None = None
    # --- mengoli acclimation only (``None`` under han) ---
    # Running sums over the current day's noon window (reset when it closes).
    win_sum_iabs: jax.Array | None = None
    win_sum_t_K: jax.Array | None = None
    win_sum_vpd_pa: jax.Array | None = None
    win_sum_co2_ppm: jax.Array | None = None
    win_sum_ps_pa: jax.Array | None = None
    win_count: jax.Array | None = None
    # Window means of the last completed day: the conditions the realised
    # traits were last updated at (chi, Jmax limitation and slope mapping).
    day_t_K: jax.Array | None = None
    day_vpd_pa: jax.Array | None = None
    day_co2_ppm: jax.Array | None = None
    day_ps_pa: jax.Array | None = None
    # Realised (memory-smoothed) daily optimal traits, pyrealm order.
    xi_real: jax.Array | None = None          # [Pa^0.5] C3
    vcmax25_real: jax.Array | None = None     # [umol/m2/s] C3 leaf-top
    jmax25_real: jax.Array | None = None      # [umol/m2/s] C3 leaf-top
    xi_c4_real: jax.Array | None = None       # [Pa^0.5] C4
    vcmax25_c4_real: jax.Array | None = None  # [umol/m2/s] C4 effective base
    # Completed windows since cold start (0 -> the next close SEEDS the traits
    # with that day's optimum instead of applying the memory; JULES / pyrealm).
    n_days: jax.Array | None = None
    # --- intrinsic_quantum_yield='sandoval' only: per-column CLIMATOLOGICAL inputs,
    # constant through the run (set once at init, carried unchanged). ---
    aridity_index: jax.Array | None = None   # [-] PET / P
    mgdd0_c: jax.Array | None = None         # [degC] mean T of days > 0 degC
    # --- windowed driver EMA (acclimation_hours != '24h'): seconds elapsed
    # since the last in-window update, so the e-folding stays calendar time.
    # None on a state written before this field existed (reads as 0). ---
    dt_since_update_s: jax.Array | None = None


_SUBDAILY_FIELDS = (
    "win_sum_iabs", "win_sum_t_K", "win_sum_vpd_pa", "win_sum_co2_ppm",
    "win_sum_ps_pa", "win_count", "day_t_K", "day_vpd_pa", "day_co2_ppm",
    "day_ps_pa", "xi_real", "vcmax25_real", "jmax25_real", "xi_c4_real",
    "vcmax25_c4_real", "n_days")


class PModelCapacities(NamedTuple):
    """Acclimated parameter predictions (leaf-top; diagnostics included)."""

    vcmax25_leaf: jax.Array  # [umol/m^2/s] leaf-top Vcmax25
    rjv25: jax.Array         # [-] Jmax25 / Vcmax25
    g1_kpa: jax.Array        # [kPa^0.5] predicted Medlyn slope (= xi)
    jmaxlim: jax.Array       # [-] Jmax limitation factor L = sqrt(1 - (c*/mj)^(2/3))
    chi: jax.Array           # [-] ci/ca at the acclimated optimum (diagnostic)
    xi_sqrt_pa: jax.Array    # [Pa^0.5] least-cost xi (diagnostic)


def pmodel_switches_active(surface_scheme=None, stomata=None) -> bool:
    """True iff any P-model switch is selected on the given config objects.

    Reads ``capacity_scheme``/``g1_source`` where present (canopy configs) and
    the same fields on a big-leaf ``StomataConfig``.  Shared by every lane's
    state-initialisation gate and driver guard so "is the P model on?" has one
    definition.
    """
    for obj in (surface_scheme, stomata):
        if obj is None:
            continue
        if getattr(obj, "capacity_scheme", "prescribed") in OPTIMALITY_CAPACITY_SCHEMES:
            return True
        if getattr(obj, "g1_source", "table") == "pmodel":
            return True
    return False


def _smooth_floor(x: jax.Array, floor: jax.Array, width: jax.Array) -> jax.Array:
    """Smooth (softplus) lower bound: ~x for x >> floor, -> floor from above.

    C-infinity everywhere (no `where`/`max` kink), so adjoints stay finite
    as x crosses the floor.
    """
    return floor + width * jax.nn.softplus((x - floor) / width)


def optimal_chi(
    T_K: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: PModelConfig,
    beta: jax.Array | float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Least-cost optimal chi (Stocker 2020 eq. 8-9), all pressures in Pa.

    Returns ``(chi, xi, ci_pa, gamma_star_pa, K_pa)``.  Kinetic mole
    fractions come from the shared ``leaf_biophysics`` Bernacchi constants,
    converted to partial pressures with ``ps_pa`` (departure 2 in the module
    docstring).
    """
    frac = 1e-6  # exact conversion umol/mol -> mol/mol
    gamma_star = GAMMA_STAR25_UMOL_MOL * frac * ps_pa * arrhenius_factor(T_K, HA_GAMMA)
    kc = KC25_UMOL_MOL * frac * ps_pa * arrhenius_factor(T_K, HA_KC)
    ko = KO25_UMOL_MOL * frac * ps_pa * arrhenius_factor(T_K, HA_KO)
    p_o2 = O2_UMOL_MOL * frac * ps_pa
    big_k = kc * (1.0 + p_o2 / ko)  # Michaelis coefficient K [Pa]

    eta_star = water_viscosity_ratio(T_K, ps_pa)
    _beta = cfg.beta_cost if beta is None else beta
    xi = jnp.sqrt(
        _beta * (big_k + gamma_star) / (DIFFUSIVITY_RATIO_H2O_CO2 * eta_star)
    )  # [Pa^0.5]

    chi, ci = chi_from_xi(xi, gamma_star, vpd_pa, co2_ppm, ps_pa, cfg)
    return chi, xi, ci, gamma_star, big_k


def chi_from_xi(
    xi: jax.Array,
    gamma_star_pa: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: PModelConfig,
) -> tuple[jax.Array, jax.Array]:
    """ci/ca and ci [Pa] for a GIVEN xi (Stocker 2020 eq. 8; pyrealm
    ``OptimalChiPrentice14.estimate_chi(xi_values=...)``).

    Shared by the instantaneous optimum (:func:`optimal_chi`) and the
    subdaily scheme, where xi is the memory-smoothed (lagged) value.  VPD gets
    the same smooth floor as everywhere in this module.
    """
    frac = 1e-6  # exact conversion umol/mol -> mol/mol
    vpd = _smooth_floor(vpd_pa, cfg.vpd_min_pa, cfg.vpd_min_pa)
    ca = co2_ppm * frac * ps_pa  # [Pa]
    chi = chi_core(xi, gamma_star_pa / ca, jnp.sqrt(vpd))
    ci = chi * ca
    return chi, ci


def chi_core(xi: jax.Array, gamma_over_ca: jax.Array,
             sqrt_vpd_pa: jax.Array) -> jax.Array:
    """chi = Gamma*/ca + (1 - Gamma*/ca) xi / (xi + sqrt(D)) (Stocker 2020
    eq. 8; xi [Pa^0.5], D [Pa]).  The ONE implementation, shared by the
    acclimated optimum and the canopy's instantaneous ci under
    ``stomatal_model='pmodel'`` (JULES ``stomata_pmodel``; pyrealm
    SubdailyPModel step 8)."""
    return gamma_over_ca + (1.0 - gamma_over_ca) * xi / (xi + sqrt_vpd_pa)


def _phi0(T_K: jax.Array, cfg: PModelConfig,
          kphio_inputs: tuple | None = None) -> jax.Array:
    """Apparent quantum yield phi0(T) [mol/mol] (rpmodel ftemp_kphio, C3).

    The Bernacchi quadratic goes negative below ~ -14 degC; a smooth floor at
    0 keeps deep-cold acclimated capacities at ~0 with finite gradients.
    """
    if _check_quantum_yield(cfg) == "sandoval":
        if kphio_inputs is None:
            raise ValueError(
                "PModelConfig.intrinsic_quantum_yield='sandoval' needs the per-column "
                "aridity index and mGDD0 (sandoval_inputs at init); this caller "
                "does not pass them (e.g. the P-hydro optimum is not wired).")
        return phi0_sandoval(T_K, *kphio_inputs, cfg.phi0_peak)
    if not cfg.kphio_temp:
        return jnp.asarray(cfg.phi0_peak) * jnp.ones_like(T_K)
    return phi0_c3(T_K, cfg.phi0_peak)


def phi0_sandoval(T_K: jax.Array, aridity_index: jax.Array,
                  mgdd0_c: jax.Array, phi0_peak: jax.Array | float) -> jax.Array:
    """Sandoval et al. (2026) C3 quantum yield phi0(T) [mol/mol].

    ``phi_peak(AI) * f(T; mGDD0)``: the peak ``phi0_peak`` (published 1/9;
    ``PModelConfig.phi0_peak``, the tunable shared with the temperature
    option) is reduced by the climatological aridity index AI = PET/P,
    ``phi_peak = phi0_peak / (1 + AI^m)^n``; ``f`` is a peaked Arrhenius function equal to 1 at Topt (the
    optimum of the plain peaked Arrhenius; with the T/Topt factor the maximum
    of ``f`` lies < 1 K above Topt and exceeds 1 by < 0.1 %), whose activation
    entropy ``dS = s0 * mGDD0^s1`` (and
    deactivation energy ``Hd = 295 dS``) follow the growth climate:

        Topt = Hd / (dS - R ln(Ha / (Hd - Ha)))
        f(T) = (T/Topt) exp(Ha (T - Topt) / (Topt R T))
               (1 + exp((Topt dS - Hd)/(R Topt))) / (1 + exp((T dS - Hd)/(R T)))

    JULES ``pmodel_kphio_mod.kphio_c3_sandoval_fn`` form (user decision
    2026-11-07), INCLUDING the T/Topt factor; pyrealm 3.0.0rc5 omits that
    factor in code (documented departure: ratio T/Topt).  ``mgdd0_c`` must lie
    in ``_MGDD0_RANGE_C`` (clamped at init, as JULES does).
    """
    r_gas = constants.R_universal
    m, n = _SANDOVAL_PEAK_MN
    phi_peak = phi0_peak / (1.0 + aridity_index ** m) ** n
    s0, s1 = _SANDOVAL_ENTROPY
    ds = s0 * mgdd0_c ** s1
    hd = _SANDOVAL_HD_FACTOR * ds
    ha = _SANDOVAL_HA
    topt = hd / (ds - r_gas * jnp.log(ha / (hd - ha)))
    f = ((T_K / topt) * jnp.exp(ha * (T_K - topt) / (topt * r_gas * T_K))
         * (1.0 + jnp.exp((topt * ds - hd) / (r_gas * topt)))
         / (1.0 + jnp.exp((T_K * ds - hd) / (r_gas * T_K))))
    return phi_peak * f


def kphio_inputs_from(acclim: "PModelAcclimState | None",
                 cfg: "PModelConfig") -> tuple | None:
    """``(aridity_index, mgdd0_c)`` from the state under intrinsic_quantum_yield=
    'sandoval' (raise if absent), else ``None``."""
    if _check_quantum_yield(cfg) != "sandoval":
        return None
    if acclim is None or acclim.aridity_index is None or acclim.mgdd0_c is None:
        raise ValueError(
            "PModelConfig.intrinsic_quantum_yield='sandoval' but the acclimation state "
            "carries no aridity_index / mgdd0_c: initialise it with "
            "init_pmodel_acclim(..., sandoval_inputs=(ai, mgdd0)).")
    return (acclim.aridity_index, acclim.mgdd0_c)


def phi0_c3(T_K: jax.Array, phi0_peak: jax.Array | float) -> jax.Array:
    """Temperature-dependent C3 quantum yield phi0(T) [mol/mol].

    ``phi0_peak * q(Tc) / q_max``, ``q = 0.352 + 0.022 Tc - 3.4e-4 Tc^2``
    (Bernacchi et al. 2003; rpmodel / pyrealm ``QuantumYieldTemperature``),
    ``q_max = q(32.35 degC) = 0.7079``, smoothly floored at 0.  Peak-normalised
    on purpose so ``phi0_peak`` IS the curve maximum, the same quantity as the
    Sandoval peak (see the ``_KPHIO_QUAD_C3_MAX`` note; rpmodel's kphio =
    phi0_peak / q_max).  Do not revert to ``kphio * q``.
    Shared by the acclimated optimum and, under
    ``CanopyConfig.electron_transport_model='smith'``, by the canopy kernel's
    instantaneous electron transport, so the phi0 that sizes Vcmax/Jmax is
    the one that uses them.
    """
    tc = T_K - constants.T_freeze
    a0, a1, a2 = _KPHIO_QUAD_C3
    quad = a0 + a1 * tc + a2 * tc * tc
    quad = _PHI0_FLOOR_WIDTH * jax.nn.softplus(quad / _PHI0_FLOOR_WIDTH)
    return phi0_peak * quad / _KPHIO_QUAD_C3_MAX


def acclimated_capacities(
    acclim: PModelAcclimState | None, cfg: PModelConfig
) -> PModelCapacities:
    """Acclimated leaf-top Vcmax25, rjv25 and predicted Medlyn g1.

    Coordination (Stocker eq. 16 / rpmodel wang17): at the acclimated-mean
    drivers, ``Vcmax_growth = phi0(Tg)*Iabs*mj'/mc`` and ``Jmax_growth =
    4*phi0(Tg)*Iabs*sqrt((1-k)/k)``; both are normalised to 25 degC with the
    host kernel's own Kattge & Knorr responses evaluated at the same growth
    temperature.  ``mj'/mc`` is computed via the exact algebraic ratio
    ``mj/mc = (ci + K)/(ci + 2 Gamma*)`` so the ci -> Gamma* cancellation is
    preserved analytically (no 0/0), and ``rjv25`` is likewise formed from
    the analytic ``Jmax/Vcmax`` ratio in which phi0 and Iabs cancel — so
    rjv25 stays finite as Iabs -> 0.
    """
    if acclim is None:
        raise ValueError(
            "P-model capacities requested but no acclimation state was "
            "supplied (pmodel_acclim=None): this land lane does not carry "
            "PModelAcclimState. Initialise it (init_pmodel_acclim) or select "
            "capacity_scheme='prescribed' / g1_source='table'."
        )
    if _check_acclimation(cfg) == "mengoli":
        return _realised_capacities(acclim, cfg)
    return capacities_at(acclim.t_mean_K, acclim.vpd_mean_pa,
                         acclim.co2_mean_ppm, acclim.ps_ema,
                         acclim.iabs_mean, cfg,
                         kphio_inputs=kphio_inputs_from(acclim, cfg))


def capacities_at(
    tg_k: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    iabs: jax.Array,
    cfg: PModelConfig,
    kphio_inputs: tuple | None = None,
) -> PModelCapacities:
    """C3 least-cost optimum at GIVEN acclimation drivers (any scheme).

    The han scheme calls this at the running means; the mengoli scheme
    at each day's noon-window means (the daily optimum it then smooths).
    """
    chi, xi, ci, gamma_star, big_k = optimal_chi(
        tg_k, vpd_pa, co2_ppm, ps_pa, cfg
    )
    vcmax25, rjv25, jmaxlim = capacities_from_chi(
        chi=chi, ci_pa=ci, gamma_star_pa=gamma_star, big_k_pa=big_k,
        tg_k=tg_k, iabs=iabs, phi0=_phi0(tg_k, cfg, kphio_inputs), cfg=cfg)

    # Medlyn slope: matching ci/ca = g1/(g1 + sqrt(D)) to the xi term of
    # Stocker eq. 8 (Gamma*/ca term neglected — standard) gives g1 = xi;
    # sqrt(1000) converts Pa^0.5 -> kPa^0.5 (medlyn_gs takes VPD in kPa).
    g1_kpa = xi / jnp.sqrt(1000.0)
    return PModelCapacities(
        vcmax25_leaf=vcmax25, rjv25=rjv25, g1_kpa=g1_kpa, jmaxlim=jmaxlim,
        chi=chi, xi_sqrt_pa=xi
    )


def _require_subdaily(acclim: PModelAcclimState) -> None:
    missing = [f for f in _SUBDAILY_FIELDS if getattr(acclim, f) is None]
    if missing:
        raise ValueError(
            "PModelConfig.acclimation_scheme='mengoli' but the acclimation "
            f"state lacks {missing}: initialise it with init_pmodel_acclim "
            "under the same config (a state built for han cannot be "
            "used).")


# Floor on the realised Vcmax25 when forming Jmax25/Vcmax25 (numerics only:
# the realised traits are seeded > 0 and decay geometrically, never to 0).
_VCMAX_RATIO_FLOOR = 1e-9  # [umol/m2/s]


def _realised_capacities(acclim: PModelAcclimState,
                         cfg: PModelConfig) -> PModelCapacities:
    """C3 capacities from the realised (memory-smoothed) daily traits.

    Vcmax25 and Jmax25 are the smoothed values themselves (pyrealm
    ``SubdailyPModel`` step 5-6); chi is recomputed from the smoothed xi at
    the last window's conditions (pyrealm step 8 form) and carries the
    Wang-2017 limitation factor the empirical-stress Jmax re-derivation
    (:func:`stressed_rjv25_factor`) needs.
    """
    _require_subdaily(acclim)
    _, _, _, gamma_star, big_k = optimal_chi(
        acclim.day_t_K, acclim.day_vpd_pa, acclim.day_co2_ppm,
        acclim.day_ps_pa, cfg)
    chi, ci = chi_from_xi(acclim.xi_real, gamma_star, acclim.day_vpd_pa,
                          acclim.day_co2_ppm, acclim.day_ps_pa, cfg)
    one = jnp.ones_like(chi)
    _, _, jmaxlim = capacities_from_chi(
        chi=chi, ci_pa=ci, gamma_star_pa=gamma_star, big_k_pa=big_k,
        tg_k=acclim.day_t_K, iabs=one, phi0=one, cfg=cfg)
    rjv25 = acclim.jmax25_real / jnp.maximum(acclim.vcmax25_real,
                                             _VCMAX_RATIO_FLOOR)
    return PModelCapacities(
        vcmax25_leaf=acclim.vcmax25_real, rjv25=rjv25,
        g1_kpa=acclim.xi_real / jnp.sqrt(1000.0), jmaxlim=jmaxlim,
        chi=chi, xi_sqrt_pa=acclim.xi_real)


def acclim_means(acclim: PModelAcclimState,
                 cfg: PModelConfig) -> PModelAcclimState:
    """The acclimation state whose MEAN fields describe the conditions the
    current traits were set at, for consumers that read them directly (the
    stomatal-slope mapping, Gamma* for the C3/C4 slopes).

    han: the state itself.  mengoli: the running means replaced by
    the last completed noon window's means (light is not read by those
    consumers and is left as the 24 h running mean).
    """
    if _check_acclimation(cfg) == "han":
        return acclim
    _require_subdaily(acclim)
    return acclim._replace(t_mean_K=acclim.day_t_K,
                           vpd_mean_pa=acclim.day_vpd_pa,
                           co2_mean_ppm=acclim.day_co2_ppm,
                           ps_ema=acclim.day_ps_pa)


def stressed_rjv25_factor(jmaxlim: jax.Array, stress: jax.Array) -> jax.Array:
    """Factor on Jmax25/Vcmax25 when an empirical stress scales Vcmax.

    rpmodel's structure (``rpmodel.R``): the soil-moisture stress ``s``
    multiplies Vcmax, and Jmax is then re-derived from the stressed Vcmax
    through the Jmax limitation ``L = s * L0`` on the Smith light response,
    ``Jmax = 4 phi0 Iabs / sqrt(1/L^2 - 1)``.  Jmax therefore falls MORE than
    Vcmax: ``Jmax_s / Jmax_0 = s * sqrt((1 - L0^2) / (1 - s^2 L0^2))``, and the
    ratio Jmax/Vcmax is multiplied by the square root returned here (1 at
    s = 1, ``sqrt(1 - L0^2)`` as s -> 0).
    """
    l0_sq = jmaxlim * jmaxlim
    return jnp.sqrt((1.0 - l0_sq) / (1.0 - stress * stress * l0_sq))


def capacities_from_chi(
    *,
    chi: jax.Array,
    ci_pa: jax.Array,
    gamma_star_pa: jax.Array,
    big_k_pa: jax.Array,
    tg_k: jax.Array,
    iabs: jax.Array,
    phi0: jax.Array,
    cfg: PModelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Coordination capacities (Vcmax25, rjv25, L) at a GIVEN optimal chi.

    The chi -> capacity algebra shared by the least-cost optimum
    (:func:`acclimated_capacities`) and the P-hydro profit optimum
    (``land/phydro.py``): Wang-2017 limitation on the smooth-floored mj,
    ``Vcmax_growth = phi0*Iabs*mj'/mc`` via the analytic ``mj/mc`` ratio,
    both normalised to 25 degC with the host Kattge & Knorr responses.
    """
    from legoesm.land.canopy.photosynthesis import (
        jmax_temperature_response,
        vcmax_temperature_response,
    )

    del chi  # capacity algebra runs on ci/Gamma*/K directly
    tg_c = tg_k - constants.T_freeze
    mj = (ci_pa - gamma_star_pa) / (ci_pa + 2.0 * gamma_star_pa)
    mj_safe = _smooth_floor(mj, C_STAR + cfg.mj_floor_eps, cfg.mj_floor_width)
    k = (C_STAR / mj_safe) ** (2.0 / 3.0)  # in (0, 1) by the floor
    sqrt_1mk = jnp.sqrt(1.0 - k)

    # mj'/mc = sqrt(1-k) * (mj/mc), with mj/mc = (ci + K)/(ci + 2 Gamma*).
    mj_over_mc = (ci_pa + big_k_pa) / (ci_pa + 2.0 * gamma_star_pa)

    f_v = vcmax_temperature_response(tg_k, tg_c)
    f_j = jmax_temperature_response(tg_k, tg_c)

    vcmax25 = phi0 * iabs * sqrt_1mk * mj_over_mc / f_v
    # Jmax_growth / Vcmax_growth = 4*sqrt((1-k)/k) / (sqrt(1-k)*mj/mc)
    #                            = 4 / (sqrt(k) * mj/mc)   (phi0, Iabs cancel)
    rjv25 = (4.0 / (jnp.sqrt(k) * mj_over_mc)) * (f_v / f_j)
    return vcmax25, rjv25, sqrt_1mk


def init_pmodel_acclim(
    ncol: int,
    *,
    t_init_K: jax.Array | float,
    ps_init_pa: jax.Array | float,
    cfg: PModelConfig,
    co2_init_ppm: jax.Array | float | None = None,
    dtype: Any = jnp.float32,
    soil_init: tuple[float, float] | None = None,
    sandoval_inputs: tuple | None = None,
) -> PModelAcclimState:
    """Cold-start acclimation state from the recorded config constants.

    ``sandoval_inputs = (aridity_index, mgdd0_c)`` (scalar or ``(ncol,)``)
    are the climatological inputs of ``intrinsic_quantum_yield='sandoval'``, required
    there and refused otherwise (they would be inert).  mGDD0 is clamped to
    ``_MGDD0_RANGE_C`` (the JULES rule; outside it Topt is undefined).

    ``t_init_K``/``ps_init_pa`` come from the caller's initial condition;
    PPFD/VPD (and CO2 unless overridden with the run's value) come from the
    ``PModelConfig`` init fields so every cold-start choice is in the run's
    resolved config.  ``soil_init = (psi_s_mpa, rsoil_leaf)`` (the
    ``PHydroConfig`` init fields) switches the P-hydro soil-state means on;
    ``None`` leaves them absent.
    """
    full = lambda v: jnp.full((ncol,), v, dtype=dtype)  # noqa: E731
    co2 = cfg.init_co2_ppm if co2_init_ppm is None else co2_init_ppm
    state = PModelAcclimState(
        iabs_mean=full(cfg.init_ppfd),
        t_mean_K=full(0.0) + jnp.asarray(t_init_K, dtype=dtype),
        vpd_mean_pa=full(cfg.init_vpd_pa),
        co2_mean_ppm=full(0.0) + jnp.asarray(co2, dtype=dtype),
        ps_ema=full(0.0) + jnp.asarray(ps_init_pa, dtype=dtype),
        psi_s_mean_mpa=None if soil_init is None else full(soil_init[0]),
        rsoil_mean=None if soil_init is None else full(soil_init[1]),
    )
    if _check_quantum_yield(cfg) == "sandoval":
        if cfg.phi0_peak == PHI0_PEAK_DEFAULT["temperature"]:
            raise ValueError(
                "PModelConfig.intrinsic_quantum_yield='sandoval' with phi0_peak "
                "left at the temperature option's default (0.0579): set "
                "phi0_peak explicitly (published Sandoval peak "
                "PHI0_PEAK_DEFAULT['sandoval'] = 1/9; the drivers do this)")
        if sandoval_inputs is None:
            raise ValueError(
                "PModelConfig.intrinsic_quantum_yield='sandoval' needs sandoval_inputs="
                "(aridity_index, mgdd0_c) at init")
        if not cfg.kphio_temp:
            raise ValueError("intrinsic_quantum_yield='sandoval' is temperature-dependent; "
                             "kphio_temp=False is not supported")
        ai, mg = sandoval_inputs
        try:   # concrete inputs (the usual case): reject what makes phi0 NaN
            ai_np, mg_np = np.asarray(ai, dtype=float), np.asarray(mg, dtype=float)
        except Exception:  # noqa: BLE001 - traced inputs: no value check
            ai_np = mg_np = None
        if ai_np is not None and (not np.all(np.isfinite(ai_np))
                                  or np.any(ai_np < 0.0)
                                  or not np.all(np.isfinite(mg_np))):
            raise ValueError(
                "Sandoval inputs must be finite with aridity index >= 0 "
                f"(got aridity {ai_np}, mGDD0 {mg_np})")
        lo, hi = _MGDD0_RANGE_C
        state = state._replace(
            aridity_index=full(0.0) + jnp.asarray(ai, dtype=dtype),
            mgdd0_c=jnp.clip(full(0.0) + jnp.asarray(mg, dtype=dtype), lo, hi))
    elif sandoval_inputs is not None:
        raise ValueError(
            "sandoval_inputs given but PModelConfig.intrinsic_quantum_yield="
            f"{cfg.intrinsic_quantum_yield!r}; they would be silently inert")
    state = state._replace(dt_since_update_s=full(0.0))
    if _check_acclimation(cfg) == "han":
        return state
    if soil_init is not None:
        raise ValueError(
            "PModelConfig.acclimation_scheme='mengoli' is not wired for "
            "P-hydro (transpiration_stress='phydro'), whose optimum is solved "
            "on the driver EMA; select acclimation_scheme='han'.")
    # Subdaily cold start: the traits hold the optimum at the cold-start
    # drivers (the same recorded init constants) only until the first noon
    # window closes, which SEEDS them with that day's optimum (n_days = 0).
    caps = capacities_at(state.t_mean_K, state.vpd_mean_pa,
                         state.co2_mean_ppm, state.ps_ema, state.iabs_mean, cfg,
                         kphio_inputs=kphio_inputs_from(state, cfg))
    caps4 = capacities_c4_at(state.t_mean_K, state.vpd_mean_pa,
                             state.co2_mean_ppm, state.ps_ema,
                             state.iabs_mean, cfg)
    zero = full(0.0)
    return state._replace(
        win_sum_iabs=zero, win_sum_t_K=zero, win_sum_vpd_pa=zero,
        win_sum_co2_ppm=zero, win_sum_ps_pa=zero, win_count=zero,
        day_t_K=state.t_mean_K, day_vpd_pa=state.vpd_mean_pa,
        day_co2_ppm=state.co2_mean_ppm, day_ps_pa=state.ps_ema,
        xi_real=caps.xi_sqrt_pa, vcmax25_real=caps.vcmax25_leaf,
        jmax25_real=caps.vcmax25_leaf * caps.rjv25,
        xi_c4_real=caps4.xi_c4_sqrt_pa,
        vcmax25_c4_real=caps4.vcmax25_c4_leaf,
        n_days=zero,
    )


def local_solar_hour(utc_seconds: jax.Array | float,
                     lon_deg: jax.Array | float) -> jax.Array:
    """Local solar hour [0, 24) from UTC seconds and longitude [deg east].

    Mean solar time (lon/15 h offset, no equation of time) — the JULES
    P-model convention (``physiol_jls_mod.F90``).
    """
    hours = (jnp.asarray(utc_seconds) / _S_PER_HOUR
             + jnp.asarray(lon_deg) / _DEG_LON_PER_HOUR)
    return jnp.mod(hours, _HOURS_PER_DAY)


def in_noon_window(solar_hour: jax.Array, dt: float,
                   cfg: PModelConfig) -> jax.Array:
    """Is this step inside the noon acclimation window? (bool array)

    The step's local solar hour is rounded to the model step (JULES rounds
    to its 30-min step the same way) and tested against
    ``[window_start_h, window_end_h]`` inclusive: 11:00-13:00 holds 3 hourly
    steps or 5 half-hourly ones; a step coarser than the window keeps only
    the step that rounds to noon (user decision 2026-11-06).
    """
    step_h = dt / _S_PER_HOUR
    h = jnp.mod(jnp.round(solar_hour / step_h) * step_h, _HOURS_PER_DAY)
    tol = 1e-6 * step_h  # coeff-ok: float tolerance on the rounded hour
    return (h >= cfg.window_start_h - tol) & (h <= cfg.window_end_h + tol)


def acclimation_window(cfg: PModelConfig, dt: float, *, solar_hour=None,
                       ppfd=None) -> tuple[jax.Array, jax.Array]:
    """``(in_window, closes)`` of this step for ``cfg.acclimation_hours``.

    ``24h``: every step is in the window, which closes on the day's last step
    (local solar hour + dt reaches 24 h).  ``noon``: the
    ``window_start_h..window_end_h`` local-solar window, closing on its last
    step.  ``daytime``: steps with PAR > 0 (no clock needed); the window
    closes on the first dark step after a lit one, so the mengoli daily
    update lands one step after sunset (``closes`` then marks a step that is
    itself OUTSIDE the window).
    """
    hours = cfg.acclimation_hours
    if hours == "daytime":
        if ppfd is None:
            raise ValueError("acclimation_hours='daytime' needs the step's PAR")
        inw = ppfd > 0.0
        return inw, ~inw          # a dark step closes a lit run (checked by the caller)
    if solar_hour is None:
        raise ValueError(
            f"acclimation_hours={hours!r} needs each step's local solar hour "
            "(solar_hour=None): this caller does not pass the clock the "
            "window is defined on.")
    if hours == "24h":
        step_h = dt / _S_PER_HOUR
        inw = jnp.ones_like(jnp.asarray(solar_hour), dtype=bool)
        close = (jnp.mod(jnp.round(solar_hour / step_h) * step_h, _HOURS_PER_DAY)
                 + step_h >= _HOURS_PER_DAY - 1e-6 * step_h)  # coeff-ok: float tolerance on the rounded hour
        return inw, close
    inw = in_noon_window(solar_hour, dt, cfg)
    close = inw & ~in_noon_window(solar_hour + dt / _S_PER_HOUR, dt, cfg)
    return inw, close


def _advance_subdaily(new: PModelAcclimState, old: PModelAcclimState, *,
                      T_K, ppfd, vpd_pa, co2_ppm, ps_pa, inw, close,
                      cfg: PModelConfig, dt: float) -> PModelAcclimState:
    """Window accumulation, daily optimum and daily trait memory (mengoli).

    Accumulate the drivers on window steps; when the window CLOSES form the
    window means, evaluate the C3 and C4 optima there, and update the
    realised traits — seeded with the optimum on the first completed window,
    else ``real += alpha * (opt - real)``, ``alpha = 1/acclimation_days``
    (pyrealm order: the memory acts on xi, Vcmax25, Jmax25 themselves).  The
    new values take effect from the next step (pyrealm ``update_point='max'``).
    ``inw``/``close`` come from :func:`acclimation_window`; for the
    ``daytime`` hours ``close`` is the first dark step after a lit run.
    """
    _require_subdaily(old)
    if cfg.acclimation_hours == "daytime":
        close = close & (old.win_count > 0.0)
    w = inw.astype(old.win_count.dtype)
    # where(), not w * x: an off-window non-finite driver must not reach the
    # sums (0 * NaN = NaN; codex review round 1).
    add = lambda s, x: s + jnp.where(inw, x, 0.0)  # noqa: E731
    sums = {
        "iabs": add(old.win_sum_iabs, ppfd),
        "t": add(old.win_sum_t_K, T_K),
        "vpd": add(old.win_sum_vpd_pa, vpd_pa),
        "co2": add(old.win_sum_co2_ppm, co2_ppm),
        "ps": add(old.win_sum_ps_pa, ps_pa),
    }
    count = old.win_count + w
    has = count > 0.0
    safe_n = jnp.where(has, count, 1.0)
    # Off-window steps (has == False) fall back to the last day's means so the
    # optimum below is always evaluated at finite, physical drivers (its
    # value is discarded unless close; this keeps adjoints NaN-free).
    mean = lambda s, fb: jnp.where(has, s / safe_n, fb)  # noqa: E731
    m_t = mean(sums["t"], old.day_t_K)
    m_vpd = mean(sums["vpd"], old.day_vpd_pa)
    m_co2 = mean(sums["co2"], old.day_co2_ppm)
    m_ps = mean(sums["ps"], old.day_ps_pa)
    m_iabs = mean(sums["iabs"], jnp.zeros_like(old.win_sum_iabs))
    opt = capacities_at(m_t, m_vpd, m_co2, m_ps, m_iabs, cfg,
                        kphio_inputs=kphio_inputs_from(old, cfg))
    opt4 = capacities_c4_at(m_t, m_vpd, m_co2, m_ps, m_iabs, cfg)
    seed = old.n_days < 0.5
    alpha = 1.0 / cfg.acclimation_days

    def upd(real, optimum):
        smoothed = jnp.where(seed, optimum, real + alpha * (optimum - real))
        return jnp.where(close, smoothed, real)

    reset = lambda x: jnp.where(close, jnp.zeros_like(x), x)  # noqa: E731
    keep_day = lambda m, d: jnp.where(close, m, d)  # noqa: E731
    return new._replace(
        win_sum_iabs=reset(sums["iabs"]), win_sum_t_K=reset(sums["t"]),
        win_sum_vpd_pa=reset(sums["vpd"]), win_sum_co2_ppm=reset(sums["co2"]),
        win_sum_ps_pa=reset(sums["ps"]), win_count=reset(count),
        day_t_K=keep_day(m_t, old.day_t_K),
        day_vpd_pa=keep_day(m_vpd, old.day_vpd_pa),
        day_co2_ppm=keep_day(m_co2, old.day_co2_ppm),
        day_ps_pa=keep_day(m_ps, old.day_ps_pa),
        xi_real=upd(old.xi_real, opt.xi_sqrt_pa),
        vcmax25_real=upd(old.vcmax25_real, opt.vcmax25_leaf),
        jmax25_real=upd(old.jmax25_real, opt.vcmax25_leaf * opt.rjv25),
        xi_c4_real=upd(old.xi_c4_real, opt4.xi_c4_sqrt_pa),
        vcmax25_c4_real=upd(old.vcmax25_c4_real, opt4.vcmax25_c4_leaf),
        n_days=old.n_days + close.astype(old.n_days.dtype),
    )


class PModelCapacitiesC4(NamedTuple):
    """C4 acclimated parameter predictions (leaf-top; rpmodel c4 method)."""

    vcmax25_c4_leaf: jax.Array  # [umol/m^2/s] leaf-top C4 Vcmax25
    g1_c4_kpa: jax.Array        # [kPa^0.5] predicted C4 Medlyn slope (= xi_c4)
    chi_c4: jax.Array           # [-] C4 ci/ca at the optimum (diagnostic)
    xi_c4_sqrt_pa: jax.Array    # [Pa^0.5] C4 least-cost xi (diagnostic)


def _phi0_c4(T_K: jax.Array, cfg: PModelConfig) -> jax.Array:
    """C4 apparent quantum yield [mol/mol] (rpmodel ftemp_kphio, c4 branch).

    ``kphio_c4 * max(quad_c4(Tc), 0)`` with the smooth softplus floor; when
    the temperature dependence is off, rpmodel freezes the quadratic at
    15 degC rather than dropping it (the magnitude lives in the quadratic).
    """
    a0, a1, a2 = _KPHIO_QUAD_C4
    if not cfg.kphio_temp:
        tc = jnp.full_like(T_K, _KPHIO_C4_REF_TC)
    else:
        tc = T_K - constants.T_freeze
    quad = a0 + a1 * tc + a2 * tc * tc
    quad = _PHI0_FLOOR_WIDTH * jax.nn.softplus(quad / _PHI0_FLOOR_WIDTH)
    return cfg.kphio_c4 * quad


def acclimated_capacities_c4(
    acclim: PModelAcclimState | None, cfg: PModelConfig
) -> PModelCapacitiesC4:
    """C4 acclimated leaf-top Vcmax25 and predicted Medlyn g1 (rpmodel c4).

    rpmodel's c4 method verbatim: chi/xi use the FULL kinetics (real Gamma*
    and K) with the C4 cost ratio ``beta_cost_c4`` (= 146/9); the CO2-
    saturated bundle sheath sets mj = mc = 1, so the Wang-2017 limitation
    factor is the constant ``sqrt(1 - c*^(2/3))`` and
    ``Vcmax_growth = phi0_c4(Tg) * Iabs * sqrt(1 - c*^(2/3))``.  The 25 degC
    inversion divides by the HOST Collatz kernel's RAW temperature factor
    (``c4_vcmax_temperature_response``, which is ~0.87 at 25 degC by the
    kernel's own convention), so the kernel reproduces the optimum at the
    growth temperature.  The returned value is therefore the EFFECTIVE
    25 degC base the kernel multiplies — not a literal 25 degC rate (GLM
    review) — exactly the semantics of the prescribed ``Vcmax25_C4`` it
    replaces.  No Jmax ratio: the Collatz kernel has none.
    """
    if acclim is None:
        raise ValueError(
            "P-model C4 capacities requested but no acclimation state was "
            "supplied (pmodel_acclim=None): this land lane does not carry "
            "PModelAcclimState. Initialise it (init_pmodel_acclim) or select "
            "capacity_scheme='prescribed' / g1_source='table'."
        )
    if _check_acclimation(cfg) == "mengoli":
        _require_subdaily(acclim)
        _, _, _, gamma_star, _ = optimal_chi(
            acclim.day_t_K, acclim.day_vpd_pa, acclim.day_co2_ppm,
            acclim.day_ps_pa, cfg, beta=cfg.beta_cost_c4)
        chi, _ = chi_from_xi(acclim.xi_c4_real, gamma_star, acclim.day_vpd_pa,
                             acclim.day_co2_ppm, acclim.day_ps_pa, cfg)
        return PModelCapacitiesC4(
            vcmax25_c4_leaf=acclim.vcmax25_c4_real,
            g1_c4_kpa=acclim.xi_c4_real / jnp.sqrt(1000.0), chi_c4=chi,
            xi_c4_sqrt_pa=acclim.xi_c4_real)
    return capacities_c4_at(acclim.t_mean_K, acclim.vpd_mean_pa,
                            acclim.co2_mean_ppm, acclim.ps_ema,
                            acclim.iabs_mean, cfg)


def capacities_c4_at(
    tg_k: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    iabs: jax.Array,
    cfg: PModelConfig,
) -> PModelCapacitiesC4:
    """C4 optimum (rpmodel c4 method) at GIVEN acclimation drivers."""
    from legoesm.land.canopy.photosynthesis import (
        c4_vcmax_temperature_response,
    )

    chi, xi, _ci, _gs, _K = optimal_chi(
        tg_k, vpd_pa, co2_ppm, ps_pa, cfg, beta=cfg.beta_cost_c4)
    mprime_c4 = jnp.sqrt(1.0 - C_STAR ** (2.0 / 3.0))  # mj = 1 (CO2-saturated)
    v_growth = _phi0_c4(tg_k, cfg) * iabs * mprime_c4
    vcmax25_c4 = v_growth / c4_vcmax_temperature_response(tg_k)
    g1_c4 = xi / jnp.sqrt(1000.0)
    return PModelCapacitiesC4(
        vcmax25_c4_leaf=vcmax25_c4, g1_c4_kpa=g1_c4, chi_c4=chi,
        xi_c4_sqrt_pa=xi)


def nitrogen_diagnostics(vcmax25_leaf: jax.Array,
                         jmax25_leaf: jax.Array):
    """Implied C3 photosynthetic nitrogen pools, per m2 LEAF (diagnostic only).

    Returns ``(n_rubisco, n_et)`` in g N m-2 leaf: nitrogen bound in Rubisco
    implied by ``vcmax25_leaf`` and in the electron-transport chain implied by
    ``jmax25_leaf`` (both leaf-top acclimated capacities, umol m-2 leaf s-1),
    via the LUNA nitrogen use efficiencies (see the constants block above).
    Deliberately NO total: light-capture N needs chlorophyll-allocation
    assumptions this model does not carry, and a sum omitting it would read
    as leaf N (GLM design review).  C3 only — the constants do not apply to
    C4.  Nothing in the model reads these values back; they exist for tapes
    and for a future N closure to compare against.
    """
    return (vcmax25_leaf / _F_C25_UMOL_GN_S,
            jmax25_leaf / _F_J25_UMOL_GN_S)


def acclim_trajectory(
    init: "PModelAcclimState",
    *,
    T_K: jax.Array,
    ppfd: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: "PModelConfig",
    dt: float,
    solar_hour: jax.Array | None = None,
) -> "PModelAcclimState":
    """Scan the acclimation update over a driver time series.

    ``solar_hour`` (same leading time axis) is required whenever
    :func:`needs_solar_hour` says so.

    Drivers have a leading time axis ((n_time, ...) per leaf); returns the
    PER-STEP state trajectory (leaves (n_time, ...)) so a diagnostic
    (state-free) driver can evaluate the acclimated capacities at every
    timestep with the causal running mean up to that step (offline pre-pass;
    e.g. run_ec_site --mode diagnostic).
    """
    def _f(carry, x):
        t_k, w, d_pa, ca, ps, sh = x
        new = advance_pmodel_acclim(
            carry, T_K=t_k, ppfd=w, vpd_pa=d_pa, co2_ppm=ca, ps_pa=ps,
            cfg=cfg, dt=dt, solar_hour=sh)
        return new, new

    # Promote the carry to the drivers' dtype up front: an f32 init state
    # (e.g. built from float32 NetCDF driver values) against f64 forcing
    # under JAX_ENABLE_X64 makes the scan carry change dtype across the
    # body, which lax.scan rejects.
    _dt = jnp.result_type(T_K, ppfd, vpd_pa, co2_ppm, ps_pa)
    init = jax.tree_util.tree_map(lambda x: x.astype(_dt), init)
    _, traj = jax.lax.scan(_f, init,
                           (T_K, ppfd, vpd_pa, co2_ppm, ps_pa, solar_hour))
    return traj


def advance_pmodel_acclim(
    acclim: PModelAcclimState,
    *,
    T_K: jax.Array,
    ppfd: jax.Array,
    vpd_pa: jax.Array,
    co2_ppm: jax.Array,
    ps_pa: jax.Array,
    cfg: PModelConfig,
    dt: float,
    psi_s_mpa: jax.Array | None = None,
    rsoil_leaf: jax.Array | None = None,
    solar_hour: jax.Array | None = None,
) -> PModelAcclimState:
    """One acclimation step: running means of every driver over the steps
    ``acclimation_hours`` selects, plus, under ``acclimation_scheme=
    'mengoli'``, the window accumulation and daily trait memory
    (:func:`_advance_subdaily`).  ``solar_hour`` [h] is needed whenever
    :func:`needs_solar_hour` says so.

    A plain exponential moving average with e-folding time ``acclimation_days``
    — every timestep, day and night, carries the same weight, so the means
    are the 24-hour (multi-day) averages the optimality literature sizes
    capacities for (daily / weekly / monthly means; Stocker et al. 2020).
    The light mean therefore includes the zero night values and decays
    toward zero through a polar night (capacities -> 0, finite).  The
    P-hydro soil-state means advance with the same gain when the step's
    ``psi_s_mpa``/``rsoil_leaf`` are given and the state carries them.
    """
    scheme = _check_acclimation(cfg)
    inw = close = None
    if scheme == "mengoli":
        # mengoli's window feeds the window sums only; the driver means stay
        # the plain 24 h EMA (diagnostics, C4 path), as under han / 24h.
        inw, close = acclimation_window(cfg, dt, solar_hour=solar_hour, ppfd=ppfd)
    if scheme == "mengoli" or cfg.acclimation_hours == "24h":
        gain = 1.0 - jnp.exp(-dt / acclimation_tau_s(cfg))
        since = acclim.dt_since_update_s
    else:
        inw, close = acclimation_window(cfg, dt, solar_hour=solar_hour, ppfd=ppfd)
        # Windowed half-hourly EMA: an in-window step carries the weight of
        # the time elapsed since the previous in-window step, so the memory
        # is ``acclimation_days`` of CALENDAR time (a plain dt/tau on noon
        # steps alone would stretch 15 days to ~180).
        prev = (jnp.zeros_like(T_K) if acclim.dt_since_update_s is None
                else acclim.dt_since_update_s)
        elapsed = prev + dt
        gain = jnp.where(inw, 1.0 - jnp.exp(-elapsed / acclimation_tau_s(cfg)), 0.0)
        since = jnp.where(inw, 0.0, elapsed)
    step = lambda mean, x: mean + gain * (x - mean)  # noqa: E731
    soil = {}
    if acclim.psi_s_mean_mpa is not None:
        if psi_s_mpa is None or rsoil_leaf is None:
            raise ValueError(
                "advance_pmodel_acclim: the acclimation state carries the "
                "P-hydro soil-state means but this step supplied no "
                "psi_s_mpa / rsoil_leaf; they would silently freeze.")
        soil = dict(psi_s_mean_mpa=step(acclim.psi_s_mean_mpa, psi_s_mpa),
                    rsoil_mean=step(acclim.rsoil_mean, rsoil_leaf))
    new = acclim._replace(
        iabs_mean=step(acclim.iabs_mean, ppfd),
        t_mean_K=step(acclim.t_mean_K, T_K),
        vpd_mean_pa=step(acclim.vpd_mean_pa, vpd_pa),
        co2_mean_ppm=step(acclim.co2_mean_ppm, co2_ppm),
        ps_ema=step(acclim.ps_ema, ps_pa),
        dt_since_update_s=since,
        **soil,
    )
    if scheme == "han":
        return new
    return _advance_subdaily(new, acclim, T_K=T_K, ppfd=ppfd, vpd_pa=vpd_pa,
                             co2_ppm=co2_ppm, ps_pa=ps_pa, inw=inw,
                             close=close, cfg=cfg, dt=dt)
