"""Canonical Farquhar-von Caemmerer-Berry (FvCB) photosynthesis for the two-leaf canopy.

Implements the canonical FvCB C3 + C4 equations (Bonan, "Climate Change and
Terrestrial Ecosystem Modeling" 2019, ch. 11; cross-checked against the CLM5
photosynthesis tech note §2.9).  Ported from DifferBESS
``process/Photosynthesis_FvCB.py``.

C3 differs from the earlier BESS-v2 P-model path that legoESM forked in three
structural ways:

1. **Jmax-bounded electron transport** (Bonan eq. 11.21-11.24): a Jmax-limited
   ``J`` (smaller root of the electron-transport quadratic) replaces the
   unbounded ``alf * APAR * (Pi - Gamma*) / (Pi + 2 Gamma*)`` and the
   ``JS = Vcmax/2`` placeholder.
2. **Kattge & Knorr (2007) acclimation** (Bonan eq. 11.62-11.64): peaked
   Arrhenius for *both* Vcmax and Jmax, with ``Jmax25/Vcmax25`` itself
   acclimated to growth temperature.
3. **Dark respiration** follows Tjoelker (2001) + Atkin (2008) with
   ``Rd0 = 0.015 * Vcmax25`` — the *25 degC* value.  This removes the previous
   double counting where ``Rd = 0.015 * Vcmax(T) * rd_response(T)`` applied the
   temperature response twice (inflating Rd by ~1.9x at 35 degC).

C4 follows Collatz (1992) / SiB2 / Bonan §11.7 with CLM5-aligned constants
(high-T deactivation at 313.15 K, low-T slope 0.2, ``Rd25 = 0.025 * Vcmax25``,
quantum yield ``alpha = 0.05``).

Public entry points (``c3_photosynthesis``, ``c4_photosynthesis``,
``photosynthesis``, ``vcmax_temperature_response``) keep the legoESM signatures
so the solver / bundle call sites are unchanged; ``alf`` and ``Ps`` are accepted
for parity but unused by the canonical C3 (which uses the PSII quantum yield and
keeps Ci in mole-fraction units throughout).

All functions are pure JAX, JIT-compatible, and differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.leaf_biophysics import (
    GAMMA_STAR25_UMOL_MOL,
    HA_GAMMA,
    HA_KC,
    HA_KO,
    KC25_UMOL_MOL,
    KO25_UMOL_MOL,
    O2_UMOL_MOL,
    T_REF_K,
    arrhenius_factor,
    peaked_arrhenius_factor,
)


# --- reference temperatures ---
# Gas constant + 25 degC reference come from leaf_biophysics (shared with the
# big-leaf land/stomata FvCB path); the gas constant was previously hardcoded
# as 8.314 here, which both drifted from constants.R_universal and duplicated
# the sibling path's Arrhenius helper.
_T_REF = T_REF_K    # [K] reference temperature (25 degC)
_T0_C = 25.0        # [degC] reference for dark respiration

# --- quadratic curvatures (CLM5 default / Sellers 1996b / Bonan ch. 11) ---
_THETA_J = 0.7         # electron-transport quadratic
_THETA_CJA_C3 = 0.98   # C3 stage-1 colimitation (Ac + Aj)  (Bonan eq. 11.33)
_THETA_IP_C3 = 0.95    # C3 stage-2 colimitation (Ai + Ap)
_THETA_CJA_C4 = 0.80   # C4 stage-1 colimitation
_THETA_IP_C4 = 0.95    # C4 stage-2 colimitation

# --- C3 photosystem (Bonan eq. 11.23, CLM5 §2.9) ---
_PHI_PSII = 0.85       # PSII quantum yield

# --- Bernacchi (2001) Rubisco kinetics at 25 degC (shared block, umol/mol) ---
_KC25 = KC25_UMOL_MOL          # [umol mol-1] Michaelis constant for CO2
_KO25 = KO25_UMOL_MOL          # [umol mol-1] Michaelis constant for O2
_GS25 = GAMMA_STAR25_UMOL_MOL  # [umol mol-1] Gamma_star at 25 degC
_OI = O2_UMOL_MOL              # [umol mol-1] intercellular O2 (atmospheric)

# --- activation/deactivation energies [J mol-1] (Bernacchi 2001 + Kattge & Knorr 2007) ---
_HA_KC = HA_KC
_HA_KO = HA_KO
_HA_GS = HA_GAMMA
_HA_VCMAX = 72000.0    # Kattge & Knorr 2007 (CLM5 Table 2.9.2)
_HA_JMAX = 50000.0     # Kattge & Knorr 2007 (CLM5 Table 2.9.2)
_HD_VCMAX = 200000.0
_HD_JMAX = 200000.0

# --- Kattge & Knorr (2007) acclimation linear fits (Bonan eq. 11.62-11.64) ---
# Vcmax/Jmax entropy terms dS [J K-1 mol-1] and the Jmax25/Vcmax25 ratio, each a
# linear function of growth temperature TgC [degC]. Values verbatim from the fit.
_DS_VCMAX_INTERCEPT = 668.39   # [J K-1 mol-1]
_DS_VCMAX_SLOPE     = 1.07     # [J K-1 mol-1 degC-1]
_DS_JMAX_INTERCEPT  = 659.70   # [J K-1 mol-1]
_DS_JMAX_SLOPE      = 0.75     # [J K-1 mol-1 degC-1]
_JV_RATIO_INTERCEPT = 2.59     # [-] Jmax25/Vcmax25 at TgC = 0 degC
_JV_RATIO_SLOPE     = 0.035    # [degC-1]

# --- Tjoelker (2001) + Atkin (2008) dark respiration (Bonan eq. 11.65-11.67) ---
_RD0_VCMAX_FRAC_C3 = 0.015   # Rd0 / Vcmax25 (basal rate at 25 degC)
_Q10_INTERCEPT = 3.22
_Q10_SLOPE = 0.046
_ATKIN_SLOPE = 0.00794

# --- C4 kinetics (Bonan eq. 11.72-11.74; CLM5 §2.9.5) ---
_Q10_C4 = 2.0
_S1_C4 = 0.3
_S2_C4 = 313.15    # [K] high-T deactivation onset (40 degC; CLM5)
_S3_C4 = 0.2       # low-T inhibition slope
_S4_C4 = 288.15    # [K]
_S5_C4 = 1.3
_S6_C4 = 328.15    # [K] high-T respiration cutoff
_ALPHA_C4 = 0.05       # quantum yield E (Bonan §11.7; CLM5)
_KP25_FRAC = 0.02      # kp25 / Vcmax25 (PEP carboxylase)
_RD25_FRAC_C4 = 0.025  # Rd25 / Vcmax25 (CLM5 §2.9.5)

# --- clip ranges (numerical / calibration safety) ---
_TGC_LO = 11.0     # [degC] Kattge & Knorr 2007 acclimation calibration range
_TGC_HI = 35.0
_TF_C_LO = 5.0     # [degC] instantaneous-T clip keeping Tjoelker Q10 > 0
_TF_C_HI = 45.0


# ---------------------------------------------------------------------------
# Temperature-response helpers
# ---------------------------------------------------------------------------

def _arrhenius(Tf: jax.Array, dHa: float) -> jax.Array:
    """Arrhenius temperature response (Bonan eq. 11.34), normalised to 1 at 25 degC.

    Thin wrapper over the shared ``leaf_biophysics.arrhenius_factor`` so the
    two-leaf and big-leaf photosynthesis paths share one Arrhenius definition
    and one gas constant.
    """
    return arrhenius_factor(Tf, dHa)


def _arrhenius_peaked(Tf: jax.Array, dHa: float, dHd: float, dS: jax.Array) -> jax.Array:
    """Peaked Arrhenius (Bonan eq. 11.34 * 11.36), normalised to 1 at 25 degC.

    Thin wrapper over the shared ``leaf_biophysics.peaked_arrhenius_factor``.
    """
    return peaked_arrhenius_factor(Tf, dHa, dHd, dS)


def _smaller_root_quadratic(theta: float, A: jax.Array, B: jax.Array,
                            eps: float = 1e-12) -> jax.Array:
    """Smaller root of ``theta x^2 - (A + B) x + A B = 0``.

    Used for the electron-transport quadratic (theta=Theta_j, A=I_PSII, B=Jmax)
    and for the two colimitation stages.  ``eps`` guards the sqrt against tiny
    negative arguments from floating-point cancellation when A~0 or B~0.
    """
    s = A + B
    disc = s * s - 4.0 * theta * A * B
    return (s - jnp.sqrt(jnp.maximum(disc, eps))) / (2.0 * theta)


def _delta_s_vcmax(TgC_a: jax.Array) -> jax.Array:
    """Vcmax entropy term (Bonan eq. 11.62, Kattge & Knorr 2007). [J K-1 mol-1]"""
    return _DS_VCMAX_INTERCEPT - _DS_VCMAX_SLOPE * TgC_a


def _delta_s_jmax(TgC_a: jax.Array) -> jax.Array:
    """Jmax entropy term (Bonan eq. 11.63, Kattge & Knorr 2007). [J K-1 mol-1]"""
    return _DS_JMAX_INTERCEPT - _DS_JMAX_SLOPE * TgC_a


def _jmax25_over_vcmax25(TgC_a: jax.Array) -> jax.Array:
    """Acclimated Jmax25/Vcmax25 ratio (Bonan eq. 11.64, Kattge & Knorr 2007)."""
    return _JV_RATIO_INTERCEPT - _JV_RATIO_SLOPE * TgC_a


def _q10_tjoelker(Tf: jax.Array) -> jax.Array:
    """Variable Q10 for dark respiration (Bonan eq. 11.66, Tjoelker 2001).

    Tf clipped to [5, 45] degC keeps Q10 strictly positive (Q10 > 1.15 at
    45 degC) and avoids runaway exponentials at extreme leaf temperatures.
    """
    T_C = jnp.clip(Tf - constants.T_freeze, _TF_C_LO, _TF_C_HI)
    return _Q10_INTERCEPT - _Q10_SLOPE * T_C


def _rd_atkin(Tf: jax.Array, TgC_a: jax.Array, Vcmax25: jax.Array) -> jax.Array:
    """Dark respiration (Bonan eq. 11.67, Atkin 2008). [umol m-2 s-1]

    Basal ``Rd0 = 0.015 * Vcmax25`` (the 25 degC value — NOT Vcmax(T), which
    would double-count the temperature response).  The growth-adjust term lowers
    the basal rate with warmer growth temperature (Atkin 2008); the Q10 term
    gives the instantaneous response (Tjoelker 2001 variable Q10).  Equals
    ``0.015 * Vcmax25`` exactly at Tf = 25 degC, TgC = 25 degC.
    """
    T_C = jnp.clip(Tf - constants.T_freeze, _TF_C_LO, _TF_C_HI)
    Q10 = _q10_tjoelker(Tf)
    growth_adjust = jnp.power(10.0, -_ATKIN_SLOPE * (TgC_a - _T0_C))
    instantaneous = jnp.power(Q10, (T_C - _T0_C) / 10.0)
    Rd0 = _RD0_VCMAX_FRAC_C3 * Vcmax25
    return growth_adjust * Rd0 * instantaneous


def co2_compensation_point(Tf: jax.Array) -> jax.Array:
    """CO2 compensation point in the absence of dark respiration, Gamma* [umol/mol].

    Bernacchi (2001) Arrhenius response of ``_GS25`` (= 42.75 umol/mol at 25 degC).
    Exposed for reuse by the SIF diagnostic (``canopy/sif.py``), which needs the
    same Gamma* the FvCB C3 electron-transport / Aj rates use — computed here so
    the two never drift.
    """
    return _GS25 * _arrhenius(Tf, _HA_GS)


def vcmax_temperature_response(Tf: jax.Array, TgC: jax.Array) -> jax.Array:
    """Normalised Vcmax temperature response (Kattge & Knorr 2007 peaked Arrhenius).

    Returns f(T) such that ``Vcmax(T) = f(T) * Vcmax25``; f == 1 at 25 degC.

    Parameters
    ----------
    Tf  : leaf temperature [K]
    TgC : growth temperature [degC] (clipped to the K&K calibration range)
    """
    TgC_a = jnp.clip(TgC, _TGC_LO, _TGC_HI)
    return _arrhenius_peaked(Tf, _HA_VCMAX, _HD_VCMAX, _delta_s_vcmax(TgC_a))


# ---------------------------------------------------------------------------
# C3 photosynthesis (canonical FvCB; Bonan ch. 11 / CLM5 §2.9)
# ---------------------------------------------------------------------------

class LeafAssimilation(NamedTuple):
    """Gross assimilation + dark respiration components of one FvCB solve.

    ``a_gross`` is the co-limited GROSS assimilation A (before dark respiration)
    and ``rd`` the dark respiration [both umol m-2 s-1]; net An = a_gross - rd.
    Exposed so the big-leaf ``land/stomata`` path can report GROSS primary
    production (GPP = a_gross) and couple stomata on net, sharing the EXACT
    canonical biochemistry with the two-leaf ``c3_photosynthesis`` (which just
    returns ``max(a_gross - rd, 0)``).
    """
    a_gross: jax.Array
    rd: jax.Array


def c3_assimilation(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25: jax.Array,
    TgC: jax.Array,
) -> LeafAssimilation:
    """Canonical FvCB C3 gross assimilation + dark respiration (Bonan ch. 11).

    The shared biochemistry kernel behind both :func:`c3_photosynthesis` (net,
    floored) and the big-leaf coupled A-gs solver.  Returns the GROSS co-limited
    rate ``A`` and dark respiration ``Rd`` separately (see :class:`LeafAssimilation`).

    Parameters as :func:`c3_photosynthesis` minus the parity-only ``Ps``/``alf``.
    """
    # Acclimation only valid for TgC in [11, 35] degC; clip to the boundary
    # acclimation state outside (avoid unphysical Kattge & Knorr extrapolation).
    TgC_a = jnp.clip(TgC, _TGC_LO, _TGC_HI)

    # Acclimation (Kattge & Knorr 2007)
    dS_v = _delta_s_vcmax(TgC_a)
    dS_j = _delta_s_jmax(TgC_a)
    Jmax25 = _jmax25_over_vcmax25(TgC_a) * Vcmax25

    # Instantaneous T-response of kinetic constants and capacities
    Kc = _KC25 * _arrhenius(Tf, _HA_KC)
    Ko = _KO25 * _arrhenius(Tf, _HA_KO)
    GammaStar = co2_compensation_point(Tf)
    Vcmax = Vcmax25 * _arrhenius_peaked(Tf, _HA_VCMAX, _HD_VCMAX, dS_v)
    Jmax = Jmax25 * _arrhenius_peaked(Tf, _HA_JMAX, _HD_JMAX, dS_j)

    # Dark respiration (Tjoelker + Atkin), basal at 0.015 * Vcmax25
    Rd = _rd_atkin(Tf, TgC_a, Vcmax25)

    # Electron transport: smaller root of theta_j J^2 - (I + Jmax) J + I Jmax = 0
    I_PSII = 0.5 * _PHI_PSII * APAR
    J = _smaller_root_quadratic(_THETA_J, I_PSII, Jmax)

    # Limited rates (Bonan eq. 11.28, 11.29, 11.32)
    Ci_safe = jnp.maximum(Ci, 1e-3)  # coeff-ok: div-by-zero guard on intercellular CO2 (mole fraction)
    Ac = Vcmax * (Ci_safe - GammaStar) / (Ci_safe + Kc * (1.0 + _OI / Ko))
    Aj = (J / 4.0) * (Ci_safe - GammaStar) / (Ci_safe + 2.0 * GammaStar)
    Ap = 0.5 * Vcmax

    # When Ci < Gamma*, Ac and Aj go negative; floor at 0 to keep the
    # smaller-root quadratic well-conditioned.  Net A is then driven by -Rd.
    Ac = jnp.maximum(Ac, 0.0)
    Aj = jnp.maximum(Aj, 0.0)

    # Co-limitation (Bonan eq. 11.33 / CLM5 eq. 2.9.8)
    Ai = _smaller_root_quadratic(_THETA_CJA_C3, Ac, Aj)
    A = _smaller_root_quadratic(_THETA_IP_C3, Ai, Ap)
    return LeafAssimilation(a_gross=A, rd=Rd)


@jax.jit
def c3_photosynthesis(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25: jax.Array,
    Ps: jax.Array,
    alf: jax.Array,
    TgC: jax.Array,
) -> jax.Array:
    """Net assimilation rate for canonical FvCB C3 photosynthesis.

    Parameters
    ----------
    Tf      : leaf temperature [K]
    Ci      : intercellular CO2 mole fraction [umol mol-1]
    APAR    : absorbed PAR [umol m-2 s-1]
    Vcmax25 : maximum carboxylation rate at 25 degC [umol m-2 s-1]
    Ps      : surface pressure [Pa] — accepted for parity, unused (mole-fraction Ci)
    alf     : legacy electron-transport quantum yield — accepted for parity, unused
              (FvCB uses the PSII quantum yield _PHI_PSII)
    TgC     : growth temperature [degC] for Kattge & Knorr acclimation

    Returns
    -------
    An : net assimilation rate [umol m-2 s-1], clamped to >= 0
    """
    del Ps, alf  # signature parity; FvCB uses _PHI_PSII and mole-fraction Ci
    r = c3_assimilation(Tf, Ci, APAR, Vcmax25, TgC)
    An = r.a_gross - r.rd
    return jnp.where(An < 0.0, 0.0, An)


# ---------------------------------------------------------------------------
# C4 photosynthesis (Collatz 1992 / SiB2 / Bonan §11.7; CLM5-aligned)
# ---------------------------------------------------------------------------

def c4_assimilation(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25: jax.Array,
) -> LeafAssimilation:
    """Canonical Collatz (1992)/SiB2 C4 gross assimilation + dark respiration.

    Shared kernel behind :func:`c4_photosynthesis` (net, floored) and the
    big-leaf C3/C4-blended solver.  Returns GROSS ``A`` and ``Rd`` separately.
    """
    item = (Tf - _T_REF) / 10.0
    q10_pow = jnp.power(_Q10_C4, item)

    Vcmax_o = Vcmax25 * q10_pow
    fH = 1.0 + jnp.exp(_S1_C4 * (Tf - _S2_C4))
    fL = 1.0 + jnp.exp(_S3_C4 * (_S4_C4 - Tf))
    Vcmax = Vcmax_o / (fH * fL)

    Rd25 = _RD25_FRAC_C4 * Vcmax25
    Rd = Rd25 * q10_pow / (1.0 + jnp.exp(_S5_C4 * (Tf - _S6_C4)))

    kp25 = _KP25_FRAC * Vcmax25
    kp = kp25 * q10_pow

    # Limited rates (Bonan eq. 11.69-11.71).  kp [mol m-2 s-1] * Ci [umol mol-1]
    # * 1e-6 [mol/umol] -> mol m-2 s-1, then * 1e6 to umol m-2 s-1: the two
    # unit factors cancel, so Ap = kp * Ci directly.
    Ac = Vcmax
    Aj = _ALPHA_C4 * APAR
    Ap = kp * Ci

    Ai = _smaller_root_quadratic(_THETA_CJA_C4, Ac, Aj)
    A = _smaller_root_quadratic(_THETA_IP_C4, Ai, Ap)
    return LeafAssimilation(a_gross=A, rd=Rd)


@jax.jit
def c4_photosynthesis(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25: jax.Array,
) -> jax.Array:
    """Net assimilation rate for canonical FvCB C4 photosynthesis.

    Parameters
    ----------
    Tf      : leaf temperature [K]
    Ci      : intercellular CO2 mole fraction [umol mol-1]
    APAR    : absorbed PAR [umol m-2 s-1]
    Vcmax25 : maximum carboxylation rate at 25 degC [umol m-2 s-1]

    Returns
    -------
    An : net assimilation rate [umol m-2 s-1], clamped to >= 0
    """
    r = c4_assimilation(Tf, Ci, APAR, Vcmax25)
    An = r.a_gross - r.rd
    return jnp.where(An < 0.0, 0.0, An)


# ---------------------------------------------------------------------------
# Mixed C3/C4 photosynthesis
# ---------------------------------------------------------------------------

@jax.jit
def photosynthesis(
    Tf: jax.Array,
    Ci: jax.Array,
    APAR: jax.Array,
    Vcmax25_C3: jax.Array,
    Vcmax25_C4: jax.Array,
    fC4: jax.Array,
    Ps: jax.Array,
    alf: jax.Array,
    TgC: jax.Array,
) -> jax.Array:
    """Net assimilation for a mixed C3/C4 canopy.

    Computes both C3 and C4 rates and combines them using a continuous
    fraction fC4, so the result is differentiable with respect to fC4.

    Parameters
    ----------
    Tf, Ci, APAR : leaf temperature [K], intercellular CO2 [umol/mol],
                   absorbed PAR [umol m-2 s-1]
    Vcmax25_C3, Vcmax25_C4 : canopy-integrated Vcmax25 [umol m-2 s-1]
    fC4     : C4 area fraction [0-1] — traced value, not branched
    Ps      : surface pressure [Pa] — parity only (unused by FvCB C3)
    alf     : legacy quantum yield — parity only (unused by FvCB C3)
    TgC     : growth temperature [degC] for acclimation

    Returns
    -------
    An : net assimilation rate [umol m-2 s-1]
    """
    An_C3 = c3_photosynthesis(Tf, Ci, APAR, Vcmax25_C3, Ps, alf, TgC)
    An_C4 = c4_photosynthesis(Tf, Ci, APAR, Vcmax25_C4)
    # Continuous weighted average — fully differentiable wrt fC4
    return (1.0 - fC4) * An_C3 + fC4 * An_C4


__physics_contract__ = {
    "summary": (
        "Canonical Farquhar-von Caemmerer-Berry (FvCB) C3 + C4 leaf "
        "photosynthesis for the two-big-leaf canopy (Bonan 2019 ch. 11; CLM5 "
        "§2.9). C3 uses a Jmax-bounded electron-transport rate and Kattge & "
        "Knorr (2007) peaked-Arrhenius acclimation of both Vcmax and Jmax "
        "(Jmax25/Vcmax25 = 2.59 - 0.035*TgC). Dark respiration follows Tjoelker "
        "(2001) + Atkin (2008) with Rd0 = 0.015*Vcmax25 (the 25 degC value, no "
        "double counting of the Vcmax T-response). C4 follows Collatz (1992)/"
        "SiB2 with CLM5 constants. Mixed canopies blend C3/C4 continuously by "
        "fC4. ported from DifferBESS process/Photosynthesis_FvCB.py."
    ),
    "inputs": {
        "Tf": "K", "Ci": "umol/mol", "APAR": "umol/m^2/s",
        "Vcmax25": "umol/m^2/s", "TgC": "degC", "fC4": "1",
    },
    "outputs": {"An": "umol/m^2/s"},
    "sign_convention": (
        "An >= 0 (gross assimilation minus dark respiration, floored at 0). "
        "Light- (Aj), Rubisco- (Ac) and product- (Ap) limited rates are each "
        ">= 0; net An is gross A minus Rd."
    ),
    "conserves": [],  # leaf-level rate, not a conservation law
    "differentiable": True,
    "reference": (
        "Farquhar, von Caemmerer & Berry (1980) Planta 149 78-90; Bernacchi+ "
        "(2001) Plant Cell Environ. 24 253-259; Kattge & Knorr (2007) Plant "
        "Cell Environ. 30 1176-1190; Tjoelker+ (2001) Glob. Change Biol. 7 "
        "223-230; Atkin+ (2008); Collatz+ (1992) Aust. J. Plant Physiol. 19 "
        "519-538; Bonan (2019) ch. 11; CLM5 tech note §2.9."
    ),
    "idealized_test": (
        "tests/land/unit/test_canopy_photosynthesis.py: An > 0 at light- and "
        "CO2-saturated 25 degC; An == 0 in the dark; Vcmax response peaks near "
        "25-30 degC; Rd0 == 0.015*Vcmax25 at 25 degC (no double count); "
        "Jmax25/Vcmax25 acclimation ratio; CLM5 C4 constants; high-APAR "
        "saturation (Jmax bound); finite grad wrt Vcmax25."
    ),
}
