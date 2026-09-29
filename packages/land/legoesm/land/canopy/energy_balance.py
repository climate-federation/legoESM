"""Leaf and soil energy balance for the two-leaf canopy model.

Two schemes are provided:
  - BT (Bulk Transfer): direct specific-humidity-gradient formulation.
      LE = λ ρ (q_f - q_c) / (Rb + rs)
    Computationally efficient; default.
  - PM (Penman-Monteith): second-order Paw & Gao (1988) solution.
    More physical but requires second derivatives of sat. vapour pressure.

Canopy air-space update: conductance-weighted mixing of heat/vapour from
leaves, soil, and free atmosphere.

All functions are pure JAX, JIT-compatible, and differentiable.
Sources: DifferBESS/process/CanopyEnergyBalance.py, Soil.py, CarbonWaterFluxes.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import (
    saturation_vapor_pressure_aerk,
    d_saturation_vapor_pressure_aerk,
    dd_saturation_vapor_pressure_aerk,
    vapor_pressure_from_specific_humidity,
)
from legoesm.land.leaf_biophysics import DIFFUSIVITY_RATIO_H2O_CO2
from legoesm.land.stomata import ball_berry_gs, medlyn_gs

# Module-local constants.
# NOTE: Stefan-Boltzmann, freezing point, latent heat of vaporisation, etc.
# are imported from ``legoesm.constants`` — do not redefine them here.
# The leaf H2O:CO2 diffusivity ratio (Fick's law; Ci = Ca − ratio·An/gs) is the
# single-source DIFFUSIVITY_RATIO_H2O_CO2 imported from leaf_biophysics above.
_Ps0   = 101325.0    # IUPAC STP pressure [Pa] used in the mol → m/s
                     # unit conversion factor _CF_MOLAR_VOLUME; distinct from
                     # ``constants.p_ref`` (1e5 Pa hydrostatic reference).
# mol m-2 s-1 → m s-1 leaf-conductance prefactor at IUPAC STP (encodes the
# reference molar volume 22.4 L/mol); scaled by (T_freeze/Tf)·(Ps/_Ps0).
_CF_MOLAR_VOLUME = 0.446

# Latent-heat-of-vaporisation temperature slope −dλ/dT [J kg-1 K-1], used as
# λ(T) = L_v − _LAMBDA_T_SLOPE·(T − T_freeze). DifferBESS canopy value; distinct
# from ``constants.L_v_sst_slope`` (2.370e3) — keep the canopy value verbatim.
_LAMBDA_T_SLOPE = 2.361e3

# Minimum cuticular (residual) stomatal conductance [mol m-2 s-1].  Stomata
# never fully close — the leaf cuticle always leaks a little — so the conductance
# has a small floor.  Numerically this is essential under the legacy default
# ``CanopyConfig.stress_b0=True``: there the stress factor scales the Ball-Berry
# slope AND intercept to zero (m = b0 = 0 ⇒ gs = 0), which makes the leaf
# gs/Ci/An subsystem degenerate and the canopy Newton Jacobian singular → NaN
# fluxes (seen at dry FLUXNET sites, e.g. US-Ton savanna at SWC near wilting
# point).  With ``stress_b0=False`` b0 stays > 0, so gs is already floored above
# this value and the guard is inactive.  Value matches the DifferBESS ``g0``
# default (CarbonWaterFluxes.py).  Binds only at near-complete stomatal closure.
_GS_MIN_MOL = 1.0e-4

# Saturation vapour pressure and its first/second temperature derivatives come
# from the shared ``legoesm.thermo`` Alduchov-Eskridge (1996) AERK water + AERKi
# ice blend (``*_aerk``), matching the DifferBESS two-big-leaf canopy oracle.
# Using the curve's OWN analytic derivatives keeps des/dT, d2es/dT2 consistent
# with e_s (a mismatched parameterisation makes the PM closure humidity-dependently
# wrong) and gives a proper over-ice branch below freezing (the over-water Magnus
# over-estimates e_s by ~10-60 % at -10..-50 degC).


# ---------------------------------------------------------------------------
# Energy-balance latent-heat (LE) cap
# ---------------------------------------------------------------------------
# Bound LE to the available energy so the leaf-temperature Newton solve stays
# well-posed under hot/dry/high-VPD forcing.  Without a cap, the large dq_sat/dT
# makes the leaf energy-balance Jacobian stiff; the Newton update
# ``Tf_new = Tc + Rb (Rn - LE) / (rho Cp)`` then overshoots, ``q_sat(Tf)``
# overflows, and the leaf temperature goes NaN (observed at sparse/dry FLUXNET
# sites — US-Ton savanna here, US-Wkg grassland in DifferBESS).
#
# ``soft`` is a smooth (softplus) UPPER bound ``LE <= max(Rn,0) + slack(Rn)`` with
# a radiation-gated slack (wide by day, tight at night).  It preserves exactly the
# physics the hard clamp ``LE in [0,max(Rn,0)]`` was removed to keep — negative LE
# (dew/condensation) passes through unchanged (it is bounded by the humidity
# gradient and cannot run away, so it needs no floor) and a moderate daytime LE>Rn
# is allowed — while stopping the positive runaway.  No lower bound is applied: one
# would map a zero raw flux to a small POSITIVE LE (spurious evaporation from
# bone-dry soil / closed stomata).  ``max(Rn,0)`` is softplus-smoothed so the
# bound has no Jacobian kink at Rn=0 (Rn depends on the Newton state via longwave).
# Slack coefficients are the DifferBESS ``CanopyEnergyBalance`` soft-cap settings.
_LE_CAP_DAY_SLACK_WM2     = 80.0    # extra LE budget above Rn when Rn is large
_LE_CAP_NIGHT_SLACK_WM2   = 30.0    # LE upper-bound slack at night (Rn <= 0)
_LE_CAP_RN_TRANSITION_WM2 = 100.0   # Rn over which the day/night slack saturates
_LE_CAP_SOFTNESS_PER_WM2  = 0.1     # softplus softness scale (~10 W m-2)
_LE_CAP_MODES = ("soft", "hard", "off")


def apply_le_cap(LE: jax.Array, Rn: jax.Array,
                 le_cap_mode: str = "soft") -> jax.Array:
    """Bound the latent-heat flux to the available energy (see module notes).

    ``le_cap_mode`` is a static Python string resolved at trace time:

    * ``"soft"`` — smooth softplus UPPER bound ``LE <= max(Rn,0)+slack(Rn)`` with a
      radiation-gated slack (wide by day, tight at night).  Default; stops the
      positive-LE runaway that diverges the leaf-T Newton solve.  A zero raw
      flux stays exactly 0 (no lower bound); elsewhere the output is shifted
      by at most ln(1+exp(-k*cap_hi))/k (<= 0.5 W m-2 at the 30 W m-2 night
      slack), e.g. dew LE = -50 -> -49.5, and the upper asymptote is
      softplus(k*cap_hi)/k, that much above cap_hi.
    * ``"hard"`` — legacy ``clip(LE, 0, max(Rn,0))`` (non-smooth; forces H>=0).
    * ``"off"`` — no cap (pre-regression behaviour; can diverge at dry sites).
    """
    if le_cap_mode == "off":
        return LE
    if le_cap_mode == "hard":
        return jnp.clip(LE, 0.0, jnp.maximum(Rn, 0.0))
    if le_cap_mode == "soft":
        k = _LE_CAP_SOFTNESS_PER_WM2
        rn_pos = jax.nn.softplus(Rn * k) / k          # smooth max(Rn, 0): no kink
        gate = jnp.minimum(rn_pos / _LE_CAP_RN_TRANSITION_WM2, 1.0)
        slack = _LE_CAP_NIGHT_SLACK_WM2 + (
            _LE_CAP_DAY_SLACK_WM2 - _LE_CAP_NIGHT_SLACK_WM2) * gate
        cap_hi = rn_pos + slack
        # Upper bound only — see module notes (no lower/dew bound).  Shifted so
        # a zero raw flux maps to EXACTLY zero: the plain form
        # cap_hi - softplus(k (cap_hi - LE))/k leaks -ln(1+exp(-k cap_hi))/k
        # (-0.1..-0.4 W m-2) at LE = 0, a leaf-area-independent spurious dew
        # that the leaf balance divides by a conductance proportional to LAI:
        # sparse leaves ran tens of K hot and the canopy solve stalled.
        return (jax.nn.softplus(cap_hi * k)
                - jax.nn.softplus((cap_hi - LE) * k)) / k
    raise ValueError(
        f"unknown le_cap_mode {le_cap_mode!r}; expected one of {_LE_CAP_MODES}")


# ---------------------------------------------------------------------------
# Meteorological helpers
# ---------------------------------------------------------------------------

@jax.jit
def saturation_specific_humidity(T: jax.Array, p: jax.Array) -> jax.Array:
    """Saturation specific humidity [kg kg-1] from T [K] and p [Pa].

    Uses the shared ``legoesm.thermo`` AERK water+ice saturation curve
    (``saturation_vapor_pressure_aerk``; CLAUDE.md: never inline Tetens),
    matching the canopy's other ``e_s`` evaluations.  The specific-humidity
    denominator ``p - (1 - ε) e_s`` differs from the mixing-ratio
    denominator in ``thermo.saturation_mixing_ratio`` — this function
    returns **specific** humidity, which is what the canopy air and
    leaf boundary layers carry throughout the two-leaf closure.
    """
    e_s = saturation_vapor_pressure_aerk(T)
    return constants.epsilon * e_s / (p - (1.0 - constants.epsilon) * e_s)


@functools.partial(jax.jit, static_argnames=("rh_cap_width",))
def canopy_met_variables(
    Ps: jax.Array,
    Tc: jax.Array,
    q_c: jax.Array,
    rh_cap_width: float,
) -> tuple[jax.Array, ...]:
    """Meteorological variables for the canopy air space.

    Parameters
    ----------
    Ps  : atmospheric pressure [Pa]
    Tc  : canopy air temperature [K]
    q_c : canopy air specific humidity [kg kg-1]
    rh_cap_width : width [-] of the smooth cap RH_c <= 1
        (``CanopyConfig.rh_cap_smoothing_width``); see the RH_c line below.

    Returns
    -------
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma
      All in Pa (or Pa K-1 for derivatives; Pa K-2 for second derivative).
    """
    # Vapour pressure from specific humidity
    e_c  = vapor_pressure_from_specific_humidity(q_c, Ps)
    TcC  = Tc - constants.T_freeze
    # Saturation vapour pressure + its analytic derivatives from the shared AERK
    # water+ice curve (one consistent curve; over-ice below freezing).  ddesTc
    # uses the saturation curve only (no actual vapour pressure) — the historical
    # PM ``e_c``-instead-of-``e_s`` second-derivative bug is fixed at the source.
    es_c   = saturation_vapor_pressure_aerk(Tc)
    desTc  = d_saturation_vapor_pressure_aerk(Tc)    # des/dT  [Pa K-1]
    ddesTc = dd_saturation_vapor_pressure_aerk(Tc)   # d²es/dT² [Pa K-2]

    VPD_c = es_c - e_c
    # Smooth cap RH_c = r - w*softplus((r - 1)/w) -> 1 as r -> inf, instead of
    # clip(r, 0, 1).  A saturated canopy air space (warm wet ground under a
    # canopy) put the hard cap's kink on the Ball-Berry gs -> Ci rows of the
    # canopy Newton solve, which then stalled (replay of 857 stalled production
    # columns: 105 -> 697 converge).  Bias -w*ln2 at r = 1, -5e-4 at r = 0.97.
    _r = e_c / jnp.maximum(es_c, 1e-6)
    # r >= 0, so RH_c >= -w*exp(-1/w) (a denormal); no lower clip (no kink).
    RH_c  = _r - rh_cap_width * jax.nn.softplus((_r - 1.0) / rh_cap_width)

    # Latent heat (temperature-corrected) and psychrometric constant
    lam   = constants.L_v - _LAMBDA_T_SLOPE * TcC
    gamma = constants.c_pd / constants.epsilon * Ps / lam   # [Pa K-1]

    return e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma


# ---------------------------------------------------------------------------
# Stomatal conductance dispatch (Ball-Berry or Medlyn)
# ---------------------------------------------------------------------------

def _compute_gs_and_ci(
    An: jax.Array,
    RH_c: jax.Array,
    VPD_c: jax.Array,
    Ca: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    stomatal_model: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Stomatal conductance and intercellular CO2 closure for a leaf.

    Dispatches between Ball-Berry (RH-based) and Medlyn (VPD-based) based
    on the ``stomatal_model`` static argument.  Both models use the same
    per-leaf ``(m, b0)`` pair — ``m`` is interpreted as the Ball-Berry
    slope ``g1_bb`` for ``"ball_berry"`` and as the Medlyn slope ``g1_med``
    [kPa^0.5] for ``"medlyn"``; ``b0`` is the residual conductance in both
    cases.

    Parameters
    ----------
    An     : net assimilation [μmol CO2 / m^2 / s]
    RH_c   : canopy-air relative humidity [-]  (used by Ball-Berry)
    VPD_c  : canopy-air vapour pressure deficit [Pa]  (converted to
             kPa inside for Medlyn; unused by Ball-Berry)
    Ca     : ambient CO2 [μmol / mol]
    Tf     : leaf temperature [K]  (for mol → m/s unit conversion)
    Ps     : surface pressure [Pa]
    m, b0  : stomatal slope and intercept (see above)
    stomatal_model : ``"ball_berry"`` | ``"medlyn"`` — static Python string
                     captured in a ``functools.partial`` closure; never
                     traced by JAX.

    Returns
    -------
    (rs [s m-1], gs [m s-1], Ci [μmol mol-1])
    """
    # ``stomatal_model`` is a static Python string (see the jitting caller's
    # static_argnames), so this dispatch is resolved at trace time — a typo must
    # raise, not silently run the other model (matches solve_coupled_farquhar_ci).
    if stomatal_model == "medlyn":
        VPD_kPa = jnp.maximum(VPD_c, 50.0) / 1000.0  # coeff-ok: 50 Pa (0.05 kPa) VPD floor; Pa→kPa
        gs_mol = medlyn_gs(An, VPD_kPa, Ca, m, b0)
    elif stomatal_model == "ball_berry":
        gs_mol = ball_berry_gs(An, RH_c, Ca, m, b0)
    else:
        raise ValueError(
            f"unknown stomatal_model {stomatal_model!r}; the stomatal "
            "conductance scheme must be one of {'ball_berry', 'medlyn'}")

    # Minimum cuticular conductance: keep gs > 0 even at full water stress with
    # the legacy ``stress_b0=True`` (m = b0 = 0), otherwise the leaf gs/Ci/An
    # subsystem degenerates and the canopy Newton Jacobian goes singular → NaN.
    # With ``stress_b0=False`` b0 stays > 0 and this floor is inactive.  See
    # ``_GS_MIN_MOL``.
    gs_mol = jnp.maximum(gs_mol, _GS_MIN_MOL)

    Ci = Ca - DIFFUSIVITY_RATIO_H2O_CO2 * An / jnp.maximum(gs_mol, 1e-9)
    # Clip Ci to the physically reasonable C3 range; mixed-PFT C3/C4
    # is handled upstream in ``photosynthesis()`` via the continuous fC4
    # fraction, so the C4 bounds are not needed here.
    Ci = jnp.clip(Ci, 0.5 * Ca, 0.9 * Ca)  # coeff-ok: clamp Ci to physical C3 range [0.5, 0.9]·Ca

    # Unit conversion: mol m-2 s-1 → m s-1 at IUPAC STP reference
    # (T_std = T_freeze, P_std = _Ps0 = 101325 Pa, V_molar = 22.4 L/mol → 0.0224 m^3).
    # ``_CF_MOLAR_VOLUME`` encodes the reference molar volume (with the 1e-2 factor
    # below giving V_molar = 1e-2 / _CF_MOLAR_VOLUME = 0.0224 m^3/mol).
    cf = _CF_MOLAR_VOLUME * (constants.T_freeze / Tf) * (Ps / _Ps0)
    rs = 1.0 / (gs_mol / cf * 1e-2)   # coeff-ok: fixed molar-volume unit factor → [s m-1]
    gs = 1.0 / rs                      # [m s-1]
    return rs, gs, Ci


# ---------------------------------------------------------------------------
# Leaf energy balance — BT (Bulk Transfer)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("stomatal_model", "le_cap_mode"))
def leaf_energy_balance_bt(
    An: jax.Array,
    ASW: jax.Array,
    ALW: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    Ca: jax.Array,
    Tc: jax.Array,
    q_f: jax.Array,
    q_c: jax.Array,
    RH_c: jax.Array,
    VPD_c: jax.Array,
    lam: jax.Array,
    Cp: jax.Array,
    rhoa: jax.Array,
    Rb: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    fwet: jax.Array = 0.0,
    stomatal_model: str = "ball_berry",
    le_cap_mode: str = "soft",
) -> tuple[jax.Array, ...]:
    """Leaf energy balance via direct bulk transfer (BT).

    LE = λ ρ (q_f - q_c) * g_lh_eff
    H  = Rn - LE
    Tf_new = Tc + Rb / (ρ Cp) * H

    ``fwet`` (wetted leaf fraction, 0..~0.05) splits the leaf into a DRY part
    that transpires through the stomata (series conductance ``g_lh``) and a WET
    part that evaporates intercepted water at the boundary-layer-limited rate
    (``1/Rb``, no stomatal resistance).  The effective conductance is the
    area-weighted blend, so a wet leaf evaporates MORE than a dry one — the
    interception-loss increase in ET.  ``fwet = 0`` recovers the pure-stomatal
    form exactly.  The wet-part water comes from the canopy store, routed by the
    caller (``LE_wet`` recomputed in ``canopy_forward``).

    Parameters
    ----------
    An     : net photosynthesis [μmol m-2 s-1]
    ASW    : absorbed shortwave [W m-2]
    ALW    : net absorbed longwave [W m-2]
    Tf     : leaf temperature [K]
    Ps     : pressure [Pa]
    Ca     : ambient CO2 [μmol mol-1]
    Tc     : canopy air temperature [K]
    q_f    : leaf saturation specific humidity [kg kg-1]
    q_c    : canopy air specific humidity [kg kg-1]
    RH_c   : canopy relative humidity [-]
    VPD_c  : canopy-air vapour pressure deficit [Pa] (used by Medlyn)
    lam    : latent heat of vaporisation [J kg-1]
    Cp     : specific heat of air [J kg-1 K-1]
    rhoa   : air density [kg m-3]
    Rb     : boundary-layer resistance [s m-1]
    m, b0  : stomatal slope and intercept (Ball-Berry or Medlyn; see
             ``_compute_gs_and_ci``)
    stomatal_model : ``"ball_berry"`` | ``"medlyn"`` — static argument.

    Returns
    -------
    Rn, LE, H, Tf_new, gs, Ci
    """
    _rs, gs, Ci = _compute_gs_and_ci(
        An, RH_c, VPD_c, Ca, Tf, Ps, m, b0, stomatal_model)

    Rn = ASW + ALW
    # Series leaf latent-heat conductance written directly in gs (= 1/rs):
    #   g_lh = 1/(Rb + rs) = gs / (gs*Rb + 1)
    # This is algebraically identical to ``num / (Rb + 1/gs)`` but its
    # reverse/forward-mode AD uses the product rule, so the Jacobian stays
    # finite as the stomata close (gs -> 0, g_lh -> 0 with d g_lh/d gs -> 1)
    # instead of the quotient form's Inf/Inf blow-up.  The denominator
    # (gs*Rb + 1) >= 1, so no small-denominator guard is needed.  (DifferBESS
    # Apr-13 conductance refactor.)
    g_lh = gs / (gs * Rb + 1.0)
    # Dry part transpires through the stomata (g_lh); wet part evaporates at the
    # boundary-layer limit 1/Rb (no stomatal resistance).  Blend by wetted area.
    # Rb guarded (an optimised leaf-width / cv could drive Rb -> 0; the 1/Rb term
    # would then NaN even on the inactive fwet=0 branch of the blend).
    g_lh_wet = 1.0 / jnp.maximum(Rb, 1e-9)
    g_lh_eff = (1.0 - fwet) * g_lh + fwet * g_lh_wet
    LE = lam * rhoa * (q_f - q_c) * g_lh_eff

    # Bound LE to the available energy (default soft cap).  This is what keeps
    # ``Tf_new`` from running away: the smooth ``soft`` mode still admits
    # nocturnal dew (LE < 0) and modest daytime LE > Rn, so it preserves the
    # behaviour the old hard clamp ``LE ∈ [0, max(Rn,0)]`` was removed to keep,
    # while preventing the q_sat(Tf) overflow that diverged the Newton solve at
    # hot/dry sites.  See ``apply_le_cap``.
    LE = apply_le_cap(LE, Rn, le_cap_mode)

    H  = Rn - LE
    # ``Tf_new`` is the leaf temperature satisfying sensible-flux closure
    # ``H = ρ Cp (Tf − Tc)/Rb``.  Left unclipped (a hard ``clip(dT, ±30)`` is
    # non-differentiable and degrades the Newton Jacobian); the LE cap above
    # plus the solver's Newton step clamp keep ``Tf_new`` bounded.
    Tf_new = Tc + Rb * H / (rhoa * Cp)

    return Rn, LE, H, Tf_new, gs, Ci


# ---------------------------------------------------------------------------
# Leaf energy balance — PM (Penman-Monteith, second-order Paw & Gao 1988)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("stomatal_model", "le_cap_mode"))
def leaf_energy_balance_pm(
    An: jax.Array,
    ASW: jax.Array,
    ALW: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    Ca: jax.Array,
    Tc: jax.Array,
    VPD_c: jax.Array,
    RH_c: jax.Array,
    desTc: jax.Array,
    ddesTc: jax.Array,
    gamma_c: jax.Array,
    Cp: jax.Array,
    rhoa: jax.Array,
    Rb: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    fwet: jax.Array = 0.0,
    stomatal_model: str = "ball_berry",
    le_cap_mode: str = "soft",
) -> tuple[jax.Array, ...]:
    """Leaf energy balance via second-order Penman-Monteith (Paw & Gao 1988).

    ``stomatal_model`` selects Ball-Berry or Medlyn; see
    ``leaf_energy_balance_bt`` for details.  ``fwet`` (wetted leaf fraction)
    blends the DRY vapour conductance ``1/(Rb+rs)`` with the WET one ``1/Rb``
    (no stomatal resistance) and uses the equivalent canopy resistance
    ``rc_eff = 1/g_v - Rb`` in the quadratic, so a wet leaf evaporates more
    (interception loss).  ``fwet = 0`` gives ``rc_eff = rs`` exactly.

    Returns
    -------
    Rn, LE, H, Tf_new, gs, Ci
    """
    rs, gs, Ci = _compute_gs_and_ci(
        An, RH_c, VPD_c, Ca, Tf, Ps, m, b0, stomatal_model)

    Rn = ASW + ALW
    # Wet/dry vapour-conductance blend -> effective canopy resistance.  The
    # ``where(fwet>0)`` keeps ``rc == rs`` BIT-IDENTICAL at fwet=0 (the algebra
    # 1/(1/(Rb+rs))-Rb only recovers rs to rounding).  Rb guarded.
    _Rb_s = jnp.maximum(Rb, 1e-9)
    g_v = (1.0 - fwet) / (_Rb_s + rs) + fwet / _Rb_s
    rc = jnp.where(fwet > 0.0, 1.0 / g_v - _Rb_s, rs)

    ddesTc_Rb2          = ddesTc * Rb**2
    gamma_Rb_rc          = gamma_c * (Rb + rc)
    rhoa_Cp_gamma_Rb_rc  = rhoa * Cp * gamma_Rb_rc

    a = 0.5 * ddesTc_Rb2 / rhoa_Cp_gamma_Rb_rc
    b = (-1.0
         - Rb * desTc / gamma_Rb_rc
         - ddesTc_Rb2 * Rn / rhoa_Cp_gamma_Rb_rc)
    c = (rhoa * Cp / gamma_Rb_rc * VPD_c
         + desTc * Rb / gamma_Rb_rc * Rn
         + 0.5 * ddesTc_Rb2 / rhoa_Cp_gamma_Rb_rc * Rn**2)

    disc = jnp.maximum(b**2 - 4.0 * a * c, 0.0)
    LE   = (-b + jnp.sign(b) * jnp.sqrt(disc)) / (2.0 * a)

    # Bound LE to the available energy — see leaf_energy_balance_bt / apply_le_cap.
    LE = apply_le_cap(LE, Rn, le_cap_mode)

    H  = Rn - LE
    # Unclipped Tf update; the LE cap + solver Newton clamp bound it.
    Tf_new = Tc + Rb * H / (rhoa * Cp)

    return Rn, LE, H, Tf_new, gs, Ci


# ---------------------------------------------------------------------------
# Below-canopy soil-surface evaporation resistance
# ---------------------------------------------------------------------------
# Sellers et al. (1992, J. Climate 5:1531) top-layer soil-surface resistance to
# bare-soil evaporation, ``r_ss = exp(a - b * W_1)`` [s/m], with ``W_1 =
# theta_1/theta_sat`` the top-layer RELATIVE saturation.  This is the RESISTANCE-
# method counterpart of the ``S_top**exp`` beta EFFICIENCY (same paper, its
# eq. for the surface resistance) — the two are alternative parameterisations of
# the SAME soil-moisture control on evaporation, so they are used mutually
# exclusively (never both, or the moisture limitation double-counts).
#
# Why a resistance (not the beta) below a canopy: the canopy soil energy balance
# limits soil evaporation with ONLY the below-canopy aerodynamic resistance
# (``raw_soil = 1/(Cs*ustar)``, ~50-100 s/m under a closed canopy).  A wet forest
# floor (W_1~0.8) then evaporates at near-potential rate — LE_soil measured at
# ~185 W/m2 midday at US-MMS (LAI~6 DBF), i.e. ~57% of total LE where <15% is
# physical.  The beta efficiency multiplies the (already too-small) conductance
# and barely helps because the surface frequently rewets to W_1~1.  Adding r_ss in
# SERIES with raw_soil supplies the soil-side vapour-diffusion resistance the
# aerodynamic-only path omits, and — unlike the beta — throttles a WET surface.
_SELLERS_RSS_A   = 8.206    # [-]  Sellers et al. (1992) intercept
_SELLERS_RSS_B   = 4.255    # [-]  Sellers et al. (1992) wetness slope
_SELLERS_RSS_MAX = 5.0e3    # [s/m] cap on r_ss (W_1->0 gives exp(8.206)~3.7e3);
                            #       bounds the AD Jacobian at the residual boundary.


@jax.jit
def soil_surface_evap_resistance(
    theta_rel: jax.Array,
    LAI: jax.Array,
    litter_resistance_s_m: jax.Array,
    litter_LAI: jax.Array | None = None,
) -> jax.Array:
    """Below-canopy soil-surface resistance to evaporation [s/m], added in
    SERIES with the below-canopy aerodynamic resistance in the soil energy
    balance.

    Two additive, physically-distinct mechanisms:

    * ``r_ss`` — Sellers et al. (1992) dry-surface-layer resistance
      ``exp(a - b * W_1)`` from the top-layer relative saturation
      ``W_1 = theta_1/theta_sat``; the vapour-diffusion resistance of the
      drying skin.  Rises as the surface dries (W_1 -> 0) and is a finite
      ~52 s/m even when saturated (W_1 = 1).

    * ``r_litter`` — Sakaguchi & Zeng (2009, JGR 114:D01107) forest-floor
      litter resistance, ``litter_resistance_s_m * (1 - exp(-0.5 LAI))``: the
      litter fractional cover ``1 - exp(-0.5 LAI)`` grows from ~0 (bare soil)
      to ~1 (closed canopy), so the resistance self-scales with canopy
      density across biomes (grassland litter << forest litter) from a single
      reference value — not a per-site tuning.

    Parameters
    ----------
    theta_rel : top-layer relative saturation ``theta_1/theta_sat`` [-], (ncol,)
    LAI       : leaf area index [m2 m-2], (ncol,)
    litter_resistance_s_m : reference forest-floor litter resistance [s/m]
    litter_LAI : STRUCTURAL leaf area index driving the litter cover [m2 m-2],
                 (ncol,).  Defaults to ``LAI`` (live).  A deciduous forest floor
                 keeps its leaf litter through the leaf-off season, so the litter
                 cover must be driven by a slowly-varying / seasonal-maximum LAI
                 rather than the instantaneous live LAI (which collapses to ~0 in
                 winter and lets the bare soil evaporate spuriously).  When the
                 caller supplies a persistent LAI here the litter cover no longer
                 vanishes in winter; when it does not, behaviour is unchanged.

    Returns
    -------
    r_soil_surface : (ncol,) soil-surface resistance [s/m]
    """
    w1 = jnp.clip(theta_rel, 1e-6, 1.0)
    r_ss = jnp.minimum(
        jnp.exp(_SELLERS_RSS_A - _SELLERS_RSS_B * w1), _SELLERS_RSS_MAX)
    # Litter fractional cover (0.5 is the standard LAI extinction coefficient,
    # matching the below-canopy Cs clumping weight exp(-0.5*CI*LAI)); driven by the
    # STRUCTURAL LAI (persistent forest-floor litter), falling back to live LAI.
    _lai_cover = LAI if litter_LAI is None else litter_LAI
    litter_frac = 1.0 - jnp.exp(-0.5 * jnp.maximum(_lai_cover, 0.0))
    r_litter = litter_resistance_s_m * litter_frac
    return r_ss + r_litter


# ---------------------------------------------------------------------------
# Soil energy balance — BT
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("le_cap_mode",))
def soil_energy_balance_bt(
    Ts: jax.Array,
    Tc: jax.Array,
    q_s: jax.Array,
    q_c: jax.Array,
    lam: jax.Array,
    rhoa: jax.Array,
    Cp: jax.Array,
    rah_soil: jax.Array,
    raw_soil: jax.Array,
    fStress: jax.Array,
    ASW_soil: jax.Array,
    ALW_soil: jax.Array,
    le_cap_mode: str = "soft",
) -> tuple[jax.Array, ...]:
    """Soil energy balance with prescribed skin temperature (BT).

    The soil LE is also energy-capped (default soft).  With ``Ts`` PRESCRIBED the
    humidity-gradient soil evaporation is not energy-constrained — a hot prescribed
    skin gives a large ``q_sat(Ts)`` and LE can vastly exceed the available soil
    net radiation (``G`` then absorbs the imbalance), which is unphysical and badly
    biases offline LE.  The cap keeps ``LE_soil`` near ``[-SOFT_LOW, Rn_soil+slack]``.
    Pass ``le_cap_mode="off"`` to recover the raw conductance form.

    The soil skin ``Ts`` is supplied by the caller from the top layer of the
    multilayer soil thermal state (``T_soil[:, 0]``).  Turbulent fluxes are
    computed from explicit bulk-transfer formulas and ``G`` closes the
    surface energy budget as a residual — it is *not* parameterised as
    ``G_alpha · Rn``.  The resulting ``G`` is then fed as the upper
    boundary condition to ``solve_soil_thermal`` in the caller (the same
    pattern as ``multilayer_land.py``).

    ``fStress`` (soil evaporation efficiency in [0, 1]) scales the
    below-canopy aerodynamic conductance to give the soil evaporation
    conductance ``fStress / raw_soil`` — equivalent to a dryness resistance
    ``raw_soil · (1/fStress - 1)`` in series with ``raw_soil``.

    Returns
    -------
    Rn_soil, LE_soil, H_soil, G
    """
    Rn = ASW_soil + ALW_soil
    # Direct bulk-transfer turbulent fluxes — Ts is prescribed so no
    # root-finding for soil T is needed.
    # Soil evaporation — BETA form: ``fStress`` (soil-evaporation efficiency)
    # times the below-canopy aerodynamic conductance, g_soil = fStress/raw_soil.
    # Expressing fStress as a multiplier (not a 1/fStress dryness resistance)
    # keeps the AD Jacobian finite as the soil dries (fStress -> 0: LE -> 0).
    # NOTE on the stress variable: the caller supplies ``fStress`` = the soil
    # pore RELATIVE HUMIDITY h_r = exp(psi_top g / (R_v T)) (Kelvin eq.) from the
    # PROGNOSTIC top-layer matric potential — so soil evaporation is governed by
    # the fast-drying SURFACE, not the root zone.  The beta form (vs the alpha
    # sub-saturated-surface q_surf=h_r*q_s) is used deliberately: with legoESM's
    # PROGNOSTIC skin T the alpha form drives excessive condensation (LE<0) onto a
    # dry surface and destabilises the surface energy balance; beta bounds LE->0
    # as h_r->0 (DifferBESS can use alpha because it PRESCRIBES Ts).
    g_soil = fStress / jnp.maximum(raw_soil, 1e-9)
    LE = lam * rhoa * (q_s - q_c) * g_soil
    LE = apply_le_cap(LE, Rn, le_cap_mode)
    H  = rhoa * Cp * (Ts - Tc) / jnp.maximum(rah_soil, 1e-6)
    # G closes the surface energy budget as a residual — positive into soil.
    G  = Rn - LE - H
    return Rn, LE, H, G


# ---------------------------------------------------------------------------
# Soil energy balance — PM
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("le_cap_mode",))
def soil_energy_balance_pm(
    Ts: jax.Array,
    Tc: jax.Array,
    q_s: jax.Array,
    q_c: jax.Array,
    lam: jax.Array,
    rhoa: jax.Array,
    Cp: jax.Array,
    rah_soil: jax.Array,
    raw_soil: jax.Array,
    fStress: jax.Array,
    ASW_soil: jax.Array,
    ALW_soil: jax.Array,
    le_cap_mode: str = "soft",
) -> tuple[jax.Array, ...]:
    """Soil energy balance with prescribed skin temperature (PM).

    With ``Ts`` prescribed by the caller, the second-order Penman-Monteith
    quadratic that originally solved for ``Ts`` is no longer needed — LE
    and H follow from explicit bulk-transfer formulas.  The PM variant is
    therefore numerically identical to the BT variant at the soil level;
    the dispatch is kept for API symmetry with the leaf pathway where PM
    and BT still differ.

    Returns
    -------
    Rn_soil, LE_soil, H_soil, G
    """
    Rn = ASW_soil + ALW_soil
    # Soil evaporation — BETA form: ``fStress`` (soil-evaporation efficiency)
    # times the below-canopy aerodynamic conductance, g_soil = fStress/raw_soil.
    # Expressing fStress as a multiplier (not a 1/fStress dryness resistance)
    # keeps the AD Jacobian finite as the soil dries (fStress -> 0: LE -> 0).
    # NOTE on the stress variable: the caller supplies ``fStress`` = the soil
    # pore RELATIVE HUMIDITY h_r = exp(psi_top g / (R_v T)) (Kelvin eq.) from the
    # PROGNOSTIC top-layer matric potential — so soil evaporation is governed by
    # the fast-drying SURFACE, not the root zone.  The beta form (vs the alpha
    # sub-saturated-surface q_surf=h_r*q_s) is used deliberately: with legoESM's
    # PROGNOSTIC skin T the alpha form drives excessive condensation (LE<0) onto a
    # dry surface and destabilises the surface energy balance; beta bounds LE->0
    # as h_r->0 (DifferBESS can use alpha because it PRESCRIBES Ts).
    g_soil = fStress / jnp.maximum(raw_soil, 1e-9)
    LE = lam * rhoa * (q_s - q_c) * g_soil
    LE = apply_le_cap(LE, Rn, le_cap_mode)
    H  = rhoa * Cp * (Ts - Tc) / jnp.maximum(rah_soil, 1e-6)
    G  = Rn - LE - H
    return Rn, LE, H, G


# ---------------------------------------------------------------------------
# Canopy air temperature and humidity update
# ---------------------------------------------------------------------------

@jax.jit
def canopy_air_update(
    Ta: jax.Array,
    q_atm: jax.Array,
    Tf_Sun: jax.Array,
    Tf_Sh: jax.Array,
    Ts: jax.Array,
    gs_Sun: jax.Array,
    gs_Sh: jax.Array,
    Rb_Sun: jax.Array,
    Rb_Sh: jax.Array,
    rah_above: jax.Array,
    raw_above: jax.Array,
    rah_below: jax.Array,
    raw_below: jax.Array,
    fStress: jax.Array,
    Ps: jax.Array,
    fwet: jax.Array = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Update canopy air temperature Tc and specific humidity q_c.

    ``fwet`` (wetted leaf fraction) blends the leaf WATER conductance with the
    wet-surface value ``1/Rb`` so the humidity balance transports the same
    wet-leaf vapour flux the leaf energy balance produces — otherwise ``q_c``
    is biased dry and the wet LE boost is amplified (must match
    ``leaf_energy_balance_bt`` ``g_lh_eff``).  ``fwet = 0`` is unchanged.

    Conductance-weighted mixing of above-canopy air, sunlit and shaded
    leaves, and soil — DifferBESS FULLY_COUPLED formulation.  Leaves and
    soil both communicate with the canopy air space (Tc, q_c).

    Returns
    -------
    Tc_new, q_c_new
    """
    ch_a   = 1.0 / jnp.maximum(rah_above, 1e-9)
    ch_sun = 1.0 / jnp.maximum(Rb_Sun,    1e-9)
    ch_sh  = 1.0 / jnp.maximum(Rb_Sh,     1e-9)
    ch_g   = 1.0 / jnp.maximum(rah_below, 1e-9)

    cw_a   = 1.0 / jnp.maximum(raw_above, 1e-9)
    # Leaf water conductances in gs-form gs/(gs*Rb + 1) (= 1/(Rb + 1/gs)) so
    # the AD Jacobian stays finite as the stomata close (gs -> 0) — no 1/gs
    # intermediate.  Soil conductance = fStress / raw_below (the soil
    # evaporation efficiency times the below-canopy aerodynamic conductance),
    # finite as the soil dries.  See the leaf/soil energy-balance notes.
    # Wet/dry blend of the leaf water conductance — matches the leaf energy
    # balance g_lh_eff so the vapour transported into the canopy air equals the
    # vapour the leaf loses (Rb guarded like the ch terms above).
    _Rb_Sun_s = jnp.maximum(Rb_Sun, 1e-9)
    _Rb_Sh_s  = jnp.maximum(Rb_Sh,  1e-9)
    cw_sun = ((1.0 - fwet) * gs_Sun / (gs_Sun * _Rb_Sun_s + 1.0)
              + fwet / _Rb_Sun_s)
    cw_sh  = ((1.0 - fwet) * gs_Sh  / (gs_Sh  * _Rb_Sh_s  + 1.0)
              + fwet / _Rb_Sh_s)
    # Soil water conductance = fStress / raw_below (beta form; fStress = soil pore
    # RH h_r).  Beta (not alpha sub-saturation) for consistency with the soil
    # energy balance under a PROGNOSTIC skin T — see soil_energy_balance_bt.
    cw_g   = fStress / jnp.maximum(raw_below, 1e-9)

    q_f_Sun = saturation_specific_humidity(Tf_Sun, Ps)
    q_f_Sh  = saturation_specific_humidity(Tf_Sh,  Ps)
    q_s     = saturation_specific_humidity(Ts,     Ps)

    Tc_new = (ch_a * Ta + ch_sun * Tf_Sun + ch_sh * Tf_Sh + ch_g * Ts) / (
        ch_a + ch_sun + ch_sh + ch_g)
    q_c_new = (cw_a * q_atm + cw_sun * q_f_Sun + cw_sh * q_f_Sh + cw_g * q_s) / (
        cw_a + cw_sun + cw_sh + cw_g)

    return Tc_new, q_c_new
