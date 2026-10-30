"""Shared warm-rain microphysics helpers.

Functions here are used by multiple microphysics backends (Seifert-Beheng,
Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.output import sedimentation_tendency
from legoesm.core.smooth import smooth_clamp, smooth_max, smooth_min, smooth_relu, smoothstep_quintic
from legoesm.thermo import (
    mixing_ratio_to_specific_humidity,
    saturation_mixing_ratio,
    saturation_mixing_ratio_ice,
    saturation_specific_humidity,
    saturation_specific_humidity_goff_gratch,
    saturation_specific_humidity_ice,
    saturation_specific_humidity_ice_goff_gratch,
)

# Default condensation/autoconversion sigmoid sharpness (used in signatures).
_DEFAULT_SAT_SHARPNESS = 50.0

# --- Hard (iterated) saturation-adjustment guard (opt-in) -------------------
# A bracketed-bisection saturation adjustment that lands q_v ON the liquid
# saturation curve wherever q_v > threshold * q_sat, conserving the moist
# enthalpy c_pd*T + L_v*q_v EXACTLY (the caller maps the returned condensation
# rate to dT_dt with +L_v/c_pd and to dq_v_dt with -1, so the enthalpy invariant
# holds by construction for any rate).  It drains local super-saturation pools
# the smooth sigmoid branch cannot: the sign of q_v - q_sat flips once q_sat
# rises with T and then smooth-caps at 1 kg/kg, so the sigmoid under-condenses
# moderate super-saturation and cannot fire on the hot-phase pools at all.
# Opt-in (``hard_adjust=True``); the ``hard_adjust=False`` default path is
# byte-identical to the historical smooth-only behaviour.
_DEFAULT_HARD_SAT_THRESHOLD = 1.1     # RH trigger: fire where q_v > thr*q_sat [-]
# Per-step latent-heating cap [K] on the hard adjustment.  Draining a large
# super-saturation pool ONTO the curve in a single step dumps its full latent
# heat at once (a 67 g/kg pool ~ 165 K) -- itself a detonation (observed in the
# field with the naive post-step cap; codex F2/F3).  Bounding the per-step
# heating to this value (equivalently the per-step condensed Delta q_v =
# cap*c_pd/L_v ~ 2 g/kg at 5 K) makes big pools drain over MANY steps,
# monotonically, without overshoot.  Mild super-saturation (whose full drain is
# below the cap) still lands ON the curve in one step (the cap is inactive).
_DEFAULT_HARD_SAT_MAX_HEATING_K = 5.0
# Fixed BISECTION step count (a loop-iteration count, never tunable/config).
# The equilibrium root is BRACKETED in [q_sat(T), q_v] and f is strictly
# monotone, so bracketed bisection converges for ANY (T, p, q_v) -- including
# extreme super-saturation (q_v up to 1 kg/kg) at low pressure and through the
# q_sat cap edge, the regimes where a Clausius-Clapeyron Newton step overshoots
# into a SUB-saturated (over-condensed) state (codex rounds 1-2: n=8 Newton at
# 200 K/100 Pa landed RH ~ 0.35; n=16 still off-curve at q_v=0.3 kg/kg).  30
# bisection steps drive |RH-1| to machine precision across p >= 100 Pa,
# T in [150,400] K, q_v in [0.066, 1.0] kg/kg (verified 8.9e-14).  The
# bracketing ``where`` reductions make the raw autodiff gradient finite but
# inaccurate, so the exact gradient is re-attached by implicit differentiation
# (see :func:`_hard_saturation_condensation`).
_HARD_SAT_BISECT_ITERS = 30
# Smooth activation ramp steepness [1/RH] for the trigger sigmoid, centred on
# the threshold: a finite ramp (not a hard step) keeps the blend continuous and
# differentiable (avoiding the discontinuity of a naive post-step cap).  At
# steepness 30 a cell reaches ~full hard adjustment (lands on the curve) within
# ~0.15 RH above the threshold, and is ~fully smooth ~0.15 RH below it; cells in
# the narrow transition band are a smooth handoff (not exactly on the curve, by
# construction -- an exact landing there would require a discontinuity).
_HARD_SAT_MASK_SHARPNESS = 30.0
# Floor on q_sat in the RH trigger q_v/q_sat, so the ratio stays finite where
# q_sat underflows (very cold cells); the bisection solve itself uses the true
# (unfloored) q_sat.  A safety floor (<= 1e-6), not a physical scale.
_HARD_SAT_QSAT_FLOOR = 1.0e-12


def safe_pow(x, p):
    """Differentiable ``x ** p`` with grad=0 wherever ``x <= 0``.

    Microphysics has many Marshall-Palmer-style fractional powers of
    hydrometeor mixing ratios (``q_r``, ``q_i``, ``q_s``, ``q_g``,
    ``N_i``, …) with exponents in (0, 1) — typically 0.5 for fall
    speeds, 1/3 for diameters, 0.525/0.875 for ventilation/accretion.
    Their analytic derivative ``p * x**(p-1)`` is unbounded at ``x=0``
    and complex for ``x<0``.  ``jnp.clip(x, 0.0) ** p`` therefore
    returns ``inf`` (at zero) or ``nan`` (at negatives) under
    ``jax.grad``, breaking AD on cold-start (no-precip) initial
    conditions.

    The double-where pattern below routes the AD graph through a
    placeholder of 1.0 in the inactive branch so the gradient never
    sees ``0**(p-1)``.

    Parameters
    ----------
    x : array
        Argument of the power.  May be zero or negative.
    p : float or array
        Exponent.  Intended for ``0 < p < 1`` where the bug applies;
        also safe for ``p >= 1``.

    Returns
    -------
    array
        ``x ** p`` for ``x > 0``, else 0.  The gradient is finite at every
        point but UNBOUNDED as ``x -> 0+`` for ``p < 1``: callers must keep
        tiny-but-positive inputs out (gate the source below QSMALL).
    """
    positive = x > 0.0
    safe_x = jnp.where(positive, x, 1.0)
    return jnp.where(positive, safe_x ** p, 0.0)


def donor_clamp_scale(q_avail, sink_total, dt, divisor_floor=1.0e-15):
    """AD-safe donor clamp scale ``min(1, q / (sink·dt))``.

    Returns the multiplicative scale that should be applied to all
    sinks of a single hydrometeor species so the per-step removal does
    not exceed the locally-available mass.  The naive
    ``q / max(sink·dt, 1e-30)`` form has a VJP of ``-q / (sink·dt)²``
    that overflows fp32 (max ≈ 3.4e38) whenever ``sink·dt`` is
    smaller than ~1e-20.  The ``divisor_floor`` (default 1e-15)
    bounds the divisor from below so the worst-case VJP magnitude is
    ``q / 1e-30 ≈ 1e30`` — comfortably within fp32.  At the floor
    ``jnp.maximum`` has zero subgradient, which is physically correct
    (no scaling, no sensitivity to the tiny sink).

    Parameters
    ----------
    q_avail : array
        Available mass per unit air (positive part of the hydrometeor
        mixing ratio).
    sink_total : array
        Combined sink rate [kg/kg/s] for the species in this step.
    dt : float or array
        Physics step [s].
    divisor_floor : float, default 1e-15
        Floor on ``sink_total · dt`` for AD safety.

    Returns
    -------
    array
        Multiplicative scale in ``[0, 1]``.  Identity (1.0) wherever
        the per-step sink is below the floor (no clamp needed).
    """
    sink_dt_safe = jnp.maximum(sink_total * jnp.maximum(dt, 1.0e-10),
                               divisor_floor)
    return jnp.minimum(1.0, q_avail / sink_dt_safe)


def kelvin_coefficient(T):
    """Köhler curvature (Kelvin) term ``A = 2 σ_w / (ρ_w R_v T)`` [m].

    SINGLE canonical definition — shared by the fast-SBM nucleation
    (oracle ``AKOE``) and the ARG (Abdul-Razzak & Ghan 2000) activation
    scheme, so the two cannot drift (no-duplicate-numerics rule).
    """
    return 2.0 * constants.sigma_water / (
        constants.rho_water * constants.R_v * T)


# --- air transport properties (SAM M2005 module_mp_graupel.f90; fixed) -------
# Hall & Pruppacher (1976) water-vapour diffusivity in air, Sutherland dynamic
# viscosity, and the derived Schmidt number / air thermal conductivity — shared
# by every ventilation-limited ice process (snow + graupel deposition / melting
# and rain evaporation). Published transport-property constants, not tunable.
_DV_PREFACTOR = 8.794e-5          # DV = _DV_PREFACTOR * T**_DV_T_EXPONENT / p
_DV_T_EXPONENT = 1.81
_MU_PREFACTOR = 1.496e-6          # mu = _MU_PREFACTOR * T**_MU_T_EXPONENT / (T + _MU_SUTHERLAND_T)
_MU_T_EXPONENT = 1.5
_MU_SUTHERLAND_T = 120.0          # Sutherland temperature offset [K]
_AIR_CONDUCTIVITY_MU_FACTOR = 1.414e3   # air thermal conductivity KAP = factor * mu [-]


def _air_transport_props(T, p, rho):
    """Shared ventilation transport coefficients ``(DV, mu, SC)``.

    DV = vapour diffusivity [m^2/s], mu = Sutherland dynamic viscosity [kg/m/s],
    SC = mu/(rho*DV) the Schmidt number. Factored out of the five SAM ice
    processes that recompute the identical block (snow/graupel deposition +
    melting, rain evaporation); callers needing the air thermal conductivity use
    ``_AIR_CONDUCTIVITY_MU_FACTOR * mu``."""
    dv = _DV_PREFACTOR * safe_pow(T, _DV_T_EXPONENT) / jnp.clip(p, 1.0)
    mu = _MU_PREFACTOR * safe_pow(T, _MU_T_EXPONENT) / (T + _MU_SUTHERLAND_T)
    sc = mu / (rho * dv)
    return dv, mu, sc


def mixed_phase_liquid_fraction(T):
    """Liquid fraction w(T) of the standard mixed-phase partition ramp.

    Linear in T over ``[T_hom_freeze, T_freeze]`` (all-ice below -40 C,
    all-liquid above 0 C) — the Morrison/IFS-style weight used to blend
    liquid/ice saturation and latent heat consistently.
    """
    return jnp.clip(
        (T - constants.T_hom_freeze)
        / (constants.T_freeze - constants.T_hom_freeze), 0.0, 1.0)


def mixed_phase_saturation_specific_humidity(T, p_full):
    """w(T)-blended liquid/ice saturation SPECIFIC humidity [kg/kg].

    Below the mixed-phase ramp this is the ICE curve — at TTL temperatures
    (~195 K) the liquid curve sits ~60% above it, so gating/draining on the
    liquid curve leaves permanent ice-supersaturation (the 20x-ERA5 TTL
    vapour bias of the first ClimateEval scorecard).
    """
    w = mixed_phase_liquid_fraction(T)
    r_blend = (w * saturation_mixing_ratio(T, p_full)
               + (1.0 - w) * saturation_mixing_ratio_ice(T, p_full))
    return mixing_ratio_to_specific_humidity(r_blend)


def mixed_phase_l_over_cp(T):
    """Blended latent heat / c_pd [K per kg/kg] at the mixed-phase weight.

    ``L_eff = w(T)*L_v + (1-w(T))*L_s`` — deposition releases the
    sublimation latent heat where the target curve is ice.  Evaluated at
    the PRE-adjustment temperature and held constant through the on-curve
    solve (per-cell frozen linearization: the solve then conserves
    ``c_pd*T + L_eff(T0)*q_v`` exactly, mirroring the scalar-L_v contract
    of the liquid-only path).
    """
    w = mixed_phase_liquid_fraction(T)
    return (w * constants.L_v + (1.0 - w) * constants.L_s) / constants.c_pd


def _hard_saturation_condensation(T, q_v, p_full, dt, q_sat,
                                  sat_fn=saturation_specific_humidity,
                                  l_over_cp=None):
    """Hard-saturation-adjustment condensation rate [kg/kg/s].

    Solves for the equilibrium vapour ``q_eq`` on the LIQUID saturation curve
    that an enthalpy-conserving moist adjustment would reach::

        q_eq = q_sat_liq(T + (L_v/c_pd)*(q_v - q_eq), p)

    The residual ``f(q_eq) = q_eq - q_sat_liq(T + (L_v/c_pd)*(q_v - q_eq), p)``
    is STRICTLY INCREASING in ``q_eq`` (q_sat rises with T, and T falls as
    q_eq rises) with ``f(q_sat(T)) <= 0 <= f(q_v)``, so the root is BRACKETED in
    ``[q_sat(T), q_v]``.  A fixed ``_HARD_SAT_BISECT_ITERS`` bracketed-bisection
    sweep is therefore unconditionally convergent for ANY (T, p, q_v):
    it is robust to the steep low-pressure curve AND to the q_sat cap edge,
    where a Clausius-Clapeyron Newton step overshoots into a sub-saturated
    (over-condensed) state.  Because ``q_eq <= q_v`` the returned rate is a pure
    DRAIN (>= 0); a sub-saturated column has ``q_sat(T) >= q_v`` so the bracket
    collapses to ``q_v`` and the rate is 0 (evaporation is left to the smooth
    branch).

    Differentiability: the bracketing ``where`` reductions give a finite but
    INACCURATE autodiff gradient, so the bisection root is detached
    (``stop_gradient``) and the EXACT gradient re-attached by implicit
    differentiation -- one Newton correction at the converged root, whose
    residual is ~0 (VALUE unchanged) but whose derivative is the
    implicit-function gradient ``dq_eq/dtheta = (dq_sat/dtheta)/psychrometric``,
    with the psychrometric factor the exact ``1 + (L_v/c_pd)*dq_sat/dT`` (a jvp
    of the capped curve) held constant.  Verified against centred finite
    differences to rel <= 5e-7 in d/dq_v, d/dT, d/dp.

    Parameters
    ----------
    T, q_v, p_full : array
        Temperature [K], vapour mixing ratio [kg/kg], pressure [Pa] (the TRUE
        model pressure the scheme receives; never rebuilt from p_s*sigma).
    dt : float
        Physics step [s] (the same dt the smooth branch uses).
    q_sat : array
        The (capped) liquid saturation mixing ratio already computed by the
        caller; the lower bracket ``q_sat(T)``.

    Returns
    -------
    array
        Condensation rate ``(q_v - q_eq)/dt`` [kg/kg/s], >= 0.
    """
    # Scalar L_v/c_pd by default (byte-identical liquid path); the mixed-phase
    # caller injects a per-cell frozen L_eff(T0)/c_pd array (broadcasts through
    # the same expressions; monotonicity of the residual needs only L > 0).
    if l_over_cp is None:
        l_over_cp = constants.L_v / constants.c_pd
    q_v_pos = jnp.maximum(q_v, 0.0)
    lo = jnp.clip(q_sat, 0.0, q_v_pos)     # f(lo) <= 0 (bracket lower bound)
    hi = q_v_pos                           # f(hi) >= 0 (bracket upper bound)
    for _ in range(_HARD_SAT_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        t_new = T + l_over_cp * (q_v_pos - mid)
        f_mid = mid - sat_fn(t_new, p_full)
        hi = jnp.where(f_mid > 0.0, mid, hi)
        lo = jnp.where(f_mid > 0.0, lo, mid)
    q_eq = jax.lax.stop_gradient(0.5 * (lo + hi))
    # Implicit-function gradient via one exact-Jacobian Newton correction at the
    # (detached) converged root: value ~ unchanged (residual ~ 0), gradient =
    # (dq_sat/dtheta)/psychrometric.  ``dqsat_dt`` is the exact derivative of the
    # (capped) curve from a jvp; psychrometric is held constant (stop_gradient)
    # so the correction contributes only the implicit gradient, not a spurious
    # second-order term.
    t_new = T + l_over_cp * (q_v_pos - q_eq)
    q_sat_new, dqsat_dt = jax.jvp(
        lambda t: sat_fn(t, p_full),
        (t_new,), (jnp.ones_like(t_new),))
    psychrometric = jax.lax.stop_gradient(1.0 + dqsat_dt * l_over_cp)
    q_eq = q_eq - (q_eq - q_sat_new) / psychrometric
    # Keep q_eq in the physical bracket so the rate is a pure DRAIN (>= 0): the
    # single correction can nudge q_eq just past q_v for a (near-)sub-saturated
    # column (there f(q_v) < 0), which the activation mask suppresses anyway, but
    # this makes the >= 0 contract exact.  Inactive (gradient-transparent) in the
    # super-saturated region where q_sat(T) < q_eq < q_v strictly.
    q_eq = jnp.clip(q_eq, 0.0, q_v_pos)
    return (q_v_pos - q_eq) / dt


def _hard_saturation_rate_limit(condensation, q_v, dt, hard_max_heating_K,
                                l_over_cp=None):
    """Bound the per-step POSITIVE condensation (shared by both entry points).

    Caps the per-step condensed ``Delta q_v = cond*dt`` so the per-step latent
    heating ``l_over_cp * Delta q_v <= hard_max_heating_K`` (a large pool drains
    over MANY steps not one) AND ``Delta q_v <= q_v`` (mass-positivity: never
    condense more than the column holds; also guards the tiny-NEGATIVE q_sat the
    smooth cap returns near 150 K).  The evaporation branch (rate < 0) is
    untouched.  ``minimum`` has a dead gradient only where a cap binds, where the
    per-step condensation is constant (correctly zero sensitivity).

    ``l_over_cp`` is the latent-heat / c_pd the CALLER heats with, so the K cap
    bounds the ACTUAL heating.  Default ``None`` -> the scalar liquid
    ``L_v / c_pd`` (the exact prior expression, byte-identical liquid path); the
    mixed-phase ice-curve drain injects its per-cell frozen ``L_eff(T0) / c_pd``
    array so the cap is exact on the ICE branch too (deposition releases L_s >
    L_v, which would otherwise heat ~13% past the stated cap for a given dq).
    """
    if l_over_cp is None:
        # Scalar liquid path -- keep the exact prior expression (bit-identical).
        dqv_cap = hard_max_heating_K * constants.c_pd / constants.L_v
    else:
        # Per-cell L_eff(T0)/c_pd the ice-curve caller deposits with: the
        # heating l_over_cp*dq is then bounded by hard_max_heating_K exactly.
        dqv_cap = hard_max_heating_K / l_over_cp
    cond_cap = jnp.minimum(dqv_cap, jnp.maximum(q_v, 0.0)) \
        / jnp.maximum(dt, 1.0e-10)
    return jnp.where(
        condensation > 0.0, jnp.minimum(condensation, cond_cap), condensation)


def hard_saturation_blend(condensation, T, q_v, p_full, dt, q_sat,
                          hard_threshold, hard_max_heating_K,
                          sat_fn=saturation_specific_humidity):
    """Blend the on-curve hard adjustment into ``condensation`` + rate-limit.

    The ONE place the reviewed hard-adjustment logic lives, shared by
    :func:`saturation_adjustment` (in-scheme: ``condensation`` is the smooth
    condensation rate) and :func:`hard_saturation_drain` (driver post-step:
    ``condensation = 0`` -> a pure activation-gated drain).

    PUBLIC entry point for a scheme that computes its OWN smooth condensation
    rate and therefore cannot route through :func:`saturation_adjustment` --
    e.g. Sundqvist diagnostic large-scale condensation, whose published
    RH_crit-gated closure is a DIFFERENT smooth form.  Such a scheme passes its
    own rate in and gets EXACTLY the guard the five bulk warm-rain schemes get
    (same on-curve solve, same activation ramp, same no-overshoot cap, same
    per-step heating + mass rate limit) -- there is no second implementation.

    Steps:

    1. Solve the bracketed-bisection on-curve root ``cond_hard`` (fp64 when
       available; extreme pools drive large trial-T excursions).
    2. Smooth activation ramp CENTRED on ``hard_threshold`` (RH trigger, using
       the capped ``q_sat`` so vapour above the capped saturation still drains);
       0.5 at the threshold, -> 1 (asymptotically) above, -> 0 below.
    3. Blend smooth -> hard, then CAP at ``cond_hard`` so the finite smooth term
       cannot push a cell PAST saturation into a re-evaporating overshoot
       (``cond_hard`` never overshoots).  The under-condensing smooth branch
       (smooth < cond_hard: below-threshold / mild) is untouched; evaporation
       (< 0 <= cond_hard) passes.
    4. Rate-limit the POSITIVE rate so the per-step condensed
       ``Delta q_v = cond*dt <= hard_max_heating_K * c_pd / L_v`` (per-step
       heating <= ``hard_max_heating_K``) AND ``<= q_v`` (mass-positivity: never
       condense more than the column holds; also guards the tiny-NEGATIVE q_sat
       the smooth cap returns near 150 K).  ``minimum`` has a dead gradient only
       where a cap binds, where the per-step condensation is constant (correctly
       zero sensitivity).

    Returns the final condensation rate [kg/kg/s].
    """
    if jax.config.jax_enable_x64:
        cond_hard = _hard_saturation_condensation(
            T.astype(jnp.float64), q_v.astype(jnp.float64),
            p_full.astype(jnp.float64), dt,
            q_sat.astype(jnp.float64), sat_fn=sat_fn,
        ).astype(q_v.dtype)
    else:
        cond_hard = _hard_saturation_condensation(T, q_v, p_full, dt, q_sat,
                                                  sat_fn=sat_fn)
    rel_humidity = q_v / jnp.maximum(q_sat, _HARD_SAT_QSAT_FLOOR)
    activation = jax.nn.sigmoid(
        _HARD_SAT_MASK_SHARPNESS * (rel_humidity - hard_threshold))
    blended = condensation + activation * (cond_hard - condensation)
    condensation = jnp.minimum(blended, cond_hard)     # no overshoot past curve
    return _hard_saturation_rate_limit(condensation, q_v, dt, hard_max_heating_K)


def hard_saturation_drain(T, q_v, p_full, dt,
                          hard_threshold=_DEFAULT_HARD_SAT_THRESHOLD,
                          hard_max_heating_K=_DEFAULT_HARD_SAT_MAX_HEATING_K,
                          ice_curve=False):
    """HARD-gated hard-saturation DRAIN rate [kg/kg/s] (>= 0).

    The condensation the hard saturation adjustment applies to drain
    super-saturation ONTO the liquid curve -- the entry point for the MPAS
    driver POST-STEP hook, which corrects the dycore's per-step vertical
    vapour-transport spike on the FINAL state (the placement the integration
    trial showed is load-bearing: the in-scheme adjustment leaves that spike
    uncorrected until the next physics call, and at dt=100 s the
    transport+latent feedback detonates within that window).

    Reuses the reviewed core EXACTLY -- the bracketed-bisection on-curve solve
    :func:`_hard_saturation_condensation` and the per-step heating & vapour
    (mass-positivity) rate limit :func:`_hard_saturation_rate_limit` -- but with
    a HARD relative-humidity gate ``q_v > hard_threshold * q_sat`` (NOT the
    in-scheme smooth sigmoid).  The hard gate matches the VALIDATED post-step cap
    exactly: it drains ONLY strongly super-saturated cells (the transport spike)
    and leaves mild sub-threshold super-saturation for the next microphysics
    call, with NO sub-threshold leakage.  A hard gate is appropriate here because
    the post-step hook is EAGER (outside jit / no autodiff through it), so the
    smooth ramp the in-scheme AD path needs is unnecessary.  The gated rate is
    ``cond_hard`` (already <= the on-curve drain), so no overshoot is possible
    and the separate on-curve cap is not needed.  Result is a PURE DRAIN (>= 0):
    no evaporation of sub-saturated cells.

    Apply (conserving ``c_pd*T + L_v*q_v`` exactly)::

        dq = rate * dt ;  q_v -= dq ;  q_c += dq ;  T += (L_v/c_pd) * dq

    Parameters
    ----------
    T, q_v, p_full : array
        Temperature [K], vapour mixing ratio [kg/kg], pressure [Pa] (the TRUE
        pressure the caller uses: ``sigma_coord.pressure_at_full(p_s)``).
    dt : float
        Physics/dycore step [s].
    hard_threshold : float, default 1.1
        RH trigger; drain fires where ``q_v > hard_threshold * q_sat``.
    hard_max_heating_K : float, default 5.0
        Per-step latent-heating cap [K] (~ 2 g/kg per step); a large pool drains
        over many steps rather than detonating.
    ice_curve : bool, default False
        Gate AND land on the mixed-phase w(T)-blended liquid/ice saturation
        curve with the blended latent heat (frozen at the pre-adjustment T)
        instead of liquid-only.  At TTL temperatures (~195 K) the liquid
        curve sits ~60% above the ice curve, so the liquid-only drain leaves
        permanent ice-supersaturation — the 20x-ERA5 TTL vapour bias.
        Default False is byte-identical to the reviewed liquid-only drain.
        The caller applying the drained vapour MUST use the matching latent
        heat (:func:`mixed_phase_l_over_cp` at the SAME pre-adjustment T)
        and route cold-cell condensate to CLOUD ICE where available.

    Returns
    -------
    array
        Drain rate [kg/kg/s], >= 0.
    """
    sat_fn = (mixed_phase_saturation_specific_humidity if ice_curve
              else saturation_specific_humidity)
    if jax.config.jax_enable_x64:
        T64 = T.astype(jnp.float64)
        l_over_cp = mixed_phase_l_over_cp(T64) if ice_curve else None
        q_sat = sat_fn(
            T64, p_full.astype(jnp.float64)).astype(q_v.dtype)
        cond_hard = _hard_saturation_condensation(
            T64, q_v.astype(jnp.float64),
            p_full.astype(jnp.float64), dt,
            q_sat.astype(jnp.float64),
            sat_fn=sat_fn, l_over_cp=l_over_cp).astype(q_v.dtype)
    else:
        l_over_cp = mixed_phase_l_over_cp(T) if ice_curve else None
        q_sat = sat_fn(T, p_full)
        cond_hard = _hard_saturation_condensation(
            T, q_v, p_full, dt, q_sat, sat_fn=sat_fn, l_over_cp=l_over_cp)
    # HARD RH gate (matches the validated post-step cap): drain ONLY where
    # q_v > hard_threshold * q_sat; no sub-threshold leakage.  ``cond_hard`` is
    # the on-curve drain, so the gated rate never overshoots.
    drain = jnp.where(
        q_v > hard_threshold * jnp.maximum(q_sat, _HARD_SAT_QSAT_FLOOR),
        cond_hard, 0.0)
    # Cap the per-step heating with the SAME per-cell latent heat the caller
    # deposits with (frozen L_eff(T0)/c_pd under ice_curve; None -> scalar
    # L_v/c_pd, byte-identical liquid path) so the K cap bounds ACTUAL heating.
    lcp_cap = None if l_over_cp is None else l_over_cp.astype(q_v.dtype)
    return _hard_saturation_rate_limit(drain, q_v, dt, hard_max_heating_K,
                                       l_over_cp=lcp_cap)


def saturation_adjustment(T, q_v, p_full, dt, sharpness=_DEFAULT_SAT_SHARPNESS, q_c=None,
                          hard_adjust=False, hard_threshold=_DEFAULT_HARD_SAT_THRESHOLD,
                          hard_max_heating_K=_DEFAULT_HARD_SAT_MAX_HEATING_K,
                          sat_fn=saturation_specific_humidity):
    """Compute smooth saturation adjustment (condensation tendency).

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature [K].
    q_v : array (ncol, nlev)
        Water vapor mixing ratio [kg/kg].
    p_full : array (ncol, nlev)
        Pressure [Pa].
    dt : float
        Time step [s].
    sharpness : float
        Sigmoid sharpness for smooth condensation switch.
    q_c : array or None
        Cloud water mixing ratio [kg/kg].  When provided, the negative
        (evaporation) branch is donor-clamped against ``q_c`` so that
        evaporation cannot drive ``q_c`` below zero in subsaturated
        clear air.  Without ``q_c`` the legacy signed return is
        produced (callers must apply their own donor clamp).
    hard_adjust : bool, default False
        Opt-in hard (bracketed-bisection) saturation adjustment.  When True,
        wherever ``q_v > hard_threshold * q_sat`` the condensation rate is
        smoothly blended toward the rate that lands q_v exactly ON the liquid
        saturation curve (:func:`_hard_saturation_condensation`), draining
        local super-saturation pools that the smooth sigmoid branch cannot.
        The blend weight is a smooth sigmoid of the relative-humidity excess
        (no discontinuity), and the moist enthalpy ``c_pd*T + L_v*q_v`` is
        conserved exactly because the caller maps the returned rate to the T
        and q_v tendencies consistently.  When False the code path — and hence
        the result — is byte-identical to the historical smooth-only scheme.
    hard_threshold : float, default 1.1
        Relative-humidity trigger for the hard adjustment (dimensionless): it
        fires where ``q_v > hard_threshold * q_sat``.  The comparison uses the
        (capped) ``q_sat``, so vapour above the capped saturation value is
        still drained — the region a naive post-step cap missed.  Ignored when
        ``hard_adjust`` is False.
    hard_max_heating_K : float, default 5.0
        Per-step latent-heating cap [K] on the hard adjustment.  The condensed
        ``Delta q_v`` per step is bounded to ``hard_max_heating_K * c_pd / L_v``
        (~2 g/kg at 5 K), so a large super-saturation pool drains ONTO the curve
        over MANY steps rather than dumping its full latent heat (~165 K for a
        67 g/kg pool) in one step — the detonation the naive cap caused.  Mild
        super-saturation (full drain below the cap) still lands on the curve in
        one step.  Ignored when ``hard_adjust`` is False.
    sat_fn : callable ``(T, p) -> q_sat``, default the shared Tetens curve
        Liquid saturation specific humidity used for ``q_sat`` and the hard
        adjustment (Morrison's CAM6 MG2 mode passes CAM's Goff-Gratch
        :func:`legoesm.thermo.saturation_specific_humidity_goff_gratch`).

    Returns
    -------
    condensation : array (ncol, nlev)
        Condensation tendency [kg/kg/s].  Positive = condensation;
        negative = evaporation (donor-clamped against ``q_c`` when
        provided).  Without ``q_c``, the legacy unclamped signed
        value is returned for backward compatibility.
    q_sat : array (ncol, nlev)
        Saturation mixing ratio [kg/kg].

    Notes
    -----
    A subsaturated column with ``q_c = 0`` would otherwise produce
    spurious negative ``q_c`` after the explicit Euler step ``q_c_new
    = q_c + condensation * dt`` — Codex audit cycle 2 finding
    "subsaturated clear air can create negative cloud water".  The
    ``q_c``-aware donor clamp on the evaporation branch is the
    minimal fix that conserves total water in both clear and cloudy
    columns.
    """
    # Issue #618: marginal warm-Cu condensate is the small residual q_v - q_sat.
    # The loss is NOT the subtraction (q_v, q_sat are close ⇒ Sterbenz-exact in
    # fp32) but q_sat's fp32 COMPUTATION: the exp / softplus / division chain in
    # saturation_mixing_ratio carries ~1.7e-8 kg/kg absolute error at BOMEX
    # conditions, so a marginal supersaturation residual ~1e-6 kg/kg is ~2% low
    # per cell — which the sigmoid condensation threshold and the
    # conserving-positive rescale amplify into the ~65% LWP deficit reported in
    # fp32 LES.  When x64 is available, compute q_sat AND the residual in fp64
    # then return to the input dtype (nothing fp64 leaks into the state/carry);
    # the production LES runs fp32 arrays under jax_enable_x64=True.  The branch
    # is a STATIC Python ``if`` on the compile-time x64 flag (not traced), so the
    # x64-off path is byte-identical to the original fp32 code AND avoids the
    # spurious "float64 truncated to float32" astype warning JAX emits otherwise.
    if jax.config.jax_enable_x64:
        _state_dtype = q_v.dtype
        q_sat64 = sat_fn(T.astype(jnp.float64), p_full.astype(jnp.float64))
        excess = (q_v.astype(jnp.float64) - q_sat64).astype(_state_dtype)
        q_sat = q_sat64.astype(_state_dtype)
    else:
        q_sat = sat_fn(T, p_full)
        excess = q_v - q_sat
    dqsdt = constants.L_v * q_sat / (constants.R_v * T ** 2)
    psychrometric = 1.0 + dqsdt * constants.L_v / constants.c_pd
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / (dt * psychrometric)
    if hard_adjust:
        # Iterated hard saturation adjustment, blended in smoothly where
        # q_v > hard_threshold * q_sat, with the per-step latent-heating rate
        # limit.  The reviewed core is shared with the driver post-step hook
        # (:func:`hard_saturation_drain`) via :func:`hard_saturation_blend`.
        condensation = hard_saturation_blend(
            condensation, T, q_v, p_full, dt, q_sat,
            hard_threshold, hard_max_heating_K, sat_fn=sat_fn)
    if q_c is not None:
        # Evaporation rate (negative ``condensation``) is bounded by
        # the available cloud water: |condensation| × dt ≤ q_c, i.e.
        # condensation ≥ -q_c / dt.  ``maximum(condensation, -q_c/dt)``
        # achieves this cleanly.  Differentiable everywhere — the
        # clamp is a smooth-ish max on the evaporation magnitude.
        q_c_avail = jnp.clip(q_c, 0.0, None)
        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
    return condensation, q_sat


# --- gSAM liquid saturation adjustment (SRC/MICRO_M2005/microphysics.f90
# satadj_liquid, :1638-1720, called by micro_proc :743 before the M2005 kernel).
_GSAM_SATADJ_P_MIN = 5000.0      # no adjustment where pres < 5000 Pa (:1664) [Pa]
_GSAM_SATADJ_THRESH_K = 0.01     # convergence: |dT| <= min(0.01, 0.01*fac_cond*qsat) (:1683) [K]
_GSAM_SATADJ_THRESH_QFRAC = 0.01  # ... 0.01 of the saturation value (:1683) [-]
# Newton iteration limit (maxiter, :1650) -- a loop count, never tunable.
_GSAM_SATADJ_MAXITER = 20
# Normaliser floor of the smooth excess [kg/kg]: fp32 q_sat underflows to 0 when
# cold, and a scale whose square underflows makes the gradient NaN.
_GSAM_SATADJ_SCALE_FLOOR = 1.0e-15


def saturation_adjustment_gsam(T, q_v, q_c, p_full, *, excess_sharpness,
                               qc_floor_sharpness, p_min_width_pa):
    """gSAM liquid saturation adjustment: the adjusted ``(T, q_v, q_c)``.

    Port of gSAM 1.8.8 ``satadj_liquid`` (SRC/MICRO_M2005/microphysics.f90
    :1638-1720) as ``micro_proc`` calls it (:743): on total water
    ``q_t = q_v + q_c`` and liquid-water temperature ``T_l = T - (L_v/c_pd) q_c``
    the state is put back on the liquid saturation curve, condensing or
    evaporating cloud as needed.  Per cell::

        q_c = 0;  skip where p < 5000 Pa or q_t <= q_sat(T_l)   (q_v = q_t, T = T_l)
        dT  = (L/c_p) max(0, q_t - q_sat(T_l)) / (1 + L^2 q_sat/(c_p R_v T_l^2))
        Newton  dT = f/f',  f = T_l - T1 + (L/c_p) max(0, q_t - q_sat(T1)),
                f' = 1 + L^2 q_sat(T1)/(c_p R_v T1^2)
        while |dT| > min(0.01, 0.01 (L/c_p) q_sat(T_l)) and niter < 20
        if niter > 19: leave the cell unadjusted (q_c = 0, q_v = q_t, T = T_l)
        q_c = max(0, T1 - T_l)/(L/c_p);  q_v = q_t - q_c;  T = T1

    The iteration count and the convergence test are reproduced exactly (a
    fixed 19-step masked loop), so the result is the oracle's iterate, not
    the exact root.  The saturation curve is legoESM's
    :func:`legoesm.thermo.saturation_specific_humidity` (the oracle fixture is
    built with the same curve).  Piecewise differentiable: the branch and
    convergence masks are ``jnp.where`` and the gradient is that of the
    iterate.  Computed in the state dtype.

    KNOWN DIFFERENCE (RULE 5): three classes of hard switch are smooth unless
    their field is at the hard limit (``math.inf``, ``math.inf``, ``0.0``):

    * saturation step (``excess_sharpness`` [-]): ``max(0, q_t - q_s)`` is
      ``s_q smooth_relu((q_t - q_s)/s_q)``, ``s_q = q_sat(T_l) + 1e-15`` kg/kg
      (the floor keeps cold fp32 cells finite), and the
      ``q_t > q_sat`` skip is dropped, so a sub-saturated cell runs the same
      Newton and condenses ``~q_sat log(1 + exp(-s |S - 1|))/s`` (``S`` the
      saturation ratio; softplus is > 0, so ``T1 >= T_l``).  A small
      sharpness condenses in dry air: at 1 a dry 280 K cell reaches negative
      vapour;
    * positivity (``qc_floor_sharpness`` [1/K]): gSAM's ``max(0, T1 - T_l)``
      is ``smooth_relu``, adding up to ``log 2/qc_floor_sharpness`` K of
      condensate-equivalent warming (``T = T_l + (L/c_p) q_c`` keeps ``T_l``
      conserved); with the smooth excess the hard floor never binds (0 of 146k
      swept cells), but just above saturation (``T1 - T_l ~ 1/s``) the smooth
      floor scales the gradient by ``sigmoid(s (T1 - T_l))``, which the tests pin;
    * on/off threshold (``p_min_width_pa`` [Pa]): the ``p >= 5000 Pa`` gate
      is ``sigmoid((p - 5000)/width)`` weighting the adjusted state.

    The Newton convergence test and the 20-iteration fallback are loop
    control, not switched here.
    """
    hard_excess = math.isinf(excess_sharpness)
    if not (excess_sharpness > 0.0 and qc_floor_sharpness > 0.0 and p_min_width_pa >= 0.0):
        raise ValueError("gSAM satadj needs excess_sharpness > 0, qc_floor_sharpness > 0 "
                         f"and p_min_width_pa >= 0, got {excess_sharpness}, "
                         f"{qc_floor_sharpness}, {p_min_width_pa}")
    fac = constants.L_v / constants.c_pd
    cp_rv = constants.c_pd * constants.R_v
    lv2 = constants.L_v ** 2
    tabs = T - fac * q_c
    q_t = q_v + q_c
    qsat1 = saturation_specific_humidity(tabs, p_full)

    def excess(qs):
        if hard_excess:
            return jnp.maximum(0.0, q_t - qs)
        scale = qsat1 + _GSAM_SATADJ_SCALE_FLOOR
        return scale * smooth_relu((q_t - qs) / scale, excess_sharpness)

    sat = (q_t > qsat1) if hard_excess else jnp.ones_like(q_t, dtype=bool)
    if p_min_width_pa == 0.0:
        sat = sat & (p_full >= _GSAM_SATADJ_P_MIN)
        p_weight = 1.0
    else:
        p_weight = jax.nn.sigmoid((p_full - _GSAM_SATADJ_P_MIN) / p_min_width_pa)
    dtabs = fac * excess(qsat1) / (1.0 + lv2 * qsat1 / (cp_rv * tabs ** 2))
    tabs1 = tabs + dtabs
    niter = jnp.ones_like(tabs)
    thresh = jnp.minimum(_GSAM_SATADJ_THRESH_K,
                         _GSAM_SATADJ_THRESH_QFRAC * fac * qsat1)

    def body(_, carry):
        dtabs, tabs1, niter = carry
        go = (jnp.abs(dtabs) > thresh) & (niter < _GSAM_SATADJ_MAXITER)
        qs = saturation_specific_humidity(tabs1, p_full)
        fff = tabs - tabs1 + fac * excess(qs)
        dfff = 1.0 + lv2 * qs / (cp_rv * tabs1 ** 2)
        step = fff / dfff
        return (jnp.where(go, step, dtabs), jnp.where(go, tabs1 + step, tabs1),
                niter + go.astype(niter.dtype))

    dtabs, tabs1, niter = jax.lax.fori_loop(
        0, _GSAM_SATADJ_MAXITER - 1, body, (dtabs, tabs1, niter))
    ok = sat & (niter <= _GSAM_SATADJ_MAXITER - 1)
    if hard_excess and math.isinf(qc_floor_sharpness):
        q_c_new = jnp.where(ok, p_weight * jnp.maximum(0.0, tabs1 - tabs) / fac, 0.0)
        T_adj = tabs1 if p_min_width_pa == 0.0 else tabs + p_weight * (tabs1 - tabs)
        return jnp.where(ok, T_adj, tabs), q_t - q_c_new, q_c_new
    dT = jnp.where(ok, p_weight, 0.0) * smooth_relu(tabs1 - tabs, qc_floor_sharpness)
    q_c_new = dT / fac
    return tabs + dT, q_t - q_c_new, q_c_new


def effective_Nc(N_c, Nc_0, *, predict_Nc=True, nc_specified_field=False):
    """Effective cloud-droplet number for the size distribution / autoconversion.

    ``predict_Nc=False`` (SAM M2005 default ``dopredictNc=.false.``) returns the
    SPECIFIED constant ``Nc_0`` EVERYWHERE — droplet number is not prognostic, so
    the ``N_c`` field is ignored here (and must not be evolved). ``predict_Nc=True``
    (Seifert-Beheng / prognostic Morrison) uses the prognostic ``N_c`` where
    physical (``N_c > 1``), falling back to ``Nc_0`` for zero/garbage values — this
    also hard-guards a negative ``N_c`` so ``x_c = q_c·ρ/N_c`` can never go < 0.

    Parameters
    ----------
    N_c : array
        Cloud droplet number concentration [1/m³] (Seifert-Beheng
        per-volume convention; see Notes).
    Nc_0 : float
        Default/specified cloud droplet number [1/m³].  Typical values:
        ``1e8`` /m³ maritime, ``1e9`` /m³ continental.
    predict_Nc : bool, default True
        If False, return ``Nc_0`` everywhere (SAM specified-Nc).
    nc_specified_field : bool, default False
        Specified-but-SPATIALLY-VARYING droplet number (the
        aerosol-CCN diagnostic, ``MicrophysicsConfig.nc_from_aerosol``):
        with ``predict_Nc=False``, use the caller-filled ``N_c`` field
        where physical (``N_c > 1``) instead of the constant ``Nc_0``.
        ``N_c`` is still NOT evolved (``dN_c/dt = 0`` upstream) — it is
        a per-column diagnostic from the aerosol forcing
        (``aerosol_activation.ccn_from_aod``), refreshed every step by
        the physics pipeline.  Ignored when ``predict_Nc=True``.

    Returns
    -------
    array : Effective N_c.

    Notes
    -----
    **Convention**: this module internally uses the Seifert-Beheng
    per-volume (`[1/m³]`) convention for cloud droplet number — the
    formulas ``x_c = q_c * rho / N_c`` and ``dN_r_au = dq_c_au * rho
    / x_star`` rely on it (audit Codex cycle 2).  An earlier
    docstring labeled ``N_c`` as `[1/kg]` (per-mass), which conflicts
    with the formulas: a per-mass ``N_c`` would give ``x_c`` in
    `[kg²/m³]` rather than `[kg]`, breaking the comparison against
    ``x_star = 2.6e-10 kg``.  The default ``Nc_0 = 1e8`` is the
    canonical maritime per-volume value.
    """
    if not predict_Nc:
        if nc_specified_field:
            # Aerosol-CCN specified field: spatially varying, not evolved.
            return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))
        return Nc_0 * jnp.ones_like(N_c)
    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))


def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=_DEFAULT_SAT_SHARPNESS, gamma_norm=1.0):
    """Seifert-Beheng mass-dependent autoconversion.

    Parameters
    ----------
    q_c : array
        Cloud water mixing ratio [kg/kg].
    N_c_eff : array
        Effective cloud droplet number [1/m³] (per-volume —
        see :func:`effective_Nc` Notes).
    rho : array
        Air density [kg/m3].
    k_au : float
        Autoconversion rate constant.
    x_star : float
        Mean droplet mass threshold [kg].
    sharpness : float
        Sigmoid sharpness.
    gamma_norm : float
        Gamma distribution correction (1.0 for SB/Morrison, != 1.0 for Thompson).

    Returns
    -------
    dq_c_au : array
        Cloud water autoconversion rate [kg/kg/s].
    dN_r_au : array
        Rain number formation rate [1/(m³·s)] — per-volume, matching
        ``N_c_eff``.
    x_c : array
        Mean cloud droplet mass [kg]; with ``N_c`` in `[1/m³]`,
        ``x_c = q_c · rho / N_c`` has units
        `(kg/kg)·(kg/m³)·m³ = kg`.
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    # SB autoconversion onset is a smooth transition at the
    # mean-droplet-mass threshold ``x_star`` (~2.6e-10 kg).  Argument to
    # the sigmoid must be NORMALISED by ``x_star`` so the switch sits
    # at the right scale: with un-normalised ``sharpness · (x_c − x_star)``
    # at ``sharpness = 50`` the sigmoid argument is O(5e-9) for any
    # physical ``x_c`` and ``onset`` stuck at ≈ 0.5 — the threshold is
    # effectively disabled and droplet-poor columns auto-converted at
    # half strength.  Normalising by ``x_star`` makes ``sharpness`` a
    # dimensionless steepness in fractional units of ``x_star``: the
    # sigmoid then sweeps from 0 (x_c ≪ x_star) to 1 (x_c ≫ x_star)
    # over an O(1/sharpness) range around x_c = x_star, matching the
    # canonical SB 2001 / Seifert 2008 switch behaviour.
    onset = jax.nn.sigmoid(sharpness * (x_c / x_star - 1.0))
    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
    # SB2001/2006 number closure: each newborn rain drop carries the
    # separation mass x_* (SB2001 Eq. for ∂N_r/∂t|_au = au/x_*), so the
    # number source is the mass rate divided by x_star ALONE.  The factor
    # 1/20 in SB2001 belongs to the MASS-rate coefficient k_au = k_cc/(20·x_*)
    # — here k_au is an independent tunable — and a transplanted ·20 divisor
    # previously made newborn drops 20× x_* (~215 µm instead of ~79 µm),
    # biasing rain fall speed fast and rain evaporation low.
    dN_r_au = dq_c_au * rho / x_star
    return dq_c_au, dN_r_au, x_c


def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
    """Rain collecting cloud water (accretion) — SIMPLIFIED BILINEAR SURROGATE.

    Rate = ``k_ac · q_c · q_r · rho · gamma_norm``.  This is the accretion
    form wired into Morrison (non-KK2000 path), Seifert-Beheng and Thompson;
    it is a tunable-coefficient SURROGATE, NOT faithful to the published
    Khairoutdinov-Kogan (2000) accretion that P3 and SAM-M2005 use
    (``module_mp_p3.f90:3615``, iparam=3 / ``module_mp_graupel.f90:1952``).
    This repo's P3 (the WRF v4.8.0 port) calls :func:`accretion_kk2000`.  The gSAM form, in a single column with the sub-grid
    cloud/precip-fraction scheme off (``iSCF=iSPF=1``, ``dum2=1``), reduces to
    the mass rate::

        PRA = 67 · (q_c · q_r)^1.15          [kg/kg/s]   (the faithful form)

    available as :func:`accretion_kk2000`.  DEPARTURES of this surrogate from
    that oracle: (1) coefficient ``k_ac`` (default 5.25) vs the published 67;
    (2) BILINEAR ``q_c·q_r`` (product exponent 1) vs KK2000's ``(q_c·q_r)^1.15``;
    (3) a spurious ``rho`` factor — KK2000 is a pure mixing-ratio rate with NO
    density dependence; (4) an optional ``gamma_norm`` PSD-shape modifier absent
    from KK2000.
    Pinned + canaried in
    ``tests/atmosphere/microphysics/unit/test_p3_accretion_faithful.py``.

    Parameters
    ----------
    q_c, q_r : array
        Cloud water and rain mixing ratios [kg/kg] (negatives clipped to 0).
    rho : array
        Air density [kg/m3].
    k_ac : float
        Accretion rate constant [m^3/(kg*s)].
    gamma_norm : float
        Gamma distribution correction (1.0 = none).

    Returns
    -------
    array : Accretion rate [kg/kg/s].
    """
    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm


# KK2000 autoconverted-drop mass = mass of a 25 µm-radius water drop
# (SAM M2005 ``CONS29 = 4/3·π·ρ_w·(25e-6)³``, module_mp_graupel.f90:595);
# the mass at which newly autoconverted droplets enter the rain category.
_KK2000_DROP_RADIUS = 25.0e-6  # [m]
_KK2000_CONS29 = (
    (4.0 / 3.0) * jnp.pi * constants.rho_water * _KK2000_DROP_RADIUS ** 3
)

# KK2000 warm-rain rate constants (SAM M2005 module_mp_graupel.f90:1813 / :1952):
#   PRC = 1350·q_c^2.47·N_c[#/cm^3]^-1.79   (autoconversion, kg/kg/s)
#   PRA = 67·(q_c·q_r)^1.15                  (accretion, kg/kg/s)
# Published fixed coefficients (Khairoutdinov & Kogan 2000), not tunable here.
_KK2000_AUTOCONV_PREFACTOR = 1350.0
_KK2000_AUTOCONV_QC_EXPONENT = 2.47
_KK2000_AUTOCONV_NC_EXPONENT = -1.79
_KK2000_ACCRETION_PREFACTOR = 67.0
_KK2000_ACCRETION_EXPONENT = 1.15

# CAM6 MG2 KK2000 autoconversion (micro_mg_utils.F90:689-736,
# kk2000_liq_autoconversion), published CAM6 constants:
#   prc = var_coef(relvar, 2.47) * 0.01 * 1350 * qcic^2.47
#         * (ncic[#/kg] * 1e-6 * rho)^-1.1          [kg/kg/s]
#   nprc  = prc / droplet_mass_25um   (rain number source, per kg)
#   nprc1 = prc * ncic / qcic         (cloud number sink, per kg)
# gated on qcic >= icsmall (1e-8, micro_mg_utils.F90:160); qcic capped at
# 5e-3 kg/kg by micro_mg2_0.F90:1226 before the call.
_KK2000_CAM6_PREFACTOR = 0.01 * 1350.0
_KK2000_CAM6_NC_EXPONENT = -1.1
_KK2000_CAM6_QC_MIN = 1.0e-8    # icsmall [kg/kg]
KK2000_CAM6_QC_MAX = 5.0e-3     # in-cloud cap [kg/kg] (shared with morrison.py)
_KK2000_CAM6_RELVAR_MIN = 1.0e-3   # relvar clip (clubb_intr.F90:2425)
_KK2000_CAM6_RELVAR_MAX = 10.0     # relvarmax, non-CLUBB deep scheme (:2418)

# --- CAM6 MG2 in-cloud warm-rain inputs (micro_mg2_0.F90, CESM2.1) ---
# lcldm = max(liqcldf, mincld) (:878), micro_mg_utils.F90:123 mincld = 1e-4;
# qcic = min(qc/lcldm, 5e-3), ncic = max(nc/lcldm, 0), both 0 where qc < qsmall
# (:1224-1236); qric = min(qr/precip_frac, 0.01), 0 where qric < qsmall
# (:1312-1322).  precip_frac = cldm, "in_cloud" method under CLUBB
# (namelist_defaults_cam.xml micro_mg_precip_frac_method, :1260-1266): only a
# level with qc < qsmall AND qi < qsmall inherits the level above's value, and
# there qcic = 0, so every warm-rain rate is zero whatever qric is.  For the
# warm-rain consumers here precip_frac therefore equals lcldm (cldm = lcldm
# with CLUBB's fraction as both).
MG2_MINCLD = 1.0e-4       # micro_mg_utils.F90:123 mincld [-]
MG2_QSMALL = 1.0e-18      # micro_mg_utils.F90 qsmall [kg/kg]
MG2_QRIC_MAX = 0.01       # micro_mg2_0.F90:1316 in-precip rain cap [kg/kg]


def mg2_presence(q, top):
    """How much of MG2's ``q >= qsmall`` condensate-presence test holds.

    CAM6 MG2 switches each process on at ``q >= qsmall = 1e-18 kg/kg``.  Here
    the switch is the C2 step, 0 at qsmall and 1 from ``top`` [kg/kg] up (user
    decision 2026-10-22, 1a: 1e-17; ``MorrisonConfig.mg2_presence_ramp_top``).
    ``top <= qsmall`` is CAM's hard test.  KNOWN DIFFERENCE: only condensate
    between 1e-18 and 1e-17 kg/kg sees a partial process.  Measured against
    the CAM-matching hard limit (max tendency change / column's largest MG2
    tendency): 0 in every oracle fixture column except one sedimentation
    column with sub-qsmall cloud water and droplets (0.999, its droplet
    number) and 9e-13 in one ice column (test_morrison_mg2_presence_ramp.py).
    """
    if top <= MG2_QSMALL:
        return jnp.where(q >= MG2_QSMALL, 1.0, 0.0)
    return smoothstep_quintic((q - MG2_QSMALL) / (top - MG2_QSMALL))

# --- CAM6 MG2 cloud-droplet size distribution (size_dist_param_liq,
# micro_mg_utils.F90:376-422 + size_dist_param_basic :481-516) ---
# pgam = 1/(1 - 0.7 exp(-0.008e-6 * ncic*rho))^2 - 1, floored at 2 (Martin et
# al. 1994 fit, :395-397); lambda bounded so the mean diameter stays in
# [2, 50] um (:409-410); ncic <= qcic/min_mean_mass_liq (:173, :493-494).
_MG2_PGAM_A = 0.7
_MG2_DROP_FALL_BC = 2.0    # micro_mg_utils.F90:133 bc, droplet fall-speed exponent
_STOKES_DRAG_FACTOR = 18.0  # Stokes law v = g rho_w D^2/(18 mu): MG2 acn, micro_mg2_0.F90:899
_MG2_PGAM_B = 0.008e-6     # [cm^3 -> m^3 conversion folded in] per (#/m^3)
_MG2_PGAM_MIN = 2.0
_MG2_DROP_DIAM_MAX = 50.0e-6   # [m] -> lambda lower bound (pgam+1)/50 um
_MG2_DROP_DIAM_MIN = 2.0e-6    # [m] -> lambda upper bound (pgam+1)/2 um
_MG2_MIN_MEAN_MASS_LIQ = 1.0e-20   # [kg]
_MG2_PGAM_UNSET = -100.0       # pgam where qcic <= qsmall (:419, "likely to cause an error")
# Droplet effective radius where the in-cloud water is below qsmall: CAM6
# REL fill 10 um (micro_mg_cam.F90:2536; micro_mg2_0.F90:2859 effc = 10).
MG2_REL_NO_LIQUID_M = 10.0e-6      # [m]
# --- CAM6 MG2 rain processes (micro_mg_utils.F90) ---
_MG2_BEHENG_SELFCOLL = 8.0     # self_collection_rain nragg = -8 nric qric rho (:1359, Beheng 1994)
_MG2_EVAP_CLOUD_QMIN = 1.0e-6  # evaporate_sublimate_precip: dum=0 if qcic+qiic < 1e-6 (:1468)
# --- CAM6 MG2 instantaneous ice melt / homogeneous freezing (micro_mg2_0.F90:2615-2685) ---
_MG2_MELT_DROP_R3 = 5.12e-16   # [m^3] (8 um)^3: droplets from melted ice, mean-volume radius (:2641)
_MG2_HOMOG_ICE_R3 = 1.563e-14  # [m^3] (25 um)^3: homogeneously frozen crystals (:2677)
_MG2_HOMOG_ICE_RHO = 500.0     # [kg/m^3] their density (:2678)
_MG2_HOMOG_PI = 3.14           # MG2's literal at :2677 (not pi), kept exact
# Smooth switch-on width W = 4/sharpness: the 12-88 % spread of the existing
# sigmoid(sharpness*x) ramps (user decision 2026-10-20, 1a).
RAMP_WIDTH_PER_SHARPNESS = 4.0
# Sharpness of the rounded "all of it / all the heat" cap, smooth_min(1, r)
# (core.smooth default; user decision 2026-10-20, 2b): -0.7 % at r = 1.
MG2_CAP_SHARPNESS = 100.0


def mg2_incloud_warm_rain_inputs(q_c, q_r, N_c, cloud_fraction, rescale_nc,
                                 precip_frac):
    """CAM6 MG2 in-cloud warm-rain inputs ``(q_c_ic, q_r_ic, N_c_ic, lcldm)``.

    ``cloud_fraction`` is MG2's ``liqcldf`` (= ``cldn`` = CAM6 ``ast`` =
    max(alst, aist) under CLUBB), shape ``(ncol, nlev)``.
    Rates evaluated on these inputs are multiplied back by ``lcldm`` to give
    grid-mean tendencies (micro_mg2_0.F90:1666, :1890, :1949, :1966).
    ``rescale_nc`` divides ``N_c`` by ``lcldm`` (a grid-mean prognostic
    number); a SPECIFIED droplet number is already the in-cloud value (MG2
    ``nccons``: ``ncic = ncnst/rho``), so it passes through.  ``q_r_ic`` is
    MG2's in-precip ``qric = min(qr/precip_frac, 0.01)`` (0 below qsmall,
    :1312-1322) with ``precip_frac`` from :func:`mg2_precip_frac`.
    """
    lcldm = jnp.maximum(cloud_fraction, MG2_MINCLD)
    has_qc = q_c >= MG2_QSMALL
    q_c_ic = jnp.where(has_qc, jnp.minimum(q_c / lcldm, KK2000_CAM6_QC_MAX),
                       0.0)
    N_c_ic = N_c
    if rescale_nc:
        N_c_ic = jnp.where(has_qc, jnp.maximum(N_c / lcldm, 0.0), 0.0)
    q_r_ic = jnp.minimum(q_r / precip_frac, MG2_QRIC_MAX)
    q_r_ic = jnp.where(q_r_ic < MG2_QSMALL, 0.0, q_r_ic)
    return q_c_ic, q_r_ic, N_c_ic, lcldm


def mg2_precip_frac(cloud_fraction, q_c, q_i):
    """CAM6 MG2 precipitation fraction, ``micro_mg_precip_frac_method =
    'in_cloud'`` (the CAM6 + CLUBB namelist default; micro_mg2_0.F90:1256-1266).

    ``precip_frac = cldm = max(cloud_fraction, mincld)``; walking DOWN from the
    model top, a level with ``qc < qsmall`` AND ``qi < qsmall`` inherits the
    level above's value.  Level 0 = model top (this module's convention, the
    same as CAM's k=1).  Shape ``(ncol, nlev)``.
    """
    cldm = jnp.maximum(cloud_fraction, MG2_MINCLD)
    clear = (q_c < MG2_QSMALL) & (q_i < MG2_QSMALL)

    def _down(prev, xs):
        c, clr = xs
        pf = jnp.where(clr, prev, c)
        return pf, pf

    top = cldm[:, 0]
    _, below = jax.lax.scan(_down, top, (cldm[:, 1:].T, clear[:, 1:].T))
    return jnp.concatenate([top[:, None], below.T], axis=1)


def mg2_var_coef(relvar, a, dtype):
    """MG2 sub-grid variance coefficient ``var_coef(relvar, a) =
    Gamma(relvar+a)/(Gamma(relvar) relvar^a)`` (micro_mg_utils.F90:574-583),
    with relvar clipped to CAM6's [0.001, relvarmax=10] (clubb_intr.F90:2425).
    """
    r = jnp.clip(jnp.asarray(relvar, dtype=dtype),
                 _KK2000_CAM6_RELVAR_MIN, _KK2000_CAM6_RELVAR_MAX)
    return jnp.exp(jax.scipy.special.gammaln(r + a)
                   - jax.scipy.special.gammaln(r)) / r ** a


def rule5_sharpness(width):
    """LogSumExp sharpness of a RULE 5 width; 0 = the hard form."""
    return math.inf if width == 0.0 else 1.0 / width


def mg2_drop_ln_number_cap(nc_kg, q_c_ic, log_width):
    """``ln min(nc_kg, q/1e-20 kg)`` (MG2 min_mean_mass_liq,
    micro_mg_utils.F90:493-494) as a LogSumExp of ``log_width`` in ln space
    (0 = the hard min).  ``nc_kg`` floored at the dtype's tiny for the log."""
    tiny = jnp.finfo(jnp.result_type(nc_kg)).tiny
    return smooth_min(jnp.log(jnp.maximum(nc_kg, tiny)),
                      jnp.log(q_c_ic / _MG2_MIN_MEAN_MASS_LIQ),
                      rule5_sharpness(log_width))


def mg2_droplet_psd(q_c_ic, N_c_ic, rho, *, number_cap_log_width=0.0,
                    slope_log_width=0.0):
    """MG2 ``size_dist_param_liq`` (micro_mg_utils.F90:376-422 ->
    size_dist_param_basic :481-516), per-VOLUME number here.

    Where ``qcic > qsmall``: ``pgam = max(1/(1-0.7 exp(-0.008e-6 N))^2 - 1, 2)``
    (``N = ncic*rho`` [#/m^3]), ``ncic <= qcic/1e-20``,
    ``lambda = (pi/6 rho_w (pgam+1)(pgam+2)(pgam+3) ncic/qcic)^(1/3)`` set to
    the bound (and ``ncic`` reset to match) when it leaves
    ``(pgam+1)/[50 um, 2 um]``.  Returns ``(active, pgam, lambda, ncic)``
    with ``ncic`` per kg; ``pgam``/``lambda``/``ncic`` are finite but
    meaningless where ``active`` is False (placeholder ``qcic = 1``).
    """
    active = q_c_ic > MG2_QSMALL
    q_safe = jnp.where(active, q_c_ic, 1.0)
    pgam = 1.0 - _MG2_PGAM_A * jnp.exp(-_MG2_PGAM_B * N_c_ic)
    pgam = jnp.maximum(1.0 / pgam ** 2 - 1.0, _MG2_PGAM_MIN)
    shape = (jnp.pi / 6.0 * constants.rho_water
             * (pgam + 1.0) * (pgam + 2.0) * (pgam + 3.0))
    lam_lo = (pgam + 1.0) / _MG2_DROP_DIAM_MAX
    lam_hi = (pgam + 1.0) / _MG2_DROP_DIAM_MIN
    if number_cap_log_width == 0.0 and slope_log_width == 0.0:
        nc_kg = jnp.minimum(N_c_ic / rho, q_safe / _MG2_MIN_MEAN_MASS_LIQ)
        lam = safe_pow(shape * nc_kg / q_safe, 1.0 / 3.0)
        nc_kg = jnp.where(lam < lam_lo, lam_lo ** 3 * q_safe / shape,
                          jnp.where(lam > lam_hi, lam_hi ** 3 * q_safe / shape,
                                    nc_kg))
        return active, pgam, jnp.clip(lam, lam_lo, lam_hi), nc_kg
    # RULE 5: the number cap and the slope clamp in ln space, each a LogSumExp
    # of its width (0 = MG2's hard form); the number follows the smooth slope.
    ln_nc = mg2_drop_ln_number_cap(N_c_ic / rho, q_safe, number_cap_log_width)
    ln_lam = (jnp.log(shape / q_safe) + ln_nc) / 3.0
    lam = jnp.exp(smooth_clamp(ln_lam, jnp.log(lam_lo), jnp.log(lam_hi),
                               rule5_sharpness(slope_log_width)))
    return active, pgam, lam, lam ** 3 * q_safe / shape


def mg2_size_dist_liq(q_c_ic, N_c_ic, rho):
    """MG2 ``size_dist_param_liq`` (micro_mg_utils.F90:376-422):
    ``(N_c_ic_adjusted, pgam, lamc)`` with the number per-VOLUME, see
    :func:`mg2_size_bounded_nc`.  ``lamc`` is the slope clipped to
    ``(pgam+1)/[50 um, 2 um]``; where ``qcic <= qsmall`` MG2 sets ``pgam=-100``
    and ``lamc=0`` and so does this.
    """
    active, pgam, lam, nc_kg = mg2_droplet_psd(q_c_ic, N_c_ic, rho)
    return (jnp.where(active, nc_kg * rho, N_c_ic),
            jnp.where(active, pgam, _MG2_PGAM_UNSET),
            jnp.where(active, lam, 0.0))


def mg2_size_bounded_nc(q_c_ic, N_c_ic, rho, **widths):
    """MG2 ``size_dist_param_liq`` droplet-number reset
    (:func:`mg2_droplet_psd`, ``widths`` its RULE 5 widths), per-VOLUME.
    Elsewhere (``qcic <= qsmall``) ``N_c_ic`` is returned unchanged.
    """
    active, _, _, nc_kg = mg2_droplet_psd(q_c_ic, N_c_ic, rho, **widths)
    return jnp.where(active, nc_kg * rho, N_c_ic)


def mg2_droplet_fall_coefficients(q_c, N_c_kg, lcldm, T, rho, *, presence_top):
    """MG2 droplet ``(fc, fnc)`` [Pa/s per unit mixing ratio] (micro_mg2_0.F90:
    2056-2120): ``g rho`` x the mass- / number-weighted Stokes fall speed of the
    in-cloud size distribution of the grid state ``(q_c, N_c_kg)``; 0 where the
    in-cloud water is below qsmall, switched on by :func:`mg2_presence`."""
    g = constants.g
    has_c = q_c / lcldm >= MG2_QSMALL
    on = mg2_presence(q_c / lcldm, presence_top)
    q_ic = jnp.where(has_c, q_c / lcldm, 0.0)
    n_ic = jnp.where(has_c, jnp.maximum(N_c_kg / lcldm, 0.0), 0.0)
    _, pgam, lamc = mg2_size_dist_liq(q_ic, n_ic * rho, rho)
    # MG2 gates the fall speed at qcic >= qsmall but the size distribution at
    # qcic > qsmall, so qcic == qsmall exactly divides by lamc = 0 there (inf
    # in the Fortran too); such a cell falls at 0 here.
    falls = has_c & (lamc > 0.0)
    pgam = jnp.where(falls, pgam, 0.0)  # MG2's -100 marker would put gammaln off its domain
    mu = _MU_PREFACTOR * safe_pow(T, _MU_T_EXPONENT) / (T + _MU_SUTHERLAND_T)
    acn = g * constants.rho_water / (_STOKES_DRAG_FACTOR * mu)
    bc = _MG2_DROP_FALL_BC
    lam_bc = safe_pow(jnp.where(falls, lamc, 1.0), bc)
    lg = jax.scipy.special.gammaln
    fc = on * jnp.where(falls, g * rho * acn * jnp.exp(lg(4.0 + bc + pgam) - lg(pgam + 4.0)) / lam_bc, 0.0)
    fnc = on * jnp.where(falls, g * rho * acn * jnp.exp(lg(1.0 + bc + pgam) - lg(pgam + 1.0)) / lam_bc, 0.0)
    return fc, fnc


def mg2_cloud_sedimentation(q_c, N_c_kg, lcldm, T, rho, p_half, dt, n_substeps_max, *,
                            presence_top):
    """CAM6 MG2 cloud-droplet sedimentation (micro_mg2_0.F90:2056-2120,
    :2204-2216, :2319-2377) on the POST-process grid state.

    ``q_c`` [kg/kg] and ``N_c_kg`` [1/kg] are ``state + tendency*dt``; ``lcldm`` =
    max(cloud fraction, mincld); ``T``/``rho`` the start-of-step temperature and
    the caller's air density; ``p_half`` interfaces [Pa], level 0 = model top.
    Fall speeds (fixed over sub-steps) from the in-cloud size distribution:
    ``acn = g rho_w/(18 mu)``, ``bc = 2``,
    ``fc = g rho acn G(4+bc+pgam)/(lamc^bc G(pgam+4))``,
    ``fnc = g rho acn G(1+bc+pgam)/(lamc^bc G(pgam+1))`` where in-cloud q_c >= qsmall.
    Sub-steps per column ``nstep = 1 + floor(max_k max(fc, fnc)/pdel * dt)``,
    clipped to ``n_substeps_max``; flux ``F = fc q``; level k gains
    ``d F(k-1)`` with ``d = min(lcldm(k)/lcldm(k-1), 1)``; the inflow share
    ``(1-d) F(k-1)`` falling into clear air evaporates (vapour source, latent
    cooling ``L_v/c_pd``).  Returns ``(dq_c, dN_c_kg, dq_v, dT, sfc_flux,
    nstep_required)`` [kg/kg/s, 1/(kg s), kg/kg/s, K/s, kg/(m^2 s), int].
    """
    g = constants.g
    pdel = p_half[:, 1:] - p_half[:, :-1]
    fc, fnc = mg2_droplet_fall_coefficients(q_c, N_c_kg, lcldm, T, rho,
                                            presence_top=presence_top)
    c0 = q_c
    n0 = mg2_presence(q_c, presence_top) * jnp.maximum(N_c_kg, 0.0)
    courant = jnp.max(jnp.maximum(fc, fnc) / pdel, axis=1) * dt
    nstep_req = 1 + jnp.floor(courant).astype(jnp.int32)
    nsub = jnp.clip(nstep_req, 1, n_substeps_max)[:, None]
    inv = 1.0 / nsub.astype(q_c.dtype)
    ratio = jnp.minimum(lcldm[:, 1:] / lcldm[:, :-1], 1.0)
    d = jnp.concatenate([jnp.zeros_like(ratio[:, :1]), ratio], axis=1)

    def _shift(x):  # value of the level above; nothing enters the top
        return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)

    def body(i, carry):
        c, n, dqc, dn, dqv, sfc = carry
        on = (i < nsub).astype(q_c.dtype)
        F, Fn = fc * c, fnc * n
        tend_c = (F - d * _shift(F)) / pdel
        tend_n = (Fn - d * _shift(Fn)) / pdel
        evap = (1.0 - d) * _shift(F) / pdel
        w = on * inv
        return (c - tend_c * dt * w, n - tend_n * dt * w, dqc - tend_c * w,
                dn - tend_n * w, dqv + evap * w, sfc + F[:, -1] / g * w[:, 0])

    z = jnp.zeros_like(q_c)
    _, _, dqc, dn, dqv, sfc = jax.lax.fori_loop(
        0, int(n_substeps_max), body, (c0, n0, z, z, z, jnp.zeros_like(q_c[:, 0])))
    return dqc, dn, dqv, -constants.L_v * dqv / constants.c_pd, sfc, nstep_req



def mg2_phase_change_fraction(excess_K, mass, width_K, cap_sharpness):
    """Fraction of ``mass`` an MG2 instant melt/freeze converts once the
    temperature is ``excess_K`` past its threshold: the heat-limited fraction
    min(excess*c_p/(L_f*mass), 1) (micro_mg2_0.F90:2621-2631, :2658-2668),
    with the min rounded by smooth_min (never above the hard cap), scaled by a
    C2 step that is exactly 0 at the threshold and 1 after ``width_K``.  Both
    are deliberate departures from CAM for differentiability (user decisions
    2026-10-20).  0 where ``mass <= 0``.  The heat-limit ratio is computed
    only where ``mass >= qsmall`` (MG2_QSMALL); below it the fraction is 1
    (user decision 2026-10-26 b): ``excess/mass`` overflows there (mass
    1e-219 gave r ~ -8e217 and a NaN gradient through smooth_min, although
    the ramp is 0).  Below qsmall the cap binds only within ~3e-16 K of
    the threshold, where the ramp is ~1e-46.
    """
    has = mass > 0.0
    big = mass >= MG2_QSMALL
    r = excess_K * constants.c_pd / (constants.L_f * jnp.where(big, mass, 1.0))
    # max(., 0): smooth_min(1, 0) is -3.7e-46, not 0; the ramp is 0 there anyway
    frac = jnp.where(big, jnp.maximum(smooth_min(jnp.ones_like(r), r, cap_sharpness), 0.0),
                     1.0)
    return jnp.where(has, smoothstep_quintic(excess_K / width_K) * frac, 0.0)


def mg2_instant_ice_melt_homog_freeze(q_c, N_c_kg, q_i, N_i_kg, T, T_homog,
                                      melt_width_K, freeze_width_K,
                                      cap_sharpness=MG2_CAP_SHARPNESS, *, presence_top):
    """CAM6 MG2 instantaneous ice melt then homogeneous droplet freezing on the
    RAW post-sedimentation grid-mean state (q + tend*dt, may be negative;
    micro_mg2_0.F90:2503-2529 pools, :2615-2685).

    Returns post-step targets ``(q_c, N_c_kg, q_i, N_i_kg, T, melt, frz,
    changed)``: ``melt``/``frz`` are the converted masses [kg/kg] and
    ``changed`` marks cells either process touched.  MG2 semantics: the donor
    is OVERWRITTEN from its clipped pool ((1-f)*pool), the recipient is
    INCREMENTED on its raw value.  Melted ice makes 8 um droplets; frozen
    droplets make 25 um, density-500 crystals counted from the frozen MASS
    (user decision 2026-10-20, 1a).  Numbers per kg.  Homogeneous freezing
    reads the pre-melt pools and the post-melt temperature, as MG2.
    """
    dumc = jnp.maximum(q_c, 0.0)
    dumi = jnp.maximum(q_i, 0.0)
    dumnc = mg2_presence(dumc, presence_top) * jnp.maximum(N_c_kg, 0.0)
    dumni = mg2_presence(dumi, presence_top) * jnp.maximum(N_i_kg, 0.0)

    x_m = T - constants.T_freeze
    f_m = mg2_phase_change_fraction(x_m, dumi, melt_width_K, cap_sharpness)
    fired_m = (dumi > 0.0) & (x_m > 0.0)
    melt = f_m * dumi
    q_i1 = jnp.where(fired_m, (1.0 - f_m) * dumi, q_i)
    N_i1 = jnp.where(fired_m, (1.0 - f_m) * dumni, N_i_kg)
    q_c1 = q_c + melt
    N_c1 = N_c_kg + 3.0 * melt / (4.0 * jnp.pi * _MG2_MELT_DROP_R3 * constants.rho_water)
    T1 = T - constants.L_f * melt / constants.c_pd

    x_h = T_homog - T1
    f_h = mg2_phase_change_fraction(x_h, dumc, freeze_width_K, cap_sharpness)
    fired_h = (dumc > 0.0) & (x_h > 0.0)
    frz = f_h * dumc
    q_c2 = jnp.where(fired_h, (1.0 - f_h) * dumc, q_c1)
    N_c2 = jnp.where(fired_h, (1.0 - f_h) * dumnc, N_c1)
    q_i2 = q_i1 + frz
    N_i2 = N_i1 + frz * 3.0 / (4.0 * _MG2_HOMOG_PI * _MG2_HOMOG_ICE_R3 * _MG2_HOMOG_ICE_RHO)
    T2 = T1 + constants.L_f * frz / constants.c_pd
    return q_c2, N_c2, q_i2, N_i2, T2, melt, frz, fired_m | fired_h

def mg2_droplet_number_readjust(q_c_post, N_c_post, N_c, cf, rho, dt, dN_c_dt, *,
                                presence_top):
    """CAM6 MG2 final droplet mean-size re-adjust (micro_mg2_0.F90:2743-2753,
    :2812-2835) on the grid-mean post-step state (per-volume number): in-cloud
    dumc = min(max(q_post,0)/cf, 5e-3), dumnc = max(N_post,0)/cf; where
    dumc >= qsmall and size_dist_param_liq moves the number, the tendency
    becomes (dumnc_adj*cf - N_c)/dt.  At dumc == qsmall exactly the size
    routine (qcic > qsmall) leaves the number, so a negative one survives, as
    in MG2.
    """
    q_ic = jnp.minimum(jnp.maximum(q_c_post, 0.0) / cf, KK2000_CAM6_QC_MAX)
    n_ic = jnp.maximum(N_c_post, 0.0) / cf
    n_adj = mg2_size_bounded_nc(q_ic, n_ic, rho)
    f = mg2_presence(q_ic, presence_top) * jnp.where(n_adj != n_ic, 1.0, 0.0)
    return f * ((n_adj * cf - N_c) / dt) + (1.0 - f) * dN_c_dt



@jax.custom_jvp
def _div_tangent_safe(x, y):
    """``x / y``, forward bitwise identical to the division.  The tangent
    ``(dx - (x/y) dy)/y`` replaces JAX's ``-x dy/y**2``, whose ``y**2``
    underflows to 0 in fp32 for ``y`` ~ 1e-19 (in-precip rain near qsmall at
    rho ~ 0.1, oracle row 17) and turns a finite derivative into inf/NaN.
    First order only: here 2x/y**3 ~ 1e48 exceeds the fp32 range, so fp32
    second derivatives of the rain chain near qsmall stay non-finite (fp64
    is finite)."""
    return x / y


@_div_tangent_safe.defjvp
def _div_tangent_safe_jvp(primals, tangents):
    (x, y), (dx, dy) = primals, tangents
    r = x / y
    return r, (dx - r * dy) / y


def mg2_droplet_effective_radius(q_c_ic, N_c_ic, rho, **widths):
    """CAM6 radiation droplet effective radius REL [m] (micro_mg_cam.F90:
    2534-2554): ``(pgam+3)/lambda/2`` from :func:`mg2_droplet_psd` where the
    in-cloud water ``qcic >= qsmall``, else 10 um.

    ``q_c_ic`` is CAM6's ``icwmrst = min(q/max(mincld, ast), 5e-3)``
    (micro_mg_cam.F90:2296), ``N_c_ic`` the in-cloud number [#/m^3].
    """
    _, pgam, lam, _ = mg2_droplet_psd(q_c_ic, N_c_ic, rho, **widths)
    return jnp.where(q_c_ic >= MG2_QSMALL, (pgam + 3.0) / lam / 2.0,
                     MG2_REL_NO_LIQUID_M)


def mg2_inprecip_rain(q_r, N_r, rho, precip_frac, lamr_min, lamr_max):
    """CAM6 MG2 in-precip rain state ``(q_ric, N_ric, lamr, n0r)``.

    ``qric = min(qr/precip_frac, 0.01)``, ``nric = max(nr/precip_frac, 0)``,
    both 0 where ``qric < qsmall`` (micro_mg2_0.F90:1312-1328), then
    ``size_dist_param_basic(mg_rain_props, ...)`` (micro_mg_utils.F90:481-516):
    where ``qric > qsmall``, ``lamr = (pi rho_w nric/qric)^(1/3)`` and, if it
    leaves ``[lamr_min, lamr_max]``, it is set to the bound and ``nric`` is
    re-adjusted to match; ``n0r = nric*lamr``; ``lamr = 0`` elsewhere.
    ``N_r``/``N_ric``/``n0r`` are per-VOLUME (this module's rain convention:
    MG2's per-kg value times rho).  At ``qric == qsmall`` exactly MG2 (and
    this port) leave ``lamr = n0r = 0``; consumers gate on ``lamr > 0``
    (oracle row 18).  The slope's division carries an fp32-safe tangent
    (:func:`_div_tangent_safe`, oracle row 17); the forward is unchanged.
    """
    q_ric = jnp.minimum(q_r / precip_frac, MG2_QRIC_MAX)
    small = q_ric < MG2_QSMALL
    q_ric = jnp.where(small, 0.0, q_ric)
    N_ric = jnp.where(small, 0.0, jnp.maximum(N_r / precip_frac, 0.0))
    active = q_ric > MG2_QSMALL
    cr = jnp.pi * constants.rho_water
    den = jnp.where(active, rho * q_ric, 1.0)
    lam_raw = safe_pow(_div_tangent_safe(cr * N_ric, den), 1.0 / 3.0)
    lam = jnp.clip(lam_raw, lamr_min, lamr_max)
    N_adj = jnp.where((lam_raw < lamr_min) | (lam_raw > lamr_max),
                      lam ** 3 * den / cr, N_ric)
    N_ric = jnp.where(active, N_adj, N_ric)
    lamr = jnp.where(active, lam, 0.0)
    return q_ric, N_ric, lamr, N_ric * lamr


def self_collection_rain_mg2(q_ric, N_ric, rho, precip_frac):
    """CAM6 MG2 rain self-collection (Beheng 1994; ``self_collection_rain``,
    micro_mg_utils.F90:1343-1364), grid-mean per-VOLUME number rate [1/m^3/s].

    In-precip ``nragg = -8 nric qric rho`` [1/kg/s] where ``qric >= qsmall``
    (``nric`` the size-adjusted in-precip number of :func:`mg2_inprecip_rain`),
    times ``precip_frac`` for the grid mean (micro_mg2_0.F90:1965-1967), times
    rho for this module's per-volume rain number.
    """
    nragg = jnp.where(q_ric >= MG2_QSMALL,
                      -_MG2_BEHENG_SELFCOLL * (N_ric / rho) * q_ric * rho, 0.0)
    return nragg * precip_frac * rho


def rain_evaporation_mg2(q_v, q_sat, T, p, rho, q_ric, lamr, n0r, lcldm,
                         precip_frac, q_cic, q_iic, config):
    """CAM6 MG2 rain evaporation, grid mean, returned POSITIVE (a q_r sink /
    q_v source) [kg/kg/s] (``evaporate_sublimate_precip``,
    micro_mg_utils.F90:1422-1536; grid = ``-pre*precip_frac``,
    micro_mg2_0.F90:1977).

    Rain evaporates only in the precipitating CLEAR air: ``dum = lcldm``
    unless ``qcic + qiic < 1e-6`` (then 0); where ``precip_frac > dum`` and
    ``qric >= qsmall``, ``am_evp_st = precip_frac - dum`` and the clear-air
    humidity is ``qclr = (q - dum*qvl)/(1 - dum)``; ``pre = eps (qclr-qvl)/ab``
    with the M2005 diffusion+ventilation ``eps`` (same core as
    :func:`rain_evaporation_m2005`) on the in-precip slope ``lamr`` and
    intercept ``n0r`` of :func:`mg2_inprecip_rain`, and the grid rate is
    ``min(pre*am_evp_st, 0)``.  ``q_cic``/``q_iic`` are MG2's in-cloud liquid /
    ice (``qcic``/``qiic``); ``q_sat`` is MG2's ``qvl``.
    """
    dum = jnp.where(q_cic + q_iic < _MG2_EVAP_CLOUD_QMIN, 0.0, lcldm)
    # lamr > 0 drops only qric == qsmall exactly, where MG2 itself leaves
    # lamr = n0r = 0 and evaluates 0 * inf (NaN, oracle row 18); every other
    # cell is unchanged.
    active = (precip_frac > dum) & (q_ric >= MG2_QSMALL) & (lamr > 0.0)
    am_evp_st = precip_frac - dum
    one_m_dum = jnp.where(active, 1.0 - dum, 1.0)
    qclr = (q_v - dum * q_sat) / one_m_dum
    lam = jnp.where(active, lamr, 1.0)
    dv, mu, sc = _air_transport_props(T, p, rho)
    arn = config.fall_a_r * safe_pow(
        config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    dqsdt = constants.L_v * q_sat / (constants.R_v * T ** 2)  # latent-ok: MG2 oracle constant (calc_ab, xxlv)
    ab = 1.0 + dqsdt * constants.L_v / constants.c_pd  # latent-ok: MG2 oracle constant (calc_ab, xxlv)
    eps = _mg2_vent_eps(n0r, lam, arn, config.fall_b_r, config.rain_vent_f1,
                        config.rain_vent_f2, rho, dv, mu, sc)
    pre = jnp.minimum(eps * (qclr - q_sat) / ab * am_evp_st, 0.0)
    return jnp.where(active, -pre, 0.0)


def _mg2_vent_eps(n0_vol, lam, a_fall, b_fall, f1, f2, rho, dv, mu, sc):
    """MG2 diffusion + ventilation ``eps`` of a precipitating species
    (``evaporate_sublimate_precip`` / ``bergeron_process_snow``,
    micro_mg_utils.F90:1487-1491, :1505-1509, :1574-1578)::

        eps = 2 pi n0 rho Dv (f1/lam^2 + f2 (a rho/mu)^1/2 Sc^1/3
                              Gamma(5/2+b/2) / lam^(5/2+b/2))

    ``n0_vol`` is the PER-VOLUME intercept (MG2's per-kg ``n0`` times rho);
    ``a_fall`` is the density-corrected fall-speed coefficient (``arn``/``asn``).
    """
    cons = _VENT_CONS_OFFSET + b_fall / 2.0
    return (2.0 * jnp.pi * n0_vol * dv
            * (f1 / (lam * lam)
               + f2 * math.gamma(cons)
               * safe_pow(a_fall * rho / mu, 0.5) * safe_pow(sc, 1.0 / 3.0)
               * safe_pow(lam, -cons)))


# --- CAM6 MG2 two-moment SNOW (micro_mg_utils.F90 / micro_mg2_0.F90,
# cam_cesm2_1_rel_60 a03b84b7).  Snow number N_s is PER-MASS [#/kg] as in MG2;
# rho_snow / lams_min / lams_max / fall_a_s / fall_b_s / snow_vent_f1,f2 /
# snow_aggregation_eii are the MorrisonConfig fields the "mg" flavour resolves
# to MG2's rhosn=250, lam_bnd_snow=1/[2000,10] um, as=11.72, bs=0.41,
# f1s=0.86, f2s=0.28, eii=0.5. ---
MG2_QSIC_MAX = 0.01            # in-precip snow cap [kg/kg] (micro_mg2_0.F90:1363, :2766)
MG2_SNOWMELT_OFFSET_K = 2.0    # snowmelt = tmelt + 2 K: all snow melts above it (:329)
_MG2_SNOW_VT_CAP = 1.2         # ums, uns <= 1.2*rhof [m/s] (:1407-1408, :2189-2190)
_MG2_STOKES_DIV = 9.0          # accrete_cloud_water_snow Stokes number dc0^2 uns rhow lams/(9 mu) (:1133)
_MG2_ECI_KNEE = 0.4            # eci = St^2/(St+0.4)^2, Thompson et al. 2004 (:1134)
# --- CAM6 MG2 Hallett-Mossop splinters (secondary_ice_production, micro_mg_utils.F90:1157-1184) ---
_MG2_SPLINTER_PER_KG = 3.5e8   # splinters per kg of rime (:1172, Cotton et al. 1986)
_MG2_SPLINTER_T_HI = 270.16    # [K] window top (:1171)
_MG2_SPLINTER_T_PEAK = 268.16  # [K] peak (:1171)
_MG2_SPLINTER_T_LO = 265.16    # [K] window bottom (:1173)
_MG2_MI0 = 4.0 / 3.0 * math.pi * 500.0 * (1.0e-6) ** 3  # [kg] 1-um ice embryo, MG2 rhoi 500 (U:126, :150)
_MG2_NSAGG_DENOM = 4.0 * 720.0  # snow_self_aggregation -1108 eii/(4*720 rhosn) (:1072)


def mg2_snow_slope(q_sic, N_sic, config, psd_sharpness=None):
    """MG2 ``size_dist_param_basic(mg_snow_props, qsic, nsic, lams)``
    (micro_mg_utils.F90:481-516): where ``qsic > qsmall``,
    ``lams = (pi rhosn nsic/qsic)^(1/3)``; outside ``[lams_min, lams_max]`` it
    is set to the bound and ``nsic`` re-adjusted to match.  ``lams = 0`` (and
    ``nsic`` untouched) elsewhere.  Returns ``(nsic, lams)``, per-mass number.
    The bound is rounded with smooth_min/smooth_max on slope/bound at
    ``psd_sharpness`` (default MG2_CAP_SHARPNESS; user decision 2026-10-21
    S10.5b, re-applied on main item 4a), the number re-adjusted to it."""
    return _mg2_size_dist_basic(
        q_sic, N_sic, config.rho_snow * jnp.pi, config.lams_min, config.lams_max,
        ratio_sharpness=MG2_CAP_SHARPNESS if psd_sharpness is None else psd_sharpness)


def _mg2_size_dist_basic(q, N, shape, lam_min, lam_max, min_mean_mass=None,
                         log_width=0.0, ratio_sharpness=None):
    """MG2 ``size_dist_param_basic`` for an exponential PSD (eff_dim 3,
    micro_mg_utils.F90:481-516) -> ``(N_adj, lam)``; see :func:`mg2_snow_slope`.
    ``min_mean_mass`` caps ``N <= q/min_mean_mass`` first (the ice props).
    ``log_width > 0`` replaces the hard slope clamp by a LogSumExp clamp of
    ``ln lam`` of that width, with ``N`` re-adjusted to the smooth slope
    everywhere; ``0`` is MG2's hard clamp.  ``ratio_sharpness`` instead rounds
    the clamp as ``lam_max smooth_min(1, lam/lam_max)`` then ``lam_min
    smooth_max(1, ./lam_min)`` at that sharpness, ``N`` re-adjusted likewise."""
    active = q > MG2_QSMALL
    q_safe = jnp.where(active, q, 1.0)
    if min_mean_mass is not None:
        N = jnp.minimum(N, q_safe / min_mean_mass)
    if ratio_sharpness is not None:
        lam_raw = safe_pow(shape * N / q_safe, 1.0 / 3.0)
        one = jnp.ones_like(lam_raw)
        lam = lam_max * smooth_min(one, lam_raw / lam_max, ratio_sharpness)
        lam = lam_min * smooth_max(one, lam / lam_min, ratio_sharpness)
        n_adj = lam ** 3 * q_safe / shape
    elif log_width > 0.0:
        arg = shape * N / q_safe
        tiny = jnp.finfo(jnp.result_type(arg)).tiny
        ln_raw = jnp.log(jnp.where(arg > tiny, arg, tiny)) / 3.0
        # jnp.log only for a traced bound (a trained dcs); a static one keeps
        # the Python log, bitwise as before.
        _ln = (lambda b: math.log(b)) if isinstance(lam_min, float) else jnp.log
        lam = jnp.exp(smooth_clamp(ln_raw, _ln(lam_min), math.log(lam_max),
                                   1.0 / log_width))
        n_adj = lam ** 3 * q_safe / shape
    else:
        lam_raw = safe_pow(shape * N / q_safe, 1.0 / 3.0)
        lam = jnp.clip(lam_raw, lam_min, lam_max)
        n_adj = jnp.where((lam_raw < lam_min) | (lam_raw > lam_max),
                          lam ** 3 * q_safe / shape, N)
    return (jnp.where(active, n_adj, N), jnp.where(active, lam, 0.0))


def mg2_inprecip_snow(q_s, N_s, precip_frac, config, psd_sharpness=None):
    """MG2 in-precip snow ``(qsic, nsic, lams, n0s)`` (micro_mg2_0.F90:
    1358-1375 then :1400-1401): ``qsic = min(qs/precip_frac, 0.01)``,
    ``nsic = max(ns/precip_frac, 0)``, both 0 where ``qsic < qsmall``, then
    :func:`mg2_snow_slope`; ``n0s = nsic*lams``.  Per-mass numbers."""
    q_sic = jnp.minimum(q_s / precip_frac, MG2_QSIC_MAX)
    small = q_sic < MG2_QSMALL
    q_sic = jnp.where(small, 0.0, q_sic)
    N_sic = jnp.maximum(jnp.where(small, 0.0, N_s / precip_frac), 0.0)
    N_sic, lams = mg2_snow_slope(q_sic, N_sic, config, psd_sharpness)
    return q_sic, N_sic, lams, N_sic * lams


def mg2_snow_rhof_asn(rho, config):
    """MG2 ``rhof = (rhosu/rho)^0.54`` and ``asn = as*rhof``
    (micro_mg2_0.F90:940-944)."""
    rhof = safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    return rhof, config.fall_a_s * rhof


def mg2_snow_fall_speeds(lams, rho, config):
    """MG2 snow mass- and number-weighted fall speeds ``(ums, uns)``
    (micro_mg2_0.F90:1403-1413 provisional, :2186-2199 final)::

        ums = min(asn Gamma(4+bs)/(6 lams^bs), 1.2 rhof)
        uns = min(asn Gamma(1+bs)/lams^bs,     1.2 rhof)

    where ``lams > 0``, else 0."""
    rhof, asn = mg2_snow_rhof_asn(rho, config)
    on = lams > 0.0
    lam_b = safe_pow(jnp.where(on, lams, 1.0), config.fall_b_s)
    cap = _MG2_SNOW_VT_CAP * rhof
    ums = jnp.minimum(asn * math.gamma(4.0 + config.fall_b_s) / (6.0 * lam_b), cap)
    uns = jnp.minimum(asn * math.gamma(1.0 + config.fall_b_s) / lam_b, cap)
    return jnp.where(on, ums, 0.0), jnp.where(on, uns, 0.0)


def mg2_snow_self_aggregation(q_sic, N_sic, T, rho, config):
    """MG2 ``snow_self_aggregation`` (micro_mg_utils.F90:1052-1078), in-precip
    per-mass number rate [#/kg/s], NEGATIVE (a sink)::

        nsagg = -1108 eii/(4*720 rhosn) asn qsic nsic rho
                ((qsic/nsic)/(rhosn pi))^((bs-1)/3)

    where ``qsic >= qsmall`` and ``T <= T_freeze``, else 0."""
    active = (q_sic >= MG2_QSMALL) & (T <= constants.T_freeze)
    q = jnp.where(active, q_sic, 1.0)
    n = jnp.where(active, N_sic, 1.0)
    _, asn = mg2_snow_rhof_asn(rho, config)
    rs = config.rho_snow
    nsagg = (-_NSAGG_CONS15_PREFACTOR * config.snow_aggregation_eii
             / (_MG2_NSAGG_DENOM * rs) * asn * q * n * rho
             * ((q / n) * (1.0 / (rs * jnp.pi))) ** ((config.fall_b_s - 1.0) / 3.0))
    return jnp.where(active, nsagg, 0.0)


def mg2_accrete_cloud_water_snow(T, rho, mu, uns, q_cic, N_cic, q_sic, pgam,
                                 lamc, lams, n0s, config, cold_width_K):
    """MG2 ``accrete_cloud_water_snow`` (micro_mg_utils.F90:1087-1151): riming
    of cloud droplets by snow, in-cloud ``(psacws [kg/kg/s], npsacws)``::

        dc0 = (pgam+1)/lamc;  St = dc0^2 uns rhow lams/(9 mu)
        eci = clip(St^2/(St+0.4)^2, 0, 1)
        rate = pi/4 asn rho n0s eci Gamma(bs+3)/lams^(bs+3)
        psacws = rate qcic,  npsacws = rate ncic

    where ``qsic >= qsmall`` and ``qcic >= qsmall``.  MG2's hard ``T <= T_freeze``
    is the C2 step, 0 at 0 C and full by -``cold_width_K`` (user decision
    2026-10-21 S10.5 1a, re-applied on main item 2a).
    ``uns`` is the PROVISIONAL in-precip snow number fall speed, ``pgam/lamc``
    from :func:`mg2_size_dist_liq`; ``N_cic`` in whatever units the caller wants
    ``npsacws`` in."""
    # lamc > 0 (oracle row 31): at qcic == qsmall exactly MG2's
    # size_dist_param_liq leaves pgam = -100, lamc = 0 (it tests qcic >
    # qsmall) while this gate is qcic >= qsmall, so dc0 = -99/0 made the rate
    # NaN forward and backward here.  The compiled oracle returns 0 there
    # (eci = max(NaN, 0) = 0 in gfortran), as this now does: lamc is 0 only
    # at that point, so nothing else changes.
    active = ((q_sic >= MG2_QSMALL) & (q_cic >= MG2_QSMALL)
              & (lamc > 0.0))
    cold = smoothstep_quintic((constants.T_freeze - T) / cold_width_K)
    lc = jnp.where(active, lamc, 1.0)
    ls = jnp.where(active, lams, 1.0)
    _, asn = mg2_snow_rhof_asn(rho, config)
    dc0 = (pgam + 1.0) / lc
    st = dc0 * dc0 * uns * constants.rho_water * ls / (_MG2_STOKES_DIV * mu)
    eci = jnp.clip(st * st / ((st + _MG2_ECI_KNEE) * (st + _MG2_ECI_KNEE)),
                   0.0, 1.0)
    bs = config.fall_b_s
    rate = (jnp.pi / 4.0 * asn * rho * n0s * eci * math.gamma(bs + 3.0)
            / ls ** (bs + 3.0))
    return (jnp.where(active, cold * rate * q_cic, 0.0),
            jnp.where(active, cold * rate * N_cic, 0.0))


def mg2_secondary_ice_production(psacws, T, sharpness):
    """CAM6 MG2 ``secondary_ice_production`` (Hallett-Mossop, micro_mg_utils.F90:
    1157-1184): ``(psacws_left, msacwi, nsacwi)`` with nsacwi [1/kg/s] =
    3.5e8 * tri(T) * psacws and msacwi = min(nsacwi*mi0, psacws).  The
    triangle tri (0 at -8 and -3 C, 1 at -5 C) has its three corners rounded,
    smooth_max(0, smooth_min((270.16-T)/2, (T-265.16)/3)) at ``sharpness``
    (user decision 2026-10-21, 2b).  msacwi/psacws <= 7.3e-7, so the min never
    binds; kept as MG2 wrote it.
    """
    tri = smooth_max(
        jnp.zeros_like(T),
        smooth_min((_MG2_SPLINTER_T_HI - T) / (_MG2_SPLINTER_T_HI - _MG2_SPLINTER_T_PEAK),
                   (T - _MG2_SPLINTER_T_LO) / (_MG2_SPLINTER_T_PEAK - _MG2_SPLINTER_T_LO),
                   sharpness),
        sharpness)
    nsacwi = _MG2_SPLINTER_PER_KG * tri * psacws
    msacwi = jnp.minimum(nsacwi * _MG2_MI0, psacws)
    return psacws - msacwi, msacwi, nsacwi


def mg2_snow_vent_eps(n0s, lams, T, p, rho, config):
    """MG2 snow diffusion+ventilation ``eps`` (:func:`_mg2_vent_eps` with the
    snow PSD, per-mass ``n0s``) and the air viscosity ``mu``, the pair the snow
    sublimation / Bergeron kernels and :func:`mg2_accrete_cloud_water_snow`
    consume."""
    dv, mu, sc = _air_transport_props(T, p, rho)
    _, asn = mg2_snow_rhof_asn(rho, config)
    lam = jnp.where(lams > 0.0, lams, 1.0)
    eps = _mg2_vent_eps(n0s * rho, lam, asn, config.fall_b_s,
                        config.snow_vent_f1, config.snow_vent_f2, rho, dv, mu, sc)
    return jnp.where(lams > 0.0, eps, 0.0), mu


def _mg2_calc_ab_ice(T, q_sat_i):
    """MG2 ``calc_ab(t, qvi, xxls)`` (micro_mg_utils.F90:361-373)."""
    dqsdt = constants.L_s * q_sat_i / (constants.R_v * T ** 2)  # latent-ok: MG2 oracle constant (xxls)
    return 1.0 + dqsdt * constants.L_s / constants.c_pd  # latent-ok: MG2 oracle constant (xxls)


def mg2_bergeron_snow(T, q_cic, q_sic, q_sat, q_sat_i, eps_s, cold_width_K):
    """MG2 ``bergeron_process_snow`` (micro_mg_utils.F90:1541-1586), in-cloud
    liquid -> snow rate ``bergs = eps (qvl - qvi)/ab`` [kg/kg/s] where
    ``qsic >= qsmall`` and ``qcic >= qsmall`` (times micro_mg_berg_eff_factor
    = 1, the CAM6 namelist value).  MG2's hard ``T < T_freeze`` is the C2 step,
    0 at 0 C and full by -``cold_width_K`` (user decision 2026-10-21 S10.6 5a,
    item 6a)."""
    active = (q_sic >= MG2_QSMALL) & (q_cic >= MG2_QSMALL)
    cold = smoothstep_quintic((constants.T_freeze - T) / cold_width_K)
    bergs = eps_s * (q_sat - q_sat_i) / _mg2_calc_ab_ice(T, q_sat_i)
    return jnp.where(active, cold * bergs, 0.0)


def mg2_ice_deposition_sublimation(q_v, q_i, N_i, icldm, T, p, rho, qvl, qvi, rho_ice,
                                   dcs, log_width, cold_width_K, *, presence_top):
    """CAM6 MG2 ``ice_deposition_sublimation`` (micro_mg_utils.F90:604-682):
    ``(vap_dep, ice_sublim, berg)``, grid mean [kg/kg/s].  On in-cloud
    ``qi/icldm``, ``ni/icldm`` (not capped, as there) and MG2's ice slope
    (:func:`mg2_ice_slope`, its rounding ``log_width``):
    ``epsi = 2 pi n0i rho Dv / lami^2``, ``ab = calc_ab(T, qvi, L_s)``,
    ``vap = epsi/ab (q - qvi) icldm``; deposition ``max(vap, 0)`` only below
    0 C, sublimation ``min(vap, 0)`` at any temperature, ``berg = max(epsi/ab
    (qvl - qvi), 0)`` below 0 C, not times icldm (as MG2).  MG2's hard
    ``t < tmelt`` is the C2 step, 0 at 0 C and full by -``cold_width_K`` (user
    decision 2026-10-21 S10.6 5a).  Gate qi >= qsmall through
    :func:`mg2_presence`; berg_eff_factor = 1."""
    has = q_i >= MG2_QSMALL
    on = mg2_presence(q_i, presence_top)
    ni_adj, lami = mg2_ice_slope(q_i / icldm, N_i / icldm, rho_ice, dcs, log_width)
    lam = jnp.where(has, lami, 1.0)
    dv, _, _ = _air_transport_props(T, p, rho)
    epsi = 2.0 * jnp.pi * ni_adj * lam * rho * dv / (lam * lam)
    ab = _mg2_calc_ab_ice(T, qvi)
    vap = epsi / ab * (q_v - qvi) * icldm
    cold = smoothstep_quintic((constants.T_freeze - T) / cold_width_K)
    vap_dep = jnp.where(has, on * cold * jnp.maximum(vap, 0.0), 0.0)
    ice_sublim = jnp.where(has, on * jnp.minimum(vap, 0.0), 0.0)
    berg = jnp.where(has, on * cold * jnp.maximum(epsi / ab * (qvl - qvi), 0.0), 0.0)
    return vap_dep, ice_sublim, berg


def mg2_snow_sublimation(q_v, q_sat, q_sat_i, T, eps_s, q_sic, lcldm,
                         precip_frac, q_cic, q_iic):
    """MG2 snow sublimation ``prds`` (``evaporate_sublimate_precip``,
    micro_mg_utils.F90:1460-1530), IN-PRECIP, <= 0 [kg/kg/s] (grid value =
    ``prds*precip_frac``).  Only in the precipitating clear air: ``dum = lcldm``
    unless ``qcic+qiic < 1e-6``; where ``precip_frac > dum`` and
    ``qsic >= qsmall``: ``qclr = (q - dum qvl)/(1 - dum)``,
    ``prds = min(eps (qclr - qvi)/ab (precip_frac - dum), 0)/precip_frac``."""
    dum = jnp.where(q_cic + q_iic < _MG2_EVAP_CLOUD_QMIN, 0.0, lcldm)
    active = (precip_frac > dum) & (q_sic >= MG2_QSMALL)
    one_m_dum = jnp.where(active, 1.0 - dum, 1.0)
    qclr = (q_v - dum * q_sat) / one_m_dum
    prds = eps_s * (qclr - q_sat_i) / _mg2_calc_ab_ice(T, q_sat_i)
    prds = jnp.minimum(prds * (precip_frac - dum), 0.0) / precip_frac
    return jnp.where(active, prds, 0.0)


def mg2_precip_sublimation_limiter(q_v, T, p, pre, prds, ice_sublim, vap_dep,
                                   mnuccd, dt, *, cap_sharpness):
    """MG2 limit on rain evaporation + snow and ice sublimation
    (micro_mg2_0.F90:1805-1850), all GRID rates [kg/kg/s] with MG2's signs
    (``pre``, ``prds``, ``ice_sublim`` <= 0; ``vap_dep``, ``mnuccd`` >= 0).

    Where they sum below -1e-20 and the end-of-step humidity
    ``q - (ice_sublim + vap_dep + mnuccd + pre + prds) dt`` would exceed liquid
    saturation at the end-of-step temperature, the three are replaced by their
    share (``dum1 = pre/sum``, ``dum2 = prds/sum``, ``1 - dum1 - dum2``) of the
    one-step adjustment to saturation from the state after deposition and
    nucleation: rain to LIQUID saturation, snow and ice to ICE saturation,
    each ``min((qtmp - qsat)/(1 + L^2 qsat/(c_p R_v ttmp^2)), 0)/dt``, with
    MG2's own ``qsat_water``/``qsat_ice`` (CAM Goff-Gratch; ``cap_sharpness``
    is their q = 1 cap, ``math.inf`` = CAM's hard cap).
    Returns ``(pre, prds, ice_sublim)``."""
    tot = pre + prds + ice_sublim
    on = tot < -1.0e-20
    cp = constants.c_pd
    qtmp = q_v - (ice_sublim + vap_dep + mnuccd + pre + prds) * dt
    ttmp = T + (pre * constants.L_v  # latent-ok: MG2 oracle constant (xxlv)
                + (prds + vap_dep + ice_sublim + mnuccd)
                * constants.L_s) * dt / cp  # latent-ok: MG2 oracle constant (xxls)
    on = on & (qtmp > saturation_specific_humidity_goff_gratch(ttmp, p, cap_sharpness=cap_sharpness))
    den = jnp.where(on, tot, -1.0)
    dum1 = pre / den
    dum2 = prds / den
    qtmp = q_v - (vap_dep + mnuccd) * dt
    ttmp = T + ((vap_dep + mnuccd) * constants.L_s) * dt / cp  # latent-ok: MG2 oracle constant (xxls)
    qvn = saturation_specific_humidity_goff_gratch(ttmp, p, cap_sharpness=cap_sharpness)
    dum = jnp.minimum((qtmp - qvn) / (1.0 + constants.L_v ** 2 * qvn  # latent-ok: MG2 oracle constant (xxlv_squared)
                                      / (cp * constants.R_v * ttmp ** 2)), 0.0)
    pre_n = dum * dum1 / dt
    qvn = saturation_specific_humidity_ice_goff_gratch(ttmp, p, cap_sharpness=cap_sharpness)
    dum = jnp.minimum((qtmp - qvn) / (1.0 + constants.L_s ** 2 * qvn  # latent-ok: MG2 oracle constant (xxls_squared)
                                      / (cp * constants.R_v * ttmp ** 2)), 0.0)
    prds_n = dum * dum2 / dt
    # ice share as ice_sublim/sum: MG2's 1 - dum1 - dum2 rounds to ~1e-16 with no ice
    ice_n = dum * (ice_sublim / den) / dt
    return (jnp.where(on, pre_n, pre), jnp.where(on, prds_n, prds),
            jnp.where(on, ice_n, ice_sublim))


def mg2_instant_snow_melt(T, q_s, N_s, width_K, cap_sharpness=MG2_CAP_SHARPNESS):
    """MG2 instantaneous snow melting above ``snowmelt = T_freeze + 2 K``
    (micro_mg2_0.F90:1147-1169 before the processes, :2538-2566 after
    sedimentation): MG2 melts the fraction ``min(1, (T - snowmelt) c_p/(L_f
    q_s))`` of the snow where ``T > snowmelt`` and ``q_s > 0`` (heat-limited so
    the melt cannot cool the air below snowmelt).  Here the switch-on is the C2
    step, 0 at snowmelt and full ``width_K`` above, and the min is rounded with
    smooth_min at ``cap_sharpness`` (user decisions 2026-10-20 2a/2b, re-applied
    on main 2026-10-21 item 1a; see :func:`mg2_phase_change_fraction`).
    Returns the melted ``(mass, number) = (f q_s, f N_s)`` [kg/kg, #/kg]."""
    excess = T - (constants.T_freeze + MG2_SNOWMELT_OFFSET_K)
    frac = mg2_phase_change_fraction(excess, q_s, width_K, cap_sharpness)
    return frac * jnp.maximum(q_s, 0.0), frac * N_s

# --- Seifert & Beheng (2001) warm-rain UNIVERSAL FUNCTIONS ---
# Faithful transcription of the gSAM M2005 IRAIN=1 path
# (module_mp_graupel.f90:1835-1844 autoconversion, :1960-1962 accretion).
# These are the PUBLISHED SB2001 closed forms (phi_au, phi_ac), structurally
# distinct from the simplified smooth proxies ``autoconversion_sb`` (q_c^2 *
# mean-mass sigmoid) and ``accretion`` (bilinear k*rho*q_c*q_r) above. All
# coefficients are fixed published SB2001 / M2005 constants, not tunable here.
_SB2001_PHI_AU_PREFACTOR = 600.0     # phi_au(tau) leading coeff (gSAM :1836)
_SB2001_PHI_AU_TAU_EXP = 0.68        # phi_au tau exponent (gSAM :1836)
_SB2001_KCC = 9.44e9                 # autoconv kernel k_cc [cm^3 g^-2 s^-1] (gSAM :1838)
_SB2001_X_STAR_CGS = 2.6e-7          # separation mass x_* [g] (gSAM :1838)
_SB2001_X_STAR_KG = 2.6e-10          # separation mass x_* [kg] (number closure au/x_*)
# k_cc/(20·x_*) — the SB2001 mass-rate prefactor (gSAM :1838). The 20 relates
# k_cc to the mass rate in SB2001; folded here to avoid an inline body literal.
_SB2001_AUTOCONV_MASS_COEFF = _SB2001_KCC / (20.0 * _SB2001_X_STAR_CGS)
# Cloud-droplet gamma shape nu for the (nu+2)(nu+4)/(nu+1)^2 factor (gSAM :1839)
# is DIAGNOSED per level, as gSAM's default (dofix_pgam=.false.) does -- see
# :func:`m2005_cloud_droplet_psd` and the _M2005_* block below.
_SB2001_KCR = 5.78e3                 # accretion kernel k_cr [cm^3 g^-1 s^-1] (gSAM :1962)
# 5.78e3·rho/1000 = 5.78·rho (gSAM :1962); the /1000 is the CGS mass-density
# conversion folded into the coefficient.
_SB2001_ACCRETION_COEFF = _SB2001_KCR / 1000.0
_SB2001_PHI_AC_KTAU = 5.0e-4         # phi_ac(tau) = (tau/(tau+k_tau))^4 knee (gSAM :1961)
# SB2001 rate gates (gSAM M2005 module_mp_graupel.f90:1806/2843 PRC needs
# QC3D >= 1e-6; :1944/3266 PRA needs QR3D >= 1e-8 and QC3D >= 1e-8).
_SB2001_PRC_QC_MIN = 1.0e-6          # [kg/kg]
_SB2001_PRA_Q_MIN = 1.0e-8           # [kg/kg]
# --- gSAM M2005 cloud-droplet gamma PSD (module_mp_graupel.f90:1664-1708,
# :2600-2640; the dofix_pgam=.false. default, micro_params.f90:40) ---
# PGAM = 1/(A*N_c[#/cm^3] + B)^2 - 1 (Martin et al. 1994), clamped to
# [PGAM_MIN, PGAM_MAX]; nu interpolated linearly in the dnu table at PGAM;
# droplet slope LAMC kept in (PGAM+1)/[60 um, 1 um] by resetting N_c.
_M2005_QSMALL = 1.0e-14              # QSMALL (GRAUPEL_INIT :477) [kg/kg]
_M2005_PGAM_A = 0.0005714            # Martin et al. (1994) fit slope (:1676)
_M2005_PGAM_B = 0.2714               # Martin et al. (1994) fit offset (:1676)
_M2005_PGAM_MIN = 2.0                # (:1678)
_M2005_PGAM_MAX = 10.0               # (:1679)
# dnu(1:16) (GRAUPEL_INIT :610-625); nu(PGAM) = dnu(i) + (dnu(i+1)-dnu(i))*(PGAM-i),
# i = int(PGAM) (1-based).
_M2005_DNU = (-0.557, -0.557, -0.430, -0.307, -0.186, -0.067, 0.050, 0.167,
              0.282, 0.397, 0.512, 0.626, 0.739, 0.853, 0.966, 0.966)
_M2005_DROP_DIAM_MAX = 60.0e-6       # LAMMIN = (PGAM+1)/60 um (:1696) [m]
_M2005_DROP_DIAM_MIN = 1.0e-6        # LAMMAX = (PGAM+1)/1 um (:1697) [m]
# Unit conversions for the CGS SB2001 mass rate (gSAM works in g/cm^3, #/cm^3):
_LWC_KGM3_TO_GCM3 = 1.0e-3           # rho·q_c [kg/m^3] -> LWC [g/cm^3] (gSAM ·/1000)
_NC_PERM3_TO_PERCM3 = 1.0e-6         # N_c [1/m^3] -> [1/cm^3] (gSAM ·/1e6; per-VOLUME)
_MASSRATE_GCM3_TO_MIXR = 1.0e3       # g/cm^3/s -> kg/m^3/s (then /rho -> kg/kg/s)

# --- shared PSD / fall-speed / ventilation structural constants (SAM M2005) ---
_RHO_FLOOR = 0.1                 # air-density floor in PSD/fall-speed divisions [kg/m^3]
_FALL_RHO_EXPONENT = 0.54        # (rho_su/rho)^0.54 fall-speed density correction
_VENT_CONS_OFFSET = 2.5          # 5/2 in ventilation gamma argument 5/2 + b/2
_BIGG_MNUCCR_PREFACTOR = 20.0    # Bigg freezing mass prefactor (20 pi^2 rho_w)
# Bigg rain freezing only below -4 C: CAM6 MG2 heterogeneous_rain_freezing
# ``t < 269.15`` (micro_mg_utils.F90:1273), SAM M2005 ``T3D < 269.15``
# (module_mp_graupel.F90:3296), WRF M2005 likewise.
_BIGG_T_MAX_K = 269.15
_UMR_FALL_CAP = 9.1              # rain mass-weighted fall-speed cap [m/s]
_UMG_FALL_CAP = 20.0             # graupel mass-weighted fall-speed cap [m/s]
_VDIFF_C1, _VDIFF_C2, _VDIFF_C3 = 1.2, 0.95, 0.08  # Wisner two-PSD VDIFF coeffs
_PRACG_BRACKET_C = 5.0           # PRACG collection-integral leading bracket coeff
_NSAGG_CONS15_PREFACTOR = 1108.0  # snow self-aggregation CONS15 prefactor
_NSAGG_GAMMA_DENOM = 720.0       # snow self-aggregation gamma denominator (4*720)
_RAIN_EVAP_VENT_EXP = 0.525      # Marshall-Palmer rain-evaporation ventilation exponent



def autoconversion_kk2000(q_c, N_c_eff, rho, dt, fact=1.0):
    """Khairoutdinov–Kogan (2000) warm-rain autoconversion — the SAM
    M2005 DEFAULT (``IRAIN=0``, ``module_mp_graupel.f90:1813``):

        PRC = 1350 · q_c^2.47 · (N_c[#/cm³])^−1.79      [kg/kg/s]

    ``N_c`` in #/cm³ = ``N_c_eff``[#/m³] / 1e6 for this module's
    per-volume convention (:func:`effective_Nc`).  Unlike Seifert–Beheng
    there is no mean-mass sigmoid onset — the ``q_c^2.47`` power gives a
    smooth, sharply-increasing onset with cloud water.  Returns the
    cloud-water sink + rain-number source (SAM ``NPRC1 = PRC/CONS29``),
    per-volume, matching :func:`autoconversion_sb`'s signature.

    Parameters
    ----------
    q_c : array
        Cloud water mixing ratio [kg/kg].
    N_c_eff : array
        Effective cloud droplet number [1/m³] (per-volume).
    rho : array
        Air density [kg/m³].
    dt : float
        Time step [s] (caps the rain-number source at ``N_c/dt``).
    fact : float
        Multiplier on PRC (``MorrisonConfig.autocon_fact``, CAM6 MG2-style),
        applied before both number caps. 1.0 = unscaled, bit-identical.

    Returns
    -------
    dq_c_au : array
        Cloud-water autoconversion rate [kg/kg/s].
    dN_r_au : array
        Rain-number formation rate [1/(m³·s)] (per-volume).
    x_c : array
        Mean cloud droplet mass [kg].
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    n_c_cm3 = jnp.clip(N_c_eff, 1.0) / 1.0e6        # #/cm³
    # ``fact`` (MorrisonConfig.autocon_fact) scales PRC before both caps,
    # so the rain-number source and cloud-number sink follow it.
    prc = (
        (fact * _KK2000_AUTOCONV_PREFACTOR)
        * safe_pow(q_c_pos, _KK2000_AUTOCONV_QC_EXPONENT)
        * safe_pow(n_c_cm3, _KK2000_AUTOCONV_NC_EXPONENT)
    )
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    # SAM's two-level cap (module_mp_graupel.f90:1823-1828), in per-volume:
    #   NPRC  = PRC·N_c/q_c   (cloud-number autoconv sink), ≤ N_c/dt
    #   NPRC1 = PRC·rho/CONS29 (rain-number source),       ≤ NPRC
    # NPRC1 ≤ NPRC guarantees rain gains no more drops than the cloud
    # loses (NPRC1 uses the 25 µm CONS29 mass, which need not satisfy that
    # on its own). q_c floored only inside the ratio (prc=0 at q_c=0).
    nprc = jnp.minimum(
        prc * N_c_eff / jnp.clip(q_c_pos, 1.0e-20),
        jnp.clip(N_c_eff, 0.0) / jnp.clip(dt, 1.0),
    )
    dN_r_au = jnp.minimum(prc * rho / _KK2000_CONS29, nprc)
    return prc, dN_r_au, x_c


def autoconversion_kk2000_cam6(q_c, N_c_eff, rho, relvar, fact=1.0, nc_floor=True,
                               size_reset=True):
    """CAM6 MG2 KK2000 autoconversion (``kk2000_liq_autoconversion``,
    micro_mg_utils.F90:689-736, CAM6 default ``microp_uniform=.false.``)::

        PRC = var_coef(relvar, 2.47) · 13.5 · q_c^2.47 · (N_c[#/cm³])^−1.1

    ``N_c[#/cm³]`` = ``N_c_eff``[#/m³]·1e-6, identical to CAM6's
    ``ncic[#/kg]·1e-6·rho``.  ``var_coef(r, a) = Γ(r+a)/(Γ(r)·r^a)`` is the
    sub-grid cloud-water variance enhancement, verbatim from
    micro_mg_utils.F90:574-583 + :333-339.  CAM6's ``relvar`` is the
    INVERSE relative variance (gamma shape ν = mean²/variance,
    clubb_intr.F90:2425 ``rcm**2/qclvar``), so large relvar = uniform cloud
    (factor → 1) and small relvar = strong enhancement.  Ported guards: ``q_c >= 1e-8``
    gate (``icsmall``), the 5e-3 kg/kg in-cloud cap, and MG2's droplet-number
    reset to the [2, 50] um mean-size window (:func:`mg2_size_bounded_nc`,
    size_dist_param_liq, applied to ``N_c_eff`` before the law and before
    ``x_c``).  Rain-number source =
    ``PRC·rho / m(25 µm)`` (CAM6 ``nprc``, per-volume here); the cloud-number
    sink is the caller's ``-PRC·rho/x_c`` = CAM6 ``nprc1`` (``x_c`` from the
    capped water).  ``N_c_eff`` must be the IN-CLOUD number, as CAM6's
    ``ncic``: true for the specified-Nc mode (production,
    ``predict_Nc=False``); with prognostic Nc it is the grid mean, the same
    known limitation as the other laws here.  ``relvar`` is clipped to
    [0.001, 10] as CAM6 does.  N_c is floored at 1 m^-3 (CAM6 has no floor;
    it would give inf at ncic=0) unless ``nc_floor=False`` (the
    prognostic_droplet_number path, exact CAM): there the reset number is used
    as is wherever the law is evaluated (``q_c >= icsmall``, where the reset
    makes it positive) and replaced by 1 only in the masked branch.
    ``size_reset=False``: ``N_c_eff`` already carries MG2's one reset (the
    caller shares it with every liquid sink, micro_mg2_0.F90:1296).  ``fact`` =
    ``MorrisonConfig.autocon_fact``.  Returns ``(dq_c_au, dN_r_au, x_c)`` like
    :func:`autoconversion_kk2000`.
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    q_c_ic = jnp.minimum(q_c_pos, KK2000_CAM6_QC_MAX)
    # MG2 resets ncic to the [2, 50] um mean-size window before the law
    # (size_dist_param_liq, called at micro_mg2_0.F90:1296 before :1306).
    if size_reset:
        N_c_eff = mg2_size_bounded_nc(q_c_ic, N_c_eff, rho)
    if nc_floor:
        N_c_eff = jnp.clip(N_c_eff, 1.0)
    else:
        # masked-branch value only: prc and x_c are unused below icsmall
        N_c_eff = jnp.where(q_c_ic >= _KK2000_CAM6_QC_MIN, N_c_eff, 1.0)
    n_c_cm3 = N_c_eff / 1.0e6        # #/cm³
    a = _KK2000_AUTOCONV_QC_EXPONENT
    var_coef = mg2_var_coef(relvar, a, q_c_pos.dtype)
    prc = jnp.where(
        q_c_ic >= _KK2000_CAM6_QC_MIN,
        (fact * _KK2000_CAM6_PREFACTOR) * var_coef
        * safe_pow(q_c_ic, a) * safe_pow(n_c_cm3, _KK2000_CAM6_NC_EXPONENT),
        0.0,
    )
    # Mean droplet mass from the CAPPED in-cloud water, so the caller's
    # -PRC*rho/x_c equals CAM6 nprc1 = prc*ncic/qcic (Morrison then limits
    # the cloud-number sink to N_c/dt, as CAM6 does at micro_mg2_0:1630).
    x_c = q_c_ic * rho / N_c_eff
    return prc, prc * rho / _KK2000_CONS29, x_c


def accretion_kk2000(q_c, q_r, fact=1.0):
    """Khairoutdinov–Kogan (2000) warm-rain accretion — SAM M2005
    (``module_mp_graupel.f90:1952``; identical to gSAM P3
    ``module_mp_p3.f90:3615``, iparam=3):

        PRA = 67 · (q_c · q_r)^1.15      [kg/kg/s]

    times ``fact`` (``MorrisonConfig.accre_enhan_fact``, MG2 ``accre_enhan``;
    1.0 = unscaled, bit-identical).

    A mixing-ratio rate (no ``rho`` factor, unlike :func:`accretion`).

    SCOPE: this is the single-column, fractions-OFF MASS-rate reduction of the
    Fortran (``iSCF=iSPF=1``, ``dum2=SPF-SPF_clr=1``) — the algebraic iparam=3
    branch ABOVE its enclosing ``qc>=qsmall`` AND ``qr>=qsmall`` gate
    (``qsmall=1e-14``).  DEPARTURES vs the full Fortran: (a) this helper omits
    the ``qsmall`` gate, so for tiny-but-positive inputs below 1e-14 it returns a
    positive rate where the Fortran returns exactly 0; (b) the number tie
    ``ncacc = qcacc·nc/qc`` is handled separately by the caller, not here.
    ``safe_pow`` clips the base to nonnegative and gives a finite AD result for
    the fractional power (its role is the negative-input extension — at
    ``q_c·q_r = 0`` the ``^1.15`` branch already has a finite ZERO slope, since
    ``d/dx x^1.15 = 1.15·x^0.15 -> 0``, so this is NOT replacing a singular slope).
    """
    dum = jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0)
    return (fact * _KK2000_ACCRETION_PREFACTOR) * safe_pow(
        dum, _KK2000_ACCRETION_EXPONENT)


def m2005_cloud_droplet_psd(q_c, N_c, rho):
    """gSAM M2005 cloud-droplet gamma PSD: ``(pgam, nu, N_c_adj)``.

    Port of module_mp_graupel.f90:1664-1708 (warm branch; identical cold
    branch :2600-2640) at gSAM's default ``dofix_pgam=.false.``.  Where
    ``q_c >= QSMALL``::

        PGAM = 1/(0.0005714*N_c[#/cm^3] + 0.2714)^2 - 1,  clamped to [2, 10]
        nu   = dnu(i) + (dnu(i+1) - dnu(i))*(PGAM - i),    i = int(PGAM)
        LAMC = (pi/6 rho_w N_c/rho (PGAM+1)(PGAM+2)(PGAM+3) / q_c)^(1/3)

    and if ``LAMC`` leaves ``(PGAM+1)/[60 um, 1 um]`` it is set to the bound
    and ``N_c`` reset to the number that bound carries (``N_c`` is first
    floored at 0, :1632/:2541).  ``N_c`` and
    ``N_c_adj`` are per-VOLUME [1/m^3] (the oracle's per-kg NC3D times rho);
    elsewhere ``N_c_adj = N_c`` and ``pgam``/``nu`` are not used by any rate.
    ``GAMMA(PGAM+4)/GAMMA(PGAM+1)`` is written as the exact product
    ``(PGAM+1)(PGAM+2)(PGAM+3)``.
    """
    N_c = jnp.maximum(0.0, N_c)                       # NC3D = MAX(0., NC3D) (:1632)
    active = q_c >= _M2005_QSMALL
    qc_s = jnp.where(active, q_c, 1.0)
    pgam = _M2005_PGAM_A * (N_c * _NC_PERM3_TO_PERCM3) + _M2005_PGAM_B
    pgam = 1.0 / pgam ** 2 - 1.0
    pgam = jnp.minimum(jnp.maximum(pgam, _M2005_PGAM_MIN), _M2005_PGAM_MAX)
    dumii = jnp.floor(pgam)
    idx = dumii.astype(jnp.int32) - 1                 # Fortran dnu(dumii), 1-based
    dnu = jnp.asarray(_M2005_DNU, dtype=pgam.dtype)
    nu = dnu[idx] + (dnu[idx + 1] - dnu[idx]) * (pgam - dumii)
    lamc, lammin, lammax, gamma_ratio, cons26 = _m2005_cloud_slope(
        qc_s, N_c, rho, pgam)
    lam_b = jnp.where(lamc < lammin, lammin, lammax)
    n_bound = lam_b ** 3 * qc_s / gamma_ratio / cons26 * rho
    clamped = active & ((lamc < lammin) | (lamc > lammax))
    return pgam, nu, jnp.where(clamped, n_bound, N_c)


def _m2005_cloud_slope(qc_s, N_c, rho, pgam):
    """Unclamped droplet slope ``LAMC`` and its bounds (:1690-1697)."""
    gamma_ratio = (pgam + 1.0) * (pgam + 2.0) * (pgam + 3.0)
    cons26 = jnp.pi / 6.0 * constants.rho_water
    lamc = safe_pow(cons26 * (N_c / rho) * gamma_ratio / qc_s, 1.0 / 3.0)
    lammin = (pgam + 1.0) / _M2005_DROP_DIAM_MAX
    lammax = (pgam + 1.0) / _M2005_DROP_DIAM_MIN
    return lamc, lammin, lammax, gamma_ratio, cons26


def autoconversion_sb2001(q_c, q_r, N_c_eff, rho, *, ramp_top):
    """Seifert & Beheng (2001) autoconversion as gSAM M2005 runs it
    (IRAIN=1, module_mp_graupel.f90:1830-1844 / :2868-2880)::

        dum    = 1 − q_c/(q_c+q_r)                       (rain-water fraction)
        phi_au = 600·dum^0.68·(1 − dum^0.68)^3
        PRC    = 9.44e9/(20·2.6e-7)·(nu+2)(nu+4)/(nu+1)^2
                 ·(rho·q_c/1000)^4 / (rho·NC3D/1e6)^2
                 ·(1 + phi_au/(1−dum)^2)·1000/rho          [kg/kg/s]
        NPRC1  = PRC/2.6e-7·1000                           [1/kg/s]

    computed only where ``q_c >= 1e-6`` (zero elsewhere), with ``nu`` and the
    slope-clamped droplet number from :func:`m2005_cloud_droplet_psd` (gSAM's
    default diagnosed shape, ``dofix_pgam=.false.``).  ``N_c_eff`` is per-VOLUME
    [1/m^3] (``rho·NC3D``); the returned rain-number source is per-volume
    (``NPRC1·rho``).

    Returns ``(dq_c_au, dN_r_au, x_c)`` matching :func:`autoconversion_sb`'s
    signature; ``x_c = q_c·rho/N_c_eff`` is the mean droplet mass Morrison's
    generic cloud-number sink uses under ``predict_Nc=True``.  NOT ported:
    gSAM's SB2001 cloud-number sink ``NPRC = 2·PRC/x_*`` (only reached with
    ``predict_Nc=True``, whose droplet activation is not ported either).

    RULE 5 (#2232): gSAM's ``q_c >= 1e-6`` step is replaced by the C2 step
    :func:`smoothstep_quintic`, exactly 0 at ``q_c = 1e-6`` and 1 from
    ``ramp_top`` [kg/kg] up (``MorrisonConfig`` / ``SeifertBehengConfig``
    ``.sb2001_autoconv_ramp_top``, default 2e-6, codex+GLM 2026-10-29);
    ``ramp_top <= 1e-6`` is gSAM's hard step.  The rate stays exactly 0 below
    1e-6.  KNOWN DIFFERENCE at the default: only cloud water in [1e-6, 2e-6)
    kg/kg sees a reduced rate (the number source scales with it).  Measured on
    the gSAM SB2001 kernel fixture: 168 of 1680 points change (those at
    q_c = 1e-6, rate 0 instead of gSAM's), largest change 1.5e-17 kg/kg/s =
    2.5e-14 of the fixture's largest rate; per-point and per-column gaps are
    pinned in test_sb2001_autoconv_ramp.py.  No invariant is broken (the rate
    only shrinks, stays >= 0).

    AD: piecewise-differentiable (gates and clamps are ``jnp.where``);
    ``dum^0.68`` goes through ``safe_pow`` (finite gradient at ``q_r = 0``).
    """
    if not ramp_top >= 0.0 or not math.isfinite(ramp_top):
        raise ValueError(
            f"ramp_top must be finite and >= 0 [kg/kg], got {ramp_top}")
    q_c_pos = jnp.clip(q_c, 0.0)
    q_r_pos = jnp.clip(q_r, 0.0)
    gate = q_c_pos >= _SB2001_PRC_QC_MIN
    qc_s = jnp.where(gate, q_c_pos, 1.0)
    _, nu, nc_adj = m2005_cloud_droplet_psd(qc_s, N_c_eff, rho)
    dum = 1.0 - qc_s / (qc_s + q_r_pos)
    dum_pow = safe_pow(dum, _SB2001_PHI_AU_TAU_EXP)
    dum1 = _SB2001_PHI_AU_PREFACTOR * dum_pow * (1.0 - dum_pow) ** 3
    shape = (nu + 2.0) * (nu + 4.0) / (nu + 1.0) ** 2
    prc = (
        _SB2001_AUTOCONV_MASS_COEFF * shape
        * (rho * qc_s * _LWC_KGM3_TO_GCM3) ** 4
        / (nc_adj * _NC_PERM3_TO_PERCM3) ** 2
        * (1.0 + dum1 / (1.0 - dum) ** 2) * _MASSRATE_GCM3_TO_MIXR / rho
    )
    if ramp_top > _SB2001_PRC_QC_MIN:
        prc = prc * smoothstep_quintic(
            (qc_s - _SB2001_PRC_QC_MIN) / (ramp_top - _SB2001_PRC_QC_MIN))
    dq_c_au = jnp.where(gate, prc, 0.0)
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    # NPRC1 = 0.5·NPRC = PRC/x_* (newborn drops carry the separation mass);
    # per-VOLUME here => ·rho.
    dN_r_au = dq_c_au * rho / _SB2001_X_STAR_KG
    return dq_c_au, dN_r_au, x_c


def accretion_sb2001(q_c, q_r, rho, *, qr_ramp_top, qc_ramp_top):
    """Seifert & Beheng (2001) accretion as gSAM M2005 runs it (IRAIN=1,
    module_mp_graupel.f90:1955-1962 / :3277-3285)::

        dum    = 1 − q_c/(q_c+q_r)
        phi_ac = (dum/(dum + 5e-4))^4
        PRA    = 5.78e3·rho/1000·q_c·q_r·phi_ac             [kg/kg/s]

    computed only where ``q_r >= 1e-8`` and ``q_c >= 1e-8`` (zero elsewhere).
    NOT ported: the cloud-number sink ``NPRA`` (only reached with
    ``predict_Nc=True``).

    RULE 5 (#2262): each of gSAM's two 1e-8 kg/kg steps is replaced by the C2
    step :func:`smoothstep_quintic`, exactly 0 at the threshold and 1 from its
    ramp top [kg/kg] up (``qr_ramp_top`` / ``qc_ramp_top`` =
    ``MorrisonConfig`` / ``SeifertBehengConfig``
    ``.sb2001_accretion_qr_ramp_top`` / ``.sb2001_accretion_qc_ramp_top``,
    default 2e-8 each, codex+GLM 2026-10-30); a top <= 1e-8 is gSAM's hard
    step for that gate.  The rate stays exactly 0 below either threshold.
    KNOWN DIFFERENCE at the default: only q_r or q_c in [1e-8, 2e-8) kg/kg
    sees a reduced rate.  Measured on the gSAM SB2001 kernel fixture: 192 of
    1680 points change (those at q_r = 1e-8, rate 0 instead of gSAM's),
    largest change 1.4e-13 kg/kg/s = 4.0e-9 of the fixture's largest rate; no
    fixture point lies in the cloud-water ramp; the whole-call gSAM columns
    change by exactly 0.  Gaps are pinned in test_sb2001_accretion_ramp.py.
    No invariant is broken (the rate only shrinks, stays >= 0).
    """
    for name, top in (("qr_ramp_top", qr_ramp_top), ("qc_ramp_top", qc_ramp_top)):
        if not top >= 0.0 or not math.isfinite(top):
            raise ValueError(f"{name} must be finite and >= 0 [kg/kg], got {top}")
    q_c_pos = jnp.clip(q_c, 0.0)
    q_r_pos = jnp.clip(q_r, 0.0)
    gate = (q_r_pos >= _SB2001_PRA_Q_MIN) & (q_c_pos >= _SB2001_PRA_Q_MIN)
    qc_s = jnp.where(gate, q_c_pos, 1.0)
    qr_s = jnp.where(gate, q_r_pos, 1.0)
    dum = 1.0 - qc_s / (qc_s + qr_s)
    phi_ac = (dum / (dum + _SB2001_PHI_AC_KTAU)) ** 4
    pra = _SB2001_ACCRETION_COEFF * rho * qc_s * qr_s * phi_ac
    for q_s, top in ((qr_s, qr_ramp_top), (qc_s, qc_ramp_top)):
        if top > _SB2001_PRA_Q_MIN:
            pra = pra * smoothstep_quintic(
                (q_s - _SB2001_PRA_Q_MIN) / (top - _SB2001_PRA_Q_MIN))
    return jnp.where(gate, pra, 0.0)


def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
    """Self-collection and breakup of rain drops.

    Parameters
    ----------
    N_r : array
        Rain drop number concentration [1/m^3] (per-VOLUME; the formulas
        ``q_r·rho/N_r`` and ``dN_r_sc ∝ N_r·q_r·rho`` use the per-volume
        convention, matching the tracer registry units and the radiation
        r_eff coupling). Returned number tendencies are likewise [1/(m^3 s)].
    q_r : array
        Rain mixing ratio [kg/kg].
    rho : array
        Air density [kg/m3].
    k_sc : float
        Self-collection rate constant.
    breakup_sharpness : float
        Sigmoid sharpness for breakup onset.
    D_eq : float
        Equilibrium drop diameter [m].

    Returns
    -------
    dN_r_sc : array
        Self-collection tendency [1/kg/s].
    dN_r_br : array
        Breakup tendency [1/kg/s].
    """
    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
    # Mean drop diameter D ~ (q_r * rho / N_r / (pi/6 * rho_water))^(1/3).
    # Cube-root has unbounded derivative at zero — guard with safe_pow.
    D_r_arg = (
        jnp.clip(q_r, 0.0) * rho
        / jnp.clip(N_r, 1.0)
        / (jnp.pi / 6.0 * constants.rho_water)
    )
    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
    # ``breakup_sharpness`` has units of [1/m] — the default
    # ``1e4 /m`` × ``(D_r − D_eq)`` with diameters O(1e-3 m) gives a
    # sigmoid argument of O(1) at the canonical ~0.1 mm transition
    # width around ``D_eq``.  Iter-98 incorrectly normalised this to
    # ``(D_r/D_eq − 1.0)`` without lowering the default, producing
    # an essentially-step-function transition; reverted in iter-99
    # after codex stop-time review.
    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
    dN_r_br = -dN_r_sc * breakup_frac
    return dN_r_sc, dN_r_br


def self_collection_breakup_sb2001(N_r, q_r, rho, config, dt=None):
    """SAM/Seifert-Beheng 2001 rain self-collection + breakup (NRAGG,
    module_mp_graupel.f90:1980-1986)::

        LAMR  = (π·ρ_w·N_r/(ρ·q_r))^⅓        (N_r per-VOLUME ⇒ /ρ; clamped)
        dum   = 1                              if 1/LAMR < d0  (self-collection)
              = 2 − exp(2300·(1/LAMR − d0))    if 1/LAMR ≥ d0  (→ breakup)
        NRAGG = −k·dum·q_r·N_r·ρ              (k=5.78, d0=300µm)

    NRAGG always relaxes N_r toward the SB EQUILIBRIUM drop size (where dum=0,
    1/LAMR = d0 + ln2/steepness ≈ 601µm): self-collection (dum>0, drops too
    small) reduces N_r; breakup (dum<0, drops too big) increases N_r. To keep
    the faithful rate while staying explicit-Euler stable, the per-step change
    is limited so it does NOT OVERSHOOT the local equilibrium N_r (codex
    iter-22 — better than a ±N_r/dt cap, which would throttle breakup and
    leave rain artificially overgrown). Without this, 1/LAMR→2800µm at the
    LAMR clamp gives exp(5.75)≈314 ⇒ runaway breakup. AD-safe via safe_pow
    (LAMR clamped positive).
    """
    qr_pos = jnp.clip(q_r, 0.0)
    nr_pos = jnp.clip(N_r, 0.0)
    lamr = safe_pow(
        jnp.pi * constants.rho_water * nr_pos
        / jnp.maximum(rho * qr_pos, 1.0e-20), 1.0 / 3.0)
    lamr = jnp.clip(lamr, config.lamr_min, config.lamr_max)
    inv_lamr = 1.0 / lamr
    d0 = config.rain_breakup_d0
    steep = config.rain_breakup_steepness
    dum = jnp.where(
        inv_lamr < d0, 1.0,
        2.0 - jnp.exp(steep * (inv_lamr - d0)))
    nragg = -config.rain_selfcoll_k * dum * qr_pos * nr_pos * rho
    nragg = jnp.where(qr_pos >= 1.0e-8, nragg, 0.0)     # SAM QR≥1e-8 gate
    if dt is not None:
        # Equilibrium number for this q_r: 1/LAMR_eq = d0 + ln2/steepness
        # (dum=0). NRAGG points toward N_r_eq; clip the step so it doesn't
        # overshoot (and so N_r stays ≥ 0, since N_r_eq ≥ 0).
        lamr_eq = jnp.clip(
            1.0 / (d0 + math.log(2.0) / steep),
            config.lamr_min, config.lamr_max)
        nr_eq = (lamr_eq ** 3 * rho * qr_pos
                 / (jnp.pi * constants.rho_water))
        delta_to_eq = nr_eq - nr_pos                    # signed target change
        nragg_dt = jnp.clip(
            nragg * dt,
            jnp.minimum(delta_to_eq, 0.0),
            jnp.maximum(delta_to_eq, 0.0))
        nragg = nragg_dt / jnp.maximum(dt, 1.0e-12)
    return nragg


def rain_freezing_bigg(q_r, N_r, T, rho, config, dt=None):
    """SAM Bigg (1953) immersion freezing of supercooled rain (MNUCCR/NNUCCR,
    module_mp_graupel.f90:3251-3257)::

        LAMR   = (π·ρ_w·N_r/(ρ·q_r))^⅓        (N_r per-VOLUME ⇒ /ρ; clamped)
        X      = exp(AIMM·(T₀−T)) − 1          (T<269.15 K only; AIMM=0.66/K, BIMM=100)
        MNUCCR = 20·π²·ρ_w·BIMM·(N_r/ρ)·X / LAMR^6     (mass [kg/kg/s])
        NNUCCR = π·N_r·BIMM·X / LAMR^3                 (number [/m³/s], per-vol)

    Returns ``(mnuccr, nnuccr)`` — the POSITIVE rain mass + number freezing
    rates, donor-clamped to q_r/dt and N_r/dt. Supercooled rain freezes to
    SNOW (legoESM has no graupel category; SAM freezes to graupel). The rate
    rises steeply with supercooling (exp), so below ≈−30 °C effectively all
    rain freezes in one step (clamped). AD-safe via safe_pow (LAMR clamped).
    """
    qr_pos = jnp.clip(q_r, 0.0)
    nr_pos = jnp.clip(N_r, 0.0)
    lamr = safe_pow(
        jnp.pi * constants.rho_water * nr_pos
        / jnp.maximum(rho * qr_pos, 1.0e-20), 1.0 / 3.0)
    lamr = jnp.clip(lamr, config.lamr_min, config.lamr_max)
    # Supercooling [K], capped at 100 K (everything is frozen well before
    # that) so AIMM·ΔT can't overflow expm1 in f32 (codex iter-24 F defensive).
    dT_sc = jnp.clip(constants.T_freeze - T, 0.0, 100.0)
    x = jnp.expm1(config.bigg_aimm * dT_sc)              # exp(AIMM·ΔT)−1 ≥ 0
    bimm = config.bigg_bimm
    lamr3 = lamr * lamr * lamr
    nnuccr = jnp.pi * nr_pos * bimm * x / lamr3
    mnuccr = (_BIGG_MNUCCR_PREFACTOR * jnp.pi ** 2 * constants.rho_water * bimm
              * (nr_pos / jnp.clip(rho, _RHO_FLOOR)) * x / (lamr3 * lamr3))
    # supercooled rain below -4 C only (SAM/MG2 T < 269.15 K), switched on by
    # the C2 step of bigg_cold_step (user decision 2026-10-22, 3a)
    cold = bigg_cold_step(T, config)
    gate = qr_pos > 1.0e-14
    mnuccr = jnp.where(gate, cold * mnuccr, 0.0)
    nnuccr = jnp.where(gate, cold * nnuccr, 0.0)
    if dt is not None:
        # JOINT clamp (codex iter-24 E): scale mass AND number by the same
        # factor so the frozen fraction stays consistent (Bigg freezes big
        # drops ⇒ mass fraction ≫ number fraction; independent clamps would
        # leave number-without-mass when the mass clamp binds).
        scale = jnp.minimum(
            1.0,
            jnp.minimum(
                qr_pos / jnp.maximum(mnuccr * dt, 1.0e-15),
                nr_pos / jnp.maximum(nnuccr * dt, 1.0e-15)))
        mnuccr = mnuccr * scale
        nnuccr = nnuccr * scale
    return mnuccr, nnuccr


_MG2_ICE_AC_TIME_S = 180.0     # ice_autoconversion ac_time [s] (micro_mg_utils.F90:873)
_FERRIER_ICE_NUMBER_DENOM = 6.0 * _MG2_ICE_AC_TIME_S   # Gamma(4) * ac_time
_MG2_ICE_D_MIN_M = 1.0e-6      # ice_lambda_bounds = 1/[2 dcs, 1 um] (micro_mg_utils.F90:294)
_MG2_MIN_MEAN_MASS_ICE = 1.0e-20  # min_mean_mass_ice [kg] (micro_mg_utils.F90:174)


def mg2_ice_lambda_bounds(dcs):
    """CAM6 MG2 cloud-ice slope bounds ``ice_lambda_bounds = 1/[2 dcs, 1 um]``
    (micro_mg_utils.F90:294) as ``(lami_min, lami_max)`` [1/m]."""
    return 1.0 / (2.0 * dcs), 1.0 / _MG2_ICE_D_MIN_M


def mg2_ice_slope(q_i, N_i, rho_ice, dcs, log_width=0.0):
    """MG2 ``size_dist_param_basic(mg_ice_props, qiic, niic, lami)``
    (micro_mg2_0.F90:1333) -> ``(niic, lami)``: ``N_i <= q_i/1e-20 kg``, then
    ``lami = (pi rhoi N_i/q_i)^(1/3)`` held inside ``1/[2 dcs, 1 um]`` with
    ``N_i`` re-adjusted to the bound.  Per-mass number; ``lami = 0`` where
    ``q_i <= qsmall``.

    KNOWN DIFFERENCE (RULE 5): MG2 clamps ``lami`` with a hard ``if``; with
    ``log_width > 0`` (``MorrisonConfig.mg2_ice_slope_log_width``) the clamp
    is a LogSumExp clamp of ``ln lami`` and ``niic`` follows the smooth slope
    everywhere, so ``lami``/``niic`` and the Ferrier rates differ from MG2
    near the bounds (measured gaps: tests/atmosphere/microphysics/unit/
    test_mg2_ice_snow_oracle.py).  ``log_width = 0`` recovers MG2 exactly.
    No invariant is lost: the slope stays inside the bounds to O(width).
    Default width 0.01 approved by user 2026-10-21."""
    return _mg2_size_dist_basic(jnp.clip(q_i, 0.0), jnp.clip(N_i, 0.0),
                                rho_ice * jnp.pi, *mg2_ice_lambda_bounds(dcs),
                                _MG2_MIN_MEAN_MASS_ICE, log_width=log_width)


def mg2_deposition_limiter(q_v, T, qvi, q_c, qcrat, vap_dep, mnuccd, nnuccd, dt,
                           width, *, presence_top):
    """MG2 cloud-ice deposition limits (micro_mg2_0.F90:1591-1621), grid rates
    [kg/kg/s, #/kg/s]: deposition keeps the part of the step without liquid,
    ``vap_dep (1 - qcrat)`` where ``qc >= qsmall``; then deposition +
    nucleation may take at most the vapour above ice saturation,
    ``(q - qvi)/(1 + L_s^2 qvi/(c_p R_v T^2))/dt``, shared pro rata, the
    nucleated number scaled with its mass (user decision 2026-10-21 S10.6 10a;
    MG2 leaves its own number unscaled).  The factor is
    :func:`mg2_qi_conservation_ratio`'s smooth min of 1 and cap/(vap_dep +
    mnuccd) at ``width`` (approved sharpness 100 = width 0.01; ``width -> 0``
    is MG2's hard form).  Returns ``(vap_dep, mnuccd, nnuccd)``."""
    vap_dep = vap_dep * (1.0 - mg2_presence(q_c, presence_top) * qcrat)
    cap = jnp.maximum((q_v - qvi) / (
        1.0 + constants.L_s ** 2 * qvi  # latent-ok: MG2 oracle constant (xxls_squared)
        / (constants.c_pd * constants.R_v * T ** 2)), 0.0)
    # max(., 0): the smooth min of 1 and 0 is -3.7e-46, not 0
    f = jnp.maximum(mg2_qi_conservation_ratio(cap, jnp.zeros_like(cap), vap_dep + mnuccd,
                                              dt, 1.0, width=width), 0.0)
    return vap_dep * f, mnuccd * f, nnuccd * f


def mg2_qi_conservation_ratio(q_avail, src, snk, dt, omsm, width=0.0):
    """MG2 conservation of qi (micro_mg2_0.F90:1717-1735): the factor on the
    ice sinks ``snk`` (autoconversion + sublimation) so that they take at most
    ``q_avail + src dt``, x omsm.  MG2: ``ratio = (q_avail/dt + src)/snk *
    omsm`` where ``(snk - src) dt > q_avail``, else 1.

    KNOWN DIFFERENCE (RULE 5): with ``width > 0``
    (``MorrisonConfig.mg2_qi_limit_width``) the factor is the LogSumExp
    smooth minimum of 1 and that ratio, which never exceeds the hard
    ``min(1, ratio)``, so q_i still cannot go negative; it trims the sinks by
    up to ``width ln 2`` where the ratio is near 1 and is 1 to rounding far
    from it (measured gaps: test_mg2_ice_snow_oracle.py).  ``width = 0`` is
    MG2's hard form exactly.  Default width 0.01 approved by user 2026-10-21."""
    # sinks below 1e-30 kg/kg/s: ratio >= q/dt/1e-30 >> 1 either way, so the
    # sentinel keeps fp32 r finite without changing the factor (1).
    has = snk > 1.0e-30
    r = jnp.where(has, (q_avail / dt + src) / jnp.where(has, snk, 1.0) * omsm,
                  1.0e30)
    if width > 0.0:
        return smooth_min(jnp.ones_like(r), r, 1.0 / width)
    return jnp.where((snk - src) * dt > q_avail, r, 1.0)


def ice_autoconversion_mg2(q_i, N_i, lami, dcs, T, cold_width_K):
    """MG2 ``ice_autoconversion`` (Ferrier 1994; micro_mg_utils.F90:859-901)
    ``(prci [kg/kg/s], nprci [#/kg/s])`` where ``q_i > 1e-14`` and
    ``T <= T_freeze``::

        d = lami dcs;  nprci = n0i/(lami 180) exp(-d) = N_i/180 exp(-d)
        prci  = (rhoi pi/6)/lami^3 nprci (((d+3) d+6) d+6)
              = q_i/1080 exp(-d) (((d+3) d+6) d+6)

    the mass rate in the 1/lami^3-free form (q_i = rhoi pi N_i/lami^3 for the
    exponential PSD), which stays finite as N_i -> 0 in fp32.  MG2's hard
    ``T <= T_freeze`` is the C2 step, 0 at 0 C and full by -``cold_width_K``
    (user decision 2026-10-21 S10.6b 5a, item 6a).

    KNOWN DIFFERENCE (user decision 2026-10-26 a): the step is exactly 0 AT
    T_freeze at every width, where MG2 (``T <= T_freeze``) converts; on the
    MG2 ice-rest fixture the 15 points at exactly T_freeze are -100 % (prci
    and nprci 0), at the hard limit and at the production 2 K width alike;
    all other points outside the separately recorded row-30 band
    (qsmall <= q_i <= 1e-14, where MG2 converts and the 1e-14 gate here does
    not) match (test_mg2_ice_rest_oracle.py, FERRIER_0C_GAP).  No width
    recovers the T_freeze point itself; ``melt_sharpness -> inf`` recovers
    MG2 below it, outside that band."""
    qi = jnp.clip(q_i, 0.0)
    on = qi > 1.0e-14
    cold = smoothstep_quintic((constants.T_freeze - T) / cold_width_K)
    d_rat = lami * dcs
    prci = ((qi / _FERRIER_ICE_NUMBER_DENOM) * jnp.exp(-d_rat)
            * (((d_rat + 3.0) * d_rat + 6.0) * d_rat + 6.0))
    nprci = jnp.clip(N_i, 0.0) / _MG2_ICE_AC_TIME_S * jnp.exp(-d_rat)
    return jnp.where(on, cold * prci, 0.0), jnp.where(on, cold * nprci, 0.0)


def bigg_cold_step(T, config):
    """How much of the Bigg rain-freezing rate is on: SAM and CAM6 MG2 freeze
    rain only where ``T < 269.15 K`` (hard).  Here the C2 step, 0 at 269.15 K
    and 1 from ``RAMP_WIDTH_PER_SHARPNESS / melt_sharpness`` below it (2 K in
    production; the same width field as the approved 0 C steps), user decision
    2026-10-22 (15a CAM lane, 3a SAM lane).  ``melt_sharpness -> inf`` is the
    hard cut.  KNOWN DIFFERENCE: rain at -4 to -6 C freezes at a reduced rate;
    on the MG2 fixture the per-column gap (max rate change / column's largest
    MG2 rate) is at most 3.3e-5 on both lanes (test_bigg_freezing_cold_step.py).
    """
    if math.isinf(config.melt_sharpness):   # static: 0*inf at 269.15 K would be NaN
        return jnp.where(T < _BIGG_T_MAX_K, 1.0, 0.0)
    return smoothstep_quintic((_BIGG_T_MAX_K - T) * config.melt_sharpness
                              / RAMP_WIDTH_PER_SHARPNESS)


def mg2_frozen_rain_ice_fraction(lamr, dcs, ramp_frac):
    """Share of MG2's frozen rain that becomes cloud ice, the rest snow.

    CAM6 MG2 (micro_mg2_0.F90:1648-1656) sends ALL of it to cloud ice where
    ``lamr > qsmall`` and the mean drop ``1/lamr < Dcs``.  Here the size test is
    the C2 step, 0 at ``1/lamr = Dcs`` and 1 from ``(1 - ramp_frac) Dcs`` down
    (user decision 2026-10-22, 15a: ramp_frac 0.1,
    ``MorrisonConfig.mg2_frozen_rain_ice_ramp_frac``); ``ramp_frac <= 0`` is
    CAM's hard test.  KNOWN DIFFERENCE: mean drops of 450-500 um (Dcs 500 um)
    send only part of the frozen rain to ice; on the MG2 fixture the largest
    per-column share difference is 0.75-1.0 (a 499 um drop: all ice in CAM,
    almost none here; test_morrison_mg2_frozen_rain_split.py).  Active only
    with do_graupel=False; the production deck has do_graupel=True.
    """
    has = lamr > MG2_QSMALL
    d = 1.0 / jnp.where(has, lamr, 1.0)
    if ramp_frac <= 0.0:
        return jnp.where(has & (d < dcs), 1.0, 0.0)
    return jnp.where(has, smoothstep_quintic((dcs - d) / (ramp_frac * dcs)), 0.0)


def rain_freezing_bigg_mg2(q_ric, N_ric, lamr, T, rho, precip_frac, config):
    """CAM6 MG2 ``heterogeneous_rain_freezing`` (micro_mg_utils.F90:1257-1285)
    on the in-precip rain of :func:`mg2_inprecip_rain` (``N_ric`` per-VOLUME,
    size-adjusted; ``lamr`` its slope), returned as GRID rates
    ``(mnuccr*precip_frac [kg/kg/s], nnuccr*precip_frac*rho [#/m^3/s])``::

        nnuccr = pi nric bimm (exp(aimm (T_freeze - T)) - 1)/lamr^3
        mnuccr = nnuccr 20 pi rhow/lamr^3

    where ``qric >= qsmall``, times :func:`bigg_cold_step` (MG2's ``T < 269.15
    K``, smoothed).  No internal clamp: the
    MG2 rain mass/number limiters bound it (micro_mg2_0.F90:1660-1709)."""
    # lamr = 0 at q_ric == qsmall exactly (the size routine needs > qsmall):
    # such a cell freezes nothing here (MG2 divides by 0 there)
    active = (q_ric >= MG2_QSMALL) & (lamr > 0.0)
    cold = bigg_cold_step(T, config)
    lam3 = jnp.where(active, lamr, 1.0) ** 3
    nnuccr = (jnp.pi * (N_ric / rho) * config.bigg_bimm
              * (jnp.exp(config.bigg_aimm * (constants.T_freeze - T)) - 1.0)
              / lam3)
    mnuccr = nnuccr * _BIGG_MNUCCR_PREFACTOR * jnp.pi * constants.rho_water / lam3
    return (jnp.where(active, cold * mnuccr * precip_frac, 0.0),
            jnp.where(active, cold * nnuccr * precip_frac * rho, 0.0))


def snow_deposition_m2005(q_v, q_s, N_s, q_sat_i, T, p, rho, config, dt=None):
    """SAM M2005 snow vapor deposition / sublimation PRDS (module_mp_graupel
    .f90:2039/3476) — the snow analog of the cloud-ice deposition::

        LAMS = (π·ρ_sn·N_s/q_s)^⅓ (clamped),  N0S = N_s·LAMS  (N_s per-mass)
        EPSS = 2π·N0S·ρ·DV·[F1S/LAMS² + F2S·CONS10·(ASN·ρ/μ)^½·SC^⅓·LAMS^(−CONS35)]
        PRDS = EPSS·(q_v−q_sat_i)/ABI

    PRDS>0 = DEPOSITION (snow grows, vapour sink, +L_s — the dominant anvil ice
    growth); PRDS<0 = SUBLIMATION (snow shrinks, vapour source, −L_s cooling —
    important in dry downdrafts), donor-clamped to q_s/dt. ABI is the ICE
    psychrometric correction (L_s). Returns the SIGNED PRDS [kg/kg/s]. F1S=0.86,
    F2S=0.28, CONS10=Γ(5/2+BS/2), CONS35=5/2+BS/2. AD-safe via safe_pow (LAMS
    clamped positive). Requires double-moment snow (N_s).
    """
    qs_pos = jnp.clip(q_s, 0.0)
    ns_pos = jnp.clip(N_s, 0.0)
    lams = safe_pow(
        config.rho_snow * jnp.pi * ns_pos
        / jnp.maximum(qs_pos, 1.0e-20), 1.0 / 3.0)
    lams = jnp.clip(lams, config.lams_min, config.lams_max)
    n0s = ns_pos * lams
    dv, mu, sc = _air_transport_props(T, p, rho)
    asn = config.fall_a_s * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    cons35 = _VENT_CONS_OFFSET + config.fall_b_s / 2.0
    cons10 = math.gamma(cons35)
    dqsidt = constants.L_s * q_sat_i / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    epss = (
        2.0 * jnp.pi * n0s * rho * dv
        * (config.snow_vent_f1 / (lams * lams)
           + config.snow_vent_f2 * cons10
           * safe_pow(asn * rho / mu, 0.5) * safe_pow(sc, 1.0 / 3.0)
           * safe_pow(lams, -cons35))
    )
    prds = epss * (q_v - q_sat_i) / abi
    prds = jnp.where(qs_pos > 1.0e-14, prds, 0.0)         # SAM QSMALL
    # Sublimation (negative) donor-clamped to available snow; deposition
    # (positive) is vapour-limited downstream (joint q_v donor clamp).
    dep_pos = jnp.maximum(prds, 0.0)
    if dt is not None:
        subl_neg = jnp.maximum(
            jnp.minimum(prds, 0.0), -qs_pos / jnp.maximum(dt, 1.0e-12))
    else:
        subl_neg = jnp.minimum(prds, 0.0)
    return dep_pos + subl_neg


def snow_riming_psacws(q_c, q_s, N_s, T, rho, config, dt=None):
    """SAM PSACWS snow riming of cloud water (module_mp_graupel.f90:2909)::

        LAMS   = (π·ρ_sn·N_s/q_s)^⅓ (clamped),  N0S = N_s·LAMS
        PSACWS = Γ(BS+3)·π/4·ECI · ASN · q_c·ρ · N0S / LAMS^(BS+3)

    The PSD-integrated continuous collection of cloud droplets by falling
    snow (ASN=AS·(ρ_su/ρ)^0.54, ECI=0.7). SUPERCOOLED droplets freeze onto the
    snow (riming) ⇒ q_c→q_s + L_f, so the rate is gated on T<T_freeze. Returns
    the POSITIVE riming rate [kg/kg/s], donor-clamped to q_c/dt. The snow
    NUMBER is unchanged (existing flakes grow). Requires double-moment snow.
    """
    qc_pos = jnp.clip(q_c, 0.0)
    qs_pos = jnp.clip(q_s, 0.0)
    ns_pos = jnp.clip(N_s, 0.0)
    lams = safe_pow(
        config.rho_snow * jnp.pi * ns_pos
        / jnp.maximum(qs_pos, 1.0e-20), 1.0 / 3.0)
    lams = jnp.clip(lams, config.lams_min, config.lams_max)
    n0s = ns_pos * lams
    asn = config.fall_a_s * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    bs = config.fall_b_s
    cons13 = math.gamma(bs + 3.0) * jnp.pi / 4.0 * config.snow_collect_eff
    psacws = cons13 * asn * qc_pos * rho * n0s / safe_pow(lams, bs + 3.0)
    # Riming = supercooled droplets freezing onto snow ⇒ cold gate (→1 below
    # 0 °C, →0 above) — the SAME sigmoid as the melting fraction, reversed.
    rime_frac = jax.nn.sigmoid(config.melt_sharpness * (constants.T_freeze - T))
    psacws = psacws * rime_frac
    psacws = jnp.where(
        (qs_pos > 1.0e-14) & (qc_pos > 1.0e-14), psacws, 0.0)
    if dt is not None:
        psacws = jnp.minimum(psacws, qc_pos / jnp.maximum(dt, 1.0e-12))
    return psacws


def snow_melting_psmlt(q_s, N_s, T, p, rho, config, dt=None):
    """SAM PSMLT heat-balance-limited snow melting (module_mp_graupel.f90:2030)::

        LAMS = (π·ρ_sn·N_s/q_s)^⅓ (clamped),  N0S = N_s·LAMS,  KAP = 1.414e3·μ
        melt = 2π·N0S·KAP·(T−T₀)₊/L_f · [F1S/LAMS²
                  + F2S·CONS10·(ASN·ρ/μ)^½·SC^⅓·LAMS^(−CONS35)]

    Snow melts to rain at the VENTILATION-limited heat-conduction rate (KAP is
    the air thermal conductivity), not a bulk rate — so snow doesn't persist
    far below the 0 °C level. Melting absorbs L_f (cooling). Returns the
    POSITIVE melt rate [kg/kg/s], donor-clamped to q_s/dt AND to the available
    above-freezing sensible heat (so it can't cool the air below 0 °C in one
    step). The SAM DUM sensible-heat-of-accreted-water correction is omitted —
    it is ≈0 in legoESM anyway (snow riming PSACWS is gated to T<0 °C and there
    is no rain-snow collision PRACS, so no warm water accretes onto melting
    snow). Requires double-moment snow (N_s).
    """
    qs_pos = jnp.clip(q_s, 0.0)
    ns_pos = jnp.clip(N_s, 0.0)
    lams = safe_pow(
        config.rho_snow * jnp.pi * ns_pos
        / jnp.maximum(qs_pos, 1.0e-20), 1.0 / 3.0)
    lams = jnp.clip(lams, config.lams_min, config.lams_max)
    n0s = ns_pos * lams
    dv, mu, sc = _air_transport_props(T, p, rho)
    kap = _AIR_CONDUCTIVITY_MU_FACTOR * mu               # air thermal conductivity
    asn = config.fall_a_s * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    cons35 = _VENT_CONS_OFFSET + config.fall_b_s / 2.0
    cons10 = math.gamma(cons35)
    vent = (config.snow_vent_f1 / (lams * lams)
            + config.snow_vent_f2 * cons10 * safe_pow(asn * rho / mu, 0.5)
            * safe_pow(sc, 1.0 / 3.0) * safe_pow(lams, -cons35))
    melt = (2.0 * jnp.pi * n0s * kap
            * jnp.maximum(T - constants.T_freeze, 0.0) / constants.L_f * vent)
    melt = jnp.where(qs_pos > 1.0e-14, melt, 0.0)
    if dt is not None:
        # Heat-balance cap (codex iter-29 D): melting absorbs L_f, so it cannot
        # melt more snow than the available ABOVE-FREEZING sensible heat would
        # allow without cooling the air below 0 °C in one step:
        #   melt·dt·L_f ≤ c_p·(T−T₀)  ⇒  melt ≤ c_p·(T−T₀)/(L_f·dt).
        melt_heat_cap = (
            constants.c_pd * jnp.maximum(T - constants.T_freeze, 0.0)
            / (constants.L_f * jnp.maximum(dt, 1.0e-12)))
        melt = jnp.minimum(melt, melt_heat_cap)
        melt = jnp.minimum(melt, qs_pos / jnp.maximum(dt, 1.0e-12))
    return melt


def snow_self_aggregation_nsagg(q_s, N_s, rho, config, dt=None):
    """SAM NSAGG snow self-aggregation (module_mp_graupel.f90:2894)::

        CONS15 = −1108·EII·π^((1−BS)/3)·ρ_sn^((−2−BS)/3)/(4·720)   (EII=0.1)
        NSAGG  = CONS15·ASN·ρ^((2+BS)/3)·q_s^((2+BS)/3)·(N_s·ρ)^((4−BS)/3)/ρ

    Passarelli-1978 / Reisner-1998 aggregation: falling snow flakes collide and
    merge into FEWER, larger flakes. A pure SINK of snow NUMBER N_s — the snow
    MASS q_s is CONSERVED (no q_s tendency, no latent heat). CONS15<0 ⇒ NSAGG<0;
    this returns the POSITIVE number-loss rate [1/kg/s] (caller subtracts it),
    donor-clamped to N_s/dt. SAM gates on q_s≥1e-8. Requires double-moment snow.
    """
    qs_pos = jnp.clip(q_s, 0.0)
    ns_pos = jnp.clip(N_s, 0.0)
    bs = config.fall_b_s
    asn = config.fall_a_s * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    cons15 = (_NSAGG_CONS15_PREFACTOR * config.snow_aggregation_eii
              * safe_pow(jnp.pi, (1.0 - bs) / 3.0)
              * config.rho_snow ** ((-2.0 - bs) / 3.0) / (4.0 * _NSAGG_GAMMA_DENOM))
    # |NSAGG| with CONS15's sign folded in (return the positive loss directly).
    loss = (cons15 * asn
            * safe_pow(rho, (2.0 + bs) / 3.0)
            * safe_pow(qs_pos, (2.0 + bs) / 3.0)
            * safe_pow(ns_pos * rho, (4.0 - bs) / 3.0)
            / jnp.clip(rho, _RHO_FLOOR))
    loss = jnp.where(qs_pos >= 1.0e-8, loss, 0.0)
    if dt is not None:
        # The instantaneous rate is super-linear in N_s (loss ∝ N_s^p,
        # p=(4−BS)/3≈1.2), so a forward-Euler step can overshoot N_s for stiff
        # (dense-snow) cells. Integrate the ISOLATED aggregation ODE
        # dN/dt = −k·N^p EXACTLY over dt instead (codex iter-30 E): with
        # r = loss·dt/N_s the explicit fractional removal,
        #   N(dt)/N₀ = [1 + (p−1)·r]^(1/(1−p))   (monotone, always > 0),
        # and the effective average sink is N₀·(1 − that)/dt. Reduces to the
        # explicit rate when r≪1 (τ≫dt, the normal regime); never drives N_s<0.
        p = (4.0 - bs) / 3.0
        r = loss * dt / jnp.maximum(ns_pos, 1.0e-30)
        ratio = safe_pow(1.0 + (p - 1.0) * r, 1.0 / (1.0 - p))
        loss = ns_pos * (1.0 - ratio) / jnp.maximum(dt, 1.0e-12)
        loss = jnp.minimum(loss, ns_pos / jnp.maximum(dt, 1.0e-12))
    return loss


def graupel_lamg(q_g, rho, config, N_g=None):
    """Graupel PSD slope LAMG (clamped to SAM's [LAMMING, LAMMAXG]).

    SINGLE-moment (``N_g is None``): exponential PSD with a FIXED per-volume
    intercept N0G [1/m⁴], ``ρ·q_g = π·ρ_g·N0G/LAMG⁴`` ⇒
    ``LAMG = (π·ρ_g·N0G/(ρ·q_g))^¼``.

    DOUBLE-moment (``N_g`` given, per-mass [1/kg]): the snow-like slope
    ``LAMG = (π·ρ_g·N_g/q_g)^⅓`` (SAM ``(CONS2·NG3D/QG3D)^(1/DG)``, DG=3, no
    ρ_air since N_g is per-mass — same convention as N_s).
    """
    qg_pos = jnp.clip(q_g, 0.0)
    rho_eff = jnp.clip(rho, _RHO_FLOOR)
    if N_g is None:
        lamg = safe_pow(
            jnp.pi * config.rho_graupel * config.n0_graupel
            / jnp.maximum(rho_eff * qg_pos, 1.0e-20), 0.25)
    else:
        lamg = safe_pow(
            jnp.pi * config.rho_graupel * jnp.clip(N_g, 0.0)
            / jnp.maximum(qg_pos, 1.0e-20), 1.0 / 3.0)
    return jnp.clip(lamg, config.lamg_min, config.lamg_max)


def graupel_lamg_n0g(q_g, rho, config, N_g=None):
    """Graupel PSD slope LAMG and PER-MASS intercept N0G_m, both moment modes.

    Returns ``(lamg, n0g_m)``. The per-mass intercept N0G_m feeds the SAM
    process formulas (PGMLT/PSACWG/PRACG use N0G_m; PRDG uses N0G_m·ρ):

    * single-moment: ``N0G_m = N0G_vol/ρ`` (so ``N0G_m·ρ = N0G_vol``);
    * double-moment: ``N0G_m = N_g·LAMG`` (per-mass, like SAM N0G = NG3D·LAMG).
    """
    lamg = graupel_lamg(q_g, rho, config, N_g=N_g)
    if N_g is None:
        n0g_m = config.n0_graupel / jnp.clip(rho, _RHO_FLOOR)
    else:
        n0g_m = jnp.clip(N_g, 0.0) * lamg
    return lamg, n0g_m


def graupel_melting_pgmlt(q_g, T, p, rho, config, dt=None, N_g=None):
    """SAM PGMLT graupel melting (module_mp_graupel.f90:2069)::

        melt = 2π·N0G_m·KAP·(T−T₀)₊/L_f · [F1S/LAMG²
                  + F2S·CONS11·(AGN·ρ/μ)^½·SC^⅓·LAMG^(−CONS36)]

    Same ventilation-limited heat-conduction form as snow PSMLT, with the
    graupel PSD (LAMG, AG/BG, F1S/F2S shared with snow, CONS11=Γ(5/2+BG/2),
    CONS36=5/2+BG/2). SAM's N0G is a PER-MASS intercept; for single-moment
    graupel the per-mass-equivalent is ``N0G_m = N0G_vol/ρ`` (converts the
    fixed per-volume intercept to per-mass, matching the validated snow PSMLT
    where N0S=N_s·LAMS is per-mass). Returns the POSITIVE melt rate [kg/kg/s]
    → rain, donor-clamped to q_g/dt AND the available above-freezing sensible
    heat (no over-cool below 0 °C).
    """
    qg_pos = jnp.clip(q_g, 0.0)
    lamg, n0g_m = graupel_lamg_n0g(q_g, rho, config, N_g=N_g)
    dv, mu, sc = _air_transport_props(T, p, rho)
    kap = _AIR_CONDUCTIVITY_MU_FACTOR * mu
    agn = config.fall_a_g * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    cons36 = _VENT_CONS_OFFSET + config.fall_b_g / 2.0
    cons11 = math.gamma(cons36)
    vent = (config.graupel_vent_f1 / (lamg * lamg)
            + config.graupel_vent_f2 * cons11 * safe_pow(agn * rho / mu, 0.5)
            * safe_pow(sc, 1.0 / 3.0) * safe_pow(lamg, -cons36))
    melt = (2.0 * jnp.pi * n0g_m * kap
            * jnp.maximum(T - constants.T_freeze, 0.0) / constants.L_f * vent)
    melt = jnp.where(qg_pos > 1.0e-14, melt, 0.0)
    if dt is not None:
        melt_heat_cap = (
            constants.c_pd * jnp.maximum(T - constants.T_freeze, 0.0)
            / (constants.L_f * jnp.maximum(dt, 1.0e-12)))
        melt = jnp.minimum(melt, melt_heat_cap)
        melt = jnp.minimum(melt, qg_pos / jnp.maximum(dt, 1.0e-12))
    return melt


def graupel_riming_psacwg(q_c, q_g, T, rho, config, dt=None, N_g=None):
    """SAM PSACWG graupel riming of cloud water (module_mp_graupel.f90:2923)::

        PSACWG = CONS14·AGN·q_c·ρ·N0G_m/LAMG^(BG+3),  CONS14=Γ(BG+3)·π/4·ECI

    The PSD-integrated continuous collection of supercooled cloud droplets by
    falling graupel (AGN=AG·(ρ_su/ρ)^0.54, ECI=0.7) — the primary graupel
    GROWTH path in updrafts. q_c→q_g + L_f, so it is gated to T<T_freeze (above
    0 °C the graupel melts anyway). Returns the POSITIVE riming rate [kg/kg/s],
    donor-clamped to q_c/dt. Mirrors snow PSACWS with the graupel PSD; N0G_m =
    N0G_vol/ρ is the per-mass intercept (single-moment graupel).
    """
    qc_pos = jnp.clip(q_c, 0.0)
    qg_pos = jnp.clip(q_g, 0.0)
    lamg, n0g_m = graupel_lamg_n0g(q_g, rho, config, N_g=N_g)
    agn = config.fall_a_g * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    bg = config.fall_b_g
    cons14 = math.gamma(bg + 3.0) * jnp.pi / 4.0 * config.graupel_collect_eff
    psacwg = cons14 * agn * qc_pos * rho * n0g_m / safe_pow(lamg, bg + 3.0)
    # Cold-cloud gate (supercooled droplets freeze onto graupel) — the SAME
    # sigmoid as the snow-riming / melting fraction.
    rime_frac = jax.nn.sigmoid(config.melt_sharpness * (constants.T_freeze - T))
    psacwg = psacwg * rime_frac
    psacwg = jnp.where(
        (qg_pos > 1.0e-8) & (qc_pos > 1.0e-14), psacwg, 0.0)
    if dt is not None:
        psacwg = jnp.minimum(psacwg, qc_pos / jnp.maximum(dt, 1.0e-12))
    return psacwg


def graupel_deposition_prdg(q_v, q_g, q_sat_i, T, p, rho, config, dt=None,
                            N_g=None):
    """SAM PRDG graupel vapor deposition/sublimation (module_mp_graupel.f90:3483)::

        EPSG = 2π·N0G·DV·[F1S/LAMG² + F2S·CONS11·(AGN·ρ/μ)^½·SC^⅓·LAMG^(−CONS36)]
        PRDG = EPSG·(q_v−q_sat_i)/ABI

    The graupel analog of snow PRDS. SAM writes EPSG=2π·N0G·ρ·DV·[…] with a
    PER-MASS N0G; for single-moment graupel the per-mass intercept is N0G_vol/ρ,
    so N0G_m·ρ = N0G_vol — the ρ cancels and we use the fixed per-VOLUME N0G
    directly (same identity used for snow's N0S·ρ). PRDG>0 = DEPOSITION (graupel
    grows, vapour sink, +L_s); PRDG<0 = SUBLIMATION (graupel shrinks, vapour
    source, −L_s cooling in dry downdrafts), donor-clamped to q_g/dt. ABI is the
    ICE psychrometric correction (L_s). Returns the SIGNED PRDG [kg/kg/s].
    """
    qg_pos = jnp.clip(q_g, 0.0)
    lamg, n0g_m = graupel_lamg_n0g(q_g, rho, config, N_g=N_g)
    dv, mu, sc = _air_transport_props(T, p, rho)
    agn = config.fall_a_g * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    cons36 = _VENT_CONS_OFFSET + config.fall_b_g / 2.0
    cons11 = math.gamma(cons36)
    dqsidt = constants.L_s * q_sat_i / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    # EPSG = 2π·N0G_m·ρ·DV·[vent] (SAM per-mass form). Single-moment: N0G_m·ρ =
    # N0G_vol; double-moment: N0G_m·ρ = N_g·LAMG·ρ.
    epsg = (
        2.0 * jnp.pi * n0g_m * jnp.clip(rho, _RHO_FLOOR) * dv
        * (config.graupel_vent_f1 / (lamg * lamg)
           + config.graupel_vent_f2 * cons11
           * safe_pow(agn * rho / mu, 0.5) * safe_pow(sc, 1.0 / 3.0)
           * safe_pow(lamg, -cons36))
    )
    prdg = epsg * (q_v - q_sat_i) / abi
    prdg = jnp.where(qg_pos > 1.0e-14, prdg, 0.0)            # SAM QSMALL
    # Sublimation (negative) donor-clamped to available graupel; deposition
    # (positive) is vapour-limited downstream (joint q_v donor clamp).
    dep_pos = jnp.maximum(prdg, 0.0)
    if dt is not None:
        subl_neg = jnp.maximum(
            jnp.minimum(prdg, 0.0), -qg_pos / jnp.maximum(dt, 1.0e-12))
    else:
        subl_neg = jnp.minimum(prdg, 0.0)
    return dep_pos + subl_neg


def graupel_rain_accretion_pracg(q_r, N_r, q_g, T, rho, config, dt=None,
                                 N_g=None):
    """SAM PRACG rain accretion by graupel — COLD branch (module_mp_graupel
    .f90:3031), the Wisner-type two-PSD gravitational collection::

        VDIFF = ((1.2·UMR−0.95·UMG)² + 0.08·UMG·UMR)^½
        PRACG = CONS41·VDIFF·ρ·N0RR·N0G/LAMR³·[5/(LAMR³·LAMG)
                  + 2/(LAMR²·LAMG²) + 0.5/(LAMR·LAMG³)],   CONS41 = π²·ECR·ρ_w

    Falling graupel sweeps up rain drops; below 0 °C the collected rain FREEZES
    onto the graupel ⇒ q_r→q_g + L_f (cold-cloud gated). Returns the POSITIVE
    accretion rate [kg/kg/s], donor-clamped to q_r/dt.

    Intercepts are PER-MASS to match SAM (legoESM ``N_r`` is per-VOLUME): N0RR =
    (N_r/ρ)·LAMR, N0G = N0G_vol/ρ; the trailing 1/LAMR³·[bracket] then yields the
    [kg/kg/s] units. UMR/UMG are the SAM-capped mass-weighted fall speeds. The
    warm-branch (T>0) graupel→rain shedding is omitted (PGMLT melts graupel to
    rain there). Single-moment graupel.
    """
    qr_pos = jnp.clip(q_r, 0.0)
    qg_pos = jnp.clip(q_g, 0.0)
    nr_pos = jnp.clip(N_r, 0.0)
    # Single ρ_eff basis for LAMR, N0RR, N0G and the explicit PRACG ρ — ρ_eff is
    # a numerical floor only (= ρ in all valid cells), codex iter-37 A.
    rho_eff = jnp.clip(rho, _RHO_FLOOR)
    dum = safe_pow(config.rho_su / rho_eff, _FALL_RHO_EXPONENT)
    # Rain PSD (per-mass: N_r per-volume ⇒ /ρ in LAMR), same as the sed block.
    lamr = safe_pow(
        jnp.pi * constants.rho_water * nr_pos
        / jnp.maximum(rho_eff * qr_pos, 1.0e-20), 1.0 / 3.0)
    lamr = jnp.clip(lamr, config.lamr_min, config.lamr_max)
    n0rr = nr_pos / rho_eff * lamr             # per-mass rain intercept
    lamg, n0g_m = graupel_lamg_n0g(q_g, rho, config, N_g=N_g)
    cons4 = math.gamma(4.0 + config.fall_b_r) / 6.0
    cons7g = math.gamma(4.0 + config.fall_b_g) / 6.0
    umr = config.fall_a_r * cons4 * safe_pow(lamr, -config.fall_b_r) * dum
    umr = jnp.minimum(umr, _UMR_FALL_CAP * dum)
    umg = config.fall_a_g * cons7g * safe_pow(lamg, -config.fall_b_g) * dum
    umg = jnp.minimum(umg, _UMG_FALL_CAP * dum)
    vdiff = safe_pow(
        (_VDIFF_C1 * umr - _VDIFF_C2 * umg) ** 2 + _VDIFF_C3 * umg * umr, 0.5)
    cons41 = jnp.pi ** 2 * config.graupel_rain_collect_eff * constants.rho_water
    bracket = (_PRACG_BRACKET_C / (safe_pow(lamr, 3.0) * lamg)
               + 2.0 / (lamr * lamr * lamg * lamg)
               + 0.5 / (lamr * safe_pow(lamg, 3.0)))
    pracg = (cons41 * vdiff * rho_eff * n0rr * n0g_m
             / safe_pow(lamr, 3.0) * bracket)
    # Cold-cloud gate (collected rain freezes onto graupel) — same sigmoid as
    # the riming / melting fraction.
    rime_frac = jax.nn.sigmoid(config.melt_sharpness * (constants.T_freeze - T))
    pracg = pracg * rime_frac
    pracg = jnp.where(
        (qr_pos > 1.0e-8) & (qg_pos > 1.0e-8), pracg, 0.0)
    if dt is not None:
        pracg = jnp.minimum(pracg, qr_pos / jnp.maximum(dt, 1.0e-12))
    return pracg


def snow_to_graupel_pgsacw(q_c, q_s, N_s, psacws, rho, config, dt):
    """SAM PGSACW heavy-rimed snow → graupel (module_mp_graupel.f90:3195)::

        PGSACW = min(PSACWS, CONS17·dt·N0S·q_c²·ASN²/(ρ·LAMS^(2BS+2)))
        CONS17 = 3·ρ_su·π·ECI²·Γ(2BS+2)/(ρ_g − ρ_sn)

    When snow rimes cloud water heavily (Rutledge-Hobbs 1984 gate: q_s ≥ 0.1
    g/kg AND q_c ≥ 0.5 g/kg), a PORTION of the riming PSACWS densifies the snow
    into GRAUPEL embryos instead of growing the snow (Reisner 1998 / Murakami
    1990). Returns PGSACW [kg/kg/s] ≤ PSACWS — the cloud-water riming redirected
    to graupel. The caller then puts (PSACWS − PGSACW) on snow and PGSACW on
    graupel (q_c still loses the full PSACWS; mass conserved), plus the snow-
    number sink NSCNG. The ``dt`` in the rate is SAM's Murakami conversion
    timescale (the formula is per-step). Requires double-moment snow (N_s).
    """
    qc_pos = jnp.clip(q_c, 0.0)
    qs_pos = jnp.clip(q_s, 0.0)
    ns_pos = jnp.clip(N_s, 0.0)
    lams = safe_pow(
        config.rho_snow * jnp.pi * ns_pos
        / jnp.maximum(qs_pos, 1.0e-20), 1.0 / 3.0)
    lams = jnp.clip(lams, config.lams_min, config.lams_max)
    n0s = ns_pos * lams
    asn = config.fall_a_s * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    bs = config.fall_b_s
    cons17 = (3.0 * config.rho_su * jnp.pi * config.snow_collect_eff ** 2
              * math.gamma(2.0 * bs + 2.0)
              / (config.rho_graupel - config.rho_snow))
    rate = (cons17 * dt * n0s * qc_pos * qc_pos * asn * asn
            / (jnp.clip(rho, _RHO_FLOOR) * safe_pow(lams, 2.0 * bs + 2.0)))
    pgsacw = jnp.minimum(jnp.clip(psacws, 0.0), rate)
    # Rutledge-Hobbs 1984 gate: enough snow AND cloud water, and active riming.
    pgsacw = jnp.where(
        (qs_pos >= 1.0e-4) & (qc_pos >= 5.0e-4) & (psacws > 0.0), pgsacw, 0.0)  # coeff-ok: Rutledge-Hobbs riming gates
    return pgsacw


def rain_evaporation(q_v, q_r, q_sat, evap_coeff, dt=None, rh_deficit_floor=0.0):
    """Compute rain evaporation in subsaturated air.

    When ``dt`` is provided the returned evaporation rate is
    donor-limited: ``evap · dt ≤ q_r``.  Without the limit one explicit
    step can evaporate more rain than exists (and over-heat/cool the
    column), since the Marshall-Palmer rate scales as ``q_r^0.525``
    rather than ``q_r``.  Codex finding iter-3 #6.

    Parameters
    ----------
    q_v : array
        Water vapor mixing ratio [kg/kg].
    q_r : array
        Rain mixing ratio [kg/kg].
    q_sat : array
        Saturation mixing ratio [kg/kg].
    evap_coeff : float
        Evaporation rate coefficient.
    dt : float, optional
        Physics step [s].  When provided, clamp the evaporation rate
        so ``evap · dt ≤ q_r`` (donor positivity).
    rh_deficit_floor : float, default 0.0
        Relative-humidity-deficit resolution floor (dimensionless,
        ``(q_sat − q_v)/q_sat``) below which rain evaporation is
        suppressed via a soft threshold ``clip(deficit − floor, 0)``.
        ``0.0`` (default) reproduces the legacy ungated behaviour.  A
        small positive value (~5e-5, i.e. RH > 99.995 %) is a documented
        float32-robustness knob — see Notes.  Negative values are floored
        to 0.

    Returns
    -------
    array : Evaporation rate [kg/kg/s].

    Notes
    -----
    **Deficit-resolution floor (float32 robustness).**  In a saturated
    cloud ``q_v = q_sat`` to float64 (the difference of two ~1e-2 numbers
    is a true zero), so the ``clip(q_sat − q_v, 0)`` deficit — and the
    evaporation — is exactly 0.  Under float32 the saturation curve has a
    ~1e-5 *relative* noise floor (measured in-cloud band ~9e-6), so
    ``q_sat − q_v`` jitters even in cloud; the one-sided ``clip(·, 0)``
    RECTIFIES that symmetric noise into a net spurious in-cloud
    evaporation (~2e-8 kg/kg/s) that recycles rain → vapour → cloud and
    inflates the cloudy-column liquid-water path (an 18–24 %
    float32-vs-float64 LWP divergence on DYCOMS-II).  A small
    ``rh_deficit_floor`` (~5e-5) suppresses evaporation only where the
    sub-saturation is below the float32 saturation resolution.  It is
    applied to the DEFICIT itself, NOT gated on cloud presence: any
    *resolved* liquid sub-saturation — including mixed-phase
    Wegener–Bergeron–Findeisen cells (deficit ~0.1 ≫ floor) and sub-cloud
    downdrafts — still evaporates, losing only the ``floor/deficit``
    fraction (≲0.05 % at WBF/sub-cloud scales).  The suppression is
    material only for genuinely near-saturated columns (RH ≳ 99.9 %),
    where rain evaporation is itself marginal: a deficit of ``2·floor``
    loses 50 %, of ``10·floor`` loses 10 %.

    The float64 answer is unchanged for an exactly-saturated cloud (the
    in-cloud deficit is already 0).  Where a float64 run holds rain in a
    *persistently near-saturated* cloud (deficit ≲ floor, e.g. the DYCOMS
    cold variant whose saturation-adjusted column sits at RH ~99.99 %),
    the floor DOES lower its LWP — but that evaporation is at a deficit
    float32 cannot represent, so removing it (identically in both
    precisions) is what lets float32 converge to float64, and is
    physically defensible: rain does not meaningfully evaporate at
    RH > 99.99 %.
    """
    floor = jnp.maximum(rh_deficit_floor, 0.0)
    deficit = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    # Soft threshold: suppress sub-resolution (RH>99.99%) deficits, pass
    # resolved ones (shifted by the constant ``floor``). At floor=0 this is
    # ``clip(deficit, 0) = deficit`` — exactly the legacy behaviour.
    subsaturation = jnp.clip(deficit - floor, 0.0)
    q_r_pos = jnp.clip(q_r, 0.0, None)
    # Marshall-Palmer ventilation factor q_r^0.525 — fractional power has
    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
    rate = evap_coeff * subsaturation * safe_pow(q_r_pos, _RAIN_EVAP_VENT_EXP)
    if dt is not None:
        max_rate = q_r_pos / jnp.maximum(dt, 1.0e-12)
        rate = jnp.minimum(rate, max_rate)
    return rate


def rain_evaporation_m2005(q_v, q_r, N_r, q_sat, T, p, rho, config, dt=None):
    """SAM M2005 rain evaporation PRE (module_mp_graupel.f90:1994-2010).

    Diffusion + ventilation evaporation of the rain PSD (vs the bulk
    ``evap_coeff·subsat·q_r^0.525``)::

        LAMR = (π·ρ_w·N_r/(ρ·q_r))^⅓        (N_r per-VOLUME ⇒ extra ÷ρ; clamped)
        N0R  = N_r·LAMR                       (per-volume PSD intercept)
        EPSR = 2π·N0R·DV·[F1R/LAMR²
                          + F2R·CONS9·(ARN·ρ/μ)^½·SC^⅓·LAMR^(−CONS34)]
        PRE  = EPSR·(q_v−q_sat)/AB            (subsaturated ⇒ PRE<0)

    Returns the POSITIVE evaporation rate ``−PRE = EPSR·(q_sat−q_v)₊/AB``
    (a q_r sink / q_v source), donor-clamped to ``q_r/dt``. DV is the vapour
    diffusivity, μ the Sutherland dynamic viscosity, SC the Schmidt number,
    AB the LIQUID psychrometric correction (L_v), CONS9=Γ(5/2+b_r/2),
    CONS34=5/2+b_r/2. The second EPSR term is the ventilation enhancement —
    falling drops evaporate faster — which the bulk form misses. AD-safe via
    ``safe_pow`` (LAMR clamped positive, so the negative power is finite).
    """
    qr_pos = jnp.clip(q_r, 0.0)
    nr_pos = jnp.clip(N_r, 0.0)
    lamr = safe_pow(
        jnp.pi * constants.rho_water * nr_pos
        / jnp.maximum(rho * qr_pos, 1.0e-20), 1.0 / 3.0)
    lamr = jnp.clip(lamr, config.lamr_min, config.lamr_max)
    n0r = nr_pos * lamr
    dv, mu, sc = _air_transport_props(T, p, rho)
    arn = config.fall_a_r * safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    cons34 = _VENT_CONS_OFFSET + config.fall_b_r / 2.0
    cons9 = math.gamma(cons34)
    dqsdt = constants.L_v * q_sat / (constants.R_v * T ** 2)
    ab = 1.0 + dqsdt * constants.L_v / constants.c_pd
    epsr = (
        2.0 * jnp.pi * n0r * dv
        * (config.rain_vent_f1 / (lamr * lamr)
           + config.rain_vent_f2 * cons9
           * safe_pow(arn * rho / mu, 0.5) * safe_pow(sc, 1.0 / 3.0)
           * safe_pow(lamr, -cons34))
    )
    evap = epsr * jnp.maximum(q_sat - q_v, 0.0) / ab
    evap = jnp.where(qr_pos >= _M2005_QSMALL, evap, 0.0)  # EPSR only where QR >= QSMALL (:1990, :3455)
    if dt is not None:
        evap = jnp.minimum(evap, qr_pos / jnp.maximum(dt, 1.0e-12))
    return evap


# --- gSAM 1.8.8 M2005, liquid only (doicemicro=.false. => ILIQ=1), SB2001
# warm rain (dosb_warm_rain=.true. => IRAIN=1), specified droplet number
# (dopredictNc=.false. => INUM=1): SRC/MICRO_M2005/module_mp_graupel.f90 ---
_M2005_SMALL_REMOVAL_RH = 0.9     # small liquid evaporated where q_v/q_sat < 0.9 (:1453, :4669) [-]
_M2005_SMALL_REMOVAL_Q = 1.0e-8   # ... when below 1e-8 kg/kg (:1454-1463, :4670-4679) [kg/kg]
_M2005_FALL_FILL_MIN = 1.0e-10    # fall speed below this takes the level above's (:4282-4310) [m/s]
_M2005_STOKES_DENOM = 18.0        # ACN = g*rho_w/(18*mu), Stokes droplet fall speed (:1546)


def m2005_rain_psd(q_r, N_r, rho, lamr_min, lamr_max):
    """M2005 rain slope with gSAM's number reset at the slope bounds.

    Port of module_mp_graupel.f90:1638-1650 (identical cold branch
    :2548-2560, post-step :4772-4783).  Where ``q_r >= QSMALL``::

        LAMR = (pi rho_w NR/QR)^(1/3)          (NR per kg = N_r/rho)
        LAMR < LAMMINR or > LAMMAXR:  LAMR = bound,
                                      N0RR = LAMR^4 QR/(pi rho_w), NR = N0RR/LAMR

    Returns ``(lamr, N_r_adj)``; ``N_r`` per-VOLUME [1/m^3] in and out.  Where
    ``q_r < QSMALL`` ``N_r_adj = N_r`` (``lamr`` is a placeholder, unused).
    """
    active = q_r >= _M2005_QSMALL
    qr_s = jnp.where(active, q_r, 1.0)
    lamr = safe_pow(jnp.pi * constants.rho_water * (N_r / rho) / qr_s, 1.0 / 3.0)
    lam_b = jnp.where(lamr < lamr_min, lamr_min, lamr_max)
    clamped = active & ((lamr < lamr_min) | (lamr > lamr_max))
    n_bound = lam_b ** 4 * qr_s / (jnp.pi * constants.rho_water) / lam_b * rho
    return (jnp.where(clamped, lam_b, lamr),
            jnp.where(clamped, n_bound, N_r))


def _m2005_fill_down(v):
    """``F(K) = F(K+1)`` where ``F(K) < 1e-10``, top to bottom (:4282-4310).

    ``v`` is (ncol, nlev) TOP-DOWN, so the level above is ``k-1``; the top level
    keeps its own value; the fill propagates (a filled level feeds the next).
    """
    def step(prev, x):
        cur = jnp.where(x < _M2005_FALL_FILL_MIN, prev, x)
        return cur, cur

    _, rest = jax.lax.scan(step, v[:, 0], v[:, 1:].T)
    return jnp.concatenate([v[:, :1], rest.T], axis=1)


def m2005_liquid_step(T, q_v, q_c, q_r, N_r, p, rho, dz, dt, N_c_spec, params,
                      n_substeps_max, *, sb2001_autoconv_ramp_top,
                      sb2001_accretion_qr_ramp_top, sb2001_accretion_qc_ramp_top):
    """One gSAM M2005 call, liquid only, SB2001 warm rain, specified N_c.

    Port of gSAM 1.8.8 ``M2005MICRO_GRAUPEL`` (SRC/MICRO_M2005/
    module_mp_graupel.f90) as ``micro_proc`` runs it with ``doicemicro=.false.``
    (ILIQ=1), ``dosb_warm_rain=.true.`` (IRAIN=1), ``dopredictNc=.false.``
    (INUM=1, N_c = ``N_c_spec``) and ``dofix_pgam=.false.``; ``ISATADJ=1`` (the
    caller runs :func:`saturation_adjustment_gsam` first, as micro_proc does).
    With every ice species absent, the warm (:1609-2214) and cold (:2525-3790)
    branches compute the same liquid processes (the cold branch's ice-forming
    terms are zeroed by ILIQ=1, :3545-3553), so one path serves every T:

    1. per level: liquid below 1e-8 kg/kg evaporates where q_v/q_sat < 0.9
       (q_sat at the incoming T, :1446-1463), then amounts below QSMALL are
       zeroed (rain number with them);
    2. rain slope with the number reset at its bounds (:func:`m2005_rain_psd`),
       droplet shape and slope (:func:`m2005_cloud_droplet_psd`);
    3. PRC/NPRC1 (:func:`autoconversion_sb2001`), PRA (:func:`accretion_sb2001`),
       NRAGG (:func:`self_collection_breakup_sb2001`), PRE
       (:func:`rain_evaporation_m2005`, q_sat and the psychrometric term from
       step 1, transport properties at the post-removal T);
    4. q_c conservation (PRC, PRA scaled; NPRC1 is NOT, :2135-2141), q_r
       conservation (PRE scaled, :2165-2171), tendencies, NSUBR (:2185-2190);
    5. sedimentation of rain mass+number and cloud mass from the POST-source
       state (:4131-4450): mass- and number-weighted fall speeds (rain capped
       at 9.1 (rho_su/rho)^0.54 m/s; droplets Stokes), filled down below the
       lowest hydrometeor, ``NSTEP = int(max V dt/dz + 1)`` explicit upwind
       sub-steps per species with the speeds held fixed, surface fallout of
       rain AND cloud = precipitation;
    6. state update, then the step-1 removal again on the new state (:4664-
       4679), QSMALL zeroing, and the rain-number reset at the slope bounds.

    ``sb2001_autoconv_ramp_top`` [kg/kg] is the autoconversion gate's RULE 5
    ramp top (:func:`autoconversion_sb2001`; <= 1e-6 = gSAM's hard step);
    ``sb2001_accretion_qr_ramp_top`` / ``sb2001_accretion_qc_ramp_top`` [kg/kg]
    are the accretion gates' (:func:`accretion_sb2001`; <= 1e-8 = hard step).

    ``N_r`` is per-VOLUME [1/m^3] in and out (gSAM's per-kg NR3D times rho).
    Arrays are (ncol, nlev) TOP-DOWN (surface last).  ``params`` supplies the
    gSAM M2005 rain constants (AR, BR, F1R, F2R, LAMMINR/LAMMAXR, RHOSU, the
    NRAGG constants): a ``MorrisonConfig`` at the sam flavour.  The
    data-dependent ``NSTEP`` runs in a fixed ``n_substeps_max`` loop (passes
    past a column's own count are masked); the per-column REQUIRED count is
    returned so the caller can refuse a column that needs more.  Sedimentation
    uses the actual ``dz`` as gSAM does (``dz_floor=False``; the shared
    routine's 1 m floor stays for the MG2-family schemes).

    Returns ``(T, q_v, q_c, q_r, N_r, precipitation [kg/m^2/s], required
    sub-steps (ncol,))``.  Piecewise differentiable (gates are ``jnp.where``
    with safe placeholders).
    """
    def remove_small(T, q_v, q, sub):
        rm = sub & (q < _M2005_SMALL_REMOVAL_Q)
        return (jnp.where(rm, T - q * constants.L_v / constants.c_pd, T),
                jnp.where(rm, q_v + q, q_v), jnp.where(rm, 0.0, q))

    # 1. preamble (:1436-1525); QVQVS is evaluated ONCE, before either removal
    q_sat = saturation_specific_humidity(T, p)
    sub = q_v / q_sat < _M2005_SMALL_REMOVAL_RH
    T, q_v, q_r = remove_small(T, q_v, q_r, sub)
    T, q_v, q_c = remove_small(T, q_v, q_c, sub)
    q_c = jnp.where(q_c < _M2005_QSMALL, 0.0, q_c)
    rain_zero = q_r < _M2005_QSMALL
    q_r = jnp.where(rain_zero, 0.0, q_r)
    N_r = jnp.where(rain_zero, 0.0, N_r)
    _, mu, _ = _air_transport_props(T, p, rho)
    dum54 = safe_pow(params.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXPONENT)
    arn = params.fall_a_r * dum54
    acn = constants.g * constants.rho_water / (_M2005_STOKES_DENOM * mu)

    # 2. size distributions (:1632-1708); N_c specified (INUM=1, :1616)
    N_r = jnp.maximum(0.0, N_r)
    _, N_r = m2005_rain_psd(q_r, N_r, rho, params.lamr_min, params.lamr_max)
    N_c = N_c_spec * jnp.ones_like(q_c)
    _, _, N_c_adj = m2005_cloud_droplet_psd(q_c, N_c, rho)

    # 3. process rates
    prc, nprc1, _ = autoconversion_sb2001(q_c, q_r, N_c, rho,
                                          ramp_top=sb2001_autoconv_ramp_top)
    pra = accretion_sb2001(q_c, q_r, rho, qr_ramp_top=sb2001_accretion_qr_ramp_top,
                           qc_ramp_top=sb2001_accretion_qc_ramp_top)
    nragg = self_collection_breakup_sb2001(N_r, q_r, rho, params)
    pre = -rain_evaporation_m2005(q_v, q_r, N_r, q_sat, T, p, rho, params)

    # 4. conservation (:2135-2171), tendencies (:2175-2190).  gSAM scales PRC
    # and PRA but NOT NPRC1 when the q_c limiter binds (:2135-2141); kept.
    dum = (prc + pra) * dt
    lim = (dum > q_c) & (q_c >= _M2005_QSMALL)
    ratio = q_c / jnp.where(lim, dum, 1.0)
    prc = jnp.where(lim, prc * ratio, prc)
    pra = jnp.where(lim, pra * ratio, pra)
    dum = (-pre - pra - prc) * dt
    lim = (dum > q_r) & (q_r >= _M2005_QSMALL)
    ratio = (q_r / dt + pra + prc) / jnp.where(lim, -pre, 1.0)
    pre = jnp.where(lim, pre * ratio, pre)
    qv_t = -pre
    t_t = pre * constants.L_v / constants.c_pd
    qc_t = -pra - prc
    qr_t = pre + pra + prc
    nr_kg = N_r / rho
    nr_t = nprc1 / rho + nragg / rho
    evap = pre < 0.0
    nsubr = (jnp.maximum(-1.0, pre * dt / jnp.where(evap, q_r, 1.0))
             * nr_kg / dt)
    nr_t = nr_t + jnp.where(evap, nsubr, 0.0)

    # 5. sedimentation from the post-source state (:4131-4450)
    dumr = q_r + qr_t * dt
    dumfnr = jnp.maximum(0.0, nr_kg + nr_t * dt)
    dumc = q_c + qc_t * dt
    on_r = dumr >= _M2005_QSMALL
    dumr_s = jnp.where(on_r, dumr, 1.0)
    dlamr = jnp.clip(
        safe_pow(jnp.pi * constants.rho_water * dumfnr / dumr_s, 1.0 / 3.0),
        params.lamr_min, params.lamr_max)
    cons4 = math.gamma(4.0 + params.fall_b_r) / 6.0
    cons6 = math.gamma(1.0 + params.fall_b_r)
    lam_br = dlamr ** params.fall_b_r
    vcap = _UMR_FALL_CAP * dum54
    umr = jnp.minimum(jnp.where(on_r, arn * cons4 / lam_br, 0.0), vcap)
    unr = jnp.minimum(jnp.where(on_r, arn * cons6 / lam_br, 0.0), vcap)
    on_c = dumc >= _M2005_QSMALL
    dumc_s = jnp.where(on_c, dumc, 1.0)
    pgam, _, _ = m2005_cloud_droplet_psd(dumc_s, N_c_adj, rho)
    lamc, lammin, lammax, _, _ = _m2005_cloud_slope(
        dumc_s, jnp.maximum(0.0, N_c_adj), rho, pgam)
    lamc2 = jnp.clip(lamc, lammin, lammax) ** 2
    # gSAM UMC = ACN*GAMMA(4+BCN+PGAM)/(LAMC**BCN*GAMMA(PGAM+4)), UNC with
    # GAMMA(1+BCN+PGAM)/GAMMA(PGAM+1); Stokes BCN = 2 makes each ratio the
    # exact product below.
    umc = jnp.where(on_c, acn * (pgam + 4.0) * (pgam + 5.0) / lamc2,  # coeff-ok: Gamma(p+6)/Gamma(p+4), BCN=2
                    0.0)
    unc = jnp.where(on_c, acn * (pgam + 1.0) * (pgam + 2.0) / lamc2, 0.0)
    umr, unr, umc, unc = (_m2005_fill_down(v) for v in (umr, unr, umc, unc))
    sed_r, sfc_r, req_r = sedimentation_tendency(
        dumr, rho, umr, dz, dt=dt, return_surface_flux=True,
        n_substeps_max=n_substeps_max, cfl_speed=unr, return_substeps=True,
        dz_floor=False)
    sed_nr = sedimentation_tendency(
        dumfnr, rho, unr, dz, dt=dt, n_substeps_max=n_substeps_max,
        cfl_speed=umr, dz_floor=False)
    sed_c, sfc_c, req_c = sedimentation_tendency(
        dumc, rho, umc, dz, dt=dt, return_surface_flux=True,
        n_substeps_max=n_substeps_max, cfl_speed=unc, return_substeps=True,
        dz_floor=False)

    # 6. update (:4512-4535) and post-step cleanup (:4664-4783)
    q_c = q_c + (qc_t + sed_c) * dt
    q_r = q_r + (qr_t + sed_r) * dt
    nr_kg = nr_kg + (nr_t + sed_nr) * dt
    T = T + t_t * dt
    q_v = q_v + qv_t * dt
    sub = q_v / saturation_specific_humidity(T, p) < _M2005_SMALL_REMOVAL_RH
    T, q_v, q_r = remove_small(T, q_v, q_r, sub)
    T, q_v, q_c = remove_small(T, q_v, q_c, sub)
    q_c = jnp.where(q_c < _M2005_QSMALL, 0.0, q_c)
    rain_zero = q_r < _M2005_QSMALL
    q_r = jnp.where(rain_zero, 0.0, q_r)
    nr_kg = jnp.where(rain_zero, 0.0, jnp.maximum(0.0, nr_kg))
    _, N_r = m2005_rain_psd(q_r, nr_kg * rho, rho, params.lamr_min,
                            params.lamr_max)
    return (T, q_v, q_c, q_r, N_r, sfc_r + sfc_c,
            jnp.maximum(req_r, req_c))
