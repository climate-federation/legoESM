"""Shared warm-rain microphysics helpers.

Functions here are used by multiple microphysics backends (Seifert-Beheng,
Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

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
        ``x ** p`` for ``x > 0``, else 0; gradient is finite
        everywhere.
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


def _hard_saturation_condensation(T, q_v, p_full, dt, q_sat):
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
    l_over_cp = constants.L_v / constants.c_pd
    q_v_pos = jnp.maximum(q_v, 0.0)
    lo = jnp.clip(q_sat, 0.0, q_v_pos)     # f(lo) <= 0 (bracket lower bound)
    hi = q_v_pos                           # f(hi) >= 0 (bracket upper bound)
    for _ in range(_HARD_SAT_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        t_new = T + l_over_cp * (q_v_pos - mid)
        f_mid = mid - saturation_mixing_ratio(t_new, p_full)
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
        lambda t: saturation_mixing_ratio(t, p_full),
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


def _hard_saturation_rate_limit(condensation, q_v, dt, hard_max_heating_K):
    """Bound the per-step POSITIVE condensation (shared by both entry points).

    Caps the per-step condensed ``Delta q_v = cond*dt`` at
    ``hard_max_heating_K * c_pd / L_v`` (per-step latent heating <=
    ``hard_max_heating_K``, so a large pool drains over MANY steps not one) AND
    at the available vapour ``q_v`` (mass-positivity: never condense more than
    the column holds; also guards the tiny-NEGATIVE q_sat the smooth cap returns
    near 150 K).  The evaporation branch (rate < 0) is untouched.  ``minimum``
    has a dead gradient only where a cap binds, where the per-step condensation
    is constant (correctly zero sensitivity).
    """
    dqv_cap = hard_max_heating_K * constants.c_pd / constants.L_v
    cond_cap = jnp.minimum(dqv_cap, jnp.maximum(q_v, 0.0)) \
        / jnp.maximum(dt, 1.0e-10)
    return jnp.where(
        condensation > 0.0, jnp.minimum(condensation, cond_cap), condensation)


def _hard_saturation_blend(condensation, T, q_v, p_full, dt, q_sat,
                           hard_threshold, hard_max_heating_K):
    """Blend the on-curve hard adjustment into ``condensation`` + rate-limit.

    The ONE place the reviewed hard-adjustment logic lives, shared by
    :func:`saturation_adjustment` (in-scheme: ``condensation`` is the smooth
    condensation rate) and :func:`hard_saturation_drain` (driver post-step:
    ``condensation = 0`` -> a pure activation-gated drain).  Steps:

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
            q_sat.astype(jnp.float64),
        ).astype(q_v.dtype)
    else:
        cond_hard = _hard_saturation_condensation(T, q_v, p_full, dt, q_sat)
    rel_humidity = q_v / jnp.maximum(q_sat, _HARD_SAT_QSAT_FLOOR)
    activation = jax.nn.sigmoid(
        _HARD_SAT_MASK_SHARPNESS * (rel_humidity - hard_threshold))
    blended = condensation + activation * (cond_hard - condensation)
    condensation = jnp.minimum(blended, cond_hard)     # no overshoot past curve
    return _hard_saturation_rate_limit(condensation, q_v, dt, hard_max_heating_K)


def hard_saturation_drain(T, q_v, p_full, dt,
                          hard_threshold=_DEFAULT_HARD_SAT_THRESHOLD,
                          hard_max_heating_K=_DEFAULT_HARD_SAT_MAX_HEATING_K):
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
        pressure the caller uses; on the pure-sigma MPAS path this is
        ``p_s * sigma_full``).
    dt : float
        Physics/dycore step [s].
    hard_threshold : float, default 1.1
        RH trigger; drain fires where ``q_v > hard_threshold * q_sat``.
    hard_max_heating_K : float, default 5.0
        Per-step latent-heating cap [K] (~ 2 g/kg per step); a large pool drains
        over many steps rather than detonating.

    Returns
    -------
    array
        Drain rate [kg/kg/s], >= 0.
    """
    if jax.config.jax_enable_x64:
        q_sat = saturation_mixing_ratio(
            T.astype(jnp.float64), p_full.astype(jnp.float64)).astype(q_v.dtype)
        cond_hard = _hard_saturation_condensation(
            T.astype(jnp.float64), q_v.astype(jnp.float64),
            p_full.astype(jnp.float64), dt,
            q_sat.astype(jnp.float64)).astype(q_v.dtype)
    else:
        q_sat = saturation_mixing_ratio(T, p_full)
        cond_hard = _hard_saturation_condensation(T, q_v, p_full, dt, q_sat)
    # HARD RH gate (matches the validated post-step cap): drain ONLY where
    # q_v > hard_threshold * q_sat; no sub-threshold leakage.  ``cond_hard`` is
    # the on-curve drain, so the gated rate never overshoots.
    drain = jnp.where(
        q_v > hard_threshold * jnp.maximum(q_sat, _HARD_SAT_QSAT_FLOOR),
        cond_hard, 0.0)
    return _hard_saturation_rate_limit(drain, q_v, dt, hard_max_heating_K)


def saturation_adjustment(T, q_v, p_full, dt, sharpness=_DEFAULT_SAT_SHARPNESS, q_c=None,
                          hard_adjust=False, hard_threshold=_DEFAULT_HARD_SAT_THRESHOLD,
                          hard_max_heating_K=_DEFAULT_HARD_SAT_MAX_HEATING_K):
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
        q_sat64 = saturation_mixing_ratio(T.astype(jnp.float64),
                                          p_full.astype(jnp.float64))
        excess = (q_v.astype(jnp.float64) - q_sat64).astype(_state_dtype)
        q_sat = q_sat64.astype(_state_dtype)
    else:
        q_sat = saturation_mixing_ratio(T, p_full)
        excess = q_v - q_sat
    dqsdt = constants.L_v * q_sat / (constants.R_v * T ** 2)
    psychrometric = 1.0 + dqsdt * constants.L_v / constants.c_pd
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / (dt * psychrometric)
    if hard_adjust:
        # Iterated hard saturation adjustment, blended in smoothly where
        # q_v > hard_threshold * q_sat, with the per-step latent-heating rate
        # limit.  The reviewed core is shared with the driver post-step hook
        # (:func:`hard_saturation_drain`) via :func:`_hard_saturation_blend`.
        condensation = _hard_saturation_blend(
            condensation, T, q_v, p_full, dt, q_sat,
            hard_threshold, hard_max_heating_K)
    if q_c is not None:
        # Evaporation rate (negative ``condensation``) is bounded by
        # the available cloud water: |condensation| × dt ≤ q_c, i.e.
        # condensation ≥ -q_c / dt.  ``maximum(condensation, -q_c/dt)``
        # achieves this cleanly.  Differentiable everywhere — the
        # clamp is a smooth-ish max on the evaporation magnitude.
        q_c_avail = jnp.clip(q_c, 0.0, None)
        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
    return condensation, q_sat


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
    form wired into Morrison (non-KK2000 path), Seifert-Beheng, Thompson and
    THIS repository's P3; it is a tunable-coefficient SURROGATE, NOT faithful
    to the published Khairoutdinov-Kogan (2000) accretion that gSAM P3 and
    SAM-M2005 use (``module_mp_p3.f90:3615``, iparam=3 /
    ``module_mp_graupel.f90:1952``) — this repo's P3 instead calls THIS
    surrogate.  The gSAM form, in a single column with the sub-grid
    cloud/precip-fraction scheme off (``iSCF=iSPF=1``, ``dum2=1``), reduces to
    the mass rate::

        PRA = 67 · (q_c · q_r)^1.15          [kg/kg/s]   (the faithful form)

    available as :func:`accretion_kk2000`.  DEPARTURES of this surrogate from
    that oracle: (1) coefficient ``k_ac`` (default 5.25) vs the published 67;
    (2) BILINEAR ``q_c·q_r`` (product exponent 1) vs KK2000's ``(q_c·q_r)^1.15``;
    (3) a spurious ``rho`` factor — KK2000 is a pure mixing-ratio rate with NO
    density dependence; (4) an optional ``gamma_norm`` PSD-shape modifier absent
    from KK2000 (P3 passes the default ``1.0``, so P3 does not incur (4)).
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
# Cloud-droplet gamma shape nu for the (nu+2)(nu+4)/(nu+1)^2 factor (gSAM :1839).
# gSAM's DEFAULT (dofix_pgam=.false.) DIAGNOSES pgam per level from N_c
# (Martin et al. 1994, module_mp_graupel.f90:1670-1680), clamps pgam to [2,10],
# then interpolates nu from the dnu lookup table (:1685-1687) -> nu in
# [dnu(2), dnu(10)] = [-0.557, 0.397]. Its FIXED-pgam mode uses pgam_fixed=10.3
# (Geoffroy et al. 2010; micro_params.f90:41) -> nu = dnu(10)+0.3·(dnu(11)−dnu(10))
# = 0.397 + 0.3·(0.512−0.397) = 0.4315. This module default is that fixed-pgam
# value: an EXPLICIT fixed-shape approximation (NOT gSAM's spatially-diagnosed
# default). Pass ``nu`` explicitly to supply a diagnosed/per-column shape.
_SB2001_NU_CLOUD = 0.4315
_SB2001_KCR = 5.78e3                 # accretion kernel k_cr [cm^3 g^-1 s^-1] (gSAM :1962)
# 5.78e3·rho/1000 = 5.78·rho (gSAM :1962); the /1000 is the CGS mass-density
# conversion folded into the coefficient.
_SB2001_ACCRETION_COEFF = _SB2001_KCR / 1000.0
_SB2001_PHI_AC_KTAU = 5.0e-4         # phi_ac(tau) = (tau/(tau+k_tau))^4 knee (gSAM :1961)
# Unit conversions for the CGS SB2001 mass rate (gSAM works in g/cm^3, #/cm^3):
_LWC_KGM3_TO_GCM3 = 1.0e-3           # rho·q_c [kg/m^3] -> LWC [g/cm^3] (gSAM ·/1000)
_NC_PERM3_TO_PERCM3 = 1.0e-6         # N_c [1/m^3] -> [1/cm^3] (gSAM ·/1e6; per-VOLUME)
_MASSRATE_GCM3_TO_MIXR = 1.0e3       # g/cm^3/s -> kg/m^3/s (then /rho -> kg/kg/s)

# --- shared PSD / fall-speed / ventilation structural constants (SAM M2005) ---
_RHO_FLOOR = 0.1                 # air-density floor in PSD/fall-speed divisions [kg/m^3]
_FALL_RHO_EXPONENT = 0.54        # (rho_su/rho)^0.54 fall-speed density correction
_VENT_CONS_OFFSET = 2.5          # 5/2 in ventilation gamma argument 5/2 + b/2
_BIGG_MNUCCR_PREFACTOR = 20.0    # Bigg freezing mass prefactor (20 pi^2 rho_w)
_UMR_FALL_CAP = 9.1              # rain mass-weighted fall-speed cap [m/s]
_UMG_FALL_CAP = 20.0             # graupel mass-weighted fall-speed cap [m/s]
_VDIFF_C1, _VDIFF_C2, _VDIFF_C3 = 1.2, 0.95, 0.08  # Wisner two-PSD VDIFF coeffs
_PRACG_BRACKET_C = 5.0           # PRACG collection-integral leading bracket coeff
_NSAGG_CONS15_PREFACTOR = 1108.0  # snow self-aggregation CONS15 prefactor
_NSAGG_GAMMA_DENOM = 720.0       # snow self-aggregation gamma denominator (4*720)
_RAIN_EVAP_VENT_EXP = 0.525      # Marshall-Palmer rain-evaporation ventilation exponent



def autoconversion_kk2000(q_c, N_c_eff, rho, dt):
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
    prc = (
        _KK2000_AUTOCONV_PREFACTOR
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


def accretion_kk2000(q_c, q_r):
    """Khairoutdinov–Kogan (2000) warm-rain accretion — SAM M2005
    (``module_mp_graupel.f90:1952``; identical to gSAM P3
    ``module_mp_p3.f90:3615``, iparam=3):

        PRA = 67 · (q_c · q_r)^1.15      [kg/kg/s]

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
    return _KK2000_ACCRETION_PREFACTOR * safe_pow(dum, _KK2000_ACCRETION_EXPONENT)


def autoconversion_sb2001(q_c, q_r, N_c_eff, rho, nu=_SB2001_NU_CLOUD):
    """Seifert & Beheng (2001) autoconversion — the PUBLISHED universal-function
    closed form (gSAM M2005 IRAIN=1, module_mp_graupel.f90:1835-1844)::

        tau    = 1 − q_c/(q_c+q_r)   = q_r/(q_c+q_r)      (rain-water fraction)
        phi_au = 600·tau^0.68·(1 − tau^0.68)^3            (universal function)
        PRC    = k_cc/(20·x_*)·(nu+2)(nu+4)/(nu+1)^2
                 ·(rho·q_c/1000)^4 / (N_c/1e6)^2
                 ·(1 + phi_au/(1−tau)^2)·1000/rho          [kg/kg/s]

    Unlike the simplified :func:`autoconversion_sb` (q_c^2·mean-mass sigmoid),
    this carries the full SB2001 q_c^4·N_c^-2 dependence and the phi_au
    universal function. tau = q_r/(q_c+q_r) is the rain-water fraction (= gSAM's
    ``dum``): tau→0 is CLOUD-dominated, tau→1 is RAIN-dominated. The extra
    ``phi_au/(1−tau)^2`` enhancement vanishes at both endpoints and is strongest
    at intermediate rain fraction.

    ``N_c_eff`` is per-VOLUME [1/m^3] (this module's convention); gSAM's nc3d is
    per-MASS so its ``rho·nc/1e6`` becomes ``N_c/1e6`` here — see
    :func:`autoconversion_kk2000`. ``nu`` is the cloud-droplet gamma shape
    (default :data:`_SB2001_NU_CLOUD`; see its provenance).

    FAITHFUL SCOPE: this reproduces the SB2001 MASS rate (PRC) and the rain-
    NUMBER source (dN_r_au = au/x_*). It does NOT reproduce gSAM's SB2001-SPECIFIC
    cloud-NUMBER autoconversion sink ``NPRC = 2·PRC·rho/x_*`` (:1843): under
    ``predict_Nc=True`` Morrison instead applies its GENERIC sink ``-PRC·rho/x_c``
    (mean-mass x_c, not the x_* separation mass and without the factor 2 — see
    morrison.py). Moot at the default ``predict_Nc=False`` (N_c not evolved).

    Returns ``(dq_c_au, dN_r_au, x_c)`` matching :func:`autoconversion_sb`'s
    signature. AD: tau^0.68 (unbounded slope at tau=0) is regularised via
    ``safe_pow`` (zero surrogate gradient at tau≤0, NOT the divergent one-sided
    SB derivative); the q_c+q_r, N_c and (1−tau) denominators are floored; the
    rate is masked to zero where there is no cloud water so ``jax.grad`` is
    FINITE (a surrogate, not the analytical boundary gradient) at q_c=0.
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    q_r_pos = jnp.clip(q_r, 0.0)
    q_tot = q_c_pos + q_r_pos
    # tau = rain fraction in [0,1] (gSAM ``dum``); zero rate where no condensate.
    tau = jnp.where(q_tot > 1.0e-20, q_r_pos / jnp.maximum(q_tot, 1.0e-20), 0.0)
    tau_pow = safe_pow(tau, _SB2001_PHI_AU_TAU_EXP)
    phi_au = _SB2001_PHI_AU_PREFACTOR * tau_pow * (1.0 - tau_pow) ** 3
    lwc_gcm3 = rho * q_c_pos * _LWC_KGM3_TO_GCM3            # g/cm^3
    nc_cm3 = jnp.clip(N_c_eff, 1.0) * _NC_PERM3_TO_PERCM3    # #/cm^3
    shape = (nu + 2.0) * (nu + 4.0) / (nu + 1.0) ** 2
    # (1−tau) = q_c/(q_c+q_r); floored so the phi_au/(1−tau)^2 enhancement is
    # AD-finite at q_c=0 (there phi_au→0 as (1−tau)^3, so the ratio →0 anyway).
    one_minus_tau = jnp.maximum(1.0 - tau, 1.0e-20)
    enhance = 1.0 + phi_au / one_minus_tau ** 2
    dq_c_au = (
        _SB2001_AUTOCONV_MASS_COEFF * shape
        * lwc_gcm3 ** 4 / nc_cm3 ** 2
        * enhance * _MASSRATE_GCM3_TO_MIXR / jnp.maximum(rho, _RHO_FLOOR)
    )
    dq_c_au = jnp.where(q_c_pos > 0.0, dq_c_au, 0.0)
    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
    # SB2001 number closure: newborn rain drops carry the separation mass x_*
    # (gSAM nprc1, :1843-1844; per-VOLUME here ⇒ ·rho). Identical form to
    # :func:`autoconversion_sb`'s ``dq_c_au·rho/x_star``.
    dN_r_au = dq_c_au * rho / _SB2001_X_STAR_KG
    return dq_c_au, dN_r_au, x_c


def accretion_sb2001(q_c, q_r, rho):
    """Seifert & Beheng (2001) accretion — the PUBLISHED universal-function form
    (gSAM M2005 IRAIN=1, module_mp_graupel.f90:1960-1962)::

        tau    = 1 − q_c/(q_c+q_r) = q_r/(q_c+q_r)
        phi_ac = (tau/(tau + 5e-4))^4                      (universal function)
        PRA    = 5.78e3·rho/1000·q_c·q_r·phi_ac
               = 5.78·rho·q_c·q_r·phi_ac                    [kg/kg/s]

    Unlike the simplified :func:`accretion` (bilinear ``k_ac·rho·q_c·q_r``, no
    tau dependence), this carries the SB2001 phi_ac(tau) universal function that
    suppresses accretion until rain water is present (tau>0) and the fixed
    published kernel 5.78 (vs the tunable ``k_ac``). AD-safe: the tau knee
    denominator is a strictly-positive constant offset.

    FAITHFUL SCOPE: this is the SB2001 MASS accretion rate (PRA). gSAM's
    cloud-NUMBER accretion sink ``NPRA = PRA·N_c/q_c`` (:1963) is applied by NO
    Morrison warm-rain path (kk2000 or SB), and is moot at the default
    ``predict_Nc=False``.
    """
    q_c_pos = jnp.clip(q_c, 0.0)
    q_r_pos = jnp.clip(q_r, 0.0)
    q_tot = q_c_pos + q_r_pos
    tau = jnp.where(q_tot > 1.0e-20, q_r_pos / jnp.maximum(q_tot, 1.0e-20), 0.0)
    phi_ac = (tau / (tau + _SB2001_PHI_AC_KTAU)) ** 4
    return _SB2001_ACCRETION_COEFF * rho * q_c_pos * q_r_pos * phi_ac


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
        X      = exp(AIMM·(T₀−T)) − 1          (T<T₀ only; AIMM=0.66/K, BIMM=100)
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
    gate = (qr_pos > 1.0e-14) & (dT_sc > 0.0)            # supercooled rain only
    mnuccr = jnp.where(gate, mnuccr, 0.0)
    nnuccr = jnp.where(gate, nnuccr, 0.0)
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
    evap = jnp.where(qr_pos > 1.0e-14, evap, 0.0)          # SAM QSMALL skip
    if dt is not None:
        evap = jnp.minimum(evap, qr_pos / jnp.maximum(dt, 1.0e-12))
    return evap
