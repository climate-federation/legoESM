"""Shared warm-rain microphysics helpers.

Functions here are used by multiple microphysics backends (Seifert-Beheng,
Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio


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


def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
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
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt
    if q_c is not None:
        # Evaporation rate (negative ``condensation``) is bounded by
        # the available cloud water: |condensation| × dt ≤ q_c, i.e.
        # condensation ≥ -q_c / dt.  ``maximum(condensation, -q_c/dt)``
        # achieves this cleanly.  Differentiable everywhere — the
        # clamp is a smooth-ish max on the evaporation magnitude.
        q_c_avail = jnp.clip(q_c, 0.0, None)
        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
    return condensation, q_sat


def effective_Nc(N_c, Nc_0):
    """Use config default cloud droplet number where N_c is zero.

    Parameters
    ----------
    N_c : array
        Cloud droplet number concentration [1/m³] (Seifert-Beheng
        per-volume convention; see Notes).
    Nc_0 : float
        Default cloud droplet number [1/m³].  Typical values:
        ``1e8`` /m³ maritime, ``1e9`` /m³ continental.

    Returns
    -------
    array : Effective N_c.

    Notes
    -----
    **Convention**: this module internally uses the Seifert-Beheng
    per-volume (`[1/m³]`) convention for cloud droplet number — the
    formulas ``x_c = q_c * rho / N_c`` and ``dN_r_au = dq_c_au * rho
    / (x_star * 20)`` rely on it (audit Codex cycle 2).  An earlier
    docstring labeled ``N_c`` as `[1/kg]` (per-mass), which conflicts
    with the formulas: a per-mass ``N_c`` would give ``x_c`` in
    `[kg²/m³]` rather than `[kg]`, breaking the comparison against
    ``x_star = 2.6e-10 kg``.  The default ``Nc_0 = 1e8`` is the
    canonical maritime per-volume value.
    """
    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))


def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
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
    dN_r_au = dq_c_au * rho / (x_star * 20.0)
    return dq_c_au, dN_r_au, x_c


def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
    """Rain collecting cloud water (accretion).

    Parameters
    ----------
    q_c, q_r : array
        Cloud water and rain mixing ratios [kg/kg].
    rho : array
        Air density [kg/m3].
    k_ac : float
        Accretion rate constant.
    gamma_norm : float
        Gamma distribution correction.

    Returns
    -------
    array : Accretion rate [kg/kg/s].
    """
    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm


def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
    """Self-collection and breakup of rain drops.

    Parameters
    ----------
    N_r : array
        Rain drop number concentration [1/kg].
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
    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
    dN_r_br = -dN_r_sc * breakup_frac
    return dN_r_sc, dN_r_br


def rain_evaporation(q_v, q_r, q_sat, evap_coeff, dt=None):
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

    Returns
    -------
    array : Evaporation rate [kg/kg/s].
    """
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    q_r_pos = jnp.clip(q_r, 0.0, None)
    # Marshall-Palmer ventilation factor q_r^0.525 — fractional power has
    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
    rate = evap_coeff * subsaturation * safe_pow(q_r_pos, 0.525)
    if dt is not None:
        max_rate = q_r_pos / jnp.maximum(dt, 1.0e-12)
        rate = jnp.minimum(rate, max_rate)
    return rate
