"""Lightweight saturation thermodynamics for legoESM.

This module provides `saturation_mixing_ratio` and
`saturation_mixing_ratio_ice` with *no* dependency on
``atmosphere.physics`` so that ``land/``, ``ice/``, ``ocean/``, and
``coupler/`` modules can import them without pulling in the full
atmosphere physics package.

All operations are pure JAX and compatible with jit, grad, vmap, scan.

Conventions — water-vapor mass variables
----------------------------------------
This module returns the **mixing ratio** ``r_sat = ε e_sat / (p - e_sat)``
(mass of water vapor per unit mass of *dry* air).  Throughout the
``atmosphere/physics`` source tree the prognostic field is named
``q_v`` and many docstrings call it "specific humidity".  In the
typical atmospheric regime where ``e_sat ≪ p``, mixing ratio and
specific humidity differ by ``q ≈ r / (1 + r)`` — about 1% for
``r = 0.01``.  The codebase uses these interchangeably; physics that
needs the distinction (vertical-flux conservation in saturated tropical
columns, q_c bookkeeping) should read this caveat carefully and
convert explicitly when the 1% drift matters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def saturation_vapor_pressure(T: jax.Array) -> jax.Array:
    """Compute saturation vapor pressure using Tetens formula.

    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))

    Parameters
    ----------
    T : jax.Array
        Temperature [K].

    Returns
    -------
    jax.Array
        Saturation vapor pressure [Pa].
    """
    T_c = T - constants.T_freeze
    return 611.2 * jnp.exp(17.67 * T_c / (T_c + 243.5))


def saturation_mixing_ratio(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio using Tetens formula.

    e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))   where T_c = T - 273.15
    q_sat = epsilon * e_sat / (p - e_sat)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation mixing ratio [kg/kg].
    """
    e_sat = saturation_vapor_pressure(T)
    # Smooth floor on denominator: preserves gradients near e_sat ≈ p
    # instead of a hard clip that creates a zero-gradient plateau.
    # softplus(x - 1) + 1 ≈ x for x >> 1, ≈ 1 for x << 1, smooth at x = 1.
    denom = jax.nn.softplus(p - e_sat - 1.0) + 1.0
    q_sat = constants.epsilon * e_sat / denom
    # Smooth cap at 1.0 kg/kg: prevents singularity at low-pressure levels
    # while allowing gradients to flow (unlike hard jnp.minimum).
    # Uses LogSumExp smooth-min: 1 - softplus(β(1 - x))/β with β = 20.
    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat)) / 20.0


def saturation_mixing_ratio_ice(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation mixing ratio over ice (Clausius-Clapeyron).

    e_sat_i = 611.2 * exp(L_s/R_v * (1/T_freeze - 1/T))
    q_sat_i = epsilon * e_sat_i / (p - e_sat_i)

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Ice saturation mixing ratio [kg/kg].
    """
    e_sat_i = 611.2 * jnp.exp(
        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
    )
    denom = jax.nn.softplus(p - e_sat_i - 1.0) + 1.0
    q_sat_i = constants.epsilon * e_sat_i / denom
    return 1.0 - jax.nn.softplus(20.0 * (1.0 - q_sat_i)) / 20.0


def saturation_mixing_ratio_blend(
    T: jax.Array,
    p: jax.Array,
    T_blend_top: float | None = None,
    T_blend_width: float = 20.0,
) -> jax.Array:
    """FV3_3D iter 720: saturation mixing ratio with liquid/ice blend.

    Reusable helper matching FV3's ``compute_qs(..., es_over_liq_and_ice=
    .true.)`` behaviour:

        w_liq  = clip((T − (T_top − width)) / width, 0, 1)
        q_sat  = w_liq · q_sat_liq + (1 − w_liq) · q_sat_ice

    Defaults:
        T_blend_top = ``constants.T_freeze``     (273.15 K)
        T_blend_width = 20.0 K

    Generalizes the inline blend in ``rh_calc_fv3 do_cmip=True``
    (iter-715) for reuse by other diagnostics.

    Parameters
    ----------
    T : jax.Array
        Temperature (K).
    p : jax.Array
        Pressure (Pa).
    T_blend_top : float, optional
        Upper temperature above which q_sat = q_sat_liq exactly.
        Default ``constants.T_freeze``.
    T_blend_width : float, default 20.0 K.
        Linear-blend width.

    Returns
    -------
    q_sat : jax.Array
        Blended saturation mixing ratio (kg/kg).
    """
    if T_blend_top is None:
        T_blend_top = constants.T_freeze
    T_blend_bot = T_blend_top - T_blend_width
    qs_liq = saturation_mixing_ratio(T, p)
    qs_ice = saturation_mixing_ratio_ice(T, p)
    w_liq = jnp.clip(
        (T - T_blend_bot) / (T_blend_top - T_blend_bot),
        0.0, 1.0,
    )
    return w_liq * qs_liq + (1.0 - w_liq) * qs_ice


def saturation_mixing_ratio_dT(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Analytic derivative d(q_sat)/dT consistent with ``saturation_mixing_ratio``.

    Uses the same Tetens vapor-pressure formula as
    ``saturation_vapor_pressure`` and the same hard ``p - e_sat`` floor as
    historically used by closure schemes (CLUBB-style PDF widths).  The
    smooth softplus floor used by ``saturation_mixing_ratio`` itself
    is intentionally NOT applied here — for derivative use cases
    (e.g. Gaussian PDF width scaling), the simpler ``max(p - e_sat, 1)``
    floor is the standard convention.

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        d(q_sat)/dT [kg/kg/K].
    """
    e_sat = saturation_vapor_pressure(T)
    T_c = T - constants.T_freeze
    de_dT = e_sat * 17.67 * 243.5 / (T_c + 243.5) ** 2
    p_eff = jnp.clip(p - e_sat, 1.0)
    return constants.epsilon * de_dT * p / p_eff ** 2


def saturation_specific_humidity(
    T: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute saturation specific humidity from saturation mixing ratio.

    q = w_sat / (1 + w_sat)

    where w_sat = epsilon * e_sat / (p - e_sat) is the saturation mixing
    ratio.  Use this function when working with specific humidity fields
    (q = m_v / (m_v + m_d)) rather than mixing ratio (w = m_v / m_d).

    Parameters
    ----------
    T : jax.Array
        Temperature [K].
    p : jax.Array
        Pressure [Pa].

    Returns
    -------
    jax.Array
        Saturation specific humidity [kg/kg].
    """
    w_sat = saturation_mixing_ratio(T, p)
    return w_sat / (1.0 + w_sat)
