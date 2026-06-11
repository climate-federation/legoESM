"""CLUBB skewness diagnostics (``Skx_module``) — gamma(Skw), Skx, LG05 ansatz.

Faithful port of ``CLUBB-JAX/.../Skx_module.py`` (mirror of ``Skx_module.F90``).
Pure, differentiable JAX. These small kernels are consumed widely:
  * ``compute_gamma_Skw`` -> ``gamma_Skw_fnc`` feeding
    :func:`clubb_helpers.compute_sigma_sqd_w` and the PDF closure;
  * ``Skx_func`` / ``LG_2005_ansatz`` / ``xp3_LG_2005_ansatz`` -> the assumed-PDF
    skewness closure and the (diagnostic) ``xp3`` ansatz used when
    ``l_advance_xp3 = .false.`` (the CAM default).

Adapted to legoESM conventions: the tunable coefficients are passed as named
scalars (from :class:`CLUBBParams`) rather than indexed out of CLUBB's
102-element ``clubb_params`` vector. ``l_gamma_Skw`` is a static model flag
(``model_flags.F90`` constant ``True``; not in the CAM namelist) and gates a
Python branch.

References
----------
Larson, V. E., & Golaz, J.-C. (2005). Using probability density functions to
derive consistent closure relationships among higher-order moments. Mon. Wea.
Rev., 133, 1023-1042 (eqs. 11, 16, 33).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, zm2zt, zt2zm

# CLUBB ``eps`` = max(1e-10, machine-eps); used only in the degenerate-gamma
# guard below. A safety tolerance, not a physical constant.
_EPS = 1.0e-10
_WP3_ON_WP2_CLIP = 1000.0   # bound on the wp3/wp2 ratio (calc_wp3_on_wp2)


def Skx_func(
    xp2: jax.Array,
    xp3: jax.Array,
    x_tol: float,
    Skw_denom_coef: float,
) -> jax.Array:
    """Skewness of ``x`` with the LG05 sensitivity-reduction denominator.

    ``Skx = xp3 * (xp2 + Skw_denom_coef * x_tol^2)^(-3/2)``
    (``Skx_module.F90:Skx_func``). With the CAM default ``Skw_denom_coef = 0``
    this reduces to ``xp3 / xp2^(3/2)``.

    Parameters
    ----------
    xp2, xp3 : jax.Array
        Second and third moments of ``x``.
    x_tol : float
        Tolerance for ``x`` (e.g. ``w_tol``/``rt_tol``/``thl_tol``).
    Skw_denom_coef : float
        Sensitivity-reduction coefficient (``CLUBBParams.Skw_denom_coef``).

    Returns
    -------
    jax.Array
        Skewness of ``x``.
    """
    denom_tol = Skw_denom_coef * x_tol ** 2
    return xp3 * (xp2 + denom_tol) ** (-1.5)


def compute_gamma_Skw(
    Skw: jax.Array,
    gamma_coef: float,
    gamma_coefb: float,
    gamma_coefc: float,
    l_gamma_Skw: bool = True,
) -> jax.Array:
    """Gamma coefficient as a Gaussian function of w-skewness.

    ``Skx_module.F90:compute_gamma_Skw``. With ``l_gamma_Skw`` on and the two
    coefficients meaningfully different::

        gamma = gamma_coefb + (gamma_coef - gamma_coefb)
                              * exp(-0.5 * (Skw / gamma_coefc)^2)

    otherwise ``gamma = gamma_coef`` (constant). The degenerate-coefficient
    branch is data-independent (depends only on the coefficients), so it is a
    ``jnp.where`` rather than a Python ``if`` — keeping the coefficients
    differentiable. ``l_gamma_Skw`` is a static model flag (Python ``if``).

    Parameters
    ----------
    Skw : jax.Array
        Skewness of w (zm or zt levels), shape ``(ngrdcol, nz)``.
    gamma_coef, gamma_coefb, gamma_coefc : float
        Tunable gamma coefficients (``CLUBBParams``).
    l_gamma_Skw : bool, default True
        Static flag; when False, returns the constant ``gamma_coef``.

    Returns
    -------
    jax.Array
        ``gamma_Skw_fnc`` with ``Skw``'s shape.
    """
    if not l_gamma_Skw:
        return gamma_coef + jnp.zeros_like(Skw)
    gc = jnp.asarray(gamma_coef)
    gb = jnp.asarray(gamma_coefb)
    gcf = jnp.asarray(gamma_coefc)
    cond = jnp.abs(gc - gb) > jnp.abs(gc + gb) * _EPS / 2.0
    varying = gb + (gc - gb) * jnp.exp(-0.5 * (Skw / gcf) ** 2)
    return jnp.where(cond, varying, gc + jnp.zeros_like(Skw))


def LG_2005_ansatz(
    Skw: jax.Array,
    wpxp: jax.Array,
    wp2: jax.Array,
    xp2: jax.Array,
    sigma_sqd_w: jax.Array,
    beta: float,
    x_tol: float,
    w_tol: float,
) -> jax.Array:
    """Skewness of ``x`` from skewness of ``w`` (LG05 eqs. 11, 16, 33).

    ``Skx_module.F90:LG_2005_ansatz``.

    Parameters
    ----------
    Skw : jax.Array
        Skewness of w.
    wpxp, wp2, xp2 : jax.Array
        ``w'x'`` flux, w-variance, x-variance.
    sigma_sqd_w : jax.Array
        PDF width parameter (< 1).
    beta : float
        Tunable LG05 coefficient (``CLUBBParams.beta``).
    x_tol, w_tol : float
        Tolerances for ``x`` and ``w`` (floors on the variances).

    Returns
    -------
    jax.Array
        Skewness of ``x``.
    """
    one_minus_ssw = 1.0 - sigma_sqd_w
    nrmlzd_corr_wx = wpxp / jnp.sqrt(
        jnp.maximum(wp2, w_tol ** 2) * jnp.maximum(xp2, x_tol ** 2) * one_minus_ssw
    )
    nrmlzd_Skw = Skw / (one_minus_ssw * jnp.sqrt(one_minus_ssw))
    return nrmlzd_Skw * nrmlzd_corr_wx * (beta + (1.0 - beta) * nrmlzd_corr_wx ** 2)


def xp3_LG_2005_ansatz(
    Skw_zt: jax.Array,
    wpxp_zt: jax.Array,
    wp2_zt: jax.Array,
    xp2_zt: jax.Array,
    sigma_sqd_w_zt: jax.Array,
    beta: float,
    x_tol: float,
    w_tol: float,
    Skw_denom_coef: float,
) -> jax.Array:
    """``<x'^3>`` from the LG05 skewness ansatz (inverse of :func:`Skx_func`).

    ``Skx_module.F90:xp3_LG_2005_ansatz``: ``xp3 = Skx * (xp2 + denom_tol)^(3/2)``
    with ``Skx`` from :func:`LG_2005_ansatz`. Used to diagnose ``xp3`` when
    ``l_advance_xp3 = .false.`` (CAM default).
    """
    Skx_denom_tol = Skw_denom_coef * x_tol ** 2
    Skx_zt = LG_2005_ansatz(
        Skw_zt, wpxp_zt, wp2_zt, xp2_zt, sigma_sqd_w_zt, beta, x_tol, w_tol
    )
    xp2_safe = xp2_zt + Skx_denom_tol
    return Skx_zt * xp2_safe * jnp.sqrt(xp2_safe)


def calc_wp3_on_wp2(wp2, wp3, w_tol, gr: CLUBBGrid):
    """Smoothed ``wp3/wp2`` ratio on zm and zt levels (``calc_wp3_on_wp2``).

    ``wp2`` is floored to ``w_tol^2`` on zt, the ratio clipped to ``[-1000,
    1000]``, then round-tripped zt->zm->zt to suppress spikes. ``wp2`` is
    zm-level, ``wp3`` zt-level. Returns ``(wp3_on_wp2, wp3_on_wp2_zt)``.
    """
    w_tol_sqd = w_tol ** 2
    wp2_zt = jnp.maximum(zm2zt(wp2, gr), w_tol_sqd)
    wp3_on_wp2_zt = jnp.clip(wp3 / jnp.maximum(wp2_zt, w_tol_sqd),
                             -_WP3_ON_WP2_CLIP, _WP3_ON_WP2_CLIP)
    wp3_on_wp2 = zt2zm(wp3_on_wp2_zt, gr)
    wp3_on_wp2_zt = zm2zt(wp3_on_wp2, gr)
    return wp3_on_wp2, wp3_on_wp2_zt


def compute_skewness_diagnostics(wp2, wp3, w_tol, Skw_denom_coef, gr: CLUBBGrid):
    """Skewness + wp3/wp2-ratio diagnostics for the moment advances.

    Assembles ``Skw`` on both grids (:func:`Skx_func`) and the smoothed
    ``wp3_on_wp2`` ratio (:func:`calc_wp3_on_wp2`) from the carried ``wp2`` (zm)
    and ``wp3`` (zt). Returns a dict with ``Skw_zm``, ``Skw_zt``, ``wp2_zt``
    (floored), ``wp3_zm``, ``wp3_on_wp2``, ``wp3_on_wp2_zt`` — the diagnostics the
    wp2/wp3, xp2/xpyp and xm/wpxp advances consume.
    """
    w_tol_sqd = w_tol ** 2
    wp2_zt = jnp.maximum(zm2zt(wp2, gr), w_tol_sqd)
    wp3_zm = zt2zm(wp3, gr)
    Skw_zt = Skx_func(wp2_zt, wp3, w_tol, Skw_denom_coef)
    Skw_zm = Skx_func(wp2, wp3_zm, w_tol, Skw_denom_coef)
    wp3_on_wp2, wp3_on_wp2_zt = calc_wp3_on_wp2(wp2, wp3, w_tol, gr)
    return dict(Skw_zm=Skw_zm, Skw_zt=Skw_zt, wp2_zt=wp2_zt, wp3_zm=wp3_zm,
                wp3_on_wp2=wp3_on_wp2, wp3_on_wp2_zt=wp3_on_wp2_zt)


__all__ = ["Skx_func", "compute_gamma_Skw", "LG_2005_ansatz", "xp3_LG_2005_ansatz",
           "calc_wp3_on_wp2", "compute_skewness_diagnostics"]
