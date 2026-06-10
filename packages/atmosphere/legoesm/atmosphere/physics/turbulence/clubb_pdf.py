"""CLUBB ADG1 assumed-PDF parameter closure (CAM default ``iiPDF_ADG1``).

Faithful port of the ADG1 branch of
``CLUBB-JAX/.../adg1_adg2_3d_luhar_pdf.py`` (mirror of
``adg1_adg2_3d_luhar_pdf.F90``). Given the mean state and second/third moments,
ADG1 closes the joint PDF of ``(w, rt, thl, u, v)`` as a **double Gaussian**:
``w`` carries the skewness (two components with means ``w_1 > wm > w_2``,
mixture fraction ``mixt_frac``); each "responder" ``x`` (rt/thl/u/v) is a
bi-normal whose component means/variances are set by the ``w``-``x`` covariance.
These component parameters are the input to the PDF moment integrals (cloud
fraction, buoyancy flux ``wpthvp``, higher-order moments) ported later.

CAM uses ``iiPDF_type = iiPDF_ADG1 (1)``, so only the ADG1 path is ported here
(the ADG2 / 3D-Luhar / new-PDF drivers are out of the CAM-default tree).

These kernels use NO physical constants — only the tunable ``beta`` and the
derived ``mixt_frac_max_mag`` cap (both from :mod:`clubb_config`) and a zero
floor — so the port is bit-exact to the reference (verified by a golden parity
test). All arrays are on thermodynamic (zt) levels, shape ``(ngrdcol, nzt)``.

References
----------
Golaz, J.-C., Larson, V. E., & Cotton, W. R. (2002). A PDF-based model for
boundary layer clouds. Part I. J. Atmos. Sci., 59, 3540-3551.
Larson, V. E., & Golaz, J.-C. (2005). MWR, 133, 1023-1042.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_ZERO = 0.0
_SKW_TOL = 1.0e-5   # |Skw| below which mixt_frac is pinned to 0.5


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(max(x,0))`` with a finite (0) gradient at ``x<=0`` (double-where)."""
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, jnp.sqrt(safe), 0.0)


def ADG1_w_closure(wm, wp2, Skw, sigma_sqd_w, sqrt_wp2, mixt_frac_max_mag):
    """Mixture fraction and w PDF component parameters (``ADG1_w_closure``).

    Returns ``(w_1, w_2, w_1_n, w_2_n, varnce_w_1, varnce_w_2, mixt_frac)``.
    ``w_1_n``/``w_2_n`` are the normalized component means
    (``w_i = wm + sqrt_wp2 * w_i_n``). By construction the two Gaussians
    reproduce ``wm``, ``wp2``, and ``Skw`` exactly.

    Parameters
    ----------
    wm, wp2, Skw, sigma_sqd_w, sqrt_wp2 : jax.Array
        Mean w, w-variance, w-skewness, PDF width parameter, ``sqrt(wp2)``.
    mixt_frac_max_mag : float
        Cap on the mixture fraction (``clubb_config.derive_mixt_frac_max_mag``).
    """
    denom_sq = 4.0 * (1.0 - sigma_sqd_w) ** 3 + Skw ** 2
    mf_formula = 0.5 * (1.0 - Skw / jnp.sqrt(denom_sq))
    mixt_frac = jnp.where(jnp.abs(Skw) <= _SKW_TOL, 0.5, mf_formula)
    mixt_frac = jnp.clip(mixt_frac, 1.0 - mixt_frac_max_mag, mixt_frac_max_mag)

    one_minus_mf = 1.0 - mixt_frac
    sigma_factor = 1.0 - sigma_sqd_w
    # _safe_sqrt: 1-sigma_sqd_w -> 0 in well-mixed/surface layers -> bare sqrt
    # has an inf reverse-mode gradient there (forward-identical, arg >= 0).
    w_1_n = _safe_sqrt(one_minus_mf / mixt_frac * sigma_factor)
    w_2_n = -_safe_sqrt(mixt_frac / one_minus_mf * sigma_factor)

    w_1 = wm + sqrt_wp2 * w_1_n
    w_2 = wm + sqrt_wp2 * w_2_n
    varnce_w_1 = sigma_sqd_w * wp2
    varnce_w_2 = sigma_sqd_w * wp2
    return w_1, w_2, w_1_n, w_2_n, varnce_w_1, varnce_w_2, mixt_frac


def ADG1_ADG2_responder_params(xm, xp2, wp2, sqrt_wp2, wpxp,
                               w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta):
    """Bi-normal component params for a responder ``x`` (rt/thl/u/v).

    ``ADG1_ADG2_responder_params``. Returns
    ``(x_1, x_2, varnce_x_1, varnce_x_2, alpha_x)``. By construction the
    bi-normal reproduces ``xm`` and the covariance ``w'x'`` (``wpxp``) exactly.

    Parameters
    ----------
    xm, xp2, wp2, sqrt_wp2, wpxp : jax.Array
        Mean/variance of x, w-variance, ``sqrt(wp2)``, ``w'x'`` covariance.
    w_1_n, w_2_n, mixt_frac, sigma_sqd_w : jax.Array
        ADG1 w-closure outputs.
    beta : float or jax.Array
        Tunable parameter (``CLUBBParams.beta``); scalar or per-column
        ``(ngrdcol,)``.
    """
    x_1 = xm - wpxp / (sqrt_wp2 * w_2_n)
    x_2 = xm - wpxp / (sqrt_wp2 * w_1_n)

    alpha_x = 0.5 * (1.0 - wpxp ** 2 / ((1.0 - sigma_sqd_w) * wp2 * xp2))
    alpha_x = jnp.clip(alpha_x, _ZERO, 1.0)

    beta_arr = jnp.asarray(beta)
    beta_bc = beta_arr[:, None] if beta_arr.ndim == 1 else beta_arr
    two_thirds_beta = (2.0 / 3.0) * beta_bc
    width_factor_1 = two_thirds_beta + 2.0 * mixt_frac * (1.0 - two_thirds_beta)

    varnce_x_1 = width_factor_1 * xp2 * alpha_x / mixt_frac
    varnce_x_2 = (2.0 - width_factor_1) * xp2 * alpha_x / (1.0 - mixt_frac)
    return x_1, x_2, varnce_x_1, varnce_x_2, alpha_x


def ADG1_pdf_driver(wm, rtm, thlm, um, vm, wp2, rtp2, thlp2, up2, vp2,
                    Skw, wprtp, wpthlp, upwp, vpwp, sqrt_wp2, sigma_sqd_w,
                    beta, mixt_frac_max_mag):
    """Top-level ADG1 PDF parameter driver (``ADG1_pdf_driver``).

    Closes the ``w`` PDF then the rt/thl/u/v responders. All inputs on zt
    levels, shape ``(ngrdcol, nzt)``.

    Returns
    -------
    dict
        Component means/variances and mixture fraction for w and each
        responder (keys ``w_1``/``w_2``/``varnce_w_*``/``mixt_frac``,
        ``rt_1``/``rt_2``/``varnce_rt_*``/``alpha_rt``, and likewise for
        ``thl``, ``u``, ``v``).
    """
    (w_1, w_2, w_1_n, w_2_n,
     varnce_w_1, varnce_w_2, mixt_frac) = ADG1_w_closure(
        wm, wp2, Skw, sigma_sqd_w, sqrt_wp2, mixt_frac_max_mag)

    rt_1, rt_2, varnce_rt_1, varnce_rt_2, alpha_rt = ADG1_ADG2_responder_params(
        rtm, rtp2, wp2, sqrt_wp2, wprtp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    thl_1, thl_2, varnce_thl_1, varnce_thl_2, alpha_thl = ADG1_ADG2_responder_params(
        thlm, thlp2, wp2, sqrt_wp2, wpthlp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    u_1, u_2, varnce_u_1, varnce_u_2, alpha_u = ADG1_ADG2_responder_params(
        um, up2, wp2, sqrt_wp2, upwp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    v_1, v_2, varnce_v_1, varnce_v_2, alpha_v = ADG1_ADG2_responder_params(
        vm, vp2, wp2, sqrt_wp2, vpwp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    return {
        "w_1": w_1, "w_2": w_2,
        "w_1_n": w_1_n, "w_2_n": w_2_n,
        "varnce_w_1": varnce_w_1, "varnce_w_2": varnce_w_2,
        "mixt_frac": mixt_frac,
        "rt_1": rt_1, "rt_2": rt_2,
        "varnce_rt_1": varnce_rt_1, "varnce_rt_2": varnce_rt_2, "alpha_rt": alpha_rt,
        "thl_1": thl_1, "thl_2": thl_2,
        "varnce_thl_1": varnce_thl_1, "varnce_thl_2": varnce_thl_2, "alpha_thl": alpha_thl,
        "u_1": u_1, "u_2": u_2,
        "varnce_u_1": varnce_u_1, "varnce_u_2": varnce_u_2, "alpha_u": alpha_u,
        "v_1": v_1, "v_2": v_2,
        "varnce_v_1": varnce_v_1, "varnce_v_2": varnce_v_2, "alpha_v": alpha_v,
    }


__all__ = ["ADG1_w_closure", "ADG1_ADG2_responder_params", "ADG1_pdf_driver"]
