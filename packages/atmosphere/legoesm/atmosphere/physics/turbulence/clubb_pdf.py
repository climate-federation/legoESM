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

import math as _math

import jax
import jax.numpy as jnp

from legoesm import constants

_ZERO = 0.0
_SKW_TOL = 1.0e-5   # |Skw| below which mixt_frac is pinned to 0.5

# Numerical tolerances / smoothing magnitudes (CLUBB constants_clubb.F90) — not
# physical constants and not tunable scheme params (safety/smoothing scales).
_EPS = 1.0e-10
_CHI_TOL = max(1.0e-8, jnp.finfo(jnp.float64).eps)   # chi tolerance [kg/kg]
_MAX_MAG_CORRELATION = 0.99       # bound keeping diagnosed correlations valid
_MIN_MAX_SMTH_MAG = 1.0e-9        # smoothing magnitude for smooth_max
_MAX_NUM_STDEVS = 5.0             # PDF truncation range for cloud-frac limits
_SQRT_2 = _math.sqrt(2.0)
_SQRT_2PI = _math.sqrt(2.0 * _math.pi)


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


# ===========================================================================
# Liquid cloud fraction / cloud water from the ADG1 PDF (pdf_closure_module)
# ===========================================================================


def smooth_corr_quotient(numerator, denominator, denom_thresh):
    """Smoothly bounded correlation quotient ``num/den`` (``pdf_utilities.F90``).

    Two ``smooth_max`` lifts keep the result a valid correlation: the
    denominator is raised to at least ``|num|/max_mag_correlation`` (so
    ``|quotient| <= max_mag_correlation``) and then to ``denom_thresh`` (never
    divides by ~0). Pure-jnp, differentiable.
    """
    num = jnp.asarray(numerator)
    den = jnp.asarray(denominator)
    coef = jnp.minimum(_MIN_MAX_SMTH_MAG, denom_thresh)

    def _smax(a, b):
        return 0.5 * ((a + b) + jnp.sqrt((a - b) ** 2 + coef ** 2))

    tmp = _smax(jnp.abs(num) / _MAX_MAG_CORRELATION, den)
    tmp = _smax(tmp, denom_thresh)
    return num / tmp


def calc_comp_corrs_binormal(xpyp, xm, ym, mu_x_1, mu_x_2, mu_y_1, mu_y_2,
                             sigma_x_1_sqd, sigma_x_2_sqd, sigma_y_1_sqd,
                             sigma_y_2_sqd, mixt_frac):
    """Shared PDF-component correlation of two bi-normal variables x, y.

    ``pdf_utilities.F90:calc_comp_corrs_binormal``. Both components share one
    correlation, solved from the overall covariance
    ``<x'y'> = sum_i w_i[(mu_x_i-<x>)(mu_y_i-<y>) + corr*sigma_x_i*sigma_y_i]``
    and bounded by :func:`smooth_corr_quotient`. Returns ``(corr, corr)``.
    """
    a = jnp.asarray(mixt_frac)
    numerator = (xpyp - a * (mu_x_1 - xm) * (mu_y_1 - ym)
                 - (1.0 - a) * (mu_x_2 - xm) * (mu_y_2 - ym))
    denominator = (a * _safe_sqrt(sigma_x_1_sqd * sigma_y_1_sqd)
                   + (1.0 - a) * _safe_sqrt(sigma_x_2_sqd * sigma_y_2_sqd))
    corr = smooth_corr_quotient(numerator, denominator, _EPS)
    return corr, corr


def transform_pdf_chi_eta_component(tl, rsatl, rt, exner_in,
                                    varnce_rt, varnce_thl, corr_rt_thl):
    """Sommeria-Deardorff (rt, thl) -> (chi, eta) transform for one PDF component.

    ``pdf_closure_module.F90:transform_pdf_chi_eta_component``. ``chi`` is the
    extended liquid water (saturation excess). The local ``cc_slope`` is the
    Clausius-Clapeyron sensitivity ``eps*L_v^2/(R_d*c_pd*tl^2)`` (NOT the tunable
    ``beta``). Returns ``(chi, crt, cthl, stdev_chi, stdev_eta, covar_chi_eta,
    corr_chi_eta)``.
    """
    cc_slope = constants.epsilon * constants.L_v ** 2 / (constants.R_d * constants.c_pd * tl ** 2)
    invrs = 1.0 / (1.0 + cc_slope * rsatl)
    chi = (rt - rsatl) * invrs
    crt = invrs
    cthl = ((1.0 + cc_slope * rt) * invrs ** 2
            * (constants.c_pd / constants.L_v) * cc_slope * rsatl * exner_in)
    vrnc_rt_t = crt ** 2 * varnce_rt
    vrnc_thl_t = cthl ** 2 * varnce_thl
    # _safe_sqrt: component variances can be exactly 0 (e.g. alpha_x clipped to 0
    # for perfectly-correlated columns), and a bare sqrt has a singular VJP
    # there. Forward-identical (variances >= 0).
    corr_t = 2.0 * corr_rt_thl * crt * cthl * _safe_sqrt(varnce_rt * varnce_thl)
    vrnc_chi = vrnc_rt_t - corr_t + vrnc_thl_t
    vrnc_eta = vrnc_rt_t + corr_t + vrnc_thl_t
    stdev_chi = _safe_sqrt(vrnc_chi)
    stdev_eta = _safe_sqrt(vrnc_eta)
    covar_chi_eta = vrnc_rt_t - vrnc_thl_t
    # smooth_corr_quotient (pdf_utilities) bounding corr_chi_eta to [-0.99, 0.99].
    corr_chi_eta = smooth_corr_quotient(covar_chi_eta, stdev_chi * stdev_eta, _CHI_TOL ** 2)
    return chi, crt, cthl, stdev_chi, stdev_eta, covar_chi_eta, corr_chi_eta


def calc_liquid_cloud_frac_component(mean_chi, stdev_chi):
    """Liquid cloud fraction + cloud water of one PDF component (Gaussian CDF of chi).

    ``pdf_closure_module.F90:calc_liquid_cloud_frac_component``, with
    +-``max_num_stdevs`` truncation to the clear / fully-cloudy limits.
    Returns ``(cloud_frac, rc)``.
    """
    mean_chi = jnp.asarray(mean_chi)
    stdev_chi = jnp.asarray(stdev_chi)
    is_clear = (((jnp.abs(mean_chi) <= _EPS) & (stdev_chi <= _CHI_TOL))
                | (mean_chi < -_MAX_NUM_STDEVS * stdev_chi))
    is_full = mean_chi > _MAX_NUM_STDEVS * stdev_chi
    # Double-where denominator guard (AD-safe in any precision). Where the PDF
    # component has resolvable width (``stdev_chi > floor``) the quotient uses
    # the true stdev (in float64 ``floor`` is below the reference 1e-100, so
    # active cells are bit-identical to the reference); elsewhere it uses a
    # dummy denominator of 1, so neither the quotient nor its tangent
    # ``-mean/safe_s^2`` can overflow regardless of ``mean_chi`` magnitude. Such
    # zero-width cells are always selected as clear or fully-cloudy below, so the
    # forward result is unchanged. ``floor = sqrt(tiny)`` keeps ``stdev_chi^2``
    # representable for the resolvable cells too.
    dt = stdev_chi.dtype
    floor = jnp.sqrt(jnp.finfo(dt).tiny)
    resolvable = stdev_chi > floor
    safe_s = jnp.where(resolvable, stdev_chi, jnp.asarray(1.0, dt))
    zeta = mean_chi / safe_s
    # The Gaussian (cf_mid/rc_mid) is the SELECTED output only on partial-cloud
    # cells (``partial``), where |zeta| <= max_num_stdevs already — there it uses
    # the RAW zeta, so the active-cell value AND gradient are reference-identical
    # (including exactly at the cutoff mean_chi = +-max_num_stdevs*stdev_chi,
    # which the strict clear/full comparisons keep in the partial branch).
    # On masked clear/full cells the Gaussian is discarded, so zeta is clamped
    # there only to stop ``exp(-0.5*zeta^2)`` from overflowing (-> NaN VJP) in
    # float32 where the masked zeta can be enormous.
    partial = ~(is_clear | is_full)
    zeta_g = jnp.where(partial, zeta, jnp.clip(zeta, -_MAX_NUM_STDEVS, _MAX_NUM_STDEVS))
    cf_mid = 0.5 * (1.0 + jax.scipy.special.erf(zeta_g / _SQRT_2))
    rc_mid = mean_chi * cf_mid + stdev_chi * jnp.exp(-0.5 * zeta_g ** 2) / _SQRT_2PI
    cf = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, cf_mid))
    rc = jnp.where(is_clear, 0.0, jnp.where(is_full, mean_chi, rc_mid))
    return cf, rc


def calc_pdf_liquid_cloud_frac(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa):
    """Liquid cloud fraction and cloud water from the ADG1 PDF components.

    Derives the bi-normal rt-thl component correlation from the resolved
    covariance ``rtpthlp``, applies the chi-eta transform per component, the
    Gaussian cloud-fraction integral per component, and combines by mixture
    fraction (``pdf_closure_module.F90`` pre-advance path). Saturation is Flatau
    (CAM default) via :mod:`clubb_saturation`.

    Parameters
    ----------
    adg1 : dict
        Output of :func:`ADG1_pdf_driver` (component means/variances + mixt_frac).
    rtpthlp, rtm, thlm, exner, p_in_Pa : jax.Array
        ``r_t' theta_l'`` covariance, mean total water, mean theta_l, Exner, and
        pressure [Pa], all on zt levels, shape ``(ngrdcol, nzt)``.

    Returns
    -------
    tuple of jax.Array
        ``(rcm, cloud_frac)`` on zt levels — cloud water [kg/kg] and liquid
        cloud fraction [-].
    """
    comp = calc_pdf_liquid_cloud_frac_components(
        adg1, rtpthlp, rtm, thlm, exner, p_in_Pa)
    return comp["rcm"], comp["cloud_frac"]


def calc_pdf_liquid_cloud_frac_components(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa):
    """Per-component liquid cloud-fraction PDF closure (the full intermediates).

    Like :func:`calc_pdf_liquid_cloud_frac` but returns every per-component
    intermediate (``chi``/``crt``/``cthl``/``stdev_chi``/``stdev_eta``/
    ``corr_ce``/``cf``/``rc`` for components 1 and 2, plus ``mixt_frac``,
    ``cloud_frac`` and ``rcm``). The cloud-water flux assembly
    (:func:`clubb_pdf_moments.calc_pdf_xprcp_fluxes`) consumes these.
    Mirrors ``pdf_closure_module.F90:calc_pdf_liquid_cloud_frac_components``.
    """
    from legoesm.atmosphere.physics.turbulence.clubb_saturation import sat_mixrat_liq

    corr_1, corr_2 = calc_comp_corrs_binormal(
        rtpthlp, rtm, thlm, adg1["rt_1"], adg1["rt_2"], adg1["thl_1"], adg1["thl_2"],
        adg1["varnce_rt_1"], adg1["varnce_rt_2"], adg1["varnce_thl_1"],
        adg1["varnce_thl_2"], adg1["mixt_frac"])

    mf = adg1["mixt_frac"]
    tl_1 = adg1["thl_1"] * exner
    tl_2 = adg1["thl_2"] * exner
    rsatl_1 = sat_mixrat_liq(p_in_Pa, tl_1)
    rsatl_2 = sat_mixrat_liq(p_in_Pa, tl_2)

    (chi_1, crt_1, cthl_1, schi_1, seta_1, _, corr_ce_1) = transform_pdf_chi_eta_component(
        tl_1, rsatl_1, adg1["rt_1"], exner, adg1["varnce_rt_1"], adg1["varnce_thl_1"], corr_1)
    (chi_2, crt_2, cthl_2, schi_2, seta_2, _, corr_ce_2) = transform_pdf_chi_eta_component(
        tl_2, rsatl_2, adg1["rt_2"], exner, adg1["varnce_rt_2"], adg1["varnce_thl_2"], corr_2)

    cf_1, rc_1 = calc_liquid_cloud_frac_component(chi_1, schi_1)
    cf_2, rc_2 = calc_liquid_cloud_frac_component(chi_2, schi_2)

    cloud_frac = mf * cf_1 + (1.0 - mf) * cf_2
    rcm = jnp.maximum(0.0, mf * rc_1 + (1.0 - mf) * rc_2)
    return {
        "mixt_frac": mf, "cloud_frac": cloud_frac, "rcm": rcm,
        "chi_1": chi_1, "chi_2": chi_2,
        "crt_1": crt_1, "crt_2": crt_2, "cthl_1": cthl_1, "cthl_2": cthl_2,
        "stdev_chi_1": schi_1, "stdev_chi_2": schi_2,
        "stdev_eta_1": seta_1, "stdev_eta_2": seta_2,
        "corr_ce_1": corr_ce_1, "corr_ce_2": corr_ce_2,
        "cf_1": cf_1, "cf_2": cf_2, "rc_1": rc_1, "rc_2": rc_2,
    }


__all__ = [
    "ADG1_w_closure",
    "ADG1_ADG2_responder_params",
    "ADG1_pdf_driver",
    "smooth_corr_quotient",
    "calc_comp_corrs_binormal",
    "transform_pdf_chi_eta_component",
    "calc_liquid_cloud_frac_component",
    "calc_pdf_liquid_cloud_frac",
    "calc_pdf_liquid_cloud_frac_components",
]
