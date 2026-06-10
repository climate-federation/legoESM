"""Diagnostic ADG1-PDF closure for the runnable CLUBB scheme (phase 2a).

Given the column mean state and the carried ``wp2`` on the CLUBB grid, this
diagnoses the second moments with standard mixing-length / down-gradient
closures, runs the **ADG1 double-Gaussian assumed-PDF** closure
(:mod:`clubb_pdf`) — the distinctive CLUBB feature absent from
:mod:`clubb_lite` (single Gaussian) — and returns the liquid cloud fraction,
cloud water ``rcm``, and the moist buoyancy flux ``wpthvp`` (including the
cloud-water-flux latent-heat term that makes a cloudy layer more buoyant).

This is the *diagnostic* coupling used by the phase-1/2a runnable
``clubb_turbulence`` entry: skewness is taken symmetric (``Skw = 0`` →
``mixt_frac = 1/2``) and the variances are mixing-length closures, pending the
fully prognostic moment advances (:mod:`clubb_moments`) carried as state. All
fields are on thermodynamic (zt) levels of the ascending CLUBB grid.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.turbulence.clubb_config import (
    CLUBBConfig,
    derive_mixt_frac_max_mag,
)
from legoesm.atmosphere.physics.turbulence.clubb_grid import CLUBBGrid, ddzt, zm2zt
from legoesm.atmosphere.physics.turbulence.clubb_pdf import (
    ADG1_pdf_driver,
    calc_pdf_liquid_cloud_frac,
)

from legoesm import constants

_EP1 = (1.0 - constants.epsilon) / constants.epsilon
_EP2 = 1.0 / constants.epsilon
_HUNDRED = 100.0


def _grad_zt(field_zt, gr: CLUBBGrid):
    """d/dz of a zt-level field, returned on zt (``zm2zt(ddzt(.))``)."""
    return zm2zt(ddzt(field_zt, gr), gr)


def diagnose_cloud_and_buoyancy(thlm, rtm, wp2, exner, p_in_Pa, thv_ds, Kh, Lscale,
                                gr: CLUBBGrid, config: CLUBBConfig):
    """ADG1-PDF cloud fraction, cloud water, and moist buoyancy flux (zt levels).

    Parameters (all ``(ncol, nzt)`` on the ascending CLUBB grid)
    ----------
    thlm, rtm : jax.Array
        Liquid-water potential temperature [K] and total water [kg/kg].
    wp2 : jax.Array
        Carried ``w'^2`` [m^2/s^2].
    exner, p_in_Pa, thv_ds : jax.Array
        Exner, pressure [Pa], dry-static virtual potential temperature [K].
    Kh : jax.Array
        Eddy diffusivity for scalars [m^2/s].
    Lscale : jax.Array
        CLUBB parcel mixing length [m].
    gr : CLUBBGrid
    config : CLUBBConfig

    Returns
    -------
    tuple of jax.Array
        ``(cloud_frac, rcm, wpthvp)`` on zt levels — liquid cloud fraction [-],
        cloud water [kg/kg], and the buoyancy flux ``w'thv'`` [K m/s].
    """
    params = config.params
    wp2 = jnp.maximum(wp2, config.tke_min)
    sqrt_wp2 = jnp.sqrt(wp2)

    ddz_thl = _grad_zt(thlm, gr)
    ddz_rt = _grad_zt(rtm, gr)

    # Down-gradient second-order fluxes and mixing-length variances.
    wpthlp = -Kh * ddz_thl
    wprtp = -Kh * ddz_rt
    thlp2 = jnp.maximum((Lscale * ddz_thl) ** 2, config.thl_tol ** 2)
    rtp2 = jnp.maximum((Lscale * ddz_rt) ** 2, config.rt_tol ** 2)
    rtpthlp = Lscale ** 2 * ddz_thl * ddz_rt
    up2 = vp2 = jnp.maximum(wp2, config.w_tol ** 2)

    # sigma_sqd_w (Skw = 0 -> gamma = gamma_coef), computed directly on zt.
    denom_thl = jnp.sqrt(wp2 * thlp2) + _HUNDRED * config.w_tol * config.thl_tol
    denom_rt = jnp.sqrt(wp2 * rtp2) + _HUNDRED * config.w_tol * config.rt_tol
    max_corr = jnp.maximum((wpthlp / denom_thl) ** 2, (wprtp / denom_rt) ** 2)
    sigma_sqd_w = jnp.clip(params.gamma_coef * (1.0 - jnp.minimum(max_corr, 1.0)), 0.0, 0.99)

    z = jnp.zeros_like(wp2)
    mfmm = derive_mixt_frac_max_mag(params.Skw_max_mag)
    adg1 = ADG1_pdf_driver(
        z, rtm, thlm, z, z, wp2, rtp2, thlp2, up2, vp2, z,
        wprtp, wpthlp, z, z, sqrt_wp2, sigma_sqd_w, params.beta, mfmm)

    rcm, cloud_frac = calc_pdf_liquid_cloud_frac(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa)

    # Moist buoyancy flux: wpthvp = wpthlp + ep1*thv_ds*wprtp + rc_coef*wprcp,
    # with a down-gradient cloud-water flux wprcp (rc_coef = Lv/(exner*Cp) - ep2*thv).
    wprcp = -Kh * _grad_zt(rcm, gr)
    rc_coef = constants.L_v / (exner * constants.c_pd) - _EP2 * thv_ds
    wpthvp = wpthlp + _EP1 * thv_ds * wprtp + rc_coef * wprcp
    return cloud_frac, rcm, wpthvp


__all__ = ["diagnose_cloud_and_buoyancy"]
