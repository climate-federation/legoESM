"""Diagnostic cloud fraction and cloud optical property computation.

Implements two cloud fraction schemes:

1. **Sundqvist (1988)**: RH-based, simple and robust.
   ``cf = clamp((RH - RH_crit) / (1 - RH_crit), 0, 1)``

2. **Xu-Randall (1996)**: RH + condensate-based, more physical.
   ``cf = RH^p * [1 - exp(-alpha * q_c / ((1 - RH) * q_s))]``

Both schemes compute cloud fraction per column per layer and derive
cloud liquid/ice water paths for RRTMGP cloud optics.

References
----------
- Sundqvist, H. (1988). Parameterization of condensation and
  associated clouds in models for weather prediction and general
  circulation simulation. *Physically-Based Modelling and Simulation
  of Climate and Climatic Change*, NATO ASI Series, 243, 433-461.
- Xu, K.-M. & Randall, D. A. (1996). A semiempirical cloudiness
  parameterization for use in climate models. *J. Atmos. Sci.*,
  53, 3084-3102.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.thermo import saturation_mixing_ratio
from legoesm import constants


class CloudProperties(NamedTuple):
    """Cloud properties for radiation coupling.

    All arrays have shape (ncol, nlev).

    Fields
    ------
    cloud_fraction : jnp.ndarray
        Cloud fraction per layer [0, 1].
    lwp : jnp.ndarray
        In-cloud liquid water path per layer [kg/m^2].  RRTMGP multiplies
        the derived optical depth by ``cloud_fraction`` internally, so
        ``lwp`` must be the in-cloud (not grid-mean) value.  The effective
        grid-mean optical depth is ``tau_in_cloud × cloud_fraction``.
    iwp : jnp.ndarray
        In-cloud ice water path per layer [kg/m^2].  Same convention as
        ``lwp``.
    r_eff_liq : jnp.ndarray
        Effective radius for liquid droplets [m].
    r_eff_ice : jnp.ndarray
        Effective radius for ice crystals [m].
    """
    cloud_fraction: jnp.ndarray
    lwp: jnp.ndarray
    iwp: jnp.ndarray
    r_eff_liq: jnp.ndarray
    r_eff_ice: jnp.ndarray


def _ice_fraction(T: jnp.ndarray, config: CloudConfig) -> jnp.ndarray:
    """Fraction of condensate that is ice, based on temperature.

    Linear ramp from 0 (all liquid) at T_freeze to 1 (all ice) at T_ice_only.
    """
    frac = (config.T_freeze - T) / jnp.maximum(
        config.T_freeze - config.T_ice_only, 1.0
    )
    return jnp.clip(frac, 0.0, 1.0)


def sundqvist_cloud_fraction(
    RH: jnp.ndarray,
    config: CloudConfig,
) -> jnp.ndarray:
    """Sundqvist (1988) cloud fraction from relative humidity.

    Parameters
    ----------
    RH : jnp.ndarray
        Relative humidity [0, 1+], shape (ncol, nlev).
    config : CloudConfig

    Returns
    -------
    jnp.ndarray
        Cloud fraction [0, 1], same shape as RH.
    """
    cf = (RH - config.rh_crit) / jnp.maximum(1.0 - config.rh_crit, 1.0e-6)
    return jnp.clip(cf, 0.0, 1.0)


def xu_randall_cloud_fraction(
    RH: jnp.ndarray,
    q_condensate: jnp.ndarray,
    q_sat: jnp.ndarray,
    config: CloudConfig,
) -> jnp.ndarray:
    """Xu-Randall (1996) cloud fraction from RH and condensate.

    Parameters
    ----------
    RH : jnp.ndarray
        Relative humidity [0, 1+], shape (ncol, nlev).
    q_condensate : jnp.ndarray
        Total cloud condensate (q_cloud + q_ice) [kg/kg], shape (ncol, nlev).
    q_sat : jnp.ndarray
        Saturation mixing ratio [kg/kg], shape (ncol, nlev).
    config : CloudConfig

    Returns
    -------
    jnp.ndarray
        Cloud fraction [0, 1].
    """
    # Avoid division by zero when RH = 1
    denominator = jnp.maximum((1.0 - RH) * q_sat, 1.0e-10)
    exponent = -config.alpha_xr * q_condensate / denominator
    cf = jnp.power(jnp.clip(RH, 0.0, 1.0), config.p_xr) * (
        1.0 - jnp.exp(exponent)
    )
    return jnp.clip(cf, 0.0, 1.0)


def compute_cloud_properties(
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    q_v: jnp.ndarray,
    dp: jnp.ndarray,
    config: CloudConfig,
    q_cloud: jnp.ndarray | None = None,
    q_ice: jnp.ndarray | None = None,
) -> CloudProperties:
    """Compute diagnostic cloud fraction and cloud optical properties.

    Parameters
    ----------
    T : jnp.ndarray
        Temperature [K], shape (ncol, nlev).
    p_full : jnp.ndarray
        Pressure at full levels [Pa], shape (ncol, nlev).
    q_v : jnp.ndarray
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    dp : jnp.ndarray
        Layer pressure thickness [Pa], shape (ncol, nlev).
        Computed as ``p_half[..., 1:] - p_half[..., :-1]`` (positive).
    config : CloudConfig
    q_cloud : jnp.ndarray or None
        Explicit cloud liquid water [kg/kg] from microphysics.
    q_ice : jnp.ndarray or None
        Explicit cloud ice [kg/kg] from microphysics.

    Returns
    -------
    CloudProperties
        Cloud fraction and water/ice paths for radiation.
    """
    # Saturation mixing ratio and relative humidity
    q_sat = saturation_mixing_ratio(T, p_full)
    RH = q_v / jnp.maximum(q_sat, 1.0e-10)

    # --- Cloud fraction ---
    if config.scheme == "xu_randall":
        q_c = jnp.zeros_like(T) if q_cloud is None else q_cloud
        q_i = jnp.zeros_like(T) if q_ice is None else q_ice
        q_condensate = q_c + q_i
        cf = xu_randall_cloud_fraction(RH, q_condensate, q_sat, config)
    elif config.scheme == "sundqvist":
        cf = sundqvist_cloud_fraction(RH, config)
        if config.sigma_bl < 1.0:
            # Boundary-layer override: use lower rh_crit in the lowest layers
            # (sigma > sigma_bl) where the warm lower troposphere keeps RH
            # structurally below the free-troposphere threshold.
            # Python `if` on static config value — compiles away when disabled.
            sigma_approx = p_full / jnp.maximum(p_full[:, -1:], 1.0)
            cf_bl = (RH - config.rh_crit_bl) / jnp.maximum(
                1.0 - config.rh_crit_bl, 1.0e-6
            )
            cf_bl = jnp.clip(cf_bl, 0.0, 1.0)
            cf = jnp.where(sigma_approx > config.sigma_bl, cf_bl, cf)
    else:
        raise ValueError(
            f"Unknown cloud scheme: {config.scheme!r}. "
            f"Valid schemes: 'sundqvist', 'xu_randall'. "
            f"(Use cloud_scheme='none' upstream to skip clouds entirely.)"
        )

    # --- Cloud condensate ---
    has_explicit_condensate = q_cloud is not None or q_ice is not None
    if has_explicit_condensate:
        # q_cloud / q_ice are grid-mean mixing ratios from the microphysics
        # tracer.  Convert to in-cloud values by dividing by cloud fraction so
        # that RRTMGP's internal ``optical_depth × cloud_fraction`` scaling
        # yields the correct grid-mean optical depth.
        # Safe denominator: where cf → 0 the condensate is also → 0, so the
        # LWP approaches 0/0; the masked form keeps it zero in clear sky.
        cf_safe = jnp.where(cf > 1.0e-6, cf, 1.0)
        q_c_raw = jnp.zeros_like(T) if q_cloud is None else jnp.maximum(q_cloud, 0.0)
        q_i_raw = jnp.zeros_like(T) if q_ice is None else jnp.maximum(q_ice, 0.0)
        # In-cloud mixing ratios (zero in clear-sky cells)
        q_c = jnp.where(cf > 1.0e-6, q_c_raw / cf_safe, 0.0)
        q_i = jnp.where(cf > 1.0e-6, q_i_raw / cf_safe, 0.0)
    else:
        # Diagnose condensate from a typical in-cloud liquid water content.
        # q_c_diagnostic is the IN-CLOUD mixing ratio; do NOT pre-multiply by
        # cf here — RRTMGP applies the cloud-fraction scaling to the optical
        # depth.  Pre-multiplying would cause tau ∝ cf², a factor-of-cf
        # underestimate of the cloud radiative effect (SW bias audit).
        f_ice = _ice_fraction(T, config)
        q_diag = config.q_c_diagnostic * (1.0 - f_ice)
        q_diag_i = config.q_c_diagnostic * f_ice
        # Zero out condensate in clear-sky cells (cf < threshold)
        q_c = jnp.where(cf > 1.0e-6, q_diag, 0.0)
        q_i = jnp.where(cf > 1.0e-6, q_diag_i, 0.0)

    # --- In-cloud water/ice paths [kg/m^2] ---
    # RRTMGP multiplies the derived optical depth by cloud_fraction, so these
    # must be in-cloud (not grid-mean) paths.  The effective grid-mean optical
    # depth seen by the radiation solver is tau_in_cloud × cloud_fraction.
    lwp = q_c * dp / constants.g
    iwp = q_i * dp / constants.g

    # Effective radii (constant for now): ``broadcast_to`` produces a
    # zero-copy logical view, whereas ``jnp.full_like(T, scalar)``
    # materialises a fresh ``(ncol, nlev)`` constant buffer every
    # physics step.  XLA folds the broadcast at trace time but the
    # broadcast form keeps the HLO graph small and avoids two
    # allocator round-trips per cloud_optics call.
    _scalar_dtype = T.dtype
    r_eff_liq = jnp.broadcast_to(
        jnp.asarray(config.r_eff_liq, dtype=_scalar_dtype), T.shape,
    )
    r_eff_ice = jnp.broadcast_to(
        jnp.asarray(config.r_eff_ice, dtype=_scalar_dtype), T.shape,
    )

    return CloudProperties(
        cloud_fraction=cf,
        lwp=lwp,
        iwp=iwp,
        r_eff_liq=r_eff_liq,
        r_eff_ice=r_eff_ice,
    )
