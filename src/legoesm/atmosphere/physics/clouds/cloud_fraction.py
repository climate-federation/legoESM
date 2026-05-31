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
        Grid-mean liquid water path per layer [kg/m^2].
    iwp : jnp.ndarray
        Grid-mean ice water path per layer [kg/m^2].
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

    def to_rrtmg_kwargs(self) -> dict:
        """Return cloud kwargs dict for ``rrtmgp_radiation`` / ``solve_columns``.

        **Deliberately omits cloud_fraction.** ``compute_cloud_properties``
        returns GRID-MEAN water paths (``lwp = q_c * dp / g``, q_c the
        grid-mean prognostic cloud water), which already carry the
        partial-coverage discount ``LWP_grid = cf · LWP_in-cloud``.
        RRTMG's optics multiplies cloud optical depth by
        ``cloud_fraction`` again ("scale cloud optical depth by cloud
        fraction for partial coverage", optics.py) — that scaling
        expects IN-CLOUD paths.  Passing grid-mean LWP *and*
        ``cloud_fraction`` double-counts the discount:
        ``τ_used = cf² · τ_in-cloud`` instead of ``cf · τ_in-cloud``,
        making clouds ~cf× too optically thin in both SW and LW
        (→ OSR too low, OLR too high).  Calibration probe found the
        bug cost 59 W/m² OSR at cf=0.6 and 113 W/m² at cf=0.3 (commit
        4c9591bb).

        Since τ is linear in LWP, "grid-mean LWP, no cf scaling" is
        mathematically identical to the correct "in-cloud LWP × cf
        scaling".  This helper centralises the right behaviour so the
        bug cannot resurface at a third call site (iter-15 restored
        it in ``physics_pipeline.py``, iter-16 fixed an independent
        copy in ``integration.py``).
        """
        return {
            "cloud_path_liq": self.lwp,
            "cloud_path_ice": self.iwp,
            "cloud_r_eff_liq": self.r_eff_liq,
            "cloud_r_eff_ice": self.r_eff_ice,
        }


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
    """Sundqvist (1989) cloud fraction from relative humidity.

    Sundqvist, Berge & Kristjánsson (1989, MWR 117) relate the cloud
    cover ``b`` to the grid-mean RH by ``(1−b)² = (1−RH)/(1−RH_crit)``,
    i.e.

        ``b = 1 − √((1−RH)/(1−RH_crit))``   for RH ≥ RH_crit, else 0.

    This √-form (used by ECHAM and most Sundqvist implementations) is the
    faithful scheme; the earlier code here used a *linear* ramp
    ``(RH−RH_crit)/(1−RH_crit)`` mislabeled as Sundqvist — the √-form
    rises faster just above RH_crit (e.g. 0.29 vs 0.5 at the midpoint).

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
    arg = (1.0 - RH) / jnp.maximum(1.0 - config.rh_crit, 1.0e-6)
    # Double-``where`` for the √-form: at RH ≥ 1 (arg ≤ 0) the cloud is
    # full so b = 1 *exactly*, while keeping ``√`` off zero so its
    # otherwise-infinite derivative cannot leak a NaN cotangent through
    # the dead branch (same AD-safe pattern as the Smagorinsky–Lilly
    # cutoff).  For RH < RH_crit, arg > 1 ⇒ b < 0 ⇒ clipped to 0.
    arg_safe = jnp.where(arg > 0.0, arg, 1.0)
    cf = jnp.where(arg > 0.0, 1.0 - jnp.sqrt(arg_safe), 1.0)
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
    # Xu-Randall (1996) denominator ((1−RH)·q_sat)^γ.  Floor the base
    # at 1e-10 *before* the power so the fractional γ<1 exponent never
    # sees 0 (1e-10^γ stays finite + positive ⇒ AD-safe).
    denominator = jnp.maximum((1.0 - RH) * q_sat, 1.0e-10) ** config.gamma_xr
    exponent = -config.alpha_xr * q_condensate / denominator
    # Floor the RH base of the fractional power at 1e-6 (not 0): p_xr < 1, so
    # ``RH**p_xr`` has an infinite derivative at RH=0 (0**-0.75), giving an inf
    # reverse-mode gradient d(cf)/d(q_v) for any dry layer (RH=0 ⇒ q_v=0, e.g.
    # upper stratosphere / dry init).  The forward is unaffected — cf -> 0 there
    # anyway via the (1 - exp) factor — and the clip zeroes the gradient chain
    # below the floor.
    cf = jnp.power(jnp.clip(RH, 1.0e-6, 1.0), config.p_xr) * (
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
    else:
        raise ValueError(
            f"Unknown cloud scheme: {config.scheme!r}. "
            f"Valid schemes: 'sundqvist', 'xu_randall'. "
            f"(Use cloud_scheme='none' upstream to skip clouds entirely.)"
        )

    # --- Cloud condensate ---
    has_explicit_condensate = q_cloud is not None or q_ice is not None
    if has_explicit_condensate:
        q_c = jnp.zeros_like(T) if q_cloud is None else jnp.maximum(q_cloud, 0.0)
        q_i = jnp.zeros_like(T) if q_ice is None else jnp.maximum(q_ice, 0.0)
    else:
        # Diagnose condensate from cloud fraction and a typical in-cloud value.
        # Total condensate = cf * q_c_diagnostic, partitioned by temperature.
        q_total = cf * config.q_c_diagnostic
        f_ice = _ice_fraction(T, config)
        q_c = q_total * (1.0 - f_ice)
        q_i = q_total * f_ice

    # --- Cloud water/ice paths [kg/m^2] ---
    # Grid-mean water/ice paths: q * dp / g
    # These are grid-mean (not in-cloud) values, which is what RRTMGP expects
    # when treating each layer independently (no overlap assumption).
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
