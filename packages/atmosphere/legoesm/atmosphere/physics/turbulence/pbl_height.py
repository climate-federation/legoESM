"""PBL height diagnosis (Task 9).

Diagnoses the planetary boundary layer height using the bulk Richardson
number method with linear interpolation between model levels for a
smooth, differentiable estimate.

Two methods are provided:
1. ``diagnose_pbl_height`` — smooth sigmoid-weighted average (fully
   JAX-differentiable, used inside turbulence schemes).
2. ``diagnose_pbl_height_interp`` — linear interpolation to the exact
   Ri_crit crossing (sharper, for diagnostics).

References
----------
- Vogelezang, D. H. P., & Holtslag, A. A. M. (1996). Evaluation and
  model impacts of alternative boundary-layer height formulations.
  Boundary-Layer Meteorol., 81, 245-269.
- Seidel, D. J., et al. (2010). Estimating climatological planetary
  boundary layer heights from radiosonde observations: Comparison of
  methods and uncertainty analysis. J. Geophys. Res., 115, D16113.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import virtual_temperature


__physics_contract__ = {
    "summary": (
        "Planetary boundary-layer height diagnosis from the bulk Richardson "
        "number, with a smooth (sigmoid-weighted / first-crossing) estimate "
        "for a differentiable h_pbl."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "u": "m/s", "v": "m/s",
        "p_full": "Pa", "z_full": "m (above surface)",
    },
    "outputs": {"h_pbl": "m"},
    "sign_convention": (
        "Diagnostic only (no state tendency). h_pbl >= 0, clipped to "
        "[h_min, h_max]; the PBL top is the lowest height where the bulk "
        "Richardson number Ri_b crosses Ri_crit. Ri_b uses wind SHEAR from "
        "the surface (not absolute wind), so a barotropic no-shear wind does "
        "not deepen the PBL. z increases upward; level index -1 is the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Vogelezang & Holtslag (1996), Boundary-Layer Meteorol. 81, 245-269; "
        "Seidel et al. (2010), JGR 115, D16113"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_turbulence.py: a column with a "
        "capping inversion returns h_pbl in [h_min, h_max] near the Ri_crit "
        "crossing; a no-shear (barotropic) wind does not deepen the PBL."
    ),
}


__param_spec__ = {
    "PBLHeightConfig": {
        "scheme_key": "atm.pblh.PBLHeightConfig",
        "excluded": {
            "h_min": "numerics: lowest-layer PBL-height floor (avoids /0)",
            "sharpness": "numerics: sigmoid-gate sharpness for the Ri-crossing detector",
        },
        "params": {
            "Ri_crit": {"units": "1", "bounds": (0.0825, 0.75), "tunable_tier": 1, "transform": "sigmoid", "category": "critical_richardson", "reference": "Vogelezang & Holtslag (1996) bulk-Ri PBL-height criterion", "shape": None},
            "h_max": {"units": "m", "bounds": (1650.0, 15000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pbl_height", "reference": "diagnosed PBL-height upper cap", "shape": None},
        },
    },
}


class PBLHeightConfig(NamedTuple):
    """Configuration for PBL height diagnosis.

    Fields
    ------
    Ri_crit : float
        Critical bulk Richardson number (default 0.25).
    h_min : float
        Minimum PBL height [m] (default 100.0).
    h_max : float
        Maximum PBL height [m] (default 5000.0).
    sharpness : float
        Sigmoid sharpness for smooth method (default 20.0).
    method : str
        CURRENTLY ADVISORY / not read internally — callers select
        ``diagnose_pbl_height`` vs ``diagnose_pbl_height_interp`` directly;
        setting this field has no effect. "smooth" (sigmoid-weighted) or
        "interp" (linear interpolation).
    """
    Ri_crit: float = 0.25
    h_min: float = 100.0
    h_max: float = 5000.0
    sharpness: float = 20.0
    method: str = "smooth"


def compute_bulk_richardson(
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
) -> jax.Array:
    """Compute bulk Richardson number profile from surface.

    Ri_b(z) = g * dz * (theta_v(z) - theta_v_sfc) / (theta_v_sfc * dV(z)^2)

    where dV(z)^2 = (u(z)-u_sfc)^2 + (v(z)-v_sfc)^2 is the wind *shear*
    from the surface, not the absolute wind speed.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    u, v : jax.Array
        Wind components [m/s], shape (ncol, nlev).
    p_full : jax.Array
        Pressure [Pa], shape (ncol, nlev).
    z_full : jax.Array
        Height above surface [m], shape (ncol, nlev).

    Returns
    -------
    Ri_bulk : jax.Array
        Bulk Richardson number, shape (ncol, nlev).
    theta_v : jax.Array
        Virtual potential temperature, shape (ncol, nlev).
    """
    # Virtual potential temperature
    exner = (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    theta_v = virtual_temperature(T, q_v) * exner

    # Surface values (bottom level)
    theta_v_sfc = theta_v[:, -1]  # (ncol,)
    z_sfc = z_full[:, -1:]        # (ncol, 1)
    u_sfc = u[:, -1:]             # (ncol, 1)
    v_sfc = v[:, -1:]             # (ncol, 1)

    # Height above surface
    dz_from_sfc = jnp.abs(z_full - z_sfc) + 1.0  # (ncol, nlev), +1 avoids /0

    # Buoyancy difference from surface
    dtheta_v = theta_v - theta_v_sfc[:, None]

    # Wind shear squared from surface (NOT absolute wind speed)
    # A barotropic wind with no shear should not deepen the PBL.
    dV2 = (u - u_sfc) ** 2 + (v - v_sfc) ** 2 + 1e-4  # coeff-ok: wind-shear floor [m^2/s^2]

    # Bulk Richardson number
    Ri_bulk = (constants.g / jnp.clip(theta_v_sfc[:, None], 1.0, None)) * (
        dtheta_v * dz_from_sfc / dV2
    )

    return Ri_bulk, theta_v


def diagnose_pbl_height(
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
    config: PBLHeightConfig = PBLHeightConfig(),
) -> jax.Array:
    """Diagnose PBL height using sigmoid-weighted bulk Richardson method.

    This method uses a smooth sigmoid weight to identify the transition
    where Ri_bulk crosses Ri_crit, producing a differentiable PBL height.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    u, v : jax.Array
        Wind components [m/s], shape (ncol, nlev).
    p_full : jax.Array
        Pressure [Pa], shape (ncol, nlev).
    z_full : jax.Array
        Height above surface [m], shape (ncol, nlev).
    config : PBLHeightConfig
        Configuration parameters.

    Returns
    -------
    h_pbl : jax.Array
        PBL height [m], shape (ncol,).
    """
    Ri_bulk, _ = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)

    # Sigmoid weights: high where Ri_bulk < Ri_crit (inside PBL)
    sigma = jax.nn.sigmoid(config.sharpness * (config.Ri_crit - Ri_bulk))

    # Transition-zone weights: sigma * (1 - sigma) peaks at the Ri_crit
    # crossing, not at the centroid of the subcritical layer.  This gives
    # a weighted average that converges to the PBL top in the sharp limit.
    weights = sigma * (1.0 - sigma) + 1e-20

    # Weighted average height — fuse num/denom into one stacked sum.
    _h_pair = jnp.sum(jnp.stack([z_full * weights, weights], axis=-1), axis=1)
    h_pbl = _h_pair[..., 0] / _h_pair[..., 1]

    return jnp.clip(h_pbl, config.h_min, config.h_max)


def diagnose_pbl_height_interp(
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
    config: PBLHeightConfig = PBLHeightConfig(),
) -> jax.Array:
    """Diagnose PBL height with linear interpolation to Ri_crit crossing.

    Scans from the surface upward and linearly interpolates between the
    last sub-critical and first super-critical levels. Uses a softmin
    (log-sum-exp) to select the lowest crossing in a differentiable way.

    Parameters
    ----------
    T, q_v, u, v, p_full, z_full : jax.Array
        Same as ``diagnose_pbl_height``.
    config : PBLHeightConfig
        Configuration parameters.

    Returns
    -------
    h_pbl : jax.Array
        PBL height [m], shape (ncol,).
    """
    Ri_bulk, _ = compute_bulk_richardson(T, q_v, u, v, p_full, z_full)
    ncol, nlev = Ri_bulk.shape

    # Work from surface upward: reverse level order (surface = index 0)
    Ri_rev = Ri_bulk[:, ::-1]   # (ncol, nlev), surface first
    z_rev = z_full[:, ::-1]     # (ncol, nlev), surface first

    # For each pair of adjacent levels, check if Ri crosses Ri_crit
    Ri_below = Ri_rev[:, :-1]   # (ncol, nlev-1)
    Ri_above = Ri_rev[:, 1:]
    z_below = z_rev[:, :-1]
    z_above = z_rev[:, 1:]

    # Soft crossing indicator using sigmoid (differentiable)
    cross_weight = (
        jax.nn.sigmoid(config.sharpness * (config.Ri_crit - Ri_below))
        * jax.nn.sigmoid(config.sharpness * (Ri_above - config.Ri_crit))
    )

    # Linear interpolation fraction at each crossing
    dRi = Ri_above - Ri_below + 1e-20
    frac = jnp.clip((config.Ri_crit - Ri_below) / dRi, 0.0, 1.0)
    z_cross = z_below + frac * (z_above - z_below)

    # Softmin: select the lowest crossing via log-sum-exp
    # Negative temperature parameter picks the minimum.
    beta = config.sharpness / config.h_max  # scale-aware sharpness
    large_val = config.h_max * 2.0
    # Non-crossings get pushed to large_val (won't dominate softmin)
    z_candidate = z_cross + (1.0 - cross_weight) * large_val
    # Weighted softmin: -1/beta * log(sum(w * exp(-beta * z)))
    log_weights = jnp.log(cross_weight + 1e-30) - beta * z_candidate
    h_pbl = -jax.nn.logsumexp(log_weights, axis=1) / beta

    # Blend with smooth-method fallback when no clear crossing exists
    total_cross_weight = jnp.sum(cross_weight, axis=1)
    blend = jax.nn.sigmoid(config.sharpness * (total_cross_weight - 0.1))  # coeff-ok: PBL-crossing weight threshold
    h_fallback = diagnose_pbl_height(T, q_v, u, v, p_full, z_full, config)
    h_pbl = blend * h_pbl + (1.0 - blend) * h_fallback

    return jnp.clip(h_pbl, config.h_min, config.h_max)
