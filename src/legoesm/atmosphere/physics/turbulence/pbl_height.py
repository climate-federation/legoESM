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
        "smooth" (sigmoid-weighted) or "interp" (linear interpolation).
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

    Ri_b(z) = g * z * (theta_v(z) - theta_v_sfc) / (theta_v_sfc * V(z)^2)

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
    theta_v = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa * (
        1.0 + 0.61 * q_v
    )

    # Surface values (bottom level)
    theta_v_sfc = theta_v[:, -1]  # (ncol,)
    z_sfc = z_full[:, -1:]        # (ncol, 1)

    # Height above surface
    dz_from_sfc = jnp.abs(z_full - z_sfc) + 1.0  # (ncol, nlev), +1 avoids /0

    # Buoyancy difference from surface
    dtheta_v = theta_v - theta_v_sfc[:, None]

    # Wind speed squared (with floor to avoid division by zero)
    V2 = u ** 2 + v ** 2 + 1e-4

    # Bulk Richardson number
    Ri_bulk = (constants.g / jnp.clip(theta_v_sfc[:, None], 1.0, None)) * (
        dtheta_v * dz_from_sfc / V2
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
    weights = jax.nn.sigmoid(config.sharpness * (config.Ri_crit - Ri_bulk))

    # Weighted average height
    h_pbl = jnp.sum(z_full * weights, axis=1) / jnp.clip(
        jnp.sum(weights, axis=1), 1e-10, None
    )

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
    last sub-critical and first super-critical levels. Uses a smooth
    maximum to select the crossing in a JAX-traceable way.

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

    # Crossing occurs where Ri_below < Ri_crit and Ri_above >= Ri_crit
    crosses = (Ri_below < config.Ri_crit) & (Ri_above >= config.Ri_crit)

    # Linear interpolation fraction at each crossing
    dRi = Ri_above - Ri_below + 1e-20
    frac = jnp.clip((config.Ri_crit - Ri_below) / dRi, 0.0, 1.0)
    z_cross = z_below + frac * (z_above - z_below)

    # Select the first (lowest) crossing using soft-min
    # Weight crossings, pick the one closest to surface
    large_val = config.h_max * 2.0
    z_candidate = jnp.where(crosses, z_cross, large_val)
    h_pbl = jnp.min(z_candidate, axis=1)

    # If no crossing found, fall back to sigmoid method
    no_crossing = jnp.all(~crosses, axis=1)
    h_fallback = diagnose_pbl_height(T, q_v, u, v, p_full, z_full, config)
    h_pbl = jnp.where(no_crossing, h_fallback, h_pbl)

    return jnp.clip(h_pbl, config.h_min, config.h_max)
