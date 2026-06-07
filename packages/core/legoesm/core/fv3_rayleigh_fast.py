"""FV3_3D iter 448: FV3 ``Ray_fast`` (fast Rayleigh friction).

Port of FV3 ``dyn_core.F90:2922-3020``.  Column Rayleigh
damping at top of model for u, v, w (NH) or u_d, v_d (PE).

FV3 formula::

    sday = 86400  ! seconds per day
    tau0 = tau * sday

    for k in 0..nlev-1:
        if pfull(k) < rf_cutoff:
            rff(k) = dt/tau0 * sin²(π/2 ·
                log(rf_cutoff/pfull(k)) /
                log(rf_cutoff/ptop))
            rff(k) = 1.0 / (1.0 + rff(k))
        else:
            rff(k) = 1.0

    u *= rff, v *= rff, w *= rff

The sin² profile gives smooth onset at p=rf_cutoff (rff=1, no
damping) and saturates near p=ptop (max damping).
"""
from __future__ import annotations


import jax.numpy as jnp

from legoesm import constants


def compute_rff_profile(
    pfull: jnp.ndarray,
    ptop: float,
    rf_cutoff: float,
    tau_days: float,
    dt: float,
) -> jnp.ndarray:
    """Compute FV3 Ray_fast multiplier ``rff(k)`` per level.

    Parameters
    ----------
    pfull : jax.Array, shape (nlev,)
        Reference pressure at full levels [Pa].
    ptop : float
        Pressure at model top [Pa].
    rf_cutoff : float
        Cutoff pressure [Pa]; levels with pfull < rf_cutoff get
        damping.
    tau_days : float
        Rayleigh friction timescale [days].
    dt : float
        Integration time step [s].

    Returns
    -------
    jax.Array, shape (nlev,)
        ``rff(k)`` multiplier; ``1.0`` where no damping is
        applied (pfull >= rf_cutoff), < 1.0 where damped.
    """
    sday = 86400.0
    tau0 = tau_days * sday
    # Mask out pfull >= rf_cutoff: those levels get rff=1 (no damping).
    # For numerical safety, clamp pfull to (eps, rf_cutoff) so the
    # log ratio is well-defined; the mask zeroes the damp where
    # pfull >= rf_cutoff.
    pfull_safe = jnp.clip(pfull, 1.0e-6, rf_cutoff - 1.0e-6)
    log_ratio_num = jnp.log(rf_cutoff / pfull_safe)
    log_ratio_den = jnp.log(rf_cutoff / ptop)
    sin_arg = 0.5 * jnp.pi * log_ratio_num / log_ratio_den
    rff_raw = (dt / tau0) * jnp.sin(sin_arg) ** 2
    rff = 1.0 / (1.0 + rff_raw)
    # Apply only where pfull < rf_cutoff; else rff=1 (no damping).
    return jnp.where(pfull < rf_cutoff, rff, 1.0)


def pfull_from_exner(
    exner_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Convert reference Exner profile to reference pressure.

    Π = (p/p_0)^(R_d/c_p) → p = p_0 * Π^(c_p/R_d).
    """
    return constants.p_ref * exner_ref ** (constants.c_pd / constants.R_d)
