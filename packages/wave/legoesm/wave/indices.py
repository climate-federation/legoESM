"""Integrated wave parameters and surface Stokes drift.

Port of the diagnostic functions in NeuralSWM
``inference/jax_core.py`` (``_compute_wave_parameters_from_device_arrays``
and ``compute_stokes_drift_vector_jax``).

Direction convention: the spectral axis is the direction the waves
propagate TOWARDS, measured clockwise from north.  The source stores the
physics angle as ``theta = (180 - phi) mod 360``; every function here
converts back to ``phi`` first.

Diagnostic conventions and adaptations:

* The Stokes diagnostic retains the released diagnostic gravity constant,
  while reusing the safe implicit dispersion solver. Land uses finite
  substitute inputs before masking. Quadrature includes the complete band.
* The source names its outputs ``stokes_u`` (= the NORTHWARD component)
  and ``stokes_v`` (= the EASTWARD component).  This module returns
  correctly named ``stokes_east`` / ``stokes_north`` components instead;
  the magnitude is what KPP consumes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.wave.constants import STOKES_GRAVITY
from legoesm.core.bulk_flux import sqrt_zero_subgradient
from legoesm.wave.physics.dispersion import solve_wavenumber
from legoesm.wave.grid import WaveEnv
from legoesm.wave.networks.embedding import data_direction_radians

# Calm-sea floor for the mean-period denominator.  The ``where`` below already
# returns zero on an exactly calm spectrum; this only keeps a tiny non-zero
# energy from overflowing the division.  The floor must stay above ~1e-19: its
# square is the divisor of the quotient's m0-derivative, and a smaller floor
# underflows to zero in float32 and turns 0/0 into NaN in reverse mode.
_M0_FLOOR = 1e-15

# A mean direction is only defined when the directional resultant is a
# non-negligible fraction of the total energy.  An isotropic sea has
# ``cf == sf == 0`` up to roundoff; without this guard atan2 picks an
# arbitrary direction from that roundoff and its derivative is ~1e18.
_DIRECTION_MIN_RESULTANT = 1e-6


def _direction_and_spacing(env: WaveEnv) -> tuple[jax.Array, jax.Array]:
    theta = data_direction_radians(env).reshape(-1)
    dtheta = env.dth
    return theta, dtheta


def wave_parameters(spectra: jax.Array, env: WaveEnv) -> dict[str, jax.Array]:
    """Significant wave height, mean direction and mean period.

    ``swh`` [m] uses ``4 sqrt(m0)``; ``mwd`` [deg] is the direction the
    waves come FROM in the meteorological convention; ``mwp`` [s] is
    ``m_{-1} / m_0``.  A calm column (``m0 == 0``) returns zero height,
    zero period and zero direction with a zero subgradient, so the first
    coupled or training step does not start from an arbitrary direction
    and a huge period derivative.  The mean direction is also undefined
    for a directionally symmetric sea: those columns return zero, and the
    direction derivative is discontinuous at the resultant threshold.
    The spectral integral uses WADEN's midpoint finite-volume widths,
    including both endpoint half-cells.
    """
    spectra = jnp.asarray(spectra)
    if spectra.ndim != 5:
        raise ValueError(
            "wave_parameters expects spectra (B, Nlon, Nlat, Ndir, Nfreq); "
            f"got {spectra.shape}")
    freq = jnp.asarray(env.freq).reshape(-1)
    theta, dtheta = _direction_and_spacing(env)

    df = env.quadrature_df
    df_b = df.reshape((1,) * (spectra.ndim - 1) + (df.shape[0],))
    freq_b = freq.reshape((1,) * (spectra.ndim - 1) + (freq.shape[0],))
    cos_t = jnp.cos(theta).reshape((1,) * (spectra.ndim - 2) + (theta.shape[0], 1))
    sin_t = jnp.sin(theta).reshape((1,) * (spectra.ndim - 2) + (theta.shape[0], 1))

    weighted = spectra * df_b * dtheta
    m0 = jnp.sum(weighted, axis=(-2, -1))
    swh = 4.0 * sqrt_zero_subgradient(m0)

    cf = jnp.sum(weighted * cos_t, axis=(-2, -1))
    sf = jnp.sum(weighted * sin_t, axis=(-2, -1))
    # atan2(0, 0) has a NaN derivative; substitute a finite direction where
    # the directional resultant is negligible before the masked branch so
    # reverse mode cannot see 0*NaN.
    valid_direction = (
        (m0 > _M0_FLOOR)
        & (cf * cf + sf * sf > (_DIRECTION_MIN_RESULTANT * m0) ** 2))
    safe_cf = jnp.where(valid_direction, cf, 1.0)
    safe_sf = jnp.where(valid_direction, sf, 0.0)
    direction = jnp.rad2deg(jnp.arctan2(safe_sf, safe_cf) + jnp.pi) % 360.0
    mwd = jnp.where(valid_direction, direction, 0.0)

    m_neg1 = jnp.sum(weighted * (freq_b**-1), axis=(-2, -1))
    mwp = jnp.where(
        m0 > _M0_FLOOR, m_neg1 / jnp.maximum(m0, _M0_FLOOR), 0.0)
    return {"swh": swh, "mwd": mwd, "mwp": mwp, "m0": m0}


def _wavenumber_from_omega(omega: jax.Array, depth: jax.Array) -> jax.Array:
    """Safe implicit dispersion root with the Stokes diagnostic gravity."""
    return solve_wavenumber(omega/(2*jnp.pi), depth[..., None, None],
                            gravity=STOKES_GRAVITY)[..., 0, :]



def stokes_drift(spectra: jax.Array, env: WaveEnv, depth: jax.Array) -> dict[str, jax.Array]:
    """Surface Stokes drift components and magnitude [m/s].

    ``depth`` is ``(B, Nlon, Nlat)`` or ``(B, Nlon, Nlat, 1, 1)``.
    """
    spectra = jnp.asarray(spectra)
    if spectra.ndim != 5:
        raise ValueError(
            "stokes_drift expects spectra (B, Nlon, Nlat, Ndir, Nfreq); "
            f"got {spectra.shape}")
    # Accept exactly the two documented spellings.  A looser squeeze loop
    # interprets (Nlon, Nlat, 1) as (B, Nlon, Nlat) and silently puts
    # longitude on the batch axis.
    depth_3d = jnp.asarray(depth)
    if depth_3d.ndim == 3:
        pass
    elif depth_3d.ndim == 5 and depth_3d.shape[-2:] == (1, 1):
        depth_3d = depth_3d[..., 0, 0]
    else:
        raise ValueError(
            "stokes_drift expects depth (B, Nlon, Nlat) or "
            f"(B, Nlon, Nlat, 1, 1); got {depth_3d.shape}")
    if depth_3d.shape[1:] != spectra.shape[1:3]:
        raise ValueError(
            "stokes_drift expects depth with the spectra spatial shape: "
            f"{depth_3d.shape[1:]} vs {spectra.shape[1:3]}")
    if depth_3d.shape[0] not in (1, spectra.shape[0]):
        raise ValueError(
            "stokes_drift depth batch must be 1 or match the spectra: "
            f"{depth_3d.shape[0]} vs {spectra.shape[0]}")

    freq = jnp.asarray(env.freq).reshape(-1)
    theta, dtheta = _direction_and_spacing(env)
    df = env.quadrature_df
    omega = 2.0 * jnp.pi * freq
    wet = depth_3d > 0
    safe_depth = jnp.where(wet, depth_3d, 1.)
    k = _wavenumber_from_omega(omega, safe_depth)

    omega_b = omega.reshape((1,) * depth_3d.ndim + omega.shape)
    df_b = df.reshape((1,) * depth_3d.ndim + df.shape)
    weight = (
        2.0 * STOKES_GRAVITY * k
        / (omega_b * jnp.tanh(2.0 * k * safe_depth[..., None]) + 1e-16)
    ) * k
    weight = jnp.where(wet[..., None], weight * df_b * dtheta, 0.)
    weight = weight[..., None, :]

    cos_t = jnp.cos(theta).reshape((1,) * (spectra.ndim - 2) + (theta.shape[0], 1))
    sin_t = jnp.sin(theta).reshape((1,) * (spectra.ndim - 2) + (theta.shape[0], 1))
    east = jnp.sum(spectra * weight * sin_t, axis=(-2, -1))
    north = jnp.sum(spectra * weight * cos_t, axis=(-2, -1))
    return {
        "stokes_east": east,
        "stokes_north": north,
        "stokes_magnitude": sqrt_zero_subgradient(east**2 + north**2),
    }


__all__ = ["stokes_drift", "wave_parameters"]
