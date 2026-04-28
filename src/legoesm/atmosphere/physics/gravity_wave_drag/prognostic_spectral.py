"""Prognostic spectral gravity wave drag parameterization.

Multi-azimuthal, multi-wavenumber spectral GWD with a prognostic
wave spectrum. Carries the wave flux array forward in time with a
relaxation timescale back to the launch source.

Uses jax.lax.scan for the vertical propagation, fully differentiable.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    PrognosticSpectralConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def prognostic_spectral_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: PrognosticSpectralConfig,
    spectrum_in: jax.Array,
) -> Tuple[GWDOutput, jax.Array]:
    """Compute prognostic spectral GWD tendencies.

    Parameters
    ----------
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
        Standard GWD backend signature.
    spectrum_in : jax.Array
        Input wave spectrum, shape (ncol, n_azimuths, n_wavenumbers).

    Returns
    -------
    GWDOutput
        Tendencies.
    spectrum_new : jax.Array
        Updated wave spectrum.
    """
    ncol, nlev = u.shape
    n_az = config.n_azimuths
    n_wn = config.n_wavenumbers

    # Brunt-Väisälä frequency
    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    dz_full = jnp.clip(dz_full, 1.0, None)
    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    N2_half = jnp.clip(N2_half, 1e-8, None)
    N_half = jnp.sqrt(N2_half)

    N_full = jnp.concatenate([
        N_half[:, :1],
        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
        N_half[:, -1:],
    ], axis=1)  # (ncol, nlev)

    # Wavenumber grid (log-spaced)
    k_grid = jnp.exp(jnp.linspace(
        jnp.log(config.k_min), jnp.log(config.k_max), n_wn
    ))  # (n_wn,)

    # Azimuthal directions
    azimuths = jnp.linspace(0.0, 2.0 * jnp.pi, n_az, endpoint=False)  # (n_az,)
    cos_az = jnp.cos(azimuths)  # (n_az,)
    sin_az = jnp.sin(azimuths)

    # Wind projection per azimuth: (ncol, n_az, nlev)
    U_proj = (
        u[:, None, :] * cos_az[None, :, None]
        + v[:, None, :] * sin_az[None, :, None]
    )

    # Phase speed per wavenumber: c = N / k
    # (ncol, nlev, n_wn) via broadcast
    c_phase = N_full[:, :, None] / jnp.clip(k_grid[None, None, :], 1e-10, None)

    # Layer thickness
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz = jnp.clip(dz, 1.0, None)  # (ncol, nlev)

    # Pressure thickness
    dp = jnp.abs(p_half[:, 1:] - p_half[:, :-1])  # (ncol, nlev)
    dp = jnp.clip(dp, 1.0, None)

    # Bottom-up scan per (azimuth, wavenumber)
    # For each spectral component, propagate flux upward
    # Saturation: tau_sat = threshold * rho * (c - U_proj)^3 / (N * 2*pi/k)
    # We scan over levels from surface to top

    # Precompute saturation stress: (ncol, n_az, n_wn, nlev)
    # c_phase: (ncol, nlev, n_wn) -> (ncol, n_wn, nlev) via transpose
    c_phase_t = jnp.transpose(c_phase, (0, 2, 1))  # (ncol, n_wn, nlev)

    # intrinsic: (ncol, n_az, n_wn, nlev)
    intrinsic = (
        c_phase_t[:, None, :, :]     # (ncol, 1, n_wn, nlev)
        - U_proj[:, :, None, :]       # (ncol, n_az, 1, nlev)
    )
    intrinsic_abs = jnp.clip(jnp.abs(intrinsic), 0.1, None)

    wavelength = 2.0 * jnp.pi / jnp.clip(k_grid, 1e-10, None)  # (n_wn,)
    N_4d = N_full[:, None, None, :]  # (ncol, 1, 1, nlev)
    rho_4d = rho[:, None, None, :]

    tau_sat = (
        config.breaking_threshold * rho_4d * intrinsic_abs ** 3
        / (jnp.clip(N_4d, 1e-6, None) * wavelength[None, None, :, None])
    )  # (ncol, n_az, n_wn, nlev)
    tau_sat = jnp.clip(tau_sat, _EPS, None)

    # Scan from surface (level -1) to top (level 0)
    # Flatten spectral dims for scan: (ncol * n_az * n_wn,)
    F_init = spectrum_in.reshape(ncol * n_az * n_wn)
    tau_sat_flat = tau_sat.reshape(ncol * n_az * n_wn, nlev)
    dp_flat = jnp.broadcast_to(
        dp[:, None, None, :], (ncol, n_az, n_wn, nlev)
    ).reshape(ncol * n_az * n_wn, nlev)

    def scan_fn(carry, k_rev):
        F_carry = carry
        k = nlev - 1 - k_rev
        f_break = jax.nn.sigmoid(
            config.breaking_sharpness * (F_carry - tau_sat_flat[:, k])
        )
        F_new = F_carry * (1.0 - f_break) + tau_sat_flat[:, k] * f_break
        drag_deposit = (F_carry - F_new) / jnp.clip(dp_flat[:, k], 1.0, None)
        return F_new, drag_deposit

    _, drag_stack = jax.lax.scan(scan_fn, F_init, jnp.arange(nlev))
    # drag_stack: (nlev, ncol*n_az*n_wn)
    drag_4d = drag_stack.T.reshape(ncol, n_az, n_wn, nlev)
    drag_4d = drag_4d[:, :, :, ::-1]  # reverse to top-first

    # Sum over spectrum to get (ncol, nlev) tendencies
    # Weight by azimuthal direction for du/dv
    # drag is stress gradient -> acceleration = -drag_deposit (already divided by dp)
    # Convert from dp-based to dz-based: multiply by dp/(rho*dz) -> just -drag
    du_dt_spec = -drag_4d * cos_az[None, :, None, None]  # (ncol, n_az, n_wn, nlev)
    dv_dt_spec = -drag_4d * sin_az[None, :, None, None]

    du_dt = jnp.sum(du_dt_spec, axis=(1, 2))  # (ncol, nlev)
    dv_dt = jnp.sum(dv_dt_spec, axis=(1, 2))

    # Frictional heating
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    # Prognostic spectrum update: relax toward launch source.  Pin the
    # broadcast dtype to the input spectrum dtype so the relaxation
    # stays at the input precision (defaulting to ``jnp.full`` allows
    # x64 mode to silently promote the spectrum to f64 even when the
    # state is f32).
    launch_source = jnp.full(
        (ncol, n_az, n_wn), config.launch_flux, dtype=spectrum_in.dtype,
    )
    spectrum_new = spectrum_in + dt * (launch_source - spectrum_in) / config.tau_decay

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd), spectrum_new
