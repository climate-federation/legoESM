"""FV3_3D iter 597: NH compressible-Euler total-energy diagnostic.

Faithful port of FV3 ``compute_total_energy`` NH branch
(fv_mapz.F90:1154-1183).  Mass-weighted column integral of:

    cv·T + 0.5·(u² + v² + w²) + g·z

For NH legoESM:
- T = theta · exner  (where theta = theta_ref + theta_prime)
- rho · dz · area gives cell mass
- z = z_full[k] (mid-level height)

Returns per-column total energy + globally-summed total.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def compute_total_energy_nh(state, grid, hc) -> tuple[jax.Array, float]:
    """Compute per-column and total NH total energy.

    Faithful to FV3 fv_mapz.F90:1154-1183 (NH branch) using legoESM
    NH conventions:
    - u, v are face-local cell-centered winds (rotated internally to
      u_east, v_north for FV3-style KE).
    - rho_full = rho_ref + rho_prime
    - theta_full = theta_ref + theta_prime
    - T = theta_full · exner_ref (Π_ref since rho perturbation is small)
    - w is half-level; KE term uses w² averaged to full levels.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid (needs lat, area, angle)
    hc : HeightCoordinate (needs rho_ref, theta_ref, exner_ref, dz,
         z_full)

    Returns
    -------
    te_column : jax.Array, shape (6, n, n)
        Per-column total energy [J/m²].
    te_total : float
        Globally-integrated total energy [J].
    """
    # Reference profile broadcasts
    dz_b = jnp.asarray(hc.dz)[None, None, None, :]            # (1,1,1,nlev)
    rho_ref_b = jnp.asarray(hc.rho_ref)[None, None, None, :]
    theta_ref_b = jnp.asarray(hc.theta_ref)[None, None, None, :]
    exner_ref_b = jnp.asarray(hc.exner_ref)[None, None, None, :]
    z_full_b = jnp.asarray(hc.z_full)[None, None, None, :]

    rho_full = rho_ref_b + state.rho_prime.data            # (6,n,n,nlev)
    theta_full = theta_ref_b + state.theta_prime.data
    T = theta_full * exner_ref_b                          # K

    # KE: 0.5·(u² + v² + w_avg²) where w_avg is full-level average of
    # half-level w (legoESM convention).
    u = state.u.data
    v = state.v.data
    w_half = state.w.data                                  # (..., nlev+1)
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])    # (..., nlev)
    ke = 0.5 * (u * u + v * v + w_full * w_full)

    # Internal energy + KE + PE per unit mass
    e_specific = constants.c_vd * T + ke + constants.g * z_full_b
    # Per-cell energy: mass × specific energy
    dm = rho_full * dz_b                                   # kg/m² per cell
    te_cell = dm * e_specific                              # J/m² per cell
    # Column integral
    te_per_m2 = jnp.sum(te_cell, axis=-1)                  # J/m² per column
    # Globally-summed: multiply by area
    te_column = te_per_m2 * grid.area                       # J per column
    te_total = float(jnp.sum(te_column))
    return te_column, te_total


def te_drift_nh(state_old, state_new, grid, hc) -> float:
    """FV3_3D iter 599: NH total-energy tendency between two states.

    Returns ``te_dt = TE(state_new) - TE(state_old)``.  Adiabatic
    flat-surface runs should have small drift (the dycore is
    energy-conserving to discretization order).

    Companion to ``aam_drift_nh`` (iter 587) — both diagnose dycore
    conservation behavior without applying corrections.

    Returns
    -------
    te_dt : float
        Total-energy tendency [J].  Positive = energy gained;
        negative = energy lost (physical or numerical).
    """
    _, te_old = compute_total_energy_nh(state_old, grid, hc)
    _, te_new = compute_total_energy_nh(state_new, grid, hc)
    return te_new - te_old
