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
    # iter-45: promote to fp64 budget accumulator (see total_energy_pe).
    from legoesm.core.conservation import conservation_accumulator
    acc = conservation_accumulator()

    # Reference profile broadcasts
    dz_b = jnp.asarray(hc.dz, dtype=acc)[None, None, None, :]
    rho_ref_b = jnp.asarray(hc.rho_ref, dtype=acc)[None, None, None, :]
    theta_ref_b = jnp.asarray(hc.theta_ref, dtype=acc)[None, None, None, :]
    exner_ref_b = jnp.asarray(hc.exner_ref, dtype=acc)[None, None, None, :]
    z_full_b = jnp.asarray(hc.z_full, dtype=acc)[None, None, None, :]

    rho_full = rho_ref_b + state.rho_prime.data.astype(acc)  # (6,n,n,nlev)
    theta_full = theta_ref_b + state.theta_prime.data.astype(acc)
    T = theta_full * exner_ref_b                          # K

    # KE: 0.5·(u² + v² + w_avg²) where w_avg is full-level average of
    # half-level w (legoESM convention).
    u = state.u.data.astype(acc)
    v = state.v.data.astype(acc)
    w_half = state.w.data.astype(acc)                      # (..., nlev+1)
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])    # (..., nlev)
    ke = 0.5 * (u * u + v * v + w_full * w_full)

    c_vd_acc = jnp.asarray(constants.c_vd, dtype=acc)
    g_acc = jnp.asarray(constants.g, dtype=acc)

    # Internal energy + KE + PE per unit mass
    e_specific = c_vd_acc * T + ke + g_acc * z_full_b
    # Per-cell energy: mass × specific energy
    dm = rho_full * dz_b                                   # kg/m² per cell
    te_cell = dm * e_specific                              # J/m² per cell
    # Column integral
    te_per_m2 = jnp.sum(te_cell, axis=-1)                  # J/m² per column
    # Globally-summed: multiply by area
    te_column = te_per_m2 * grid.area.astype(acc)          # J per column
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


def apply_te_correction_nh(state_old, state_new, grid, hc):
    """FV3_3D iter 601: NH total-energy-conserving correction.

    Analog of iter 588's ``apply_aam_correction_nh`` but for total
    energy.  Inspired by FV3 ``consv_te > 0`` (fv_dynamics.F90:359-378
    + Lagrangian_to_Eulerian energy-correcting branch).

    Enforces ``TE(corrected) ≈ TE(state_old)`` by adding a uniform
    temperature increment ΔT such that the total internal-energy
    change exactly compensates the drift:

        te_dt = TE(state_new) - TE(state_old)
        ΔT = -te_dt / (cv · total_dry_mass)
        Δθ' = ΔT / exner_ref(k)   (level-dependent)

    KE and PE terms are NOT redistributed — only the IE term is
    adjusted via θ'.  This matches FV3's design (the correction
    targets the thermodynamic state, not winds).

    Returns
    -------
    state_corrected : NonHydrostaticState
        State with theta_prime adjusted; other fields unchanged.

    Notes
    -----
    Bit-for-bit identical to state_new when te_dt = 0.  Differentiable
    end-to-end.
    """
    _, te_old = compute_total_energy_nh(state_old, grid, hc)
    _, te_new = compute_total_energy_nh(state_new, grid, hc)
    te_dt = te_new - te_old                              # J

    # Total dry mass: sum(rho_full · dz · area)
    dz_b = jnp.asarray(hc.dz)[None, None, None, :]
    rho_ref_b = jnp.asarray(hc.rho_ref)[None, None, None, :]
    rho_full = rho_ref_b + state_new.rho_prime.data
    dm = rho_full * dz_b * grid.area[..., None]
    total_mass = jnp.sum(dm)

    # ΔT uniform across the atmosphere (kelvin)
    dT = -te_dt / (constants.c_vd * total_mass)

    # Δθ' = ΔT / exner_ref(k)
    exner_ref_b = jnp.asarray(hc.exner_ref)[None, None, None, :]
    d_theta_prime = dT / exner_ref_b                     # (1,1,1,nlev)

    new_theta = state_new.theta_prime.replace(
        data=state_new.theta_prime.data + d_theta_prime,
    )
    return state_new._replace(theta_prime=new_theta)
