"""Richards equation solver — mixed-form modified Picard iteration (Task 8C).

Solves the 1D vertical unsaturated flow equation:
    ∂θ/∂t = ∂/∂z [K(ψ) · (∂ψ/∂z + 1)] - S(z)

Uses the Celia et al. (1990) mass-conservative mixed-form discretization
with Picard iteration for nonlinearity.

All operations are JAX-differentiable. The Picard loop uses
jax.lax.fori_loop with masked updates after convergence.

References
----------
- Celia et al. (1990): A general mass-conservative numerical solution for the
  unsaturated flow equation. Water Resources Research, 26(7), 1483-1496.
- Miller et al. (1998): A spatially distributed model for forestation and
  land-use change. Ecological Modelling, 108, 47-63.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.land.soil_grid import SoilGrid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig,
    theta_from_psi,
    hydraulic_conductivity,
    moisture_capacity,
    interblock_K,
)


class RichardsConfig(NamedTuple):
    """Configuration for the Richards equation solver."""
    max_iter: int = 10
    theta_tol: float = 1e-6       # convergence tolerance [m3/m3]
    bottom_bc: str = "free_drainage"  # "free_drainage" or "zero_flux"


class RichardsOutput(NamedTuple):
    """Output from the Richards equation solver."""
    psi_new: jnp.ndarray       # (ncol, n_layers) updated matric potential [m]
    theta_new: jnp.ndarray     # (ncol, n_layers) updated water content [m3/m3]
    runoff_surface: jnp.ndarray   # (ncol,) surface runoff [kg/m2/s]
    runoff_subsurface: jnp.ndarray  # (ncol,) subsurface runoff [kg/m2/s]
    n_iter: jnp.ndarray        # (ncol,) iterations used


def solve_richards(
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    grid: SoilGrid,
    hydro_config: SoilHydraulicsConfig,
    richards_config: RichardsConfig,
    flux_top: jnp.ndarray,
    sink: jnp.ndarray,
    dt: float,
) -> RichardsOutput:
    """Solve the Richards equation for one time step.

    Parameters
    ----------
    psi : jnp.ndarray
        Matric potential [m], shape (ncol, n_layers).
    theta : jnp.ndarray
        Volumetric water content [m3/m3], shape (ncol, n_layers).
    grid : SoilGrid
        Vertical soil grid.
    hydro_config : SoilHydraulicsConfig
        Hydraulic property parameters.
    richards_config : RichardsConfig
        Solver parameters.
    flux_top : jnp.ndarray
        Net downward flux at surface [m/s] (precip - evap in water depth).
        Shape (ncol,).
    sink : jnp.ndarray
        Root water uptake [m3/m3/s], shape (ncol, n_layers).
    dt : float
        Time step [s].

    Returns
    -------
    RichardsOutput
    """
    ncol, nlayers = psi.shape
    dz = grid.dz                  # (nlayers,)
    dz_if = grid.dz_interface     # (nlayers-1,)

    theta_n = theta  # θ at time level n (saved for mass conservation)

    # --- Infiltration capacity ---
    K_top = hydraulic_conductivity(psi[:, 0], theta[:, 0], hydro_config)
    psi_top_abs = jnp.abs(psi[:, 0])
    infil_capacity = K_top * (1.0 + psi_top_abs / dz[0])  # Darcy infiltration limit

    # Surface runoff: excess over infiltration capacity
    flux_infiltrated = jnp.minimum(flux_top, infil_capacity)
    # Also cap at what would saturate top layer in one dt
    max_flux = (hydro_config.theta_sat - theta[:, 0]) * dz[0] / dt
    flux_infiltrated = jnp.minimum(flux_infiltrated, jnp.maximum(max_flux, 0.0))
    runoff_surface = jnp.maximum(flux_top - flux_infiltrated, 0.0)

    # --- Picard iteration ---
    psi_m = psi  # iterate

    def picard_body(m, carry):
        psi_m, theta_m, converged = carry

        # Recompute hydraulic properties at current iterate
        K_m = hydraulic_conductivity(psi_m, theta_m, hydro_config)  # (ncol, nlayers)
        C_m = moisture_capacity(psi_m, theta_m, hydro_config)       # (ncol, nlayers)

        # Interblock conductivity (geometric mean)
        K_half = interblock_K(K_m[:, :-1], K_m[:, 1:])  # (ncol, nlayers-1)

        # Build tridiagonal system: [C/dt + A] * dpsi = rhs
        # A is the diffusion operator from Darcy's law

        # Sub-diagonal (lower), diagonal, super-diagonal (upper)
        a = jnp.zeros((ncol, nlayers))  # sub-diagonal
        b = jnp.zeros((ncol, nlayers))  # diagonal
        c = jnp.zeros((ncol, nlayers))  # super-diagonal
        rhs = jnp.zeros((ncol, nlayers))

        # Interior fluxes: q_{k+1/2} = K_{k+1/2} * [(psi_{k+1} - psi_k)/dz_{k+1/2} + 1]
        # The "+1" is gravitational drainage (z positive downward)

        # Diffusion coefficients
        coeff = K_half / dz_if  # (ncol, nlayers-1)

        # Diagonal: C/dt + contributions from above and below interfaces
        diag = C_m / dt
        # From interface above (k-1/2): for layers 1..nlayers-1
        diag = diag.at[:, 1:].add(coeff / dz[1:])
        # From interface below (k+1/2): for layers 0..nlayers-2
        diag = diag.at[:, :-1].add(coeff / dz[:-1])

        # Sub-diagonal: -K_{k-1/2} / (dz_if * dz_k)
        sub = -coeff / dz[1:]  # (ncol, nlayers-1)

        # Super-diagonal: -K_{k+1/2} / (dz_if * dz_k)
        sup = -coeff / dz[:-1]  # (ncol, nlayers-1)

        # RHS: -(theta_m - theta_n)/dt - sink + gravity flux divergence
        rhs = -(theta_m - theta_n) / dt - sink

        # Gravitational flux: K_{k+1/2} enters from above, exits below
        grav_flux_in = jnp.zeros((ncol, nlayers))
        grav_flux_out = jnp.zeros((ncol, nlayers))
        grav_flux_in = grav_flux_in.at[:, 1:].set(K_half / dz[1:])
        grav_flux_out = grav_flux_out.at[:, :-1].set(K_half / dz[:-1])
        rhs = rhs + (grav_flux_in - grav_flux_out)

        # Top BC: flux = flux_infiltrated (Neumann)
        rhs = rhs.at[:, 0].add(flux_infiltrated / dz[0])

        # Bottom BC
        if richards_config.bottom_bc == "free_drainage":
            # Gravitational flux only: q_bottom = K_N (downward)
            K_bot = K_m[:, -1]
            rhs = rhs.at[:, -1].add(-K_bot / dz[-1])
        # zero_flux: no additional term (natural BC)

        # Solve tridiagonal system: a*dpsi[k-1] + b*dpsi[k] + c*dpsi[k+1] = rhs
        # Assemble full arrays for Thomas algorithm
        a_full = jnp.zeros((ncol, nlayers))
        a_full = a_full.at[:, 1:].set(sub)
        c_full = jnp.zeros((ncol, nlayers))
        c_full = c_full.at[:, :-1].set(sup)

        dpsi = _thomas_solve_batch(a_full, diag, c_full, rhs)

        # Update psi and theta
        psi_new = psi_m + dpsi
        theta_new = theta_from_psi(psi_new, hydro_config)
        theta_new = jnp.clip(theta_new, hydro_config.theta_r, hydro_config.theta_sat)

        # Check convergence
        max_dtheta = jnp.max(jnp.abs(theta_new - theta_m), axis=1)  # (ncol,)
        newly_converged = max_dtheta < richards_config.theta_tol

        # Only update non-converged columns
        update_mask = ~converged
        psi_out = jnp.where(update_mask[:, None], psi_new, psi_m)
        theta_out = jnp.where(update_mask[:, None], theta_new, theta_m)
        converged_out = converged | newly_converged

        return psi_out, theta_out, converged_out

    converged_init = jnp.zeros(ncol, dtype=bool)
    theta_m_init = theta_from_psi(psi_m, hydro_config)
    theta_m_init = jnp.clip(theta_m_init, hydro_config.theta_r, hydro_config.theta_sat)

    psi_final, theta_final, converged_final = jax.lax.fori_loop(
        0, richards_config.max_iter,
        picard_body,
        (psi_m, theta_m_init, converged_init),
    )

    # Subsurface runoff: gravitational drainage at bottom
    if richards_config.bottom_bc == "free_drainage":
        K_bot = hydraulic_conductivity(psi_final[:, -1], theta_final[:, -1], hydro_config)
        runoff_subsurface = K_bot  # [m/s]
    else:
        runoff_subsurface = jnp.zeros(ncol)

    # Convert runoff from m/s of water to kg/m2/s
    rho_w = 1000.0
    runoff_surface_kgm2s = runoff_surface * rho_w
    runoff_subsurface_kgm2s = runoff_subsurface * rho_w

    return RichardsOutput(
        psi_new=psi_final,
        theta_new=theta_final,
        runoff_surface=runoff_surface_kgm2s,
        runoff_subsurface=runoff_subsurface_kgm2s,
        n_iter=jnp.sum(~converged_final).astype(jnp.float64) * jnp.ones(ncol),
    )


def _thomas_solve_batch(a, b, c, d):
    """Solve tridiagonal system for each column.

    Parameters
    ----------
    a, b, c, d : (ncol, nlayers) sub/main/super diagonal and RHS.

    Returns
    -------
    x : (ncol, nlayers) solution.
    """
    ncol, n = b.shape

    # Forward sweep
    def forward_step(carry, k):
        c_prev, d_prev = carry
        w = a[:, k] / jnp.clip(b[:, k] - a[:, k] * c_prev, 1e-30, None)
        # Actually this is wrong for the standard algorithm. Let me fix.
        # Standard Thomas:
        # c'[0] = c[0]/b[0], d'[0] = d[0]/b[0]
        # w = a[k] / (b[k] - a[k]*c'[k-1])
        # ... this is simpler with fori_loop

        return (c_prev, d_prev), None

    # Use the simpler fori_loop approach
    def solve_single(a_col, b_col, c_col, d_col):
        """Solve single column tridiagonal system."""
        # Forward elimination
        def fwd(carry, k):
            c_p, d_p = carry
            denom = b_col[k] - a_col[k] * c_p
            denom = jnp.where(jnp.abs(denom) < 1e-30,
                             jnp.sign(denom) * 1e-30 + 1e-30, denom)
            c_new = c_col[k] / denom
            d_new = (d_col[k] - a_col[k] * d_p) / denom
            return (c_new, d_new), (c_new, d_new)

        denom0 = jnp.where(jnp.abs(b_col[0]) < 1e-30, 1e-30, b_col[0])
        init = (c_col[0] / denom0, d_col[0] / denom0)
        _, (c_primes, d_primes) = jax.lax.scan(fwd, init, jnp.arange(1, n))

        # Prepend first values
        c_all = jnp.concatenate([jnp.array([init[0]]), c_primes])
        d_all = jnp.concatenate([jnp.array([init[1]]), d_primes])

        # Back substitution
        def bwd(x_next, k):
            x_k = d_all[k] - c_all[k] * x_next
            return x_k, x_k

        x_last = d_all[-1]
        _, x_rev = jax.lax.scan(bwd, x_last, jnp.arange(n - 2, -1, -1))
        x = jnp.concatenate([jnp.flip(x_rev), jnp.array([x_last])])
        return x

    # Vectorize over columns
    return jax.vmap(solve_single)(a, b, c, d)
