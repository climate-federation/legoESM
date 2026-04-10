"""Richards equation solver — mixed-form Picard iteration.

Solves the 1D vertical unsaturated flow equation:
    ∂θ/∂t = ∂/∂z [K(ψ) · (∂ψ/∂z + 1)] - S(z)

Uses the Celia et al. (1990) mass-conservative mixed-form discretization
with Picard iteration for nonlinearity.

All operations are JAX-differentiable. The Picard loop uses
``jax.lax.fori_loop`` with a **fixed iteration count** (default 10).
No early-termination convergence check is performed: converged columns
simply get near-zero updates on subsequent iterations. The ``n_iter``
diagnostic always equals ``max_iter``.

References
----------
- Celia et al. (1990): A general mass-conservative numerical solution for the
  unsaturated flow equation. Water Resources Research, 26(7), 1483-1496.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.soil_grid import SoilGrid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig,
    theta_from_psi,
    hydraulic_conductivity,
    moisture_capacity,
    interblock_K,
)
from legoesm.land.tridiag import thomas_solve_batch


class RichardsConfig(NamedTuple):
    """Configuration for the Richards equation solver.

    The solver runs a fixed number of Picard iterations (``max_iter``)
    per time step.  ``theta_tol`` is retained for future use but is
    **not** checked during the loop.
    """
    max_iter: int = 10
    theta_tol: float = 1e-6       # reserved for future convergence check [m3/m3]
    bottom_bc: str = "free_drainage"  # "free_drainage" or "zero_flux"


class RichardsOutput(NamedTuple):
    """Output from the Richards equation solver."""
    psi_new: jnp.ndarray       # (ncol, n_layers) updated matric potential [m]
    theta_new: jnp.ndarray     # (ncol, n_layers) updated water content [m3/m3]
    runoff_surface: jnp.ndarray   # (ncol,) surface runoff [kg/m2/s]
    runoff_subsurface: jnp.ndarray  # (ncol,) subsurface runoff [kg/m2/s]
    n_iter: jnp.ndarray        # (ncol,) always equals max_iter (fixed-iteration solver)


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

    # Surface runoff: excess over Darcy infiltration capacity.
    # Do NOT additionally cap by top-layer saturation — the implicit Picard
    # solve redistributes water downward, so capping here creates spurious
    # runoff before deeper layers can absorb the infiltrating water.
    flux_infiltrated = jnp.minimum(flux_top, infil_capacity)
    runoff_surface = jnp.maximum(flux_top - flux_infiltrated, 0.0)

    # --- Picard iteration ---
    psi_m = psi  # iterate
    n_iter_count = jnp.zeros(ncol)  # per-column iteration counter

    def picard_body(m, carry):
        psi_m, theta_m = carry

        # Recompute hydraulic properties at current iterate
        K_m = hydraulic_conductivity(psi_m, theta_m, hydro_config)  # (ncol, nlayers)
        C_m = moisture_capacity(psi_m, theta_m, hydro_config)       # (ncol, nlayers)

        # Interblock conductivity (geometric mean)
        K_half = interblock_K(K_m[:, :-1], K_m[:, 1:])  # (ncol, nlayers-1)

        # Build tridiagonal system: [C/dt + A] * dpsi = rhs
        # A is the diffusion operator from Darcy's law

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

        dpsi = thomas_solve_batch(a_full, diag, c_full, rhs)

        # Update psi and theta unconditionally.  Converged columns get
        # near-zero dpsi, so extra iterations are effectively no-ops.
        psi_new = psi_m + dpsi
        theta_new = theta_from_psi(psi_new, hydro_config)
        theta_new = jnp.clip(theta_new, hydro_config.theta_r, hydro_config.theta_sat)

        return psi_new, theta_new

    theta_m_init = theta_from_psi(psi_m, hydro_config)
    theta_m_init = jnp.clip(theta_m_init, hydro_config.theta_r, hydro_config.theta_sat)

    psi_final, theta_final = jax.lax.fori_loop(
        0, richards_config.max_iter,
        picard_body,
        (psi_m, theta_m_init),
    )
    # Fixed iteration count (no convergence check; always equals max_iter)
    n_iter_final = jnp.full(ncol, float(richards_config.max_iter))

    # Subsurface runoff: gravitational drainage at bottom
    if richards_config.bottom_bc == "free_drainage":
        K_bot = hydraulic_conductivity(psi_final[:, -1], theta_final[:, -1], hydro_config)
        runoff_subsurface = K_bot  # [m/s]
    else:
        runoff_subsurface = jnp.zeros(ncol)

    # Convert runoff from m/s of water to kg/m2/s
    runoff_surface_kgm2s = runoff_surface * constants.rho_water
    runoff_subsurface_kgm2s = runoff_subsurface * constants.rho_water

    return RichardsOutput(
        psi_new=psi_final,
        theta_new=theta_final,
        runoff_surface=runoff_surface_kgm2s,
        runoff_subsurface=runoff_subsurface_kgm2s,
        n_iter=n_iter_final,
    )


