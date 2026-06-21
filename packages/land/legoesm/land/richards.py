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


__param_spec__ = {
    "RichardsConfig": {
        "scheme_key": "land.richards",
        "excluded": {
            "theta_tol": "numerics: Newton convergence tolerance",
        },
        "params": {
        },
    },
}


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
    ncol = psi.shape[0]
    dz = grid.dz                  # (nlayers,)
    dz_if = grid.dz_interface     # (nlayers-1,)

    theta_n = theta  # θ at time level n (saved for mass conservation)

    # --- Infiltration capacity ---
    # Darcy: q_max = K_top · (1 − dpsi/dz), with the head gradient
    # taken across the half-distance from the surface (assumed psi=0)
    # to the first node at depth ``z_node[0] = 0.5 · dz[0]``:
    #     grad = (psi[0] − 0) / (0.5 · dz[0]) = psi[0] / (0.5 · dz[0]).
    # For unsaturated soil (psi[0] < 0) the gradient is negative and
    # capacity > K_top (suction draws water down).  For ponded /
    # saturated surfaces (psi[0] ≥ 0) the gradient is positive and the
    # capacity is reduced — clamping to 0 prevents a positive head from
    # producing artificially-enhanced infiltration.  Using ``abs(psi)``
    # would always increase the capacity, which is wrong for ponded
    # cells (Codex GPT-5 review caught the sign).
    # Keep the column axis (``[:, :1]`` not ``[:, 0]``) so a PER-COLUMN
    # ``hydro_config`` (van-Genuchten fields shaped ``(ncol, 1)`` for spatial soil)
    # broadcasts; squeeze back to ``(ncol,)``.  Bit-identical for a scalar config.
    K_top = hydraulic_conductivity(psi[:, :1], theta[:, :1], hydro_config)[:, 0]
    head_grad = psi[:, 0] / (0.5 * dz[0])
    infil_capacity = jnp.maximum(K_top * (1.0 - head_grad), 0.0)

    # Surface runoff: excess over Darcy infiltration capacity.
    # Do NOT additionally cap by top-layer saturation — the implicit Picard
    # solve redistributes water downward, so capping here creates spurious
    # runoff before deeper layers can absorb the infiltrating water.
    flux_infiltrated = jnp.minimum(flux_top, infil_capacity)
    runoff_surface = jnp.maximum(flux_top - flux_infiltrated, 0.0)

    if richards_config.bottom_bc not in ("free_drainage", "zero_flux"):
        raise ValueError(
            f"Unknown Richards bottom_bc {richards_config.bottom_bc!r}; "
            "expected one of: 'free_drainage', 'zero_flux'."
        )

    # --- Picard iteration ---
    psi_m = psi  # iterate

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

        # RHS for the Picard iteration on dpsi = psi^{m+1} - psi^m:
        #   [C/dt - L^m] · dpsi = L^m psi^m - (theta^m - theta^n)/dt
        #                         + grav_div - sink + BCs
        # The previous formulation omitted the L^m psi^m term, so the
        # converged solution satisfied (theta - theta^n)/dt = grav_div
        # − sink + BCs — i.e. *gravity-drainage only*, with no capillary
        # redistribution.  Adding L^m psi^m closes the equation back to
        # the full Richards form (Celia 1990, eq. 17).
        rhs = -(theta_m - theta_n) / dt - sink

        # L^m psi^m as a flux divergence using the same coefficients
        # as the LHS matrix.  Express as
        #     (L psi)_k = (F_in_k − F_out_k) / dz_k
        # with F_{k+1/2} = coeff_k · (psi_k − psi_{k+1}) the upward
        # Darcy flux at interface k+1/2.  Boundary cells (k=0 and
        # k=N-1) naturally pick up only one flux contribution (the
        # missing interface flux is replaced by the explicit Neumann
        # BCs added below).  Two pads + one subtraction keeps the
        # trace size minimal vs. computing each flux side separately.
        F_iface = coeff * (psi_m[:, :-1] - psi_m[:, 1:])  # (ncol, N-1)
        F_in = jnp.pad(F_iface, ((0, 0), (1, 0)))
        F_out = jnp.pad(F_iface, ((0, 0), (0, 1)))
        rhs = rhs + (F_in - F_out) / dz

        # Gravitational flux: K_{k+1/2} enters from above, exits below.
        # Use ``jnp.pad`` instead of ``zeros + .at[].set`` — one Pad
        # HLO op vs alloc-then-scatter.  This block fires every Picard
        # iteration (up to 10) inside the land step.
        K_half_in = K_half / dz[1:]
        K_half_out = K_half / dz[:-1]
        grav_flux_in = jnp.pad(K_half_in, ((0, 0), (1, 0)))
        grav_flux_out = jnp.pad(K_half_out, ((0, 0), (0, 1)))
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
        # Assemble full arrays for Thomas algorithm via ``jnp.pad``
        # (one HLO op each vs ``zeros + .at[].set``).
        a_full = jnp.pad(sub, ((0, 0), (1, 0)))
        c_full = jnp.pad(sup, ((0, 0), (0, 1)))

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

    # Subsurface runoff: gravitational drainage at bottom.  Keep the column axis
    # (``[:, -1:]`` then squeeze) so a PER-COLUMN hydro_config (ncol,1) broadcasts —
    # same fix as the K_top boundary call; bit-identical for a scalar config.
    if richards_config.bottom_bc == "free_drainage":
        K_bot = hydraulic_conductivity(psi_final[:, -1:], theta_final[:, -1:],
                                       hydro_config)[:, 0]
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


