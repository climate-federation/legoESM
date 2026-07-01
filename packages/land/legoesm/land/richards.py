"""Richards equation solver — mixed-form Picard iteration.

Solves the 1D vertical unsaturated flow equation:
    ∂θ/∂t = ∂/∂z [K(ψ) · (∂ψ/∂z + 1)] - S(z)

Uses the Celia et al. (1990) mass-conservative mixed-form discretization
with Picard iteration for nonlinearity.

The PDE above is written in z-up notation, but the discretization uses an
index-increasing-downward soil grid (layer ``k`` sits above layer ``k+1``).
Both the gravity and capillary terms use this same index direction, so the
signs are internally consistent and the result is correct.

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
from legoesm.timestepping.tridiagonal import thomas_solve


__param_spec__ = {
    "RichardsConfig": {
        "scheme_key": "land.richards",
        "excluded": {
            "theta_tol": "numerics: Newton convergence tolerance",
            "pond_max": "numerics: surface ponding cap before overland runoff [m]",
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
    # Surface ponding: max depth [m] held on the surface before it overflows to
    # runoff (overland flow).  Excess precip ponds up to this depth (a coupled
    # surface cell) and infiltrates on later steps; above it, overflows to runoff.
    pond_max: float = 0.05


class RichardsOutput(NamedTuple):
    """Output from the Richards equation solver."""
    psi_new: jnp.ndarray       # (ncol, n_layers) updated matric potential [m]
    theta_new: jnp.ndarray     # (ncol, n_layers) updated water content [m3/m3]
    runoff_surface: jnp.ndarray   # (ncol,) surface runoff [kg/m2/s]
    runoff_subsurface: jnp.ndarray  # (ncol,) subsurface runoff [kg/m2/s]
    n_iter: jnp.ndarray        # (ncol,) always equals max_iter (fixed-iteration solver)
    surface_water: jnp.ndarray  # (ncol,) updated surface ponding depth [m]


def solve_richards(
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    grid: SoilGrid,
    hydro_config: SoilHydraulicsConfig,
    richards_config: RichardsConfig,
    flux_top: jnp.ndarray,
    sink: jnp.ndarray,
    dt: float,
    surface_water: jnp.ndarray | None = None,
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

    # --- Coupled surface ponding cell (ParFlow/CliMA overland store) ---
    # A surface water store h_s [m] is solved IMPLICITLY together with the soil
    # column.  The surface<->soil flux is an internal Darcy face flux
    #     q01 = K_sat * ((h_s - psi_0) / half0 + 1),
    # so a DRY surface (psi_0 << h_s) infiltrates fast via its suction (recharge),
    # and a SATURATED surface infiltrates at ~K_sat (the over-saturation drains DOWN,
    # not back up).  Because q01 enters the surface-cell balance and the layer-0
    # balance with OPPOSITE signs, the column water budget closes BY CONSTRUCTION at
    # convergence, with no Hortonian infiltration cap, no theta clip, and no max()
    # mass-creation (the codex-flagged failure modes of the decoupled Robin BC).
    # h_s is eliminated by a Schur complement folded into layer-0's diagonal + rhs;
    # the surface balance is  dh_s/dt = flux_top - q01.  Overland runoff is NOT in
    # this balance: after Picard the converged h_s is split into pond (<= pond_max)
    # and runoff (the excess), which keeps  pond_new + runoff*dt == h_s  exactly.
    _half0 = 0.5 * dz[0]
    # Top-layer saturated conductivity, per column.  Broadcast K_sat against the
    # full (ncol, nlayers) state exactly as hydraulic_conductivity does, then take
    # layer 0 — this handles scalar, (ncol,1), (ncol,nlayers) and (nlayers,) configs
    # identically and never mis-maps layer variation onto columns (codex).
    _Ksat = jnp.broadcast_to(hydro_config.K_sat, theta.shape)[:, 0]   # (ncol,)
    _Kc = _Ksat / _half0                                    # surface conductance [1/s]
    _pond_max = richards_config.pond_max
    # Default zero pond carries flux_top's dtype so it never widens _work_dtype on its
    # own in a mixed-precision run (codex dtype hygiene; matches the _work_dtype care).
    h_s0 = (jnp.zeros_like(flux_top) if surface_water is None
            else jnp.broadcast_to(surface_water, (ncol,)))   # pond at time n [m]

    # Max infiltration the surface can supply this step [m/s] = existing pond drained
    # in one step + net input.  When flux_top < -h_s0/dt (evaporative demand exceeds
    # the pond), _avail_rate < 0 and the cap forces q01 < 0 — i.e. water flows soil ->
    # surface -> atmosphere.  That is the intended BARE-SOIL EVAPORATION: with no pond
    # the demand is met from the soil, and the cell reduces to the old Neumann flux_top
    # top BC (q01 == flux_top).  flux_top is already supply-limited upstream by the
    # top-layer evaporative-resistance throttle, so this does not over-extract.
    _avail_rate = h_s0 / dt + flux_top

    def _surface_terms(h_s, psi0):
        """Return q01 (infiltration into layer 0 [m/s]), its conductance Kc_eff =
        -dq01/dpsi0, the surface residual R_s, and D_s = dR_s/dh_s, at the iterate.

        q01 is the Darcy flux K_sat*((h_s-psi0)/half0 + 1) CAPPED at the available
        surface supply (existing pond + net input).  The cap is the nonnegative-depth
        complementarity: with no pond and no input the supply is 0, so a dry surface
        cannot drive phantom suction-infiltration into the soil (which would create
        water).  Capped, q01 is constant in (h_s, psi0) so its conductance is 0."""
        q01_robin = _Ksat * ((h_s - psi0) / _half0 + 1.0)
        capped = q01_robin > _avail_rate
        q01 = jnp.where(capped, _avail_rate, q01_robin)
        Kc_eff = jnp.where(capped, 0.0, _Kc)                 # -dq01/dpsi0 = dq01/dh_s
        # No overland-runoff sink in the surface balance: h_s accumulates the full
        # (input - infiltration), and the post-Picard step splits the converged
        # h_s into pond (<= pond_max) and runoff (the excess).  Putting the runoff
        # in the balance HERE would drain water the reported runoff cannot see
        # (h_s would converge to ~pond_max, so max(h_s-pond_max,0) reads ~0) — a
        # silent leak.  Splitting post-hoc keeps pond_new + runoff*dt == h_s exactly.
        R_s = (h_s - h_s0) / dt - flux_top + q01
        D_s = 1.0 / dt + Kc_eff                              # dR_s/dh_s (>0)
        return q01, Kc_eff, R_s, D_s

    if richards_config.bottom_bc not in ("free_drainage", "zero_flux"):
        raise ValueError(
            f"Unknown Richards bottom_bc {richards_config.bottom_bc!r}; "
            "expected one of: 'free_drainage', 'zero_flux'."
        )

    # --- Picard iteration ---
    psi_m = psi  # iterate

    def picard_body(m, carry):
        h_s_m, psi_m, theta_m = carry

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
        # with F_{k+1/2} = coeff_k · (psi_k − psi_{k+1}) the downward
        # Darcy flux (index-increasing-downward grid) at interface
        # k+1/2: positive F drains layer k into deeper layer k+1.
        # Boundary cells (k=0 and
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

        # Top BC: coupled surface ponding cell, eliminated by a Schur complement.
        # The surface unknown h_s satisfies  D_s*dh_s - Kc*dpsi_0 = -R_s ; eliminating
        # dh_s = (-R_s + Kc*dpsi_0)/D_s folds into layer 0:
        #   diag_0 += Kc/dz0 * (1 - Kc/D_s)      (implicit infiltration conductance,
        #                                         damped by the surface storage)
        #   rhs_0  += q01/dz0 - Kc*R_s/(dz0*D_s) (explicit infiltration + storage)
        # q01 appears with opposite signs in the surface and layer-0 balances, so the
        # eliminated system is mass-conservative by construction.
        # Kc_eff is the LINEARIZED conductance -dq01/dpsi0 (= _Kc when the Darcy flux
        # is below the surface supply, 0 when capped) — use it, not _Kc, so a capped
        # (supply-limited) cell decouples from the soil and adds only an explicit
        # source q01/dz0, conserving exactly.
        q01_m, Kc_m, R_s_m, D_s_m = _surface_terms(h_s_m, psi_m[:, 0])
        diag = diag.at[:, 0].add(Kc_m / dz[0] * (1.0 - Kc_m / D_s_m))
        rhs = rhs.at[:, 0].add(q01_m / dz[0] - Kc_m * R_s_m / (dz[0] * D_s_m))

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

        dpsi = thomas_solve(a_full, diag, c_full, rhs)

        # Recover the surface-cell increment from the Schur relation, then update.
        dh_s = (-R_s_m + Kc_m * dpsi[:, 0]) / D_s_m
        h_s_new = h_s_m + dh_s

        # Update psi and theta unconditionally.  Converged columns get
        # near-zero dpsi, so extra iterations are effectively no-ops.
        psi_new = psi_m + dpsi
        # No theta clip: theta_from_psi is bounded below at theta_r by the van-
        # Genuchten asymptote, and ABOVE theta_sat it is the physical elastic /
        # ponding storage (theta_sat + S_s*theta_sat*psi).  Clipping to theta_sat
        # silently destroyed that ponded water (non-conservative); the specific-
        # storage variable switch makes the clip unnecessary.
        theta_new = theta_from_psi(psi_new, hydro_config)

        return h_s_new, psi_new, theta_new

    theta_m_init = theta_from_psi(psi_m, hydro_config)

    # The Picard body upcasts ``psi_m + dpsi`` to the working precision implied
    # by EVERY float that feeds the Thomas solve (the grid spacings, the
    # infiltration/sink forcing, and the van-Genuchten hydraulics via
    # ``theta_m_init``), so the fori_loop OUTPUT carry is that dtype.  If the
    # soil state arrives as float32 in an x64 run (a downcast somewhere upstream
    # between coupled segments), the INPUT carry would be float32 while the
    # output is float64 -> "scan body carry input and output must have equal
    # types" at compile.  Promote the initial carry to the result type of all
    # those contributors so input == output for ANY input precision — narrowing
    # this to ``(psi_m, dz)`` alone would still mismatch a mixed-dtype config
    # whose hydraulics are wider than ``dz`` (codex).  Byte-identical for a
    # uniform float64 / true float32 run.
    _work_dtype = jnp.result_type(
        psi_m, theta_n, theta_m_init, dz, dz_if, flux_top, _Ksat, h_s0, sink)
    h_s_final, psi_final, theta_final = jax.lax.fori_loop(
        0, richards_config.max_iter,
        picard_body,
        (h_s0.astype(_work_dtype), psi_m.astype(_work_dtype),
         theta_m_init.astype(_work_dtype)),
    )
    # Fixed iteration count (no convergence check; always equals max_iter)
    n_iter_final = jnp.full(ncol, float(richards_config.max_iter))

    # Surface store + overland runoff, split from the converged surface cell.  The
    # surface balance R_s = 0 gives dh_s/dt = flux_top - q01, so the soil's +q01 and
    # the pond's -q01 cancel and d(soil + pond) = flux_top - sink - drainage exactly.
    # The cap q01 <= h_s0/dt + flux_top guarantees h_s_final >= 0 AT convergence; with
    # the fixed (non-converged) 10-iteration Picard, h_s_final can dip slightly
    # negative on Picard slack.  Clamping that to 0 would CREATE water (codex), so the
    # negative slack (the over-infiltration the pond could not actually supply) is
    # returned to soil layer 0 below — pond_new + runoff*dt + soil_debit == h_s_final
    # exactly, conserving regardless of Picard convergence.
    h_pos = jnp.maximum(h_s_final, 0.0)
    pond_deficit = h_pos - h_s_final                        # = max(-h_s_final, 0) >= 0
    runoff_surface = jnp.maximum(h_pos - richards_config.pond_max, 0.0) / dt
    surface_water_new = jnp.minimum(h_pos, richards_config.pond_max)
    # Un-infiltrate the (Picard-slack) over-draw so the pond clamp creates no water.
    # O(slack) for realistic forcing; psi_final is left as-is (the O(slack) psi/theta
    # mismatch at layer 0 re-equilibrates on the next step's Picard solve).
    theta_final = theta_final.at[:, 0].add(-pond_deficit / dz[0])

    # Subsurface runoff: gravitational drainage at bottom.  Evaluate K on the FULL
    # (ncol, nlayers) state, then slice the bottom layer — a layer-varying K_sat
    # ((nlayers,) or (ncol,nlayers)) cannot broadcast against a sliced (ncol,1) input,
    # and slicing first would pair the bottom psi with the WRONG layer's K_sat (codex).
    # Bit-identical for scalar / (ncol,1) configs; correct for layer-varying ones.
    if richards_config.bottom_bc == "free_drainage":
        K_bot = hydraulic_conductivity(psi_final, theta_final, hydro_config)[:, -1]
        runoff_subsurface = K_bot  # [m/s]
    else:
        runoff_subsurface = jnp.zeros_like(flux_top)  # dtype-matched (codex)

    # Convert runoff from m/s of water to kg/m2/s
    runoff_surface_kgm2s = runoff_surface * constants.rho_water
    runoff_subsurface_kgm2s = runoff_subsurface * constants.rho_water

    return RichardsOutput(
        psi_new=psi_final,
        theta_new=theta_final,
        runoff_surface=runoff_surface_kgm2s,
        runoff_subsurface=runoff_subsurface_kgm2s,
        n_iter=n_iter_final,
        surface_water=surface_water_new,
    )


