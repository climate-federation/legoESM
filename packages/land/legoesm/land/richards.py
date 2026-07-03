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
    psi_from_theta,
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

# --- numerics: explicit-flux CFL limiter (Courant 1928) ---
# Courant-number safety factor for the explicit gravity-drainage / infiltration
# fluxes.  The stability bound is K*dt/dz <= 1; we cap at 0.9 of it to leave a
# margin against the geometric-mean interface K and the fixed-iteration Picard.
# Applied identically to the interior interface K (K_half), the surface
# infiltration conductivity (_Ksat) and the free-drainage bottom flux (K_bot).
_CFL_SAFETY = 0.9

# --- numerics: dry-side effective-saturation floor (Richards degeneracy guard) ---
# As a layer dries toward theta_r the matric potential psi -> -inf and BOTH the
# moisture capacity C(psi)=dtheta/dpsi AND the conductivity K(psi) collapse to ~0.
# The Picard tridiagonal diagonal (C/dt + interface conductances) then degenerates,
# so dpsi = rhs/diag explodes and the Thomas solve returns NaN — reachable whenever a
# top layer is driven to theta_r under sustained ET with no infiltration (a dry-season
# / desert column).  Clamp psi each Picard iterate at the matric potential of a small
# effective saturation Se = _SE_DRY_FLOOR, i.e. bound the ARTIFICIAL retained water at
# _SE_DRY_FLOOR*(theta_sat - theta_r) ~ 3e-5 m3/m3 INDEPENDENT of the retention-curve
# shape.  A FIXED psi floor is NOT shape-robust: for a low-n_vg soil psi = -1e7 still
# maps to Se ~ 0.4 (O(0.1) spurious water); flooring on Se instead pins the water error
# regardless of (alpha, n) — the psi floor at Se is computed once per solve from the
# (possibly per-column) hydraulics via psi_from_theta.  CLM (smpmin) / Noah-MP use the
# analogous guard.  Only ever binds below wilting; the wet side (psi >= 0) is untouched.
_SE_DRY_FLOOR = 1.0e-4
# For a near-flat retention curve (n_vg -> 1) the psi at Se=_SE_DRY_FLOOR is astronomic
# (~ -1e80): FINITE in float64 but OVERFLOWS to -inf in float32, which would make
# jnp.maximum(psi, -inf) == psi and defeat the guard for a float32 land state (codex).
# Cap the floor magnitude at a value that is representable in BOTH precisions (|.| <<
# float32 max 3.4e38) and far below any realistic dry psi (~ -1e10), so it never binds
# for a normal soil but keeps the floor finite (and the guard effective) for the flat-
# curve edge — at the cost of a larger, still bounded, water error there (~1e-2).
_PSI_FLOOR_MIN = -1.0e30


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

    # Depth-decay of saturated conductivity (Niu et al. 2005): K_sat(z) =
    # K_sat * exp(-z/k_sat_decay_m), applied per layer to the WHOLE K(theta) curve
    # to impede drainage OUT OF the root zone.  0.0 = disabled (uniform K, exact
    # backward-compat).  Static Python gate on the config float (feature switch).
    if hydro_config.k_sat_decay_m > 0.0:
        _kdecay = jnp.exp(-grid.z_node / hydro_config.k_sat_decay_m)   # (nlayers,)
    else:
        _kdecay = jnp.ones_like(grid.z_node)

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
    _Ksat = jnp.broadcast_to(hydro_config.K_sat, theta.shape)[:, 0] * _kdecay[0]  # (ncol,)
    # CFL cap on the surface infiltration conductivity (stability at high K_sat).
    # The Robin infiltration q01 = _Ksat*((h_s-psi0)/half0 + 1) drives the layer-0
    # Schur update EXPLICITLY (rhs_0 += q01/dz0) and its linearized conductance
    # _Kc = _Ksat/half0 sets the surface coupling; with _Ksat >> dz0/dt (a sandy
    # K_sat ~ 4e-5 m/s at a 30-min step) that explicit source moves more than one
    # top cell of water per iteration and psi_0/theta_0 diverge via the unbounded
    # specific-storage term.  Cap _Ksat at the resolvable rate 0.9*dz0/dt — the same
    # CFL bound applied to the interior grav flux (K_half) below and the bottom
    # drainage (K_bot).  No-op for loam (K_sat ~ 3e-6 << dz0/dt); the water the
    # capped surface cannot infiltrate stays as ponding and leaves as overland
    # runoff in the post-Picard split, so mass is conserved.  z convention: dz>0
    # downward, q01>0 downward into the soil.
    _Ksat = jnp.minimum(_Ksat, _CFL_SAFETY * dz[0] / dt)
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

    # Config-aware dry-side psi floor (see _SE_DRY_FLOOR): matric potential at
    # Se = _SE_DRY_FLOOR for the active retention curve + (possibly per-column)
    # hydraulics, computed ONCE.  jnp.maximum(psi, _psi_dry_floor) then bounds the
    # dry limit with a water error <= _SE_DRY_FLOOR*(theta_sat-theta_r) for ANY (alpha, n).
    _psi_dry_floor = psi_from_theta(
        hydro_config.theta_r
        + _SE_DRY_FLOOR * (hydro_config.theta_sat - hydro_config.theta_r),
        hydro_config)
    # Keep the floor FINITE in the working precision (float32 overflows the flat-curve
    # psi to -inf, defeating the guard — see _PSI_FLOOR_MIN).  Never binds for a normal
    # soil (floor ~ -1e6 >> -1e30).
    _psi_dry_floor = jnp.maximum(_psi_dry_floor, _PSI_FLOOR_MIN)

    def picard_body(m, carry):
        h_s_m, psi_m, theta_m = carry

        # Recompute hydraulic properties at current iterate
        K_m = hydraulic_conductivity(psi_m, theta_m, hydro_config) * _kdecay[None, :]  # (ncol, nlayers)
        C_m = moisture_capacity(psi_m, theta_m, hydro_config)       # (ncol, nlayers)

        # Interblock conductivity (geometric mean)
        K_half = interblock_K(K_m[:, :-1], K_m[:, 1:])  # (ncol, nlayers-1)

        # CFL flux limiter (stability at high K_sat).  The gravity-drainage flux
        # is applied EXPLICITLY (grav_flux term in the rhs below), so it must not
        # move more than one cell of water per step: K_half * dt / dz <= 1.  For a
        # 30-min step a high-K sandy soil (K_sat ~ 4e-5 m/s >> dz/dt) violates this
        # and psi/theta run away via the UNBOUNDED specific-storage term (theta =
        # theta_sat + S_s*theta_sat*psi has no upper clip).  Cap the interface
        # conductivity at 0.9*min(dz_adjacent)/dt.  No-op for low-K soils
        # (loam K_sat ~ 3e-6 << dz/dt); mass-conserving (the excess drainage is
        # simply deferred to the next step, not discarded).
        K_half = jnp.minimum(K_half, _CFL_SAFETY * jnp.minimum(dz[:-1], dz[1:]) / dt)

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
            # Gravitational flux only: q_bottom = K_N (downward, z>0 down).  Applied
            # EXPLICITLY (rhs term), so CFL-cap it at the resolvable rate 0.9*dz_N/dt
            # exactly as the interior grav flux and the surface infiltration above.
            # dz_N is the thick bottom layer, so this rarely bites, but keeps every
            # explicit K path bounded so a high-K profile cannot drain > one cell/step.
            K_bot = jnp.minimum(K_m[:, -1], _CFL_SAFETY * dz[-1] / dt)
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

        # Update psi and theta unconditionally.  Converged columns get near-zero
        # dpsi, so extra iterations are effectively no-ops.  Clamp psi at the config-
        # aware dry-side floor (_psi_dry_floor, see _SE_DRY_FLOOR): a fully-dried
        # layer (C, K -> 0) would otherwise drive dpsi -> ±inf and NaN the Thomas
        # solve.  maximum() leaves the whole normal + wet range untouched.
        psi_new = jnp.maximum(psi_m + dpsi, _psi_dry_floor)
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
        # Must use the SAME CFL cap as the Picard bottom BC (line ~309), otherwise the
        # reported drainage exceeds what the solve actually removed and the soil-water
        # budget leaks.  0.9*dz_N/dt; no-op unless the bottom-layer K exceeds it.
        K_bot = hydraulic_conductivity(psi_final, theta_final, hydro_config)[:, -1] * _kdecay[-1]
        K_bot = jnp.minimum(K_bot, _CFL_SAFETY * dz[-1] / dt)
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


