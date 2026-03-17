"""Hydrostatic Primitive Equations on MPAS Voronoi meshes.

Implements the vector-invariant form of the hydrostatic PE using the
TRiSK discretization (Ringler et al. 2010) on the C-grid.  Reuses
the existing Voronoi operators and shape-agnostic vertical coordinate
functions (sigma and hybrid) from the core library.

Equations (per level k):
    du/dt = F_pv - grad(B_k) - R_d T_k grad(ln p_s) + visc + vert_adv
    dT/dt = -v·∇T - σ̇ ∂T/∂σ + κ T ω/p
    dp_s/dt = -(1/σ_range) ∫ div(p_s v) dσ

where B_k = KE + Φ_k (Bernoulli function), F_pv is the potential-
vorticity flux, and σ̇ is diagnosed from the continuity equation.

References
----------
- Ringler, T. D., et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
- Skamarock, W. C., et al. (2012). Mon. Wea. Rev., 140, 3090-3105.
- Simmons & Burridge (1981). Mon. Wea. Rev., 109, 758-766.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState, MPASHydrostaticTendencies
from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    kinetic_energy_cell,
    potential_vorticity_vertex,
    pv_flux_energy_conserving,
    pv_flux_enstrophy_conserving,
    vector_laplacian_del2,
    vector_laplacian_del4,
    edge_thickness,
    apvm_correction,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot,
    compute_mass_flux_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class MPASPrimitiveEquationConfig(NamedTuple):
    """Configuration for the MPAS hydrostatic primitive equation model."""
    g: float = constants.g
    nu_del2: float = 0.0          # del2 viscosity [m²/s]
    nu_del4: float = 0.0          # del4 viscosity [m⁴/s]
    K_h: float = 0.0              # scalar diffusion [m²/s]
    pv_scheme: str = "energy"     # "energy" or "enstrophy"
    apvm_scale: float = 0.0       # APVM upwinding (0 = off)
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"


# ============================================================================
# Tendency computation
# ============================================================================

def mpas_hydrostatic_tendencies(
    state: MPASHydrostaticState,
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    config: MPASPrimitiveEquationConfig = MPASPrimitiveEquationConfig(),
    physics_tendency: MPASHydrostaticTendencies | None = None,
    dt: float = 0.0,
) -> MPASHydrostaticTendencies:
    """Compute tendencies for the hydrostatic PE on an MPAS mesh.

    Parameters
    ----------
    state : MPASHydrostaticState
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig
    physics_tendency : MPASHydrostaticTendencies, optional
    dt : float
        Time step (needed for APVM correction).

    Returns
    -------
    MPASHydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u_3d = state.u.data        # (nEdges, nlev)
    T_3d = state.T.data        # (nCells, nlev)
    p_s = state.p_s.data       # (nCells,)
    phis = state.phis.data     # (nCells,)

    R_d = constants.R_d
    kappa = constants.kappa
    nlev = T_3d.shape[-1]
    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]

    # --- 1. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)   # (nCells, nlev)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T_3d, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T_3d, p_s, sigma_coord, phis)

    # --- 3. Precompute ln(p_s) gradient at edges ---
    ln_ps = jnp.log(p_s)
    grad_ln_ps = gradient_edge(ln_ps, mesh)  # (nEdges,)

    # --- Per-level tendencies via scan ---
    def _level_tendencies(carry, k):
        u_k = u_3d[:, k]        # (nEdges,)
        T_k = T_3d[:, k]        # (nCells,)
        Phi_k = Phi[:, k]       # (nCells,)
        p_k = p_full[:, k]      # (nCells,)

        # Kinetic energy
        ke = kinetic_energy_cell(u_k, mesh)  # (nCells,)

        # Bernoulli function: KE + Phi
        bernoulli = ke + Phi_k

        # Bernoulli gradient at edges
        grad_B = gradient_edge(bernoulli, mesh)  # (nEdges,)

        # Pressure gradient correction: R_d * T_edge * grad(ln p_s)
        T_edge = 0.5 * (T_k[c1] + T_k[c2])  # (nEdges,)
        pg_corr = R_d * T_edge * grad_ln_ps

        # PV flux
        # For the hydrostatic PE, h is proportional to dp/g.
        # Use a proxy thickness = dp/g (pressure thickness).
        if _hybrid:
            h_proxy = dp[:, k] / config.g  # (nCells,)
        else:
            h_proxy = p_s * sigma_coord.dsigma[k] / config.g

        q_v = potential_vorticity_vertex(u_k, h_proxy, mesh.fVertex, mesh)

        if config.apvm_scale > 0 and dt > 0:
            q_v = apvm_correction(q_v, u_k, mesh, config.apvm_scale * dt)

        if config.pv_scheme == "enstrophy":
            pv_flux = pv_flux_enstrophy_conserving(u_k, h_proxy, q_v, mesh)
        else:
            pv_flux = pv_flux_energy_conserving(u_k, h_proxy, q_v, mesh)

        # Momentum tendency
        du_dt_k = -grad_B - pg_corr + pv_flux

        # Viscosity
        if config.nu_del2 > 0:
            du_dt_k = du_dt_k + config.nu_del2 * vector_laplacian_del2(u_k, mesh)
        if config.nu_del4 > 0:
            du_dt_k = du_dt_k + config.nu_del4 * vector_laplacian_del4(u_k, mesh)

        # Divergence for continuity / sigma-dot
        div_k = divergence_cell(u_k, mesh)  # (nCells,)

        # Temperature advection: -v·∇T ≈ centered tracer flux form
        T_edge_centered = 0.5 * (T_k[c1] + T_k[c2])
        flux_T = u_k * T_edge_centered  # (nEdges,)
        div_uT = divergence_cell(flux_T, mesh)  # (nCells,)
        horiz_adv_T = -div_uT + T_k * div_k  # advective form

        # Scalar diffusion
        if config.K_h > 0:
            grad_T = gradient_edge(T_k, mesh)
            horiz_adv_T = horiz_adv_T + config.K_h * divergence_cell(grad_T, mesh)

        return carry, (du_dt_k, div_k, horiz_adv_T, p_k)

    _, (du_dt_all, div_all, horiz_adv_T_all, p_all) = jax.lax.scan(
        _level_tendencies, None, jnp.arange(nlev),
    )
    # scan outputs: (nlev, nEdges/nCells)
    du_dt_3d = jnp.moveaxis(du_dt_all, 0, -1)       # (nEdges, nlev)
    div_3d = jnp.moveaxis(div_all, 0, -1)            # (nCells, nlev)
    horiz_adv_T_3d = jnp.moveaxis(horiz_adv_T_all, 0, -1)  # (nCells, nlev)

    # --- 4. Surface pressure tendency and vertical velocity ---
    if _hybrid:
        div_ps_v = div_3d * p_s[:, None]  # approximate
        dp_s_dt = -jnp.sum(
            div_ps_v * dp / p_s[:, None], axis=-1,
        ) / sigma_coord.B_range

        mass_flux = compute_mass_flux_hybrid(div_3d, p_s, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T_3d, mass_flux, p_s, sigma_coord)
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt, sigma_coord)
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        D_total = jnp.sum(div_3d * dsigma, axis=-1)  # (nCells,)
        dp_s_dt = -p_s * D_total / sigma_range

        sigma_dot = compute_sigma_dot(div_3d, sigma_coord)
        vert_adv_T = vertical_advection(T_3d, sigma_dot, sigma_coord)
        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma_coord)

    # Vertical advection of u: approximate via edge-averaged sigma-dot
    if _hybrid:
        vert_adv_u = _vertical_advection_edge(u_3d, mass_flux, sigma_coord, mesh, hybrid=True, p_s=p_s)
    else:
        vert_adv_u = _vertical_advection_edge(u_3d, sigma_dot, sigma_coord, mesh, hybrid=False)

    du_dt_3d = du_dt_3d + vert_adv_u

    # --- 5. Thermodynamic equation ---
    # Adiabatic heating: κ·T·ω/p
    adiabatic = kappa * T_3d * omega / p_full

    # Missing v·∇(ln p_s) contribution:
    # Need u_edge * grad(ln p_s) averaged to cells.
    # flux = u * grad_ln_ps per edge, then divergence gives cell avg
    # Actually: v·∇(ln p_s) at cells ≈ div(u * ln_ps) - ln_ps * div(u)
    # Simpler: direct edge-to-cell reconstruction
    u_grad_lnps = u_3d * grad_ln_ps[:, None]  # (nEdges, nlev)

    def _cell_avg_scalar_product(k):
        """Reconstruct v·∇(ln p_s) at cells for level k."""
        # Use divergence of flux minus scalar times divergence
        flux_k = u_grad_lnps[:, k]  # nEdges
        # Weighted sum at cells using edge contributions
        eoc = mesh.edgesOnCell  # (maxEdges, nCells)
        mask = (eoc >= 0).astype(flux_k.dtype)
        eoc_safe = jnp.maximum(eoc, 0)
        # dcEdge * dvEdge gives edge area; simplify with area averaging
        vals = flux_k[eoc_safe] * mesh.dvEdge[eoc_safe] * mesh.edgeSignOnCell * mask
        # This is basically divergence * p_s of ln_ps... Actually let's use
        # the simpler approach: v·∇(ln p_s) ≈ (1/A_c) Σ_e u_e * grad_ln_ps_e * l_e
        # but grad_ln_ps is already the normal gradient.  The dot product
        # v·∇φ at cell c ≈ div(u * φ) - φ * div(u) = divergence_cell(u*ln_ps) - ln_ps * div(u)
        ln_ps_edge = 0.5 * (ln_ps[c1] + ln_ps[c2])
        return divergence_cell(u_3d[:, k] * ln_ps_edge, mesh) - ln_ps * div_3d[:, k]

    # Vectorize over levels
    _, v_grad_lnps_all = jax.lax.scan(
        lambda carry, k: (carry, _cell_avg_scalar_product(k)),
        None, jnp.arange(nlev),
    )
    v_grad_lnps = jnp.moveaxis(v_grad_lnps_all, 0, -1)  # (nCells, nlev)

    adiabatic = adiabatic + kappa * T_3d * v_grad_lnps

    dT_dt_3d = horiz_adv_T_3d + vert_adv_T + adiabatic

    # --- 6. Add physics tendencies ---
    if physics_tendency is not None:
        du_dt_3d = du_dt_3d + physics_tendency.du_dt.data
        dT_dt_3d = dT_dt_3d + physics_tendency.dT_dt.data
        dp_s_dt = dp_s_dt + physics_tendency.dp_s_dt.data

    return MPASHydrostaticTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT_dt_3d, name="dT_dt",
                    dims=("nCells", "nlev"), units="K/s"),
        dp_s_dt=Field(data=dp_s_dt, name="dp_s_dt",
                      dims=("nCells",), units="Pa/s"),
        dphis_dt=Field(data=jnp.zeros_like(phis), name="dphis_dt",
                       dims=("nCells",), units="m²/s³"),
    )


def _vertical_advection_edge(
    u_3d, vert_vel, sigma_coord, mesh, hybrid=False, p_s=None,
):
    """Vertical advection of edge-based velocity.

    Interpolates the vertical velocity (sigma-dot or mass flux) from
    cells to edges, then applies the standard vertical advection.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]

    if hybrid:
        # mass_flux is (nCells, nlev+1), average to edges
        mass_flux_edge = 0.5 * (vert_vel[c1] + vert_vel[c2])
        p_s_edge = 0.5 * (p_s[c1] + p_s[c2])
        return vertical_advection_hybrid(u_3d, mass_flux_edge, p_s_edge, sigma_coord)
    else:
        # sigma_dot is (nCells, nlev+1), average to edges
        sigma_dot_edge = 0.5 * (vert_vel[c1] + vert_vel[c2])
        return vertical_advection(u_3d, sigma_dot_edge, sigma_coord)


# ============================================================================
# Model class
# ============================================================================

class MPASPrimitiveEquationModel(IntegrationMixin):
    """Hydrostatic primitive equation model on an MPAS Voronoi mesh.

    Parameters
    ----------
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: MPASPrimitiveEquationConfig | None = None,
    ):
        self.mesh = mesh
        self.sigma_coord = sigma_coord
        self.config = config or MPASPrimitiveEquationConfig()

    def tendencies(
        self,
        state: MPASHydrostaticState,
        dt: float = 0.0,
        physics_tendency: MPASHydrostaticTendencies | None = None,
    ) -> MPASHydrostaticTendencies:
        return mpas_hydrostatic_tendencies(
            state, self.mesh, self.sigma_coord, self.config,
            physics_tendency=physics_tendency, dt=dt,
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def step(
        self,
        state: MPASHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> MPASHydrostaticState:
        """Advance one time step.

        Parameters
        ----------
        state : MPASHydrostaticState
        dt : float
        physics_fn : callable, optional
            Function (state, mesh, sigma_coord) -> MPASHydrostaticTendencies.

        Returns
        -------
        MPASHydrostaticState
        """
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.mesh, self.sigma_coord)
            tend = mpas_hydrostatic_tendencies(
                s, self.mesh, self.sigma_coord, self.config,
                physics_tendency=phys, dt=dt,
            )
            return MPASHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        if self.config.fix_mass:
            state_new = _fix_mass_mpas_hydro(state_new, state, self.mesh)

        return state_new

    # integrate() and integrate_scan() inherited from IntegrationMixin


def _fix_mass_mpas_hydro(state_new, state_old, mesh):
    """Fix mass conservation: uniform additive correction to p_s."""
    area = mesh.areaCell
    mass_old = jnp.sum(state_old.p_s.data * area)
    mass_new = jnp.sum(state_new.p_s.data * area)
    total_area = jnp.sum(area)
    correction = (mass_old - mass_new) / total_area
    p_s_fixed = state_new.p_s.replace(data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)
