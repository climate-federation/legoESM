"""MPAS shallow water solver on Voronoi meshes.

Implements the vector-invariant form of the shallow water equations
using the TRiSK discretization from Ringler et al. (2010).

Equations (vector-invariant form):
    dh/dt = -div(h_edge * u)
    du/dt = -grad(KE + g*(h + h_s)) + F_q + diffusion

where F_q is the PV flux term (energy or enstrophy conserving).

References
----------
- Ringler, T. D., et al. (2010). A unified approach to energy conservation
  and PV dynamics for arbitrarily-structured C-grids. JCP 229, 3065-3090.
- Skamarock, W. C., et al. (2012). A multiscale nonhydrostatic atmospheric
  model using centroidal Voronoi tessellations and C-grid staggering.
  Mon. Wea. Rev., 140, 3090-3105.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)

from legoesm.core.field import Field
from legoesm.core.state import MPASShallowWaterState, MPASShallowWaterTendencies
from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    kinetic_energy_cell,
    potential_vorticity_vertex,
    pv_flux_energy_conserving,
    pv_flux_enstrophy_conserving,
    thickness_flux,
    vector_laplacian_del2,
    vector_laplacian_del4,
    apvm_correction,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class MPASShallowWaterConfig(NamedTuple):
    """Configuration for the MPAS shallow water model."""
    g: float = constants.g
    nu_del2: float = 0.0        # del2 viscosity coefficient [m²/s]
    nu_del4: float = 0.0        # del4 viscosity coefficient [m⁴/s]
    pv_scheme: str = "energy"   # "energy" or "enstrophy"
    thickness_order: int = 2    # 2 (centered) or 3 (upwind-biased)
    apvm_scale: float = 0.0    # APVM upwinding (0 = off, 1 = full)
    fix_mass: bool = True
    fix_energy: bool = False
    time_integrator: str = "rk4"  # "rk4", "ssp_rk3"


def mpas_shallow_water_tendencies(
    state: MPASShallowWaterState,
    mesh: VoronoiMesh,
    config: MPASShallowWaterConfig = MPASShallowWaterConfig(),
    dt: float = 0.0,
) -> MPASShallowWaterTendencies:
    """Compute tendencies for the MPAS shallow water equations.

    Parameters
    ----------
    state : MPASShallowWaterState
    mesh : VoronoiMesh
    config : MPASShallowWaterConfig
    dt : float
        Time step (needed for APVM correction).

    Returns
    -------
    MPASShallowWaterTendencies
    """
    h = state.h.data       # (nCells,)
    u = state.u.data       # (nEdges,)
    h_s = state.h_s.data   # (nCells,)
    g = config.g

    # --- Thickness tendency: dh/dt = -div(h_edge * u) ---
    h_flux = thickness_flux(h, u, mesh, order=config.thickness_order)
    dh_dt_data = -divergence_cell(h_flux, mesh)

    # --- Kinetic energy at cells ---
    ke = kinetic_energy_cell(u, mesh)  # (nCells,)

    # --- Bernoulli function: B = KE + g*(h + h_s) ---
    bernoulli = ke + g * (h + h_s)

    # --- Pressure/Bernoulli gradient ---
    grad_B = gradient_edge(bernoulli, mesh)  # (nEdges,)

    # --- Potential vorticity ---
    q = potential_vorticity_vertex(u, h, mesh.fVertex, mesh)

    # Apply APVM if requested
    if config.apvm_scale > 0 and dt > 0:
        q = apvm_correction(q, u, mesh, config.apvm_scale * dt)

    # --- PV flux ---
    if config.pv_scheme == "enstrophy":
        F_pv = pv_flux_enstrophy_conserving(u, h, q, mesh)
    else:
        F_pv = pv_flux_energy_conserving(u, h, q, mesh)

    # --- Velocity tendency ---
    du_dt_data = -grad_B + F_pv

    # --- Diffusion ---
    if config.nu_del2 > 0:
        del2_u = vector_laplacian_del2(u, mesh)
        du_dt_data = du_dt_data + config.nu_del2 * del2_u

    if config.nu_del4 > 0:
        del4_u = vector_laplacian_del4(u, mesh)
        du_dt_data = du_dt_data + config.nu_del4 * del4_u

    dh_dt = state.h.replace(data=dh_dt_data, name="dh_dt", units="m/s")
    du_dt = state.u.replace(data=du_dt_data, name="du_dt", units="m/s^2")

    return MPASShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt)


# ============================================================================
# Model class
# ============================================================================

class MPASShallowWaterModel(IntegrationMixin):
    """MPAS shallow water model using TRiSK discretization.

    Parameters
    ----------
    mesh : VoronoiMesh
    config : MPASShallowWaterConfig, optional
    """

    def __init__(
        self,
        mesh: VoronoiMesh,
        config: MPASShallowWaterConfig | None = None,
    ):
        self.mesh = mesh
        self.config = config or MPASShallowWaterConfig()

    def tendencies(
        self,
        state: MPASShallowWaterState,
        dt: float = 0.0,
    ) -> MPASShallowWaterTendencies:
        return mpas_shallow_water_tendencies(
            state, self.mesh, self.config, dt=dt)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MPASShallowWaterState, dt: float) -> MPASShallowWaterState:
        """Advance one time step."""
        from legoesm.core.precision import cast_pytree
        state = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            tend = mpas_shallow_water_tendencies(
                s, self.mesh, self.config, dt=dt)
            # Return same pytree structure as state for RK arithmetic
            return MPASShallowWaterState(
                h=s.h.replace(data=tend.dh_dt.data),
                u=s.u.replace(data=tend.du_dt.data),
                h_s=s.h_s.replace(data=jnp.zeros_like(s.h_s.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Conservation fixers
        if self.config.fix_mass:
            state_new = _fix_mass_mpas(state_new, state, self.mesh)
        if self.config.fix_energy:
            state_new = _fix_energy_mpas(
                state_new, state, self.mesh, self.config.g)

        return cast_pytree(state_new, None, "storage")

    # integrate() and integrate_scan() inherited from IntegrationMixin


# ============================================================================
# Conservation fixers
# ============================================================================

def _fix_mass_mpas(state_new, state_old, mesh):
    """Fix mass conservation: uniform additive correction to h.

    Upcasts to float64 for the global reduction to avoid catastrophic
    cancellation in the mass difference (float32 sums lose ~7 digits).
    Three sums batched into one allreduce for multi-rank scaling.
    """
    area = mesh.areaCell.astype(jnp.float64)
    local = jnp.stack([
        jnp.sum(state_old.h.data.astype(jnp.float64) * area),
        jnp.sum(state_new.h.data.astype(jnp.float64) * area),
        jnp.sum(area),
    ])
    if jax.process_count() > 1:
        from legoesm.parallel.reductions import global_sum_mpi
        local = global_sum_mpi(local)
    mass_old, mass_new, total_area = local[0], local[1], local[2]
    correction = (mass_old - mass_new) / total_area
    h_fixed = state_new.h.replace(
        data=state_new.h.data + correction.astype(state_new.h.data.dtype),
    )
    return state_new._replace(h=h_fixed)


def _fix_energy_mpas(state_new, state_old, mesh, g):
    """Fix energy conservation: velocity scaling.

    The four contributing sums (E_old, KE_new, PE_new, plus KE_old +
    PE_old that go into E_old) are computed locally and reduced
    together — one MPI allreduce instead of four when the mesh is
    sharded across ranks.
    """
    area = mesh.areaCell

    def _ke_pe_terms(state):
        h = state.h.data
        u = state.u.data
        h_s = state.h_s.data
        ke = jnp.sum(kinetic_energy_cell(u, mesh) * h * area)
        pe = jnp.sum(0.5 * g * (h + h_s) ** 2 * area)
        return ke, pe

    KE_old, PE_old = _ke_pe_terms(state_old)
    KE_new, PE_new = _ke_pe_terms(state_new)

    local = jnp.stack([KE_old, PE_old, KE_new, PE_new])
    if jax.process_count() > 1:
        from legoesm.parallel.reductions import global_sum_mpi
        local = global_sum_mpi(local)
    KE_old, PE_old, KE_new, PE_new = local[0], local[1], local[2], local[3]
    E_old = KE_old + PE_old

    KE_target = jnp.maximum(E_old - PE_new, 0.0)
    scale = jnp.where(KE_new > _TINY, jnp.sqrt(KE_target / KE_new), 1.0)

    u_fixed = state_new.u.replace(data=state_new.u.data * scale)
    return state_new._replace(u=u_fixed)
