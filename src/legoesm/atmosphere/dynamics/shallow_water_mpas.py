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
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
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
# RK4 time integrator (MPAS default)
# ============================================================================

def _rk4_step(state, tendency_fn, dt):
    """Classical 4th-order Runge-Kutta step.

    Parameters
    ----------
    state : pytree
    tendency_fn : callable
    dt : float

    Returns
    -------
    pytree : state advanced by dt
    """
    k1 = tendency_fn(state)
    s1 = jax.tree.map(lambda s, k: s + 0.5 * dt * k, state, k1)

    k2 = tendency_fn(s1)
    s2 = jax.tree.map(lambda s, k: s + 0.5 * dt * k, state, k2)

    k3 = tendency_fn(s2)
    s3 = jax.tree.map(lambda s, k: s + dt * k, state, k3)

    k4 = tendency_fn(s3)
    state_new = jax.tree.map(
        lambda s, a, b, c, d: s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        state, k1, k2, k3, k4,
    )
    return state_new


# ============================================================================
# Model class
# ============================================================================

class MPASShallowWaterModel:
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
        """Advance one time step.

        Parameters
        ----------
        state : MPASShallowWaterState
        dt : float

        Returns
        -------
        MPASShallowWaterState
        """
        def tendency_fn(s):
            tend = mpas_shallow_water_tendencies(
                s, self.mesh, self.config, dt=dt)
            # Return same pytree structure as state for RK arithmetic
            return MPASShallowWaterState(
                h=s.h.replace(data=tend.dh_dt.data),
                u=s.u.replace(data=tend.du_dt.data),
                h_s=s.h_s.replace(data=jnp.zeros_like(s.h_s.data)),
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("rk4", "runge_kutta_4"):
            state_new = _rk4_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk3", "ssp3", "rk3"):
            state_new = ssp_rk3_step(state, tendency_fn, dt)
        else:
            raise ValueError(
                f"Unsupported time_integrator={self.config.time_integrator!r}")

        # Conservation fixers
        if self.config.fix_mass:
            state_new = _fix_mass_mpas(state_new, state, self.mesh)
        if self.config.fix_energy:
            state_new = _fix_energy_mpas(
                state_new, state, self.mesh, self.config.g)

        return state_new

    def integrate(
        self,
        state: MPASShallowWaterState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[MPASShallowWaterState, list[MPASShallowWaterState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : MPASShallowWaterState
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state, trajectory
        """
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: MPASShallowWaterState,
        n_steps: int,
        dt: float,
    ) -> tuple[MPASShallowWaterState, MPASShallowWaterState]:
        """Integrate using jax.lax.scan (differentiable).

        Parameters
        ----------
        state : MPASShallowWaterState
        n_steps : int
        dt : float

        Returns
        -------
        final_state, trajectory
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps))
        return final_state, trajectory


# ============================================================================
# Conservation fixers
# ============================================================================

def _fix_mass_mpas(state_new, state_old, mesh):
    """Fix mass conservation: uniform additive correction to h."""
    area = mesh.areaCell
    mass_old = jnp.sum(state_old.h.data * area)
    mass_new = jnp.sum(state_new.h.data * area)
    total_area = jnp.sum(area)
    correction = (mass_old - mass_new) / total_area
    h_fixed = state_new.h.replace(data=state_new.h.data + correction)
    return state_new._replace(h=h_fixed)


def _fix_energy_mpas(state_new, state_old, mesh, g):
    """Fix energy conservation: velocity scaling."""
    area = mesh.areaCell
    dc = mesh.dcEdge
    dv = mesh.dvEdge

    def total_energy(state):
        h = state.h.data
        u = state.u.data
        h_s = state.h_s.data
        # KE at cells
        ke = kinetic_energy_cell(u, mesh) * h
        pe = 0.5 * g * (h + h_s) ** 2
        return jnp.sum((ke + pe) * area)

    E_old = total_energy(state_old)

    h_new = state_new.h.data
    u_new = state_new.u.data
    KE_cells = kinetic_energy_cell(u_new, mesh)
    KE_new = jnp.sum(KE_cells * h_new * area)
    PE_new = jnp.sum(0.5 * g * (h_new + state_new.h_s.data) ** 2 * area)

    KE_target = jnp.maximum(E_old - PE_new, 0.0)
    scale = jnp.where(KE_new > 1e-30, jnp.sqrt(KE_target / KE_new), 1.0)

    u_fixed = state_new.u.replace(data=u_new * scale)
    return state_new._replace(u=u_fixed)
