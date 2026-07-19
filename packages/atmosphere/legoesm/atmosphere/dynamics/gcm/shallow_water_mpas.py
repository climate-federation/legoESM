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

from legoesm import constants
from legoesm.core.conservation import fix_energy_mpas, fix_mass_mpas
from legoesm.core.operators_voronoi import (
    apvm_correction,
    kinetic_energy_cell,
    potential_vorticity_vertex,
    pv_flux_energy_conserving,
    pv_flux_enstrophy_conserving,
    thickness_flux,
    vector_laplacian_del2,
    vector_laplacian_del4,
)
from legoesm.core.precision import cast_pytree
from legoesm.core.state import MPASShallowWaterState, MPASShallowWaterTendencies
from legoesm.grids.operator_adapters import mpas_edge_operators
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin


class MPASShallowWaterConfig(NamedTuple):
    """Configuration for the MPAS shallow water model."""
    g: float = constants.g
    nu_del2: float = 0.0        # del2 viscosity coefficient [m²/s]
    nu_del4: float = 0.0        # del4 viscosity coefficient [m⁴/s]
    pv_scheme: str = "energy"   # "energy" or "enstrophy"
    thickness_order: int = 2    # 2 (centered) or 3 (upwind-biased)
    apvm_scale: float = 0.0    # APVM upwinding (0 = off, 1 = full)
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False  # Mirror PE/SW: anchor fixer to initial mass
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

    # The grid-dispatched edge-normal operator interface (B2): the TRiSK
    # divergence/gradient flow through ``ops`` rather than the bare free functions,
    # so the core calls operators without naming the mesh (design L2).  The adapter
    # delegates to the same core.operators_voronoi kernels -> byte-identical (its
    # delegation is pinned in test_grid_operators_adapter).
    ops = mpas_edge_operators(mesh)

    # --- Thickness tendency: dh/dt = -div(h_edge * u) ---
    h_flux = thickness_flux(h, u, mesh, order=config.thickness_order)
    dh_dt_data = -ops.divergence(h_flux)

    # --- Kinetic energy at cells ---
    ke = kinetic_energy_cell(u, mesh)  # (nCells,)

    # --- Bernoulli function: B = KE + g*(h + h_s) ---
    bernoulli = ke + g * (h + h_s)

    # --- Pressure/Bernoulli gradient ---
    grad_B = ops.gradient(bernoulli)  # (nEdges,)

    # --- Potential vorticity ---
    q = potential_vorticity_vertex(u, h, mesh.fVertex, mesh)

    # Apply APVM if requested
    if config.apvm_scale > 0 and dt > 0:
        q = apvm_correction(q, u, mesh, config.apvm_scale * dt)

    # --- PV flux ---
    if config.pv_scheme == "energy":
        F_pv = pv_flux_energy_conserving(u, h, q, mesh)
    elif config.pv_scheme == "enstrophy":
        F_pv = pv_flux_enstrophy_conserving(u, h, q, mesh)
    else:
        raise ValueError(
            f"Unknown pv_scheme {config.pv_scheme!r}; "
            "expected one of: 'energy', 'enstrophy'."
        )

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
        # iter-6: anchored mass target (fp64, lazy).  Mirrors
        # ``CGridLatLonShallowWaterModel._target_mass``.
        self._target_mass: jax.Array | None = None



    def compute_mass(self, state: MPASShallowWaterState) -> jax.Array:
        """Compute total dry mass ``∫ h dA`` in the fp64 budget acc."""
        return jnp.sum(
            state.h.data.astype(jnp.float64)
            * self.mesh.areaCell.astype(jnp.float64),
        )

    def tendencies(
        self,
        state: MPASShallowWaterState,
        dt: float = 0.0,
    ) -> MPASShallowWaterTendencies:
        return mpas_shallow_water_tendencies(
            state, self.mesh, self.config, dt=dt)

    def step(self, state: MPASShallowWaterState, dt: float) -> MPASShallowWaterState:
        """Outer wrapper: snapshots initial mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT)."""
        target_mass = self._target_mass
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and target_mass is None):
            target_mass = self.compute_mass(state)
            if not isinstance(target_mass, jax.core.Tracer):
                # Designed eager path: cache the concrete t=0 mass so
                # later segments keep anchoring to the same constant.
                self._target_mass = target_mass
            # Traced path (step() called inside an OUTER jit/grad/scan):
            # NEVER cache — a tracer stored on self leaks into the next
            # trace (UnexpectedTracerError; gh-417, same class as the
            # primitive_eq_cdgrid A1-gate bug).  Thread the per-call
            # pre-step mass instead; the fixer telescopes post-step mass
            # back to pre-step mass, matching the non-anchor semantics.
        return self._step_jit(state, dt, target_mass)

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: MPASShallowWaterState,
        dt: float,
        target_mass: jax.Array | None = None,
    ) -> MPASShallowWaterState:
        """Advance one time step."""
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

        # Conservation fixers — shared MPI-aware helpers in legoesm-core
        # (federation dedup; same algorithm, now Metal-safe via the policy-aware
        # accumulator), not a dycore-local copy.
        if self.config.fix_mass:
            state_new = fix_mass_mpas(
                state_new, state, self.mesh, target_mass=target_mass,
            )
        if self.config.fix_energy:
            state_new = fix_energy_mpas(
                state_new, state, self.mesh, self.config.g)

        return cast_pytree(state_new, None, "storage")

    # integrate() and integrate_scan() inherited from IntegrationMixin


# ============================================================================
# Conservation fixers
# ============================================================================

# The MPAS mass / energy conservation fixers live in the shared substrate
# (legoesm.core.conservation) — the same MPI-aware, batched-allreduce algorithm,
# made precision-policy-aware (Metal-safe) — imported above, not duplicated here.
