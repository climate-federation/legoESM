"""FV3-style C-D grid Shallow Water Equations on the cubed-sphere.

True staggered C-D grid discretisation following Lin (2004):

* D-grid winds (cell corners, shape ``(6, n+1, n+1)``) are prognostic.
* C-grid velocities (cell edges) are diagnosed for mass transport.
* Vorticity is computed from circulation around cell boundaries
  (exact on the D-grid, avoids Hollingsworth-Kallberg instability).
* Bernoulli gradient uses Arakawa-Lamb 4-point formula at corners.
* Mass transport uses upwind (1st-order) face values with C-grid
  velocities naturally colocated at cell interfaces.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    cgrid_to_dgrid,
    cgrid_mass_flux_divergence,
    cdgrid_momentum_tendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants


# ==============================================================================
# State and Config
# ==============================================================================

class CDGridShallowWaterState(NamedTuple):
    """Shallow water state on the FV3 C-D grid.

    h : (6, n, n) — height at cell centres
    u_d : (6, n+1, n+1) — x-velocity at cell corners (D-grid)
    v_d : (6, n+1, n+1) — y-velocity at cell corners (D-grid)
    h_s : (6, n, n) — surface topography at cell centres
    """
    h: jax.Array
    u_d: jax.Array
    v_d: jax.Array
    h_s: jax.Array


class CDGridShallowWaterConfig(NamedTuple):
    """Configuration for C-D grid shallow water model."""
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0  # Biharmonic hyperdiffusion
    use_conservation_fixer: bool = True
    fix_mass: bool = True


# ==============================================================================
# Tendencies
# ==============================================================================

def cdgrid_shallow_water_tendencies(
    state: CDGridShallowWaterState,
    cdgrid: CubedSphereCDGrid,
    config: CDGridShallowWaterConfig = CDGridShallowWaterConfig(),
):
    """Compute C-D grid shallow water tendencies.

    Parameters
    ----------
    state : CDGridShallowWaterState
    cdgrid : CubedSphereCDGrid
    config : CDGridShallowWaterConfig

    Returns
    -------
    (dh_dt, du_d_dt, dv_d_dt) : tuple of jax.Array
    """
    h, u_d, v_d, h_s = state

    # 1. Mass transport via C-grid velocities
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

    # Zero-mean correction for mass conservation
    total_area = jnp.sum(cdgrid.base.area)
    dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # 2. Momentum tendencies (vector-invariant form)
    du_d_dt, dv_d_dt = cdgrid_momentum_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        g=config.g, A_h=config.A_h,
        hyperdiff_coeff=config.hyperdiff_coeff,
    )

    return dh_dt, du_d_dt, dv_d_dt


# ==============================================================================
# Model class
# ==============================================================================

class CDGridShallowWaterModel:
    """FV3-style C-D grid shallow water model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
        Base A-grid cubed-sphere.
    config : CDGridShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: CDGridShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()

    def tendencies(self, state: CDGridShallowWaterState):
        """Compute tendencies (pure function wrapper)."""
        return cdgrid_shallow_water_tendencies(
            state, self.cdgrid, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: CDGridShallowWaterState, dt: float,
    ) -> CDGridShallowWaterState:
        """Advance one time step using SSP-RK3."""
        def tendency_fn(s):
            dh, du, dv = cdgrid_shallow_water_tendencies(
                s, self.cdgrid, self.config,
            )
            return CDGridShallowWaterState(
                h=dh, u_d=du, v_d=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = ssp_rk3_step(state, tendency_fn, dt)

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            mass_old = jnp.sum(state.h * self.cdgrid.base.area)
            mass_new = jnp.sum(state_new.h * self.cdgrid.base.area)
            total_area = jnp.sum(self.cdgrid.base.area)
            h_fixed = state_new.h + (mass_old - mass_new) / total_area
            state_new = state_new._replace(h=h_fixed)

        return state_new

    def integrate(
        self,
        state: CDGridShallowWaterState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[CDGridShallowWaterState, list[CDGridShallowWaterState]]:
        """Integrate forward for a given duration."""
        n_steps = int(duration / dt)
        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory
