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
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
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
    time_integrator: str = "ssp_rk3"


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

class CDGridShallowWaterModel(IntegrationMixin):
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
        self._target_mass = None  # Set on first step for drift-free conservation

    def set_initial_mass(self, state: CDGridShallowWaterState):
        """Anchor conservation fixer to initial state mass.

        Call once before time integration to prevent cumulative mass drift
        in long runs (AMIP/CMIP). Without this, the fixer anchors to the
        previous step, and O(eps) errors accumulate over millions of steps.
        """
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    def _sync_dgrid_boundary(self, state: CDGridShallowWaterState):
        """Synchronize D-grid boundary winds between cubed-sphere faces.

        Converts D-grid corners to geographic (east/north) first — this is
        safe to average across faces — then averages to A-grid centres,
        converts to face-local, uses ``agrid_to_dgrid_vector`` (which does
        proper cross-face vector halo exchange), and replaces ONLY the
        boundary rows/columns.
        """
        from legoesm.core.operators_cdgrid import agrid_to_dgrid_vector

        u_d, v_d = state.u_d, state.v_d
        ca_c = self.cdgrid.cos_angle_corner
        sa_c = self.cdgrid.sin_angle_corner

        # 1. D-grid corners → geographic (safe to average across faces)
        u_east_d = ca_c * u_d - sa_c * v_d
        v_north_d = sa_c * u_d + ca_c * v_d

        # 2. Average geographic velocities to A-grid centres
        u_east_a = 0.25 * (u_east_d[:, :-1, :-1] + u_east_d[:, 1:, :-1]
                           + u_east_d[:, :-1, 1:] + u_east_d[:, 1:, 1:])
        v_north_a = 0.25 * (v_north_d[:, :-1, :-1] + v_north_d[:, 1:, :-1]
                            + v_north_d[:, :-1, 1:] + v_north_d[:, 1:, 1:])

        # 3. Geographic A-grid → face-local A-grid
        ca = self.grid.cos_angle
        sa = self.grid.sin_angle
        u_a = ca * u_east_a + sa * v_north_a
        v_a = -sa * u_east_a + ca * v_north_a

        # 4. A-grid → D-grid with cross-face vector halo exchange
        u_d_sync, v_d_sync = agrid_to_dgrid_vector(u_a, v_a, self.cdgrid)

        # 5. Replace boundary rows/columns only
        n = self.grid.n
        u_new = u_d.at[:, 0, :].set(u_d_sync[:, 0, :])
        u_new = u_new.at[:, n, :].set(u_d_sync[:, n, :])
        u_new = u_new.at[:, :, 0].set(u_d_sync[:, :, 0])
        u_new = u_new.at[:, :, n].set(u_d_sync[:, :, n])

        v_new = v_d.at[:, 0, :].set(v_d_sync[:, 0, :])
        v_new = v_new.at[:, n, :].set(v_d_sync[:, n, :])
        v_new = v_new.at[:, :, 0].set(v_d_sync[:, :, 0])
        v_new = v_new.at[:, :, n].set(v_d_sync[:, :, n])

        return state._replace(u_d=u_new, v_d=v_new)

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

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Synchronize D-grid boundary corners across cubed-sphere faces
        state_new = self._sync_dgrid_boundary(state_new)

        # Conservation fixer — anchored to initial mass when available,
        # otherwise to previous step's mass.
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            # Use initial target mass if set (drift-free for long runs);
            # otherwise anchor to previous step.
            if self._target_mass is not None:
                mass_target = self._target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            state_new = state_new._replace(h=h_fixed)

        return state_new

