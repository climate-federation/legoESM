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
    div_damp: float = 0.0         # Divergence damping coefficient [m^2/s]
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
        div_damp=config.div_damp,
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

        FV3-style direct corner-to-corner sync: convert D-grid corner
        velocities to geographic (east/north) at each shared face edge,
        average with the neighbouring face's geographic velocity at each
        shared corner, then convert back to face-local.

        Edge corners (shared by 2 faces) get a pairwise average.
        Vertex corners (shared by 3 faces) get a 3-way average.
        """
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

        u_d, v_d = state.u_d, state.v_d
        n = self.grid.n
        ca_c = self.cdgrid.cos_angle_corner
        sa_c = self.cdgrid.sin_angle_corner

        # 1. Convert all corners to geographic (read-only reference)
        ue = ca_c * u_d - sa_c * v_d
        vn = sa_c * u_d + ca_c * v_d

        # 2. Edge sync: pairwise average (read originals, write to output)
        ue_out = ue
        vn_out = vn

        def _get_strip(arr, face, edge):
            if edge == WEST:    return arr[face, 0, :]
            elif edge == EAST:  return arr[face, n, :]
            elif edge == SOUTH: return arr[face, :, 0]
            else:               return arr[face, :, n]

        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                nbr_ue = _get_strip(ue, nbr_face, nbr_edge)
                nbr_vn = _get_strip(vn, nbr_face, nbr_edge)
                if is_reversed:
                    nbr_ue = nbr_ue[::-1]
                    nbr_vn = nbr_vn[::-1]
                local_ue = _get_strip(ue, face, edge)
                local_vn = _get_strip(vn, face, edge)
                avg_ue = 0.5 * (local_ue + nbr_ue)
                avg_vn = 0.5 * (local_vn + nbr_vn)
                if edge == WEST:
                    ue_out = ue_out.at[face, 0, :].set(avg_ue)
                    vn_out = vn_out.at[face, 0, :].set(avg_vn)
                elif edge == EAST:
                    ue_out = ue_out.at[face, n, :].set(avg_ue)
                    vn_out = vn_out.at[face, n, :].set(avg_vn)
                elif edge == SOUTH:
                    ue_out = ue_out.at[face, :, 0].set(avg_ue)
                    vn_out = vn_out.at[face, :, 0].set(avg_vn)
                else:
                    ue_out = ue_out.at[face, :, n].set(avg_ue)
                    vn_out = vn_out.at[face, :, n].set(avg_vn)

        # 3. Vertex sync: 3-way average at cube vertices (8 vertices)
        _vtx = [
            [(0, 0, 0), (3, n, 0), (5, 0, n)],
            [(0, n, 0), (1, 0, 0), (5, n, n)],
            [(0, 0, n), (3, n, n), (4, 0, 0)],
            [(0, n, n), (1, 0, n), (4, n, 0)],
            [(1, n, 0), (2, 0, 0), (5, n, 0)],
            [(1, n, n), (2, 0, n), (4, n, n)],
            [(2, n, 0), (3, 0, 0), (5, 0, 0)],
            [(2, n, n), (3, 0, n), (4, 0, n)],
        ]
        for vtx in _vtx:
            ue_avg = sum(ue[f, i, j] for f, i, j in vtx) / 3.0
            vn_avg = sum(vn[f, i, j] for f, i, j in vtx) / 3.0
            for f, i, j in vtx:
                ue_out = ue_out.at[f, i, j].set(ue_avg)
                vn_out = vn_out.at[f, i, j].set(vn_avg)

        # 4. Smooth first interior row: blend with boundary to reduce the
        #    sharp transition between synced boundary and interior corners.
        #    This 3-point filter operates in geographic coords (isotropic).
        w = 0.45  # weight on boundary / 2nd-interior; 0.10 on local
        for face in range(6):
            # WEST (i=1): blend with i=0 (synced) and i=2 (interior)
            ue_out = ue_out.at[face, 1, :].set(
                w * ue_out[face, 0, :] + (1 - 2*w) * ue_out[face, 1, :]
                + w * ue_out[face, 2, :])
            vn_out = vn_out.at[face, 1, :].set(
                w * vn_out[face, 0, :] + (1 - 2*w) * vn_out[face, 1, :]
                + w * vn_out[face, 2, :])
            # EAST (i=n-1)
            ue_out = ue_out.at[face, n-1, :].set(
                w * ue_out[face, n, :] + (1 - 2*w) * ue_out[face, n-1, :]
                + w * ue_out[face, n-2, :])
            vn_out = vn_out.at[face, n-1, :].set(
                w * vn_out[face, n, :] + (1 - 2*w) * vn_out[face, n-1, :]
                + w * vn_out[face, n-2, :])
            # SOUTH (j=1)
            ue_out = ue_out.at[face, :, 1].set(
                w * ue_out[face, :, 0] + (1 - 2*w) * ue_out[face, :, 1]
                + w * ue_out[face, :, 2])
            vn_out = vn_out.at[face, :, 1].set(
                w * vn_out[face, :, 0] + (1 - 2*w) * vn_out[face, :, 1]
                + w * vn_out[face, :, 2])
            # NORTH (j=n-1)
            ue_out = ue_out.at[face, :, n-1].set(
                w * ue_out[face, :, n] + (1 - 2*w) * ue_out[face, :, n-1]
                + w * ue_out[face, :, n-2])
            vn_out = vn_out.at[face, :, n-1].set(
                w * vn_out[face, :, n] + (1 - 2*w) * vn_out[face, :, n-1]
                + w * vn_out[face, :, n-2])

        # 5. Convert back to face-local
        u_d_new = ca_c * ue_out + sa_c * vn_out
        v_d_new = -sa_c * ue_out + ca_c * vn_out

        return state._replace(u_d=u_d_new, v_d=v_d_new)

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
            # Synchronise D-grid boundary corners at every RK stage so
            # that each intermediate state has consistent cross-face winds.
            s = self._sync_dgrid_boundary(s)
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

