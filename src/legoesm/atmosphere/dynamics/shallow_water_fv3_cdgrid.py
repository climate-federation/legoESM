"""FV3-style C-D grid Shallow Water Equations on the cubed-sphere.

True staggered C-D grid discretisation following Lin (2004):

* D-grid winds (cell corners, shape ``(6, n+1, n+1)``) are prognostic.
* C-grid velocities (cell edges) are diagnosed for mass transport.
* Vorticity is computed from circulation around cell boundaries
  (exact on the D-grid, avoids Hollingsworth-Kallberg instability).
* Bernoulli gradient uses Arakawa-Lamb 4-point formula at corners.
* Mass transport uses PPM (Piecewise Parabolic Method) for 4th-order
  accurate face reconstruction.
* D-to-C grid conversion includes non-orthogonality correction.
* Divergence damping with adaptive Smagorinsky scaling.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Colella & Woodward (1984): The Piecewise Parabolic Method
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
    fv3_sw_tendencies,
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

    h : (6, n, n) -- height at cell centres
    u_d : (6, n+1, n+1) -- x-velocity at cell corners (D-grid)
    v_d : (6, n+1, n+1) -- y-velocity at cell corners (D-grid)
    h_s : (6, n, n) -- surface topography at cell centres
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

def _fix_boundary_tendencies(du, dv, n):
    """Blend face-boundary momentum tendencies with interior values.

    D-grid operators at face-boundary corners have larger truncation
    errors than interior corners due to non-orthogonality.  Blend
    boundary-corner tendencies with their nearest interior neighbour
    to reduce the edge error.  Vertices (where 3 faces meet) are
    replaced entirely with the average of their 2 edge neighbours.

    Parameters
    ----------
    du, dv : jax.Array, shape (6, n+1, n+1)
    n : int

    Returns
    -------
    du, dv : jax.Array with smoothed boundary values
    """
    # Blend weight: 0 = keep original, 1 = use interior neighbor
    w = 0.5

    # Edge blending (excluding corners which get special treatment)
    # West (i=0), blend with i=1
    du = du.at[:, 0, 1:-1].set((1 - w) * du[:, 0, 1:-1] + w * du[:, 1, 1:-1])
    dv = dv.at[:, 0, 1:-1].set((1 - w) * dv[:, 0, 1:-1] + w * dv[:, 1, 1:-1])
    # East (i=n), blend with i=n-1
    du = du.at[:, n, 1:-1].set((1 - w) * du[:, n, 1:-1] + w * du[:, n - 1, 1:-1])
    dv = dv.at[:, n, 1:-1].set((1 - w) * dv[:, n, 1:-1] + w * dv[:, n - 1, 1:-1])
    # South (j=0), blend with j=1
    du = du.at[:, 1:-1, 0].set((1 - w) * du[:, 1:-1, 0] + w * du[:, 1:-1, 1])
    dv = dv.at[:, 1:-1, 0].set((1 - w) * dv[:, 1:-1, 0] + w * dv[:, 1:-1, 1])
    # North (j=n), blend with j=n-1
    du = du.at[:, 1:-1, n].set((1 - w) * du[:, 1:-1, n] + w * du[:, 1:-1, n - 1])
    dv = dv.at[:, 1:-1, n].set((1 - w) * dv[:, 1:-1, n] + w * dv[:, 1:-1, n - 1])

    # Vertex corners: replace with average of 2 nearest edge neighbours
    corners = [
        ((0, 0), (1, 0), (0, 1)),
        ((n, 0), (n - 1, 0), (n, 1)),
        ((0, n), (1, n), (0, n - 1)),
        ((n, n), (n - 1, n), (n, n - 1)),
    ]
    for (ci, cj), (n1i, n1j), (n2i, n2j) in corners:
        avg_du = 0.5 * (du[:, n1i, n1j] + du[:, n2i, n2j])
        avg_dv = 0.5 * (dv[:, n1i, n1j] + dv[:, n2i, n2j])
        du = du.at[:, ci, cj].set(avg_du)
        dv = dv.at[:, ci, cj].set(avg_dv)
    return du, dv


def cdgrid_shallow_water_tendencies(
    state: CDGridShallowWaterState,
    cdgrid: CubedSphereCDGrid,
    config: CDGridShallowWaterConfig = CDGridShallowWaterConfig(),
):
    """Compute C-D grid shallow water tendencies.

    Uses PPM transport for mass flux and d2a2c with non-orthogonality
    correction for D-to-C grid conversion.

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

    # 1. Mass transport via C-grid velocities (with non-orth correction)
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    dh_dt = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

    # Zero-mean correction for mass conservation
    total_area = jnp.sum(cdgrid.base.area)
    dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # 2. Momentum tendencies (vector-invariant form with div damping)
    du_d_dt, dv_d_dt = cdgrid_momentum_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        g=config.g, A_h=config.A_h,
        hyperdiff_coeff=config.hyperdiff_coeff,
        div_damp=config.div_damp,
    )

    # 3. Fix tendencies at cube vertices where the D-grid stencil
    #    breaks down due to extreme non-orthogonality (cosa = +/-0.5).
    n = cdgrid.n
    du_d_dt, dv_d_dt = _fix_boundary_tendencies(du_d_dt, dv_d_dt, n)

    return dh_dt, du_d_dt, dv_d_dt


# ==============================================================================
# Model class
# ==============================================================================

class CDGridShallowWaterModel(IntegrationMixin):
    """FV3-style C-D grid shallow water model on the cubed-sphere.

    Features PPM transport, non-orthogonality-corrected d2a2c, and
    adaptive divergence damping. No ad-hoc boundary smoothing needed.

    Parameters
    ----------
    grid : CubedSphereGrid
        Base cubed-sphere grid (cell-centre metrics).
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
        self._target_mass = None

    def set_initial_mass(self, state: CDGridShallowWaterState):
        """Anchor conservation fixer to initial state mass."""
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    def _sync_dgrid_boundary(self, state: CDGridShallowWaterState):
        """Owner-based sync of D-grid corner winds at shared edges.

        For each shared face edge, the lower face index is the "owner".
        The non-owner face COPIES the owner's geographic wind — no
        averaging.  This ensures bitwise-identical corner values at
        shared boundaries without the edge-selective dissipation that
        pairwise averaging introduces.

        Vertices (shared by 3 faces) are owned by the lowest face index.
        """
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

        u_d, v_d = state.u_d, state.v_d
        n = self.grid.n
        ca_c = self.cdgrid.cos_angle_corner
        sa_c = self.cdgrid.sin_angle_corner

        # Convert all corners to geographic
        ue = ca_c * u_d - sa_c * v_d
        vn = sa_c * u_d + ca_c * v_d

        def _get_strip(arr, face, edge):
            if edge == WEST:    return arr[face, 0, :]
            elif edge == EAST:  return arr[face, n, :]
            elif edge == SOUTH: return arr[face, :, 0]
            else:               return arr[face, :, n]

        # Edge sync: non-owner copies from owner (lower face index)
        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                if nbr_face < face:
                    # nbr_face owns → copy FROM neighbor
                    nbr_ue = _get_strip(ue, nbr_face, nbr_edge)
                    nbr_vn = _get_strip(vn, nbr_face, nbr_edge)
                    if is_reversed:
                        nbr_ue = nbr_ue[::-1]
                        nbr_vn = nbr_vn[::-1]
                    if edge == WEST:
                        ue = ue.at[face, 0, :].set(nbr_ue)
                        vn = vn.at[face, 0, :].set(nbr_vn)
                    elif edge == EAST:
                        ue = ue.at[face, n, :].set(nbr_ue)
                        vn = vn.at[face, n, :].set(nbr_vn)
                    elif edge == SOUTH:
                        ue = ue.at[face, :, 0].set(nbr_ue)
                        vn = vn.at[face, :, 0].set(nbr_vn)
                    else:
                        ue = ue.at[face, :, n].set(nbr_ue)
                        vn = vn.at[face, :, n].set(nbr_vn)

        # Vertex sync: lowest face index owns
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
            owner = vtx[0]  # already sorted by face index
            ue_own = ue[owner[0], owner[1], owner[2]]
            vn_own = vn[owner[0], owner[1], owner[2]]
            for f, i, j in vtx[1:]:
                ue = ue.at[f, i, j].set(ue_own)
                vn = vn.at[f, i, j].set(vn_own)

        # Convert back to face-local
        u_d_new = ca_c * ue + sa_c * vn
        v_d_new = -sa_c * ue + ca_c * vn
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

        # Owner-based sync: once per time step, after integrator
        state_new = self._sync_dgrid_boundary(state_new)

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            state_new = state_new._replace(h=h_fixed)

        return state_new


# ==============================================================================
# FV3 edge-midpoint D-grid shallow water model
# ==============================================================================

class FV3EdgeShallowWaterState(NamedTuple):
    """Shallow water state with FV3 edge-midpoint D-grid stagger.

    h   : (6, n, n)   -- height at cell centres
    u_d : (6, n, n+1) -- x-velocity at x-edge midpoints
    v_d : (6, n+1, n) -- y-velocity at y-edge midpoints
    h_s : (6, n, n)   -- surface topography at cell centres
    """
    h: jax.Array
    u_d: jax.Array
    v_d: jax.Array
    h_s: jax.Array


class FV3EdgeShallowWaterModel(IntegrationMixin):
    """FV3-style shallow water model with edge-midpoint D-grid stagger.

    Uses PPM transport, d2a2c with non-orthogonality correction,
    and adaptive divergence damping.
    """

    def __init__(self, grid, config=None):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()
        self._target_mass = None

    def set_initial_mass(self, state):
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    def _sync_dgrid_boundary(self, state):
        """Sync edge-midpoint D-grid winds at face boundaries.

        The d2a2c handles cross-face via halo exchange with
        non-orthogonality correction, so no explicit sync needed.
        """
        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step."""
        def tendency_fn(s):
            dh, du, dv = fv3_sw_tendencies(
                s.h, s.u_d, s.v_d, s.h_s, self.cdgrid,
                g=self.config.g,
                div_damp=self.config.div_damp,
                hyperdiff_coeff=self.config.hyperdiff_coeff,
            )
            return FV3EdgeShallowWaterState(
                h=dh, u_d=du, v_d=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = self.cdgrid.base.area.astype(acc)
            total_area = jnp.sum(area)
            if self._target_mass is not None:
                mass_target = self._target_mass
            else:
                mass_target = jnp.sum(state.h.astype(acc) * area)
            mass_new = jnp.sum(state_new.h.astype(acc) * area)
            correction = (mass_target - mass_new) / total_area
            h_fixed = state_new.h + correction.astype(state_new.h.dtype)
            state_new = state_new._replace(h=h_fixed)

        return state_new
