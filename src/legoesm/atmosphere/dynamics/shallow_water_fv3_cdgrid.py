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
    _extrapolate_boundary_corners,
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
    use_experimental_csw: bool = False  # EXPERIMENTAL: C-grid tendencies via RK3 (NOT the FV3 forward-backward scheme). Known unstable.
    boundary_fix: bool = True  # Replace boundary corner tendencies with interior

    # FV3 d_sw5 corner divergence damping knobs (fv_arrays.F90 defaults).
    d2_bg: float = 0.0         # Background del-2 coefficient
    dddmp: float = 0.0         # Adaptive Smagorinsky coefficient
    d4_bg: float = 0.16        # Background del-4+ coefficient
    nord: int = 1              # Damping order: 0=del-2, 1=del-4, 2=del-6

    # FV3 d_sw6 vorticity damping (vtdm4, do_vort_damp in Fortran).
    damp_v: float = 0.0        # Vorticity damping coefficient
    # FV3 derives nord_v(k) = min(2, flagstruct%nord) at runtime
    # (dyn_core.F90:757,1258).  With default nord=1 this is 1.
    nord_v: int = 1            # Vorticity damping order


# ==============================================================================
# Tendencies
# ==============================================================================

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

    # 2. Momentum tendencies (vector-invariant form with div damping)
    du_d_dt, dv_d_dt = cdgrid_momentum_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        g=config.g, A_h=config.A_h,
        hyperdiff_coeff=config.hyperdiff_coeff,
        div_damp=config.div_damp,
    )

    # Boundary-corner fix: halo interpolation gives O(dx) gradient error
    # at all face-boundary corners (12-60x larger than interior).
    # Replace with nearest-interior values that have O(dx^2) accuracy.
    n = cdgrid.n
    du_d_dt, dv_d_dt = _extrapolate_boundary_corners(du_d_dt, dv_d_dt, n)

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
        from legoesm.core.precision import cast_pytree
        # Cast state to compute precision at the boundary.
        state_c = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            dh, du, dv = cdgrid_shallow_water_tendencies(
                s, self.cdgrid, self.config,
            )
            return CDGridShallowWaterState(
                h=dh, u_d=du, v_d=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = dispatch_integrator(
            state_c, tendency_fn, dt, self.config.time_integrator,
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

        # Cast back to storage precision.
        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# FV3 Forward-Backward Shallow Water Model (EXPERIMENTAL — DO NOT USE)
# ==============================================================================

class FV3FBShallowWaterModel:
    """EXPERIMENTAL: FV3 forward-backward shallow water model.

    **NOT PRODUCTION-READY.** Known unstable (85 m/s v-wind after 1 day,
    3% mass error). Use ``FV3EdgeShallowWaterModel`` with the default
    ``use_experimental_csw=False`` for production work.

    This model uses the three-phase FV3 forward-backward scheme from
    ``fv3_sw_core.fv3_fb_sw_step``:
    1. c_sw: C-grid half-step (KE + vorticity, forward)
    2. p_grad_c: pressure gradient at C-grid (backward, using h_star)
    3. d_sw: D-grid full-step (mass transport + wind update, no A-L gradient)

    Blocker: The forward-backward coupling is unstable for finite dt without
    additional dissipation at the c_sw/d_sw interface. A faithful port would
    require FV3's exact dissipation control (del2/del4 at specific phases).

    Parameters
    ----------
    grid : CubedSphereGrid
    config : CDGridShallowWaterConfig, optional
    """

    def __init__(self, grid, config=None):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()
        self._target_mass = None

    def set_initial_mass(self, state):
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step using FV3 forward-backward."""
        from legoesm.core.precision import cast_pytree
        from legoesm.core.fv3_sw_core import fv3_fb_sw_step

        state_c = cast_pytree(state, None, "compute")

        h_new, u_new, v_new = fv3_fb_sw_step(
            state_c.h, state_c.u_d, state_c.v_d, state_c.h_s,
            self.cdgrid, dt, g=self.config.g,
            div_damp=self.config.div_damp,
            d2_bg=self.config.d2_bg,
            dddmp=self.config.dddmp,
            d4_bg=self.config.d4_bg,
            nord=self.config.nord,
            damp_v=self.config.damp_v,
            nord_v=self.config.nord_v,
        )

        state_new = FV3EdgeShallowWaterState(
            h=h_new, u_d=u_new, v_d=v_new, h_s=state_c.h_s)

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

        return cast_pytree(state_new, None, "storage")


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
    """PRODUCTION shallow water model with FV3-inspired edge-midpoint D-grid stagger.

    This is a **stabilized research path**, not a faithful FV3 port.
    Key differences from FV3:
    - Uses Arakawa-Lamb 4-point gradient (FV3 uses 2-point c_sw gradient)
    - Uses RK3 time integration (FV3 uses forward-backward splitting)
    - Edge-midpoint stagger avoids boundary sync (FV3 uses tile-edge coupling)

    Edge-midpoint D-grid winds sit half a cell from any face boundary,
    eliminating boundary sync entirely and removing edge artifacts.

    Momentum tendencies are computed at cell corners (compact stencil)
    and averaged to edge-midpoint positions.
    """

    def __init__(self, grid, config=None):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or CDGridShallowWaterConfig()
        self._target_mass = None

    def set_initial_mass(self, state):
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, dt):
        """Advance one time step."""
        from legoesm.core.precision import cast_pytree
        state = cast_pytree(state, None, "compute")

        if self.config.use_experimental_csw:
            # EXPERIMENTAL — known unstable (NaN by step ~50).
            # This wraps fv3_csw_tendencies in RK3, NOT the actual
            # forward-backward step (fv3_forward_backward_step).
            # See docs/cubed_sphere_edge_artifacts.md iterations 7-14.
            import warnings
            warnings.warn(
                "use_experimental_csw=True is experimental and known unstable. "
                "It runs fv3_csw_tendencies (C-grid half only) through "
                "RK3, NOT the actual FV3 forward-backward scheme "
                "(fv3_forward_backward_step). See "
                "docs/cubed_sphere_edge_artifacts.md.",
                stacklevel=2,
            )
            from legoesm.core.fv3_sw_core import fv3_csw_tendencies

            def tendency_fn_csw(s):
                dh, du, dv = fv3_csw_tendencies(
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
                state, tendency_fn_csw, dt, self.config.time_integrator,
            )
        else:
            def tendency_fn(s):
                dh, du, dv = fv3_sw_tendencies(
                    s.h, s.u_d, s.v_d, s.h_s, self.cdgrid,
                    g=self.config.g,
                    div_damp=self.config.div_damp,
                    hyperdiff_coeff=self.config.hyperdiff_coeff,
                    boundary_fix=self.config.boundary_fix,
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

        return cast_pytree(state_new, None, "storage")
