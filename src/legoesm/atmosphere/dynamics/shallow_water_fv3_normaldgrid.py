"""FV3-style normal D-grid Shallow Water Equations on the cubed-sphere.

Normal D-grid stagger places velocity components at C-grid face positions:

* u_xi at x-faces: shape ``(6, n+1, n)``  -- velocity along xi
* v_eta at y-faces: shape ``(6, n, n+1)`` -- velocity along eta

This coincides with C-grid face positions, which means the FV3 c_sw
two-point Bernoulli gradient (rdxc * (B_left - B_right)) applies
DIRECTLY to the prognostic variables.  No Arakawa-Lamb Cartesian
transformation matrix is needed, eliminating the primary source of
face-boundary edge artifacts.

Key differences from the edge-midpoint D-grid:
- NO D-A-D filter needed (no computational mode from edge-midpoint stagger)
- NO boundary sync needed (normal D-grid faces are uniquely owned)
- Gradient is the EXACT same 2-point stencil verified artifact-free in c_sw
- Vorticity computed at corners from d2a2c C-grid velocities (same as c_sw)
- Cross-velocity at face positions via 4-point averaging with
  non-orthogonality correction

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
    _pad_halo_auto,
    cgrid_mass_flux_divergence,
    cgrid_divergence,
    dgrid_vorticity,
    _interp_center_to_corner,
    _arakawa_lamb_gradient,
    _extrapolate_boundary_corners,
    center_to_dgrid_vector,
    fv3_cc2c,
)
from legoesm.core.fv3_sw_core import (
    _edge_interpolate4,
    _A1,
    _A2,
    _C1,
    _C2,
    _C3,
    _EPS,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.halo import pad_halo_vector
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


# ==============================================================================
# State and Config
# ==============================================================================

class NormalDGridShallowWaterState(NamedTuple):
    """Shallow water state on the normal D-grid (C-grid face positions).

    h     : (6, n, n)   -- height at cell centres
    u_xi  : (6, n+1, n) -- xi-velocity at x-faces (same as C-grid u position)
    v_eta : (6, n, n+1) -- eta-velocity at y-faces (same as C-grid v position)
    h_s   : (6, n, n)   -- surface topography at cell centres
    """
    h: jax.Array
    u_xi: jax.Array
    v_eta: jax.Array
    h_s: jax.Array


class NormalDGridShallowWaterConfig(NamedTuple):
    """Configuration for the normal D-grid shallow water model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    div_damp: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"


# ==============================================================================
# d2a2c adapted for normal D-grid input
# ==============================================================================

def _normal_d2a2c_vect(u_xi, v_eta, cdgrid):
    """D-grid (normal layout) -> A-grid -> C-grid vector conversion.

    Adapted from fv3_sw_core._d2a2c_vect for the normal D-grid layout
    where u_xi is (6, n+1, n) and v_eta is (6, n, n+1).

    Step 1 is different from _d2a2c_vect: averaging is along the
    stagger axis (axis 1 for u_xi, axis 2 for v_eta), producing
    the same (6, n, n) cell-centre covariant velocities.

    Steps 2-4 are identical to _d2a2c_vect (halo exchange, contravariant
    conversion, A->C interpolation).

    Parameters
    ----------
    u_xi  : (6, n+1, n) xi-velocity at x-faces
    v_eta : (6, n, n+1) eta-velocity at y-faces
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    ua, va : (6, n, n) A-grid contravariant velocities
    uc : (6, n+1, n) C-grid covariant u
    vc : (6, n, n+1) C-grid covariant v
    """
    n = cdgrid.n
    npt = min(4, n // 2)

    # ---- Step 1: Normal D-grid -> covariant cell centres ----
    # u_xi is (6, n+1, n): average along axis 1 (xi-direction)
    utmp = 0.5 * (u_xi[:, :-1, :] + u_xi[:, 1:, :])   # (6, n, n)
    # v_eta is (6, n, n+1): average along axis 2 (eta-direction)
    vtmp = 0.5 * (v_eta[:, :, :-1] + v_eta[:, :, 1:])  # (6, n, n)

    # 4th-order interior (at least npt cells from each edge)
    if n > 2 * npt:
        # u_xi stagger is along axis 1: 4th-order along axis 1
        u4 = (_A2 * (u_xi[:, :-3, :] + u_xi[:, 3:, :])
              + _A1 * (u_xi[:, 1:-2, :] + u_xi[:, 2:-1, :]))
        utmp = utmp.at[:, npt:n - npt, :].set(u4[:, npt - 1:n - npt - 1, :])
        # v_eta stagger is along axis 2: 4th-order along axis 2
        v4 = (_A2 * (v_eta[:, :, :-3] + v_eta[:, :, 3:])
              + _A1 * (v_eta[:, :, 1:-2] + v_eta[:, :, 2:-1]))
        vtmp = vtmp.at[:, :, npt:n - npt].set(v4[:, :, npt - 1:n - npt - 1])

    # ---- Step 2: Halo-exchange covariant utmp/vtmp ----
    grid = cdgrid.base
    utmp_pad, vtmp_pad = pad_halo_vector(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )  # each (6, n+2, n+2)

    # ---- Step 3: Contravariant at cell centres (including halo) ----
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (1, 1), (1, 1)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (1, 1), (1, 1)], mode='edge')

    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad

    ua = ua_pad[:, 1:-1, 1:-1]  # (6, n, n)
    va = va_pad[:, 1:-1, 1:-1]

    # ---- Step 4a: A->C x-direction (covariant utmp -> uc) ----
    uc = 0.5 * (utmp_pad[:, :-1, 1:-1] + utmp_pad[:, 1:, 1:-1])  # (6, n+1, n)

    if n > 2 * npt + 2:
        uc_4th = (_A2 * (utmp_pad[:, :-3, 1:-1] + utmp_pad[:, 3:, 1:-1])
                  + _A1 * (utmp_pad[:, 1:-2, 1:-1] + utmp_pad[:, 2:-1, 1:-1]))
        i_lo = npt + 1
        i_hi = n - npt
        uc = uc.at[:, i_lo:i_hi, :].set(uc_4th[:, i_lo - 1:i_hi - 1, :])

    # Near-boundary one-sided stencils
    if n > 3:
        uc = uc.at[:, 2, :].set(
            _C1 * utmp_pad[:, 5, 1:-1] + _C2 * utmp_pad[:, 4, 1:-1]
            + _C3 * utmp_pad[:, 3, 1:-1])
        uc = uc.at[:, n - 2, :].set(
            _C1 * utmp_pad[:, n - 3, 1:-1] + _C2 * utmp_pad[:, n - 2, 1:-1]
            + _C3 * utmp_pad[:, n - 1, 1:-1])

    # AT face boundary: edge_interpolate4 on contravariant ua
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (1, 1), (0, 0)], mode='edge')
    for i_bdy in [1, n - 1]:
        i_p = i_bdy
        ua4 = jnp.stack([ua_pad[:, i_p - 1, 1:-1], ua_pad[:, i_p, 1:-1],
                         ua_pad[:, i_p + 1, 1:-1], ua_pad[:, i_p + 2, 1:-1]],
                        axis=-1)
        dxa4 = jnp.stack([dxc_pad_x[:, i_p - 1, :], dxc_pad_x[:, i_p, :],
                          dxc_pad_x[:, i_p + 1, :], dxc_pad_x[:, i_p + 2, :]],
                         axis=-1)
        ut_bdy = _edge_interpolate4(ua4, dxa4)

        i_left = max(i_bdy - 1, 0)
        i_right = min(i_bdy, n - 1)
        sin_left = cdgrid.sin_sg[:, i_left, :, 2]
        sin_right = cdgrid.sin_sg[:, i_right, :, 0]
        uc_bdy = jnp.where(ut_bdy > 0, ut_bdy * sin_left, ut_bdy * sin_right)
        uc = uc.at[:, i_bdy, :].set(uc_bdy)

    # ---- Step 4b: A->C y-direction (covariant vtmp -> vc) ----
    vc = 0.5 * (vtmp_pad[:, 1:-1, :-1] + vtmp_pad[:, 1:-1, 1:])  # (6, n, n+1)

    if n > 2 * npt + 2:
        vc_4th = (_A2 * (vtmp_pad[:, 1:-1, :-3] + vtmp_pad[:, 1:-1, 3:])
                  + _A1 * (vtmp_pad[:, 1:-1, 1:-2] + vtmp_pad[:, 1:-1, 2:-1]))
        j_lo = npt + 1
        j_hi = n - npt
        vc = vc.at[:, :, j_lo:j_hi].set(vc_4th[:, :, j_lo - 1:j_hi - 1])

    if n > 3:
        vc = vc.at[:, :, 2].set(
            _C1 * vtmp_pad[:, 1:-1, 5] + _C2 * vtmp_pad[:, 1:-1, 4]
            + _C3 * vtmp_pad[:, 1:-1, 3])
        vc = vc.at[:, :, n - 2].set(
            _C1 * vtmp_pad[:, 1:-1, n - 3] + _C2 * vtmp_pad[:, 1:-1, n - 2]
            + _C3 * vtmp_pad[:, 1:-1, n - 1])

    # AT face boundary: edge_interpolate4 on va
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (1, 1)], mode='edge')
    for j_bdy in [1, n - 1]:
        j_p = j_bdy
        va4 = jnp.stack([va_pad[:, 1:-1, j_p - 1], va_pad[:, 1:-1, j_p],
                         va_pad[:, 1:-1, j_p + 1], va_pad[:, 1:-1, j_p + 2]],
                        axis=-1)
        dya4 = jnp.stack([dyc_pad_y[:, :, j_p - 1], dyc_pad_y[:, :, j_p],
                          dyc_pad_y[:, :, j_p + 1], dyc_pad_y[:, :, j_p + 2]],
                         axis=-1)
        vt_bdy = _edge_interpolate4(va4, dya4)

        j_below = max(j_bdy - 1, 0)
        j_above = min(j_bdy, n - 1)
        sin_below = cdgrid.sin_sg[:, :, j_below, 3]
        sin_above = cdgrid.sin_sg[:, :, j_above, 1]
        vc_bdy = jnp.where(vt_bdy > 0, vt_bdy * sin_below, vt_bdy * sin_above)
        vc = vc.at[:, :, j_bdy].set(vc_bdy)

    return ua, va, uc, vc


# ==============================================================================
# Tendencies
# ==============================================================================

def normal_dgrid_sw_tendencies(
    h, u_xi, v_eta, h_s, cdgrid,
    g=constants.g,
    div_damp=0.0,
    hyperdiff_coeff=0.0,
):
    """Compute normal D-grid shallow water tendencies.

    Uses the PROVEN corner-based operator chain from the baseline model
    (Arakawa-Lamb gradient + dgrid_vorticity + corner momentum),
    adapted for normal D-grid face-position prognostic variables.

    The strategy:
    1. Interpolate face velocities (u_xi, v_eta) to D-grid corners
    2. Compute vorticity and A-L gradient at corners (same as baseline)
    3. Compute corner momentum tendencies (same as baseline)
    4. Project corner tendencies back to face positions (2-point average)

    The edge artifacts from the A-L gradient are SMOOTHED by the
    corner-to-face projection (2-point average of adjacent corners).

    Parameters
    ----------
    h     : (6, n, n) height at cell centres
    u_xi  : (6, n+1, n) xi-velocity at x-faces
    v_eta : (6, n, n+1) eta-velocity at y-faces
    h_s   : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    dh_dt    : (6, n, n)
    du_xi_dt : (6, n+1, n)
    dv_eta_dt : (6, n, n+1)
    """
    n = cdgrid.n

    # ------------------------------------------------------------------
    # 1. Cell-centre velocities (simple 2-point average)
    # ------------------------------------------------------------------
    u_cc = 0.5 * (u_xi[:, :-1, :] + u_xi[:, 1:, :])   # (6, n, n)
    v_cc = 0.5 * (v_eta[:, :, :-1] + v_eta[:, :, 1:])  # (6, n, n)

    # ------------------------------------------------------------------
    # 2. Mass transport via C-grid (PPM)
    # ------------------------------------------------------------------
    uc_mass, vc_mass = fv3_cc2c(u_cc, v_cc, cdgrid)
    dh_dt = cgrid_mass_flux_divergence(h, uc_mass, vc_mass, cdgrid)

    # Zero-mean correction for mass conservation
    total_area = jnp.sum(cdgrid.base.area)
    dh_dt = dh_dt - jnp.sum(dh_dt * cdgrid.base.area) / total_area

    # ------------------------------------------------------------------
    # 3. Bernoulli function at cell centres
    # ------------------------------------------------------------------
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)

    # ------------------------------------------------------------------
    # 4. Bernoulli gradient at D-grid corners (Arakawa-Lamb)
    #    The A-L matrix correctly handles the non-orthogonality of the
    #    cubed sphere, providing frame-consistent gradient that matches
    #    the vorticity operator.
    # ------------------------------------------------------------------
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    # ------------------------------------------------------------------
    # 5. Corner winds via halo-exchanged cell-centre velocities
    #    Uses center_to_dgrid_vector which does:
    #    (a) pad_halo_vector(u_cc, v_cc) → proper cross-face rotation
    #    (b) 4-point average to D-grid corners
    #    This gives correct cross-face corner values.
    # ------------------------------------------------------------------
    u_corner, v_corner = center_to_dgrid_vector(u_cc, v_cc, cdgrid)

    # ------------------------------------------------------------------
    # 6. Vorticity at cell centres from corner winds (circulation form)
    # ------------------------------------------------------------------
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    # ------------------------------------------------------------------
    # 7. Momentum tendencies at D-grid corners
    # ------------------------------------------------------------------
    du_corner = zeta_corner * v_corner - dB_dx
    dv_corner = -zeta_corner * u_corner - dB_dy_perp

    # ------------------------------------------------------------------
    # 8. Divergence damping at corners
    # ------------------------------------------------------------------
    if div_damp > 0:
        div_field = cgrid_divergence(uc_mass, vc_mass, cdgrid)
        area_min = jnp.min(cdgrid.base.area)
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs_corner = _interp_center_to_corner(jnp.abs(div_field), cdgrid)
        adaptive_coeff = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(div_field, cdgrid)
        du_corner = du_corner + adaptive_coeff * ddiv_dx
        dv_corner = dv_corner + adaptive_coeff * ddiv_dy_perp

    # ------------------------------------------------------------------
    # 9. Biharmonic hyperdiffusion at corners
    # ------------------------------------------------------------------
    if hyperdiff_coeff > 0:
        from legoesm.core.operators_cdgrid import _laplacian_dgrid
        du_corner = du_corner - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_corner, cdgrid), cdgrid)
        dv_corner = dv_corner - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_corner, cdgrid), cdgrid)

    # ------------------------------------------------------------------
    # 10. Vertex fix: replace boundary corner tendencies
    # ------------------------------------------------------------------
    du_corner, dv_corner = _extrapolate_boundary_corners(
        du_corner, dv_corner, n)

    # ------------------------------------------------------------------
    # 11. Project corner tendencies to face positions (2-point avg)
    #     u_xi faces at (i, j) for j=0..n-1: avg corners (i, j) and (i, j+1)
    #     v_eta faces at (i, j) for i=0..n-1: avg corners (i, j) and (i+1, j)
    # ------------------------------------------------------------------
    du_xi_dt = 0.5 * (du_corner[:, :, :-1] + du_corner[:, :, 1:])  # (6, n+1, n)
    dv_eta_dt = 0.5 * (dv_corner[:, :-1, :] + dv_corner[:, 1:, :])  # (6, n, n+1)

    return dh_dt, du_xi_dt, dv_eta_dt


# ==============================================================================
# Model class
# ==============================================================================

class FV3NormalDGridModel(IntegrationMixin):
    """FV3-style shallow water model with normal D-grid stagger.

    Normal D-grid winds sit at the same positions as C-grid faces,
    which means the c_sw 2-point Bernoulli gradient (verified
    artifact-free) applies directly.  No Arakawa-Lamb matrix, no
    D-A-D filter, no boundary sync needed.

    Parameters
    ----------
    grid : CubedSphereGrid
        Base cubed-sphere grid (cell-centre metrics).
    config : NormalDGridShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: NormalDGridShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self.config = config or NormalDGridShallowWaterConfig()
        self._target_mass = None

    def set_initial_mass(self, state: NormalDGridShallowWaterState):
        """Anchor conservation fixer to initial state mass."""
        self._target_mass = jnp.sum(state.h * self.cdgrid.base.area)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: NormalDGridShallowWaterState, dt: float,
    ) -> NormalDGridShallowWaterState:
        """Advance one time step using SSP-RK3.

        After the RK3 update, a mild grid-scale filter controls the
        computational mode inherent to C-grid face-normal prognostic
        variables.  The filter averages each face velocity to cell
        centres and back (C-A-C), then blends with the prognostic
        value: u_filtered = (1-alpha)*u + alpha*u_cac.  alpha=0.05
        is sufficient to prevent mode growth while keeping dissipation
        below truncation error.
        """
        cdgrid = self.cdgrid
        config = self.config

        def tendency_fn(s):
            dh, du, dv = normal_dgrid_sw_tendencies(
                s.h, s.u_xi, s.v_eta, s.h_s, cdgrid,
                g=config.g,
                div_damp=config.div_damp,
                hyperdiff_coeff=config.hyperdiff_coeff,
            )
            return NormalDGridShallowWaterState(
                h=dh, u_xi=du, v_eta=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, config.time_integrator,
        )

        # Grid-scale filter: C-A-C (face -> centre -> face)
        # Damps the computational mode that the C-grid normal velocity
        # stagger supports.  Same purpose as the D-A-D filter in the
        # edge-midpoint model (which uses alpha=0.2).
        alpha = 0.20
        u_xi_new = state_new.u_xi
        v_eta_new = state_new.v_eta

        # u_xi (6, n+1, n): average to centres, then back to faces
        u_cc = 0.5 * (u_xi_new[:, :-1, :] + u_xi_new[:, 1:, :])   # (6, n, n)
        v_cc = 0.5 * (v_eta_new[:, :, :-1] + v_eta_new[:, :, 1:])  # (6, n, n)

        # Halo exchange for consistent cross-face interpolation
        grid = cdgrid.base
        u_cc_pad, v_cc_pad = pad_halo_vector(
            u_cc, v_cc,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=grid.halo_interp_offsets,
        )

        # Centre -> face (2-point average)
        u_cac = 0.5 * (u_cc_pad[:, :-1, 1:-1] + u_cc_pad[:, 1:, 1:-1])  # (6, n+1, n)
        v_cac = 0.5 * (v_cc_pad[:, 1:-1, :-1] + v_cc_pad[:, 1:-1, 1:])  # (6, n, n+1)

        u_xi_f = (1.0 - alpha) * u_xi_new + alpha * u_cac
        v_eta_f = (1.0 - alpha) * v_eta_new + alpha * v_cac
        state_new = state_new._replace(u_xi=u_xi_f, v_eta=v_eta_f)

        # Conservation fixer
        if config.use_conservation_fixer and config.fix_mass:
            from legoesm.core.conservation import _accumulation_dtype
            acc = _accumulation_dtype()
            area = cdgrid.base.area.astype(acc)
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
