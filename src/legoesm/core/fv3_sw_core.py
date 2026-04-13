"""FV3-inspired shallow water forward-backward core (EXPERIMENTAL).

Implements a c_sw + d_sw forward-backward step adapted from GFDL's
FV3 dynamical core (sw_core.F90).  The operators use FV3's covariant
velocity convention with sin_sg flux scaling.

**Status**: experimental.  The full forward-backward step
(``fv3_fb_sw_step``) is known to be unstable — use ``fv3_sw_tendencies``
(operators_cdgrid.py) or ``fv3_csw_tendencies`` for production work.

The c_sw half-step operates at C-grid face positions (gradient and
vorticity at the same stagger — essential for geostrophic balance).
The d_sw half-step operates at D-grid edge-midpoint positions (reusing
the existing Arakawa-Lamb gradient which is consistent at that stagger).

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Mouallem, Harris & Chen (2023): Duo-Grid edge effect fix
- GFDL sw_core.F90: c_sw (lines 79-488), d2a2c_vect (lines 3006-3345)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo_vector
from legoesm.core.operators_cdgrid import (
    _pad_halo_auto,
    cgrid_mass_flux_divergence,
    dgrid_vorticity,
    _interp_center_to_corner,
    _arakawa_lamb_gradient,
    _extrapolate_boundary_corners,
    cgrid_divergence,
)

_EPS = float(jnp.finfo(jnp.float32).eps)


def _edge_interpolate4(ua4, dxa4):
    """FV3 non-uniform 4-point interpolation to the interface between cells 2 and 3.

    Exact port of ``edge_interpolate4`` from sw_core.F90.
    """
    t1 = dxa4[..., 0] + dxa4[..., 1]
    t2 = dxa4[..., 2] + dxa4[..., 3]
    return 0.5 * (
        ((t1 + dxa4[..., 1]) * ua4[..., 1] - dxa4[..., 1] * ua4[..., 0]) / t1
        + ((t2 + dxa4[..., 2]) * ua4[..., 2] - dxa4[..., 2] * ua4[..., 3]) / t2
    )


# FV3 interpolation constants (from sw_core.F90)
_A1 = 0.5625     # 4th-order Lagrange
_A2 = -0.0625
_C1 = -2.0 / 14.0  # one-sided cubic at face boundary
_C2 = 11.0 / 14.0
_C3 = 5.0 / 14.0


# ==============================================================================
# d2a2c_vect: D-grid → A-grid → C-grid (FV3 covariant convention)
# ==============================================================================

def _d2a2c_vect_duogrid(u_d, v_d, cdgrid):
    """FV3 D→A→C with Duo-Grid: 4th-order everywhere, no edge specials.

    When the Duo-Grid is active, halo data has been remapped to the
    extended grid, so stencils are smooth across face boundaries.  This
    matches the ``gridstruct%dg%is_initialized`` branch in FV3
    ``sw_core.F90:3419-3704``:
    - 4th-order D→A for the full domain except outermost halo rows
    - 4th-order A→C for all C-grid positions
    - ALL corner fixes and edge specials SKIPPED
    - NO ``sin_sg`` upwinding at face boundaries
    """
    n = cdgrid.n
    h = 2  # halo depth for smooth 4th-order stencil coverage
    grid = cdgrid.base
    dg = grid.duogrid

    # ---- Step 0: D-grid staggered halo exchange (FV3 ext_vector) ----
    # FV3 exchanges D-grid winds via ext_vector BEFORE d2a2c_vect,
    # providing halo-extended D-grid data for 4th-order D→A everywhere.
    from legoesm.grids.duogrid import pad_halo_dgrid
    u_d_ext, v_d_ext = pad_halo_dgrid(
        u_d, v_d,
        cdgrid.cos_angle_edge_x, cdgrid.sin_angle_edge_x,
        cdgrid.cos_angle_edge_y, cdgrid.sin_angle_edge_y,
        dg)
    # u_d_ext: (6, n, n+3) — u_d with 1 halo on each j-side
    # v_d_ext: (6, n+3, n) — v_d with 1 halo on each i-side

    # ---- Step 1: D-grid → covariant cell centres (utmp, vtmp) ----
    # FV3 sw_core.F90:3421-3447 — duogrid path:
    #   Interior (jsd+1..jed-1): 4th-order
    #   Extreme boundary (jsd, jed): use u(i, jsd+1) (copy of inner edge)
    # u_d_ext: (6, n, n+3) — edges at indices 0..n+2, cells at 0..n-1.
    # Edge index j in u_d_ext corresponds to: 0=south halo, 1..n+1=original, n+2=north halo.
    # Cell j uses edges (j+1, j+2) — offset by 1 for halo.
    # FV3 sw_core.F90:3421-3447 duogrid path: 4th-order D→A everywhere
    # except at the outermost halo rows (jsd, jed) of the HALOCATED domain.
    # Our n cells correspond to FV3's INTERIOR cells (is:ie), NOT the halo
    # boundary rows. The halo is provided by the secondary geographic exchange
    # (Step 2 below). So all n cells get 4th-order here.
    utmp = 0.5 * (u_d_ext[:, :, 1:-2] + u_d_ext[:, :, 2:-1])  # (6, n, n) 2nd-order fallback
    vtmp = 0.5 * (v_d_ext[:, 1:-2, :] + v_d_ext[:, 2:-1, :])  # (6, n, n)
    if n >= 4:
        u4 = (_A2 * (u_d_ext[:, :, :-3] + u_d_ext[:, :, 3:])
              + _A1 * (u_d_ext[:, :, 1:-2] + u_d_ext[:, :, 2:-1]))
        utmp = u4  # 4th-order for ALL n interior cells
        v4 = (_A2 * (v_d_ext[:, :-3, :] + v_d_ext[:, 3:, :])
              + _A1 * (v_d_ext[:, 1:-2, :] + v_d_ext[:, 2:-1, :]))
        vtmp = v4

    # ---- Step 2: Halo-exchange covariant utmp/vtmp ----
    # FV3 ext_vector converts covariant → lat/lon → remap → grid-aligned.
    # The critical step is covariant→contravariant BEFORE geographic rotation,
    # which accounts for non-orthogonality (cosa_s, rsin2).
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell

    # Covariant → contravariant (FV3 c2l_ord2 step 1)
    ua = (utmp - vtmp * cos_sg5) * rsin2
    va = (vtmp - utmp * cos_sg5) * rsin2
    # Contravariant → geographic (FV3 c2l_ord2 step 2)
    u_east = grid.cos_angle * ua - grid.sin_angle * va
    v_north = grid.sin_angle * ua + grid.cos_angle * va
    # Halo-exchange geographic components as scalars (with duogrid remap)
    from legoesm.grids.halo import pad_halo
    u_east_pad = pad_halo(u_east, halo=h, duogrid=dg)
    v_north_pad = pad_halo(v_north, halo=h, duogrid=dg)
    # Geographic → grid-aligned on padded domain
    cap = grid.cos_angle_padded_h2
    sap = grid.sin_angle_padded_h2
    utmp_pad = cap * u_east_pad + sap * v_north_pad
    vtmp_pad = -sap * u_east_pad + cap * v_north_pad

    # ---- Step 3: Contravariant at cell centres over FULL padded domain ----
    # FV3 ref: sw_core.F90:3449-3454 — compute ua/va for isd:ied, jsd:jed
    # cos_sg5 and rsin2 already computed in Step 2 for ext_vector_dgrid
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')

    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad

    ua = ua_pad[:, h:-h, h:-h]  # (6, n, n)
    va = va_pad[:, h:-h, h:-h]

    # ---- Step 4a: A→C x-direction — 4th-order EVERYWHERE ----
    # FV3 ref: sw_core.F90:3557-3563 with ifirst=is-1, ilast=ie+2
    # (no clamping because dg%is_initialized)
    # With halo=2: utmp_pad indices 0..n+3, 4th-order uc needs utmp at
    # [i, i+1, i+2, i+3] → covers uc indices 0..n (all n+1 C-grid positions)
    uc_4th = (_A2 * (utmp_pad[:, :-3, h:-h] + utmp_pad[:, 3:, h:-h])
              + _A1 * (utmp_pad[:, 1:-2, h:-h] + utmp_pad[:, 2:-1, h:-h]))
    # uc_4th has shape (6, n+1, n) — exactly the C-grid u-positions
    uc = uc_4th

    # Contravariant ut from covariant uc
    cosa_u = cdgrid.cosa_u
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u ** 2, _EPS))
    ut = (uc - v_d * cosa_u) / jnp.maximum(sina_u, _EPS)

    # ---- Step 4b: A→C y-direction — 4th-order EVERYWHERE ----
    vc_4th = (_A2 * (vtmp_pad[:, h:-h, :-3] + vtmp_pad[:, h:-h, 3:])
              + _A1 * (vtmp_pad[:, h:-h, 1:-2] + vtmp_pad[:, h:-h, 2:-1]))
    vc = vc_4th

    cosa_v = cdgrid.cosa_v
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v ** 2, _EPS))
    vt = (vc - u_d * cosa_v) / jnp.maximum(sina_v, _EPS)

    return ua, va, uc, vc, ut, vt


def _d2a2c_vect(u_d, v_d, cdgrid):
    """FV3 D-grid → A-grid → C-grid vector conversion.

    Adapted from GFDL sw_core.F90 d2a2c_vect.  Returns C-grid
    velocities in FV3's COVARIANT convention (uc = interpolated covariant
    utmp, NOT the physical face-normal velocity).

    When the Duo-Grid is active on the base grid, dispatches to the
    FV3-faithful Duo-Grid path (``_d2a2c_vect_duogrid``) which uses
    4th-order interpolation everywhere and skips all edge/corner specials.
    This matches the ``dg%is_initialized`` branch in FV3 sw_core.F90.

    Parameters
    ----------
    u_d : (6, n, n+1) D-grid x-velocity at x-edge midpoints
    v_d : (6, n+1, n) D-grid y-velocity at y-edge midpoints
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    ua, va : (6, n, n) A-grid contravariant
    uc : (6, n+1, n) C-grid covariant u
    vc : (6, n, n+1) C-grid covariant v
    ut : (6, n+1, n) C-grid contravariant u (transport)
    vt : (6, n, n+1) C-grid contravariant v (transport)
    """
    # Dispatch to Duo-Grid path when active (FV3 sw_core.F90:3419).
    # FV3 gates on `dg%is_initialized` which is true when the duogrid
    # structure is fully allocated with sufficient halo. Here the `ng >= 2`
    # guard ensures the k2e-remapped halo covers both depths needed by the
    # 4th-order A→C stencil (halo=2). For tiny grids (n<4) where
    # ng = n//2 < 2, fall through to the legacy path.
    dg = cdgrid.base.duogrid
    if dg is not None and dg.ng >= 2:
        return _d2a2c_vect_duogrid(u_d, v_d, cdgrid)

    n = cdgrid.n
    npt = min(4, n // 2)

    # ---- Step 1: D-grid → covariant cell centres (utmp, vtmp) ----
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n)
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])   # (6, n, n)

    # 4th-order interior (at least npt cells from each edge).
    # Guard: needs n > 2*npt AND npt > 0 (i.e., n >= 2).
    if n > 2 * npt and npt > 0:
        u4 = (_A2 * (u_d[:, :, :-3] + u_d[:, :, 3:])
              + _A1 * (u_d[:, :, 1:-2] + u_d[:, :, 2:-1]))
        utmp = utmp.at[:, :, npt:n - npt].set(u4[:, :, npt - 1:n - npt - 1])
        v4 = (_A2 * (v_d[:, :-3, :] + v_d[:, 3:, :])
              + _A1 * (v_d[:, 1:-2, :] + v_d[:, 2:-1, :]))
        vtmp = vtmp.at[:, npt:n - npt, :].set(v4[:, npt - 1:n - npt - 1, :])

    # ---- Step 2: Halo-exchange COVARIANT utmp/vtmp ----
    # pad_halo_vector rotates wind components across face boundaries —
    # correct for covariant (grid-aligned) winds.
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

    # ---- Step 4a: A→C x-direction (covariant utmp → uc) ----
    # Interior: 4th-order interpolation of COVARIANT utmp
    uc = 0.5 * (utmp_pad[:, :-1, 1:-1] + utmp_pad[:, 1:, 1:-1])  # (6, n+1, n)

    if n > 2 * npt + 2:
        uc_4th = (_A2 * (utmp_pad[:, :-3, 1:-1] + utmp_pad[:, 3:, 1:-1])
                  + _A1 * (utmp_pad[:, 1:-2, 1:-1] + utmp_pad[:, 2:-1, 1:-1]))
        i_lo = npt + 1
        i_hi = n - npt
        uc = uc.at[:, i_lo:i_hi, :].set(uc_4th[:, i_lo - 1:i_hi - 1, :])

    # Near-boundary one-sided stencils (i=2, i=n-2 only — NOT at face edge itself)
    if n > 3:
        uc = uc.at[:, 2, :].set(
            _C1 * utmp_pad[:, 5, 1:-1] + _C2 * utmp_pad[:, 4, 1:-1]
            + _C3 * utmp_pad[:, 3, 1:-1])
        uc = uc.at[:, n - 2, :].set(
            _C1 * utmp_pad[:, n - 3, 1:-1] + _C2 * utmp_pad[:, n - 2, 1:-1]
            + _C3 * utmp_pad[:, n - 1, 1:-1])

    # AT face boundary (i=1, i=n-1): edge_interpolate4 on CONTRAVARIANT ua
    # then uc = ut * sin_sg_upwind (covariant from contravariant).
    # Requires n >= 2 for the 4-point edge stencil to have valid indices.
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (1, 1), (0, 0)], mode='edge')
    for i_bdy in ([1, n - 1] if n >= 2 else []):
        i_p = i_bdy  # padded offset
        ua4 = jnp.stack([ua_pad[:, i_p - 1, 1:-1], ua_pad[:, i_p, 1:-1],
                         ua_pad[:, i_p + 1, 1:-1], ua_pad[:, i_p + 2, 1:-1]],
                        axis=-1)
        dxa4 = jnp.stack([dxc_pad_x[:, i_p - 1, :], dxc_pad_x[:, i_p, :],
                          dxc_pad_x[:, i_p + 1, :], dxc_pad_x[:, i_p + 2, :]],
                         axis=-1)
        ut_bdy = _edge_interpolate4(ua4, dxa4)  # contravariant

        i_left = max(i_bdy - 1, 0)
        i_right = min(i_bdy, n - 1)
        sin_left = cdgrid.sin_sg[:, i_left, :, 2]   # E-edge of left cell
        sin_right = cdgrid.sin_sg[:, i_right, :, 0]  # W-edge of right cell
        uc_bdy = jnp.where(ut_bdy > 0, ut_bdy * sin_left, ut_bdy * sin_right)
        uc = uc.at[:, i_bdy, :].set(uc_bdy)

    # Contravariant ut from covariant uc
    cosa_u = cdgrid.cosa_u
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u ** 2, _EPS))
    ut = (uc - v_d * cosa_u) / jnp.maximum(sina_u, _EPS)

    # At face boundaries: ut = uc / sin_sg_upwind (not the cosa/sina formula)
    for i_bdy in [0, 1, n - 1, n]:
        i_left = max(i_bdy - 1, 0)
        i_right = min(i_bdy, n - 1)
        sin_left = cdgrid.sin_sg[:, i_left, :, 2]
        sin_right = cdgrid.sin_sg[:, i_right, :, 0]
        sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
        ut = ut.at[:, i_bdy, :].set(
            uc[:, i_bdy, :] / jnp.maximum(sin_upwind, _EPS))

    # ---- Step 4b: A→C y-direction (covariant vtmp → vc) ----
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

    # AT face boundary (j=1, j=n-1): edge_interpolate4 on va.
    # Requires n >= 2 for the 4-point edge stencil.
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (1, 1)], mode='edge')
    for j_bdy in ([1, n - 1] if n >= 2 else []):
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

    cosa_v = cdgrid.cosa_v
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v ** 2, _EPS))
    vt = (vc - u_d * cosa_v) / jnp.maximum(sina_v, _EPS)

    for j_bdy in [0, 1, n - 1, n]:
        j_below = max(j_bdy - 1, 0)
        j_above = min(j_bdy, n - 1)
        sin_below = cdgrid.sin_sg[:, :, j_below, 3]
        sin_above = cdgrid.sin_sg[:, :, j_above, 1]
        sin_upwind = jnp.where(vc[:, :, j_bdy] > 0, sin_below, sin_above)
        vt = vt.at[:, :, j_bdy].set(
            vc[:, :, j_bdy] / jnp.maximum(sin_upwind, _EPS))

    return ua, va, uc, vc, ut, vt


# ==============================================================================
# c_sw: C-grid half-step
# ==============================================================================

def _c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
    """FV3 c_sw: C-grid half of the forward-backward step.

    Updates mass (h) via first-order upwind transport and C-grid
    covariant velocities (uc, vc) via vorticity flux + KE gradient.

    Parameters
    ----------
    h : (6, n, n) height at cell centres
    u_d : (6, n, n+1) D-grid x-velocity
    v_d : (6, n+1, n) D-grid y-velocity
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    dt : float — full time step (half-step scaling applied internally)
    g : float — gravitational acceleration

    Returns
    -------
    h_star : (6, n, n) transported mass
    uc_new : (6, n+1, n) updated C-grid covariant u
    vc_new : (6, n, n+1) updated C-grid covariant v
    ua : (6, n, n) A-grid contravariant u (for diagnostics)
    va : (6, n, n) A-grid contravariant v
    """
    n = cdgrid.n
    dt2 = 0.5 * dt  # half-step

    # 1. d2a2c_vect
    ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

    # 2. Scale transport velocities with dt/2 * edge_length * sin_sg_upwind
    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    sg = cdgrid.sin_sg

    # x-direction: ut → scaled flux
    sin_sg_left_x = jnp.pad(sg[:, :, :, 2], [(0, 0), (1, 0), (0, 0)], mode='edge')
    sin_sg_right_x = jnp.pad(sg[:, :, :, 0], [(0, 0), (0, 1), (0, 0)], mode='edge')
    sin_upwind_x = jnp.where(ut > 0, sin_sg_left_x, sin_sg_right_x)
    ut_scaled = dt2 * ut * dy * sin_upwind_x

    # y-direction: vt → scaled flux
    sin_sg_below_y = jnp.pad(sg[:, :, :, 3], [(0, 0), (0, 0), (1, 0)], mode='edge')
    sin_sg_above_y = jnp.pad(sg[:, :, :, 1], [(0, 0), (0, 0), (0, 1)], mode='edge')
    sin_upwind_y = jnp.where(vt > 0, sin_sg_below_y, sin_sg_above_y)
    vt_scaled = dt2 * vt * dx * sin_upwind_y

    # 3. First-order upwind mass transport
    h_pad = _pad_halo_auto(h, cdgrid)
    # x-fluxes: upwind h * scaled_ut
    h_left = h_pad[:, :-1, 1:-1]   # (6, n+1, n)
    h_right = h_pad[:, 1:, 1:-1]
    fx = jnp.where(ut_scaled > 0, h_left, h_right) * ut_scaled

    # y-fluxes
    h_bot = h_pad[:, 1:-1, :-1]    # (6, n, n+1)
    h_top = h_pad[:, 1:-1, 1:]
    fy = jnp.where(vt_scaled > 0, h_bot, h_top) * vt_scaled

    rarea = 1.0 / cdgrid.base.area
    h_star = h + (fx[:, :-1, :] - fx[:, 1:, :] + fy[:, :, :-1] - fy[:, :, 1:]) * rarea

    # 4. KE at cell centres
    uc_left = uc[:, :-1, :]   # (6, n, n)
    uc_right = uc[:, 1:, :]
    ke_u = jnp.where(ua > 0, uc_left, uc_right)

    vc_bot = vc[:, :, :-1]    # (6, n, n)
    vc_top = vc[:, :, 1:]
    ke_v = jnp.where(va > 0, vc_bot, vc_top)

    # ke = dt/4 * (ua * uc_upwind + va * vc_upwind)
    # NOTE: c_sw uses ONLY kinetic energy for the C-grid gradient, NOT
    # the full Bernoulli function (g*h is in d_sw only).  This is per
    # FV3 sw_core.F90 where ke has dt4 = 0.25*dt scaling.
    ke_total = dt2 * 0.5 * (ua * ke_u + va * ke_v)

    # 5. Vorticity at D-grid corners from C-grid covariant velocities
    # FV3 c_sw uses dxc/dyc (center-to-center distances), NOT edge lengths.
    # uc is covariant along x → uc*dxc = line integral along that path.
    fx_circ = uc * cdgrid.dxc    # (6, n+1, n) — FV3: fx = uc * dxc
    fy_circ = vc * cdgrid.dyc    # (6, n, n+1) — FV3: fy = vc * dyc

    fx_pad = jnp.pad(fx_circ, [(0, 0), (0, 0), (1, 1)], mode='edge')
    fy_pad = jnp.pad(fy_circ, [(0, 0), (1, 1), (0, 0)], mode='edge')

    circ = (fx_pad[:, :, :-1] - fx_pad[:, :, 1:]
            + fy_pad[:, 1:, :] - fy_pad[:, :-1, :])

    # Cube vertex corrections — FV3 sw_core.F90:395-401:
    # SKIPPED when duogrid is active (.not. flagstruct%duogrid).
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2
    if not use_duogrid:
        circ = circ.at[:, 0, 0].add(fy_pad[:, 0, 0])
        circ = circ.at[:, n, 0].add(-fy_pad[:, n + 1, 0])
        circ = circ.at[:, n, n].add(-fy_pad[:, n + 1, n])
        circ = circ.at[:, 0, n].add(fy_pad[:, 0, n])

    vort = circ * cdgrid.rarea_c
    vort_abs = vort + cdgrid.f_corner

    # 6. Vorticity flux at C-grid face positions
    # FV3 sw_core.F90:622-726: when duogrid is active, use simple formula
    # everywhere (no face-boundary overrides). When not active, override
    # fy1 at face boundaries with dt2 * v_d.
    cosa_u = cdgrid.cosa_u
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u ** 2, _EPS))
    fy1 = dt2 * (v_d - uc * cosa_u) / jnp.maximum(sina_u, _EPS)
    if not use_duogrid:
        # At face edges: sin_sg cancellation → fy1 = dt2 * v_d
        fy1 = fy1.at[:, 0, :].set(dt2 * v_d[:, 0, :])
        fy1 = fy1.at[:, 1, :].set(dt2 * v_d[:, 1, :])
        fy1 = fy1.at[:, n - 1, :].set(dt2 * v_d[:, n - 1, :])
        fy1 = fy1.at[:, n, :].set(dt2 * v_d[:, n, :])

    vort_x = jnp.where(fy1 > 0, vort_abs[:, :, :-1], vort_abs[:, :, 1:])

    # y-face: cross-velocity in i-direction
    cosa_v = cdgrid.cosa_v
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v ** 2, _EPS))
    fx1 = dt2 * (u_d - vc * cosa_v) / jnp.maximum(sina_v, _EPS)
    if not use_duogrid:
        fx1 = fx1.at[:, :, 0].set(dt2 * u_d[:, :, 0])
        fx1 = fx1.at[:, :, 1].set(dt2 * u_d[:, :, 1])
        fx1 = fx1.at[:, :, n - 1].set(dt2 * u_d[:, :, n - 1])
        fx1 = fx1.at[:, :, n].set(dt2 * u_d[:, :, n])

    vort_y = jnp.where(fx1 > 0, vort_abs[:, :-1, :], vort_abs[:, 1:, :])

    # 7. KE gradient at C-grid face positions (2-point difference)
    ke_pad = _pad_halo_auto(ke_total, cdgrid)
    dke_x = cdgrid.rdxc * (ke_pad[:, :-1, 1:-1] - ke_pad[:, 1:, 1:-1])
    dke_y = cdgrid.rdyc * (ke_pad[:, 1:-1, :-1] - ke_pad[:, 1:-1, 1:])

    # 8. Update C-grid covariant velocities
    uc_new = uc + fy1 * vort_x + dke_x
    vc_new = vc - fx1 * vort_y + dke_y

    return h_star, uc_new, vc_new, ua, va


# ==============================================================================
# C-grid tendency for RK3 integration
# ==============================================================================

def fv3_csw_tendencies(h, u_d, v_d, h_s, cdgrid, g=9.80616,
                       div_damp=0.0, hyperdiff_coeff=0.0):
    """FV3 c_sw-style shallow water tendencies for RK3 integration.

    Computes the FULL Bernoulli gradient + vorticity flux at C-grid
    face positions (same stagger → geostrophic balance preserved), then
    projects the TOTAL C-grid tendency to D-grid edge-midpoint positions.

    Projecting the SUM (which is near zero for balanced flow) preserves
    balance, unlike projecting gradient and vorticity separately.

    Parameters
    ----------
    h : (6, n, n) height at cell centres
    u_d : (6, n, n+1) D-grid x-velocity
    v_d : (6, n+1, n) D-grid y-velocity
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    dh_dt : (6, n, n)
    du_dt : (6, n, n+1)
    dv_dt : (6, n+1, n)
    """
    n = cdgrid.n

    from legoesm.core.operators_cdgrid import fv3_d2cc, fv3_cc2c

    # 1. d2a2c_vect: D→A→C for KE and vorticity (covariant convention)
    ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

    # 2. Mass transport uses the PROVEN fv3_cc2c (physical face-normal).
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    uc_mass, vc_mass = fv3_cc2c(u_cc, v_cc, cdgrid)
    dh_dt = cgrid_mass_flux_divergence(h, uc_mass, vc_mass, cdgrid)

    # 3. KE at cell centres (FV3 upwind formula)
    uc_left = uc[:, :-1, :]
    uc_right = uc[:, 1:, :]
    ke_u = jnp.where(ua > 0, uc_left, uc_right)

    vc_bot = vc[:, :, :-1]
    vc_top = vc[:, :, 1:]
    ke_v = jnp.where(va > 0, vc_bot, vc_top)

    ke = 0.5 * (ua * ke_u + va * ke_v)
    B = ke + g * (h + h_s)

    # 4. Bernoulli gradient at C-grid face positions (2-point difference)
    B_pad = _pad_halo_auto(B, cdgrid)
    dB_x = cdgrid.rdxc * (B_pad[:, :-1, 1:-1] - B_pad[:, 1:, 1:-1])  # (6, n+1, n)
    dB_y = cdgrid.rdyc * (B_pad[:, 1:-1, :-1] - B_pad[:, 1:-1, 1:])  # (6, n, n+1)

    # 5. Vorticity at D-grid corners from C-grid covariant velocities
    # FV3: fx = uc * dxc, fy = vc * dyc (center-to-center distances)
    fx_circ = uc * cdgrid.dxc
    fy_circ = vc * cdgrid.dyc
    fx_pad = jnp.pad(fx_circ, [(0, 0), (0, 0), (1, 1)], mode='edge')
    fy_pad = jnp.pad(fy_circ, [(0, 0), (1, 1), (0, 0)], mode='edge')

    circ = (fx_pad[:, :, :-1] - fx_pad[:, :, 1:]
            + fy_pad[:, 1:, :] - fy_pad[:, :-1, :])
    # FV3 sw_core.F90:395-401: corner corrections skipped with duogrid
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2
    if not use_duogrid:
        circ = circ.at[:, 0, 0].add(fy_pad[:, 0, 0])
        circ = circ.at[:, n, 0].add(-fy_pad[:, n + 1, 0])
        circ = circ.at[:, n, n].add(-fy_pad[:, n + 1, n])
        circ = circ.at[:, 0, n].add(fy_pad[:, 0, n])

    vort_abs = circ * cdgrid.rarea_c + cdgrid.f_corner

    # 6. Vorticity flux at C-grid face positions
    # FV3 sw_core.F90:622: when duogrid active, use uniform formula
    cosa_u = cdgrid.cosa_u
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u ** 2, _EPS))
    fy1 = (v_d - uc * cosa_u) / jnp.maximum(sina_u, _EPS)
    if not use_duogrid:
        fy1 = fy1.at[:, 0, :].set(v_d[:, 0, :])
        fy1 = fy1.at[:, 1, :].set(v_d[:, 1, :])
        fy1 = fy1.at[:, n - 1, :].set(v_d[:, n - 1, :])
        fy1 = fy1.at[:, n, :].set(v_d[:, n, :])
    vort_x = jnp.where(fy1 > 0, vort_abs[:, :, :-1], vort_abs[:, :, 1:])

    cosa_v = cdgrid.cosa_v
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v ** 2, _EPS))
    fx1 = (u_d - vc * cosa_v) / jnp.maximum(sina_v, _EPS)
    if not use_duogrid:
        fx1 = fx1.at[:, :, 0].set(u_d[:, :, 0])
        fx1 = fx1.at[:, :, 1].set(u_d[:, :, 1])
        fx1 = fx1.at[:, :, n - 1].set(u_d[:, :, n - 1])
        fx1 = fx1.at[:, :, n].set(u_d[:, :, n])
    vort_y = jnp.where(fx1 > 0, vort_abs[:, :-1, :], vort_abs[:, 1:, :])

    # 7. TOTAL C-grid tendency = vorticity flux + Bernoulli gradient
    duc = fy1 * vort_x + dB_x    # (6, n+1, n)
    dvc = -fx1 * vort_y + dB_y   # (6, n, n+1)

    # 8. Divergence damping (optional)
    if div_damp > 0:
        div_field = cgrid_divergence(uc, vc, cdgrid)
        div_pad = _pad_halo_auto(div_field, cdgrid)
        ddiv_x = cdgrid.rdxc * (div_pad[:, :-1, 1:-1] - div_pad[:, 1:, 1:-1])
        ddiv_y = cdgrid.rdyc * (div_pad[:, 1:-1, :-1] - div_pad[:, 1:-1, 1:])
        duc = duc + div_damp * ddiv_x
        dvc = dvc + div_damp * ddiv_y

    # 9. Project TOTAL C-grid tendency → D-grid edge midpoints (4-point avg)
    # Projecting the SUM (near zero for balanced flow) preserves balance.
    duc_pad = jnp.pad(duc, [(0, 0), (0, 0), (1, 1)], mode='edge')
    du_dt = 0.25 * (duc_pad[:, :-1, :-1] + duc_pad[:, 1:, :-1]
                     + duc_pad[:, :-1, 1:] + duc_pad[:, 1:, 1:])

    dvc_pad = jnp.pad(dvc, [(0, 0), (1, 1), (0, 0)], mode='edge')
    dv_dt = 0.25 * (dvc_pad[:, :-1, :-1] + dvc_pad[:, 1:, :-1]
                     + dvc_pad[:, :-1, 1:] + dvc_pad[:, 1:, 1:])

    return dh_dt, du_dt, dv_dt


# ==============================================================================
# d_sw: D-grid half-step
# ==============================================================================

def _d_sw(h, h_star, u_d, v_d, h_s, uc_new, vc_new, cdgrid, dt, g,
          div_damp=0.0, hyperdiff_coeff=0.0):
    """FV3 d_sw: D-grid half of the forward-backward step.

    Uses the UPDATED C-grid velocities from c_sw for mass transport
    (PPM), then updates D-grid winds using the existing Arakawa-Lamb
    gradient + corner vorticity (internally consistent at D-grid stagger).

    The Bernoulli gradient uses the ORIGINAL h (same time level as u_d/v_d)
    to maintain geostrophic balance.  The mass update uses h_star from c_sw.

    Parameters
    ----------
    h : (6, n, n) ORIGINAL height (for Bernoulli gradient, same time level as u_d)
    h_star : (6, n, n) mass from c_sw (for mass transport)
    u_d : (6, n, n+1) OLD D-grid x-velocity
    v_d : (6, n+1, n) OLD D-grid y-velocity
    h_s : (6, n, n) surface topography
    uc_new : (6, n+1, n) UPDATED C-grid covariant u from c_sw
    vc_new : (6, n, n+1) UPDATED C-grid covariant v from c_sw
    cdgrid : CubedSphereCDGrid
    dt, g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    h_new : (6, n, n)
    u_d_new : (6, n, n+1)
    v_d_new : (6, n+1, n)
    """
    n = cdgrid.n

    # 1. Convert updated covariant uc/vc to physical face-normal velocities.
    # cgrid_mass_flux_divergence builds fluxes as h_face * u_c * dy, so u_c
    # must be the physical face-normal velocity, not the covariant projection.
    # Physical face-normal: uc_phys = (uc_cov - v_at_u * cosa_u) * rsin_u
    # We approximate v_at_u from the D-grid v_d (same time level as uc_new
    # was derived from) via simple averaging to the u-face position.
    cosa_u = cdgrid.cosa_u     # (6, n+1, n)
    rsin_u = cdgrid.rsin_u     # (6, n+1, n)
    cosa_v = cdgrid.cosa_v     # (6, n, n+1)
    rsin_v = cdgrid.rsin_v     # (6, n, n+1)

    # v_d (6, n+1, n) is co-located with cosa_u — use directly as the
    # cross-velocity at u-face positions (same approximation as _uc_to_ut).
    v_at_u = v_d  # (6, n+1, n)

    # u_d (6, n, n+1) is co-located with cosa_v — use directly.
    u_at_v = u_d  # (6, n, n+1)

    uc_phys = (uc_new - v_at_u * cosa_u) * rsin_u
    vc_phys = (vc_new - u_at_v * cosa_v) * rsin_v

    # 2. Mass transport (PPM) with physical face-normal velocities (dt/2 half-step).
    dh = cgrid_mass_flux_divergence(h_star, uc_phys, vc_phys, cdgrid)
    h_new = h_star + 0.5 * dt * dh

    # 3. D-grid momentum update using existing corner operators.
    # Uses ORIGINAL h (same time level as u_d/v_d) for geostrophic balance.
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
    B = KE + g * (h + h_s)
    dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)

    # Corner winds for vorticity
    u_d_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    u_corner = 0.5 * (u_d_pad[:, :-1, :] + u_d_pad[:, 1:, :])
    v_d_pad = jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')
    v_corner = 0.5 * (v_d_pad[:, :, :-1] + v_d_pad[:, :, 1:])

    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
    zeta_abs = zeta + cdgrid.base.f
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    du_corner = zeta_corner * v_corner - dB_dx
    dv_corner = -zeta_corner * u_corner - dB_dy_perp

    # Divergence damping
    if div_damp > 0:
        div_field = cgrid_divergence(uc_new, vc_new, cdgrid)
        area_min = jnp.min(cdgrid.base.area)
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs_corner = _interp_center_to_corner(jnp.abs(div_field), cdgrid)
        adaptive_coeff = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        ddiv_dx, ddiv_dy = _arakawa_lamb_gradient(div_field, cdgrid)
        du_corner = du_corner + adaptive_coeff * ddiv_dx
        dv_corner = dv_corner + adaptive_coeff * ddiv_dy

    # Biharmonic hyperdiffusion
    if hyperdiff_coeff > 0:
        from legoesm.core.operators_cdgrid import _laplacian_dgrid
        du_corner = du_corner - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_corner, cdgrid), cdgrid)
        dv_corner = dv_corner - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_corner, cdgrid), cdgrid)

    # Boundary fix (consistent at D-grid stagger)
    du_corner, dv_corner = _extrapolate_boundary_corners(du_corner, dv_corner, n)

    # Average corner tendencies to edge-midpoint D-grid
    du_d_dt = 0.5 * (du_corner[:, :-1, :] + du_corner[:, 1:, :])
    dv_d_dt = 0.5 * (dv_corner[:, :, :-1] + dv_corner[:, :, 1:])

    # Forward Euler update with full dt (d_sw is the primary D-grid update)
    u_d_new = u_d + dt * du_d_dt
    v_d_new = v_d + dt * dv_d_dt

    # Note: The c_sw half already contributed a KE gradient correction
    # at C-grid positions (dt/4 scaling). The d_sw provides the FULL
    # Bernoulli gradient + vorticity at D-grid positions (dt scaling).
    # The forward-backward coupling is through the mass field (h_star)
    # and the updated uc_new/vc_new used for transport.

    return h_new, u_d_new, v_d_new


# ==============================================================================
# Complete forward-backward step
# ==============================================================================

def fv3_forward_backward_step(h, u_d, v_d, h_s, cdgrid, dt, g=9.80616,
                               div_damp=0.0, hyperdiff_coeff=0.0):
    """EXPERIMENTAL: One complete FV3 forward-backward time step for shallow water.

    Known unstable — produces large errors by step ~50.  Use
    ``fv3_sw_tendencies`` (operators_cdgrid.py) with RK3 integration
    for production work.

    Combines c_sw (C-grid half) + d_sw (D-grid half).  The c_sw half
    uses FV3's covariant velocity convention with sin_sg flux scaling;
    the d_sw half uses the existing corner-based operators (consistent
    at D-grid stagger).  The forward-backward coupling connects them
    without projecting between staggers.

    Parameters
    ----------
    h : (6, n, n) height at cell centres
    u_d : (6, n, n+1) D-grid x-velocity at x-edge midpoints
    v_d : (6, n+1, n) D-grid y-velocity at y-edge midpoints
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    dt : float — full time step
    g : float
    div_damp : float — divergence damping coefficient [m²/s]
    hyperdiff_coeff : float — biharmonic hyperdiffusion coefficient

    Returns
    -------
    h_new : (6, n, n)
    u_d_new : (6, n, n+1)
    v_d_new : (6, n+1, n)
    """
    # C-grid half: mass transport + C-grid velocity update
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # D-grid half: PPM mass transport + D-grid velocity update
    h_new, u_d_new, v_d_new = _d_sw(
        h, h_star, u_d, v_d, h_s, uc_new, vc_new, cdgrid, dt, g,
        div_damp=div_damp, hyperdiff_coeff=hyperdiff_coeff)

    return h_new, u_d_new, v_d_new


# ==============================================================================
# NEW: FV3-faithful forward-backward WITHOUT Arakawa-Lamb gradient
# ==============================================================================

def _p_grad_c(h_star, h_s, cdgrid, dt2, g):
    """Backward pressure gradient at C-grid positions.

    Applies g*grad(h_star + h_s) at C-grid face positions using a 2-point
    divided difference.  This is the "backward-in-time" step that couples
    mass and momentum implicitly, providing stability for gravity waves.

    Parameters
    ----------
    h_star : (6, n, n) — half-step height from c_sw
    h_s : (6, n, n) — surface topography
    cdgrid : CubedSphereCDGrid
    dt2 : float — dt/2
    g : float

    Returns
    -------
    dp_x : (6, n+1, n) — pressure gradient contribution to uc
    dp_y : (6, n, n+1) — pressure gradient contribution to vc
    """
    p = g * (h_star + h_s)
    p_pad = _pad_halo_auto(p, cdgrid)
    # 2-point gradient at C-grid faces (same sign convention as c_sw KE gradient)
    dp_x = dt2 * cdgrid.rdxc * (p_pad[:, :-1, 1:-1] - p_pad[:, 1:, 1:-1])
    dp_y = dt2 * cdgrid.rdyc * (p_pad[:, 1:-1, :-1] - p_pad[:, 1:-1, 1:])
    return dp_x, dp_y


def _uc_to_ut(uc, vc, u_d, v_d, cdgrid):
    """Convert updated C-grid covariant (uc, vc) to contravariant (ut, vt).

    Uses metric correction at interior and sin_sg upwinding at face boundaries.

    Parameters
    ----------
    uc : (6, n+1, n) — updated covariant C-grid u
    vc : (6, n, n+1) — updated covariant C-grid v
    u_d : (6, n, n+1) — D-grid x-wind (for metric correction)
    v_d : (6, n+1, n) — D-grid y-wind (for metric correction)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    ut : (6, n+1, n) — contravariant transport u at C-grid x-faces
    vt : (6, n, n+1) — contravariant transport v at C-grid y-faces
    """
    n = cdgrid.n
    cosa_u = cdgrid.cosa_u     # (6, n+1, n)
    rsin_u = cdgrid.rsin_u     # (6, n+1, n)
    cosa_v = cdgrid.cosa_v     # (6, n, n+1)
    rsin_v = cdgrid.rsin_v     # (6, n, n+1)

    # v_d (6, n+1, n) has the same shape as cosa_u — use directly as
    # the cross-velocity at u-face positions (co-located approximation,
    # same as _d2a2c_vect line 167).  Similarly u_d (6, n, n+1) matches cosa_v.
    ut = (uc - v_d * cosa_u) * rsin_u
    vt = (vc - u_d * cosa_v) * rsin_v

    # At face boundaries: ut = uc / sin_sg_upwind
    sg = cdgrid.sin_sg
    for i_bdy in [0, 1, n - 1, n]:
        i_left = max(i_bdy - 1, 0)
        i_right = min(i_bdy, n - 1)
        sin_left = sg[:, i_left, :, 2]    # E-edge of left cell
        sin_right = sg[:, i_right, :, 0]  # W-edge of right cell
        sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
        ut = ut.at[:, i_bdy, :].set(
            uc[:, i_bdy, :] / jnp.maximum(sin_upwind, _EPS))

    for j_bdy in [0, 1, n - 1, n]:
        j_below = max(j_bdy - 1, 0)
        j_above = min(j_bdy, n - 1)
        sin_below = sg[:, :, j_below, 3]  # N-edge of cell below
        sin_above = sg[:, :, j_above, 1]  # S-edge of cell above
        sin_upwind = jnp.where(vc[:, :, j_bdy] > 0, sin_below, sin_above)
        vt = vt.at[:, :, j_bdy].set(
            vc[:, :, j_bdy] / jnp.maximum(sin_upwind, _EPS))

    return ut, vt


def _d_sw_native(h, u_d, v_d, uc, vc, ua, va, cdgrid, dt, g,
                 div_damp=0.0):
    """D-grid full-step without Arakawa-Lamb gradient.

    Mass transport uses PPM via the updated C-grid velocities (which already
    include the backward pressure gradient from p_grad_c).  D-grid winds are
    updated using:
    - KE at corners (interpolated from cell centres, NO A-L stencil)
    - Vorticity transport to D-grid edges via fv_tp_2d
    - Divergence damping at corners

    The pressure gradient is NOT in this function — it was already incorporated
    into uc/vc by p_grad_c.  The D-grid wind update involves only the (small)
    KE gradient and vorticity flux.  For balanced geostrophic flow (Williamson 2),
    both are near zero, so halo errors have minimal impact.

    Parameters
    ----------
    h : (6, n, n) — ORIGINAL height (for PPM mass transport)
    u_d : (6, n, n+1) — OLD D-grid x-velocity
    v_d : (6, n+1, n) — OLD D-grid y-velocity
    uc : (6, n+1, n) — UPDATED covariant C-grid u (from c_sw + p_grad_c)
    vc : (6, n, n+1) — UPDATED covariant C-grid v (from c_sw + p_grad_c)
    ua, va : (6, n, n) — A-grid contravariant (from c_sw's d2a2c_vect)
    cdgrid : CubedSphereCDGrid
    dt : float — full time step
    g : float
    div_damp : float

    Returns
    -------
    h_new, u_d_new, v_d_new
    """
    from legoesm.core.fv_tp_2d import (
        compute_transport_quantities, fv_tp_2d, transport_step,
    )
    n = cdgrid.n

    # === 1. Contravariant transport velocity from updated C-grid ===
    ut, vt = _uc_to_ut(uc, vc, u_d, v_d, cdgrid)

    # === 2. PPM mass transport using ORIGINAL h ===
    h_new = transport_step(h, ut, vt, dt, cdgrid)

    # === 3. Cell-centre vorticity from D-grid circulation ===
    dx_u = cdgrid.dx_edge_y  # (6, n, n+1) — edge length for u_d
    dy_v = cdgrid.dy_edge_x  # (6, n+1, n) — edge length for v_d

    vt_circ = u_d * dx_u  # (6, n, n+1) — u circulation
    ut_circ = v_d * dy_v  # (6, n+1, n) — v circulation

    rarea = 1.0 / cdgrid.base.area  # (6, n, n)
    # CCW circulation: bottom - top + right - left
    zeta = rarea * (vt_circ[:, :, :-1] - vt_circ[:, :, 1:]
                    + ut_circ[:, 1:, :] - ut_circ[:, :-1, :])
    zeta_abs = zeta + cdgrid.base.f  # (6, n, n)

    # === 4. KE at cell centres (contravariant × covariant) ===
    utmp = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])  # (6, n, n)
    vtmp = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])  # (6, n, n)
    ke_cell = 0.5 * (ua * utmp + va * vtmp)  # (6, n, n)

    # === 5. KE at corners via 4-point average with halo ===
    ke_corner = _interp_center_to_corner(ke_cell, cdgrid)  # (6, n+1, n+1)

    # === 6. Divergence damping at corners (optional) ===
    if div_damp > 0:
        div_field = cgrid_divergence(uc, vc, cdgrid)
        area_min = float(jnp.min(cdgrid.base.area))
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs = jnp.abs(div_field)
        div_abs_corner = _interp_center_to_corner(div_abs, cdgrid)
        damp_coeff = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        div_corner = _interp_center_to_corner(div_field, cdgrid)
        ke_corner = ke_corner + damp_coeff * div_corner

    # === 7. KE gradient at D-grid edges (2-point corner difference) ===
    # u_d[i, j] sits between corners (i, j) and (i+1, j) in the i-direction
    ke_diff_u = ke_corner[:, :-1, :] - ke_corner[:, 1:, :]  # (6, n, n+1)
    # But ke_diff has wrong shape for u_d: (6, n+1+1-1=n+1, n+1) → need to
    # trim j to match u_d's n+1 j-values... Actually ke_corner is (n+1, n+1)
    # and :-1 / 1: in dim1 gives (n, n+1) ← matches u_d!

    # v_d[i, j] sits between corners (i, j) and (i, j+1) in the j-direction
    ke_diff_v = ke_corner[:, :, :-1] - ke_corner[:, :, 1:]  # (6, n+1, n)

    # Scale to circulation: dt * ke_diff has units s × m²/s² = m²/s
    ke_diff_u_scaled = dt * ke_diff_u
    ke_diff_v_scaled = dt * ke_diff_v

    # === 8. Vorticity transport to D-grid edges via fv_tp_2d ===
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx_vort, fy_vort = fv_tp_2d(
        zeta_abs, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    # fx_vort: (6, n+1, n) — vorticity flux at x-interfaces (v_d positions)
    # fy_vort: (6, n, n+1) — vorticity flux at y-interfaces (u_d positions)

    # === 9. D-grid wind update ===
    # Circulation form: u_new * dx = u_old * dx + ke_diff + fy_vort
    #                   v_new * dy = v_old * dy + ke_diff - fx_vort
    rdx_u = 1.0 / jnp.maximum(dx_u, _EPS)  # (6, n, n+1)
    rdy_v = 1.0 / jnp.maximum(dy_v, _EPS)  # (6, n+1, n)

    u_d_new = u_d + (ke_diff_u_scaled + fy_vort) * rdx_u
    v_d_new = v_d + (ke_diff_v_scaled - fx_vort) * rdy_v

    return h_new, u_d_new, v_d_new


def fv3_fb_sw_step(h, u_d, v_d, h_s, cdgrid, dt, g=9.80616,
                   div_damp=0.0):
    """EXPERIMENTAL: Complete FV3 forward-backward shallow water time step.

    Known unstable (85 m/s v-wind after 1 day, 3% mass error).
    Use ``fv3_sw_tendencies`` with RK3 for production work.

    Three phases:
    1. c_sw (forward, dt/2): d2a2c_vect + mass transport + KE/vorticity
       update at C-grid.
    2. p_grad_c (backward, dt/2): pressure gradient at C-grid using
       transported mass (h_star).  This implicit coupling provides stability
       for gravity waves and keeps the large pressure gradient at C-grid
       where the 2-point stencil is well-conditioned.
    3. d_sw (full dt): PPM mass transport + D-grid wind update using ONLY
       KE gradient (corner differences) and vorticity transport.  No
       Arakawa-Lamb gradient — the pressure gradient is already in uc/vc.

    For balanced geostrophic flow (Williamson 2), the D-grid winds change
    by only the small KE and vorticity terms.  This makes the scheme
    insensitive to halo interpolation errors at face boundaries, unlike
    the RK3 approach where the full Bernoulli gradient (dominated by g*h)
    must be computed at D-grid corners with haloed cell-centre data.

    Parameters
    ----------
    h : (6, n, n) height
    u_d : (6, n, n+1) D-grid x-velocity (edge midpoints)
    v_d : (6, n+1, n) D-grid y-velocity (edge midpoints)
    h_s : (6, n, n) surface topography
    cdgrid : CubedSphereCDGrid
    dt : float
    g : float
    div_damp : float

    Returns
    -------
    h_new, u_d_new, v_d_new
    """
    dt2 = 0.5 * dt

    # Phase 1: c_sw — forward half-step at C-grid (KE + vorticity only)
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # Phase 2: p_grad_c — backward pressure gradient at C-grid
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    # Phase 3: d_sw — full-step D-grid update (no A-L gradient)
    h_new, u_d_new, v_d_new = _d_sw_native(
        h, u_d, v_d, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp)

    return h_new, u_d_new, v_d_new
