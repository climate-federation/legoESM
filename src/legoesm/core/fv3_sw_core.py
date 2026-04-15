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
from legoesm.grids.halo import pad_halo, pad_halo_vector
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


def _d_sw1_recompute_ut_vt(uc, vc, cdgrid, dt):
    """FV3 d_sw1 transport velocity recomputation with boundary handling.

    Recomputes contravariant transport velocities (ut, vt) from covariant
    C-grid velocities (uc, vc) using FV3's 4-cell cross-velocity average,
    face-boundary overrides (sin_sg upwind), adjacent strip recomputation,
    and corner 2×2 solve. Matches sw_core.F90:618-812.

    This replaces the ut/vt from d2a2c_vect with more accurate transport
    velocities that have cross-face consistent boundary handling.

    Parameters
    ----------
    uc : (6, n+1, n) covariant C-grid u
    vc : (6, n, n+1) covariant C-grid v
    cdgrid : CubedSphereCDGrid
    dt : float — time step (for upwind sign test)

    Returns
    -------
    ut : (6, n+1, n) contravariant transport u
    vt : (6, n, n+1) contravariant transport v
    """
    n = cdgrid.n
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    cosa_u = cdgrid.cosa_u  # (6, n+1, n)
    cosa_v = cdgrid.cosa_v  # (6, n, n+1)
    rsin_u = cdgrid.rsin_u  # (6, n+1, n)
    rsin_v = cdgrid.rsin_v  # (6, n, n+1)
    sg = cdgrid.sin_sg

    # === Part 1: Interior ut/vt from 4-cell vc/uc average ===
    # ut(I,j) = (uc(I,j) - 0.25*cosa_u*(vc(I-1,j)+vc(I,j)+vc(I-1,j+1)+vc(I,j+1)))*rsin_u
    # Need vc padded in axis 1 (rows) for the I-1 stencil
    vc_pad = jnp.pad(vc, [(0, 0), (1, 1), (0, 0)], mode='edge')  # (6, n+2, n+1)
    # vc_pad[:, I, :] = vc at padded row I → original row I-1
    # 4-cell average at each u-face (I, j): vc(I-1,j)+vc(I,j)+vc(I-1,j+1)+vc(I,j+1)
    vc_avg = (vc_pad[:, :-1, :-1] + vc_pad[:, 1:, :-1]
              + vc_pad[:, :-1, 1:] + vc_pad[:, 1:, 1:])  # (6, n+1, n)
    ut = (uc - 0.25 * cosa_u * vc_avg) * rsin_u

    # vt(i,J) = (vc(i,J) - 0.25*cosa_v*(uc(i,J-1)+uc(i+1,J-1)+uc(i,J)+uc(i+1,J)))*rsin_v
    uc_pad = jnp.pad(uc, [(0, 0), (0, 0), (1, 1)], mode='edge')  # (6, n+1, n+2)
    uc_avg = (uc_pad[:, :-1, :-1] + uc_pad[:, 1:, :-1]
              + uc_pad[:, :-1, 1:] + uc_pad[:, 1:, 1:])  # (6, n, n+1)
    vt = (vc - 0.25 * cosa_v * uc_avg) * rsin_v

    if use_duogrid:
        return ut, vt

    # === Part 2: Non-duogrid face-boundary overrides ===
    # West face (I=0): ut = uc / sin_sg(upwind)
    sin_w_left = sg[:, :, :, 2]   # E-edge of cell to the left
    sin_w_right = sg[:, :, :, 0]  # W-edge of cell to the right
    # Pad sin_sg for cross-face upwind selection
    from legoesm.grids.halo import pad_halo
    grid = cdgrid.base
    offsets = grid.halo_interp_offsets
    se_pad = pad_halo(sin_w_left, interp_offsets=offsets)
    sw_pad = pad_halo(sin_w_right, interp_offsets=offsets)

    # Face boundary at I=0 (Python) = Fortran I=1
    sin_upwind_w = jnp.where(uc[:, 0, :] * dt > 0,
                             se_pad[:, :n+1, 1:-1][:, 0, :],
                             sw_pad[:, 1:n+2, 1:-1][:, 0, :])
    ut = ut.at[:, 0, :].set(uc[:, 0, :] / jnp.maximum(jnp.abs(sin_upwind_w), _EPS))

    # East face (I=n)
    sin_upwind_e = jnp.where(uc[:, n, :] * dt > 0,
                             se_pad[:, :n+1, 1:-1][:, n, :],
                             sw_pad[:, 1:n+2, 1:-1][:, n, :])
    ut = ut.at[:, n, :].set(uc[:, n, :] / jnp.maximum(jnp.abs(sin_upwind_e), _EPS))

    # South face (J=0): vt = vc / sin_sg(upwind)
    sin_s_below = sg[:, :, :, 3]  # N-edge of cell below
    sin_s_above = sg[:, :, :, 1]  # S-edge of cell above
    sn_pad = pad_halo(sin_s_below, interp_offsets=offsets)
    ss_pad = pad_halo(sin_s_above, interp_offsets=offsets)

    sin_upwind_s = jnp.where(vc[:, :, 0] * dt > 0,
                             sn_pad[:, 1:-1, :n+1][:, :, 0],
                             ss_pad[:, 1:-1, 1:n+2][:, :, 0])
    vt = vt.at[:, :, 0].set(vc[:, :, 0] / jnp.maximum(jnp.abs(sin_upwind_s), _EPS))

    # North face (J=n)
    sin_upwind_n = jnp.where(vc[:, :, n] * dt > 0,
                             sn_pad[:, 1:-1, :n+1][:, :, n],
                             ss_pad[:, 1:-1, 1:n+2][:, :, n])
    vt = vt.at[:, :, n].set(vc[:, :, n] / jnp.maximum(jnp.abs(sin_upwind_n), _EPS))

    # === Part 3: Adjacent strip recomputation ===
    # After overriding boundary ut, recompute vt at the 2 rows nearest
    # each face boundary using the corrected ut values.
    # FV3 sw_core.F90:666-726.
    # West edge: vt at rows 0 and 1, for interior j range
    # vt(row,J) = vc(row,J) - 0.25*cosa_v(row,J)*(ut(row,J-1)+ut(row+1,J-1)+ut(row,J)+ut(row+1,J))
    if n > 4:
        jlo = 2         # Fortran max(3,js) → Python 2
        jhi = n - 1     # Fortran min(npy-2,je+1) → Python n-1

        # West: rows 0, 1
        for row in [0, 1]:
            # ut at rows row and row+1, cols J-1 and J
            ut_left = ut[:, row, :-1] + ut[:, row + 1, :-1]   # (6, n-1)  J=0..n-2
            ut_right = ut[:, row, 1:] + ut[:, row + 1, 1:]    # (6, n-1)  J=1..n-1
            # 4-cell average at J positions 1..n-1 (interior v-face positions)
            ut4 = ut_left[:, :-1] + ut_right[:, 1:]  # wait, need J-1 and J
            # Actually: for v-face J, ut(row, J-1) + ut(row+1, J-1) + ut(row, J) + ut(row+1, J)
            # J ranges 0..n. ut has n columns (0..n-1). So J-1 valid for J>=1, J valid for J<=n-1.
            # Interior range: J = jlo..jhi = 2..n-1
            for J in range(jlo, jhi + 1):
                if J - 1 >= 0 and J - 1 < n and J < n:
                    avg = ut[:, row, J-1] + ut[:, row+1, J-1] + ut[:, row, J] + ut[:, row+1, J]
                    vt = vt.at[:, row, J].set(vc[:, row, J] - 0.25 * cosa_v[:, row, J] * avg)

        # East: rows n-2, n-1
        for row in [n-2, n-1]:
            if row >= 0 and row < n and row + 1 <= n:
                for J in range(jlo, jhi + 1):
                    if J - 1 >= 0 and J - 1 < n and J < n:
                        avg = ut[:, row, J-1] + ut[:, row+1, J-1] + ut[:, row, J] + ut[:, row+1, J]
                        vt = vt.at[:, row, J].set(vc[:, row, J] - 0.25 * cosa_v[:, row, J] * avg)

        # South: cols 0, 1 — recompute ut using corrected vt
        ilo = 2; ihi = n - 1
        for col in [0, 1]:
            if col + 1 <= n:
                for I in range(ilo, ihi + 1):
                    if I - 1 >= 0 and I - 1 < n and I < n:
                        avg = vt[:, I-1, col] + vt[:, I, col] + vt[:, I-1, col+1] + vt[:, I, col+1]
                        ut = ut.at[:, I, col].set(uc[:, I, col] - 0.25 * cosa_u[:, I, col] * avg)

        # North: cols n-1, n
        for col in [n-1, n]:
            if col - 1 >= 0 and col < n + 1:
                for I in range(ilo, ihi + 1):
                    if I - 1 >= 0 and I - 1 < n and I < n and col - 1 >= 0:
                        avg = vt[:, I-1, col-1] + vt[:, I, col-1] + vt[:, I-1, col] + vt[:, I, col]
                        ut = ut.at[:, I, col].set(uc[:, I, col] - 0.25 * cosa_u[:, I, col] * avg)

    # === Part 4: Corner 2×2 solve ===
    # At each cube vertex, solve a coupled system for the ut/vt values
    # near the corner. FV3 sw_core.F90:739-811.
    # For the SW corner:
    #   damp = 1/(1 - 0.0625*cosa_u(2,1)*cosa_v(1,2))
    #   ut(2,1) = (uc(2,1) - 0.25*cosa_u(2,1)*(vt(1,1)+vt(2,1)+vt(2,2)+vc(1,2)
    #              - 0.25*cosa_v(1,2)*(ut(1,1)+ut(1,2)+ut(2,2)))) * damp
    # In Python 0-based: Fortran (2,1) → Python (1,0)

    # SW corner: Fortran (2,1) → Python ut(1, 0), vt(0, 1)
    cu = cosa_u  # (6, n+1, n)
    cv = cosa_v  # (6, n, n+1)

    # Interior solve: Fortran ut(2,1) → Python ut[:, 1, 0]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, 1, 0] * cv[:, 0, 1])
    ut = ut.at[:, 1, 0].set(
        (uc[:, 1, 0] - 0.25 * cu[:, 1, 0] * (
            vt[:, 0, 0] + vt[:, 1, 0] + vt[:, 1, 1] + vc[:, 0, 1]
            - 0.25 * cv[:, 0, 1] * (ut[:, 0, 0] + ut[:, 0, 1] + ut[:, 1, 1])
        )) * damp)
    vt = vt.at[:, 0, 1].set(
        (vc[:, 0, 1] - 0.25 * cv[:, 0, 1] * (
            ut[:, 0, 0] + ut[:, 0, 1] + ut[:, 1, 1] + uc[:, 1, 0]
            - 0.25 * cu[:, 1, 0] * (vt[:, 0, 0] + vt[:, 1, 0] + vt[:, 1, 1])
        )) * damp)

    # SE corner: Fortran ut(npx-1,1) → Python ut[:, n-1, 0], vt(npx-1,2) → vt[:, n-2, 1]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, n-1, 0] * cv[:, n-2, 1])
    ut = ut.at[:, n-1, 0].set(
        (uc[:, n-1, 0] - 0.25 * cu[:, n-1, 0] * (
            vt[:, n-2, 0] + vt[:, n-3, 0] + vt[:, n-3, 1] + vc[:, n-2, 1]
            - 0.25 * cv[:, n-2, 1] * (ut[:, n, 0] + ut[:, n, 1] + ut[:, n-1, 1])
        )) * damp)
    vt = vt.at[:, n-2, 1].set(
        (vc[:, n-2, 1] - 0.25 * cv[:, n-2, 1] * (
            ut[:, n, 0] + ut[:, n, 1] + ut[:, n-1, 1] + uc[:, n-1, 0]
            - 0.25 * cu[:, n-1, 0] * (vt[:, n-2, 0] + vt[:, n-3, 0] + vt[:, n-3, 1])
        )) * damp)

    # NE corner: Fortran ut(npx-1,npy-1) → Python ut[:, n-1, n-1], vt(npx-1,npy-1) → vt[:, n-2, n-1]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, n-1, n-1] * cv[:, n-2, n-1])
    ut = ut.at[:, n-1, n-1].set(
        (uc[:, n-1, n-1] - 0.25 * cu[:, n-1, n-1] * (
            vt[:, n-2, n] + vt[:, n-3, n] + vt[:, n-3, n-1] + vc[:, n-2, n-1]
            - 0.25 * cv[:, n-2, n-1] * (ut[:, n, n-1] + ut[:, n, n-2] + ut[:, n-1, n-2])
        )) * damp)
    vt = vt.at[:, n-2, n-1].set(
        (vc[:, n-2, n-1] - 0.25 * cv[:, n-2, n-1] * (
            ut[:, n, n-1] + ut[:, n, n-2] + ut[:, n-1, n-2] + uc[:, n-1, n-1]
            - 0.25 * cu[:, n-1, n-1] * (vt[:, n-2, n] + vt[:, n-3, n] + vt[:, n-3, n-1])
        )) * damp)

    # NW corner: Fortran ut(2,npy-1) → Python ut[:, 1, n-1], vt(1,npy-1) → vt[:, 0, n-1]
    damp = 1.0 / (1.0 - 0.0625 * cu[:, 1, n-1] * cv[:, 0, n-1])
    ut = ut.at[:, 1, n-1].set(
        (uc[:, 1, n-1] - 0.25 * cu[:, 1, n-1] * (
            vt[:, 0, n] + vt[:, 1, n] + vt[:, 1, n-1] + vc[:, 0, n-1]
            - 0.25 * cv[:, 0, n-1] * (ut[:, 0, n-1] + ut[:, 0, n-2] + ut[:, 1, n-2])
        )) * damp)
    vt = vt.at[:, 0, n-1].set(
        (vc[:, 0, n-1] - 0.25 * cv[:, 0, n-1] * (
            ut[:, 0, n-1] + ut[:, 0, n-2] + ut[:, 1, n-2] + uc[:, 1, n-1]
            - 0.25 * cu[:, 1, n-1] * (vt[:, 0, n] + vt[:, 1, n] + vt[:, 1, n-1])
        )) * damp)

    return ut, vt


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
    """D→A→C with Duo-Grid: FV3-faithful via D-grid halo exchange.

    Matches the ``gridstruct%dg%is_initialized`` branch in FV3
    ``sw_core.F90:3419-3454``:
    1. D-grid staggered halo exchange (pad_halo_dgrid)
    2. 4th-order D→A averaging on haloed domain
    3. Covariant→contravariant via cosa_s/rsin2 (FV3 line 3451-3452)
    4. Scalar halo exchange of utmp/vtmp for A→C
    5. 4th-order A→C interpolation
    """
    from legoesm.grids.duogrid import pad_halo_dgrid

    n = cdgrid.n
    h = 2  # halo depth for 4th-order A→C stencil
    grid = cdgrid.base
    dg = grid.duogrid

    # ---- Step 1: D-grid staggered halo exchange ----
    # FV3 uses mpp_update_domains(DGRID_NE) before d2a2c_vect.
    u_d_pad, v_d_pad = pad_halo_dgrid(
        u_d, v_d,
        cdgrid.cos_angle_edge_x, cdgrid.sin_angle_edge_x,
        cdgrid.cos_angle_edge_y, cdgrid.sin_angle_edge_y,
        duogrid=dg,
    )  # u_d_pad: (6, n, n+3), v_d_pad: (6, n+3, n)

    # ---- Step 2: 4th-order D→A on haloed domain (FV3 sw_core.F90:3421-3435) ----
    if n > 3:
        utmp = (_A2 * (u_d_pad[:, :, :-3] + u_d_pad[:, :, 3:])
                + _A1 * (u_d_pad[:, :, 1:-2] + u_d_pad[:, :, 2:-1]))  # (6, n, n)
        vtmp = (_A2 * (v_d_pad[:, :-3, :] + v_d_pad[:, 3:, :])
                + _A1 * (v_d_pad[:, 1:-2, :] + v_d_pad[:, 2:-1, :]))  # (6, n, n)
    else:
        utmp = 0.5 * (u_d_pad[:, :, 1:-1][:, :, :-1]
                       + u_d_pad[:, :, 1:-1][:, :, 1:])
        vtmp = 0.5 * (v_d_pad[:, 1:-1, :][:, :-1, :]
                       + v_d_pad[:, 1:-1, :][:, 1:, :])

    # ---- Step 3: Covariant→contravariant at cell centres ----
    # FV3 sw_core.F90:3451-3452:
    #   ua(i,j) = (utmp(i,j)-vtmp(i,j)*cosa_s(i,j)) * rsin2(i,j)
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell
    ua = (utmp - vtmp * cos_sg5) * rsin2
    va = (vtmp - utmp * cos_sg5) * rsin2

    # ---- Step 4: VECTOR halo exchange for A→C ----
    # utmp/vtmp are covariant grid-axis projections that change meaning
    # across face boundaries — must use vector rotation (not scalar exchange).
    utmp_pad, vtmp_pad = pad_halo_vector(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        halo=h, duogrid=dg,
    )  # each (6, n+4, n+4)

    # ---- Step 5: A→C interpolation — 4th-order ----
    uc = (_A2 * (utmp_pad[:, :-3, h:-h] + utmp_pad[:, 3:, h:-h])
          + _A1 * (utmp_pad[:, 1:-2, h:-h] + utmp_pad[:, 2:-1, h:-h]))

    ut = (uc - v_d * cdgrid.cosa_u) * cdgrid.rsin_u

    vc = (_A2 * (vtmp_pad[:, h:-h, :-3] + vtmp_pad[:, h:-h, 3:])
          + _A1 * (vtmp_pad[:, h:-h, 1:-2] + vtmp_pad[:, h:-h, 2:-1]))

    vt = (vc - u_d * cdgrid.cosa_v) * cdgrid.rsin_v

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
    # halo=2 for edge_interpolate4 at face boundaries (FV3 sw_core.F90:3587).
    grid = cdgrid.base
    h = 2
    utmp_pad, vtmp_pad = pad_halo_vector(
        utmp, vtmp,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets,
        halo=h,
    )  # each (6, n+4, n+4)

    # ---- Step 3: Contravariant at cell centres (including halo) ----
    cos_sg5 = cdgrid.cos_sg[:, :, :, 4]
    rsin2 = cdgrid.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')

    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad

    ua = ua_pad[:, h:-h, h:-h]  # (6, n, n)
    va = va_pad[:, h:-h, h:-h]

    # ---- Step 4a: A→C x-direction (covariant utmp → uc) ----
    # With halo=2: padded cell k → utmp_pad[:, k+h, ...]
    # u-face i sits between cells i-1 and i → padded indices i+h-1 and i+h
    uc = 0.5 * (utmp_pad[:, h-1:n+h, h:-h]
                + utmp_pad[:, h:n+h+1, h:-h])  # (6, n+1, n)

    if n > 2 * npt + 2:
        uc_4th = (_A2 * (utmp_pad[:, h-2:n+h-1, h:-h]
                         + utmp_pad[:, h+1:n+h+2, h:-h])
                  + _A1 * (utmp_pad[:, h-1:n+h, h:-h]
                           + utmp_pad[:, h:n+h+1, h:-h]))
        i_lo = npt + 1
        i_hi = n - npt
        uc = uc.at[:, i_lo:i_hi, :].set(uc_4th[:, i_lo - 1:i_hi - 1, :])

    # One-sided c1/c2/c3 stencil at i=1, n-1 (FV3 sw_core.F90:3586,3594)
    if n > 3:
        uc = uc.at[:, 1, :].set(
            _C1 * utmp_pad[:, h + 2, h:-h] + _C2 * utmp_pad[:, h + 1, h:-h]
            + _C3 * utmp_pad[:, h, h:-h])
        uc = uc.at[:, n - 1, :].set(
            _C1 * utmp_pad[:, n + h - 3, h:-h] + _C2 * utmp_pad[:, n + h - 2, h:-h]
            + _C3 * utmp_pad[:, n + h - 1, h:-h])

    # AT face boundary (i=0, i=n): edge_interpolate4 on CONTRAVARIANT ua
    # (FV3 sw_core.F90:3587,3603). With halo=2 the 4-point stencil
    # straddles the face boundary correctly.
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    for i_bdy in ([0, n] if n >= 2 else []):
        i_p = i_bdy + h  # padded offset: cell i → padded index i+h
        # ua_pad stencil: 4 cells centred on u-face i_bdy
        ua4 = jnp.stack([ua_pad[:, i_p - 1, h:-h], ua_pad[:, i_p, h:-h],
                         ua_pad[:, i_p + 1, h:-h], ua_pad[:, i_p + 2, h:-h]],
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

    # Contravariant ut from covariant uc (FV3 rsin_u = 1/sin²)
    ut = (uc - v_d * cdgrid.cosa_u) * cdgrid.rsin_u

    # At face boundaries (i=0, n): FV3 sets ut = edge_interpolate4(ua) DIRECTLY
    # (sw_core.F90:3587,3603), not via (uc - v*cos)*rsin_u.
    # Since uc = ut_ei4*sin_sg, dividing by the same sin_sg recovers ut_ei4.
    # Positions 1 and n-1 (C1/C2/C3 stencil) use the standard formula
    # (sw_core.F90:3596,3610): ut = (uc - v*cosa)*rsin_u. No override needed.
    for i_bdy in ([0, n] if n >= 2 else []):
        i_left = max(i_bdy - 1, 0)
        i_right = min(i_bdy, n - 1)
        sin_left = cdgrid.sin_sg[:, i_left, :, 2]
        sin_right = cdgrid.sin_sg[:, i_right, :, 0]
        sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
        ut = ut.at[:, i_bdy, :].set(
            uc[:, i_bdy, :] / jnp.maximum(sin_upwind, _EPS))

    # ---- Step 4b: A→C y-direction (covariant vtmp → vc) ----
    # vtmp_pad axis 2 has n+2h elements.  v-face j → padded j+h-1 and j+h.
    vc = 0.5 * (vtmp_pad[:, h:-h, h-1:n+h] + vtmp_pad[:, h:-h, h:n+h+1])  # (6, n, n+1)

    if n > 2 * npt + 2:
        vc_4th = (_A2 * (vtmp_pad[:, h:-h, h-2:n+h-1]
                         + vtmp_pad[:, h:-h, h+1:n+h+2])
                  + _A1 * (vtmp_pad[:, h:-h, h-1:n+h]
                           + vtmp_pad[:, h:-h, h:n+h+1]))
        j_lo = npt + 1
        j_hi = n - npt
        vc = vc.at[:, :, j_lo:j_hi].set(vc_4th[:, :, j_lo - 1:j_hi - 1])

    # One-sided c1/c2/c3 stencil at j=1, n-1
    if n > 3:
        vc = vc.at[:, :, 1].set(
            _C1 * vtmp_pad[:, h:-h, h + 2] + _C2 * vtmp_pad[:, h:-h, h + 1]
            + _C3 * vtmp_pad[:, h:-h, h])
        vc = vc.at[:, :, n - 1].set(
            _C1 * vtmp_pad[:, h:-h, n + h - 3] + _C2 * vtmp_pad[:, h:-h, n + h - 2]
            + _C3 * vtmp_pad[:, h:-h, n + h - 1])

    # AT face boundary (j=0, j=n): edge_interpolate4 on va (halo=2 straddles boundary)
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    for j_bdy in ([0, n] if n >= 2 else []):
        j_p = j_bdy + h
        va4 = jnp.stack([va_pad[:, h:-h, j_p - 1], va_pad[:, h:-h, j_p],
                         va_pad[:, h:-h, j_p + 1], va_pad[:, h:-h, j_p + 2]],
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

    vt = (vc - u_d * cdgrid.cosa_v) * cdgrid.rsin_v

    # Override only at edge_interpolate4 positions (j=0, n) to recover
    # edge_interpolate4 result: ut = uc/sin_sg.
    for j_bdy in [0, n]:
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
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    # 1. d2a2c_vect
    ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

    # 2. Scale transport velocities with dt/2 * edge_length * sin_sg_upwind
    # Use proper halo exchange for sin_sg (not mode='edge') so cross-face
    # upwinding is correct at panel boundaries (FV3 fv_grid_utils.F90:570).
    from legoesm.grids.halo import pad_halo
    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)
    sg = cdgrid.sin_sg
    grid = cdgrid.base

    # x-direction: ut → scaled flux (same pattern as compute_transport_quantities)
    sin_east = sg[:, :, :, 2]   # E-edge of each cell
    sin_west = sg[:, :, :, 0]   # W-edge of each cell
    se_pad = pad_halo(sin_east, interp_offsets=grid.halo_interp_offsets)
    sw_pad = pad_halo(sin_west, interp_offsets=grid.halo_interp_offsets)
    sin_upwind_x = jnp.where(ut > 0, se_pad[:, :n+1, 1:-1],
                                      sw_pad[:, 1:n+2, 1:-1])
    ut_scaled = dt2 * ut * dy * sin_upwind_x

    # y-direction: vt → scaled flux
    sin_north = sg[:, :, :, 3]  # N-edge of each cell
    sin_south = sg[:, :, :, 1]  # S-edge of each cell
    sn_pad = pad_halo(sin_north, interp_offsets=grid.halo_interp_offsets)
    ss_pad = pad_halo(sin_south, interp_offsets=grid.halo_interp_offsets)
    sin_upwind_y = jnp.where(vt > 0, sn_pad[:, 1:-1, :n+1],
                                      ss_pad[:, 1:-1, 1:n+2])
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

    # 4. KE at cell centres (FV3 sw_core.F90:303-372)
    # Duogrid/bounded: simple upwind uc/vc at all cells.
    # Non-duogrid: face-boundary cells use sin_sg/cos_sg conversion
    # to get the coordinate-parallel wind (sw_core.F90:323-365).
    uc_left = uc[:, :-1, :]   # (6, n, n)
    uc_right = uc[:, 1:, :]
    ke_u = jnp.where(ua > 0, uc_left, uc_right)

    vc_bot = vc[:, :, :-1]    # (6, n, n)
    vc_top = vc[:, :, 1:]
    ke_v = jnp.where(va > 0, vc_bot, vc_top)

    if not use_duogrid:
        sg = cdgrid.sin_sg
        cg = cdgrid.cos_sg
        # x-direction face-boundary KE conversion
        # Left edge (cell 0, ua > 0): uc*sin_sg(W) + v*cos_sg(W)
        ke_bdy_left = uc[:, 0, :] * sg[:, 0, :, 0] + v_d[:, 0, :] * cg[:, 0, :, 0]
        ke_u = ke_u.at[:, 0, :].set(
            jnp.where(ua[:, 0, :] > 0, ke_bdy_left, ke_u[:, 0, :]))
        # Right edge (cell n-1, ua <= 0): uc*sin_sg(E) + v*cos_sg(E)
        ke_bdy_right = (uc[:, n, :] * sg[:, n - 1, :, 2]
                        + v_d[:, n, :] * cg[:, n - 1, :, 2])
        ke_u = ke_u.at[:, n - 1, :].set(
            jnp.where(ua[:, n - 1, :] > 0, ke_u[:, n - 1, :], ke_bdy_right))
        # y-direction face-boundary KE conversion
        # Bottom edge (cell j=0, va > 0): vc*sin_sg(S) + u*cos_sg(S)
        ke_bdy_bot = vc[:, :, 0] * sg[:, :, 0, 1] + u_d[:, :, 0] * cg[:, :, 0, 1]
        ke_v = ke_v.at[:, :, 0].set(
            jnp.where(va[:, :, 0] > 0, ke_bdy_bot, ke_v[:, :, 0]))
        # Top edge (cell j=n-1, va <= 0): vc*sin_sg(N) + u*cos_sg(N)
        ke_bdy_top = (vc[:, :, n] * sg[:, :, n - 1, 3]
                      + u_d[:, :, n] * cg[:, :, n - 1, 3])
        ke_v = ke_v.at[:, :, n - 1].set(
            jnp.where(va[:, :, n - 1] > 0, ke_v[:, :, n - 1], ke_bdy_top))

    # ke = dt/4 * (ua * uc_upwind + va * vc_upwind)
    # NOTE: c_sw uses ONLY kinetic energy for the C-grid gradient, NOT
    # the full Bernoulli function (g*h is in d_sw only).  This is per
    # FV3 sw_core.F90 where ke has dt4 = 0.25*dt scaling.
    ke_total = dt2 * 0.5 * (ua * ke_u + va * ke_v)

    # 5. Vorticity at D-grid corners from C-grid covariant velocities.
    # Compute cell-centre vorticity from C-grid circulation (uses only
    # interior C-grid values), then halo-exchange and interpolate to corners.
    # This avoids edge-copy padding that gave zero vorticity at boundaries.
    fx_circ = uc * cdgrid.dxc    # (6, n+1, n) — FV3: fx = uc * dxc
    fy_circ = vc * cdgrid.dyc    # (6, n, n+1) — FV3: fy = vc * dyc

    # Cell-centre vorticity: circulation around each cell using interior values
    cell_vort = (fx_circ[:, :-1, :] - fx_circ[:, 1:, :]
                 + fy_circ[:, :, 1:] - fy_circ[:, :, :-1])
    rarea = 1.0 / cdgrid.base.area
    cell_vort = cell_vort * rarea + cdgrid.base.f

    # Halo exchange + interpolate to corners (4-point average)
    vort_abs = _interp_center_to_corner(cell_vort, cdgrid)

    # 6. Vorticity flux at C-grid face positions
    # FV3 sw_core.F90:416-423: c_sw vorticity flux uses /sina (1/sin),
    # NOT *rsin_u (1/sin²). The FV3 comment says: "we only divide by
    # sin instead of sin**2 in the interior". rsin_u is for d2a2c_vect only.
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_u**2, _EPS))
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_v**2, _EPS))
    fy1 = dt2 * (v_d - uc * cdgrid.cosa_u) / jnp.maximum(sina_u, _EPS)
    if not use_duogrid:
        # At face boundaries only (i=1,npx in FV3 → Python 0,n):
        # sin_sg cancellation → fy1 = dt2 * v_d (sw_core.F90:445-449).
        # Adjacent cells (1, n-1) use the standard formula.
        fy1 = fy1.at[:, 0, :].set(dt2 * v_d[:, 0, :])
        fy1 = fy1.at[:, n, :].set(dt2 * v_d[:, n, :])

    vort_x = jnp.where(fy1 > 0, vort_abs[:, :, :-1], vort_abs[:, :, 1:])

    # y-face: FV3 fx1 = dt2*(u - vc*cosa_v)/sina_v  (1/sin, not 1/sin²)
    fx1 = dt2 * (u_d - vc * cdgrid.cosa_v) / jnp.maximum(sina_v, _EPS)
    if not use_duogrid:
        # FV3 sw_core.F90:458-460: only at face boundaries (j=1, npy → Python 0, n).
        fx1 = fx1.at[:, :, 0].set(dt2 * u_d[:, :, 0])
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

    # 3. KE at cell centres (FV3 upwind formula, sw_core.F90:303-372)
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2
    uc_left = uc[:, :-1, :]
    uc_right = uc[:, 1:, :]
    ke_u = jnp.where(ua > 0, uc_left, uc_right)

    vc_bot = vc[:, :, :-1]
    vc_top = vc[:, :, 1:]
    ke_v = jnp.where(va > 0, vc_bot, vc_top)

    if not use_duogrid:
        sg = cdgrid.sin_sg
        cg = cdgrid.cos_sg
        ke_bdy_l = uc[:, 0, :] * sg[:, 0, :, 0] + v_d[:, 0, :] * cg[:, 0, :, 0]
        ke_u = ke_u.at[:, 0, :].set(
            jnp.where(ua[:, 0, :] > 0, ke_bdy_l, ke_u[:, 0, :]))
        ke_bdy_r = uc[:, n, :] * sg[:, n-1, :, 2] + v_d[:, n, :] * cg[:, n-1, :, 2]
        ke_u = ke_u.at[:, n-1, :].set(
            jnp.where(ua[:, n-1, :] > 0, ke_u[:, n-1, :], ke_bdy_r))
        ke_bdy_b = vc[:, :, 0] * sg[:, :, 0, 1] + u_d[:, :, 0] * cg[:, :, 0, 1]
        ke_v = ke_v.at[:, :, 0].set(
            jnp.where(va[:, :, 0] > 0, ke_bdy_b, ke_v[:, :, 0]))
        ke_bdy_t = vc[:, :, n] * sg[:, :, n-1, 3] + u_d[:, :, n] * cg[:, :, n-1, 3]
        ke_v = ke_v.at[:, :, n-1].set(
            jnp.where(va[:, :, n-1] > 0, ke_v[:, :, n-1], ke_bdy_t))

    if use_duogrid:
        # Duogrid: use D→A physical velocities (consistent frame) to avoid
        # the contravariant×covariant cross-product ua*uc which amplifies
        # interpolation mismatches at face boundaries by 1/sin².
        utmp_ke = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])   # (6, n, n)
        vtmp_ke = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
        ke = 0.5 * (utmp_ke**2 + vtmp_ke**2)
    else:
        # Always use physical-frame KE for the Bernoulli function.
        # FV3's contravariant formula (ua*ke_u + va*ke_v) has 1700x worse
        # balance at face boundaries due to the 1/sin² metric amplification,
        # causing the CSW path to be unstable. Physical-frame KE gives the
        # same value but with much better geostrophic balance.
        utmp_ke = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
        vtmp_ke = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
        ke = 0.5 * (utmp_ke**2 + vtmp_ke**2)
    B = ke + g * (h + h_s)

    # 4. Bernoulli gradient at C-grid face positions (2-point difference)
    B_pad = _pad_halo_auto(B, cdgrid)
    dB_x = cdgrid.rdxc * (B_pad[:, :-1, 1:-1] - B_pad[:, 1:, 1:-1])  # (6, n+1, n)
    dB_y = cdgrid.rdyc * (B_pad[:, 1:-1, :-1] - B_pad[:, 1:-1, 1:])  # (6, n, n+1)

    # 5. Vorticity at D-grid corners from C-grid covariant velocities.
    # Compute CELL-CENTRE vorticity from C-grid circulation (uses only
    # interior C-grid values, no halo needed), then scalar halo exchange
    # for boundary, then interpolate to corners. This avoids the edge-copy
    # padding that caused zero vorticity at face-boundary columns and the
    # resulting linear instability.
    dg = cdgrid.base.duogrid
    use_duogrid = dg is not None and dg.ng >= 2

    fx_circ = uc * cdgrid.dxc   # (6, n+1, n)
    fy_circ = vc * cdgrid.dyc   # (6, n, n+1)

    # Cell-centre vorticity: circulation around cell (i,j)
    # Uses uc at faces i and i+1 (rows), vc at faces j and j+1 (cols)
    # All values are interior C-grid positions — no halo needed.
    cell_vort = (fx_circ[:, :-1, :] - fx_circ[:, 1:, :]
                 + fy_circ[:, :, 1:] - fy_circ[:, :, :-1])
    rarea = 1.0 / cdgrid.base.area
    cell_vort = cell_vort * rarea

    # Add planetary vorticity at cell centres
    cell_vort_abs = cell_vort + cdgrid.base.f

    # Halo exchange and interpolate to corners (4-point average)
    vort_abs = _interp_center_to_corner(cell_vort_abs, cdgrid)

    # 6. Vorticity flux at C-grid face positions
    # FV3 sw_core.F90:416-423: c_sw uses /sina (1/sin), NOT *rsin_u (1/sin²)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_u**2, _EPS))
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cdgrid.cosa_v**2, _EPS))
    fy1 = (v_d - uc * cdgrid.cosa_u) / jnp.maximum(sina_u, _EPS)
    if not use_duogrid:
        # FV3 sw_core.F90:445-449: override only at face boundaries (0, n).
        fy1 = fy1.at[:, 0, :].set(v_d[:, 0, :])
        fy1 = fy1.at[:, n, :].set(v_d[:, n, :])
    vort_x = jnp.where(fy1 > 0, vort_abs[:, :, :-1], vort_abs[:, :, 1:])

    # FV3: fx1 = (u - vc*cosa_v)/sina_v
    fx1 = (u_d - vc * cdgrid.cosa_v) / jnp.maximum(sina_v, _EPS)
    if not use_duogrid:
        # FV3 sw_core.F90:458-460: only at face boundaries (0, n).
        fx1 = fx1.at[:, :, 0].set(u_d[:, :, 0])
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

    # 9. Project TOTAL C-grid tendency → D-grid edge midpoints via
    # halo-exchanged cell-centre averaging. The previous edge-copy padding
    # created a linear instability at face corners (blowup at ~2h).
    # Now: C-grid → cell-centre average → vector halo exchange → D-grid.
    from legoesm.grids.halo import pad_halo_vector
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    # Step 1: average C-grid to cell centres
    duc_cc = 0.5 * (duc[:, :-1, :] + duc[:, 1:, :])   # (6, n, n)
    dvc_cc = 0.5 * (dvc[:, :, :-1] + dvc[:, :, 1:])   # (6, n, n)
    # Step 2: vector halo exchange (rotates tendencies across face boundaries)
    duc_pad, dvc_pad = pad_halo_vector(
        duc_cc, dvc_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    # Step 3: average haloed cell centres to D-grid edge midpoints
    du_dt = 0.5 * (duc_pad[:, 1:-1, :-1] + duc_pad[:, 1:-1, 1:])   # (6, n, n+1)
    dv_dt = 0.5 * (dvc_pad[:, :-1, 1:-1] + dvc_pad[:, 1:, 1:-1])   # (6, n+1, n)

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
    # must be the PHYSICAL face-normal velocity = (uc - v*cosa)/sin, NOT
    # the contravariant velocity (rsin_u = 1/sin²).
    # This is a hybrid (non-FV3) path — the FV3 c_sw path uses ut*sin_sg*dy.
    cosa_u = cdgrid.cosa_u     # (6, n+1, n)
    cosa_v = cdgrid.cosa_v     # (6, n, n+1)
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, _EPS))
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v**2, _EPS))

    # v_d (6, n+1, n) is co-located with cosa_u — use directly as the
    # cross-velocity at u-face positions (same approximation as _uc_to_ut).
    v_at_u = v_d  # (6, n+1, n)

    # u_d (6, n, n+1) is co-located with cosa_v — use directly.
    u_at_v = u_d  # (6, n, n+1)

    uc_phys = (uc_new - v_at_u * cosa_u) / jnp.maximum(sina_u, _EPS)
    vc_phys = (vc_new - u_at_v * cosa_v) / jnp.maximum(sina_v, _EPS)

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
    # Phase 1: C-grid half-step (c_sw) — mass transport + velocity update
    h_star, uc_new, vc_new, ua, va = _c_sw(
        h, u_d, v_d, h_s, cdgrid, dt, g)

    # Phase 2: Backward pressure gradient at C-grid (p_grad_c)
    dt2 = 0.5 * dt
    dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
    uc_new = uc_new + dp_x
    vc_new = vc_new + dp_y

    # Phase 3: D-grid half-step using FV3-native transport operators
    # (NOT Arakawa-Lamb — uses PPM mass transport + KE/vort transport)
    h_new, u_d_new, v_d_new = _d_sw_native(
        h, u_d, v_d, h_s, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp)

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

    # At edge_interpolate4 positions (i=1, n-1): ut = uc / sin_sg recovers
    # the edge_interpolate4 result.  Face boundaries (i=0, n) keep the
    # standard formula with cross-velocity (FV3 sw_core.F90:3595-3596).
    sg = cdgrid.sin_sg
    for i_bdy in [1, n - 1]:
        i_left = max(i_bdy - 1, 0)
        i_right = min(i_bdy, n - 1)
        sin_left = sg[:, i_left, :, 2]
        sin_right = sg[:, i_right, :, 0]
        sin_upwind = jnp.where(uc[:, i_bdy, :] > 0, sin_left, sin_right)
        ut = ut.at[:, i_bdy, :].set(
            uc[:, i_bdy, :] / jnp.maximum(sin_upwind, _EPS))

    for j_bdy in [1, n - 1]:
        j_below = max(j_bdy - 1, 0)
        j_above = min(j_bdy, n - 1)
        sin_below = sg[:, :, j_below, 3]
        sin_above = sg[:, :, j_above, 1]
        sin_upwind = jnp.where(vc[:, :, j_bdy] > 0, sin_below, sin_above)
        vt = vt.at[:, :, j_bdy].set(
            vc[:, :, j_bdy] / jnp.maximum(sin_upwind, _EPS))

    return ut, vt


def _d_sw_native(h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, dt, g,
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
    # Use FV3 d_sw1 boundary handling (adjacent strips + corner 2×2 solve)
    # for cross-face consistent transport at panel boundaries.
    ut, vt = _d_sw1_recompute_ut_vt(uc, vc, cdgrid, dt)

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

    # === 5. KE at corners (NO g*h — pressure gradient already in uc/vc via p_grad_c) ===
    # NOTE: This is a GRADIENT, not FV3's d_sw3 B-grid KE TRANSPORT.
    # Full FV3 fidelity requires porting ytp_v/xtp_u staggered transport.
    B_corner = _interp_center_to_corner(ke_cell, cdgrid)  # (6, n+1, n+1)

    # === 6. Divergence damping at corners (optional) ===
    if div_damp > 0:
        div_field = cgrid_divergence(uc, vc, cdgrid)
        area_min = float(jnp.min(cdgrid.base.area))
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs_corner = _interp_center_to_corner(jnp.abs(div_field), cdgrid)
        damp_coeff = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        div_corner = _interp_center_to_corner(div_field, cdgrid)
        B_corner = B_corner + damp_coeff * div_corner

    # === 7. Bernoulli gradient at D-grid edges ===
    ke_diff_u = B_corner[:, :-1, :] - B_corner[:, 1:, :]  # (6, n, n+1)
    ke_diff_v = B_corner[:, :, :-1] - B_corner[:, :, 1:]  # (6, n+1, n)
    ke_diff_u_scaled = dt * ke_diff_u
    ke_diff_v_scaled = dt * ke_diff_v

    # === 8. Vorticity transport to D-grid edges via fv_tp_2d ===
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx_vort, fy_vort = fv_tp_2d(
        zeta_abs, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    # fx_vort: (6, n+1, n) — vorticity flux at x-interfaces (v_d positions)
    # fy_vort: (6, n, n+1) — vorticity flux at y-interfaces (u_d positions)

    # === 9. D-grid wind update (incremental) ===
    # FV3's d_sw6 uses a REPLACEMENT formula (u = vt + ke + fy) but that
    # requires covariant convention. Our D-grid winds are in the geographic
    # rotation convention, so we use incremental updates.
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
        h, u_d, v_d, h_s, uc_new, vc_new, ua, va, cdgrid, dt, g,
        div_damp=div_damp)

    return h_new, u_d_new, v_d_new
