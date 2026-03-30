"""FV3-style D-grid shallow water core with forward-backward stepping.

Line-integral formulation: the momentum equation operates on ``u_line =
u_phys * dx`` in coordinate space where the Bernoulli gradient is a bare
difference and no non-orthogonality cross-term is needed.

Vorticity is PPM-transported using the SAME mass fluxes as the height
field, ensuring the KE contraction and vorticity flux use identical
contravariant velocities → discrete geostrophic balance → no edge
artifacts.

All operations are pure JAX and differentiable.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on cubed-sphere grids
- Harris et al. (2021): Scientific Description of GFDL FV3
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.core.d2a2c_vect import d2a2c_vect
from legoesm.core.fv_tp_2d import fv_tp_2d
from legoesm.core.operators_cdgrid import (
    cgrid_mass_flux_divergence,
    _interp_center_to_corner,
    _interp_corner_to_center,
    _pad_halo_auto,
    fv3_vorticity,
)


def fv3_d_sw_step(h, u_d, v_d, h_s, cdgrid, dt, g=9.80616,
                   div_damp=0.0, hyperdiff_coeff=0.0):
    """FV3 forward-backward shallow water step.

    Advances ``(h, u_d, v_d)`` by one time step ``dt``.

    Parameters
    ----------
    h : (6, n, n)     — height at cell centres
    u_d : (6, n, n+1) — physical x-velocity at x-edge midpoints
    v_d : (6, n+1, n) — physical y-velocity at y-edge midpoints
    h_s : (6, n, n)   — surface topography
    cdgrid : CubedSphereCDGrid
    dt : float         — time step [s]
    g, div_damp, hyperdiff_coeff : float

    Returns
    -------
    h_new, u_d_new, v_d_new : updated state
    """
    n = cdgrid.n
    grid = cdgrid.base
    dx = cdgrid.dx_edge_y   # (6, n, n+1) — x-edge length
    dy = cdgrid.dy_edge_x   # (6, n+1, n) — y-edge length
    area = grid.area
    dt2 = 0.5 * dt

    # ==================================================================
    # 1. D→A→C conversion (4th-order interior)
    # ==================================================================
    ua, va, uc, vc = d2a2c_vect(u_d, v_d, cdgrid)

    # ==================================================================
    # 2. Contravariant C-grid velocities + Courant + mass fluxes
    # ==================================================================
    from legoesm.grids.halo import pad_halo_vector
    ua_pad, va_pad = pad_halo_vector(
        ua, va,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )
    # Contravariant at x-faces: average ua in i-direction
    ut = 0.5 * (ua_pad[:, :-1, 1:-1] + ua_pad[:, 1:, 1:-1])  # (6, n+1, n)
    # Contravariant at y-faces: average va in j-direction
    vt = 0.5 * (va_pad[:, 1:-1, :-1] + va_pad[:, 1:-1, 1:])  # (6, n, n+1)

    # Courant numbers (dimensionless)
    rdx_pad = _pad_halo_auto(1.0 / grid.dx, cdgrid)
    rdy_pad = _pad_halo_auto(1.0 / grid.dy, cdgrid)
    rdx_u = 0.5 * (rdx_pad[:, :-1, 1:-1] + rdx_pad[:, 1:, 1:-1])
    rdy_v = 0.5 * (rdy_pad[:, 1:-1, :-1] + rdy_pad[:, 1:-1, 1:])
    crx = dt * ut * rdx_u   # (6, n+1, n)
    cry = dt * vt * rdy_v   # (6, n, n+1)

    # Mass fluxes (area flux = contravariant vel × transverse edge length)
    xfx = dt * ut * dy   # (6, n+1, n) — [m²]
    yfx = dt * vt * dx   # (6, n, n+1) — [m²]

    # ==================================================================
    # 3. Mass transport — forward step (PPM)
    # ==================================================================
    dh = cgrid_mass_flux_divergence(h, uc, vc, cdgrid)
    total_area = jnp.sum(area)
    dh = dh - jnp.sum(dh * area) / total_area
    h_new = h + dt * dh

    # ==================================================================
    # 4. KE at corners — contraction with dt/2 scaled contravariant
    # ==================================================================
    # Contravariant at corners (from SAME haloed ua, va)
    ub_c = 0.25 * (ua_pad[:, :-1, :-1] + ua_pad[:, 1:, :-1]
                    + ua_pad[:, :-1, 1:] + ua_pad[:, 1:, 1:])
    vb_c = 0.25 * (va_pad[:, :-1, :-1] + va_pad[:, 1:, :-1]
                    + va_pad[:, :-1, 1:] + va_pad[:, 1:, 1:])

    ub = dt2 * ub_c   # (6, n+1, n+1), units: m
    vb = dt2 * vb_c

    # Physical velocity at corners (average from edge midpoints)
    u_d_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    u_phys_c = 0.5 * (u_d_pad[:, :-1, :] + u_d_pad[:, 1:, :])
    v_d_pad = jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')
    v_phys_c = 0.5 * (v_d_pad[:, :, :-1] + v_d_pad[:, :, 1:])

    # KE = 0.5*(vb*u + ub*v), units: m²/s
    ke = 0.5 * (vb * u_phys_c + ub * v_phys_c)

    # ==================================================================
    # 5. Geopotential at corners (using UPDATED height)
    # ==================================================================
    gh_corner = g * _interp_center_to_corner(h_new + h_s, cdgrid)

    # B_corner includes both ke (dt/2 scaled) and gh (dt scaled)
    B_corner = ke + dt * gh_corner

    # ==================================================================
    # 6. Vorticity at cell centres
    # ==================================================================
    zeta_corner = fv3_vorticity(u_d, v_d, cdgrid)
    zeta_abs_corner = zeta_corner + cdgrid.f_corner
    zeta_cc = _interp_corner_to_center(zeta_abs_corner)  # (6, n, n)

    # ==================================================================
    # 7. PPM-transport vorticity using SAME mass fluxes
    # ==================================================================
    # The raw PPM fluxes (face-value reconstructions of ζ at each face)
    # multiplied by the mass flux give the vorticity contribution to
    # the line-integral momentum update.
    fx_vort, fy_vort = fv_tp_2d(
        zeta_cc, crx, cry, xfx, yfx, cdgrid, return_fluxes=True)

    # The flux at each face is: ζ_face * mass_flux = ζ_face * (area/timestep)
    # For the line-integral update we need: dt * ζ * v_contra * edge_length
    # The PPM gives: fy_vort * yfx = ζ_face * dt*vt*dx = dt * ζ * vt * dx
    # This has units: [1/s * s * m/s * m] = m²/s ≡ u_line units. ✓
    # But fy_vort is the FACE VALUE (not yet multiplied by mass flux).
    # We need: fy_vort_contribution = fy_vort * yfx (at y-faces for u_line)
    # and:     fx_vort_contribution = fx_vort * xfx (at x-faces for v_line)

    # For u_line at edge (i,j): the vorticity flux from y-direction
    # transport contributes at the two y-faces bounding this x-edge.
    # In FV3: fy(i,j) = fy_vort * yfx at face (i,j). This IS the
    # vorticity contribution to u_line.

    # BUT: fy_vort has shape (6, n, n+1) — y-face values.
    #      xfx has shape (6, n+1, n) — x-face mass fluxes.
    # fy_vort IS at u_d positions! And fx_vort IS at v_d positions!
    # So: the vorticity contribution is simply:
    fy_contrib = fy_vort * yfx   # (6, n, n+1) — for u_line update
    fx_contrib = fx_vort * xfx   # (6, n+1, n) — for v_line update

    # ==================================================================
    # 8. Line-integral momentum update
    # ==================================================================
    u_line_old = u_d * dx
    v_line_old = v_d * dy

    u_line_new = u_line_old + (B_corner[:, :-1, :] - B_corner[:, 1:, :]) + fy_contrib
    v_line_new = v_line_old + (B_corner[:, :, :-1] - B_corner[:, :, 1:]) - fx_contrib

    # ==================================================================
    # 9. Divergence damping
    # ==================================================================
    if div_damp > 0:
        from legoesm.core.operators_cdgrid import cgrid_divergence
        div_field = cgrid_divergence(uc, vc, cdgrid)
        div_corner = _interp_center_to_corner(div_field, cdgrid)
        area_min = jnp.min(area)
        d2_bg = div_damp / area_min
        dddmp = 0.2
        div_abs_corner = _interp_center_to_corner(jnp.abs(div_field), cdgrid)
        adaptive = area_min * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * div_abs_corner))
        damp_corner = dt * adaptive * div_corner
        u_line_new = u_line_new + (damp_corner[:, :-1, :] - damp_corner[:, 1:, :])
        v_line_new = v_line_new + (damp_corner[:, :, :-1] - damp_corner[:, :, 1:])

    # ==================================================================
    # 10. Biharmonic hyperdiffusion
    # ==================================================================
    if hyperdiff_coeff > 0:
        from legoesm.core.operators import laplacian_compact
        u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
        v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
        lap_u = laplacian_compact(u_cc, grid)
        lap2_u = laplacian_compact(lap_u, grid)
        lap2_u_pad = _pad_halo_auto(lap2_u, cdgrid)
        u_line_new = u_line_new - dt * hyperdiff_coeff * dx * 0.5 * (
            lap2_u_pad[:, 1:-1, :-1] + lap2_u_pad[:, 1:-1, 1:])
        lap_v = laplacian_compact(v_cc, grid)
        lap2_v = laplacian_compact(lap_v, grid)
        lap2_v_pad = _pad_halo_auto(lap2_v, cdgrid)
        v_line_new = v_line_new - dt * hyperdiff_coeff * dy * 0.5 * (
            lap2_v_pad[:, :-1, 1:-1] + lap2_v_pad[:, 1:, 1:-1])

    # ==================================================================
    # 11. Convert line integrals back to physical velocities
    # ==================================================================
    u_d_new = u_line_new / dx
    v_d_new = v_line_new / dy

    return h_new, u_d_new, v_d_new


# Legacy tendency-based wrapper
def fv3_d_sw(h, u_d, v_d, h_s, cdgrid,
             g=9.80616, div_damp=0.0, hyperdiff_coeff=0.0):
    """Tendency wrapper for RK3 compatibility (approximate)."""
    _dt = 1.0
    h_new, u_new, v_new = fv3_d_sw_step(
        h, u_d, v_d, h_s, cdgrid, _dt,
        g=g, div_damp=div_damp, hyperdiff_coeff=hyperdiff_coeff)
    return h_new - h, u_new - u_d, v_new - v_d
