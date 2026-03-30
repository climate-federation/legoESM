"""FV3-style D-grid shallow water core with forward-backward stepping.

True covariant velocity formulation: the momentum equation operates in
**coordinate space** where the Bernoulli gradient is a bare difference
and no non-orthogonality cross-term is needed.  Internally converts
between physical (stored) and covariant (computed) velocities.

Forward-backward split: mass is advanced first, then the new mass is
used for the momentum update.  KE at corners uses the same contravariant
velocities as the vorticity flux, ensuring discrete geostrophic balance.

This is a FULL-STEP function (advances state by dt), not a tendency.

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
from legoesm.core.operators_cdgrid import (
    cgrid_mass_flux_divergence,
    _interp_center_to_corner,
    _pad_halo_auto,
    fv3_vorticity,
)

_EPS = 1e-30


def fv3_d_sw_step(h, u_d, v_d, h_s, cdgrid, dt, g=9.80616,
                   div_damp=0.0, hyperdiff_coeff=0.0):
    """FV3 forward-backward shallow water step.

    Advances (h, u_d, v_d) by one time step ``dt``.

    The momentum equation is in **coordinate space** (covariant):

        u_line_new = u_line_old + dt * [-dB + vort_flux]

    where ``u_line = u_phys * dx`` is the line integral.
    This avoids the non-orthogonality cross-term entirely.

    Parameters
    ----------
    h : (6, n, n)     — height at cell centres
    u_d : (6, n, n+1) — physical x-velocity at x-edge midpoints
    v_d : (6, n+1, n) — physical y-velocity at y-edge midpoints
    h_s : (6, n, n)   — surface topography
    cdgrid : CubedSphereCDGrid
    dt : float         — time step [s]
    g : float          — gravity
    div_damp, hyperdiff_coeff : float

    Returns
    -------
    h_new, u_d_new, v_d_new : updated state (same shapes)
    """
    n = cdgrid.n
    grid = cdgrid.base
    dx = cdgrid.dx_edge_y   # (6, n, n+1) — x-edge physical length
    dy = cdgrid.dy_edge_x   # (6, n+1, n) — y-edge physical length
    area = grid.area         # (6, n, n)
    dt2 = 0.5 * dt

    # ==================================================================
    # Step 1: D→A→C conversion (4th-order interior)
    # ==================================================================
    ua, va, uc, vc = d2a2c_vect(u_d, v_d, cdgrid)

    # ==================================================================
    # Step 2: Mass transport — forward step
    # ==================================================================
    dh = cgrid_mass_flux_divergence(h, uc, vc, cdgrid)
    total_area = jnp.sum(area)
    dh = dh - jnp.sum(dh * area) / total_area
    h_new = h + dt * dh

    # ==================================================================
    # Step 3: Contravariant transport velocities at corners
    # ==================================================================
    # Use haloed cell-centre contravariant from d2a2c_vect.
    from legoesm.grids.halo import pad_halo_vector
    ua_pad, va_pad = pad_halo_vector(
        ua, va,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )
    # Average contravariant to corners — used for BOTH KE and vort flux
    ub_c = 0.25 * (ua_pad[:, :-1, :-1] + ua_pad[:, 1:, :-1]
                    + ua_pad[:, :-1, 1:] + ua_pad[:, 1:, 1:])  # (6, n+1, n+1)
    vb_c = 0.25 * (va_pad[:, :-1, :-1] + va_pad[:, 1:, :-1]
                    + va_pad[:, :-1, 1:] + va_pad[:, 1:, 1:])  # (6, n+1, n+1)

    # Scale by dt/2 for KE (following FV3 convention)
    ub = dt2 * ub_c
    vb = dt2 * vb_c

    # ==================================================================
    # Step 4: KE at corners — contraction of physical vel × contravariant
    # ==================================================================
    # FV3 KE: ke = 0.5*(vb * u_phys + ub * v_phys) where vb = dt/2*v_contra.
    # u_phys at corners: average u_d (physical velocity, NOT line integral)
    u_d_pad = jnp.pad(u_d, [(0, 0), (1, 1), (0, 0)], mode='edge')
    u_c = 0.5 * (u_d_pad[:, :-1, :] + u_d_pad[:, 1:, :])      # (6, n+1, n+1)
    v_d_pad = jnp.pad(v_d, [(0, 0), (0, 0), (1, 1)], mode='edge')
    v_c = 0.5 * (v_d_pad[:, :, :-1] + v_d_pad[:, :, 1:])       # (6, n+1, n+1)

    # ke has units [m²/s]: dt/2 * m/s * m/s = m²/s.
    ke = 0.5 * (vb * u_c + ub * v_c)

    # ==================================================================
    # Step 5: Geopotential height at corners (using UPDATED height)
    # ==================================================================
    gh_corner = g * _interp_center_to_corner(h_new + h_s, cdgrid)

    # ==================================================================
    # Step 6: Vorticity at corners — exact Stokes circulation
    # ==================================================================
    zeta = fv3_vorticity(u_d, v_d, cdgrid)     # (6, n+1, n+1)
    zeta_abs = zeta + cdgrid.f_corner

    # ==================================================================
    # Step 7: Vorticity flux at edges using the SAME ub_c/vb_c
    # ==================================================================
    # Average absolute vorticity to edges
    zeta_u = 0.5 * (zeta_abs[:, :-1, :] + zeta_abs[:, 1:, :])  # (6, n, n+1)
    zeta_v = 0.5 * (zeta_abs[:, :, :-1] + zeta_abs[:, :, 1:])  # (6, n+1, n)

    # Perpendicular contravariant at edges (from SAME corner data)
    vb_at_u = 0.5 * (vb_c[:, :-1, :] + vb_c[:, 1:, :])   # (6, n, n+1)
    ub_at_v = 0.5 * (ub_c[:, :, :-1] + ub_c[:, :, 1:])    # (6, n+1, n)

    # Vorticity flux in line-integral units [m²/s]:
    # fy = dt * ζ * v_contra * dx_edge (perpendicular flux × edge length)
    fy_u = dt * zeta_u * vb_at_u * dx                        # (6, n, n+1)
    fx_v = dt * zeta_v * ub_at_v * dy                        # (6, n+1, n)

    # ==================================================================
    # Step 8: Momentum update in COORDINATE SPACE (line integrals)
    # ==================================================================
    # u_line_new = u_line_old + [ke(i,j) - ke(i+1,j)] + fy
    #   where ke already includes dt/2 factor
    # The ke difference: ke at corner (i,j) minus ke at corner (i+1,j)
    #   for u_d at x-edge between these corners.

    u_line_old = u_d * dx   # (6, n, n+1)
    v_line_old = v_d * dy   # (6, n+1, n)

    # Bernoulli gradient in coordinate space (includes gh + ke):
    # B_corner = ke + dt * gh_corner  (ke already has dt/2 from step 4)
    B_corner = ke + dt * gh_corner

    u_line_new = u_line_old + (B_corner[:, :-1, :] - B_corner[:, 1:, :]) + fy_u
    v_line_new = v_line_old + (B_corner[:, :, :-1] - B_corner[:, :, 1:]) - fx_v

    # ==================================================================
    # Step 9: Divergence damping
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
        # Add damping to ke (modifies the B gradient)
        damp_corner = dt * adaptive * div_corner
        u_line_new = u_line_new + (damp_corner[:, :-1, :] - damp_corner[:, 1:, :])
        v_line_new = v_line_new + (damp_corner[:, :, :-1] - damp_corner[:, :, 1:])

    # ==================================================================
    # Step 10: Biharmonic hyperdiffusion
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
    # Step 11: Convert line integrals back to physical velocities
    # ==================================================================
    u_d_new = u_line_new / dx
    v_d_new = v_line_new / dy

    return h_new, u_d_new, v_d_new


# Legacy tendency-based wrapper for RK3 compatibility
def fv3_d_sw(h, u_d, v_d, h_s, cdgrid,
             g=9.80616, div_damp=0.0, hyperdiff_coeff=0.0):
    """Tendency wrapper around :func:`fv3_d_sw_step` for RK3 integrators.

    NOTE: This is an APPROXIMATION — the forward-backward split in
    ``fv3_d_sw_step`` uses the updated height for the momentum step,
    which cannot be exactly represented as a tendency.  For proper
    FV3 behavior, use ``fv3_d_sw_step`` directly.
    """
    # Use a tiny dt for tendency estimation
    # The caller's RK3 will apply its own dt scaling
    _dt = 1.0  # unit time step — tendencies are per second
    h_new, u_new, v_new = fv3_d_sw_step(
        h, u_d, v_d, h_s, cdgrid, _dt,
        g=g, div_damp=div_damp, hyperdiff_coeff=hyperdiff_coeff)
    return h_new - h, u_new - u_d, v_new - v_d
