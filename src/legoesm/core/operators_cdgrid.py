"""C-D grid operators on the cubed-sphere.

Provides the core operators for the FV3-style C-D grid discretisation,
unified for both 2D (shallow water) and 3D (atmosphere PE / ocean PE):

* D-grid vorticity (at cell centres from corner winds via circulation)
* D-grid to C-grid wind interpolation
* C-grid divergence and mass flux
* Vector-invariant momentum tendencies with Arakawa-Lamb gradient
* Laplacian and biharmonic diffusion on the D-grid

All operators automatically handle both 2D (shape ``(6, n+1, n+1)``)
and 3D (shape ``(6, n+1, n+1, nlev)``) inputs.

D-grid winds are prognostic (at cell corners).
C-grid winds are diagnostic (at cell edges).

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo
from legoesm.core.operators_fv import _ppm_reconstruct_x, _ppm_reconstruct_y


# ==============================================================================
# Internal: halo padding that works for both 2D and 3D
# ==============================================================================

def _pad_halo_auto(field, cdgrid):
    """Pad halo for a 2D or 3D cell-centre field.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n) or (6, n, n, nlev)

    Returns
    -------
    jax.Array, shape (6, n+2, n+2) or (6, n+2, n+2, nlev)
    """
    if field.ndim == 3:
        return pad_halo(field, interp_offsets=cdgrid.base.halo_interp_offsets)
    # 3D: vmap over levels
    f_t = jnp.moveaxis(field, -1, 0)

    def pad_one(fk):
        return pad_halo(fk, interp_offsets=cdgrid.base.halo_interp_offsets)

    f_pad_t = jax.vmap(pad_one)(f_t)
    return jnp.moveaxis(f_pad_t, 0, -1)


def _pad_halo_auto_h2(field, cdgrid):
    """Pad halo=2 for a 2D or 3D cell-centre field.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n) or (6, n, n, nlev)

    Returns
    -------
    jax.Array, shape (6, n+4, n+4) or (6, n+4, n+4, nlev)
    """
    if field.ndim == 3:
        return pad_halo(field, halo=2,
                        interp_offsets=cdgrid.base.halo_interp_offsets_h2)
    f_t = jnp.moveaxis(field, -1, 0)

    def pad_one(fk):
        return pad_halo(fk, halo=2,
                        interp_offsets=cdgrid.base.halo_interp_offsets_h2)

    f_pad_t = jax.vmap(pad_one)(f_t)
    return jnp.moveaxis(f_pad_t, 0, -1)


def _broadcast_metric(metric, field):
    """Broadcast a 2D metric (6, ...) to match field's trailing nlev dim."""
    if field.ndim > metric.ndim:
        return metric[..., None]
    return metric


# ==============================================================================
# D-grid to C-grid interpolation
# ==============================================================================

def dgrid_to_cgrid(u_d, v_d, cdgrid):
    """Interpolate D-grid corner winds to C-grid edge-normal velocities.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.

    Parameters
    ----------
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])
    """
    u_c = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_c = 0.5 * (v_d[:, :-1] + v_d[:, 1:])
    return u_c, v_c


def cgrid_to_dgrid(u_c, v_c, cdgrid):
    """Interpolate C-grid edge velocities to D-grid corners.

    Works for both 2D and 3D inputs.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])

    Returns
    -------
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    pad_u = [(0, 0), (0, 0), (1, 1)] + [(0, 0)] * (u_c.ndim - 3)
    u_c_pad = jnp.pad(u_c, pad_u, mode='edge')
    u_d = 0.5 * (u_c_pad[:, :, :-1] + u_c_pad[:, :, 1:])

    pad_v = [(0, 0), (1, 1), (0, 0)] + [(0, 0)] * (v_c.ndim - 3)
    v_c_pad = jnp.pad(v_c, pad_v, mode='edge')
    v_d = 0.5 * (v_c_pad[:, :-1] + v_c_pad[:, 1:])
    return u_d, v_d


# ==============================================================================
# D-grid vorticity (circulation form)
# ==============================================================================

def dgrid_vorticity(u_d, v_d, cdgrid):
    """Relative vorticity at cell centres from D-grid corner winds.

    Uses the integral circulation form: zeta = (1/A) oint v . dl, which is
    exact for the D-grid and avoids the Hollingsworth-Kallberg instability.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.

    Parameters
    ----------
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    zeta : jax.Array, shape (6, n, n[, nlev])
    """
    u_south = 0.5 * (u_d[:, :-1, :-1] + u_d[:, 1:, :-1])
    u_north = 0.5 * (u_d[:, :-1, 1:] + u_d[:, 1:, 1:])
    v_east = 0.5 * (v_d[:, 1:, :-1] + v_d[:, 1:, 1:])
    v_west = 0.5 * (v_d[:, :-1, :-1] + v_d[:, :-1, 1:])

    dx_south = _broadcast_metric(cdgrid.dx_edge_y[:, :, :-1], u_d)
    dx_north = _broadcast_metric(cdgrid.dx_edge_y[:, :, 1:], u_d)
    dy_west = _broadcast_metric(cdgrid.dy_edge_x[:, :-1, :], u_d)
    dy_east = _broadcast_metric(cdgrid.dy_edge_x[:, 1:, :], u_d)

    circ = (u_south * dx_south + v_east * dy_east
            - u_north * dx_north - v_west * dy_west)

    area = _broadcast_metric(cdgrid.base.area, u_d)
    return circ / area


# ==============================================================================
# C-grid divergence
# ==============================================================================

def cgrid_divergence(u_c, v_c, cdgrid):
    """Exact flux-form divergence at cell centres.

    Works for both 2D and 3D inputs.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])

    Returns
    -------
    div : jax.Array, shape (6, n, n[, nlev])
    """
    dy = _broadcast_metric(cdgrid.dy_edge_x, u_c)
    dx = _broadcast_metric(cdgrid.dx_edge_y, v_c)
    flux_x = u_c * dy
    flux_y = v_c * dx
    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    area = _broadcast_metric(cdgrid.base.area, net_x)
    return (net_x + net_y) / area


# ==============================================================================
# C-grid mass flux
# ==============================================================================

def cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid):
    """Conservative mass flux divergence with PPM face values.

    Uses PPM (Piecewise Parabolic Method) reconstruction with
    Colella-Woodward monotonicity limiting for high-order face
    states, then applies upwind selection based on C-grid velocity.

    Works for both 2D and 3D inputs.

    Parameters
    ----------
    h : jax.Array, shape (6, n, n[, nlev])
    u_c : jax.Array, shape (6, n+1, n[, nlev])
    v_c : jax.Array, shape (6, n, n+1[, nlev])

    Returns
    -------
    dh_dt : jax.Array, shape (6, n, n[, nlev])
    """
    if h.ndim == 4:
        # 3D: apply PPM per level via vmap
        h_t = jnp.moveaxis(h, -1, 0)       # (nlev, 6, n, n)
        u_c_t = jnp.moveaxis(u_c, -1, 0)   # (nlev, 6, n+1, n)
        v_c_t = jnp.moveaxis(v_c, -1, 0)   # (nlev, 6, n, n+1)

        def flux_div_one(args):
            hk, uk, vk = args
            return cgrid_mass_flux_divergence(hk, uk, vk, cdgrid)

        result_t = jax.vmap(flux_div_one)((h_t, u_c_t, v_c_t))
        return jnp.moveaxis(result_t, 0, -1)

    # 2D case: PPM reconstruction
    h_pad_h2 = _pad_halo_auto_h2(h, cdgrid)  # (6, n+4, n+4)

    # PPM left/right states at x-interfaces: (6, n+1, n)
    h_L_x, h_R_x = _ppm_reconstruct_x(h_pad_h2)
    # PPM left/right states at y-interfaces: (6, n, n+1)
    h_L_y, h_R_y = _ppm_reconstruct_y(h_pad_h2)

    # Upwind selection
    h_face_x = jnp.where(u_c > 0, h_L_x, h_R_x)
    h_face_y = jnp.where(v_c > 0, h_L_y, h_R_y)

    dy = cdgrid.dy_edge_x   # (6, n+1, n)
    dx = cdgrid.dx_edge_y   # (6, n, n+1)

    flux_x = h_face_x * u_c * dy
    flux_y = h_face_y * v_c * dx

    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]

    return -(net_x + net_y) / cdgrid.base.area


# ==============================================================================
# Scalar advection on C-grid
# ==============================================================================

def cgrid_scalar_advection(q, u_c, v_c, cdgrid):
    """Advective transport of scalar q by C-grid velocities (upwind).

    Works for both 2D and 3D inputs.
    """
    return cgrid_mass_flux_divergence(q, u_c, v_c, cdgrid)


# ==============================================================================
# Arakawa-Lamb gradient at D-grid corners
# ==============================================================================

def _arakawa_lamb_gradient(B, cdgrid):
    """4-point Arakawa-Lamb gradient of cell-centre field at D-grid corners.

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    B : jax.Array, shape (6, n, n[, nlev])

    Returns
    -------
    dB_dx, dB_dy : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    B_pad = _pad_halo_auto(B, cdgrid)

    if B.ndim == 3:
        B_sw = B_pad[:, :-1, :-1]
        B_se = B_pad[:, 1:, :-1]
        B_nw = B_pad[:, :-1, 1:]
        B_ne = B_pad[:, 1:, 1:]
    else:
        B_sw = B_pad[:, :-1, :-1, :]
        B_se = B_pad[:, 1:, :-1, :]
        B_nw = B_pad[:, :-1, 1:, :]
        B_ne = B_pad[:, 1:, 1:, :]

    dx_pad = jnp.pad(cdgrid.base.dx, ((0, 0), (1, 1), (1, 1)), mode='edge')
    dy_pad = jnp.pad(cdgrid.base.dy, ((0, 0), (1, 1), (1, 1)), mode='edge')
    dx_dual = 0.25 * (dx_pad[:, :-1, :-1] + dx_pad[:, 1:, :-1]
                       + dx_pad[:, :-1, 1:] + dx_pad[:, 1:, 1:])
    dy_dual = 0.25 * (dy_pad[:, :-1, :-1] + dy_pad[:, 1:, :-1]
                       + dy_pad[:, :-1, 1:] + dy_pad[:, 1:, 1:])

    dx_dual_b = _broadcast_metric(dx_dual, B_sw)
    dy_dual_b = _broadcast_metric(dy_dual, B_sw)

    dB_dx = ((B_se + B_ne) - (B_sw + B_nw)) / jnp.maximum(dx_dual_b, 1e-10)
    dB_dy = ((B_nw + B_ne) - (B_sw + B_se)) / jnp.maximum(dy_dual_b, 1e-10)

    return dB_dx, dB_dy


# ==============================================================================
# Interpolation helpers
# ==============================================================================

def _interp_center_to_corner(field, cdgrid):
    """Interpolate cell-centre field to D-grid corners (4-point average).

    Works for both 2D (6, n, n) and 3D (6, n, n, nlev) inputs.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n[, nlev])

    Returns
    -------
    jax.Array, shape (6, n+1, n+1[, nlev])
    """
    f_pad = _pad_halo_auto(field, cdgrid)

    if field.ndim == 3:
        return 0.25 * (f_pad[:, :-1, :-1] + f_pad[:, 1:, :-1]
                        + f_pad[:, :-1, 1:] + f_pad[:, 1:, 1:])
    return 0.25 * (f_pad[:, :-1, :-1, :] + f_pad[:, 1:, :-1, :]
                    + f_pad[:, :-1, 1:, :] + f_pad[:, 1:, 1:, :])


def _interp_corner_to_center(field_d):
    """Interpolate D-grid corners to cell centres (4-point average).

    Parameters
    ----------
    field_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    jax.Array, shape (6, n, n[, nlev])
    """
    if field_d.ndim == 3:
        return 0.25 * (field_d[:, :-1, :-1] + field_d[:, 1:, :-1]
                        + field_d[:, :-1, 1:] + field_d[:, 1:, 1:])
    return 0.25 * (field_d[:, :-1, :-1, :] + field_d[:, 1:, :-1, :]
                    + field_d[:, :-1, 1:, :] + field_d[:, 1:, 1:, :])


# ==============================================================================
# Laplacian at D-grid corners
# ==============================================================================

def _laplacian_dgrid(u_d, cdgrid):
    """Laplacian of a D-grid field using A-grid round-trip.

    Interpolates D-grid → A-grid centres, applies the compact
    A-grid Laplacian (which uses proper inter-face halo exchange),
    then interpolates back to D-grid corners.

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev).
    For 3D, applies the Laplacian level-by-level.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1[, nlev])

    Returns
    -------
    jax.Array, shape (6, n+1, n+1[, nlev])
    """
    if u_d.ndim == 4:
        u_t = jnp.moveaxis(u_d, -1, 0)

        def lap_one(uk):
            return _laplacian_dgrid(uk, cdgrid)

        result = jax.vmap(lap_one)(u_t)
        return jnp.moveaxis(result, 0, -1)

    # 1. D-grid → A-grid (corner_to_center): (6, n+1, n+1) → (6, n, n)
    u_a = _interp_corner_to_center(u_d)

    # 2. A-grid Laplacian with proper halo exchange
    from legoesm.core.operators import laplacian_compact
    lap_a = laplacian_compact(u_a, cdgrid.base)  # (6, n, n)

    # 3. A-grid → D-grid (center_to_corner): (6, n, n) → (6, n+1, n+1)
    return _interp_center_to_corner(lap_a, cdgrid)


# ==============================================================================
# Vector-invariant momentum tendencies (unified 2D/3D)
# ==============================================================================

def cdgrid_momentum_tendencies(
    h_or_p, u_d, v_d, h_s_or_p_prime, cdgrid,
    g=9.80616, A_h=0.0, hyperdiff_coeff=0.0,
    rho_0=None, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
):
    """D-grid momentum tendencies (vector-invariant form).

    Unified for both shallow water (2D) and 3D primitive equations.

    For 2D (shallow water):
        du_d/dt = +zeta_abs * v_d - dB/dx + viscosity
        dv_d/dt = -zeta_abs * u_d - dB/dy + viscosity
        where B = KE + g*(h + h_s)

    For 3D (primitive equations / ocean):
        du_d/dt = zeta*v_d + f*v' - dKE/dx - (1/rho_0)*dp'/dx + viscosity
        dv_d/dt = -zeta*u_d - f*u' - dKE/dy - (1/rho_0)*dp'/dy + viscosity

    Parameters
    ----------
    h_or_p : jax.Array, shape (6, n, n[, nlev])
        Height (2D shallow water) or baroclinic pressure perturbation (3D).
    u_d, v_d : jax.Array, shape (6, n+1, n+1[, nlev])
    h_s_or_p_prime : jax.Array, shape (6, n, n)
        Surface topography (2D) — unused in 3D (pass zeros).
    cdgrid : CubedSphereCDGrid
    g : float
        Gravity (used for 2D Bernoulli function).
    A_h : float
        Laplacian viscosity [m^2/s].
    hyperdiff_coeff : float
        Biharmonic hyperdiffusion coefficient.
    rho_0 : float or None
        Reference density (3D ocean only).
    div_v : jax.Array or None
        Velocity divergence for skew-symmetric correction (3D).
    f_3d : jax.Array or None
        Coriolis parameter broadcast to 3D.
    u_prime, v_prime : jax.Array or None
        Baroclinic velocity deviation (3D ocean).

    Returns
    -------
    du_d_dt, dv_d_dt : jax.Array, shape (6, n+1, n+1[, nlev])
    """
    is_3d = u_d.ndim == 4

    # 1. Vorticity at cell centres
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    if is_3d:
        zeta_abs = zeta  # Coriolis handled separately via f_3d + u_prime/v_prime
    else:
        zeta_abs = zeta + cdgrid.base.f

    # 2. KE at cell centres from C-grid velocities
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    if is_3d:
        u_center = 0.5 * (u_c[:, :-1, :, :] + u_c[:, 1:, :, :])
        v_center = 0.5 * (v_c[:, :, :-1, :] + v_c[:, :, 1:, :])
    else:
        u_center = 0.5 * (u_c[:, :-1, :] + u_c[:, 1:, :])
        v_center = 0.5 * (v_c[:, :, :-1] + v_c[:, :, 1:])
    KE = 0.5 * (u_center ** 2 + v_center ** 2)

    # 3. Gradients at corners (Arakawa-Lamb)
    if is_3d:
        # 3D: separate KE and pressure gradients
        dKE_dx, dKE_dy = _arakawa_lamb_gradient(KE, cdgrid)
        dp_dx, dp_dy = _arakawa_lamb_gradient(h_or_p, cdgrid)
    else:
        # 2D: Bernoulli function B = KE + g*(h + h_s)
        B = KE + g * (h_or_p + h_s_or_p_prime)
        dB_dx, dB_dy = _arakawa_lamb_gradient(B, cdgrid)

    # 4. Vorticity at corners
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    # 5. Tendencies
    if is_3d:
        du_d_dt = zeta_corner * v_d - dKE_dx
        dv_d_dt = -zeta_corner * u_d - dKE_dy

        # Pressure gradient
        if rho_0 is not None:
            du_d_dt = du_d_dt - dp_dx / rho_0
            dv_d_dt = dv_d_dt - dp_dy / rho_0

        # Coriolis on baroclinic deviation
        if f_3d is not None and u_prime is not None and v_prime is not None:
            f_corner = cdgrid.f_corner[..., None]
            du_d_dt = du_d_dt + f_corner * v_prime
            dv_d_dt = dv_d_dt - f_corner * u_prime

        # Skew-symmetric correction
        if div_v is not None:
            div_corner = _interp_center_to_corner(div_v, cdgrid)
            du_d_dt = du_d_dt - 0.5 * u_d * div_corner
            dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner
    else:
        du_d_dt = zeta_corner * v_d - dB_dx
        dv_d_dt = -zeta_corner * u_d - dB_dy

    # 6. Laplacian viscosity
    if A_h > 0:
        du_d_dt = du_d_dt + A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + A_h * _laplacian_dgrid(v_d, cdgrid)

    # 7. Biharmonic hyperdiffusion
    if hyperdiff_coeff > 0:
        du_d_dt = du_d_dt - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_d, cdgrid), cdgrid)
        dv_d_dt = dv_d_dt - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_d, cdgrid), cdgrid)

    return du_d_dt, dv_d_dt


# ==============================================================================
# Legacy aliases for backward compatibility
# ==============================================================================

def dgrid_vorticity_3d(u_d, v_d, cdgrid):
    """Alias: dgrid_vorticity handles both 2D and 3D."""
    return dgrid_vorticity(u_d, v_d, cdgrid)


def cgrid_divergence_3d(u_c, v_c, cdgrid):
    """Alias: cgrid_divergence handles both 2D and 3D."""
    return cgrid_divergence(u_c, v_c, cdgrid)


def cgrid_mass_flux_divergence_3d(h, u_c, v_c, cdgrid):
    """Alias: cgrid_mass_flux_divergence handles both 2D and 3D."""
    return cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)


def _arakawa_lamb_gradient_3d(B, cdgrid):
    """Alias: _arakawa_lamb_gradient handles both 2D and 3D."""
    return _arakawa_lamb_gradient(B, cdgrid)


def _interp_center_to_corner_3d(field, cdgrid):
    """Alias: _interp_center_to_corner handles both 2D and 3D."""
    return _interp_center_to_corner(field, cdgrid)


def cdgrid_momentum_tendencies_3d(
    u_d, v_d, p_prime, cdgrid, rho_0,
    A_h=0.0, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
):
    """Alias: cdgrid_momentum_tendencies handles both 2D and 3D."""
    return cdgrid_momentum_tendencies(
        p_prime, u_d, v_d, jnp.zeros(cdgrid.base.area.shape), cdgrid,
        rho_0=rho_0, A_h=A_h, div_v=div_v, f_3d=f_3d,
        u_prime=u_prime, v_prime=v_prime,
    )
