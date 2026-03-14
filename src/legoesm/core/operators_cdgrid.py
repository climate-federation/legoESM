"""C-D grid operators on the cubed-sphere.

Provides the core operators for the FV3-style C-D grid discretisation:

* D-grid vorticity (at cell centres from corner winds via circulation)
* D-grid to C-grid wind interpolation
* C-grid divergence and mass flux
* Vector-invariant momentum tendencies with Arakawa-Lamb gradient
* 3D extensions for the primitive equation / ocean solvers

D-grid winds are prognostic (at cell corners, shape ``(6, n+1, n+1)``).
C-grid winds are diagnostic (at cell edges, ``(6, n+1, n)`` for ``u_c``
and ``(6, n, n+1)`` for ``v_c``).

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo


# ==============================================================================
# D-grid to C-grid interpolation
# ==============================================================================

def dgrid_to_cgrid(u_d, v_d, cdgrid):
    """Interpolate D-grid corner winds to C-grid edge-normal velocities.

    u_c at the x-interface between cells (i,j) and (i+1,j) is the average
    of u_d at the two corners of that edge: (i+1,j) and (i+1,j+1) →
    effectively an average along the y-direction (axis 2).

    Works for both 2D (6, n+1, n+1) and 3D (6, n+1, n+1, nlev) inputs.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1) or (6, n+1, n+1, nlev)
    v_d : jax.Array, shape (6, n+1, n+1) or (6, n+1, n+1, nlev)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    u_c : jax.Array, shape (6, n+1, n) or (6, n+1, n, nlev)
    v_c : jax.Array, shape (6, n, n+1) or (6, n, n+1, nlev)
    """
    # Average along axis 2 (y-direction) for u_c
    u_c = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    # Average along axis 1 (x-direction) for v_c
    v_c = 0.5 * (v_d[:, :-1] + v_d[:, 1:])
    return u_c, v_c


def cgrid_to_dgrid(u_c, v_c, cdgrid):
    """Interpolate C-grid edge velocities to D-grid corners.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n) or (6, n+1, n, nlev)
    v_c : jax.Array, shape (6, n, n+1) or (6, n, n+1, nlev)

    Returns
    -------
    u_d, v_d : jax.Array, shape (6, n+1, n+1) or (6, n+1, n+1, nlev)
    """
    # Pad axis 2 (y) for u_c
    pad_u = [(0, 0), (0, 0), (1, 1)] + [(0, 0)] * (u_c.ndim - 3)
    u_c_pad = jnp.pad(u_c, pad_u, mode='edge')
    u_d = 0.5 * (u_c_pad[:, :, :-1] + u_c_pad[:, :, 1:])

    # Pad axis 1 (x) for v_c
    pad_v = [(0, 0), (1, 1), (0, 0)] + [(0, 0)] * (v_c.ndim - 3)
    v_c_pad = jnp.pad(v_c, pad_v, mode='edge')
    v_d = 0.5 * (v_c_pad[:, :-1] + v_c_pad[:, 1:])
    return u_d, v_d


# ==============================================================================
# D-grid vorticity (circulation form)
# ==============================================================================

def dgrid_vorticity(u_d, v_d, cdgrid):
    """Relative vorticity at cell centres from D-grid corner winds.

    Uses the integral circulation form: ζ = (1/A) ∮ v · dl, which is
    exact for the D-grid and avoids the Hollingsworth-Kallberg instability.

    For cell (i,j) with corners (i,j), (i+1,j), (i+1,j+1), (i,j+1),
    the counterclockwise circulation is:

        Γ = u_south * dx_south + v_east * dy_east
          - u_north * dx_north - v_west * dy_west

    where each edge velocity is the average of the D-grid winds at the
    two corners of that edge.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1)
    v_d : jax.Array, shape (6, n+1, n+1)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    zeta : jax.Array, shape (6, n, n)
    """
    # Average D-grid winds along each cell edge
    u_south = 0.5 * (u_d[:, :-1, :-1] + u_d[:, 1:, :-1])   # (6, n, n)
    u_north = 0.5 * (u_d[:, :-1, 1:] + u_d[:, 1:, 1:])     # (6, n, n)
    v_east = 0.5 * (v_d[:, 1:, :-1] + v_d[:, 1:, 1:])       # (6, n, n)
    v_west = 0.5 * (v_d[:, :-1, :-1] + v_d[:, :-1, 1:])     # (6, n, n)

    # Edge lengths
    # dx_edge_y: shape (6, n, n+1) — distance corner (i,j) to (i+1,j)
    # dy_edge_x: shape (6, n+1, n) — distance corner (i,j) to (i,j+1)
    dx_south = cdgrid.dx_edge_y[:, :, :-1]   # (6, n, n)
    dx_north = cdgrid.dx_edge_y[:, :, 1:]    # (6, n, n)
    dy_west = cdgrid.dy_edge_x[:, :-1, :]    # (6, n, n)
    dy_east = cdgrid.dy_edge_x[:, 1:, :]     # (6, n, n)

    circ = (u_south * dx_south + v_east * dy_east
            - u_north * dx_north - v_west * dy_west)

    return circ / cdgrid.base.area


def dgrid_vorticity_3d(u_d, v_d, cdgrid):
    """Vorticity at cell centres for 3D fields.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1, nlev)
    v_d : jax.Array, shape (6, n+1, n+1, nlev)

    Returns
    -------
    zeta : jax.Array, shape (6, n, n, nlev)
    """
    u_south = 0.5 * (u_d[:, :-1, :-1, :] + u_d[:, 1:, :-1, :])
    u_north = 0.5 * (u_d[:, :-1, 1:, :] + u_d[:, 1:, 1:, :])
    v_east = 0.5 * (v_d[:, 1:, :-1, :] + v_d[:, 1:, 1:, :])
    v_west = 0.5 * (v_d[:, :-1, :-1, :] + v_d[:, :-1, 1:, :])

    dx_south = cdgrid.dx_edge_y[:, :, :-1, None]
    dx_north = cdgrid.dx_edge_y[:, :, 1:, None]
    dy_west = cdgrid.dy_edge_x[:, :-1, :, None]
    dy_east = cdgrid.dy_edge_x[:, 1:, :, None]

    circ = (u_south * dx_south + v_east * dy_east
            - u_north * dx_north - v_west * dy_west)

    return circ / cdgrid.base.area[..., None]


# ==============================================================================
# C-grid divergence
# ==============================================================================

def cgrid_divergence(u_c, v_c, cdgrid):
    """Exact flux-form divergence at cell centres.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n)
    v_c : jax.Array, shape (6, n, n+1)

    Returns
    -------
    div : jax.Array, shape (6, n, n)
    """
    flux_x = u_c * cdgrid.dy_edge_x   # (6, n+1, n)
    flux_y = v_c * cdgrid.dx_edge_y   # (6, n, n+1)
    net_x = flux_x[:, 1:, :] - flux_x[:, :-1, :]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    return (net_x + net_y) / cdgrid.base.area


def cgrid_divergence_3d(u_c, v_c, cdgrid):
    """Exact flux-form divergence for 3D fields.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n, nlev)
    v_c : jax.Array, shape (6, n, n+1, nlev)

    Returns
    -------
    div : jax.Array, shape (6, n, n, nlev)
    """
    dy = cdgrid.dy_edge_x[:, :, :, None]   # (6, n+1, n, 1)
    dx = cdgrid.dx_edge_y[:, :, :, None]   # (6, n, n+1, 1)
    flux_x = u_c * dy
    flux_y = v_c * dx
    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]
    return (net_x + net_y) / cdgrid.base.area[:, :, :, None]


# ==============================================================================
# C-grid mass flux
# ==============================================================================

def cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid):
    """Conservative mass flux divergence with upwind face values.

    Parameters
    ----------
    h : jax.Array, shape (6, n, n)
    u_c : jax.Array, shape (6, n+1, n)
    v_c : jax.Array, shape (6, n, n+1)

    Returns
    -------
    dh_dt : jax.Array, shape (6, n, n)
    """
    h_pad = pad_halo(h, interp_offsets=cdgrid.base.halo_interp_offsets)

    h_left_x = h_pad[:, :-1, 1:-1]
    h_right_x = h_pad[:, 1:, 1:-1]
    h_face_x = jnp.where(u_c > 0, h_left_x, h_right_x)

    h_left_y = h_pad[:, 1:-1, :-1]
    h_right_y = h_pad[:, 1:-1, 1:]
    h_face_y = jnp.where(v_c > 0, h_left_y, h_right_y)

    flux_x = h_face_x * u_c * cdgrid.dy_edge_x
    flux_y = h_face_y * v_c * cdgrid.dx_edge_y

    net_x = flux_x[:, 1:, :] - flux_x[:, :-1, :]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]

    return -(net_x + net_y) / cdgrid.base.area


def cgrid_mass_flux_divergence_3d(h, u_c, v_c, cdgrid):
    """Conservative mass flux divergence for 3D fields (upwind).

    Parameters
    ----------
    h : jax.Array, shape (6, n, n, nlev)
    u_c : jax.Array, shape (6, n+1, n, nlev)
    v_c : jax.Array, shape (6, n, n+1, nlev)

    Returns
    -------
    dh_dt : jax.Array, shape (6, n, n, nlev)
    """
    import jax

    # Pad each level individually via vmap
    def pad_one(h_k):
        return pad_halo(h_k, interp_offsets=cdgrid.base.halo_interp_offsets)

    h_transposed = jnp.moveaxis(h, -1, 0)      # (nlev, 6, n, n)
    h_pad_t = jax.vmap(pad_one)(h_transposed)   # (nlev, 6, n+2, n+2)
    h_pad = jnp.moveaxis(h_pad_t, 0, -1)        # (6, n+2, n+2, nlev)

    # h at x-interfaces: upwind from u_c sign
    # u_c has shape (6, n+1, n, nlev)
    h_left_x = h_pad[:, :-1, 1:-1, :]    # (6, n+1, n, nlev)
    h_right_x = h_pad[:, 1:, 1:-1, :]    # (6, n+1, n, nlev)
    h_face_x = jnp.where(u_c > 0, h_left_x, h_right_x)

    # h at y-interfaces: upwind from v_c sign
    # v_c has shape (6, n, n+1, nlev)
    h_left_y = h_pad[:, 1:-1, :-1, :]    # (6, n, n+1, nlev)
    h_right_y = h_pad[:, 1:-1, 1:, :]    # (6, n, n+1, nlev)
    h_face_y = jnp.where(v_c > 0, h_left_y, h_right_y)

    dy = cdgrid.dy_edge_x[:, :, :, None]   # (6, n+1, n, 1)
    dx = cdgrid.dx_edge_y[:, :, :, None]   # (6, n, n+1, 1)

    flux_x = h_face_x * u_c * dy
    flux_y = h_face_y * v_c * dx

    net_x = flux_x[:, 1:] - flux_x[:, :-1]
    net_y = flux_y[:, :, 1:] - flux_y[:, :, :-1]

    return -(net_x + net_y) / cdgrid.base.area[:, :, :, None]


# ==============================================================================
# Scalar advection on C-grid
# ==============================================================================

def cgrid_scalar_advection(q, u_c, v_c, cdgrid):
    """Advective transport of scalar q by C-grid velocities (upwind).

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    u_c, v_c : C-grid velocities

    Returns
    -------
    dq_dt : jax.Array, shape (6, n, n)
    """
    return cgrid_mass_flux_divergence(q, u_c, v_c, cdgrid)


# ==============================================================================
# Arakawa-Lamb gradient at D-grid corners
# ==============================================================================

def _arakawa_lamb_gradient(B, cdgrid):
    """4-point Arakawa-Lamb gradient of cell-centre field at D-grid corners.

    At corner (i,j), the gradient uses the 4 surrounding cell values:
    (i-1,j-1), (i,j-1), (i-1,j), (i,j). The x-gradient is the average
    of the east pair minus the west pair, divided by the cell width.

    Parameters
    ----------
    B : jax.Array, shape (6, n, n)
        Cell-centre field.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    dB_dx, dB_dy : jax.Array, shape (6, n+1, n+1)
        Gradients at D-grid corners.
    """
    B_pad = pad_halo(B, interp_offsets=cdgrid.base.halo_interp_offsets)
    # (6, n+2, n+2), interior at [1:-1, 1:-1]

    # 4 cells surrounding each corner
    B_sw = B_pad[:, :-1, :-1]   # (6, n+1, n+1)
    B_se = B_pad[:, 1:, :-1]
    B_nw = B_pad[:, :-1, 1:]
    B_ne = B_pad[:, 1:, 1:]

    # Dual-cell spacing: average of 4 surrounding cell metrics
    dx_pad = jnp.pad(cdgrid.base.dx, ((0, 0), (1, 1), (1, 1)), mode='edge')
    dy_pad = jnp.pad(cdgrid.base.dy, ((0, 0), (1, 1), (1, 1)), mode='edge')
    dx_dual = 0.25 * (dx_pad[:, :-1, :-1] + dx_pad[:, 1:, :-1]
                       + dx_pad[:, :-1, 1:] + dx_pad[:, 1:, 1:])
    dy_dual = 0.25 * (dy_pad[:, :-1, :-1] + dy_pad[:, 1:, :-1]
                       + dy_pad[:, :-1, 1:] + dy_pad[:, 1:, 1:])

    dB_dx = 0.5 * ((B_se + B_ne) - (B_sw + B_nw)) / jnp.maximum(dx_dual, 1e-10)
    dB_dy = 0.5 * ((B_nw + B_ne) - (B_sw + B_se)) / jnp.maximum(dy_dual, 1e-10)

    return dB_dx, dB_dy


def _arakawa_lamb_gradient_3d(B, cdgrid):
    """4-point Arakawa-Lamb gradient for 3D fields.

    Parameters
    ----------
    B : jax.Array, shape (6, n, n, nlev)

    Returns
    -------
    dB_dx, dB_dy : jax.Array, shape (6, n+1, n+1, nlev)
    """
    import jax

    nlev = B.shape[-1]
    B_t = jnp.moveaxis(B, -1, 0)   # (nlev, 6, n, n)

    def pad_one(b):
        return pad_halo(b, interp_offsets=cdgrid.base.halo_interp_offsets)

    B_pad_t = jax.vmap(pad_one)(B_t)   # (nlev, 6, n+2, n+2)
    B_pad = jnp.moveaxis(B_pad_t, 0, -1)  # (6, n+2, n+2, nlev)

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

    dB_dx = 0.5 * ((B_se + B_ne) - (B_sw + B_nw)) / jnp.maximum(dx_dual[..., None], 1e-10)
    dB_dy = 0.5 * ((B_nw + B_ne) - (B_sw + B_se)) / jnp.maximum(dy_dual[..., None], 1e-10)

    return dB_dx, dB_dy


# ==============================================================================
# Interpolation helpers
# ==============================================================================

def _interp_center_to_corner(field, cdgrid):
    """Interpolate cell-centre field to D-grid corners (4-point average).

    Parameters
    ----------
    field : jax.Array, shape (6, n, n)

    Returns
    -------
    jax.Array, shape (6, n+1, n+1)
    """
    f_pad = pad_halo(field, interp_offsets=cdgrid.base.halo_interp_offsets)
    return 0.25 * (f_pad[:, :-1, :-1] + f_pad[:, 1:, :-1]
                    + f_pad[:, :-1, 1:] + f_pad[:, 1:, 1:])


def _interp_center_to_corner_3d(field, cdgrid):
    """Interpolate cell-centre 3D field to D-grid corners.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n, nlev)

    Returns
    -------
    jax.Array, shape (6, n+1, n+1, nlev)
    """
    import jax

    nlev = field.shape[-1]
    f_t = jnp.moveaxis(field, -1, 0)

    def pad_one(fk):
        return pad_halo(fk, interp_offsets=cdgrid.base.halo_interp_offsets)

    f_pad_t = jax.vmap(pad_one)(f_t)
    f_pad = jnp.moveaxis(f_pad_t, 0, -1)

    return 0.25 * (f_pad[:, :-1, :-1, :] + f_pad[:, 1:, :-1, :]
                    + f_pad[:, :-1, 1:, :] + f_pad[:, 1:, 1:, :])


# ==============================================================================
# Laplacian at D-grid corners
# ==============================================================================

def _laplacian_dgrid(u_d, cdgrid):
    """5-point Laplacian of a D-grid field.

    Parameters
    ----------
    u_d : jax.Array, shape (6, n+1, n+1)

    Returns
    -------
    jax.Array, shape (6, n+1, n+1)
    """
    dx_pad = jnp.pad(cdgrid.base.dx, ((0, 0), (1, 1), (1, 1)), mode='edge')
    dx_dual = 0.25 * (dx_pad[:, :-1, :-1] + dx_pad[:, 1:, :-1]
                       + dx_pad[:, :-1, 1:] + dx_pad[:, 1:, 1:])
    dy_dual = dx_dual  # approximate isotropy

    u_pad = jnp.pad(u_d, ((0, 0), (1, 1), (1, 1)), mode='edge')
    return (
        u_pad[:, 2:, 1:-1] + u_pad[:, :-2, 1:-1]
        + u_pad[:, 1:-1, 2:] + u_pad[:, 1:-1, :-2]
        - 4.0 * u_d
    ) / (dx_dual * dy_dual)


# ==============================================================================
# Vector-invariant momentum (2D shallow water)
# ==============================================================================

def cdgrid_momentum_tendencies(
    h, u_d, v_d, h_s, cdgrid, g=9.80616,
    A_h=0.0, hyperdiff_coeff=0.0,
):
    """D-grid momentum tendencies (vector-invariant form, shallow water).

    du_d/dt = +ζ_abs * v_d - ∂B/∂x + viscosity
    dv_d/dt = -ζ_abs * u_d - ∂B/∂y + viscosity

    where B = KE + g*(h + h_s) and ζ_abs = ζ + f.

    Parameters
    ----------
    h : jax.Array, shape (6, n, n)
    u_d, v_d : jax.Array, shape (6, n+1, n+1)
    h_s : jax.Array, shape (6, n, n)
    cdgrid : CubedSphereCDGrid
    g : float
    A_h : float
    hyperdiff_coeff : float

    Returns
    -------
    du_d_dt, dv_d_dt : jax.Array, shape (6, n+1, n+1)
    """
    # 1. Vorticity at cell centres
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    zeta_abs = zeta + cdgrid.base.f

    # 2. KE at cell centres from C-grid velocities
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    u_center = 0.5 * (u_c[:, :-1, :] + u_c[:, 1:, :])
    v_center = 0.5 * (v_c[:, :, :-1] + v_c[:, :, 1:])
    KE = 0.5 * (u_center ** 2 + v_center ** 2)

    # 3. Bernoulli function
    B = KE + g * (h + h_s)

    # 4. Gradient at corners (Arakawa-Lamb)
    dB_dx, dB_dy = _arakawa_lamb_gradient(B, cdgrid)

    # 5. Vorticity at corners (interpolated from centres)
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)

    # 6. Tendencies
    du_d_dt = zeta_corner * v_d - dB_dx
    dv_d_dt = -zeta_corner * u_d - dB_dy

    # 7. Laplacian viscosity
    if A_h > 0:
        du_d_dt = du_d_dt + A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + A_h * _laplacian_dgrid(v_d, cdgrid)

    # 8. Biharmonic hyperdiffusion
    if hyperdiff_coeff > 0:
        du_d_dt = du_d_dt - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_d, cdgrid), cdgrid)
        dv_d_dt = dv_d_dt - hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_d, cdgrid), cdgrid)

    return du_d_dt, dv_d_dt


# ==============================================================================
# 3D momentum tendencies (for PE / ocean solvers)
# ==============================================================================

def cdgrid_momentum_tendencies_3d(
    u_d, v_d, p_prime, cdgrid, rho_0,
    A_h=0.0, div_v=None, f_3d=None,
    u_prime=None, v_prime=None,
):
    """D-grid momentum tendencies for 3D primitive equations.

    du_d/dt = ζ·v_d + f·v_prime - ∂(KE)/∂x - (1/ρ₀)·∂p'/∂x - ½u·div(v)
    dv_d/dt = -ζ·u_d - f·u_prime - ∂(KE)/∂y - (1/ρ₀)·∂p'/∂y - ½v·div(v)

    Parameters
    ----------
    u_d, v_d : jax.Array, shape (6, n+1, n+1, nlev)
    p_prime : jax.Array, shape (6, n, n, nlev)
        Baroclinic pressure perturbation at cell centres.
    cdgrid : CubedSphereCDGrid
    rho_0 : float
    A_h : float
    div_v : jax.Array, shape (6, n, n, nlev) or None
        Velocity divergence for skew-symmetric correction.
    f_3d : jax.Array, shape (6, n, n, 1) or None
        Coriolis parameter broadcast to 3D.
    u_prime, v_prime : jax.Array, shape (6, n+1, n+1, nlev) or None
        Baroclinic velocity deviation.

    Returns
    -------
    du_d_dt, dv_d_dt : jax.Array, shape (6, n+1, n+1, nlev)
    """
    # 1. Vorticity at cell centres
    zeta = dgrid_vorticity_3d(u_d, v_d, cdgrid)

    # 2. KE at cell centres
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    u_center = 0.5 * (u_c[:, :-1, :, :] + u_c[:, 1:, :, :])
    v_center = 0.5 * (v_c[:, :, :-1, :] + v_c[:, :, 1:, :])
    KE = 0.5 * (u_center ** 2 + v_center ** 2)

    # 3. Gradients at corners
    dKE_dx, dKE_dy = _arakawa_lamb_gradient_3d(KE, cdgrid)
    dp_dx, dp_dy = _arakawa_lamb_gradient_3d(p_prime, cdgrid)

    # 4. Vorticity at corners
    zeta_corner = _interp_center_to_corner_3d(zeta, cdgrid)

    # 5. Tendencies
    du_d_dt = zeta_corner * v_d - dKE_dx - dp_dx / rho_0
    dv_d_dt = -zeta_corner * u_d - dKE_dy - dp_dy / rho_0

    # 6. Coriolis on baroclinic deviation (avoid double-counting barotropic)
    if f_3d is not None and u_prime is not None and v_prime is not None:
        f_corner = cdgrid.f_corner[..., None]  # (6, n+1, n+1, 1)
        du_d_dt = du_d_dt + f_corner * v_prime
        dv_d_dt = dv_d_dt - f_corner * u_prime

    # 7. Skew-symmetric correction for divergent flow
    if div_v is not None:
        div_corner = _interp_center_to_corner_3d(div_v, cdgrid)
        du_d_dt = du_d_dt - 0.5 * u_d * div_corner
        dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner

    # 8. Laplacian viscosity
    if A_h > 0:
        for k in range(u_d.shape[-1]):
            lap_u_k = _laplacian_dgrid(u_d[..., k], cdgrid)
            lap_v_k = _laplacian_dgrid(v_d[..., k], cdgrid)
            du_d_dt = du_d_dt.at[..., k].add(A_h * lap_u_k)
            dv_d_dt = dv_d_dt.at[..., k].add(A_h * lap_v_k)

    return du_d_dt, dv_d_dt
