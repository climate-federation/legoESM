"""Consistent cubed-sphere FV operators for atmospheric dynamics.

Provides a shared interface-flux layer so that mass, tracer, pressure,
and momentum updates all use the same face-based discretization.

Key operators
-------------
fv_divergence       : 2D FV divergence consistent with PPM mass flux
fv_divergence_3d    : vmap wrapper for 3D fields
fv_divergence_damping   : 2nd + 4th order selective divergence damping
fv_divergence_damping_3d: vmap wrapper
default_div_damp_coeffs : Resolution-aware damping coefficients

The FV divergence uses the same interface velocity computation as
``fv_flux_divergence`` in ``operators_fv.py``, ensuring that the
divergence seen by sigma_dot / continuity is discretely compatible
with the PPM mass transport.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.halo import pad_halo, pad_halo_vector
from legoesm.grids.cubed_sphere import CubedSphereGrid


# ==============================================================================
# FV divergence — consistent with PPM mass flux
# ==============================================================================

def fv_divergence(u, v, grid):
    """Compute 2D divergence using the same interface velocities as PPM mass flux.

    This ensures that the divergence used for sigma_dot / vertical velocity
    is discretely compatible with the PPM mass transport, preventing the
    mass–momentum coupling instability.

    The interface velocity is the arithmetic mean of the two flanking
    cell-center values (same as in ``fv_flux_divergence``), and the
    interface metric is the mean of flanking half-edge lengths.

    Parameters
    ----------
    u, v : jax.Array, shape (6, n, n)
        Grid-aligned velocity components at cell centers.
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array, shape (6, n, n)
        Divergence at cell centers.
    """
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2, halo=2,
    )

    # --- X-direction: interface velocity × metric ---
    u_strip = u_pad[:, :, 2:-2]   # (6, n+4, n)
    u_iface = 0.5 * (u_strip[:, 1:-2, :] + u_strip[:, 2:-1, :])  # (6, n+1, n)

    hy = grid.hy_ext_h2[:, :, 2:-2]  # (6, n+4, n)
    hy_iface = 0.5 * (hy[:, 1:-2, :] + hy[:, 2:-1, :])  # (6, n+1, n)

    Phi_x = u_iface * hy_iface   # (6, n+1, n)

    # --- Y-direction ---
    v_strip = v_pad[:, 2:-2, :]   # (6, n, n+4)
    v_iface = 0.5 * (v_strip[:, :, 1:-2] + v_strip[:, :, 2:-1])  # (6, n, n+1)

    hx = grid.hx_ext_h2[:, 2:-2, :]  # (6, n, n+4)
    hx_iface = 0.5 * (hx[:, :, 1:-2] + hx[:, :, 2:-1])  # (6, n, n+1)

    Phi_y = v_iface * hx_iface   # (6, n, n+1)

    # --- Net flux divergence ---
    net_x = Phi_x[:, 1:, :] - Phi_x[:, :-1, :]
    net_y = Phi_y[:, :, 1:] - Phi_y[:, :, :-1]

    return (net_x + net_y) / grid.area


def fv_divergence_3d(u_3d, v_3d, grid):
    """FV divergence at all vertical levels via vmap.

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array, shape (6, n, n, nlev)
    """
    def _single_level(u_k, v_k):
        return fv_divergence(u_k, v_k, grid)

    u_t = jnp.moveaxis(u_3d, -1, 0)   # (nlev, 6, n, n)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    result = jax.vmap(_single_level)(u_t, v_t)  # (nlev, 6, n, n)
    return jnp.moveaxis(result, 0, -1)


# ==============================================================================
# Divergence damping
# ==============================================================================

def fv_divergence_damping(u, v, grid, nu2, nu4=0.0):
    """Selective divergence damping on the cubed-sphere.

    Damps only the divergent component of the flow:
        du_damp = +nu2 * d(div)/dx  -  nu4 * d(lap(div))/dx
        dv_damp = +nu2 * d(div)/dy  -  nu4 * d(lap(div))/dy

    The divergence is computed using the FV-consistent operator
    (same interface velocities as the mass flux).

    Parameters
    ----------
    u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    nu2 : float
        2nd-order divergence damping coefficient [m²/s].
    nu4 : float
        4th-order divergence damping coefficient [m⁴/s].

    Returns
    -------
    du_damp, dv_damp : jax.Array, shape (6, n, n)
    """
    du_damp = jnp.zeros_like(u)
    dv_damp = jnp.zeros_like(v)

    if nu2 <= 0.0 and nu4 <= 0.0:
        return du_damp, dv_damp

    # Divergence using FV-consistent operator
    div = fv_divergence(u, v, grid)  # (6, n, n)

    if nu2 > 0.0:
        div_pad = pad_halo(div, interp_offsets=grid.halo_interp_offsets)
        grad_div_x = (div_pad[:, 2:, 1:-1] - div_pad[:, :-2, 1:-1]) / grid.dx
        grad_div_y = (div_pad[:, 1:-1, 2:] - div_pad[:, 1:-1, :-2]) / grid.dy
        du_damp = du_damp + nu2 * grad_div_x
        dv_damp = dv_damp + nu2 * grad_div_y

    if nu4 > 0.0:
        div_pad = pad_halo(div, interp_offsets=grid.halo_interp_offsets)
        # Laplacian of divergence
        lap_div = (
            (div_pad[:, 2:, 1:-1] - 2 * div_pad[:, 1:-1, 1:-1] + div_pad[:, :-2, 1:-1])
            / (grid.dx / 2.0) ** 2
            + (div_pad[:, 1:-1, 2:] - 2 * div_pad[:, 1:-1, 1:-1] + div_pad[:, 1:-1, :-2])
            / (grid.dy / 2.0) ** 2
        )
        # Gradient of laplacian
        lap_div_pad = pad_halo(lap_div, interp_offsets=grid.halo_interp_offsets)
        grad_lap_x = (lap_div_pad[:, 2:, 1:-1] - lap_div_pad[:, :-2, 1:-1]) / grid.dx
        grad_lap_y = (lap_div_pad[:, 1:-1, 2:] - lap_div_pad[:, 1:-1, :-2]) / grid.dy
        du_damp = du_damp - nu4 * grad_lap_x
        dv_damp = dv_damp - nu4 * grad_lap_y

    return du_damp, dv_damp


def fv_divergence_damping_3d(u_3d, v_3d, grid, nu2, nu4=0.0):
    """Divergence damping at all vertical levels via vmap.

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    nu2, nu4 : float

    Returns
    -------
    du_damp, dv_damp : jax.Array, shape (6, n, n, nlev)
    """
    def _single_level(u_k, v_k):
        return fv_divergence_damping(u_k, v_k, grid, nu2, nu4)

    u_t = jnp.moveaxis(u_3d, -1, 0)
    v_t = jnp.moveaxis(v_3d, -1, 0)
    du_t, dv_t = jax.vmap(_single_level)(u_t, v_t)
    return jnp.moveaxis(du_t, 0, -1), jnp.moveaxis(dv_t, 0, -1)


# ==============================================================================
# Resolution-aware defaults
# ==============================================================================

def default_div_damp_coeffs(grid, dt=None):
    """Compute resolution-aware divergence damping coefficients.

    2nd-order:  nu2 ~ 0.12 * dx_min² / dt_ref
    4th-order:  nu4 ~ 0.005 * dx_min⁴ / dt_ref

    If *dt* is not given, uses a CFL-safe estimate:
        dt_ref = dx_min / (2 * c)  with c = 340 m/s (sound speed proxy)

    Parameters
    ----------
    grid : CubedSphereGrid
    dt : float or None

    Returns
    -------
    nu2, nu4 : float
    """
    dx_min = float(jnp.min(grid.dx / 2.0))  # single-cell width
    c_ref = 340.0   # sound speed proxy

    if dt is None:
        dt_ref = dx_min / (2.0 * c_ref)
    else:
        dt_ref = float(dt)

    nu2 = 0.12 * dx_min ** 2 / dt_ref
    nu4 = 0.005 * dx_min ** 4 / dt_ref

    return nu2, nu4
