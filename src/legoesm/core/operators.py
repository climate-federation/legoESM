"""Discrete differential operators on the cubed-sphere grid.

All operators are pure functions (no side effects) operating on JAX arrays.
They are compatible with jit, grad, vmap, and scan.

For the A-grid (collocated) shallow-water equations, all fields live at
cell centers. We use second-order centered finite differences within each
face, with proper inter-face halo exchange at face boundaries via
pad_halo (scalar) and pad_halo_vector (vector).

For the C-grid formulation (future), velocities live on edges and scalars
at cell centers, requiring different operator implementations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector


# ==============================================================================
# Core Finite-Difference Operators (A-grid, 2nd order)
# ==============================================================================

def gradient_x(field: Field, grid: CubedSphereGrid) -> Field:
    """Compute d(field)/dx using centered differences.

    Uses 2nd-order centered differences along the grid x-axis (axis=1).
    Proper inter-face halo exchange ensures correct boundary values.

    Parameters
    ----------
    field : Field
        Scalar field at cell centers, shape (6, n, n).
    grid : CubedSphereGrid
        The grid with metric terms.

    Returns
    -------
    Field : d(field)/dx, shape (6, n, n).
    """
    padded = pad_halo(field.data)
    # Centered difference: (f[i+1,j] - f[i-1,j]) / (2*dx)
    # In padded array: i+1 = padded[:, 2:, 1:-1], i-1 = padded[:, :-2, 1:-1]
    df_dx = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / grid.dx
    return field.replace(data=df_dx, name=f"d{field.name}_dx", units=f"{field.units}/m")


def gradient_y(field: Field, grid: CubedSphereGrid) -> Field:
    """Compute d(field)/dy using centered differences.

    Same as gradient_x but along axis=2 (y-direction).
    """
    padded = pad_halo(field.data)
    # j+1 = padded[:, 1:-1, 2:], j-1 = padded[:, 1:-1, :-2]
    df_dy = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / grid.dy
    return field.replace(data=df_dy, name=f"d{field.name}_dy", units=f"{field.units}/m")


def gradient(field: Field, grid: CubedSphereGrid) -> tuple[Field, Field]:
    """Compute the horizontal gradient of a scalar field.

    Returns (dfield/dx, dfield/dy) in grid-aligned coordinates.
    """
    return gradient_x(field, grid), gradient_y(field, grid)


def divergence(u_field: Field, v_field: Field, grid: CubedSphereGrid) -> Field:
    """Compute horizontal divergence of a vector field.

    div(u, v) = (1/(h_x*h_y)) * [d(u*h_y)/di + d(v*h_x)/dj]

    where h_x = grid.dx/2, h_y = grid.dy/2 are single-cell edge lengths
    (grid.dx/dy span 2 cells: from cell i-1 to i+1), i/j are integer
    indices, and the centered difference d/di = (f[i+1]-f[i-1])/2.

    Uses vector halo exchange to correctly handle velocity component
    rotation at face boundaries.

    Parameters
    ----------
    u_field, v_field : Field
        Vector components at cell centers, shape (6, n, n).
    grid : CubedSphereGrid
        The grid.

    Returns
    -------
    Field : Divergence, shape (6, n, n).
    """
    u = u_field.data
    v = v_field.data

    # Exchange velocity components with proper rotation, then apply
    # locally extrapolated half-metrics on the padded stencil.
    #
    # This avoids assuming dx == dy across face boundaries; on this grid,
    # anisotropy can be significant near edges/corners.
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
    )

    flux_x_pad = u_pad * grid.hy_ext
    flux_y_pad = v_pad * grid.hx_ext

    # Centered differences of metric-weighted fluxes:
    # d(u*h_y)/di ≈ (u*h_y)[i+1] - (u*h_y)[i-1]
    d_flux_x = flux_x_pad[:, 2:, 1:-1] - flux_x_pad[:, :-2, 1:-1]
    d_flux_y = flux_y_pad[:, 1:-1, 2:] - flux_y_pad[:, 1:-1, :-2]

    # div = (d_flux_x + d_flux_y) / (2 * area)
    # Factor of 2 from centered difference: d/di = (f[i+1]-f[i-1])/2
    div_data = (d_flux_x + d_flux_y) / (2.0 * grid.area)

    return Field(data=div_data, name="divergence", dims=u_field.dims,
                 units="1/s", staggering="cell")


def curl_z(u_field: Field, v_field: Field, grid: CubedSphereGrid) -> Field:
    """Compute the vertical component of curl (vorticity).

    vorticity = dv/dx - du/dy (in grid-aligned coordinates)

    Uses vector halo exchange for proper boundary treatment.

    Parameters
    ----------
    u_field, v_field : Field
        Vector components at cell centers, shape (6, n, n).
    grid : CubedSphereGrid
        The grid.

    Returns
    -------
    Field : Relative vorticity, shape (6, n, n).
    """
    u = u_field.data
    v = v_field.data

    # Pad vector components with proper rotation (using precomputed trig)
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
    )

    # Orthogonal-curvilinear finite-volume form:
    # zeta = (1/area) * [d(v*h_y)/di - d(u*h_x)/dj] / 2
    # with h_x = dx/2, h_y = dy/2 on the local (padded) stencil.
    vort_x = v_pad * grid.hy_ext
    vort_y = u_pad * grid.hx_ext

    d_vort_x = vort_x[:, 2:, 1:-1] - vort_x[:, :-2, 1:-1]
    d_vort_y = vort_y[:, 1:-1, 2:] - vort_y[:, 1:-1, :-2]

    vort_data = (d_vort_x - d_vort_y) / (2.0 * grid.area)

    return Field(data=vort_data, name="vorticity", dims=u_field.dims,
                 units="1/s", staggering="cell")


def laplacian(field: Field, grid: CubedSphereGrid) -> Field:
    """Compute the Laplacian of a scalar field.

    laplacian(f) = d^2f/dx^2 + d^2f/dy^2

    Uses 2nd-order centered differences with halo exchange.
    """
    data = field.data
    padded = pad_halo(data)

    # d^2f/dx^2: (f[i+1] - 2*f[i] + f[i-1]) / (dx/2)^2
    d2f_dx2 = (
        padded[:, 2:, 1:-1] - 2.0 * data + padded[:, :-2, 1:-1]
    ) / (grid.dx**2 / 4.0)  # dx is the distance over 2 cells

    # d^2f/dy^2
    d2f_dy2 = (
        padded[:, 1:-1, 2:] - 2.0 * data + padded[:, 1:-1, :-2]
    ) / (grid.dy**2 / 4.0)

    lap_data = d2f_dx2 + d2f_dy2

    return field.replace(data=lap_data, name=f"laplacian_{field.name}")


# ==============================================================================
# Higher-order advection operators
# ==============================================================================

def advect_upwind(
    q: Field, u: Field, v: Field, grid: CubedSphereGrid
) -> Field:
    """First-order upwind advection of scalar q by velocity (u, v).

    -u * dq/dx - v * dq/dy, using upwind differencing for stability.
    """
    q_pad = pad_halo(q.data)
    u_data = u.data
    v_data = v.data
    q_data = q.data

    # Upwind in x
    dq_fwd_x = q_pad[:, 2:, 1:-1] - q_data      # q[i+1] - q[i]
    dq_bwd_x = q_data - q_pad[:, :-2, 1:-1]      # q[i] - q[i-1]
    dq_dx = jnp.where(u_data > 0, dq_bwd_x, dq_fwd_x) / (grid.dx / 2.0)

    # Upwind in y
    dq_fwd_y = q_pad[:, 1:-1, 2:] - q_data       # q[j+1] - q[j]
    dq_bwd_y = q_data - q_pad[:, 1:-1, :-2]      # q[j] - q[j-1]
    dq_dy = jnp.where(v_data > 0, dq_bwd_y, dq_fwd_y) / (grid.dy / 2.0)

    adv = -(u_data * dq_dx + v_data * dq_dy)
    return q.replace(data=adv, name=f"advect_{q.name}")


def advect_centered(
    q: Field, u: Field, v: Field, grid: CubedSphereGrid
) -> Field:
    """Second-order centered advection of scalar q by velocity (u, v).

    -u * dq/dx - v * dq/dy, using centered differences.
    Note: centered advection is non-dissipative but can be unstable
    without explicit diffusion. Pair with SSP-RK3 for stability.
    """
    dq_dx = gradient_x(q, grid)
    dq_dy = gradient_y(q, grid)

    adv = -(u.data * dq_dx.data + v.data * dq_dy.data)
    return q.replace(data=adv, name=f"advect_{q.name}")


def hyperdiffusion(field: Field, grid: CubedSphereGrid, coeff: float) -> Field:
    """Fourth-order hyperdiffusion: -coeff * nabla^4(field).

    Provides scale-selective damping of grid-scale noise while
    preserving large-scale features. Essential for stability of
    centered advection schemes.
    """
    lap1 = laplacian(field, grid)
    lap2 = laplacian(lap1, grid)
    return field.replace(data=-coeff * lap2.data, name=f"hyperdiff_{field.name}")


# ==============================================================================
# Global integrals (for conservation)
# ==============================================================================

def global_integral(field: Field, grid: CubedSphereGrid) -> jax.Array:
    """Compute the area-weighted global integral of a field.

    integral = sum(field * area) over all cells.

    When running under MPI (distributed halo backend), the local
    partial sum is combined across all ranks via ``allreduce(SUM)``.

    Parameters
    ----------
    field : Field
        Scalar field at cell centers, shape (6, n, n).
    grid : CubedSphereGrid
        The grid with cell areas.

    Returns
    -------
    scalar : The global integral.
    """
    local_sum = jnp.sum(field.data * grid.area)
    if _is_distributed():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_sum)
    return local_sum


def _is_distributed() -> bool:
    """Check if the MPI halo backend is active."""
    from legoesm.grids.halo import get_halo_backend
    return get_halo_backend() == "mpi"


def global_mean(field: Field, grid: CubedSphereGrid) -> jax.Array:
    """Compute the area-weighted global mean of a field."""
    return global_integral(field, grid) / grid.total_area
