"""Discrete differential operators on the latitude-longitude grid.

Parallel implementation to operators.py (cubed-sphere), providing the
same set of operators for the lat-lon grid. All operators are pure
functions compatible with jit, grad, vmap, and scan.

Key differences from cubed-sphere:
- Longitude: periodic boundary (wrap-around)
- Latitude: zero-gradient boundary at poles (no face connectivity)
- Spherical metric terms: cos(lat) factors in divergence/curl
- dx and dy span 2 cells, matching cubed-sphere convention
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.halo_latlon import (
    pad_halo_latlon, pad_halo_latlon_vector, pad_halo_vector_latlon,
)


# ==============================================================================
# Core Finite-Difference Operators (A-grid, 2nd order)
# ==============================================================================

def gradient_x(field: Field, grid: LatLonGrid) -> Field:
    """Compute d(field)/dx using centered differences in longitude.

    Parameters
    ----------
    field : Field
        Scalar field at cell centers, shape (n_lat, n_lon).
    grid : LatLonGrid
        The grid with metric terms.

    Returns
    -------
    Field : d(field)/dx, shape (n_lat, n_lon).
    """
    padded = pad_halo_latlon(field.data)
    # Centered difference: (f[j, i+1] - f[j, i-1]) / dx
    # In padded array: i+1 = padded[1:-1, 2:], i-1 = padded[1:-1, :-2]
    df_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / grid.dx
    return field.replace(data=df_dx, name=f"d{field.name}_dx", units=f"{field.units}/m")


def gradient_y(field: Field, grid: LatLonGrid) -> Field:
    """Compute d(field)/dy using centered differences in latitude.

    Parameters
    ----------
    field : Field
        Scalar field at cell centers, shape (n_lat, n_lon).
    grid : LatLonGrid
        The grid with metric terms.

    Returns
    -------
    Field : d(field)/dy, shape (n_lat, n_lon).
    """
    padded = pad_halo_latlon(field.data)
    # Centered difference: (f[j+1, i] - f[j-1, i]) / dy
    df_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / grid.dy
    return field.replace(data=df_dy, name=f"d{field.name}_dy", units=f"{field.units}/m")


def gradient(field: Field, grid: LatLonGrid) -> tuple[Field, Field]:
    """Compute the horizontal gradient of a scalar field."""
    return gradient_x(field, grid), gradient_y(field, grid)


def divergence(u_field: Field, v_field: Field, grid: LatLonGrid) -> Field:
    """Compute horizontal divergence of a vector field on the sphere.

    Spherical flux-form divergence:
        div = (1/A) * [Δ_lon(u * dy/2) + Δ_lat(v * dx/2)]

    where dy/2 and dx/2 are single-cell edge lengths, and the centered
    difference d/di = (f[i+1]-f[i-1])/2 provides the factor of 2.

    Note: v*dx varies with latitude because dx = R*2*dlon*cos(lat),
    which captures the d(v*cos(lat))/dlat metric term.

    Parameters
    ----------
    u_field, v_field : Field
        Vector components at cell centers, shape (n_lat, n_lon).
    grid : LatLonGrid
        The grid.

    Returns
    -------
    Field : Divergence, shape (n_lat, n_lon).
    """
    u = u_field.data
    v = v_field.data

    # Form fluxes: multiply by single-cell edge length (half of 2-cell span)
    flux_x = u * (grid.dy * 0.5)    # (n_lat, n_lon), dy is constant
    flux_y = v * (grid.dx * 0.5)    # (n_lat, n_lon), dx varies with lat

    flux_x_pad, flux_y_pad = pad_halo_vector_latlon(flux_x, flux_y)

    # Centered differences of fluxes
    d_flux_x = flux_x_pad[1:-1, 2:] - flux_x_pad[1:-1, :-2]  # d/dlon
    d_flux_y = flux_y_pad[2:, 1:-1] - flux_y_pad[:-2, 1:-1]  # d/dlat

    # div = (d_flux_x + d_flux_y) / (2 * area)
    div_data = (d_flux_x + d_flux_y) / (2.0 * grid.area)

    return Field(data=div_data, name="divergence", dims=u_field.dims,
                 units="1/s", staggering="cell")


def curl_z(u_field: Field, v_field: Field, grid: LatLonGrid) -> Field:
    """Compute the vertical component of curl (relative vorticity).

    Spherical vorticity:
        ζ = (1/A) * [Δ_lon(v * dy/2) - Δ_lat(u * dx/2)]

    Parameters
    ----------
    u_field, v_field : Field
        Vector components at cell centers, shape (n_lat, n_lon).
    grid : LatLonGrid
        The grid.

    Returns
    -------
    Field : Relative vorticity, shape (n_lat, n_lon).
    """
    u = u_field.data
    v = v_field.data

    # Form metric-weighted fields
    v_metric = v * (grid.dy * 0.5)   # v * single-cell dy
    u_metric = u * (grid.dx * 0.5)   # u * single-cell dx (varies with lat)

    v_pad = pad_halo_latlon_vector(v_metric)
    u_pad = pad_halo_latlon_vector(u_metric)

    dv_dx = v_pad[1:-1, 2:] - v_pad[1:-1, :-2]  # d/dlon
    du_dy = u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]  # d/dlat

    vort_data = (dv_dx - du_dy) / (2.0 * grid.area)

    return Field(data=vort_data, name="vorticity", dims=u_field.dims,
                 units="1/s", staggering="cell")


def laplacian(field: Field, grid: LatLonGrid) -> Field:
    """Compute the Laplacian of a scalar field.

    Uses 2nd-order centered differences with halo exchange.
    dx and dy span 2 cells, so single-cell spacing = dx/2 and dy/2.
    """
    data = field.data
    padded = pad_halo_latlon(data)

    # d^2f/dx^2: (f[i+1] - 2*f[i] + f[i-1]) / (dx/2)^2
    d2f_dx2 = (
        padded[1:-1, 2:] - 2.0 * data + padded[1:-1, :-2]
    ) / (grid.dx**2 / 4.0)

    # d^2f/dy^2
    d2f_dy2 = (
        padded[2:, 1:-1] - 2.0 * data + padded[:-2, 1:-1]
    ) / (grid.dy**2 / 4.0)

    lap_data = d2f_dx2 + d2f_dy2

    return field.replace(data=lap_data, name=f"laplacian_{field.name}")


def hyperdiffusion(field: Field, grid: LatLonGrid, coeff: float) -> Field:
    """Fourth-order hyperdiffusion: -coeff * nabla^4(field)."""
    lap1 = laplacian(field, grid)
    lap2 = laplacian(lap1, grid)
    return field.replace(data=-coeff * lap2.data, name=f"hyperdiff_{field.name}")


# ==============================================================================
# Global integrals (for conservation)
# ==============================================================================

def global_integral(field: Field, grid: LatLonGrid) -> jax.Array:
    """Compute the area-weighted global integral of a field.

    Uses an fp64 accumulator (via
    ``legoesm.core.conservation._conservation_accumulator``) so the
    result is well-conditioned even when ``field.data`` and
    ``grid.area`` are stored in fp32.  A plain ``jnp.sum`` over an
    fp32 product loses ~log2(n_cells) bits of precision and produces
    spurious O(0.1%-1%) "mass drift" on N=720x1440 lat-lon grids,
    while the cubed-sphere path (``operators.global_integral``)
    already promotes to fp64 before summing.

    Supports single-device and MPI distributed execution; the
    JAX/XLA NamedSharding multi-device path is handled implicitly
    via ``jnp.sum`` on a sharded array.
    """
    from legoesm.core.conservation import _conservation_accumulator
    acc = _conservation_accumulator()
    prod = field.data.astype(acc) * grid.area.astype(acc)
    local_sum = jnp.sum(prod)

    from legoesm.core.operators import _is_distributed
    if _is_distributed():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_sum)
    return local_sum


def global_mean(field: Field, grid: LatLonGrid) -> jax.Array:
    """Compute the area-weighted global mean of a field."""
    integ = global_integral(field, grid)
    return integ / grid.total_area.astype(integ.dtype)
