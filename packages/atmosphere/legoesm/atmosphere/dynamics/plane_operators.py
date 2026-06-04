"""Discrete operators on the doubly-periodic Cartesian plane.

Staged-not-integrated: these operators feed the future plane non-
hydrostatic dycore (PR2 of the CRM rollout). They are exercised in
isolation by ``tests/unit/test_plane_operators.py`` and
``tests/unit/test_plane_halo.py``.

Staggering convention
---------------------
Arakawa-C, doubly periodic. All horizontal arrays carry shape ``(...,
ny, nx)`` with no duplicated periodic endpoint.

- scalar (T-point, cell centre): ``phi[..., j, i]`` lives at
  ``(xc[i], yc[j])`` where ``xc[i] = (i + 0.5) dx``, ``yc[j] = (j +
  0.5) dy``.
- u (x-face): ``u[..., j, i]`` lives at ``(xu[i], yc[j])`` where
  ``xu[i] = i dx``. It is the face between cell ``i-1`` and cell ``i``
  (mod nx).
- v (y-face): ``v[..., j, i]`` lives at ``(xc[i], yv[j])`` where
  ``yv[j] = j dy``. It is the face between cell ``j-1`` and cell ``j``
  (mod ny).
- vorticity / corner: ``q[..., j, i]`` lives at the SW corner of cell
  ``(j, i)``, i.e. ``(xu[i], yv[j])``.
- vertical velocity w lives on cell interfaces with shape
  ``(nlev+1, ny, nx)`` and is not consumed by the horizontal operators
  in this module.

Index formulas (periodic mod nx, ny)
------------------------------------
- ``grad_x(phi)[..., j, i] = (phi[..., j, i] - phi[..., j, i-1]) / dx``
- ``grad_y(phi)[..., j, i] = (phi[..., j, i] - phi[..., j-1, i]) / dy``
- ``divergence(u, v)[..., j, i] = (u[..., j, i+1] - u[..., j, i]) / dx
                                + (v[..., j+1, i] - v[..., j, i]) / dy``
- ``curl(u, v)[..., j, i] = (v[..., j, i] - v[..., j, i-1]) / dx
                          - (u[..., j, i] - u[..., j-1, i]) / dy``
- ``laplacian(phi)[..., j, i] =
       (phi[..., j, i+1] - 2 phi[..., j, i] + phi[..., j, i-1]) / dx**2
     + (phi[..., j+1, i] - 2 phi[..., j, i] + phi[..., j-1, i]) / dy**2``

Inner-product weights and adjoint pairing
-----------------------------------------
Cells have uniform area ``dx * dy``. For arbitrary periodic ``phi``,
``u``, ``v`` the discrete integration-by-parts identity

    sum(phi * div(u, v)) * dx * dy
        == - sum(u * grad_x(phi)) * dx * dy
           - sum(v * grad_y(phi)) * dx * dy

holds *exactly* (to round-off) — see
``tests/unit/test_plane_operators.py::test_energy_consistent_pg_div_adjoint``.
This is the discrete equivalent of ``⟨φ, ∇·F⟩ = -⟨F, ∇φ⟩`` and is the
condition required for energy-consistent pressure-gradient /
divergence coupling in the future plane NH dycore.

Exact discrete identities
-------------------------
- ``divergence(F, G) == 0`` for any spatially constant ``F``, ``G``.
- ``grad_x(c) == 0``, ``grad_y(c) == 0`` for any spatially constant ``c``.
- ``curl(grad_x(phi), grad_y(phi)) == 0`` for any ``phi`` because the
  4-point mixed differences commute on the staggered grid.
- ``sum(divergence(u, v)) == 0`` for any ``u``, ``v`` under periodic
  BC (discrete Stokes).

Discrete Laplacian spectrum
---------------------------
For ``phi[..., j, i] = exp(2*pi*i*(k*i/nx + l*j/ny))`` (a discrete
plane wave) the eigenvalue of the 5-point Laplacian is

    -4 * sin^2(pi*k/nx) / dx^2 - 4 * sin^2(pi*l/ny) / dy^2 .

The continuous eigenvalue ``-(2*pi*k/Lx)^2 - (2*pi*l/Ly)^2`` is
recovered only in the long-wave limit (k, l ≪ nx, ny). Tests assert
the discrete formula to machine epsilon and grid-refinement
convergence at second order.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------- #
# Shape validation                                                      #
# --------------------------------------------------------------------- #


def _check_horizontal_shape(arr: jax.Array, grid, name: str) -> None:
    """Validate that ``arr`` has rank >= 3 and trailing axes ``(grid.ny,
    grid.nx)``.

    A transposed array, a wrong-sized horizontal block, or a 2D array
    would otherwise silently produce numerically valid-looking but
    incorrect stencils because every operator slices the last two
    axes via ``jnp.roll``.
    """
    if arr.ndim < 3:
        raise ValueError(
            f"{name} must have ndim >= 3 (..., ny, nx); got ndim={arr.ndim}, "
            f"shape={arr.shape}"
        )
    if arr.shape[-2:] != (grid.ny, grid.nx):
        raise ValueError(
            f"{name} trailing axes must be (ny={grid.ny}, nx={grid.nx}); "
            f"got shape {arr.shape}"
        )


# --------------------------------------------------------------------- #
# Horizontal differential operators                                     #
# --------------------------------------------------------------------- #


def grad_x_3d(phi: jax.Array, grid) -> jax.Array:
    """Cell-centre to x-face gradient.

    ``out[..., j, i] = (phi[..., j, i] - phi[..., j, i-1]) / dx``

    Parameters
    ----------
    phi : jax.Array
        Cell-centred scalar with horizontal shape ``(..., ny, nx)``.
    grid : PlaneGrid
        Provides ``dx``.

    Returns
    -------
    jax.Array
        x-face values with the same shape as ``phi``.
    """
    _check_horizontal_shape(phi, grid, "phi")
    return (phi - jnp.roll(phi, 1, axis=-1)) / grid.dx


def grad_y_3d(phi: jax.Array, grid) -> jax.Array:
    """Cell-centre to y-face gradient.

    ``out[..., j, i] = (phi[..., j, i] - phi[..., j-1, i]) / dy``
    """
    _check_horizontal_shape(phi, grid, "phi")
    return (phi - jnp.roll(phi, 1, axis=-2)) / grid.dy


def divergence_3d(u: jax.Array, v: jax.Array, grid) -> jax.Array:
    """Cell-centre divergence of an Arakawa-C face vector field.

    ``out[..., j, i] = (u[..., j, i+1] - u[..., j, i]) / dx
                     + (v[..., j+1, i] - v[..., j, i]) / dy``

    Parameters
    ----------
    u : jax.Array
        x-face component, shape ``(..., ny, nx)``.
    v : jax.Array
        y-face component, shape ``(..., ny, nx)``.
    grid : PlaneGrid
        Provides ``dx``, ``dy``.

    Returns
    -------
    jax.Array
        Cell-centred divergence, shape ``(..., ny, nx)``.
    """
    _check_horizontal_shape(u, grid, "u")
    _check_horizontal_shape(v, grid, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"u and v must have equal shape; got {u.shape} and {v.shape}"
        )
    du_dx = (jnp.roll(u, -1, axis=-1) - u) / grid.dx
    dv_dy = (jnp.roll(v, -1, axis=-2) - v) / grid.dy
    return du_dx + dv_dy


def curl_3d(u: jax.Array, v: jax.Array, grid) -> jax.Array:
    """Vertical component of curl at SW corner points.

    ``out[..., j, i] = (v[..., j, i] - v[..., j, i-1]) / dx
                     - (u[..., j, i] - u[..., j-1, i]) / dy``

    The corner grid is collocated with shape ``(..., ny, nx)`` — point
    ``(j, i)`` is at the SW corner of cell ``(j, i)`` (or equivalently
    the NE corner of cell ``(j-1, i-1)``).
    """
    _check_horizontal_shape(u, grid, "u")
    _check_horizontal_shape(v, grid, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"u and v must have equal shape; got {u.shape} and {v.shape}"
        )
    dv_dx = (v - jnp.roll(v, 1, axis=-1)) / grid.dx
    du_dy = (u - jnp.roll(u, 1, axis=-2)) / grid.dy
    return dv_dx - du_dy


def laplacian_3d(phi: jax.Array, grid) -> jax.Array:
    """5-point scalar Laplacian at cell centres.

    ``out[..., j, i] =
        (phi[..., j, i+1] - 2 phi[..., j, i] + phi[..., j, i-1]) / dx**2
      + (phi[..., j+1, i] - 2 phi[..., j, i] + phi[..., j-1, i]) / dy**2``
    """
    _check_horizontal_shape(phi, grid, "phi")
    lap_x = (
        jnp.roll(phi, -1, axis=-1) - 2.0 * phi + jnp.roll(phi, 1, axis=-1)
    ) / (grid.dx ** 2)
    lap_y = (
        jnp.roll(phi, -1, axis=-2) - 2.0 * phi + jnp.roll(phi, 1, axis=-2)
    ) / (grid.dy ** 2)
    return lap_x + lap_y


# --------------------------------------------------------------------- #
# Periodic halo padding                                                 #
# --------------------------------------------------------------------- #


def pad_halo_plane_4d(field: jax.Array, halo_width: int) -> jax.Array:
    """Periodic halo pad for a 4D ``(batch, nlev, ny, nx)`` field.

    Returns an array of shape ``(batch, nlev, ny + 2h, nx + 2h)`` whose
    halo cells contain the periodically wrapped values from the
    opposite side of the domain. Corners (NE, NW, SE, SW) are filled
    consistently with two-axis wrap, matching ``jnp.pad`` mode
    ``'wrap'``.

    Parameters
    ----------
    field : jax.Array
        Input with shape ``(batch, nlev, ny, nx)``. ``batch`` and
        ``nlev`` may be 1 or larger; only the last two axes are
        padded.
    halo_width : int
        Number of halo cells on each side. Must satisfy
        ``0 < halo_width < min(ny, nx)`` so that the halo never wraps
        the interior cells more than once.

    Returns
    -------
    jax.Array
        Padded field of shape ``(batch, nlev, ny + 2h, nx + 2h)``.

    Raises
    ------
    ValueError
        If ``field`` is not 4D, or if ``halo_width`` is out of range.
    """
    if field.ndim != 4:
        raise ValueError(
            f"pad_halo_plane_4d expects a 4D (batch, nlev, ny, nx) input; "
            f"got ndim={field.ndim}, shape={field.shape}"
        )
    if not isinstance(halo_width, int):
        raise ValueError(
            f"halo_width must be a Python int; got type {type(halo_width)}"
        )
    if halo_width <= 0:
        raise ValueError(
            f"halo_width must be positive; got {halo_width}"
        )
    _, _, ny, nx = field.shape
    if halo_width >= min(ny, nx):
        raise ValueError(
            f"halo_width={halo_width} must be strictly less than "
            f"min(ny={ny}, nx={nx})"
        )
    return jnp.pad(
        field,
        ((0, 0), (0, 0), (halo_width, halo_width), (halo_width, halo_width)),
        mode="wrap",
    )
