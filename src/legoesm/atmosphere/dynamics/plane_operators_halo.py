"""Halo-aware versions of plane stencil operators for MPI dycore.

Operators in :mod:`plane_operators` use ``jnp.roll`` for periodic
boundaries. Under MPI domain decomposition, ``jnp.roll`` wraps to
the wrong side (local boundary instead of neighbor rank's
opposite-edge data). This module provides equivalent operators
that operate on a HALO-PADDED array (shape
``(ny_local + 2h, nx_local + 2h, ...)``), using slicing into the
halo cells instead of wrapping.

Caller workflow:

1. Local interior state has shape ``(ny_local, nx_local, ...)``.
2. Call :func:`legoesm.parallel.plane_mpi.exchange_halo_plane_yxz`
   to fill halos → padded shape ``(ny_local + 2h, nx_local + 2h, ...)``.
3. Apply these halo-aware operators on the padded array → they
   return INTERIOR-shape results ``(ny_local, nx_local, ...)``
   that can be combined into tendencies.

Single-rank fallback: with ``layout.n_ranks == 1``,
``exchange_halo_plane_yxz`` uses ``jnp.pad(mode='wrap')`` so the
padded array contains the correct periodic data — the halo-aware
operators then produce bit-identical results to the unpadded
``jnp.roll`` versions.

Most operators in this module operate at ``halo == 1`` (5-point
Laplacian, 1st-order upwind, face/cell averages — single-cell
stencil radius). The WENO5 advection operators
(:func:`weno5_advection_x_halo`, :func:`weno5_advection_y_halo`)
require ``halo >= 3`` because the WENO-Z reconstruction reaches
``f[i-2..i+3]`` for the i+1/2 face and ``f[i-3..i+2]`` for the
i-1/2 face. All operators use the last-two-axes convention
``(..., ny, nx)`` to match ``plane_operators.py``.

Equivalence: for any function ``f`` in :mod:`plane_operators` with
stencil radius 1, the relation

    f(arr) == _f_halo(jnp.pad(arr, ((0,0),)*(arr.ndim-2)+((1,1),(1,1)),
                              mode='wrap'))

holds bit-for-bit (no extra arithmetic, just slice indexing). The
multi-rank case differs only in that the padded halos come from
neighbor ranks rather than ``jnp.pad(mode='wrap')``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def grad_x_3d_halo(phi_pad: jax.Array, grid, halo: int = 1) -> jax.Array:
    """Halo-aware ``grad_x_3d``: (phi[j,i] - phi[j,i-1])/dx.

    ``phi_pad`` shape ``(..., ny_local + 2h, nx_local + 2h)``.
    Returns interior ``(..., ny_local, nx_local)``.
    """
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_x = phi_pad[..., h:-h, h - 1 : -h - 1]
    return (interior - shifted_minus_x) / grid.dx


def grad_y_3d_halo(phi_pad: jax.Array, grid, halo: int = 1) -> jax.Array:
    """Halo-aware ``grad_y_3d``: (phi[j,i] - phi[j-1,i])/dy."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_y = phi_pad[..., h - 1 : -h - 1, h:-h]
    return (interior - shifted_minus_y) / grid.dy


def divergence_3d_halo(
    u_pad: jax.Array, v_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``divergence_3d``: (u[i+1]-u[i])/dx + (v[j+1]-v[j])/dy."""
    h = halo
    u_int = u_pad[..., h:-h, h:-h]
    u_shifted_plus_x = u_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    v_int = v_pad[..., h:-h, h:-h]
    v_shifted_plus_y = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    du_dx = (u_shifted_plus_x - u_int) / grid.dx
    dv_dy = (v_shifted_plus_y - v_int) / grid.dy
    return du_dx + dv_dy


def laplacian_3d_halo(
    phi_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware 5-point Laplacian."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    plus_x = phi_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    minus_x = phi_pad[..., h:-h, h - 1 : -h - 1]
    plus_y = phi_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    minus_y = phi_pad[..., h - 1 : -h - 1, h:-h]
    lap_x = (plus_x - 2.0 * interior + minus_x) / (grid.dx ** 2)
    lap_y = (plus_y - 2.0 * interior + minus_y) / (grid.dy ** 2)
    return lap_x + lap_y


def interp_cell_to_xface_halo(
    phi_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (phi[i] + phi[i-1])`` at x-face."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_x = phi_pad[..., h:-h, h - 1 : -h - 1]
    return 0.5 * (interior + shifted_minus_x)


def interp_cell_to_yface_halo(
    phi_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (phi[j] + phi[j-1])`` at y-face."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_y = phi_pad[..., h - 1 : -h - 1, h:-h]
    return 0.5 * (interior + shifted_minus_y)


def interp_xface_to_cell_halo(
    u_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (u[i] + u[i+1])`` at cell centre."""
    h = halo
    interior = u_pad[..., h:-h, h:-h]
    shifted_plus_x = u_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    return 0.5 * (interior + shifted_plus_x)


def interp_yface_to_cell_halo(
    v_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (v[j] + v[j+1])`` at cell centre."""
    h = halo
    interior = v_pad[..., h:-h, h:-h]
    shifted_plus_y = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    return 0.5 * (interior + shifted_plus_y)


def interp_yface_to_xface_halo(
    v_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware 4-point corner average: v at y-face → x-face."""
    h = halo
    v_int = v_pad[..., h:-h, h:-h]
    v_im1 = v_pad[..., h:-h, h - 1 : -h - 1]
    v_jp1 = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    v_im1_jp1 = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h - 1 : -h - 1]
    return 0.25 * (v_int + v_im1 + v_jp1 + v_im1_jp1)


def interp_xface_to_yface_halo(
    u_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware 4-point corner average: u at x-face → y-face."""
    h = halo
    u_int = u_pad[..., h:-h, h:-h]
    u_jm1 = u_pad[..., h - 1 : -h - 1, h:-h]
    u_ip1 = u_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    u_jm1_ip1 = u_pad[..., h - 1 : -h - 1, h + 1 : -h + 1 if h > 1 else None]
    return 0.25 * (u_int + u_jm1 + u_ip1 + u_jm1_ip1)


# --------------------------------------------------------------------- #
# Vlast (vertical-last) wrappers + advection helpers                    #
# Mirror the (ny, nx, nlev) → (ny, nx, nlev) shape contract used by     #
# plane_compressible_euler_slow_tendencies. The padded inputs have      #
# shape (ny_local + 2h, nx_local + 2h, nlev).                           #
# --------------------------------------------------------------------- #


def _vlast_to_back(field_yxz: jax.Array) -> jax.Array:
    """``(ny, nx, nlev) -> (nlev, ny, nx)``."""
    return jnp.moveaxis(field_yxz, -1, 0)


def _vlast_from_back(field_zyx: jax.Array) -> jax.Array:
    """``(nlev, ny, nx) -> (ny, nx, nlev)``."""
    return jnp.moveaxis(field_zyx, 0, -1)


def grad_x_vlast_halo(phi_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        grad_x_3d_halo(_vlast_to_back(phi_pad_yxz), grid, halo)
    )


def grad_y_vlast_halo(phi_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        grad_y_3d_halo(_vlast_to_back(phi_pad_yxz), grid, halo)
    )


def divergence_vlast_halo(u_pad_yxz, v_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        divergence_3d_halo(
            _vlast_to_back(u_pad_yxz),
            _vlast_to_back(v_pad_yxz),
            grid, halo,
        )
    )


def laplacian_vlast_halo(phi_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        laplacian_3d_halo(_vlast_to_back(phi_pad_yxz), grid, halo)
    )


def interp_cell_to_xface_vlast_halo(phi_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        interp_cell_to_xface_halo(_vlast_to_back(phi_pad_yxz), grid, halo)
    )


def interp_cell_to_yface_vlast_halo(phi_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        interp_cell_to_yface_halo(_vlast_to_back(phi_pad_yxz), grid, halo)
    )


def interp_xface_to_cell_vlast_halo(u_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        interp_xface_to_cell_halo(_vlast_to_back(u_pad_yxz), grid, halo)
    )


def interp_yface_to_cell_vlast_halo(v_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        interp_yface_to_cell_halo(_vlast_to_back(v_pad_yxz), grid, halo)
    )


def interp_yface_to_xface_vlast_halo(v_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        interp_yface_to_xface_halo(_vlast_to_back(v_pad_yxz), grid, halo)
    )


def interp_xface_to_yface_vlast_halo(u_pad_yxz, grid, halo=1):
    return _vlast_from_back(
        interp_xface_to_yface_halo(_vlast_to_back(u_pad_yxz), grid, halo)
    )


def upwind_advection_x_halo(
    field_pad_yxz: jax.Array,
    u_at_field_pad_yxz: jax.Array,
    dx: float,
    halo: int = 1,
) -> jax.Array:
    """Halo-aware first-order upwind ``-u df/dx``.

    Both ``field`` and ``u_at_field`` must be halo-padded
    ``(ny_local + 2h, nx_local + 2h, nlev)`` and CO-LOCATED on the
    Arakawa-C grid (caller invariant). Returns interior-shape
    tendency ``(ny_local, nx_local, nlev)``.
    """
    h = halo
    interior = field_pad_yxz[h:-h, h:-h, :]
    minus_x = field_pad_yxz[h:-h, h - 1 : -h - 1, :]
    plus_x = field_pad_yxz[h:-h, h + 1 : -h + 1 if h > 1 else None, :]
    f_backward = (interior - minus_x) / dx
    f_forward = (plus_x - interior) / dx
    u_int = u_at_field_pad_yxz[h:-h, h:-h, :]
    u_pos = jnp.maximum(u_int, 0.0)
    u_neg = jnp.minimum(u_int, 0.0)
    return -(u_pos * f_backward + u_neg * f_forward)


def upwind_advection_y_halo(
    field_pad_yxz: jax.Array,
    v_at_field_pad_yxz: jax.Array,
    dy: float,
    halo: int = 1,
) -> jax.Array:
    """Halo-aware first-order upwind ``-v df/dy``."""
    h = halo
    interior = field_pad_yxz[h:-h, h:-h, :]
    minus_y = field_pad_yxz[h - 1 : -h - 1, h:-h, :]
    plus_y = field_pad_yxz[h + 1 : -h + 1 if h > 1 else None, h:-h, :]
    f_backward = (interior - minus_y) / dy
    f_forward = (plus_y - interior) / dy
    v_int = v_at_field_pad_yxz[h:-h, h:-h, :]
    v_pos = jnp.maximum(v_int, 0.0)
    v_neg = jnp.minimum(v_int, 0.0)
    return -(v_pos * f_backward + v_neg * f_forward)


def _slice_axis_shift(arr, halo, axis, shift):
    """Return interior-shape view of ``arr`` at neighbour ``i+shift``.

    For padded ``arr`` of shape ``(ny+2h, nx+2h, nlev)``, the interior
    starts at index ``h`` along each horizontal axis. The slice that
    picks ``arr_int[i+shift]`` for every interior ``i`` along the
    chosen axis is ``arr[h+shift : (h+n)+shift, ..., :]`` (with the
    matching identity slice on the other axes). Requires
    ``|shift| <= halo``.
    """
    h = halo
    if abs(shift) > h:
        raise ValueError(
            f"shift {shift} exceeds halo {h}; WENO5 stencil needs |shift|≤halo."
        )
    start = h + shift
    end_neg = -h + shift  # exclusive end relative to the trailing axis
    end = end_neg if end_neg != 0 else None
    if axis == 0:
        return arr[start:end, h:-h, :]
    elif axis == 1:
        return arr[h:-h, start:end, :]
    raise ValueError(f"axis must be 0 (y) or 1 (x); got {axis}.")


def weno5_advection_x_halo(
    field_pad_yxz: jax.Array,
    u_at_field_pad_yxz: jax.Array,
    dx: float,
    halo: int = 3,
) -> jax.Array:
    """Halo-aware 5th-order WENO-Z upwind ``-u df/dx``.

    Mirror of ``compressible_euler_plane._weno5_advection_x`` using
    slice arithmetic on halo-padded inputs. Requires ``halo >= 3``
    (face i+1/2 stencil reaches ``f[i-2..i+3]`` and the i-1/2 face
    via roll-by-1 reaches ``f[i-3..i+2]``).

    Single-rank equivalence: when ``layout.n_ranks == 1`` and the
    padded inputs come from ``jnp.pad(mode='wrap')``, the slices
    pick the same neighbours as the serial ``jnp.roll`` stencils →
    bit-identical to the serial WENO5.
    """
    if halo < 3:
        raise ValueError(
            f"weno5_advection_x_halo requires halo>=3; got halo={halo}."
        )
    from legoesm.core.weno import weno5_z
    f_pad = field_pad_yxz
    # Stencil for face i+1/2: f[i-2..i+3]. Use the helper to slice.
    stencil_R = [
        _slice_axis_shift(f_pad, halo, axis=1, shift=-2),  # f[i-2]
        _slice_axis_shift(f_pad, halo, axis=1, shift=-1),  # f[i-1]
        _slice_axis_shift(f_pad, halo, axis=1, shift=0),   # f[i]
        _slice_axis_shift(f_pad, halo, axis=1, shift=1),   # f[i+1]
        _slice_axis_shift(f_pad, halo, axis=1, shift=2),   # f[i+2]
        _slice_axis_shift(f_pad, halo, axis=1, shift=3),   # f[i+3]
    ]
    fR_plus, fR_minus = weno5_z(stencil_R)

    # Face velocity at i+1/2 = 0.5 * (u[i] + u[i+1]).
    u_int = _slice_axis_shift(u_at_field_pad_yxz, halo, axis=1, shift=0)
    u_xp1 = _slice_axis_shift(u_at_field_pad_yxz, halo, axis=1, shift=1)
    u_face_R = 0.5 * (u_int + u_xp1)

    phi_R = jnp.where(u_face_R >= 0.0, fR_plus, fR_minus)
    flux_R = u_face_R * phi_R

    # Face i-1/2 is just the flux for i shifted left by one. Build
    # the same WENO reconstruction at the shifted positions.
    stencil_L = [
        _slice_axis_shift(f_pad, halo, axis=1, shift=-3),  # f[i-3]
        _slice_axis_shift(f_pad, halo, axis=1, shift=-2),  # f[i-2]
        _slice_axis_shift(f_pad, halo, axis=1, shift=-1),  # f[i-1]
        _slice_axis_shift(f_pad, halo, axis=1, shift=0),   # f[i]
        _slice_axis_shift(f_pad, halo, axis=1, shift=1),   # f[i+1]
        _slice_axis_shift(f_pad, halo, axis=1, shift=2),   # f[i+2]
    ]
    fL_plus, fL_minus = weno5_z(stencil_L)
    u_xm1 = _slice_axis_shift(u_at_field_pad_yxz, halo, axis=1, shift=-1)
    u_face_L = 0.5 * (u_xm1 + u_int)
    phi_L = jnp.where(u_face_L >= 0.0, fL_plus, fL_minus)
    flux_L = u_face_L * phi_L

    # Advective form: -d(F)/dx + f*div(u).
    f_int = _slice_axis_shift(f_pad, halo, axis=1, shift=0)
    return -(flux_R - flux_L) / dx + f_int * (u_face_R - u_face_L) / dx


def weno5_advection_y_halo(
    field_pad_yxz: jax.Array,
    v_at_field_pad_yxz: jax.Array,
    dy: float,
    halo: int = 3,
) -> jax.Array:
    """Halo-aware 5th-order WENO-Z upwind ``-v df/dy`` (axis=0)."""
    if halo < 3:
        raise ValueError(
            f"weno5_advection_y_halo requires halo>=3; got halo={halo}."
        )
    from legoesm.core.weno import weno5_z
    f_pad = field_pad_yxz
    stencil_R = [
        _slice_axis_shift(f_pad, halo, axis=0, shift=-2),
        _slice_axis_shift(f_pad, halo, axis=0, shift=-1),
        _slice_axis_shift(f_pad, halo, axis=0, shift=0),
        _slice_axis_shift(f_pad, halo, axis=0, shift=1),
        _slice_axis_shift(f_pad, halo, axis=0, shift=2),
        _slice_axis_shift(f_pad, halo, axis=0, shift=3),
    ]
    fR_plus, fR_minus = weno5_z(stencil_R)
    v_int = _slice_axis_shift(v_at_field_pad_yxz, halo, axis=0, shift=0)
    v_yp1 = _slice_axis_shift(v_at_field_pad_yxz, halo, axis=0, shift=1)
    v_face_R = 0.5 * (v_int + v_yp1)
    phi_R = jnp.where(v_face_R >= 0.0, fR_plus, fR_minus)
    flux_R = v_face_R * phi_R

    stencil_L = [
        _slice_axis_shift(f_pad, halo, axis=0, shift=-3),
        _slice_axis_shift(f_pad, halo, axis=0, shift=-2),
        _slice_axis_shift(f_pad, halo, axis=0, shift=-1),
        _slice_axis_shift(f_pad, halo, axis=0, shift=0),
        _slice_axis_shift(f_pad, halo, axis=0, shift=1),
        _slice_axis_shift(f_pad, halo, axis=0, shift=2),
    ]
    fL_plus, fL_minus = weno5_z(stencil_L)
    v_ym1 = _slice_axis_shift(v_at_field_pad_yxz, halo, axis=0, shift=-1)
    v_face_L = 0.5 * (v_ym1 + v_int)
    phi_L = jnp.where(v_face_L >= 0.0, fL_plus, fL_minus)
    flux_L = v_face_L * phi_L

    f_int = _slice_axis_shift(f_pad, halo, axis=0, shift=0)
    return -(flux_R - flux_L) / dy + f_int * (v_face_R - v_face_L) / dy


def van_leer_advection_x_halo(
    field_pad_yxz: jax.Array,
    u_at_field_pad_yxz: jax.Array,
    dx: float,
    halo: int = 2,
) -> jax.Array:
    """Halo-aware 2nd-order Van Leer TVD upwind ``-u df/dx``.

    Mirror of ``compressible_euler_plane._van_leer_advection_x``
    using slice arithmetic on halo-padded inputs. Requires
    ``halo >= 2`` (face i+1/2 stencil reaches ``f[i-1..i+2]`` and
    the i-1/2 face shifted-left reaches ``f[i-2..i+1]``).

    Single-rank equivalence: when ``layout.n_ranks == 1`` and the
    padded inputs come from ``jnp.pad(mode='wrap')``, the slices
    pick the same neighbours as the serial ``jnp.roll`` stencils →
    bit-identical to the serial Van Leer.
    """
    if halo < 2:
        raise ValueError(
            f"van_leer_advection_x_halo requires halo>=2; got halo={halo}."
        )
    from legoesm.core.flux_limiters import van_leer_face_values
    f_pad = field_pad_yxz
    # Stencils for the i+1/2 (R) and i-1/2 (L) faces. HD-1-correct r sign lives
    # in the shared helper, so serial and halo stay bit-identical.
    f_im2 = _slice_axis_shift(f_pad, halo, axis=1, shift=-2)
    f_im1 = _slice_axis_shift(f_pad, halo, axis=1, shift=-1)
    f_i   = _slice_axis_shift(f_pad, halo, axis=1, shift=0)
    f_ip1 = _slice_axis_shift(f_pad, halo, axis=1, shift=1)
    f_ip2 = _slice_axis_shift(f_pad, halo, axis=1, shift=2)
    phi_pos, phi_neg = van_leer_face_values(f_im1, f_i, f_ip1, f_ip2)       # i+1/2
    phi_pos_L, phi_neg_L = van_leer_face_values(f_im2, f_im1, f_i, f_ip1)   # i-1/2
    u_int = _slice_axis_shift(u_at_field_pad_yxz, halo, axis=1, shift=0)
    u_xp1 = _slice_axis_shift(u_at_field_pad_yxz, halo, axis=1, shift=1)
    u_face_R = 0.5 * (u_int + u_xp1)
    phi_R = jnp.where(u_face_R >= 0.0, phi_pos, phi_neg)
    flux_R = u_face_R * phi_R
    u_xm1 = _slice_axis_shift(u_at_field_pad_yxz, halo, axis=1, shift=-1)
    u_face_L = 0.5 * (u_xm1 + u_int)
    phi_L = jnp.where(u_face_L >= 0.0, phi_pos_L, phi_neg_L)
    flux_L = u_face_L * phi_L
    # Advective form.
    return -(flux_R - flux_L) / dx + f_i * (u_face_R - u_face_L) / dx


def van_leer_advection_y_halo(
    field_pad_yxz: jax.Array,
    v_at_field_pad_yxz: jax.Array,
    dy: float,
    halo: int = 2,
) -> jax.Array:
    """Halo-aware 2nd-order Van Leer TVD upwind ``-v df/dy`` (axis=0)."""
    if halo < 2:
        raise ValueError(
            f"van_leer_advection_y_halo requires halo>=2; got halo={halo}."
        )
    from legoesm.core.flux_limiters import van_leer_face_values
    f_pad = field_pad_yxz
    f_jm2 = _slice_axis_shift(f_pad, halo, axis=0, shift=-2)
    f_jm1 = _slice_axis_shift(f_pad, halo, axis=0, shift=-1)
    f_j   = _slice_axis_shift(f_pad, halo, axis=0, shift=0)
    f_jp1 = _slice_axis_shift(f_pad, halo, axis=0, shift=1)
    f_jp2 = _slice_axis_shift(f_pad, halo, axis=0, shift=2)
    phi_pos, phi_neg = van_leer_face_values(f_jm1, f_j, f_jp1, f_jp2)       # j+1/2
    phi_pos_L, phi_neg_L = van_leer_face_values(f_jm2, f_jm1, f_j, f_jp1)   # j-1/2
    v_int = _slice_axis_shift(v_at_field_pad_yxz, halo, axis=0, shift=0)
    v_yp1 = _slice_axis_shift(v_at_field_pad_yxz, halo, axis=0, shift=1)
    v_face_R = 0.5 * (v_int + v_yp1)
    phi_R = jnp.where(v_face_R >= 0.0, phi_pos, phi_neg)
    flux_R = v_face_R * phi_R
    v_ym1 = _slice_axis_shift(v_at_field_pad_yxz, halo, axis=0, shift=-1)
    v_face_L = 0.5 * (v_ym1 + v_int)
    phi_L = jnp.where(v_face_L >= 0.0, phi_pos_L, phi_neg_L)
    flux_L = v_face_L * phi_L
    return -(flux_R - flux_L) / dy + f_j * (v_face_R - v_face_L) / dy


def variable_K_diffusion_vlast_halo(
    field_pad_yxz: jax.Array,
    K_pad_yxz: jax.Array,
    grid,
    halo: int = 1,
) -> jax.Array:
    """Halo-aware conservative ``∇·(K ∇field)`` on vertical-last layout.

    Both ``field`` and ``K`` halo-padded
    ``(ny_local + 2h, nx_local + 2h, *)``. Returns interior tendency
    ``(ny_local, nx_local, *)``.
    """
    if K_pad_yxz.shape != field_pad_yxz.shape:
        raise ValueError(
            f"K shape {K_pad_yxz.shape} must equal field shape "
            f"{field_pad_yxz.shape}."
        )
    h = halo
    f_int = field_pad_yxz[h:-h, h:-h, :]
    f_xm1 = field_pad_yxz[h:-h, h - 1 : -h - 1, :]
    f_xp1 = field_pad_yxz[h:-h, h + 1 : -h + 1 if h > 1 else None, :]
    f_ym1 = field_pad_yxz[h - 1 : -h - 1, h:-h, :]
    f_yp1 = field_pad_yxz[h + 1 : -h + 1 if h > 1 else None, h:-h, :]
    K_int = K_pad_yxz[h:-h, h:-h, :]
    K_xm1 = K_pad_yxz[h:-h, h - 1 : -h - 1, :]
    K_xp1 = K_pad_yxz[h:-h, h + 1 : -h + 1 if h > 1 else None, :]
    K_ym1 = K_pad_yxz[h - 1 : -h - 1, h:-h, :]
    K_yp1 = K_pad_yxz[h + 1 : -h + 1 if h > 1 else None, h:-h, :]
    K_xface_plus = 0.5 * (K_int + K_xp1)
    K_xface_minus = 0.5 * (K_int + K_xm1)
    K_yface_plus = 0.5 * (K_int + K_yp1)
    K_yface_minus = 0.5 * (K_int + K_ym1)
    flux_xp = K_xface_plus * (f_xp1 - f_int)
    flux_xm = K_xface_minus * (f_int - f_xm1)
    flux_yp = K_yface_plus * (f_yp1 - f_int)
    flux_ym = K_yface_minus * (f_int - f_ym1)
    return (flux_xp - flux_xm) / (grid.dx ** 2) + (
        flux_yp - flux_ym
    ) / (grid.dy ** 2)
