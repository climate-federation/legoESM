"""Doubly-periodic plane non-hydrostatic compressible Euler — building blocks.

Staged-not-integrated (CRM rollout, PR2a):

* PR2a (this file) ships the *building blocks* needed by the future
  plane non-hydrostatic dycore: vertical-last operator wrappers around
  the PR1 plane stencils, a flat terrain-metric constructor, the
  dry-mass integral, and a uniform-additive mass fixer. None of these
  are wired into any factory dispatch.
* PR2b will add the full ``PlaneCompressibleEulerModel`` class, slow
  tendency, ``plane_acoustic_substeps`` (a thin wrapper around the
  shared :func:`compressible_euler._acoustic_column_kernel`), and the
  dry-validation harness (rising thermal + Straka).

State convention
----------------
All plane prognostic arrays are vertical-LAST (matches
:class:`legoesm.core.state.PlaneNonHydrostaticState` and the existing
cubed-sphere / MPAS convention) so that the shared acoustic column
kernel slices the same axis. PR1 plane operators in
:mod:`legoesm.atmosphere.dynamics.plane_operators` expect
``(..., ny, nx)`` (vertical-trailing-horizontal). The vlast wrappers
here use :func:`jax.numpy.moveaxis` to bridge the two conventions
without modifying PR1 operators.

Field unwrap policy
-------------------
The dycore exchanges :class:`legoesm.core.field.Field` objects with
its caller through :class:`PlaneNonHydrostaticState`. Inside this
module, helpers operate on raw ``jax.Array`` data (via ``Field.data``)
and the public entry points wrap the result back into ``Field``
objects via ``.replace(data=...)`` so metadata (units, dims,
staggering) is preserved. No public helper here returns a bare array.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics import plane_operators as _plane_ops
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
    _acoustic_column_kernel,
    _semi_implicit_acoustic_column_kernel,
    _sponge_profile,
    precompute_si_tridiag_bands,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.core.field import Field
from legoesm.core.state import (
    PlaneNonHydrostaticState,
    PlaneNonHydrostaticTendencies,
)
from legoesm.grids.plane import PlaneGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.split_explicit import (
    SplitExplicitConfig,
    split_explicit_step,
)


# --------------------------------------------------------------------- #
# Vertical-last operator wrappers                                       #
# --------------------------------------------------------------------- #


def _move_vertical_to_front(field_yxz: jax.Array) -> jax.Array:
    """``(ny, nx, nlev) -> (nlev, ny, nx)`` view for a PR1 operator call."""
    return jnp.moveaxis(field_yxz, -1, 0)


def _move_vertical_to_back(field_zyx: jax.Array) -> jax.Array:
    """``(nlev, ny, nx) -> (ny, nx, nlev)`` inverse of
    :func:`_move_vertical_to_front`."""
    return jnp.moveaxis(field_zyx, 0, -1)


def grad_x_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """``grad_x`` on a vertical-last ``(ny, nx, nlev)`` scalar.

    Returns x-face values with the same shape as the input. Delegates
    to :func:`plane_operators.grad_x_3d` via two
    :func:`jax.numpy.moveaxis` calls; XLA folds these into the kernel.
    """
    return _move_vertical_to_back(
        _plane_ops.grad_x_3d(_move_vertical_to_front(phi_yxz), grid)
    )


def grad_y_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """``grad_y`` on a vertical-last scalar."""
    return _move_vertical_to_back(
        _plane_ops.grad_y_3d(_move_vertical_to_front(phi_yxz), grid)
    )


def divergence_vlast(
    u_yxz: jax.Array, v_yxz: jax.Array, grid: PlaneGrid
) -> jax.Array:
    """``divergence`` of an Arakawa-C face vector field, vertical-last."""
    u_zyx = _move_vertical_to_front(u_yxz)
    v_zyx = _move_vertical_to_front(v_yxz)
    return _move_vertical_to_back(
        _plane_ops.divergence_3d(u_zyx, v_zyx, grid)
    )


def curl_vlast(
    u_yxz: jax.Array, v_yxz: jax.Array, grid: PlaneGrid
) -> jax.Array:
    """``curl`` at SW corners, vertical-last."""
    u_zyx = _move_vertical_to_front(u_yxz)
    v_zyx = _move_vertical_to_front(v_yxz)
    return _move_vertical_to_back(
        _plane_ops.curl_3d(u_zyx, v_zyx, grid)
    )


def laplacian_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """5-point scalar Laplacian, vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.laplacian_3d(_move_vertical_to_front(phi_yxz), grid)
    )


# --------------------------------------------------------------------- #
# Two-point centered face/cell interpolations (vertical-last).          #
# Re-added after main merge dropped them; required by                   #
# plane_compressible_euler_slow_tendencies (non-halo path).             #
# Periodic in both horizontal axes (jnp.roll wrap).                     #
# --------------------------------------------------------------------- #


def interp_cell_to_xface_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """Cell-centre → x-face: average phi[i,j] with phi[i-1,j] (axis=1)."""
    return 0.5 * (phi_yxz + jnp.roll(phi_yxz, 1, axis=1))


def interp_cell_to_yface_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """Cell-centre → y-face: average phi[i,j] with phi[i,j-1] (axis=0)."""
    return 0.5 * (phi_yxz + jnp.roll(phi_yxz, 1, axis=0))


def interp_xface_to_cell_vlast(u_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """x-face → cell-centre: average u[i,j] with u[i+1,j]."""
    return 0.5 * (u_yxz + jnp.roll(u_yxz, -1, axis=1))


def interp_yface_to_cell_vlast(v_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """y-face → cell-centre: average v[i,j] with v[i,j+1]."""
    return 0.5 * (v_yxz + jnp.roll(v_yxz, -1, axis=0))


def interp_yface_to_xface_vlast(v_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """y-face → x-face (4-point corner average)."""
    v_cell = interp_yface_to_cell_vlast(v_yxz, grid)
    return interp_cell_to_xface_vlast(v_cell, grid)


def interp_xface_to_yface_vlast(u_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """x-face → y-face (4-point corner average)."""
    u_cell = interp_xface_to_cell_vlast(u_yxz, grid)
    return interp_cell_to_yface_vlast(u_cell, grid)


# --------------------------------------------------------------------- #
# Flat terrain metric for a plane                                       #
# --------------------------------------------------------------------- #


def make_flat_plane_terrain_metric(
    grid: PlaneGrid, height_coord: HeightCoordinate
) -> TerrainMetric:
    """Construct a ``TerrainMetric`` for a flat-surface plane.

    On a flat plane the surface elevation ``z_s`` is zero, so the
    Jacobian ``J = (H - z_s) / H`` is identically ``1`` and the
    physical-z arrays equal the z-star arrays at every horizontal
    cell.

    Parameters
    ----------
    grid : PlaneGrid
        Provides ``ny``, ``nx``, ``area_T.dtype``.
    height_coord : HeightCoordinate
        Supplies ``z_full``, ``z_half``, ``n_levels``.

    Returns
    -------
    TerrainMetric
        ``z_s = 0``, ``jacobian = 1`` (shape ``(ny, nx)``), and
        broadcast 3D arrays at full and half levels.
    """
    dtype = grid.area_T.dtype
    z_s = jnp.zeros((grid.ny, grid.nx), dtype=dtype)
    jacobian = jnp.ones((grid.ny, grid.nx), dtype=dtype)
    z_full_3d = (
        jnp.ones((grid.ny, grid.nx, 1), dtype=dtype)
        * height_coord.z_full.astype(dtype)
    )
    z_half_3d = (
        jnp.ones((grid.ny, grid.nx, 1), dtype=dtype)
        * height_coord.z_half.astype(dtype)
    )
    return TerrainMetric(
        z_s=z_s,
        jacobian=jacobian,
        z_full_3d=z_full_3d,
        z_half_3d=z_half_3d,
    )


def _assert_flat_terrain(
    terrain_metric: TerrainMetric, tol: float = 1.0e-10
) -> None:
    """Raise ``ValueError`` if ``terrain_metric`` is not flat.

    Host-only check: converts the Jacobian to a NumPy scalar so the
    boolean test happens on the host, never under JIT tracing. PR2a
    only supports flat terrain on the plane; sloped terrain support
    lands in a follow-up PR together with the metric-consistent mass
    integral.
    """
    import numpy as np
    j = np.asarray(terrain_metric.jacobian)
    z_s = np.asarray(terrain_metric.z_s)
    if not np.allclose(j, 1.0, atol=tol):
        raise ValueError(
            "Plane PR2a requires flat terrain (Jacobian == 1 everywhere); "
            f"got max|J - 1| = {float(np.max(np.abs(j - 1.0)))}"
        )
    if not np.allclose(z_s, 0.0, atol=tol):
        raise ValueError(
            "Plane PR2a requires flat terrain (z_s == 0 everywhere); "
            f"got max|z_s| = {float(np.max(np.abs(z_s)))}"
        )


# --------------------------------------------------------------------- #
# Dry-air mass integral + uniform-additive mass fixer                   #
# --------------------------------------------------------------------- #


def compute_dry_mass_plane(
    state: PlaneNonHydrostaticState,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
) -> jax.Array:
    """Discrete dry-air mass on the plane.

    ``M = sum_{j,i,k} (rho_ref[k] + rho_prime[j,i,k]) * J[j,i] *
    area_T[j,i] * dz[k]``

    The terrain Jacobian ``J`` is uniformly ``1`` for a flat plane,
    but the formula is written generically so the mass fixer remains
    correct when sloped terrain support is added in a future PR.

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        Provides ``rho_prime.data`` of shape ``(ny, nx, nlev)``.
    grid : PlaneGrid
        Provides ``area_T`` of shape ``(ny, nx)``.
    height_coord : HeightCoordinate
        Provides ``rho_ref`` of shape ``(nlev,)`` and ``dz`` of shape
        ``(nlev,)``.
    terrain_metric : TerrainMetric
        Provides ``jacobian`` of shape ``(ny, nx)``.

    Returns
    -------
    jax.Array
        Scalar total dry-air mass in the natural model units
        (``rho`` * ``area`` * ``dz``).
    """
    rho_p = state.rho_prime.data
    rho_total = height_coord.rho_ref + rho_p   # broadcast (nlev,) → (ny,nx,nlev)
    weight_horizontal = (
        terrain_metric.jacobian * grid.area_T
    )[:, :, None]                              # (ny, nx, 1)
    cell_mass = rho_total * weight_horizontal * height_coord.dz
    return jnp.sum(cell_mass)


def _plane_volume_weight(
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
) -> jax.Array:
    """Total weighted volume ``sum(J * area_T * dz)`` used by the fixer."""
    weight_horizontal = (
        terrain_metric.jacobian * grid.area_T
    )[:, :, None]
    return jnp.sum(weight_horizontal * height_coord.dz)


def fix_mass_nonhydrostatic_plane(
    state: PlaneNonHydrostaticState,
    target_mass: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
) -> PlaneNonHydrostaticState:
    """Restore dry mass to ``target_mass`` via a uniform additive
    correction to ``rho_prime``.

    The correction is

        delta = (target_mass - current_mass) / sum(J * area_T * dz)
        rho_prime' = rho_prime + delta

    Adding a spatially uniform ``delta`` preserves every spatial
    gradient of ``rho_prime`` used in the slow-tendency assembly, so
    momentum / theta / w tendencies are unchanged by the fixer. This
    matches the MPAS pattern in
    :func:`compressible_euler_mpas.fix_mass_nonhydrostatic_mpas`.

    Differentiability
    -----------------
    The fixer is a pure additive correction with no ``jnp.where`` /
    clip / branch, so ``jax.grad`` flows through cleanly. The global
    sum is differentiable in JAX.

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        State whose ``rho_prime`` is to be corrected.
    target_mass : jax.Array
        Scalar target total dry-air mass.
    grid, height_coord, terrain_metric : as in
        :func:`compute_dry_mass_plane`.

    Returns
    -------
    PlaneNonHydrostaticState
        Same state with ``rho_prime`` shifted by ``delta`` everywhere.
    """
    current = compute_dry_mass_plane(state, grid, height_coord, terrain_metric)
    weighted_volume = _plane_volume_weight(grid, height_coord, terrain_metric)
    delta = (target_mass - current) / weighted_volume
    rho_p_new = state.rho_prime.data + delta
    return PlaneNonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w,
        theta_prime=state.theta_prime,
        rho_prime=state.rho_prime.replace(data=rho_p_new),
        phis=state.phis,
        tracers=state.tracers,
    )


# --------------------------------------------------------------------- #
# Convenience: rest state on the plane                                  #
# --------------------------------------------------------------------- #


def make_rest_state(
    grid: PlaneGrid, height_coord: HeightCoordinate, dtype=None
) -> PlaneNonHydrostaticState:
    """Build a hydrostatically balanced rest state on the plane.

    All perturbations and velocities are zero. Useful as the starting
    point for rest-state preservation tests in PR2a and as the
    background state for benchmark initial conditions in PR2b.
    """
    if dtype is None:
        dtype = grid.area_T.dtype
    ny, nx = grid.ny, grid.nx
    nlev = height_coord.n_levels
    zeros_3d = jnp.zeros((ny, nx, nlev), dtype=dtype)
    zeros_w = jnp.zeros((ny, nx, nlev + 1), dtype=dtype)
    zeros_2d = jnp.zeros((ny, nx), dtype=dtype)
    zeros_tracers = jnp.zeros((ny, nx, nlev, 0), dtype=dtype)
    return PlaneNonHydrostaticState(
        u=Field(zeros_3d, name="u", dims=("y", "x", "z"), units="m/s",
                staggering="edge"),
        v=Field(zeros_3d, name="v", dims=("y", "x", "z"), units="m/s",
                staggering="edge"),
        w=Field(zeros_w, name="w", dims=("y", "x", "z_half"),
                units="m/s", staggering="cell"),
        theta_prime=Field(
            zeros_3d, name="theta_prime", dims=("y", "x", "z"),
            units="K", staggering="cell",
        ),
        rho_prime=Field(
            zeros_3d, name="rho_prime", dims=("y", "x", "z"),
            units="kg/m^3", staggering="cell",
        ),
        phis=Field(zeros_2d, name="phis", dims=("y", "x"),
                   units="m^2/s^2", staggering="cell"),
        tracers=Field(
            zeros_tracers, name="tracers",
            dims=("y", "x", "z", "tracer"), units="kg/kg",
            staggering="cell",
        ),
    )


# --------------------------------------------------------------------- #
# Horizontal advection scheme registry — single source of truth for     #
# valid scheme names + their halo-width requirements. iter-186 added    #
# this map after Codex flagged that --use-dd + --advection van_leer     #
# crashed because the driver constructed a halo-1 layout while the     #
# halo dispatch requires halo>=2 for van_leer (and >=3 for weno5).     #
# Both the production driver (run_rce_mpi_long.py) and the halo       #
# dispatch (compressible_euler_plane_halo.py) consult this so the      #
# minimum-halo contract cannot drift between caller and callee.        #
#                                                                       #
# Stencil widths:                                                       #
#   upwind1  — 2-cell (i-1, i+1) → halo 1.                              #
#   van_leer — 4-cell (i-1, i, i+1, i+2) for i+1/2 face; halo 2.        #
#   weno5    — 6-cell (i-2..i+3) for i+1/2; halo 3.                     #
# --------------------------------------------------------------------- #
HORIZONTAL_ADVECTION_HALO_REQUIREMENT: dict[str, int] = {
    "upwind1": 1,
    "van_leer": 2,
    "weno5": 3,
}


# --------------------------------------------------------------------- #
# First-order upwind on the Arakawa-C grid                              #
# --------------------------------------------------------------------- #


def _upwind_advection_x(field_yxz: jax.Array, u_at_field: jax.Array,
                        dx: float) -> jax.Array:
    """First-order upwind contribution to ``-u df/dx``.

    ``field`` and ``u_at_field`` must live at the SAME horizontal
    location (a Arakawa-C-grid invariant). For each point pick the
    one-sided difference on the side the local advector flows from::

        out = -max(u, 0) * (f[i] - f[i-1]) / dx
              - min(u, 0) * (f[i+1] - f[i]) / dx

    Periodic neighbours via ``jnp.roll`` (axis=1 = x for the
    ``(ny, nx, nlev)`` vlast layout). Upwind biasing damps the
    2-Δx dispersive mode triggered by sharp plume gradients.

    Differentiability
    -----------------
    ``jnp.maximum`` / ``jnp.minimum`` against ``0.0`` introduce a
    kink at ``u = 0``. JAX returns a finite subgradient at the kink,
    so ``jax.grad`` flows cleanly even through zero-velocity cells.
    """
    f_backward = (field_yxz - jnp.roll(field_yxz, 1, axis=1)) / dx
    f_forward = (jnp.roll(field_yxz, -1, axis=1) - field_yxz) / dx
    u_pos = jnp.maximum(u_at_field, 0.0)
    u_neg = jnp.minimum(u_at_field, 0.0)
    return -(u_pos * f_backward + u_neg * f_forward)


def _upwind_advection_y(field_yxz: jax.Array, v_at_field: jax.Array,
                        dy: float) -> jax.Array:
    """First-order upwind contribution to ``-v df/dy``. Same convention
    as :func:`_upwind_advection_x`: ``field`` and ``v_at_field`` must
    co-locate."""
    f_backward = (field_yxz - jnp.roll(field_yxz, 1, axis=0)) / dy
    f_forward = (jnp.roll(field_yxz, -1, axis=0) - field_yxz) / dy
    v_pos = jnp.maximum(v_at_field, 0.0)
    v_neg = jnp.minimum(v_at_field, 0.0)
    return -(v_pos * f_backward + v_neg * f_forward)


# --------------------------------------------------------------------- #
# Van Leer TVD flux-form advection (2nd-order, monotone, bounded).      #
# Drop-in replacement for _upwind_advection_x/y with the same           #
# co-located (field, velocity) convention. Cheaper than WENO5 (stencil  #
# width 4 vs 6) but still 2nd-order accurate in smooth regions; falls   #
# back to 1st-order upwind at extrema. Less numerical diffusion than    #
# upwind1 → relaxes the advective CFL bound and lets larger dt run      #
# stably (iter-178 motivation).                                         #
# --------------------------------------------------------------------- #


def _van_leer_advection_x(
    field_yxz: jax.Array, u_at_field: jax.Array, dx: float,
) -> jax.Array:
    """2nd-order Van Leer TVD upwind contribution to ``-u df/dx``.

    Reconstruct field at face i+1/2 from a 4-cell stencil
    [f[i-1], f[i], f[i+1], f[i+2]] using a slope-limited 2nd-order
    extrapolation. Upwind side selected by face velocity sign.
    Convert flux-form divergence to advective form via the
    ``-d(uf)/dx + f·du/dx`` identity (same trick as WENO5).

    Face velocity is the 2-point centred average of ``u_at_field``
    (consistent with the WENO5 path).

    Differentiability: Van Leer's analytic ``(r+|r|)/(1+|r|)`` is
    smooth everywhere except a single subgradient kink at ``r=0``
    (an extremum). ``jnp.where`` for sign selection contributes a
    second well-behaved subgradient. ``jax.grad`` flows cleanly.
    """
    from legoesm.core.flux_limiters import van_leer_limiter
    eps = 1e-30
    f = field_yxz
    f_im1 = jnp.roll(f,  1, axis=1)  # f[i-1]
    f_i   = f                         # f[i]
    f_ip1 = jnp.roll(f, -1, axis=1)  # f[i+1]
    f_ip2 = jnp.roll(f, -2, axis=1)  # f[i+2]
    # Positive-velocity reconstruction at face i+1/2 (upwind from left).
    # delta = f[i+1] - f[i]; r = (f[i] - f[i-1]) / delta.
    delta_pos = f_ip1 - f_i
    r_pos = (f_i - f_im1) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    phi_pos = f_i + 0.5 * van_leer_limiter(r_pos) * delta_pos
    # Negative-velocity reconstruction at face i+1/2 (upwind from right).
    # delta = f[i] - f[i+1]; r = (f[i+2] - f[i+1]) / delta.
    delta_neg = f_i - f_ip1
    r_neg = (f_ip2 - f_ip1) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    phi_neg = f_ip1 + 0.5 * van_leer_limiter(r_neg) * delta_neg
    # Face velocity: 2-point centred average of co-located cell velocities.
    u_face_R = 0.5 * (u_at_field + jnp.roll(u_at_field, -1, axis=1))
    phi_R = jnp.where(u_face_R >= 0.0, phi_pos, phi_neg)
    flux_R = u_face_R * phi_R
    flux_L = jnp.roll(flux_R, 1, axis=1)
    u_face_L = jnp.roll(u_face_R, 1, axis=1)
    return -(flux_R - flux_L) / dx + f * (u_face_R - u_face_L) / dx


def _van_leer_advection_y(
    field_yxz: jax.Array, v_at_field: jax.Array, dy: float,
) -> jax.Array:
    """2nd-order Van Leer TVD upwind contribution to ``-v df/dy``
    (axis=0). Same convention as :func:`_van_leer_advection_x`."""
    from legoesm.core.flux_limiters import van_leer_limiter
    eps = 1e-30
    f = field_yxz
    f_jm1 = jnp.roll(f,  1, axis=0)
    f_j   = f
    f_jp1 = jnp.roll(f, -1, axis=0)
    f_jp2 = jnp.roll(f, -2, axis=0)
    delta_pos = f_jp1 - f_j
    r_pos = (f_j - f_jm1) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    phi_pos = f_j + 0.5 * van_leer_limiter(r_pos) * delta_pos
    delta_neg = f_j - f_jp1
    r_neg = (f_jp2 - f_jp1) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    phi_neg = f_jp1 + 0.5 * van_leer_limiter(r_neg) * delta_neg
    v_face_R = 0.5 * (v_at_field + jnp.roll(v_at_field, -1, axis=0))
    phi_R = jnp.where(v_face_R >= 0.0, phi_pos, phi_neg)
    flux_R = v_face_R * phi_R
    flux_L = jnp.roll(flux_R, 1, axis=0)
    v_face_L = jnp.roll(v_face_R, 1, axis=0)
    return -(flux_R - flux_L) / dy + f * (v_face_R - v_face_L) / dy


# --------------------------------------------------------------------- #
# WENO5-Z flux-form upwind advection (5th-order, monotone).             #
# Drop-in replacements for _upwind_advection_x/y with the same          #
# co-located (field, velocity) convention. Periodic in both axes        #
# via jnp.roll. Stencil width 6; needs only single-rank periodic        #
# wrap (no halo exchange on a doubly-periodic plane).                   #
# --------------------------------------------------------------------- #


def _weno5_advection_x(
    field_yxz: jax.Array, u_at_field: jax.Array, dx: float,
) -> jax.Array:
    """5th-order WENO-Z upwind contribution to ``-u df/dx``.

    Computes the face flux ``F_{i+1/2} = u_face · phi_face`` where
    ``phi_face`` is the WENO5-Z left-/right-biased reconstruction
    selected by the sign of the face velocity. The conservative flux
    divergence is corrected by ``phi * div(u)`` so the returned tendency
    is advective form ``-u dphi/dx``, matching
    :func:`_upwind_advection_x`.

    Face velocity uses a 2-point centred average of ``u_at_field``
    (consistent with the co-located upwind it replaces). At any face
    where ``u_face = 0`` the average of the two reconstructions is
    used so the scheme stays smooth across zero crossings.
    """
    from legoesm.core.weno import weno5_z
    f = field_yxz
    # Stencil for face i+1/2: [f[i-2], f[i-1], f[i], f[i+1], f[i+2], f[i+3]]
    stencil_R = [
        jnp.roll(f,  2, axis=1),
        jnp.roll(f,  1, axis=1),
        f,
        jnp.roll(f, -1, axis=1),
        jnp.roll(f, -2, axis=1),
        jnp.roll(f, -3, axis=1),
    ]
    fR_plus, fR_minus = weno5_z(stencil_R)  # at face i+1/2
    # Face velocity = arithmetic average of co-located cell velocities.
    u_face_R = 0.5 * (u_at_field + jnp.roll(u_at_field, -1, axis=1))
    # Upwind selection.
    phi_R = jnp.where(u_face_R >= 0.0, fR_plus, fR_minus)
    flux_R = u_face_R * phi_R
    # Face i-1/2 is just the rolled face i+1/2 of the previous cell.
    flux_L = jnp.roll(flux_R, 1, axis=1)
    u_face_L = jnp.roll(u_face_R, 1, axis=1)
    return -(flux_R - flux_L) / dx + f * (u_face_R - u_face_L) / dx


def _weno5_advection_y(
    field_yxz: jax.Array, v_at_field: jax.Array, dy: float,
) -> jax.Array:
    """5th-order WENO-Z upwind contribution to ``-v df/dy`` (axis=0)."""
    from legoesm.core.weno import weno5_z
    f = field_yxz
    stencil_R = [
        jnp.roll(f,  2, axis=0),
        jnp.roll(f,  1, axis=0),
        f,
        jnp.roll(f, -1, axis=0),
        jnp.roll(f, -2, axis=0),
        jnp.roll(f, -3, axis=0),
    ]
    fR_plus, fR_minus = weno5_z(stencil_R)
    v_face_R = 0.5 * (v_at_field + jnp.roll(v_at_field, -1, axis=0))
    phi_R = jnp.where(v_face_R >= 0.0, fR_plus, fR_minus)
    flux_R = v_face_R * phi_R
    flux_L = jnp.roll(flux_R, 1, axis=0)
    v_face_L = jnp.roll(v_face_R, 1, axis=0)
    return -(flux_R - flux_L) / dy + f * (v_face_R - v_face_L) / dy


def _variable_K_diffusion_vlast(
    field_yxz: jax.Array, K_yxz: jax.Array, grid: PlaneGrid,
) -> jax.Array:
    """Conservative variable-coefficient diffusion ``∇·(K ∇field)``.

    Discrete flux-form on a uniform doubly-periodic plane:

        ∂_x(K ∂_x f)|_i ≈ (K_{i+1/2}(f_{i+1}-f_i) - K_{i-1/2}(f_i-f_{i-1})) / dx²

    where ``K_{i±1/2} = 0.5(K_i + K_{i±1})`` is the K average to the
    face. The same in y, summed. Conservative: ``sum(out * area) =
    0`` under periodic BC. Dissipative: ``sum(f * out * area) ≤ 0``
    when ``K ≥ 0`` (because the form is ``-sum_faces K |grad f|²`` via
    discrete integration by parts — verified in
    :func:`tests/unit/test_plane_nh_smagorinsky.py::test_smag_tendency_is_dissipative`).

    Replaces the naive ``K * laplacian(f)`` which is not conservative
    when ``K`` is spatially variable.

    Parameters
    ----------
    field_yxz : jax.Array
        The diffused field. Any shape ``(ny, nx, *)`` works.
    K_yxz : jax.Array
        Diffusion coefficient. **Must match ``field_yxz.shape``** —
        the caller is responsible for interpolating K to the right
        vertical layout (e.g. half-level K for half-level w).
    """
    if K_yxz.shape != field_yxz.shape:
        raise ValueError(
            f"K_yxz shape {K_yxz.shape} must equal field_yxz shape "
            f"{field_yxz.shape}; pre-interpolate K to the right "
            "vertical layout (full level vs half level)."
        )
    # Face-averaged K (axis 1 = nx, axis 0 = ny in vlast layout).
    K_xface_plus = 0.5 * (K_yxz + jnp.roll(K_yxz, -1, axis=1))
    K_xface_minus = 0.5 * (K_yxz + jnp.roll(K_yxz, 1, axis=1))
    K_yface_plus = 0.5 * (K_yxz + jnp.roll(K_yxz, -1, axis=0))
    K_yface_minus = 0.5 * (K_yxz + jnp.roll(K_yxz, 1, axis=0))

    flux_xp = K_xface_plus * (jnp.roll(field_yxz, -1, axis=1) - field_yxz)
    flux_xm = K_xface_minus * (field_yxz - jnp.roll(field_yxz, 1, axis=1))
    flux_yp = K_yface_plus * (jnp.roll(field_yxz, -1, axis=0) - field_yxz)
    flux_ym = K_yface_minus * (field_yxz - jnp.roll(field_yxz, 1, axis=0))

    return (flux_xp - flux_xm) / (grid.dx ** 2) + (
        flux_yp - flux_ym
    ) / (grid.dy ** 2)


def _safe_sqrt_strain(strain_mag_sq: jax.Array) -> jax.Array:
    """``sqrt(strain_mag_sq)`` with AD-safe ``d/dx sqrt(0) = 0``.

    Forward: zero where input is zero, ``sqrt(input)`` elsewhere
    (preserves bit-exact zero on the rest state). Backward:
    evaluates the gradient at a safe positive argument so
    ``1/(2*sqrt)`` stays finite, then masks the zero-strain
    contribution. Mirrors the double-where trick in
    :func:`legoesm.core._smagorinsky_visc.compute_smagorinsky_ah_2d`.
    """
    safe = jnp.where(strain_mag_sq > 0.0, strain_mag_sq, 1.0)
    return jnp.where(strain_mag_sq > 0.0, jnp.sqrt(safe), 0.0)


def _compute_smagorinsky_K_m_plane(
    u_yxz: jax.Array,
    v_yxz: jax.Array,
    w_yxz_half: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    c_s: float,
) -> jax.Array:
    """Full 3D Smagorinsky-Lilly eddy viscosity on the C-grid plane.

    ``K_m = (C_s · Δ)² · |S|`` with isotropic mixing length
    ``Δ = (dx · dy · dz)^(1/3)`` and FULL strain-rate magnitude
    ``|S| = sqrt(2 S_ij S_ij)`` including vertical-shear components
    ``S_13``, ``S_23``, ``S_33``. Returned at cell centres.

    Energy-consistent C-grid evaluation
    -----------------------------------
    Each strain component is evaluated at its NATURAL Arakawa-C
    location, then ``S_ij²`` is interpolated to cell centres so the
    aggregated ``|S|²`` lives where ``K_m`` does:

    * ``S11 = ∂u/∂x`` at cell centre (u face-difference along x).
    * ``S22 = ∂v/∂y`` at cell centre.
    * ``S33 = ∂w/∂z`` at cell centre (w half-level difference).
    * ``S12 = 0.5(∂u/∂y + ∂v/∂x)`` at SW corner via the existing
      :func:`plane_operators.curl_3d`-style stencil, then averaged
      to cell centre.
    * ``S13 = 0.5(∂u/∂z + ∂w/∂x)`` at x-face / vertical full level;
      averaged to cell centre.
    * ``S23 = 0.5(∂v/∂z + ∂w/∂y)`` at y-face / vertical full level;
      averaged to cell centre.

    No A-grid simplification anywhere — every gradient uses the
    proper face-difference on the Arakawa-C state, replacing the
    horizontal-only pilot whose ``K_m`` was identically zero under
    pure vertical shear (Codex review request 2026-05-24).

    Parameters
    ----------
    u_yxz : jax.Array
        x-face zonal wind, shape ``(ny, nx, nlev)``.
    v_yxz : jax.Array
        y-face meridional wind, shape ``(ny, nx, nlev)``.
    w_yxz_half : jax.Array
        Half-level vertical velocity, shape ``(ny, nx, nlev+1)``.
    grid : PlaneGrid
    height_coord : HeightCoordinate
        Provides ``dz`` (per-level full-level thickness, shape
        ``(nlev,)``) and ``dz_half`` (interface-to-interface
        spacing, shape ``(nlev-1,)``) for the vertical gradients.
    c_s : float

    Returns
    -------
    K_m : jax.Array
        Eddy viscosity at cell centres, shape ``(ny, nx, nlev)``.
    """
    nlev = u_yxz.shape[-1]

    # --- Horizontal C-grid gradients ---
    # u at x-face → ∂u/∂x = (u[..., j, i+1] - u[..., j, i]) / dx at cell centre.
    du_dx_center = (jnp.roll(u_yxz, -1, axis=1) - u_yxz) / grid.dx
    # v at y-face → ∂v/∂y = (v[..., j+1, i] - v[..., j, i]) / dy at cell centre.
    dv_dy_center = (jnp.roll(v_yxz, -1, axis=0) - v_yxz) / grid.dy

    # --- S12 at SW corner ---
    # ∂u/∂y at corner = (u[..., j, i] - u[..., j-1, i]) / dy
    du_dy_corner = (u_yxz - jnp.roll(u_yxz, 1, axis=0)) / grid.dy
    # ∂v/∂x at corner = (v[..., j, i] - v[..., j, i-1]) / dx
    dv_dx_corner = (v_yxz - jnp.roll(v_yxz, 1, axis=1)) / grid.dx
    S12_corner = 0.5 * (du_dy_corner + dv_dx_corner)
    # Corner→centre average (SW + SE + NW + NE).
    S12_sq_center = 0.25 * (
        S12_corner ** 2
        + jnp.roll(S12_corner, -1, axis=1) ** 2
        + jnp.roll(S12_corner, -1, axis=0) ** 2
        + jnp.roll(jnp.roll(S12_corner, -1, axis=0), -1, axis=1) ** 2
    )

    # --- Vertical gradients (w at half levels, u/v at full levels) ---
    # ∂w/∂z = (w_half[k+1] - w_half[k]) / dz_full[k] at full level (cell centre).
    dz_full = height_coord.dz                          # (nlev,)
    dw_dz_center = (
        w_yxz_half[..., 1:] - w_yxz_half[..., :-1]
    ) / dz_full
    S33_center = dw_dz_center

    # ∂u/∂z at x-face, vertical full-level: needs interior centred
    # difference of u between full levels k+1, k-1 (centred). Edges
    # use one-sided one-level differences.
    # Build du/dz_full at x-face (same staggering as u).
    du_dz = _full_level_centred_d_dz(u_yxz, height_coord)
    dv_dz = _full_level_centred_d_dz(v_yxz, height_coord)

    # ∂w/∂x at x-face (cell-centre w_full needed first), ∂w/∂y at y-face.
    # Build w at full level (vertical midpoint of half-level pair) then
    # take face-difference along x / y → result lives at x-face/y-face.
    w_full = 0.5 * (w_yxz_half[..., :-1] + w_yxz_half[..., 1:])  # cell centre
    dw_dx_xface = (w_full - jnp.roll(w_full, 1, axis=1)) / grid.dx
    dw_dy_yface = (w_full - jnp.roll(w_full, 1, axis=0)) / grid.dy

    S13_xface = 0.5 * (du_dz + dw_dx_xface)            # at x-face
    S23_yface = 0.5 * (dv_dz + dw_dy_yface)            # at y-face

    # Face → centre via simple two-point average on the appropriate axis.
    S13_sq_center = 0.5 * (
        S13_xface ** 2 + jnp.roll(S13_xface, -1, axis=1) ** 2
    )
    S23_sq_center = 0.5 * (
        S23_yface ** 2 + jnp.roll(S23_yface, -1, axis=0) ** 2
    )

    # --- Aggregate strain magnitude at cell centre ---
    # |S|² = 2 S_ij S_ij = 2 (S11² + S22² + S33² + 2 S12² + 2 S13² + 2 S23²)
    strain_mag_sq = 2.0 * (
        du_dx_center ** 2
        + dv_dy_center ** 2
        + S33_center ** 2
        + 2.0 * S12_sq_center
        + 2.0 * S13_sq_center
        + 2.0 * S23_sq_center
    )
    strain_mag = _safe_sqrt_strain(strain_mag_sq)

    # LES boundary-layer mixing length:
    #   l_m = min(c_s · Δ, κ · z)
    # — caps the Smagorinsky length-scale at the von Kármán
    # wall-distance scaling so K_m → 0 at the surface and the
    # log-layer mean velocity profile is recovered under uniform
    # shear (Mason 1989, Pope 2000 §10.4). Without this cap, ``K_m``
    # at the first cell ≈ (c_s · Δ)² · |S|, which gives a SHEAR
    # STRESS far too large at the surface and a quartic-in-z
    # spurious near-wall acceleration. ``z_full`` is positive
    # height above the surface (``z_half[-1] = 0`` so
    # ``z_full[nlev-1] = 0.5 · dz_sfc`` at the lowest centre).
    # ``κ`` is the von Kármán constant from
    # :mod:`legoesm.constants` so callers wanting a different
    # convention (e.g. 0.41) update the central definition once.
    from legoesm import constants
    delta = (grid.dx * grid.dy * dz_full) ** (1.0 / 3.0)   # (nlev,)
    l_smag = c_s * delta                                   # (nlev,)
    l_wall = constants.kappa_von_karman * height_coord.z_full   # (nlev,)
    l_m = jnp.minimum(l_smag, l_wall)
    l_m_sq = l_m ** 2                                      # (nlev,)
    return l_m_sq * strain_mag


def _full_level_centred_d_dz(
    field_yxz: jax.Array, height_coord: HeightCoordinate
) -> jax.Array:
    """Vertical derivative ``∂f/∂(level index)`` at full level.

    Uses :attr:`HeightCoordinate.dz_half` which IS the
    full-level-to-full-level distance (``dz_half[k] =
    z_full[k] - z_full[k+1]``), positive under the top-to-bottom
    storage order. The denominator for the interior centred
    difference is therefore ``dz_half[k-1] + dz_half[k] =
    z_full[k-1] - z_full[k+1]``, the total distance between
    ``f[k-1]`` and ``f[k+1]``.

    Top-down convention
    -------------------
    ``z_full`` decreases with level index ``k`` (index 0 = model
    top, index ``nlev-1`` = surface), so the returned value is
    ``+(f[k+1] - f[k-1]) / (z[k-1] - z[k+1])``, i.e. the derivative
    with respect to LEVEL INDEX. The sign relative to the physical
    vertical coordinate ``z`` is opposite; this helper is consumed
    exclusively inside SQUARED strain components ``S_13²``,
    ``S_23²`` so the sign drops out of ``|S|²``. Do NOT use the raw
    return value where a signed ``∂f/∂z_physical`` is required
    without flipping the sign.

    Interior uses the 2-level centred difference; boundaries fall
    back to one-sided one-step differences (top: ``(f[0] - f[1]) /
    dz_half[0]`` written in level-index direction; bottom analogous).
    Result has the same shape as the input.
    """
    nlev = field_yxz.shape[-1]
    if nlev < 2:
        return jnp.zeros_like(field_yxz)
    dz_half = height_coord.dz_half                          # (nlev-1,)
    interior = (
        field_yxz[..., 2:] - field_yxz[..., :-2]
    ) / (dz_half[:-1] + dz_half[1:])
    bottom = (field_yxz[..., 1:2] - field_yxz[..., 0:1]) / dz_half[0]
    top = (field_yxz[..., -1:] - field_yxz[..., -2:-1]) / dz_half[-1]
    return jnp.concatenate([bottom, interior, top], axis=-1)


def _vertical_advection_plane(
    field_yxz: jax.Array,
    w_yxz_half: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
) -> jax.Array:
    """``-(w/J) df/dz`` at cell centres, periodic over the horizontal axes.

    Parameters
    ----------
    field_yxz : jax.Array
        Prognostic at full levels, shape ``(ny, nx, nlev)``.
    w_yxz_half : jax.Array
        Vertical velocity at interface levels, shape ``(ny, nx, nlev+1)``.
        ``w[..., 0]`` and ``w[..., -1]`` are the rigid boundaries.
    height_coord : HeightCoordinate
        Provides ``dz_half`` for the centred vertical gradient.
    J : jax.Array
        Jacobian of shape ``(ny, nx)`` (broadcasts over the level axis).

    Returns
    -------
    jax.Array
        Vertical advection tendency, shape ``(ny, nx, nlev)``.

    Notes
    -----
    Mirrors :func:`compressible_euler_mpas._vertical_advection_height_1d`
    but works on the ``(ny, nx, nlev)`` plane layout without a flatten
    / reshape pair.
    """
    dz_half = height_coord.dz_half  # (nlev-1,)
    nlev = field_yxz.shape[-1]
    w_full = 0.5 * (w_yxz_half[..., :-1] + w_yxz_half[..., 1:])  # (ny, nx, nlev)

    if nlev > 2:
        inner = (field_yxz[..., :-2] - field_yxz[..., 2:]) / (
            dz_half[:-1] + dz_half[1:]
        )
        pad_axes = ((0, 0),) * (field_yxz.ndim - 1)
        df_dz = jnp.pad(inner, (*pad_axes, (1, 1)))
    else:
        df_dz = jnp.zeros_like(field_yxz)

    return -w_full / J[:, :, None] * df_dz


# --------------------------------------------------------------------- #
# Slow tendencies                                                       #
# --------------------------------------------------------------------- #


def plane_compressible_euler_slow_tendencies(
    state: PlaneNonHydrostaticState,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CompressibleEulerConfig,
) -> PlaneNonHydrostaticTendencies:
    """Slow (advective + PG + Coriolis) tendencies for the plane NH dycore.

    Evaluated once per outer SSP-RK3 stage and held constant for the
    acoustic substeps. See the module docstring for the equations
    being solved.

    Energy-consistent C-grid pairing
    --------------------------------
    All horizontal differential operators consume ``u`` at the
    x-face (between cells ``i-1`` and ``i``), ``v`` at the y-face,
    and scalars at cell centres — the Arakawa-C convention encoded
    by :mod:`plane_operators`. The PG / divergence pairing is the
    discrete adjoint pair documented in that module's
    "Inner-product weights and adjoint pairing" docstring:
    ``sum(phi · div(u, v)) == -sum(u · grad_x(phi)) -
    sum(v · grad_y(phi))`` to machine epsilon under periodic BC,
    which is the discrete condition for energy-consistent
    pressure-gradient / divergence coupling. No A-grid
    simplification anywhere in the body (replaces the PR2b
    centred-difference shortcut).

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        Prognostic state at the start of the RK stage.
    grid : PlaneGrid
        Provides ``dx``, ``dy``, ``f_y`` (Coriolis field), ``area_T``.
    height_coord : HeightCoordinate
        Reference profile + vertical metric.
    terrain_metric : TerrainMetric
        Jacobian (always ``1`` on a flat plane).
    config : CompressibleEulerConfig

    Returns
    -------
    PlaneNonHydrostaticTendencies
    """
    u = state.u.data           # x-face zonal wind, (ny, nx, nlev)
    v = state.v.data           # y-face meridional wind, (ny, nx, nlev)
    w = state.w.data           # half-level vertical wind, (ny, nx, nlev+1)
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    c_p = jnp.asarray(0.0, dtype=u.dtype) + _c_pd_constant()
    theta_0 = height_coord.theta_ref            # (nlev,)
    rho_0 = height_coord.rho_ref                # (nlev,)
    J = terrain_metric.jacobian                 # (ny, nx)

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # 1. Exner perturbation (column-local; reused from shared module).
    from legoesm.atmosphere.dynamics.compressible_euler import (
        compute_exner_perturbation,
    )
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # 2. C-grid pressure gradient. ``grad_x_vlast(pi_p)`` returns
    #    the x-face gradient ``(pi[j,i] - pi[j,i-1])/dx`` — exactly
    #    where ``u`` lives. ``theta_total`` is at cell centres;
    #    interpolate to the x-face so the PG operand co-locates with
    #    its target tendency.
    grad_pi_x_xface = grad_x_vlast(pi_p, grid)
    grad_pi_y_yface = grad_y_vlast(pi_p, grid)
    theta_xface = interp_cell_to_xface_vlast(theta_total, grid)
    theta_yface = interp_cell_to_yface_vlast(theta_total, grid)
    du_pg = -c_p * theta_xface * grad_pi_x_xface
    dv_pg = -c_p * theta_yface * grad_pi_y_yface

    # 3. Coriolis (optional). f at cell centre; interpolate to the
    #    target face so f·u_cross acts at the proper Arakawa-C
    #    location. f varies in y → on the beta-plane the y-face has
    #    a different f than the cell centre (Codex review
    #    2026-05-24); f_xface is identical to f_centre because
    #    x-face shares yc with the cell, but we still interpolate
    #    for symmetry + so the f-plane / beta-plane branch is
    #    handled uniformly.
    if config.use_coriolis:
        f_3d = grid.f_y[:, :, None]
        f_xface = interp_cell_to_xface_vlast(f_3d, grid)
        f_yface = interp_cell_to_yface_vlast(f_3d, grid)
        v_xface = interp_yface_to_xface_vlast(v, grid)
        u_yface = interp_xface_to_yface_vlast(u, grid)
        du_cor = f_xface * v_xface
        dv_cor = -f_yface * u_yface
    else:
        du_cor = jnp.zeros_like(u)
        dv_cor = jnp.zeros_like(v)

    # 4. Mass continuity (flux form): ``drho'/dt = -div(rho · u)``.
    #    ``rho_total`` lives at cell centres; interpolate to each face
    #    so the mass flux has the same staggering as the velocity.
    rho_xface = interp_cell_to_xface_vlast(rho_total, grid)
    rho_yface = interp_cell_to_yface_vlast(rho_total, grid)
    drho_p_dt = -divergence_vlast(rho_xface * u, rho_yface * v, grid)

    # 5. Theta advection (advective form). Theta at cell centre;
    #    advect with the cell-centre velocity formed by face→cell
    #    averaging of u, v. Scheme selected by config.
    #    horizontal_advection_scheme:
    #    - "upwind1" (cheap, 1st-order, dispersive, dt-constrained).
    #    - "van_leer" (2nd-order TVD, stencil 4, monotone — iter-183
    #      production default: 3x wall-time speedup vs upwind1 at
    #      dt=20 thanks to lower numerical diffusion).
    #    - "weno5" (5th-order WENO-Z, stencil 6 — least grid-scale
    #      dispersion but ~3x per-step cost; opt-in for sharp-front
    #      problems where dispersion matters more than throughput).
    scheme = getattr(config, "horizontal_advection_scheme", "upwind1")
    if scheme == "weno5":
        adv_x, adv_y = _weno5_advection_x, _weno5_advection_y
    elif scheme == "van_leer":
        adv_x, adv_y = _van_leer_advection_x, _van_leer_advection_y
    elif scheme == "upwind1":
        adv_x, adv_y = _upwind_advection_x, _upwind_advection_y
    else:
        # iter-194: list the active registry instead of a hardcoded
        # string so a future fourth scheme in
        # HORIZONTAL_ADVECTION_HALO_REQUIREMENT automatically surfaces
        # in the error message.
        raise ValueError(
            f"Unknown horizontal_advection_scheme: {scheme!r}. "
            f"Expected one of "
            f"{sorted(HORIZONTAL_ADVECTION_HALO_REQUIREMENT)}."
        )
    u_center = interp_xface_to_cell_vlast(u, grid)
    v_center = interp_yface_to_cell_vlast(v, grid)
    dtheta_p_dt = (
        adv_x(theta_p, u_center, grid.dx)
        + adv_y(theta_p, v_center, grid.dy)
    )

    # 6. Horizontal momentum advection — Arakawa-C. u lives at x-face;
    #    advect by (u-at-x-face, v-at-x-face). v→x-face via 4-pt
    #    corner average. Symmetric for v. Same scheme as theta.
    v_at_xface = interp_yface_to_xface_vlast(v, grid)
    u_at_yface = interp_xface_to_yface_vlast(u, grid)
    du_adv = (
        adv_x(u, u, grid.dx)
        + adv_y(u, v_at_xface, grid.dy)
    )
    dv_adv = (
        adv_x(v, u_at_yface, grid.dx)
        + adv_y(v, v, grid.dy)
    )

    # 7. Vertical advection of u, v by full-level w.
    du_vert = _vertical_advection_plane(u, w, height_coord, J)
    dv_vert = _vertical_advection_plane(v, w, height_coord, J)

    du_dt = du_adv + du_vert + du_pg + du_cor
    dv_dt = dv_adv + dv_vert + dv_pg + dv_cor

    # 8. w slow part: horizontal advection of w (advective form on
    #    half-level w; interpolate u, v to cell centre then average
    #    to interface for the upwind side selection).
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    dw_full = (
        adv_x(w_full, u_center, grid.dx)
        + adv_y(w_full, v_center, grid.dy)
    )
    # Re-map to half levels: interior is the average of adjacent full
    # values; top and bottom interfaces stay rigid (zero) so the
    # acoustic substep's BC convention is preserved.
    pad_axes = ((0, 0),) * (dw_full.ndim - 1)
    dw_dt = jnp.pad(
        0.5 * (dw_full[..., :-1] + dw_full[..., 1:]),
        (*pad_axes, (1, 1)),
    )

    # 9. Rayleigh sponge layer (top-of-model damping). Zero when
    #    ``config.sponge_coeff == 0`` — the profile evaluates to zero
    #    below ``H - sponge_width`` and reaches ``sponge_coeff`` at
    #    the model top. The same sponge taper is applied to u, v, and
    #    theta' at full levels and to w at half levels; this matches
    #    the MPAS NH dycore pattern in
    #    :func:`compressible_euler_mpas.mpas_compressible_euler_slow_tendencies`.
    sponge_full = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )                                         # (nlev,)
    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )                                         # (nlev+1,)
    du_dt = du_dt - sponge_full * u
    dv_dt = dv_dt - sponge_full * v
    dtheta_p_dt = dtheta_p_dt - sponge_full * theta_p
    drho_p_dt = drho_p_dt - sponge_full * rho_p
    dw_dt = dw_dt - sponge_half * w

    # 10. Biharmonic hyperdiffusion. ``-coeff * ∇⁴f`` with the
    #     discrete 5-point laplacian applied twice; biharmonic damping
    #     is monotone over all nonzero wavenumbers but the eigenvalue
    #     grows as ``sin⁴(π k/nx)``, so for the discrete stencil
    #     2-Δx modes are damped ``(sin(π/2)/sin(π/8))⁴ ≈ 47×`` faster
    #     than 8-Δx modes (preferential grid-scale damping while
    #     leaving long wavelengths nearly untouched in CRM-relevant
    #     timescales). Zero coefficients (default
    #     ``CompressibleEulerConfig`` values) reproduce the PR2d
    #     no-hyperdiff behaviour exactly. Explicit-Euler biharmonic
    #     CFL on a 5-point stencil is ``dt * K * (4/dx² + 4/dy²)² <
    #     2`` for SSP-RK3 stability; callers exceed this at their own
    #     risk and the dycore does not guard at runtime (the cost of
    #     the check inside JIT outweighs the rare benefit; the same
    #     convention as the cubed-sphere / MPAS NH dycores). The
    #     hyperdiff CFL bound is documented at the module docstring
    #     level for users that want to set coefficients near the
    #     stability limit.
    if config.hyperdiff_coeff > 0.0:
        du_dt = du_dt - config.hyperdiff_coeff * laplacian_vlast(
            laplacian_vlast(u, grid), grid,
        )
        dv_dt = dv_dt - config.hyperdiff_coeff * laplacian_vlast(
            laplacian_vlast(v, grid), grid,
        )
        dtheta_p_dt = dtheta_p_dt - config.hyperdiff_coeff * laplacian_vlast(
            laplacian_vlast(theta_p, grid), grid,
        )
    if config.hyperdiff_rho_coeff > 0.0:
        drho_p_dt = drho_p_dt - config.hyperdiff_rho_coeff * laplacian_vlast(
            laplacian_vlast(rho_p, grid), grid,
        )
    # Explicit vertical Laplacian dissipation on theta_p.
    # Form: nu_v * (theta_p[k+1] - 2*theta_p[k] + theta_p[k-1]) / dz_k^2
    # with rigid (zero-gradient → here zero-value) boundary at k=0 and
    # k=nlev-1 via jnp.pad. Damps the buoyancy/PG feedback loop that
    # blows up the dycore at dt > 0.5 s on dx ~ 2 km coarse-vertical
    # grids. dz uses the full-level half-spacing for consistency with
    # the acoustic-substep theta gradient.
    if getattr(config, "vertical_theta_diffusion", 0.0) > 0.0:
        nu_v = config.vertical_theta_diffusion
        # Centred 2nd-order Laplacian on the (..., k) axis with rigid
        # BC. dz_half[k] is the spacing between full levels k and k+1.
        dz_half = height_coord.dz_half  # shape (nlev-1,)
        # Compute interior 2nd derivative.
        tp_above = theta_p[..., 2:]
        tp_below = theta_p[..., :-2]
        tp_centre = theta_p[..., 1:-1]
        dz_avg = 0.5 * (dz_half[:-1] + dz_half[1:])  # (nlev-2,)
        d2_theta_inner = (tp_above - 2.0 * tp_centre + tp_below) / (dz_avg ** 2)
        # Pad with zeros at top/bottom so boundary tendencies are zero.
        pad_axes_v = ((0, 0),) * (theta_p.ndim - 1)
        d2_theta = jnp.pad(d2_theta_inner, (*pad_axes_v, (1, 1)))
        dtheta_p_dt = dtheta_p_dt + nu_v * d2_theta

    if config.hyperdiff_w_coeff > 0.0:
        # ``w`` lives on half levels; the laplacian wrapper applies in
        # the horizontal only (last two axes via ``moveaxis``), so the
        # half-level layout is preserved end-to-end. The rigid w
        # boundary values at ``[..., 0]`` and ``[..., -1]`` stay at
        # zero because laplacian of a uniformly-zero boundary plane is
        # zero.
        dw_dt = dw_dt - config.hyperdiff_w_coeff * laplacian_vlast(
            laplacian_vlast(w, grid), grid,
        )

    # 11. Smagorinsky-Lilly LES eddy viscosity (PR3c). Adds
    #     ``K_m * Lap(u/v)`` to horizontal momentum, ``K_m / Pr *
    #     Lap(theta')`` to potential temperature, and
    #     ``K_m_half * Lap(w)`` to vertical momentum (``K_m`` averaged
    #     from full to half levels). ``c_s = 0`` skips the branch
    #     entirely (cheap Python on/off gate).
    if config.smagorinsky_cs > 0.0:
        # Full 3D Smag strain (replaces horizontal-only pilot). Takes
        # half-level w so the vertical-shear components S13, S23, S33
        # contribute to |S|. K_m lives at cell centres.
        K_m = _compute_smagorinsky_K_m_plane(
            u, v, w, grid, height_coord, config.smagorinsky_cs,
        )
        # u at x-face, v at y-face on the Arakawa-C grid →
        # interpolate K_m to each face before the flux-form
        # diffusion so the operand and diffusivity co-locate.
        K_m_xface = interp_cell_to_xface_vlast(K_m, grid)
        K_m_yface = interp_cell_to_yface_vlast(K_m, grid)
        du_dt = du_dt + _variable_K_diffusion_vlast(u, K_m_xface, grid)
        dv_dt = dv_dt + _variable_K_diffusion_vlast(v, K_m_yface, grid)
        K_h = K_m / config.smagorinsky_prandtl
        dtheta_p_dt = dtheta_p_dt + _variable_K_diffusion_vlast(
            theta_p, K_h, grid,
        )
        # ``K_m`` at full levels; interpolate to half levels for w
        # (rigid boundary K stays zero — no spurious tendency at top
        # / bottom interfaces).
        K_m_half_interior = 0.5 * (K_m[..., :-1] + K_m[..., 1:])
        pad_axes = ((0, 0),) * (K_m_half_interior.ndim - 1)
        K_m_half = jnp.pad(
            K_m_half_interior, (*pad_axes, (1, 1)),
        )
        dw_dt = dw_dt + _variable_K_diffusion_vlast(w, K_m_half, grid)

    # 12. Tracer advection. Advective form via upwind on cell-centre
    #     velocities (face-averaged from ``u``, ``v`` — the C-grid
    #     pairing for cell-centred scalars). vmap over the trailing
    #     tracer axis.
    tracers = state.tracers.data
    if tracers.shape[-1] > 0:
        def _tracer_tend_one(q):
            return (
                adv_x(q, u_center, grid.dx)
                + adv_y(q, v_center, grid.dy)
                + _vertical_advection_plane(q, w, height_coord, J)
            )
        dtracers_dt = jax.vmap(_tracer_tend_one, in_axes=-1, out_axes=-1)(
            tracers,
        )
    else:
        dtracers_dt = jnp.zeros_like(tracers)
    zero_phis = jnp.zeros_like(state.phis.data)

    return PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=du_dt),
        dv_dt=state.v.replace(data=dv_dt),
        dw_dt=state.w.replace(data=dw_dt),
        dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_dt),
        drho_prime_dt=state.rho_prime.replace(data=drho_p_dt),
        dphis_dt=state.phis.replace(data=zero_phis),
        dtracers_dt=state.tracers.replace(data=dtracers_dt),
    )


def _c_pd_constant() -> float:
    """Local accessor for the dry-air specific heat at constant pressure.

    Lifted out of the slow tendency so the function body stays
    Python-pure; the actual value lives in
    :mod:`legoesm.constants`.
    """
    from legoesm import constants
    return constants.c_pd


# --------------------------------------------------------------------- #
# Acoustic substep wrapper                                              #
# --------------------------------------------------------------------- #


def plane_acoustic_substeps(
    state: PlaneNonHydrostaticState,
    slow_tend: PlaneNonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
) -> PlaneNonHydrostaticState:
    """Run ``n_substeps`` forward-backward acoustic substeps on the plane.

    Thin wrapper around
    :func:`compressible_euler._acoustic_column_kernel` — no duplicated
    vertical algebra. The kernel is driven by ``jax.lax.fori_loop`` and
    operates on the ``(ny, nx, nlev)`` plane layout via the natural
    last-axis broadcast (the kernel slices ``[..., k]`` for the
    vertical index).

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        State after the slow-tendency axpy update.
    slow_tend : PlaneNonHydrostaticTendencies
        Slow tendencies; unused by the column kernel (column-local
        vertical update only), retained in the signature for parity
        with the MPAS / cubed-sphere acoustic-substep interface.
    dt_s : float
        Substep size in seconds.
    n_substeps : int
        Number of substeps per outer stage.
    config : SplitExplicitConfig
        Outer split-explicit configuration (unused inside; kept for
        signature compatibility).
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
        Provides ``jacobian`` of shape ``(ny, nx)`` — broadcasts to the
        horizontal leading axes of the kernel inputs.
    euler_config : CompressibleEulerConfig
        Carries ``g`` and ``acoustic_off_centering``.

    Returns
    -------
    PlaneNonHydrostaticState
        Updated ``(w, theta_prime, rho_prime)``. ``u``, ``v``, ``phis``,
        and ``tracers`` are unchanged because the acoustic substep is
        vertical-only.
    """
    del slow_tend, config  # signature parity with other dycores
    g = euler_config.g
    J = terrain_metric.jacobian
    beta = euler_config.acoustic_off_centering

    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    # Python-loop unroll (n_substeps is compile-time static via
    # SplitExplicitConfig). See semi-implicit variant for full rationale.
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _ in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = _acoustic_column_kernel(
            w_final, theta_p_final, rho_p_final,
            height_coord, J, dt_s, beta, g,
        )

    return PlaneNonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


def plane_acoustic_substeps_semi_implicit(
    state: PlaneNonHydrostaticState,
    slow_tend: PlaneNonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
) -> PlaneNonHydrostaticState:
    """Semi-implicit acoustic substeps on the plane via per-column
    Thomas tridiagonal solve.

    Lifts the vertical acoustic CFL constraint that limits the
    forward-backward variant to ``dt ≲ dx/c_s``. Permits outer
    ``dt`` ~ 30-60 s at dx=2 km (advective CFL bound only).

    Thin wrapper around
    :func:`compressible_euler._semi_implicit_acoustic_column_kernel`
    — column-local algebra, no duplicated vertical math. Signature
    parity with :func:`plane_acoustic_substeps`.
    """
    del slow_tend, config  # signature parity with other dycores
    g = euler_config.g
    J = terrain_metric.jacobian
    beta = euler_config.acoustic_off_centering
    implicit_buoyancy = euler_config.implicit_buoyancy

    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    # Loop-invariant tridiag bands — hoisted once, reused n_substeps times.
    tri_bands = precompute_si_tridiag_bands(
        height_coord, J, dt_s, g, implicit_buoyancy,
        nlev=theta_p.shape[-1],
    )

    # n_substeps is compile-time static (from SplitExplicitConfig field),
    # so a Python for-loop fully unrolls the substep sequence — XLA then
    # has straight-line HLO across iterations and can fuse the post-cuSPARSE
    # tail of one substep with the pre-cuSPARSE head of the next. This
    # replaces ``lax.fori_loop`` which kept the substeps as a while-loop and
    # prevented inter-iteration fusion.
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _ in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = (
            _semi_implicit_acoustic_column_kernel(
                w_final, theta_p_final, rho_p_final,
                height_coord, J, dt_s, beta, g,
                implicit_buoyancy=implicit_buoyancy,
                precomputed_tridiag=tri_bands,
            )
        )

    return PlaneNonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


# --------------------------------------------------------------------- #
# Config validation                                                     #
# --------------------------------------------------------------------- #


def validate_plane_config(config: CompressibleEulerConfig) -> None:
    """Reject ``CompressibleEulerConfig`` flags unsupported in PR2b.

    Raises :class:`NotImplementedError` with a clear pointer to which
    later PR enables the flag, so users see a single source of truth
    instead of obscure runtime errors deep in the dycore.

    Supported in PR2b:
    - ``g``, ``n_acoustic_substeps``, ``small_earth_factor``
    - ``use_coriolis``
    - ``acoustic_off_centering``
    - ``fix_mass``, ``anchor_mass_to_initial``
    - ``outer_integrator`` (any valid SSP-RK choice)

    Unsupported in PR2b (raise):
    - ``semi_implicit_acoustic`` — needs tridiagonal column solve in
      the plane wrapper (lands in PR3 alongside hyperdiffusion).
    - ``sponge_coeff > 0`` — sponge layer lands with PR2c validation
      benchmarks (rising-thermal / Straka use a top sponge).
    - ``hyperdiff_coeff / hyperdiff_rho_coeff / hyperdiff_w_coeff > 0``
      — biharmonic damping lands in PR3.
    """
    # semi_implicit_acoustic now supported via
    # plane_acoustic_substeps_semi_implicit (per-column Thomas solve
    # using the shared _semi_implicit_acoustic_column_kernel).
    # ``sponge_coeff > 0`` is now supported (PR2d) — the Rayleigh
    # sponge is applied inside ``plane_compressible_euler_slow_tendencies``
    # to ``u``, ``v``, ``theta'`` at full levels and ``w`` at half
    # levels via the shared ``_sponge_profile`` taper.
    # ``hyperdiff_coeff`` and friends are now supported (PR3a) — the
    # biharmonic ``-coeff * Lap(Lap(field))`` term is added to the
    # slow tendency in
    # :func:`plane_compressible_euler_slow_tendencies`. Setting all
    # three to zero (the ``CompressibleEulerConfig`` default for
    # plane-friendly setups) reproduces the PR2b behaviour exactly.
    # Negative coefficients would invert the damping sign and produce
    # exponential growth — almost certainly a user error — so reject
    # them up front rather than silently treating them as off.
    for name, value in (
        ("hyperdiff_coeff", config.hyperdiff_coeff),
        ("hyperdiff_rho_coeff", config.hyperdiff_rho_coeff),
        ("hyperdiff_w_coeff", config.hyperdiff_w_coeff),
    ):
        if value < 0.0:
            raise ValueError(
                f"{name}={value!r} must be non-negative; biharmonic "
                "hyperdiffusion damps only with a positive coefficient. "
                "Pass 0.0 to disable."
            )
    if config.smagorinsky_cs < 0.0:
        raise ValueError(
            f"smagorinsky_cs={config.smagorinsky_cs!r} must be "
            "non-negative; Smagorinsky-Lilly K_m = (C_s Δ)^2 |S| "
            "only damps for C_s >= 0. Pass 0.0 to disable."
        )
    if config.smagorinsky_cs > 0.0 and config.smagorinsky_prandtl <= 0.0:
        raise ValueError(
            f"smagorinsky_prandtl={config.smagorinsky_prandtl!r} must "
            "be > 0 when smagorinsky_cs > 0; K_h = K_m / Pr inverts "
            "or NaNs for Pr <= 0."
        )


# --------------------------------------------------------------------- #
# Driver class                                                          #
# --------------------------------------------------------------------- #


class PlaneCompressibleEulerModel:
    """Outer driver for the doubly-periodic plane NH compressible Euler dycore.

    PR2b MVP. The dycore is reachable only by direct construction
    (no public-config grid literal yet — that lands in PR2c). Tracer
    transport, sponge, hyperdiffusion, semi-implicit acoustic, and
    sloped terrain are explicitly rejected at construction time so
    users see a single source of truth for the supported feature set.

    Parameters
    ----------
    grid : PlaneGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
        Must be flat — ``J == 1`` and ``z_s == 0`` everywhere.
    config : CompressibleEulerConfig, optional
        Defaults to a permissive setup; the constructor enforces every
        PR2b restriction explicitly via :func:`validate_plane_config`.

    Examples
    --------
    >>> grid = create_plane_grid(nx=16, ny=16, nlev=10,
    ...                          dx=1.0e3, dy=1.0e3)
    >>> height_coord = create_height_coordinate(grid.nlev, H=20.0e3)
    >>> terrain = make_flat_plane_terrain_metric(grid, height_coord)
    >>> model = PlaneCompressibleEulerModel(grid, height_coord, terrain)
    >>> state = make_rest_state(grid, height_coord)
    >>> next_state = model.step(state, dt=1.0)
    """

    def __init__(
        self,
        grid: PlaneGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: CompressibleEulerConfig | None = None,
    ):
        if not isinstance(grid, PlaneGrid):
            raise TypeError(
                f"PlaneCompressibleEulerModel expects a PlaneGrid; "
                f"got {type(grid).__name__}"
            )
        _assert_flat_terrain(terrain_metric)
        if config is None:
            config = CompressibleEulerConfig()
        validate_plane_config(config)
        self.grid = grid
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config
        self._target_mass: jax.Array | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def precompute_target_mass(
        self,
        state: PlaneNonHydrostaticState,
    ) -> None:
        """Cache target dry mass from ``state`` for fix_mass use under JIT.

        ``step()`` lazily caches ``_target_mass`` from its first input,
        which leaks the traced array when called inside ``jax.lax.scan``
        with ``fix_mass=True``. Calling this method BEFORE the scan
        pre-populates the cache with a concrete (non-traced) array, so
        the scan body sees a closed-over constant.

        Usage::

            model._target_mass = None  # if already set
            model.precompute_target_mass(initial_state)
            out = jax.lax.scan(lambda s,_: (model.step(s, dt), None),
                               initial_state, None, length=N)[0]

        Required when fix_mass=True + anchor_mass_to_initial=True is
        combined with lax.scan-based time integration.
        """
        if self.config.fix_mass:
            self._target_mass = compute_dry_mass_plane(
                state, self.grid, self.height_coord, self.terrain_metric,
            )

    def step(
        self,
        state: PlaneNonHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> PlaneNonHydrostaticState:
        """Advance the dycore by ``dt`` seconds.

        Calls the shared SSP-RK3 split-explicit driver under the hood.

        Parameters
        ----------
        state : PlaneNonHydrostaticState
            Current prognostic state.
        dt : float
            Outer time step in seconds.
        physics_fn : callable, optional
            Physics tendency callable (PR3d). Signature
            ``physics_fn(state, grid, height_coord, terrain_metric) ->
            PlaneNonHydrostaticTendencies``. Added to the slow
            tendency once per outer RK stage (matches MPAS / cubed-
            sphere convention). ``None`` (default) skips the call.

        Returns
        -------
        PlaneNonHydrostaticState
            State after one full SSP-RK3 step with acoustic substeps.
        """
        if self.config.fix_mass and self.config.anchor_mass_to_initial:
            if self._target_mass is None:
                self._target_mass = compute_dry_mass_plane(
                    state, self.grid, self.height_coord, self.terrain_metric,
                )
        target = self._target_mass
        if target is None and self.config.fix_mass:
            target = compute_dry_mass_plane(
                state, self.grid, self.height_coord, self.terrain_metric,
            )
        return self._step_jit(state, dt, target, physics_fn)

    def step_halo(
        self,
        state_local: PlaneNonHydrostaticState,
        dt: float,
        layout,
        f_pad_cached: jax.Array | None = None,
        owned_mask: jax.Array | None = None,
    ) -> PlaneNonHydrostaticState:
        """Domain-decomposed step using halo-aware slow tendency.

        Each rank owns a local slab. Halo exchange via
        :func:`packed_exchange_halo_plane_yxz` per slow-tendency call.
        Acoustic substeps stay vertical-only (column-local) — no MPI.

        Eager-mode only on multi-rank (mpi4jax sendrecv branch not
        jit-safe on macOS shared-mem). Single-rank gets jit speedup
        via the existing :meth:`step` path.

        Mass fixer is SKIPPED on multi-rank (compute_dry_mass_plane
        is a global reduction — separate MPI variant needed). Single
        rank with ``fix_mass=True`` falls back to :meth:`step`.

        Smagorinsky LES + vertical-θ diffusion now supported on the
        halo path (R4/R5 — see ``compressible_euler_plane_halo``).
        Mass fixer R7: pass ``owned_mask`` (rank-local 0/1 mask of
        cells the rank owns) to use the MPI-aware dry-mass fixer
        :func:`legoesm.atmosphere.dynamics.rce_mpi.fix_mass_nonhydrostatic_plane_mpi`.
        Without ``owned_mask`` the multi-rank fix_mass branch is
        skipped (mass not anchored to target) — single-rank still
        uses the serial fixer via ``step``.
        """
        from legoesm.atmosphere.dynamics.compressible_euler_plane_halo import (
            plane_compressible_euler_slow_tendencies_halo,
        )
        from legoesm.timestepping.split_explicit import (
            split_explicit_step, SplitExplicitConfig,
        )

        if layout.n_ranks == 1:
            # Single-rank: halo path is bit-identical to the standard
            # jit'd step (proven by test_halo_equiv_*) — route to it
            # for the JIT speedup. Skips packed exchange overhead +
            # gives ~14x faster per-step on small grids.
            return self.step(state_local, dt)

        # Multi-rank correctness gate (Codex 2026-05 review): silently
        # skipping the mass fixer when fix_mass=True but no owned_mask
        # is provided lets dry mass drift unbounded across a long
        # production run while reporting "fix_mass=True". Surface
        # immediately so callers wire owned_mask explicitly.
        if self.config.fix_mass and owned_mask is None:
            raise ValueError(
                "step_halo on multi-rank with config.fix_mass=True "
                "requires owned_mask (shape (ny_local, nx_local), "
                "1.0 on owned cells, 0.0 on duplicated halo rows). "
                "Compute it once from layout.iy_start/iy_end + "
                "layout.ix_start/ix_end and pass it via the "
                "owned_mask kwarg, or set config.fix_mass=False to "
                "opt out of mass anchoring on this multi-rank path."
            )

        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            tend = plane_compressible_euler_slow_tendencies_halo(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.config, layout, f_pad_cached=f_pad_cached,
            )
            # Wrap as state for split_explicit_step (same pytree shape).
            return PlaneNonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(
                    data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(
                    data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=tend.dphis_dt.data),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        if self.config.semi_implicit_acoustic:
            def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                return plane_acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                )
        else:
            def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                return plane_acoustic_substeps(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                )

        stepped = split_explicit_step(
            state_local, slow_tendency_fn, acoustic_update_fn,
            dt, se_config,
        )

        # MPI-aware dry-mass fixer (R7). Only fires when the user
        # passes an owned_mask + config.fix_mass is True. Without the
        # mask we can't compute a non-double-counted global mass.
        if self.config.fix_mass and owned_mask is not None:
            from legoesm.atmosphere.dynamics.rce_mpi import (
                compute_dry_mass_plane_mpi,
                fix_mass_nonhydrostatic_plane_mpi,
            )
            if self.config.anchor_mass_to_initial:
                if self._target_mass is None:
                    # Initial reduction across all ranks so every rank
                    # captures the same anchor.
                    self._target_mass = compute_dry_mass_plane_mpi(
                        state_local, self.grid, self.height_coord,
                        self.terrain_metric, layout, owned_mask,
                    )
                target = self._target_mass
            else:
                target = compute_dry_mass_plane_mpi(
                    state_local, self.grid, self.height_coord,
                    self.terrain_metric, layout, owned_mask,
                )
            stepped = fix_mass_nonhydrostatic_plane_mpi(
                stepped, target, self.grid, self.height_coord,
                self.terrain_metric, layout, owned_mask,
            )
        return stepped

    # ------------------------------------------------------------------
    # JIT boundary
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def _step_jit(
        self,
        state: PlaneNonHydrostaticState,
        dt: float,
        target_mass: jax.Array | None,
        physics_fn=None,
    ) -> PlaneNonHydrostaticState:
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s: PlaneNonHydrostaticState):
            tend = plane_compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.config,
            )
            if physics_fn is not None:
                phys = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
                # Add physics tendencies to dycore slow tendency.
                tend = PlaneNonHydrostaticTendencies(
                    du_dt=tend.du_dt.replace(
                        data=tend.du_dt.data + phys.du_dt.data),
                    dv_dt=tend.dv_dt.replace(
                        data=tend.dv_dt.data + phys.dv_dt.data),
                    dw_dt=tend.dw_dt.replace(
                        data=tend.dw_dt.data + phys.dw_dt.data),
                    dtheta_prime_dt=tend.dtheta_prime_dt.replace(
                        data=tend.dtheta_prime_dt.data
                        + phys.dtheta_prime_dt.data),
                    drho_prime_dt=tend.drho_prime_dt.replace(
                        data=tend.drho_prime_dt.data
                        + phys.drho_prime_dt.data),
                    dphis_dt=tend.dphis_dt,
                    dtracers_dt=tend.dtracers_dt.replace(
                        data=tend.dtracers_dt.data
                        + phys.dtracers_dt.data),
                )
            # Rewrap as ``PlaneNonHydrostaticState`` so the shared
            # ``pytree_axpy`` inside ``split_explicit_step`` can align
            # the leaves — it requires matching NamedTuple types
            # between state and tendency. Same pattern as the MPAS NH
            # dycore. ``phis`` tendency is zero (static surface);
            # ``tracers`` tendency is a zero array of the right shape
            # (PR2b only supports ``n_tracers == 0`` so this is just
            # the empty-trailing-axis array).
            return PlaneNonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(
                    data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(
                    data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=tend.dphis_dt.data),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        if self.config.semi_implicit_acoustic:
            def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                return plane_acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                )
        else:
            def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                return plane_acoustic_substeps(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                )

        state_new = split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

        if self.config.fix_mass and target_mass is not None:
            state_new = fix_mass_nonhydrostatic_plane(
                state_new, target_mass, self.grid, self.height_coord,
                self.terrain_metric,
            )
        return state_new
