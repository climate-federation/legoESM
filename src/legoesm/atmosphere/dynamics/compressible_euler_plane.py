"""Doubly-periodic plane non-hydrostatic compressible Euler dycore.

Stages of the CRM rollout
-------------------------
PR2a (already merged in this file) shipped the building blocks:
vertical-last operator wrappers around the PR1 plane stencils, a
flat-terrain metric constructor, the dry-mass integral and a uniform-
additive mass fixer. PR2b (this revision) wires those building blocks
into a working dry plane dycore by adding:

* :func:`plane_compressible_euler_slow_tendencies` — RK-stage slow
  tendency (horizontal advection of mass / theta / momentum,
  horizontal pressure-gradient, optional Coriolis, horizontal w slow
  part). Vertical acoustic terms are deferred to the substep kernel.
* :func:`plane_acoustic_substeps` — thin wrapper around
  :func:`compressible_euler._acoustic_column_kernel` (extracted in
  PR2a) that drives the substep loop and rewraps the result into
  ``PlaneNonHydrostaticState``.
* :class:`PlaneCompressibleEulerModel` — outer driver with strict
  PR2b config validation (rejects semi-implicit acoustic, sponge,
  hyperdiffusion, tracers, non-flat terrain, ``physics_fn``).

PR2c will publicly expose this dycore in the driver config and
``supported_matrix`` so experiments can dispatch into it. PR3 adds
microphysics + LES SGS + hyperdiffusion + tracer transport.

Equations solved (dry, perturbation form)
-----------------------------------------
Prognostic ``(u, v, w, theta', rho')`` with full state
``theta = theta_ref(z) + theta'`` and ``rho = rho_ref(z) + rho'``
where ``(theta_ref, rho_ref)`` is the hydrostatically balanced 1D
reference profile from :class:`HeightCoordinate`.

Continuity (mass, flux-form so the discrete dry-mass integral is
conserved up to mass-fixer correction)

    drho'/dt = -d/dx (rho_total * u) - d/dy (rho_total * v)
             - (vertical acoustic mass flux divergence, in substep)

Potential temperature (advective form; vertical advection by w is
absorbed into the substep)

    dtheta'/dt = -u * d theta_total / dx - v * d theta_total / dy
                - (w * d theta_total / dz, in substep)

Horizontal momentum (advective form; vertical advection by w via
:func:`_vertical_advection_plane` per column)

    du/dt = -u du/dx - v du/dy - w du/dz
           - c_p * theta_total * d pi' / dx + f * v
    dv/dt = -u dv/dx - v dv/dy - w dv/dz
           - c_p * theta_total * d pi' / dy - f * u

Vertical velocity (horizontal slow advection + vertical PG and
buoyancy via substep)

    dw/dt = -u dw/dx - v dw/dy
           + (-c_p * theta * d pi' / dz + g * theta' / theta_ref, in substep)

Operator splitting
------------------
The split-explicit infrastructure
(:func:`legoesm.timestepping.split_explicit.split_explicit_step`)
applies the slow tendency at each outer SSP-RK3 stage, then runs
``n_substeps`` acoustic substeps (vertical-only) on ``(w, theta',
rho')``. Slow updates to ``u, v, w`` (and the horizontal slow part of
``rho'``) are applied via an axpy on the full state pytree before each
substep loop. Tracer transport is deferred to PR3, so PR2b accepts
only ``n_tracers == 0``.

Hyperdiffusion stability (PR3a)
-------------------------------
Biharmonic damping ``-K_h ∇⁴`` is integrated explicitly through the
slow-tendency path, so the user-supplied ``hyperdiff_coeff``,
``hyperdiff_rho_coeff``, ``hyperdiff_w_coeff`` must satisfy the
explicit-Euler CFL bound on the 5-point stencil::

    dt * K * (4/dx² + 4/dy²)² < CFL_max

with ``CFL_max ≈ 2`` for SSP-RK3 (slightly looser than forward
Euler). For the CI test grid (``dx = dy = 200 m``, ``dt = 0.5 s``)
the eigenvalue is ``(2·4/200²)² = 4 × 10⁻⁸``, so the bound becomes
``K < 2 / (0.5 · 4 × 10⁻⁸) = 1 × 10⁸ m⁴/s``; the CI tests use
``K = 1 × 10⁶`` and ``1 × 10⁷``, well inside the bound. The plane dycore does not check
the bound at runtime — the same convention as the cubed-sphere and
MPAS NH dycores. Setting ``K`` near the bound trades increased
damping for risk of grid-scale oscillation; documented user
responsibility.

A-grid simplification used in PR2b
----------------------------------
The state class stores ``u`` at x-faces and ``v`` at y-faces per the
PR1 Arakawa-C convention, but the PR2b slow tendency assembles
horizontal advection of ``u, v, w_full, theta_total, rho_total`` using
**cell-centred centred differences** of the field as stored
(equivalent to treating the values as A-grid). The discrete divergence
used for ``rho'`` continuity is also cell-centred. This intentionally
trades the energy-consistent C-grid PG / divergence pairing for code
simplicity in the first runnable dycore — the rest-state preservation
and dry-mass conservation tests still hold (proof: every tendency
contains at least one factor of a velocity or perturbation, all zero
on the balanced rest state). Energy-consistent C-grid discretisation
lands together with the rising-thermal + Straka validation in a
follow-up PR once the MVP is wired publicly. See
:func:`plane_compressible_euler_slow_tendencies` for the explicit
operator choices.

State convention
----------------
All plane prognostic arrays are vertical-LAST (matches
:class:`legoesm.core.state.PlaneNonHydrostaticState` and the existing
cubed-sphere / MPAS convention) so that the shared acoustic column
kernel slices the same axis. PR1 plane operators in
:mod:`legoesm.atmosphere.dynamics.plane_operators` expect
``(..., ny, nx)`` (vertical-trailing-horizontal). The vlast wrappers
in this module use :func:`jax.numpy.moveaxis` to bridge the two
conventions without modifying PR1 operators.

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
    _sponge_profile,
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
# Centred-difference helpers (A-grid simplification for PR2b)           #
# --------------------------------------------------------------------- #


def _d_dx_centered(field_yxz: jax.Array, dx: float) -> jax.Array:
    """``df/dx`` at cell centres by centred difference (2 dx wide).

    ``out[..., j, i, k] = (field[..., j, i+1, k] - field[..., j, i-1, k]) /
    (2 dx)``, periodic via ``jnp.roll``. The horizontal layout is
    ``(ny, nx, nlev)`` so the x axis is ``axis=1``.
    """
    return (jnp.roll(field_yxz, -1, axis=1) - jnp.roll(field_yxz, 1, axis=1)) / (
        2.0 * dx
    )


def _d_dy_centered(field_yxz: jax.Array, dy: float) -> jax.Array:
    """``df/dy`` at cell centres by centred difference. Same as
    :func:`_d_dx_centered` but along ``axis=0``."""
    return (jnp.roll(field_yxz, -1, axis=0) - jnp.roll(field_yxz, 1, axis=0)) / (
        2.0 * dy
    )


def _upwind_advection_x(field_yxz: jax.Array, u_yxz: jax.Array,
                        dx: float) -> jax.Array:
    """First-order upwind contribution to ``-u df/dx`` at cell centres.

    For each cell pick the one-sided difference on the side the local
    ``u`` flows from::

        out = -max(u, 0) * (f[i] - f[i-1]) / dx
              - min(u, 0) * (f[i+1] - f[i]) / dx

    Periodic neighbours via ``jnp.roll`` (axis=1 = x for the
    ``(ny, nx, nlev)`` layout). Upwind biasing damps the 2-Δx
    dispersive mode that destabilises centred differencing on
    advection-dominated plume flow — same role as upwind / flux-
    limited reconstruction in the ocean ``_van_leer_limiter`` and
    cubed-sphere ``advect_upwind`` paths.

    A-grid simplification (PR2b carry-over)
    ---------------------------------------
    ``field_yxz`` and ``u_yxz`` are assumed to live at the same
    horizontal location — the cell centre. PR2b made the explicit
    choice to treat the state's nominally Arakawa-C ``u``, ``v``
    values as cell-centered for the slow tendency, trading
    energy-consistent C-grid PG/divergence pairing for simpler
    advection. PR3b inherits that simplification — the upwind side
    is picked from the local cell-centered velocity rather than a
    proper face-staggered interpolation. The energy-consistent
    C-grid refactor (which would interpolate ``v`` to x-faces for
    ``u``-advection and so on) lands in a follow-up PR alongside
    the LES SGS closure.

    Differentiability
    -----------------
    ``jnp.maximum`` / ``jnp.minimum`` against ``0.0`` introduce a
    kink at ``u = 0`` (or ``v = 0``). JAX returns a finite
    subgradient at the kink, so ``jax.grad`` flows cleanly even
    through zero-velocity cells — verified by
    ``tests/unit/test_plane_nh_upwind.py::test_upwind_grad_through_zero_velocity``.
    """
    f_backward = (field_yxz - jnp.roll(field_yxz, 1, axis=1)) / dx
    f_forward = (jnp.roll(field_yxz, -1, axis=1) - field_yxz) / dx
    u_pos = jnp.maximum(u_yxz, 0.0)
    u_neg = jnp.minimum(u_yxz, 0.0)
    return -(u_pos * f_backward + u_neg * f_forward)


def _upwind_advection_y(field_yxz: jax.Array, v_yxz: jax.Array,
                        dy: float) -> jax.Array:
    """First-order upwind contribution to ``-v df/dy`` at cell centres.
    Same algebra as :func:`_upwind_advection_x` but along ``axis=0``."""
    f_backward = (field_yxz - jnp.roll(field_yxz, 1, axis=0)) / dy
    f_forward = (jnp.roll(field_yxz, -1, axis=0) - field_yxz) / dy
    v_pos = jnp.maximum(v_yxz, 0.0)
    v_neg = jnp.minimum(v_yxz, 0.0)
    return -(v_pos * f_backward + v_neg * f_forward)


def _horizontal_divergence_centered(
    u_yxz: jax.Array, v_yxz: jax.Array, grid: PlaneGrid
) -> jax.Array:
    """Cell-centred divergence ``du/dx + dv/dy`` from centred differences.

    This is the A-grid analogue of :func:`divergence_vlast`; both
    reduce to the same operator on a doubly-periodic plane when the
    inputs live at cell centres because the centred operator is
    equivalent to (forward face flux − backward face flux) / 2.
    """
    return _d_dx_centered(u_yxz, grid.dx) + _d_dy_centered(v_yxz, grid.dy)


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
    being solved and the A-grid simplification used for horizontal
    momentum advection in PR2b.

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        Prognostic state at the start of the RK stage.
    grid : PlaneGrid
        Provides ``dx``, ``dy``, ``f_y`` (Coriolis field), ``area_T``.
    height_coord : HeightCoordinate
        Reference profile + vertical metric.
    terrain_metric : TerrainMetric
        Jacobian (always ``1`` on a flat plane in PR2b).
    config : CompressibleEulerConfig
        Acoustic / sponge / hyperdiffusion knobs. PR2b only uses
        ``config.use_coriolis``; other flags must be the off defaults
        and are checked by :class:`PlaneCompressibleEulerModel` at
        construction time, not here.

    Returns
    -------
    PlaneNonHydrostaticTendencies
        Slow tendencies for every prognostic. ``dphis_dt`` and
        ``dtracers_dt`` are zero arrays (PR2b does not advect either).
    """
    u = state.u.data           # (ny, nx, nlev)
    v = state.v.data           # (ny, nx, nlev)
    w = state.w.data           # (ny, nx, nlev+1)
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

    # 2. Horizontal pressure gradient at cell centres.
    grad_pi_x = _d_dx_centered(pi_p, grid.dx)
    grad_pi_y = _d_dy_centered(pi_p, grid.dy)
    du_pg = -c_p * theta_total * grad_pi_x
    dv_pg = -c_p * theta_total * grad_pi_y

    # 3. Coriolis (optional; ``f_y`` already broadcasts to (ny, nx, 1)).
    if config.use_coriolis:
        f_3d = grid.f_y[:, :, None]
        du_cor = f_3d * v
        dv_cor = -f_3d * u
    else:
        du_cor = jnp.zeros_like(u)
        dv_cor = jnp.zeros_like(v)

    # 4. Mass continuity (flux form) — drho'/dt = -d/dx(rho u) - d/dy(rho v).
    drho_p_dt = -_horizontal_divergence_centered(
        rho_total * u, rho_total * v, grid,
    )

    # 5. Theta horizontal advection (advective form, first-order
    #    upwind so the 2-Δx mode triggered by sharp plume gradients
    #    is damped at the advection step — see PR3a docstring).
    #    NB: ``u``, ``v`` are passed in as cell-centered velocities
    #    per the module-level "A-grid simplification used in PR2b"
    #    section — the same convention is honoured for Sections 6, 7,
    #    8 below. Field locations therefore match where the upwind
    #    helper expects them; the Arakawa-C state labels remain in
    #    place for the future C-grid refactor.
    dtheta_p_dt = (
        _upwind_advection_x(theta_total, u, grid.dx)
        + _upwind_advection_y(theta_total, v, grid.dy)
    )

    # 6. Horizontal momentum advection (advective form, first-order
    #    upwind on cell-centred ``u``, ``v``).
    du_adv = (
        _upwind_advection_x(u, u, grid.dx)
        + _upwind_advection_y(u, v, grid.dy)
    )
    dv_adv = (
        _upwind_advection_x(v, u, grid.dx)
        + _upwind_advection_y(v, v, grid.dy)
    )

    # 7. Vertical advection of u, v by full-level w (cell-centred A-grid).
    du_vert = _vertical_advection_plane(u, w, height_coord, J)
    dv_vert = _vertical_advection_plane(v, w, height_coord, J)

    du_dt = du_adv + du_vert + du_pg + du_cor
    dv_dt = dv_adv + dv_vert + dv_pg + dv_cor

    # 8. w slow part: horizontal advection of w (advective form on
    #    full-level interpolation; rigid w boundaries at interfaces).
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    dw_full = (
        _upwind_advection_x(w_full, u, grid.dx)
        + _upwind_advection_y(w_full, v, grid.dy)
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

    zero_tracers = jnp.zeros_like(state.tracers.data)
    zero_phis = jnp.zeros_like(state.phis.data)

    return PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=du_dt),
        dv_dt=state.v.replace(data=dv_dt),
        dw_dt=state.w.replace(data=dw_dt),
        dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_dt),
        drho_prime_dt=state.rho_prime.replace(data=drho_p_dt),
        dphis_dt=state.phis.replace(data=zero_phis),
        dtracers_dt=state.tracers.replace(data=zero_tracers),
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

    def substep_body(_, carry):
        w_c, theta_p_c, rho_p_c = carry
        return _acoustic_column_kernel(
            w_c, theta_p_c, rho_p_c,
            height_coord, J, dt_s, beta, g,
        )

    w_final, theta_p_final, rho_p_final = jax.lax.fori_loop(
        0, n_substeps, substep_body, (w, theta_p, rho_p),
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
    if config.semi_implicit_acoustic:
        raise NotImplementedError(
            "PR2b plane dycore only supports forward-backward acoustic "
            "substeps; set semi_implicit_acoustic=False or wait for the "
            "follow-up PR that lifts the kernel into the tridiagonal "
            "solver path."
        )
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

    def step(
        self,
        state: PlaneNonHydrostaticState,
        dt: float,
    ) -> PlaneNonHydrostaticState:
        """Advance the dycore by ``dt`` seconds.

        Calls the shared SSP-RK3 split-explicit driver under the hood.
        PR2b does not accept a ``physics_fn`` — the dycore is dry and
        does not couple to any physics until PR3 wires microphysics
        and the LES closures.

        Parameters
        ----------
        state : PlaneNonHydrostaticState
            Current prognostic state.
        dt : float
            Outer time step in seconds.

        Returns
        -------
        PlaneNonHydrostaticState
            State after one full SSP-RK3 step with acoustic substeps.
        """
        if state.tracers.data.shape[-1] != 0:
            raise NotImplementedError(
                "PR2b plane dycore only supports n_tracers == 0; tracer "
                "transport lands in PR3 alongside microphysics."
            )
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
        return self._step_jit(state, dt, target)

    # ------------------------------------------------------------------
    # JIT boundary
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def _step_jit(
        self,
        state: PlaneNonHydrostaticState,
        dt: float,
        target_mass: jax.Array | None,
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
