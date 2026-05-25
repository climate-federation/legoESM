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

Arakawa-C discretisation with energy-consistent PG/divergence pair
------------------------------------------------------------------
The slow tendency honours the Arakawa-C staggering in every term:
``u`` at x-faces, ``v`` at y-faces, scalars at cell centres,
``w`` at half levels. The horizontal pressure gradient uses
:func:`grad_x_vlast` / :func:`grad_y_vlast` which return values at
the x/y-face — exactly where ``u/v`` live — and the discrete
divergence uses :func:`divergence_vlast` on the face-staggered
mass fluxes ``rho_face · u`` and ``rho_face · v``. This is the
adjoint pair documented in
:mod:`plane_operators` — discrete integration by parts
``sum(phi · div(u, v)) == -sum(u · grad_x(phi)) -
sum(v · grad_y(phi))`` holds to machine epsilon, the discrete
condition for energy-consistent PG / divergence coupling.

Cross-component velocities for momentum advection use the 4-point
corner-average interpolators
(:func:`interp_yface_to_xface_vlast`,
:func:`interp_xface_to_yface_vlast`); scalar advection uses
face→centre averages (:func:`interp_xface_to_cell_vlast`,
:func:`interp_yface_to_cell_vlast`). Smagorinsky-Lilly LES
(below) is the full 3D strain tensor — every component is
evaluated at its natural Arakawa-C location, then ``S_ij²`` is
averaged to cell centres for the eddy-viscosity ``K_m``. No
A-grid simplification anywhere.

Note: horizontal advection of momentum, theta and tracers is
first-order upwind (dissipative). Full energy-conserving
vector-invariant momentum advection is a separate item; the
PG/div pairing alone is the adjoint identity that
:mod:`plane_operators` proves to machine epsilon.

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


def interp_cell_to_xface_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """Cell-centre → x-face interpolation, vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.interp_cell_to_xface(_move_vertical_to_front(phi_yxz), grid)
    )


def interp_cell_to_yface_vlast(phi_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """Cell-centre → y-face interpolation, vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.interp_cell_to_yface(_move_vertical_to_front(phi_yxz), grid)
    )


def interp_xface_to_cell_vlast(u_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """x-face → cell-centre interpolation, vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.interp_xface_to_cell(_move_vertical_to_front(u_yxz), grid)
    )


def interp_yface_to_cell_vlast(v_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """y-face → cell-centre interpolation, vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.interp_yface_to_cell(_move_vertical_to_front(v_yxz), grid)
    )


def interp_yface_to_xface_vlast(v_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """y-face → x-face (4-pt corner average), vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.interp_yface_to_xface(_move_vertical_to_front(v_yxz), grid)
    )


def interp_xface_to_yface_vlast(u_yxz: jax.Array, grid: PlaneGrid) -> jax.Array:
    """x-face → y-face (4-pt corner average), vertical-last."""
    return _move_vertical_to_back(
        _plane_ops.interp_xface_to_yface(_move_vertical_to_front(u_yxz), grid)
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

    # Isotropic mixing length: cube-root of cell volume.
    delta = (grid.dx * grid.dy * dz_full) ** (1.0 / 3.0)   # (nlev,)
    delta_sq = (c_s * delta) ** 2                          # (nlev,)
    return delta_sq * strain_mag


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

    # 5. Theta advection (advective form, first-order upwind). Theta
    #    at cell centre; advect with the cell-centre velocity formed
    #    by face→cell averaging of u, v.
    u_center = interp_xface_to_cell_vlast(u, grid)
    v_center = interp_yface_to_cell_vlast(v, grid)
    dtheta_p_dt = (
        _upwind_advection_x(theta_total, u_center, grid.dx)
        + _upwind_advection_y(theta_total, v_center, grid.dy)
    )

    # 6. Horizontal momentum advection — Arakawa-C upwind. u lives at
    #    x-face; advect by (u-at-x-face, v-at-x-face). v→x-face via
    #    4-pt corner average. Symmetric for v.
    v_at_xface = interp_yface_to_xface_vlast(v, grid)
    u_at_yface = interp_xface_to_yface_vlast(u, grid)
    du_adv = (
        _upwind_advection_x(u, u, grid.dx)
        + _upwind_advection_y(u, v_at_xface, grid.dy)
    )
    dv_adv = (
        _upwind_advection_x(v, u_at_yface, grid.dx)
        + _upwind_advection_y(v, v, grid.dy)
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
        _upwind_advection_x(w_full, u_center, grid.dx)
        + _upwind_advection_y(w_full, v_center, grid.dy)
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
                _upwind_advection_x(q, u_center, grid.dx)
                + _upwind_advection_y(q, v_center, grid.dy)
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
