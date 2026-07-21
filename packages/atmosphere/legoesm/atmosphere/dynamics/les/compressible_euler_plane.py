"""Doubly-periodic plane non-hydrostatic compressible Euler — building blocks.

Staged-not-integrated (CRM rollout, PR2a):

* PR2a (this file) ships the *building blocks* needed by the future
  plane non-hydrostatic dycore: vertical-last operator wrappers around
  the PR1 plane stencils, a flat terrain-metric constructor, the
  dry-mass integral, and a uniform-additive mass fixer. None of these
  are wired into any factory dispatch.
* PR2b will add the full ``PlaneCompressibleEulerModel`` class, slow
  tendency, ``plane_acoustic_substeps`` (a thin wrapper around the
  shared :func:`compressible_euler.acoustic_column_kernel`), and the
  dry-validation harness (rising thermal + Straka).

State convention
----------------
All plane prognostic arrays are vertical-LAST (matches
:class:`legoesm.core.state.PlaneNonHydrostaticState` and the existing
cubed-sphere / MPAS convention) so that the shared acoustic column
kernel slices the same axis. PR1 plane operators in
:mod:`legoesm.atmosphere.dynamics.les.plane_operators` expect
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

from legoesm.atmosphere.dynamics.les import plane_operators as _plane_ops
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
    acoustic_column_kernel,
    semi_implicit_acoustic_column_kernel,
    sponge_profile,
    precompute_si_tridiag_bands,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.atmosphere.physics.turbulence.lasd_core import lasd_cs2 as _lasd_cs2
from legoesm.atmosphere.physics.turbulence.vreman import (
    vreman_nu_t as _vreman_nu_t_core)
from legoesm.atmosphere.physics.turbulence.amd import (
    amd_nu_t as _amd_nu_t_core)
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
from legoesm.timestepping.integration import (
    refuse_unthreaded_stateful_physics,
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


def _plane_horizontal_weight(
    grid: PlaneGrid,
    terrain_metric: TerrainMetric,
) -> jax.Array:
    """Shared fixer weight ``(J * area_T)[:, :, None]`` of shape
    ``(ny, nx, 1)``.

    Single source of the mass-fixer weight expression so every mass
    integral below multiplies bit-identical factors in the same order.
    """
    return (terrain_metric.jacobian * grid.area_T)[:, :, None]


def _plane_weighted_mass_sum(
    rho_like: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
) -> jax.Array:
    """``sum(rho_like * J * area_T * dz)`` — the mass-fixer integral.

    ``rho_like`` broadcasts against ``(ny, nx, 1)``; either ``(nlev,)``
    (a reference profile) or ``(ny, nx, nlev)`` (a density field).
    The products associate left-to-right (``(rho_like * w_h) * dz``)
    and the final ``jnp.sum`` reduces one ``(ny, nx, nlev)`` array, so
    every caller — :func:`compute_dry_mass_plane`,
    :func:`_plane_background_mass`, and the anomaly sum in
    :func:`fix_mass_nonhydrostatic_plane` — accumulates in exactly the
    same order. The mass fixer's rest-state bit-exactness relies on
    this shared op order (see :func:`_plane_background_mass`).
    """
    weight_horizontal = _plane_horizontal_weight(grid, terrain_metric)
    return jnp.sum(rho_like * weight_horizontal * height_coord.dz)


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
    return _plane_weighted_mass_sum(
        rho_total, grid, height_coord, terrain_metric,
    )


def _plane_volume_weight(
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
) -> jax.Array:
    """Total weighted volume ``sum(J * area_T * dz)`` used by the fixer."""
    weight_horizontal = _plane_horizontal_weight(grid, terrain_metric)
    return jnp.sum(weight_horizontal * height_coord.dz)


def _plane_background_mass(
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
) -> jax.Array:
    """State-independent part of :func:`compute_dry_mass_plane`:
    ``sum(rho_ref * J * area_T * dz)``.

    Bit-match invariant: this must equal
    ``compute_dry_mass_plane(state, ...)`` BIT-EXACTLY when
    ``state.rho_prime`` is identically zero and both are evaluated
    eagerly. That holds because ``rho_ref + 0.0 == rho_ref`` exactly
    and both route through the SAME :func:`_plane_weighted_mass_sum`
    (same broadcast shapes, same left-to-right association, same
    final ``jnp.sum`` over ``(ny, nx, nlev)``, same cached
    executable). The mass fixer's rest-state exactness
    (``test_one_step_at_rest_with_mass_fixer``) relies on it.
    """
    return _plane_weighted_mass_sum(
        height_coord.rho_ref, grid, height_coord, terrain_metric,
    )


def fix_mass_nonhydrostatic_plane(
    state: PlaneNonHydrostaticState,
    target_mass: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    background_mass: jax.Array | None = None,
) -> PlaneNonHydrostaticState:
    """Restore dry mass to ``target_mass`` via a uniform additive
    correction to ``rho_prime``.

    The correction is computed in ANOMALY form, cancelling the
    state-independent background integral ``B = sum(rho_ref * J *
    area_T * dz)`` analytically instead of numerically:

        delta = ((target_mass - B) - sum(rho' * J * area_T * dz))
                / sum(J * area_T * dz)
        rho_prime' = rho_prime + delta

    This is algebraically identical to the naive
    ``(target_mass - current_mass) / V`` (``current_mass = B +
    anomaly``) but avoids subtracting two ~1e11-scale totals whose
    true difference is near zero. The naive form is NOT exact at rest:
    the traced ``current_mass`` reduce gets fused into the step graph
    by XLA and can land 1 ulp away from the eagerly computed
    ``target_mass``, injecting a spurious ``ulp(M)/V`` (~1e-16 kg/m³)
    uniform shift into a bit-zero ``rho_prime`` every step (caught by
    ``test_one_step_at_rest_with_mass_fixer``). In anomaly form the
    only traced reduction sums ``rho' * w`` — exactly zero in any
    fusion/reduction order when ``rho'`` is bit-zero — while
    ``target_mass - B`` subtracts two eagerly computed scalars that
    are bit-identical at rest (see :func:`_plane_background_mass`).
    For ``B/2 <= target_mass <= 2B`` (always, since |rho'| << rho_ref)
    the subtraction is exact by Sterbenz's lemma, so the anomaly form
    is also strictly more accurate off rest. This refines the MPAS
    pattern in
    :func:`compressible_euler_mpas.fix_mass_nonhydrostatic_mpas`.

    ``background_mass`` lets the caller pass an EAGERLY precomputed
    ``B`` (``PlaneCompressibleEulerModel`` computes it once in
    ``__init__`` and threads it through ``_step_jit``). This matters
    under ``jax.jit``: omnistaging stages even the all-constant
    ``B`` reduce into the graph, and XLA's constant folding
    (HloEvaluator) can accumulate in a different order than the eager
    executable — measured 6 ulps of ``M`` on the 4x4x6 rest test —
    reintroducing a spurious shift. An eagerly computed ``B`` is
    embedded as a literal constant with its bits preserved. When
    ``background_mass`` is ``None`` (eager utility callers, e.g. the
    single-rank MPI short-circuit and unit tests), it is computed
    inline, which is bit-safe outside a trace.

    Sign convention: mass deficit (``current < target``) gives
    ``delta > 0`` → ``rho'`` increases → mass increases by
    ``delta * V = target - current``; the budget closes exactly.

    Differentiability
    -----------------
    The fixer is a pure additive correction with no ``jnp.where`` /
    clip / branch, so ``jax.grad`` flows through cleanly. The anomaly
    sum carries the same linear dependence on ``rho_prime`` as the
    total-mass form (``d delta / d rho'[j,i,k] = -J*area_T*dz/V``),
    so gradients are unchanged.

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
    # Static feature-gate style Python branch on a non-traced value
    # (background_mass is either None or a concrete array — never a
    # tracer in any caller).
    if background_mass is None:
        background_mass = _plane_background_mass(
            grid, height_coord, terrain_metric,
        )
    anomaly = _plane_weighted_mass_sum(
        state.rho_prime.data, grid, height_coord, terrain_metric,
    )
    weighted_volume = _plane_volume_weight(grid, height_coord, terrain_metric)
    delta = ((target_mass - background_mass) - anomaly) / weighted_volume
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
    "centered": 1,    # 2nd-order centred (gSAM advect2_mom); momentum-leg only
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
    from legoesm.core.flux_limiters import van_leer_face_values
    f = field_yxz
    # Face i+1/2 reconstruction (HD-1-correct r sign) via the shared helper —
    # stencil [f[i-1], f[i], f[i+1], f[i+2]].
    phi_pos, phi_neg = van_leer_face_values(
        jnp.roll(f, 1, axis=1), f, jnp.roll(f, -1, axis=1), jnp.roll(f, -2, axis=1))
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
    from legoesm.core.flux_limiters import van_leer_face_values
    f = field_yxz
    phi_pos, phi_neg = van_leer_face_values(
        jnp.roll(f, 1, axis=0), f, jnp.roll(f, -1, axis=0), jnp.roll(f, -2, axis=0))
    v_face_R = 0.5 * (v_at_field + jnp.roll(v_at_field, -1, axis=0))
    phi_R = jnp.where(v_face_R >= 0.0, phi_pos, phi_neg)
    flux_R = v_face_R * phi_R
    flux_L = jnp.roll(flux_R, 1, axis=0)
    v_face_L = jnp.roll(v_face_R, 1, axis=0)
    return -(flux_R - flux_L) / dy + f * (v_face_R - v_face_L) / dy


def _centered_advection_x(
    field_yxz: jax.Array, u_at_field: jax.Array, dx: float,
) -> jax.Array:
    """2nd-order CENTERED flux-form advection ``-u df/dx`` = gSAM `advect2_mom`.

    ADV-SPLIT #86 (codex iter-68): a 2nd-order CENTERED, NON-diffusive momentum
    scheme matching SAM's face reconstruction (``advect2_mom_xy.f90:27``:
    ``flux=0.25·(c_face)·(φ_face)`` — no upwind bias, no flux limiter). The TVD
    van_leer default adds limiter diffusion that SUPPRESSES convective updraft
    cores / w-variance tails (CONFIRMED: identical-IC max|w| caps ~2.6 m/s vs
    low-diffusion ~5-15; centered restores ~5 in a gray burst). For SAM-faithful
    convective EXTREMES the momentum legs (u/v/w) use this scheme; the SCALAR legs
    keep van_leer (monotone ≈ MPDATA, positivity). NOTE (codex iter-68 [S1]): this
    uses the ADVECTIVE form ``−d(uf)/dx + f·du/dx`` (same identity as van_leer/
    weno5 — preserves a constant f under divergent flow), which is NOT proven
    discretely KE-conserving like SAM's pure-flux ``advect2_mom`` — call it
    "centered non-diffusive matching SAM's face reconstruction", not "energy-
    conserving". DISPERSIVE (2Δ modes) — relies on the ∇⁴ hyperdiff + Smagorinsky
    SGS to control grid-scale noise (as SAM relies on SGS + monotone scalars).
    **Stability validated only to ~5 m/s (gray burst); high-k KE growth at a
    10-15 m/s RRTM burst is UNPROVEN (codex [S2]) ⇒ OPT-IN, not a default yet.**
    Face value = centred 2-pt avg; stencil [i-1, i, i+1] ⇒ halo 1 (≤ van_leer's 2).
    """
    f = field_yxz
    phi_R = 0.5 * (f + jnp.roll(f, -1, axis=1))         # centred face i+1/2
    u_face_R = 0.5 * (u_at_field + jnp.roll(u_at_field, -1, axis=1))
    flux_R = u_face_R * phi_R
    flux_L = jnp.roll(flux_R, 1, axis=1)
    u_face_L = jnp.roll(u_face_R, 1, axis=1)
    return -(flux_R - flux_L) / dx + f * (u_face_R - u_face_L) / dx


def _centered_advection_y(
    field_yxz: jax.Array, v_at_field: jax.Array, dy: float,
) -> jax.Array:
    """2nd-order CENTERED flux-form advection ``-v df/dy`` (axis=0). Momentum-leg
    counterpart of :func:`_centered_advection_x` (= gSAM `advect2_mom`)."""
    f = field_yxz
    phi_R = 0.5 * (f + jnp.roll(f, -1, axis=0))
    v_face_R = 0.5 * (v_at_field + jnp.roll(v_at_field, -1, axis=0))
    flux_R = v_face_R * phi_R
    flux_L = jnp.roll(flux_R, 1, axis=0)
    v_face_L = jnp.roll(v_face_R, 1, axis=0)
    return -(flux_R - flux_L) / dy + f * (v_face_R - v_face_L) / dy


# --------------------------------------------------------------------- #
# WENO5-Z flux-form upwind advection (5th-order; NOT monotone /         #
# NOT positivity-preserving — a positive 6-cell stencil can still       #
# reconstruct a negative face value. Safe for signed fields (θ′,        #
# momentum); positive-definite tracers are guarded onto van_leer at     #
# the dispatch below).                                                  #
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


def _vertical_K_diffusion_full(
    field_yxz: jax.Array, K_yxz: jax.Array, height_coord: HeightCoordinate,
) -> jax.Array:
    """Conservative vertical SGS flux ``∂_z(K ∂_z field)`` for FULL-level fields.

    SGS-VERT (#81): SAM's SGS is fully 3D — ``diffuse_scalar``/``diffuse_mom``
    add the VERTICAL flux on top of the horizontal one. The plane CRM previously
    applied the horizontal flux (:func:`_variable_K_diffusion_vlast`) ONLY,
    under-mixing the vertical sub-grid transport of u/v/θ'/tracers
    (entrainment-detrainment, BL mixing).

    MASS-WEIGHTED flux-form (codex iter-64 [HIGH] — confirmed against gSAM
    ``diffuse_scalar_z.f90:56`` / ``diffuse_mom3D.f90:113,169``: SAM weights the
    vertical flux by the reference density, ``∂_t φ = (1/ρ)·∂_z(ρ_w·K·∂_z φ)``,
    NOT the plain ``∂_z(K ∂_z φ)``). Vertical — unlike horizontal — cannot hide
    the ``ρ(z)`` factor, so it must be carried explicitly::

        F_{k+1/2} = ρ_w[k+1/2] · K_{k+1/2} · (f_{k+1} − f_k) / dz_half[k]
        tend_k    = (F_{k+1/2} − F_{k-1/2}) / (ρ_ref[k] · dz_full[k])

    with full-level K averaged to the interface ``K_{k+1/2}=0.5(K_k+K_{k+1})``,
    ``ρ_w`` the interface reference density (``rho_ref_half``) and ``ρ_ref`` the
    cell density. NO-FLUX domain boundaries (``F=0`` at top+surface) — interior
    SGS only; surface/top fluxes are injected SEPARATELY by the surface scheme,
    matching SAM where ``diffuse_scalar`` adds ``fluxb``/``fluxt`` to the
    boundary cell (``diffuse_scalar_z.f90:61``) rather than through the interior
    stencil — so a no-flux interior stencil avoids double-counting.

    Conservative in the MASS measure: ``Σ_k tend_k·ρ_ref[k]·dz_full[k] = F_top −
    F_bot = 0``. Dissipative: ``Σ_k f_k·tend_k·ρ_ref[k]·dz_full[k] = −Σ ρ_w·K·
    (f_{k+1}−f_k)²/dz_half ≤ 0`` for K≥0. (Uses the anelastic reference density
    ρ₀(z) like SAM; the ρ' part is second-order and absent from SAM's anelastic
    SGS.) ``field_yxz``/``K_yxz`` are full-level ``(ny, nx, nlev)``.
    """
    dz_full = height_coord.dz                       # (nlev,) cell thickness
    dz_half = height_coord.dz_half                  # (nlev-1,) full→full dist
    rho_full = height_coord.rho_ref                 # (nlev,) cell ρ₀
    # interior interfaces sit at z_half[1..nlev-1] ⇒ rho_ref_half[1:-1].
    rho_iface = height_coord.rho_ref_half[1:-1]     # (nlev-1,) interface ρ₀
    # K at the nlev-1 interior interfaces (average adjacent full-level K).
    K_iface = 0.5 * (K_yxz[..., :-1] + K_yxz[..., 1:])          # (..., nlev-1)
    flux = rho_iface * K_iface * (
        field_yxz[..., 1:] - field_yxz[..., :-1]) / dz_half
    # No-flux at the top + bottom domain interfaces ⇒ pad the interface flux
    # with a zero at each end before differencing back to full levels.
    pad_axes = ((0, 0),) * (flux.ndim - 1)
    flux_padded = jnp.pad(flux, (*pad_axes, (1, 1)))           # (..., nlev+1)
    return (flux_padded[..., 1:] - flux_padded[..., :-1]) / (rho_full * dz_full)


def _vertical_K_diffusion_w(
    w_yxz: jax.Array, K_full: jax.Array, height_coord: HeightCoordinate,
) -> jax.Array:
    """Vertical SGS flux ``∂_z(K ∂_z w)`` for the HALF-level ``w`` (SGS-VERT #81).

    Dual-grid counterpart of :func:`_vertical_K_diffusion_full`: ``w`` lives at
    the ``nlev+1`` interfaces (``z_half``), ``K_m`` at the ``nlev`` cell centres
    (``z_full``). The vertical w-gradient and its flux therefore live at the
    CELL CENTRES, and the flux divergence returns to the interfaces. MASS-WEIGHTED
    like SAM ``diffuse_mom3D.f90:112,179`` — the w leg weights the cell-centre
    flux by the CELL density ρ_ref and normalises the interface divergence by the
    INTERFACE density ρ_w (the mirror of the scalar/u/v weighting)::

        F_k          = ρ_ref[k] · K_k · (w_{k+1} − w_k) / dz_full[k]   (centre k)
        tend_{k+1/2} = (F_{k+1} − F_k) / (ρ_w[k+1/2] · dz_half[k])      (iface)

    The rigid top/bottom boundaries (``w=0`` there) are preserved by padding the
    interior-interface tendency with zeros — the boundary w is held by the BC,
    not diffused. Dissipative for the resolved KE in the mass measure
    (``Σ w·tend·ρ_w·dz_half ≤ 0``).
    """
    dz_full = height_coord.dz                       # (nlev,) cell thickness
    dz_half = height_coord.dz_half                  # (nlev-1,) full→full dist
    rho_full = height_coord.rho_ref                 # (nlev,) cell ρ₀
    rho_iface = height_coord.rho_ref_half[1:-1]     # (nlev-1,) interior iface ρ₀
    # ρ-weighted SGS flux F = ρ_ref · K · ∂_z w at cell centres.
    flux_full = rho_full * K_full * (
        w_yxz[..., 1:] - w_yxz[..., :-1]) / dz_full       # (..., nlev)
    # Divergence back to the nlev-1 INTERIOR interfaces, normalised by the
    # interface density; the two rigid boundary interfaces (w=0) get zero.
    tend_interior = (
        flux_full[..., 1:] - flux_full[..., :-1]) / (rho_iface * dz_half)
    pad_axes = ((0, 0),) * (tend_interior.ndim - 1)
    return jnp.pad(tend_interior, (*pad_axes, (1, 1)))         # (..., nlev+1)


def safe_sqrt_strain(strain_mag_sq: jax.Array) -> jax.Array:
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


def sgs_brunt_vaisala_sq(
    theta_total: jax.Array,
    tracers: jax.Array,
    height_coord: HeightCoordinate,
    phase_blend_width_K: float = 20.0,
    sat_blend_rel_width: float = 0.02,
) -> jax.Array:
    """Sub-grid Brunt–Väisälä frequency ``N²`` for the SAM ``dosmagor``
    stratification correction, switching between a CLEAR (virtual-θ) and a
    SATURATED (moist-adiabatic) formula per grid point.

    SAM faithfulness (``SGS_TKE/tke_full.f90``)
    -------------------------------------------
    SAM forms ``buoy_sgs`` at each interface in two regimes and feeds it
    to ``tk = (Cs·Δ)²·sqrt(max(0, def2 − Pr·buoy_sgs))`` (line 298):

    - **Unsaturated** (``tke_full.f90:154-159``): virtual-temperature
      buoyancy gradient with vapour + condensate loading.  In potential-
      temperature form, ``N²_dry = (g/θ_v)·∂θ_v/∂z`` with
      ``θ_v = θ·(1 + ε_v·q_v − Σq_cond)``, ``ε_v = 1/ε − 1 ≈ 0.61``.
    - **Saturated** (``tke_full.f90:198-226``): when the interface total
      water exceeds saturation, SAM switches to the moist (cloudy)
      buoyancy that accounts for latent-heat release on adiabatic ascent.
      SAM's ``lstarn``/``dqsat`` algebra is the discretised Durran–Klemp
      (1982, JAS 39, Eq. 36) saturated ``N²``:

          N²_moist = g·[ A·(∂lnθ/∂z + (L/(c_pd·T))·∂q_s/∂z) − ∂q_w/∂z ]
          A = (1 + L·q_s/(R_d·T)) / (1 + ε·L²·q_s/(c_pd·R_d·T²))

      ``L`` = latent heat, ``q_s`` the saturation mixing ratio, ``q_w``
      total water.  The saturated ``N²`` is SMALLER (often negative) than
      the dry value, so the ``dosmagor`` shutoff lets the LES MIX inside
      cloud (conditional instability) while still damping the clear,
      stable environment.  Without this switch the closure over-damps
      cloudy interfaces exactly where the CRM is meant to resolve
      convective overturning.

    Deliberate departures from SAM (differentiability + robustness)
    ---------------------------------------------------------------
    legoESM is end-to-end differentiable, so three SAM choices that are
    fine in a forward-only Fortran model are replaced by smooth,
    AD-safe surrogates (Codex adversarial-review iter-2):

    - **Phase split is TEMPERATURE-based**, not SAM's condensate ratio
      ``ω = q_c/(q_c+q_i)``.  ``ω(q_c,q_i)`` has an unbounded adjoint as
      ``q_c+q_i → 0`` and injects a spurious ``(q_sw−q_si)·∂ω/∂z``
      hydrometeor-phase term into ``∂q_s/∂z``.  We use the shared
      :func:`legoesm.thermo.saturation_mixing_ratio_blend` (``w_liq(T)``
      ramp) so ``q_s`` and ``L`` are smooth functions of ``T`` and
      ``∂q_s/∂z`` is a clean thermodynamic gradient on the moist adiabat.
    - **Clear↔moist transition is a SMOOTH sigmoid** in fractional
      supersaturation ``(q_nonprecip − q_s)/q_s`` (width
      ``sat_blend_rel_width``), not SAM's hard ``qtot > qsat`` branch
      (``tke_full.f90:198``).  A hard ``where`` branch-selects the
      gradient at cloud edges; the narrow ramp tracks SAM's sharp switch
      in the forward pass while keeping adjoints finite and smooth.

    Known faithfulness caveats (logged, acceptable):
    - The switch is evaluated at CELL CENTRES (legoESM stores ``N²`` at
      centres) whereas SAM tests at INTERFACES — the transition can sit
      within half a level of SAM's, where ``dosmagor`` is sensitive.
    - ``T``/``p`` come from the REFERENCE Exner profile
      (``T = θ·Π_ref``, ``p = p_ref·Π_ref^(1/κ)``; SAM uses base-state
      ``presi``).  This ignores pressure perturbations, so ``q_s`` and
      the switch can bias in strong cold pools / compressed regions.

    ``q_s`` uses the shared :mod:`legoesm.thermo` saturation routines
    (CLAUDE.md — no re-implemented Clausius–Clapeyron).  ``N²_moist``
    needs only ``q_s`` and its vertical gradient, NOT ``dq_s/dT`` — the
    Durran–Klemp ``A`` factor is algebraic in ``(L, q_s, T)``.

    Parameters
    ----------
    theta_total : jax.Array
        Dry potential temperature ``θ = θ_ref + θ'`` at cell centres,
        shape ``(ny, nx, nlev)``.
    tracers : jax.Array
        Tracer array ``(ny, nx, nlev, n_tracers)``; slot 0 = ``q_v``,
        slots 1.. = condensate mass (q_c, q_r, q_i, q_s, q_g) then number
        concentrations.  Robust to ``n_tracers < 6`` (warm-rain layouts):
        absent ice/precip slots are treated as zero.
    height_coord : HeightCoordinate
        Reference Exner (``exner_ref``) for T/p and ``dz_half`` for the
        vertical gradients (via :func:`full_level_centred_d_dz`).
    phase_blend_width_K : float
        Width [K] of the temperature ramp that splits liquid/ice for the
        mixed-phase ``q_s`` and ``L`` (``w_liq = 1`` above
        ``T_freeze``, ``0`` below ``T_freeze − width``).  Matches the
        :func:`legoesm.thermo.saturation_mixing_ratio_blend` default so
        ``q_s`` and ``L_eff`` use one consistent ramp.
    sat_blend_rel_width : float
        Fractional-supersaturation width of the smooth clear↔moist
        sigmoid: ``f_moist = σ((q_nonprecip − q_s)/(width·q_s))``.
        Small (default 2 %) so the forward transition stays close to
        SAM's hard switch while the adjoint stays finite.

    Returns
    -------
    jax.Array
        ``N²_sgs`` at cell centres ``(ny, nx, nlev)`` [1/s²].
    """
    from legoesm import constants
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.thermo import saturation_mixing_ratio_blend

    # ``full_level_centred_d_dz`` returns the LEVEL-INDEX derivative =
    # −∂/∂z_physical under top-down storage, so every N² carries a leading
    # minus to recover the physical-z stratification (matches the dry term
    # the SMAG-1 commit introduced and the kernel docstring).
    n_tr = tracers.shape[-1]
    if n_tr == 0:
        dtheta_dlev = full_level_centred_d_dz(theta_total, height_coord)
        return -(constants.g / jnp.clip(theta_total, 1.0, None)) * dtheta_dlev

    q_v = tracers[..., 0]
    n_cond = min(n_tr, 6)                      # slots 1..5 = condensate mass
    if n_cond > 1:
        q_cond = jnp.sum(tracers[..., 1:n_cond], axis=-1)
    else:
        q_cond = jnp.zeros_like(q_v)
    q_w = q_v + q_cond                         # total water mixing ratio

    # Cloud condensate (cloud water slot 1, cloud ice slot 3) sets the
    # non-precip total water for the saturation switch; precip slots
    # (2,4,5) load buoyancy via q_w but do not by themselves saturate.
    q_c = tracers[..., 1] if n_tr > 1 else jnp.zeros_like(q_v)
    q_i = tracers[..., 3] if n_tr > 3 else jnp.zeros_like(q_v)

    exner = height_coord.exner_ref                             # (nlev,)
    T = theta_total * exner                                    # (ny,nx,nlev)
    p = constants.p_ref * exner ** (1.0 / constants.kappa)     # (nlev,)

    # Clear-air virtual-θ N² (SAM unsaturated branch).
    theta_v = virtual_temperature(theta_total, q_v) - theta_total * q_cond
    n2_dry = -(constants.g / jnp.clip(theta_v, 1.0, None)) * (
        full_level_centred_d_dz(theta_v, height_coord)
    )

    # Saturated moist-adiabatic N² (Durran–Klemp 1982).  TEMPERATURE-based
    # mixed phase (w_liq ramp) keeps q_s / L smooth + differentiable and
    # makes ∂q_s/∂z thermodynamic (no q_cloud-in-denominator blow-up, no
    # hydrometeor-phase ∂ω/∂z artifact — Codex iter-2).
    q_sat = saturation_mixing_ratio_blend(
        T, p, T_blend_width=phase_blend_width_K,
    )
    w_liq = jnp.clip(
        (T - (constants.T_freeze - phase_blend_width_K)) / phase_blend_width_K,
        0.0, 1.0,
    )
    L_eff = w_liq * constants.L_v + (1.0 - w_liq) * constants.L_s
    A_moist = (
        1.0 + L_eff * q_sat / (constants.R_d * T)
    ) / (
        1.0
        + constants.epsilon * L_eff ** 2 * q_sat
        / (constants.c_pd * constants.R_d * T ** 2)
    )
    ln_theta = jnp.log(jnp.clip(theta_total, 1.0, None))
    n2_moist = -constants.g * (
        A_moist * (
            full_level_centred_d_dz(ln_theta, height_coord)
            + (L_eff / (constants.c_pd * T))
            * full_level_centred_d_dz(q_sat, height_coord)
        )
        - full_level_centred_d_dz(q_w, height_coord)
    )

    # Smooth clear↔moist blend (differentiable surrogate for SAM's hard
    # qtot>qsat switch, tke_full.f90:198).  σ in fractional supersaturation
    # so adjoints stay finite across cloud edges; narrow ramp tracks the
    # sharp forward switch.  q_nonprecip = q_v + cloud (SAM non-precip).
    q_nonprecip = q_v + q_c + q_i
    q_sat_safe = jnp.clip(q_sat, 1.0e-12, None)
    f_moist = jax.nn.sigmoid(
        (q_nonprecip - q_sat) / (sat_blend_rel_width * q_sat_safe)
    )
    return f_moist * n2_moist + (1.0 - f_moist) * n2_dry


def _compute_smagorinsky_K_m_plane(
    u_yxz: jax.Array,
    v_yxz: jax.Array,
    w_yxz_half: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    c_s: float,
    n2_sgs: jax.Array | None = None,
    prandtl: float = 1.0,
    wall_damping: bool = True,
    delta_max: float = 1.0e30,
    stability_length: bool = False,
) -> jax.Array:
    """Full 3D Smagorinsky-Lilly eddy viscosity on the C-grid plane.

    ``K_m = (C_s · Δ)² · sqrt(max(0, |S|² − Pr · N²))`` with isotropic
    mixing length ``Δ = (dx · dy · dz)^(1/3)`` and FULL strain-rate
    magnitude ``|S|² = 2 S_ij S_ij`` including vertical-shear components
    ``S_13``, ``S_23``, ``S_33``. Returned at cell centres.

    Stratification (Lilly) correction — SAM faithfulness
    ----------------------------------------------------
    SAM's ``dosmagor`` closure forms ``tk = (Cs·Δ)² · sqrt(max(0,
    def2 − Pr·buoy_sgs))`` (``SGS_TKE/tke_full.f90:298``), subtracting
    the sub-grid buoyancy frequency ``N² = buoy_sgs`` inside the sqrt so
    that mixing SHUTS OFF in stably-stratified layers (trade inversion,
    cloud tops, free troposphere, tropopause) and is ENHANCED where the
    column is statically unstable.  ``def2`` in SAM is ``2 S_ij S_ij``
    — the SAME convention as ``strain_mag_sq`` here.  Pass the precomputed
    sub-grid ``N²`` (clear-vs-moist switched) from
    :func:`sgs_brunt_vaisala_sq` as ``n2_sgs`` to activate the term;
    ``n2_sgs=None`` reduces the closure to the pure-strain form (used by
    the dry-strain unit tests).  Keeping ``N²`` in a single shared helper
    (rather than inline here AND in the halo kernel) prevents serial/MPI
    divergence.  Without it the closure over-mixes every stratified layer,
    which directly corrupts the domain-mean ``θ``/``q`` profiles this CRM
    is validated against.

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
    n2_sgs : jax.Array | None
        Precomputed sub-grid Brunt–Väisälä frequency ``N²`` at cell
        centres, shape ``(ny, nx, nlev)`` (from
        :func:`sgs_brunt_vaisala_sq`, which switches between clear and
        saturated stratification).  When provided, the SAM correction
        ``− Pr · N²`` is subtracted inside the strain sqrt.  ``None``
        (default) recovers the pure-strain Smagorinsky form.
    prandtl : float
        Turbulent Prandtl number ``Pr`` multiplying ``N²`` in the
        stratification term — matches SAM's ``def2 − Pr·buoy_sgs``
        (``Pr = 1`` in SAM ``dosmagor``).

    Returns
    -------
    K_m : jax.Array
        Eddy viscosity at cell centres, shape ``(ny, nx, nlev)``.
    """
    u_yxz.shape[-1]

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
    # (Level-index difference = −∂w/∂z_physical under top-down storage; S33
    # enters |S|² only SQUARED, so the sign is inert HERE — do not reuse this
    # value where a signed ∂w/∂z_physical is required.)
    dz_full = height_coord.dz                          # (nlev,)
    dw_dz_center = (
        w_yxz_half[..., 1:] - w_yxz_half[..., :-1]
    ) / dz_full
    S33_center = dw_dz_center

    # ∂u/∂z at x-face, vertical full-level: needs interior centred
    # difference of u between full levels k+1, k-1 (centred). Edges
    # use one-sided one-level differences.
    # Sign convention: z is positive UP, level index k runs TOP→DOWN, so
    # full_level_centred_d_dz returns −∂/∂z_physical. Negate to the physical
    # sign: S13/S23 MIX these with the already-physical ∂w/∂x, ∂w/∂y, so the
    # sign does NOT square out — the 2·(∂u/∂z)(∂w/∂x) cross term in S13² flips.
    du_dz = -full_level_centred_d_dz(u_yxz, height_coord)  # +∂u/∂z_phys, x-face
    dv_dz = -full_level_centred_d_dz(v_yxz, height_coord)  # +∂v/∂z_phys, y-face

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
    # ``strain_mag`` (the sqrt) is deferred to AFTER the optional
    # stratification subtraction below so the SAM ``max(0, def2 −
    # Pr·N²)`` shutoff lives inside a single AD-safe ``_safe_sqrt``.

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
    # SAM-faithful cap on the horizontal grid spacing in the mixing length
    # (SGS_TKE/tke_full.f90:42 coef=min(delta_max,dx·mu)·min(delta_max,dy·ady),
    # delta_max=1000 m): without it the isotropic Δ over-grows on coarse grids
    # (dx>1 km) and the SGS over-mixes the resolved convective variance.
    dx_eff = jnp.minimum(delta_max, grid.dx)
    dy_eff = jnp.minimum(delta_max, grid.dy)
    delta = (dx_eff * dy_eff * dz_full) ** (1.0 / 3.0)     # (nlev,)
    l_smag = c_s * delta                                   # (nlev,)
    # ``wall_damping`` (static config bool) caps the length at the von Kármán
    # wall scaling (Mason 1989). SAM's ``dosmagor`` does NOT (``smix=grd``; the
    # ``tke_full.f90`` wall correction is commented out) — so the SAM-faithful
    # CRM runs set ``smagorinsky_wall_damping=False`` ⇒ ``l_m = c_s·Δ`` at every
    # level (the cap is active only in the lowest cell where ``κz < c_s·Δ``).
    if wall_damping:
        l_wall = constants.kappa_von_karman * height_coord.z_full   # (nlev,)
        l_m = jnp.minimum(l_smag, l_wall)
    else:
        l_m = l_smag
    l_m_sq = l_m ** 2                                      # (nlev,)

    # --- SAM dosmagor stratification (Lilly) correction ---
    # K_m = (Cs·Δ)² · sqrt(max(0, |S|² − Pr·N²)).  ``safe_sqrt_strain``
    # applies the ``max(0, ·)`` shutoff AND keeps ``d/dx sqrt(0) = 0``,
    # so a strongly-stable column (N² ≫ |S|²) yields K_m = 0 exactly,
    # matching SAM ``tke_full.f90:298``.  N² is precomputed by the shared
    # ``sgs_brunt_vaisala_sq`` helper (clear↔moist switched) so the serial
    # and halo kernels stay bit-identical.
    if n2_sgs is not None:
        strain_arg = strain_mag_sq - prandtl * n2_sgs
    else:
        strain_arg = strain_mag_sq
    strain_mag = safe_sqrt_strain(strain_arg)
    if stability_length and n2_sgs is not None:
        # SAM dosmagor stable-layer Deardorff mixing-length limit
        # (SGS_TKE/tke_full.f90:285-298): in STABLE layers (N²>0) shrink the
        # mixing length smix=min(grd, max(0.1·grd, √(0.76·tk/(Ck·√N²)))) and
        # vary Cee=Ce1+Ce2·(smix/grd); the eddy viscosity becomes
        # tk=√(Ck³/Cee·(|S|²−Pr·N²))·smix² (Ck=0.1, Ce=Ck³/Cs⁴,
        # Ce1=Ce/0.7·0.19, Ce2=Ce/0.7·0.51). SAM lags tk across the step; the
        # diagnostic plane uses the same-step smix=grd estimate (predictor).
        # In UNSTABLE layers (N²≤0) smix=grd ⇒ reduces EXACTLY to the default
        # (Cs·grd)²·|S| form. Use with wall_damping=False (SAM dosmagor caps
        # neither length); grd is the isotropic delta (delta_max-capped).
        Ck = 0.1
        Ce = Ck ** 3 / c_s ** 4
        tk_grd = c_s ** 2 * delta ** 2 * strain_mag        # smix=grd estimate
        n2_pos = jnp.maximum(n2_sgs, 1.0e-10)
        smix_stable = jnp.minimum(delta, jnp.maximum(
            0.1 * delta,
            jnp.sqrt(0.76 * tk_grd / (Ck * jnp.sqrt(n2_pos)))))
        smix = jnp.where(n2_sgs > 0.0, smix_stable, delta)
        ratio = smix / jnp.clip(delta, 1.0e-12, None)
        Cee = Ce / 0.7 * (0.19 + 0.51 * ratio)
        return jnp.sqrt(Ck ** 3 / Cee) * smix ** 2 * strain_mag
    return l_m_sq * strain_mag


def _horizontal_box_filter_plane(field_yxz: jax.Array) -> jax.Array:
    """Separable 3-point top-hat TEST filter over the periodic horizontal
    (y, x) directions — the Germano dynamic-Smagorinsky test filter at width
    ratio α=2 relative to the grid filter. Vertical is left UNfiltered (the
    LES is homogeneous only in the horizontal, so the dynamic average + test
    filter act in (x, y) — standard for atmospheric boundary-layer LES)."""
    fx = (jnp.roll(field_yxz, 1, axis=1) + field_yxz
          + jnp.roll(field_yxz, -1, axis=1)) / 3.0
    return (jnp.roll(fx, 1, axis=0) + fx + jnp.roll(fx, -1, axis=0)) / 3.0


def _centre_velocities_and_strain_plane(
    u_yxz: jax.Array,
    v_yxz: jax.Array,
    w_yxz_half: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
):
    """A-grid cell-centre velocities + full resolved strain tensor ``S_ij``.

    Shared by BOTH dynamic-Smagorinsky paths (standard Germano
    :func:`_compute_dynamic_smag_cs_plane` and scale-dependent
    :func:`_compute_scale_dependent_dynamic_smag_cs_plane`) so the two closures
    diagnose ``C_s`` from byte-identical resolved strain. Centred A-grid
    differences in the horizontal; the existing full-level centred operator in
    the vertical. ``Smag = √(2 S_ij S_ij)`` (the SAM/jax-alfa convention)."""
    (uc, vc, wc, dudx, dudy, dudz, dvdx, dvdy, dvdz,
     dwdx, dwdy, dwdz) = _velocity_gradients_plane(
        u_yxz, v_yxz, w_yxz_half, grid, height_coord)
    S11, S22, S33 = dudx, dvdy, dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)
    Smag = jnp.sqrt(jnp.maximum(
        2.0 * (S11 ** 2 + S22 ** 2 + S33 ** 2
               + 2.0 * (S12 ** 2 + S13 ** 2 + S23 ** 2)),
        1.0e-30))
    return uc, vc, wc, S11, S22, S33, S12, S13, S23, Smag


def _velocity_gradients_plane(u_yxz, v_yxz, w_yxz_half, grid, height_coord):
    """A-grid cell-centre velocities + the NINE resolved velocity gradients
    ``a_cd = ∂u_c/∂x_d`` at cell centres (centred ``jnp.roll`` differences in the
    horizontal, the full-level centred operator in the vertical). Shared by
    :func:`_centre_velocities_and_strain_plane` (which symmetrises to S_ij) and the
    Vreman closure (which needs the asymmetric tensor)."""
    uc = 0.5 * (u_yxz + jnp.roll(u_yxz, -1, axis=1))
    vc = 0.5 * (v_yxz + jnp.roll(v_yxz, -1, axis=0))
    wc = 0.5 * (w_yxz_half[..., :-1] + w_yxz_half[..., 1:])
    dudx = (jnp.roll(uc, -1, axis=1) - jnp.roll(uc, 1, axis=1)) / (2.0 * grid.dx)
    dvdx = (jnp.roll(vc, -1, axis=1) - jnp.roll(vc, 1, axis=1)) / (2.0 * grid.dx)
    dwdx = (jnp.roll(wc, -1, axis=1) - jnp.roll(wc, 1, axis=1)) / (2.0 * grid.dx)
    dudy = (jnp.roll(uc, -1, axis=0) - jnp.roll(uc, 1, axis=0)) / (2.0 * grid.dy)
    dvdy = (jnp.roll(vc, -1, axis=0) - jnp.roll(vc, 1, axis=0)) / (2.0 * grid.dy)
    dwdy = (jnp.roll(wc, -1, axis=0) - jnp.roll(wc, 1, axis=0)) / (2.0 * grid.dy)
    # z is positive UP, level index k runs TOP→DOWN ⇒ full_level_centred_d_dz
    # returns −∂/∂z_physical. Negate so this helper honours its contract
    # (physically-signed a_cd = ∂u_c/∂x_d): the mixed strains S13/S23 and the
    # Vreman/AMD/Germano gradient PRODUCTS are odd in these entries, so the
    # sign does not square out downstream.
    dudz = -full_level_centred_d_dz(uc, height_coord)   # +∂u/∂z_phys
    dvdz = -full_level_centred_d_dz(vc, height_coord)   # +∂v/∂z_phys
    dwdz = -full_level_centred_d_dz(wc, height_coord)   # +∂w/∂z_phys
    return uc, vc, wc, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz


def _compute_vreman_K_m_plane(u_yxz, v_yxz, w_yxz_half, grid, height_coord,
                              c_vreman):
    """Vreman (2004) eddy viscosity ``K_m`` at cell centres for the plane CRM —
    OPTIONAL alternative to :func:`_compute_smagorinsky_K_m_plane` (the SAM-faithful
    default). Computes the nine A-grid velocity gradients and defers the algebra to
    the shared :func:`legoesm.atmosphere.physics.turbulence.vreman.vreman_nu_t`
    (the SAME core the spectral LES uses). Per-direction filter widths
    (Δx, Δy, Δz(z)) make it well-behaved on anisotropic grids; it is purely local
    (no plane average) so it is MPI-safe. No N² cutoff / wall cap (Vreman already
    vanishes in laminar/near-wall 1-D shear)."""
    (uc, vc, wc, dudx, dudy, dudz, dvdx, dvdy, dvdz,
     dwdx, dwdy, dwdz) = _velocity_gradients_plane(
        u_yxz, v_yxz, w_yxz_half, grid, height_coord)
    # a_cd = ∂u_c/∂x_d ; dz per-level (nlev,) broadcasts on the trailing axis.
    return _vreman_nu_t_core(
        dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz,
        grid.dx, grid.dy, height_coord.dz, c_vreman)


def _compute_amd_K_m_plane(u_yxz, v_yxz, w_yxz_half, grid, height_coord,
                           c_amd):
    """Anisotropic Minimum-Dissipation eddy viscosity ``K_m`` at cell centres —
    OPTIONAL alternative to :func:`_compute_smagorinsky_K_m_plane`. Computes the
    nine A-grid velocity gradients (the SAME helper Vreman uses) and defers the
    algebra to the shared
    :func:`legoesm.atmosphere.physics.turbulence.amd.amd_nu_t`. Per-direction
    filter widths handle Δx≠Δz; purely local (no plane average) ⇒ MPI-safe
    algebra. The minimum-dissipation ``max(N,0)`` already gives zero K_m in
    resolved laminar/1-D shear, so no N² cutoff / wall cap is needed."""
    (uc, vc, wc, dudx, dudy, dudz, dvdx, dvdy, dvdz,
     dwdx, dwdy, dwdz) = _velocity_gradients_plane(
        u_yxz, v_yxz, w_yxz_half, grid, height_coord)
    return _amd_nu_t_core(
        dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz,
        grid.dx, grid.dy, height_coord.dz, c_amd)


def _compute_dynamic_smag_cs_plane(
    u_yxz: jax.Array,
    v_yxz: jax.Array,
    w_yxz_half: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    cs_max: float = 0.4,
) -> jax.Array:
    """Dynamic Smagorinsky coefficient ``C_s(z)`` (Germano 1991; Lilly 1992).

    Returns a per-LEVEL coefficient (shape ``(nlev,)``) computed from the
    resolved field by a horizontal test filter + the Germano identity, then
    plane-averaged over the homogeneous ``(x, y)`` directions:

        L_ij = F(u_i u_j) − F(u_i) F(u_j)                       (Leonard stress)
        M_ij = 2 Δ² [ F(|S| S_ij) − α² |F(S)| F(S_ij) ]          (α = 2)
        C_s² = ⟨L_ij^d M_ij⟩_xy / ⟨M_ij M_ij⟩_xy ,  clipped to [0, cs_max²]

    where ``F`` is the 3-point top-hat test filter, ``Δ=(dx·dy·dz)^⅓`` the grid
    filter width, and ``L^d`` the deviatoric Leonard stress (trace removed so a
    weakly-compressible ``S_kk≠0`` does not bias the contraction). The plane
    average is the standard stabilisation for horizontally-homogeneous LES
    (GABLS1, Wangara): it removes the well-known ill-conditioning of the pointwise
    Germano ratio. ``C_s(z)`` then feeds the SAME ``(C_s·Δ)²·|S|`` operator as the
    static path, so wall-damping / stratification cutoff / diffusion are unchanged.

    Single-rank plane average: ``jnp.mean`` is over the LOCAL (y, x) tile, so
    under a horizontal MPI decomposition this is a per-rank average (an
    approximation). The dynamic path is intended for single-rank (GPU) LES;
    MPI runs use the static closure (validated serial=MPI). AD/JIT-safe.
    """
    u_yxz.shape[-1]
    uc, vc, wc, S11, S22, S33, S12, S13, S23, Smag = (
        _centre_velocities_and_strain_plane(
            u_yxz, v_yxz, w_yxz_half, grid, height_coord))

    F = _horizontal_box_filter_plane
    alpha2 = 4.0  # (test/grid filter-width ratio)² = 2²

    # --- Leonard stress L_ij = F(u_i u_j) − F(u_i)F(u_j) (deviatoric) ---
    Fu, Fv, Fw = F(uc), F(vc), F(wc)
    L11 = F(uc * uc) - Fu * Fu
    L22 = F(vc * vc) - Fv * Fv
    L33 = F(wc * wc) - Fw * Fw
    L12 = F(uc * vc) - Fu * Fv
    L13 = F(uc * wc) - Fu * Fw
    L23 = F(vc * wc) - Fv * Fw
    Ltr = (L11 + L22 + L33) / 3.0
    L11d, L22d, L33d = L11 - Ltr, L22 - Ltr, L33 - Ltr

    # --- M_ij = 2 Δ² [ F(|S| S_ij) − α² |F(S)| F(S_ij) ] ---
    dz_full = height_coord.dz
    delta = (grid.dx * grid.dy * dz_full) ** (1.0 / 3.0)   # (nlev,)
    two_d2 = 2.0 * delta ** 2                              # (nlev,)
    # |F(S)| from the test-filtered strain components.
    FS11, FS22, FS33 = F(S11), F(S22), F(S33)
    FS12, FS13, FS23 = F(S12), F(S13), F(S23)
    FSmag = jnp.sqrt(jnp.maximum(
        2.0 * (FS11 ** 2 + FS22 ** 2 + FS33 ** 2
               + 2.0 * (FS12 ** 2 + FS13 ** 2 + FS23 ** 2)),
        1.0e-30))

    def m_comp(Sij, FSij):
        return two_d2 * (F(Smag * Sij) - alpha2 * FSmag * FSij)
    M11 = m_comp(S11, FS11)
    M22 = m_comp(S22, FS22)
    M33 = m_comp(S33, FS33)
    M12 = m_comp(S12, FS12)
    M13 = m_comp(S13, FS13)
    M23 = m_comp(S23, FS23)

    # --- Contractions L_ij M_ij and M_ij M_ij (off-diagonals ×2) ---
    LM = (L11d * M11 + L22d * M22 + L33d * M33
          + 2.0 * (L12 * M12 + L13 * M13 + L23 * M23))
    MM = (M11 ** 2 + M22 ** 2 + M33 ** 2
          + 2.0 * (M12 ** 2 + M13 ** 2 + M23 ** 2))

    # --- Plane-average per level, C_s² = ⟨LM⟩/⟨MM⟩, clip to [0, cs_max²] ---
    LM_z = jnp.mean(LM, axis=(0, 1))                       # (nlev,)
    MM_z = jnp.mean(MM, axis=(0, 1))
    cs_sq = LM_z / jnp.maximum(MM_z, 1.0e-30)
    # Clip backscatter (LM<0) + the ill-conditioned ratio to [0, cs_max²]. The
    # tiny positive floor (1e-24) keeps ``d/dx √(cs_sq)`` FINITE at the clip-to-
    # zero boundary (√(0) has an infinite derivative ⇒ a 0·∞ NaN under AD);
    # √1e-24 = 1e-12 is a negligible C_s floor.
    cs_sq = jnp.clip(cs_sq, 1.0e-24, cs_max ** 2)
    return jnp.sqrt(cs_sq)                                 # C_s(z), (nlev,)


def _compute_scale_dependent_dynamic_smag_cs_plane(
    u_yxz: jax.Array,
    v_yxz: jax.Array,
    w_yxz_half: jax.Array,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    cs_max: float = 1.0,
) -> jax.Array:
    """Scale-dependent dynamic Smagorinsky ``C_s`` (Bou-Zeid–Meneveau–Parlange
    2005, the LASD closure of the jax-alfa LES oracle).

    Faithful port of ``DynamicSGS_LASDD_SM.LASDD``. The standard Germano
    procedure (:func:`_compute_dynamic_smag_cs_plane`) assumes ``C_s`` is
    SCALE-INVARIANT — ``β = C_s²(2Δ)/C_s²(Δ) = 1`` — which over-dissipates near
    the wall where ``Δ`` is no longer ≪ the integral scale. LASD relaxes this by
    adding a SECOND test filter at ``4Δ`` and solving the Germano identity at
    both ratios for ``β`` per level, then forms

        M_ij = 2Δ²·F̂(|S|S_ij) − 2(2Δ)²·β·|Ŝ|·Ŝ_ij
        C_s²(x,y,z) = Imfilter(L_ij^d M_ij) / Imfilter(M_ij M_ij)

    with ``L_ij`` the resolved (Leonard) stress at ``2Δ`` and ``F̂`` the sharp
    spectral test filter. The β polynomial coefficients are plane-averaged (per
    level), exactly as the oracle; the final ``C_s²`` is LOCALLY averaged
    (3×3 Imfilter) and returned as a 3D field — the operator
    :func:`_compute_smagorinsky_K_m_plane` broadcasts a 3D ``c_s`` over ``Δ``.

    Returns ``C_s`` (NOT squared), shape ``(ny, nx, nlev)``. Single-rank/GPU LES
    (plane-mean β + spectral filter over the LOCAL tile); MPI uses the static
    closure, like the Germano path.
    """
    u_yxz.shape[-1]
    uc, vc, wc, S11, S22, S33, S12, S13, S23, S = (
        _centre_velocities_and_strain_plane(
            u_yxz, v_yxz, w_yxz_half, grid, height_coord))
    delta = (grid.dx * grid.dy * height_coord.dz) ** (1.0 / 3.0)   # (nlev,)
    cs2 = _lasd_cs2(uc, vc, wc, S11, S22, S33, S12, S13, S23, S,
                    delta, cs_max=cs_max)
    # Tiny floor so d/dx √(cs2) stays finite at the clip boundary (AD-safe).
    return jnp.sqrt(jnp.maximum(cs2, 1.0e-24))                    # C_s (ny,nx,nz)


def full_level_centred_d_dz(
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
    with respect to LEVEL INDEX = ``−∂f/∂z_physical``. The sign does
    NOT "drop out when squared" for MIXED strain components —
    ``S_13² = (0.5(∂u/∂z + ∂w/∂x))²`` carries an odd
    ``(∂u/∂z)(∂w/∂x)`` cross term — nor in gradient products
    (Vreman/AMD/Germano ``M_ij``) or in N². EVERY consumer needing a
    signed ``∂f/∂z_physical`` must negate the raw return value; all
    current call sites do (leading minus at the N² sites, negation at
    the strain/gradient sites).

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


def vertical_advection_van_leer_plane(
    field_yxz: jax.Array,
    w_yxz_half: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
) -> jax.Array:
    """Monotone (van-Leer TVD) ``-(w/J) df/dz`` at cell centres — for SCALARS.

    Drop-in monotone replacement for :func:`_vertical_advection_plane` for the
    TRACER vertical transport (D5). The centred scheme overshoots into negative
    tracer values at the sharp vertical gradients of a convective updraft (then
    clipped on the microphysics read → mass loss + distorted profiles); the
    van-Leer flux limiter forbids new extrema, so a positive tracer stays
    positive. SAM advects scalars with a monotone scheme (ULTIMATE-MACHO /
    MPDATA) and momentum with a centred one — so this is wired for tracers
    only, leaving u/v on the centred scheme.

    Numerics — the SAME advective form as the horizontal
    :func:`_van_leer_advection_x` (flux→advective identity
    ``-w·∂q/∂z = -∂(wq)/∂z + q·∂w/∂z``), so the horizontal + vertical operators
    combine to the advective ``-u·∇q`` (not a flux/advective mix). The flux
    reconstruction lives on the ``w`` INTERFACE grid where ``w`` natively sits
    (no half→full averaging): at interface ``j`` (between cell ``j-1`` above and
    cell ``j`` below) the face value ``q_face[j]`` is the slope-limited 2nd-order
    upwind value chosen by ``sign(w[j])``, ``F[j]=w[j]·q_face[j]``, and the cell
    tendency is ``(F[k+1]-F[k])/dz[k] + q[k]·(w[k]-w[k+1])/dz[k]``, all ``/J``.
    Rigid boundaries (``w[...,0]=w[...,-1]=0``) ⇒ zero flux through the
    top/surface; the field is edge-padded (zero-gradient) so the 4-cell stencil
    degrades to 1st-order upwind at the boundary-adjacent interfaces (standard
    TVD boundary behaviour). Reduces to the centred scheme for a smooth field
    with uniform ``w``. AD: van-Leer's ``(r+|r|)/(1+|r|)`` is smooth except a
    single subgradient kink at ``r=0``; the ``where``-guarded denominators flow
    cleanly through ``jax.grad`` (same construction as ``_van_leer_advection_x``).
    """
    from legoesm.core.flux_limiters import van_leer_face_values
    nlev = field_yxz.shape[-1]
    if nlev <= 2:
        # too few cells for the 4-point stencil — fall back to centred.
        return _vertical_advection_plane(field_yxz, w_yxz_half, height_coord, J)
    f = field_yxz                                   # (ny, nx, nlev)
    w = w_yxz_half                                  # (ny, nx, nlev+1)
    dz = height_coord.dz                            # (nlev,)
    # Edge-pad the field by 2 cells each end so every interface j=0..nlev has
    # its [j-2, j-1, j, j+1] stencil (zero-gradient ghosts at the rigid lids).
    pad_axes = ((0, 0),) * (f.ndim - 1)
    fp = jnp.pad(f, (*pad_axes, (2, 2)), mode="edge")   # (ny, nx, nlev+4)
    f_jm2 = fp[..., 0:nlev + 1]                     # cell j-2 at interface j
    f_jm1 = fp[..., 1:nlev + 2]                     # cell j-1 (above the face)
    f_j = fp[..., 2:nlev + 3]                       # cell j   (below the face)
    f_jp1 = fp[..., 3:nlev + 4]                     # cell j+1
    # Face reconstruction at interface j (the i+1/2 face with left cell j-1,
    # right cell j) via the shared HD-1-correct helper. Velocity along
    # increasing-k is -w, so phi_pos (face-vel>0) is downward flow (w<=0,
    # upwind cell j-1 above) and phi_neg (face-vel<0) is upward (w>0, cell j).
    phi_pos, phi_neg = van_leer_face_values(f_jm2, f_jm1, f_j, f_jp1)
    q_face = jnp.where(w <= 0.0, phi_pos, phi_neg)  # (ny, nx, nlev+1)
    flux = w * q_face                               # F[j] at interfaces
    # Rigid lids: explicitly enforce NO flux through the top/surface interfaces
    # (``w`` is already 0 there by construction; this is defensive so a future
    # non-zero-lid w cannot leak tracer mass — codex iter-44 C).
    flux = flux.at[..., 0].set(0.0).at[..., -1].set(0.0)
    # Advective tendency = -∂(wq)/∂z + q·∂w/∂z, both per cell thickness dz[k].
    flux_div = (flux[..., 1:] - flux[..., :-1]) / dz       # (F[k+1]-F[k])/dz
    dwdz = f * (w[..., :-1] - w[..., 1:]) / dz             # q·(w[k]-w[k+1])/dz
    return (flux_div + dwdz) / J[:, :, None]


# --------------------------------------------------------------------- #
# Slow tendencies                                                       #
# --------------------------------------------------------------------- #


def moisture_buoyancy_w_half(
    tracers: jax.Array,
    theta_prime: jax.Array,
    height_coord: HeightCoordinate,
    hmean_fn,
) -> jax.Array:
    """SAM moist buoyancy on ``w`` at half levels (``buoyancy.f90``).

    The dry θ'-buoyancy in the acoustic substep
    (``compressible_euler._acoustic_substep``) carries only the LEADING
    ``g·θ'/θ₀`` (the ``×1`` thermal coefficient).  SAM's full buoyancy
    (``buoyancy.f90:35-46``), with ``bet = g/T₀`` factored out, is

        buo = g·(ε_v·q_v′ − q_cond′)                       # vapour + load
            + g·(T′/T₀)·(1 + ε_v·q̄_v − q̄_cond)            # moist thermal

    where ``′`` = deviation from the HORIZONTAL MEAN (SAM's evolving
    ``qv0``/``qn0``/``qp0``/``tabs0`` base state) and ``ε_v = 1/ε − 1 ≈
    0.61``.  The acoustic substep supplies the ``g·(T′/T₀)·1`` piece;
    this helper supplies the rest:

        B_moist = g·[ ε_v·q_v′ − q_cond′
                      + (θ′−⟨θ′⟩)/θ₀ · (ε_v·q̄_v − q̄_cond) ]

    — the first-order vapour-virtual + condensate loading PLUS the moist
    modification of the thermal coefficient (``ε_v·q̄_v − q̄_cond``).  The
    thermal correction multiplies ``θ′−⟨θ′⟩`` (deviation from the
    horizontal mean = SAM's ``T′`` relative to ``tabs0``), so the whole
    term has ZERO horizontal mean and adds NO spurious mean updraft
    against legoESM's DRY reference (full ``q_v`` would inject
    ``g·ε_v·⟨q_v⟩ ≈ 0.07 m/s²`` of unbalanced mean buoyancy in RCE).
    Dry-substep + ``B_moist`` then reproduce SAM's full ``buoyancy.f90``
    expression (up to the uniform-grid half-level average noted below).

    Moisture is frozen across the acoustic substeps, so this is evaluated
    ONCE per RK stage and added to the slow ``dw/dt`` (SAM likewise adds
    buoyancy once per Adams–Bashforth step), unlike the dry θ'-buoyancy
    re-evaluated every substep for the acoustic / gravity-wave coupling.

    Faithfulness caveats (logged):
    - Full→half interpolation is a uniform ``0.5·(b[k]+b[k+1])``, matching
      the dry ``theta_p_half`` convention; SAM uses ``adz``-weighted
      ``betu``/``betd`` — a parity gap only on STRETCHED vertical grids,
      shared by the dry buoyancy (fix both together if it matters).
    - No SAM energy-compensation term (``factor=coef·buo·w`` on the
      prognostic ``t``); legoESM's dry buoyancy likewise omits it (that
      correction is specific to SAM's anelastic liquid-static-energy
      formulation — legoESM's compressible θ/ρ thermodynamics handle the
      buoyancy work via the pressure/continuity coupling instead).

    Parameters
    ----------
    tracers : jax.Array
        ``(ny, nx, nlev, n_tracers)`` interior cell-centre tracers.  Slot
        0 = ``q_v``; slots 1..min(5) = condensate mass (q_c, q_r, q_i,
        q_s, q_g).  Robust to ``n_tracers < 6`` and ``== 0``.
    theta_prime : jax.Array
        ``(ny, nx, nlev)`` potential-temperature perturbation θ − θ_ref
        (cell centres).  Only its deviation from the horizontal mean is
        used (the moist thermal-coefficient correction).
    height_coord : HeightCoordinate
        Provides ``theta_ref`` (θ₀) and ``z_half`` (interface count).
    hmean_fn : Callable[[jax.Array], jax.Array]
        Horizontal-mean reduction of a ``(ny, nx, nlev)`` field to a
        broadcastable mean profile.  Serial passes
        ``lambda f: jnp.mean(f, axis=(0, 1), keepdims=True)``; the MPI
        halo path passes a GLOBAL mean (``global_sum_mpi`` over interior /
        global point count) — bit-identical at ``n_ranks == 1`` and
        physically correct for ``n_ranks > 1``.

    Returns
    -------
    jax.Array
        ``B_moist`` at half levels ``(ny, nx, nlev+1)`` with rigid (zero)
        top + bottom interfaces, ready to add to the slow ``dw/dt``.
    """
    from legoesm import constants

    nlev_half = height_coord.z_half.shape[-1]
    n_tr = tracers.shape[-1]
    if n_tr == 0:
        ny, nx = tracers.shape[0], tracers.shape[1]
        return jnp.zeros((ny, nx, nlev_half), dtype=tracers.dtype)

    q_v = tracers[..., 0]
    n_cond = min(n_tr, 6)                      # slots 1..5 = condensate mass
    if n_cond > 1:
        q_cond = jnp.sum(tracers[..., 1:n_cond], axis=-1)
    else:
        q_cond = jnp.zeros_like(q_v)

    eps_v = 1.0 / constants.epsilon - 1.0
    qv_bar = hmean_fn(q_v)                     # (1,1,nlev) — SAM qv0
    qcond_bar = hmean_fn(q_cond)              # (1,1,nlev) — SAM qn0+qp0
    # Moist thermal-coefficient correction: (θ′−⟨θ′⟩)/θ₀ · (ε_v·q̄_v − q̄_cond).
    # θ′−⟨θ′⟩ = θ − ⟨θ⟩ = SAM's T′ relative to the horizontal-mean base
    # state; keeps the whole term zero-mean.
    theta_dev = theta_prime - hmean_fn(theta_prime)
    thermal_coef = eps_v * qv_bar - qcond_bar
    b_full = constants.g * (
        eps_v * (q_v - qv_bar) - (q_cond - qcond_bar)
        + (theta_dev / height_coord.theta_ref) * thermal_coef
    )                                          # (ny, nx, nlev)

    # Full levels -> interior interfaces (two-point average); rigid (zero)
    # top + bottom to match the w BC of the horizontal-advection dw/dt.
    pad_axes = ((0, 0),) * (b_full.ndim - 1)
    return jnp.pad(
        0.5 * (b_full[..., :-1] + b_full[..., 1:]),
        (*pad_axes, (1, 1)),
    )


def _acoustic_moist_buoyancy_w(state, height_coord, euler_config, layout=None):
    """Frozen SAM moist buoyancy on the w half-levels for the ACOUSTIC loop.

    Returns ``B_moist`` (vapour-virtual + condensate loading, :func:`
    moisture_buoyancy_w_half`) evaluated ONCE from the stage-initial state, to be
    added to w EACH acoustic substep — so the condensate-loading drag acts at the
    SAME frequency as the dry θ' buoyancy in the substep loop. Applying it only once
    per RK stage (the old ``slow``-tendency path) let latent-heated updrafts feel
    the full dry warming 6×/step but the moist drag 1×/step → runaway convection
    (max|w|→40 m/s). Returns None when moist/acoustic-moist buoyancy is off.

    MPI note: the perturbation mean is a LOCAL ``jnp.mean`` (= the true domain mean
    at n_ranks==1, so serial==halo parity holds + the halo parity tests pass). Under
    a horizontal decomposition (n_ranks>1) it is a rank-local mean — a small
    approximation, the same convention the dynamic-Smagorinsky plane average uses;
    the serial / single-GPU path (the RCE driver) is exact."""
    if not (getattr(euler_config, "moist_buoyancy", False)
            and getattr(euler_config, "acoustic_moist_buoyancy", True)):
        return None
    # Horizontal-mean function for the SAM perturbation.  Default = rank-local
    # ``jnp.mean`` (zero comm; exact at n_ranks==1).  Opt-in exact GLOBAL mean
    # (one allreduce/field) when ``acoustic_moist_global_mean`` is set AND we
    # are actually decomposed — the global cell count is known from the layout
    # (``ny_global*nx_global``), so only the per-level SUM needs an allreduce,
    # which ``global_sum_mpi`` carries with an AD-safe VJP.
    if (getattr(euler_config, "acoustic_moist_global_mean", False)
            and layout is not None and layout.n_ranks > 1):
        from legoesm.parallel.reductions import global_sum_mpi
        _gn = float(layout.ny_global * layout.nx_global)

        def _hmean(f):
            return global_sum_mpi(
                jnp.sum(f, axis=(0, 1), keepdims=True)) / _gn
    else:
        def _hmean(f):
            return jnp.mean(f, axis=(0, 1), keepdims=True)
    return moisture_buoyancy_w_half(
        state.tracers.data, state.theta_prime.data, height_coord, _hmean)


# Valid SGS turbulence closures — single source shared by validate_plane_config()
# (fail-early, at model __init__) and the slow_tendencies hot-path guard below, so
# the two allowlists cannot drift.  A direct slow_tendencies caller (tests, the
# public dynamics registry) bypasses the __init__ validator, so both guards exist.
_VALID_TURBULENCE_CLOSURES = ("smagorinsky", "molecular", "none", "vreman", "amd")


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
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        compute_exner_perturbation,
    )
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # 2. C-grid pressure gradient. ``grad_x_vlast(pi_p)`` returns
    #    the x-face gradient ``(pi[j,i] - pi[j,i-1])/dx`` — exactly
    #    where ``u`` lives. ``theta_total`` is at cell centres;
    #    interpolate to the x-face so the PG operand co-locates with
    #    its target tendency.
    # When substep_horizontal_acoustic is set, the horizontal PG moves
    # into the acoustic substep loop (full Skamarock-Klemp split). Skip
    # it here so it is not double-counted (and not integrated at the
    # unstable outer-dt CFL).
    substep_horiz = getattr(config, "substep_horizontal_acoustic", False)
    if substep_horiz:
        du_pg = jnp.zeros_like(u)
        dv_pg = jnp.zeros_like(v)
    else:
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
        # SAM coriolis.f90: f acts on the DEPARTURE from the geostrophic
        # reference wind (ug0(k), vg0(k)), not the full wind — otherwise the
        # large-scale balanced mean wind is spuriously spun up by an inertial
        # oscillation. ug0/vg0 are (nlev,) base-state profiles on height_coord;
        # None ⇒ zero (RCE / no-mean-wind ⇒ reduces to the full-wind form).
        v_ref = (0.0 if height_coord.v_geo0 is None
                 else height_coord.v_geo0[None, None, :])
        u_ref = (0.0 if height_coord.u_geo0 is None
                 else height_coord.u_geo0[None, None, :])
        du_cor = f_xface * (v_xface - v_ref)
        dv_cor = -f_yface * (u_yface - u_ref)
    else:
        du_cor = jnp.zeros_like(u)
        dv_cor = jnp.zeros_like(v)

    # 4. Mass continuity (flux form): ``drho'/dt = -div(rho · u)``.
    #    ``rho_total`` lives at cell centres; interpolate to each face
    #    so the mass flux has the same staggering as the velocity.
    # Horizontal mass-flux divergence. With substep_horizontal_acoustic
    # the FULL continuity (horizontal + vertical) is integrated in the
    # acoustic substep, so the slow tendency contributes no continuity
    # term here (only sponge/hyperdiff on rho' below).
    if substep_horiz:
        drho_p_dt = jnp.zeros_like(rho_p)
    else:
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
    _ADV_PAIRS = {
        "weno5": (_weno5_advection_x, _weno5_advection_y),
        "van_leer": (_van_leer_advection_x, _van_leer_advection_y),
        "centered": (_centered_advection_x, _centered_advection_y),
        "upwind1": (_upwind_advection_x, _upwind_advection_y),
    }
    scheme = getattr(config, "horizontal_advection_scheme", "upwind1")
    if scheme not in _ADV_PAIRS:
        # iter-194: list the active registry instead of a hardcoded string so a
        # future scheme in HORIZONTAL_ADVECTION_HALO_REQUIREMENT auto-surfaces.
        raise ValueError(
            f"Unknown horizontal_advection_scheme: {scheme!r}. "
            f"Expected one of {sorted(HORIZONTAL_ADVECTION_HALO_REQUIREMENT)}."
        )
    adv_x, adv_y = _ADV_PAIRS[scheme]
    # POSITIVITY GUARD (codex CRM-dycore review): the CRM tracers
    # (q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r, N_i, …) are all
    # positive-definite. WENO5-Z is 5th-order but NOT positivity-
    # preserving, so it drives q<0 that then feeds microphysics as a
    # spurious source. Advect TRACERS with the monotone van_leer
    # limiter when weno5 is selected; θ′ (signed) keeps weno5 for its
    # low tropopause dispersion. van_leer's 2-cell halo ⊆ weno5's
    # 3-cell halo ⇒ never under-halos. Mirrors SAM (monotone scalars,
    # non-diffusive θ/momentum) and the ADV-SPLIT momentum path below.
    if scheme == "weno5":
        tadv_x, tadv_y = _van_leer_advection_x, _van_leer_advection_y
    else:
        tadv_x, tadv_y = adv_x, adv_y
    # ADV-SPLIT (#86): MOMENTUM legs (u/v/w) may use a separate scheme (e.g.
    # "centered" = SAM-faithful non-diffusive advect2_mom) while scalars keep the
    # monotone van_leer. None ⇒ momentum = scalar scheme (legacy, both same).
    mscheme = getattr(
        config, "horizontal_momentum_advection_scheme", None) or scheme
    if mscheme not in _ADV_PAIRS:
        raise ValueError(
            f"Unknown horizontal_momentum_advection_scheme: {mscheme!r}. "
            f"Expected one of {sorted(HORIZONTAL_ADVECTION_HALO_REQUIREMENT)}."
        )
    madv_x, madv_y = _ADV_PAIRS[mscheme]
    u_center = interp_xface_to_cell_vlast(u, grid)
    v_center = interp_yface_to_cell_vlast(v, grid)
    dtheta_p_dt = (
        adv_x(theta_p, u_center, grid.dx)
        + adv_y(theta_p, v_center, grid.dy)
    )

    # 6. Horizontal momentum advection — Arakawa-C. u lives at x-face;
    #    advect by (u-at-x-face, v-at-x-face). v→x-face via 4-pt
    #    corner average. Symmetric for v. Uses the MOMENTUM scheme
    #    (madv; ADV-SPLIT #86 — may be centred/non-diffusive while
    #    scalars stay monotone van_leer).
    v_at_xface = interp_yface_to_xface_vlast(v, grid)
    u_at_yface = interp_xface_to_yface_vlast(u, grid)
    du_adv = (
        madv_x(u, u, grid.dx)
        + madv_y(u, v_at_xface, grid.dy)
    )
    dv_adv = (
        madv_x(v, u_at_yface, grid.dx)
        + madv_y(v, v, grid.dy)
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
        madv_x(w_full, u_center, grid.dx)
        + madv_y(w_full, v_center, grid.dy)
    )
    # Re-map to half levels: interior is the average of adjacent full
    # values; top and bottom interfaces stay rigid (zero) so the
    # acoustic substep's BC convention is preserved.
    pad_axes = ((0, 0),) * (dw_full.ndim - 1)
    dw_dt = jnp.pad(
        0.5 * (dw_full[..., :-1] + dw_full[..., 1:]),
        (*pad_axes, (1, 1)),
    )

    # 8b. SAM moist buoyancy on w (vapour-virtual + condensate loading),
    #     frozen for this RK stage. The dry θ' buoyancy lives in the
    #     acoustic substep; this adds the moist part SAM's buoyancy.f90
    #     carries. Perturbation from the SERIAL horizontal mean.
    # Apply the moist buoyancy here (once per RK stage) ONLY when the acoustic
    # loop does NOT carry it. With acoustic_moist_buoyancy=True (default) the moist
    # buoyancy is added INSIDE the acoustic substeps instead (consistent frequency
    # with the dry θ' buoyancy), which fixes the convective-updraft runaway.
    if config.moist_buoyancy and not getattr(
            config, "acoustic_moist_buoyancy", True):
        dw_dt = dw_dt + moisture_buoyancy_w_half(
            state.tracers.data, theta_p, height_coord,
            lambda f: jnp.mean(f, axis=(0, 1), keepdims=True),
        )

    # 9. Rayleigh sponge layer (top-of-model damping). Zero when
    #    ``config.sponge_coeff == 0`` — the profile evaluates to zero
    #    below ``H - sponge_width`` and reaches ``sponge_coeff`` at
    #    the model top. The same sponge taper is applied to u, v, and
    #    theta' at full levels and to w at half levels; this matches
    #    the MPAS NH dycore pattern in
    #    :func:`compressible_euler_mpas.mpas_compressible_euler_slow_tendencies`.
    _sponge_shape = getattr(config, "sponge_profile_shape", "sin2")
    sponge_full = sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff, shape=_sponge_shape,
    )                                         # (nlev,)
    sponge_half = sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff, shape=_sponge_shape,
    )                                         # (nlev+1,)
    # SAM ``damping.f90`` damps the VERTICAL velocity ONLY (``w/(1+taudamp)``);
    # u/v/θ/ρ are untouched, so the sponge absorbs gravity waves while leaving
    # the anvil-level horizontal wind + thermodynamics intact. With
    # ``sponge_w_only=True`` (static config bool) legoESM matches that; the
    # default damps all 5 fields (the MPAS-style sponge). The CRM run-scripts
    # set True + ``sponge_width=0.4·H`` (SAM ``nub=0.6``, top 40%).
    if not getattr(config, "sponge_w_only", False):
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
    tracer_sgs_tend = None     # SGS-SCALAR: K_h diffusion on tracers (set below)
    # Turbulence-closure mode (DNS-LES, iter-177). Both branches build a
    # cell-centre eddy/molecular viscosity K_m + a Prandtl number; the
    # diffusion operators below are SHARED, so LES and DNS reuse the CRM
    # machinery unchanged — only K_m differs.
    _closure = getattr(config, "turbulence_closure", "smagorinsky")
    if _closure not in _VALID_TURBULENCE_CLOSURES:
        raise ValueError(
            f"unknown turbulence_closure {_closure!r}; must be one of "
            "'smagorinsky', 'molecular', 'none', 'vreman', 'amd' — a typo "
            "would otherwise silently disable ALL SGS turbulence "
            "(indistinguishable from the intentional 'none' inviscid mode)")
    _use_smag = _closure == "smagorinsky" and config.smagorinsky_cs > 0.0
    _use_mol = (_closure == "molecular"
                and getattr(config, "molecular_viscosity", 0.0) > 0.0)
    _use_vreman = _closure == "vreman" and config.vreman_c > 0.0
    _use_amd = _closure == "amd" and getattr(config, "amd_c", 0.0) > 0.0
    if _use_smag or _use_mol or _use_vreman or _use_amd:
        if _use_mol:
            # DNS: CONSTANT molecular kinematic viscosity ν everywhere — no
            # eddy model, no stratification cutoff. K_h = ν / Pr for the heat
            # + scalar legs. Needs sgs_vertical_diffusion=True for the full
            # 3-D ν∇²; dx must resolve ~the Kolmogorov scale.
            K_m = jnp.full(u.shape, config.molecular_viscosity, dtype=u.dtype)
            sgs_prandtl = config.molecular_prandtl
        elif _use_vreman:
            # Vreman (2004) EDDY viscosity — an OPTIONAL alternative to
            # Smagorinsky (NOT the SAM-faithful default). It vanishes for
            # well-resolved laminar/2-D shear and uses per-direction filter
            # widths, so it behaves on anisotropic Δx≠Δz grids. Purely local
            # (no plane average) ⇒ MPI-safe. K_h = K_m / Pr (same Prandtl).
            K_m = _compute_vreman_K_m_plane(
                u, v, w, grid, height_coord, config.vreman_c)
            sgs_prandtl = config.smagorinsky_prandtl
        elif _use_amd:
            # Anisotropic Minimum-Dissipation (Rozema 2015 / Abkar-Bae-Moin
            # 2016) EDDY viscosity — an OPTIONAL alternative to Smagorinsky.
            # Gives ZERO SGS viscosity where the resolved flow needs none
            # (minimum dissipation) and never negative, with per-direction
            # filter widths for anisotropic Δx≠Δz grids. Purely local (no plane
            # average) ⇒ MPI-safe. K_h = K_m / Pr (same Prandtl).
            K_m = _compute_amd_K_m_plane(
                u, v, w, grid, height_coord, config.amd_c)
            sgs_prandtl = config.smagorinsky_prandtl
        else:
            # CRM/LES: Smagorinsky-Lilly EDDY viscosity. Full 3D strain (takes
            # half-level w so the vertical-shear components S13, S23, S33
            # contribute to |S|). Sub-grid N² (clear↔moist switched) applies
            # the SAM dosmagor stratification cutoff (tke_full.f90:154-226).
            # K_m at cell centres; shared with the halo kernel so serial/MPI
            # stay bit-identical.
            n2_sgs = sgs_brunt_vaisala_sq(
                theta_total, state.tracers.data, height_coord,
            )
            # DYNAMIC Smagorinsky (Germano/Lilly): replace the fixed
            # config.smagorinsky_cs with a plane-averaged C_s(z) diagnosed from
            # the resolved field each step. Falls back to the static scalar when
            # the flag is off (byte-identical). The dynamic C_s(z) array then
            # feeds the SAME (C_s·Δ)²·|S| operator (c_s broadcasts per-level).
            if getattr(config, "smagorinsky_dynamic", False):
                cs_max = getattr(config, "smagorinsky_dynamic_cs_max", 0.4)
                if getattr(config, "smagorinsky_scale_dependent", False):
                    # Bou-Zeid et al. (2005) scale-dependent dynamic (LASD) —
                    # adds a 4Δ test filter + per-level β solve; returns a 3D
                    # C_s field (broadcasts over Δ in the K_m operator).
                    c_s_arg = _compute_scale_dependent_dynamic_smag_cs_plane(
                        u, v, w, grid, height_coord, cs_max=cs_max,
                    )
                else:
                    c_s_arg = _compute_dynamic_smag_cs_plane(
                        u, v, w, grid, height_coord, cs_max=cs_max,
                    )
            else:
                c_s_arg = config.smagorinsky_cs
            K_m = _compute_smagorinsky_K_m_plane(
                u, v, w, grid, height_coord, c_s_arg,
                n2_sgs=n2_sgs, prandtl=config.smagorinsky_prandtl,
                wall_damping=getattr(config, "smagorinsky_wall_damping", True),
                delta_max=config.smagorinsky_delta_max,
                stability_length=getattr(
                    config, "smagorinsky_stability_length", False),
            )
            sgs_prandtl = config.smagorinsky_prandtl
        # u at x-face, v at y-face on the Arakawa-C grid →
        # interpolate K_m to each face before the flux-form
        # diffusion so the operand and diffusivity co-locate.
        K_m_xface = interp_cell_to_xface_vlast(K_m, grid)
        K_m_yface = interp_cell_to_yface_vlast(K_m, grid)
        du_dt = du_dt + _variable_K_diffusion_vlast(u, K_m_xface, grid)
        dv_dt = dv_dt + _variable_K_diffusion_vlast(v, K_m_yface, grid)
        K_h = K_m / sgs_prandtl
        dtheta_p_dt = dtheta_p_dt + _variable_K_diffusion_vlast(
            theta_p, K_h, grid,
        )
        # SGS-VERT (#81): add the VERTICAL SGS flux ∂_z(K ∂_z φ) so the
        # Smagorinsky closure is fully 3D like SAM (gated; default off).
        # u/v use their face-co-located K (same diffusivity as the
        # horizontal leg); θ'/tracers use K_h; w uses cell-centre K_m. The
        # no-flux scalar BC + rigid-w BC live inside the helpers (surface
        # fluxes are applied separately by the surface scheme).
        if config.sgs_vertical_diffusion:
            du_dt = du_dt + _vertical_K_diffusion_full(u, K_m_xface, height_coord)
            dv_dt = dv_dt + _vertical_K_diffusion_full(v, K_m_yface, height_coord)
            dtheta_p_dt = dtheta_p_dt + _vertical_K_diffusion_full(
                theta_p, K_h, height_coord,
            )
        # SGS-SCALAR (iter-60): SAM `sgs.f90:664-675` SGS-diffuses EVERY scalar
        # (`do k=1,nmicro_fields: call diffuse_scalar(micro_field(:,:,:,k),tkh)`)
        # — q_v + all hydrometeor mass + number fields — with the same eddy
        # conductivity as `t` (SAM Pr=1 ⇒ tkh=tk=K_m). legoESM previously diffused
        # θ' ONLY, leaving moisture/condensate under-mixed (wrong cloud-edge
        # dilution, q'²). Apply the SAME K_h to EVERY tracer slot — matching both
        # the θ' treatment above AND the tracer ADVECTION below (which also vmaps
        # all slots), so no slot is advected-but-not-diffused. SAM `flag_advect=1`
        # for all M2005 fields with docloud+doprecip on ⇒ diffuse-all is faithful
        # (a non-advected/padded slot, if ever added, would need masking in BOTH
        # advection and this diffusion together). The VERTICAL leg is added below
        # when sgs_vertical_diffusion is on (SGS-VERT #81 — now wired, was a gap).
        if state.tracers.data.shape[-1] > 0:
            tracer_sgs_tend = jax.vmap(
                lambda q: _variable_K_diffusion_vlast(q, K_h, grid),
                in_axes=-1, out_axes=-1,
            )(state.tracers.data)
            if config.sgs_vertical_diffusion:
                tracer_sgs_tend = tracer_sgs_tend + jax.vmap(
                    lambda q: _vertical_K_diffusion_full(q, K_h, height_coord),
                    in_axes=-1, out_axes=-1,
                )(state.tracers.data)
        # ``K_m`` at full levels; interpolate to half levels for w
        # (rigid boundary K stays zero — no spurious tendency at top
        # / bottom interfaces).
        K_m_half_interior = 0.5 * (K_m[..., :-1] + K_m[..., 1:])
        pad_axes = ((0, 0),) * (K_m_half_interior.ndim - 1)
        K_m_half = jnp.pad(
            K_m_half_interior, (*pad_axes, (1, 1)),
        )
        dw_dt = dw_dt + _variable_K_diffusion_vlast(w, K_m_half, grid)
        if config.sgs_vertical_diffusion:
            dw_dt = dw_dt + _vertical_K_diffusion_w(w, K_m, height_coord)

    # 12. Tracer advection. Advective form via upwind on cell-centre
    #     velocities (face-averaged from ``u``, ``v`` — the C-grid
    #     pairing for cell-centred scalars). vmap over the trailing
    #     tracer axis. The VERTICAL scheme is config-selected (D5):
    #     "centered" (default) or monotone "van_leer" (positive-definite,
    #     SAM-faithful for scalars). u/v momentum stay centred regardless.
    vert_tracer_scheme = getattr(
        config, "vertical_tracer_advection", "centered")
    if vert_tracer_scheme == "van_leer":
        _vertical_tracer_adv = vertical_advection_van_leer_plane
    elif vert_tracer_scheme == "centered":
        _vertical_tracer_adv = _vertical_advection_plane
    else:
        raise ValueError(
            f"Unknown vertical_tracer_advection: {vert_tracer_scheme!r}. "
            f"Expected 'centered' or 'van_leer'."
        )
    tracers = state.tracers.data
    if tracers.shape[-1] > 0:
        def _tracer_tend_one(q):
            # tadv_* = van_leer under weno5 (positivity guard above);
            # == adv_* for every monotone scheme.
            return (
                tadv_x(q, u_center, grid.dx)
                + tadv_y(q, v_center, grid.dy)
                + _vertical_tracer_adv(q, w, height_coord, J)
            )
        dtracers_dt = jax.vmap(_tracer_tend_one, in_axes=-1, out_axes=-1)(
            tracers,
        )
    else:
        dtracers_dt = jnp.zeros_like(tracers)
    # SGS-SCALAR (iter-60): add the horizontal SGS eddy diffusion of every tracer
    # (computed with K_h in the Smagorinsky block above) — SAM diffuses all
    # micro_field scalars, not just θ'. Conservative flux-form ⇒ tracer mass
    # preserved under periodic BC.
    if tracer_sgs_tend is not None:
        dtracers_dt = dtracers_dt + tracer_sgs_tend
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
    layout=None,
) -> PlaneNonHydrostaticState:
    """Run ``n_substeps`` forward-backward acoustic substeps on the plane.

    Thin wrapper around
    :func:`compressible_euler.acoustic_column_kernel` — no duplicated
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
    b_moist = _acoustic_moist_buoyancy_w(state, height_coord, euler_config, layout)
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _ in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = acoustic_column_kernel(
            w_final, theta_p_final, rho_p_final,
            height_coord, J, dt_s, beta, g,
            theta_vert_van_leer=(
                getattr(euler_config, "acoustic_theta_advection", "centered")
                == "van_leer"),
        )
        if b_moist is not None:  # frozen moist buoyancy each substep (see si_horizontal)
            w_final = (w_final + dt_s * b_moist).at[..., 0].set(0.0).at[..., -1].set(0.0)

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
    layout=None,
) -> PlaneNonHydrostaticState:
    """Semi-implicit acoustic substeps on the plane via per-column
    Thomas tridiagonal solve.

    Lifts the vertical acoustic CFL constraint that limits the
    forward-backward variant to ``dt ≲ dx/c_s``. Permits outer
    ``dt`` ~ 30-60 s at dx=2 km (advective CFL bound only).

    Thin wrapper around
    :func:`compressible_euler.semi_implicit_acoustic_column_kernel`
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
    si_w_filter_nu = float(getattr(
        euler_config, "si_w_vertical_filter_nu", 0.0,
    ))
    b_moist = _acoustic_moist_buoyancy_w(state, height_coord, euler_config, layout)
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _ in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = (
            semi_implicit_acoustic_column_kernel(
                w_final, theta_p_final, rho_p_final,
                height_coord, J, dt_s, beta, g,
                implicit_buoyancy=implicit_buoyancy,
                precomputed_tridiag=tri_bands,
                si_w_vertical_filter_nu=si_w_filter_nu,
                theta_vert_van_leer=(
                    getattr(euler_config, "acoustic_theta_advection", "centered")
                    == "van_leer"),
            )
        )
        if b_moist is not None:  # frozen moist buoyancy each substep (see si_horizontal)
            w_final = (w_final + dt_s * b_moist).at[..., 0].set(0.0).at[..., -1].set(0.0)

    return PlaneNonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


def plane_acoustic_substeps_si_horizontal(
    state: PlaneNonHydrostaticState,
    slow_tend: PlaneNonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
    grid: PlaneGrid,
    layout=None,
) -> PlaneNonHydrostaticState:
    """Full Skamarock-Klemp split-explicit acoustic substep on the plane.

    Unlike :func:`plane_acoustic_substeps_semi_implicit` (vertical-only),
    this integrates BOTH the horizontal and vertical acoustic terms on the
    short substep ``dt_s = dt/n_substeps``:

    Per substep (forward-backward):
      1. pi' from (rho', theta') via the EOS perturbation.
      2. u,v forward update by the horizontal pressure gradient
         ``-c_p * theta_face * grad(pi')`` (the term removed from the
         slow tendency when ``substep_horizontal_acoustic=True``).
      3. w vertical implicit solve + vertical continuity + vertical theta
         advection via the shared column kernel (SI in the vertical).
      4. rho' backward update with the HORIZONTAL mass-flux divergence
         using the just-updated u,v (the vertical part is already in the
         column kernel).

    This lowers the horizontal-acoustic CFL from ``c_s*dt/dx`` (unstable
    at fine dx when the PG sits in the slow tendency) to
    ``c_s*dt/(n_substeps*dx)``. Requires the caller to pass ``grid`` for
    the horizontal C-grid operators.

    ``slow_tend`` is unused here: the slow forcing is applied once per
    RK3 stage (``_rk_stage_with_acoustics``) before the substep loop, the
    same convention the vertical-only SI substep uses.
    """
    del slow_tend
    g = euler_config.g
    J = terrain_metric.jacobian
    beta = euler_config.acoustic_off_centering
    implicit_buoyancy = euler_config.implicit_buoyancy
    c_p = _c_pd_constant()
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    si_w_filter_nu = float(getattr(
        euler_config, "si_w_vertical_filter_nu", 0.0,
    ))

    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    tri_bands = precompute_si_tridiag_bands(
        height_coord, J, dt_s, g, implicit_buoyancy,
        nlev=theta_p.shape[-1],
    )

    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        compute_exner_perturbation,
    )

    b_moist = _acoustic_moist_buoyancy_w(state, height_coord, euler_config, layout)
    u_c, v_c, w_c, theta_p_c, rho_p_c = u, v, w, theta_p, rho_p
    for _ in range(int(n_substeps)):
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c, rho_0 + rho_p_c,
        )
        # 1+2. Horizontal pressure gradient -> forward u, v update.
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
        grad_pi_x = grad_x_vlast(pi_p, grid)
        grad_pi_y = grad_y_vlast(pi_p, grid)
        theta_xface = interp_cell_to_xface_vlast(theta_total, grid)
        theta_yface = interp_cell_to_yface_vlast(theta_total, grid)
        u_new = u_c + dt_s * (-c_p * theta_xface * grad_pi_x)
        v_new = v_c + dt_s * (-c_p * theta_yface * grad_pi_y)

        # 3. Vertical acoustic implicit (w) + vertical continuity +
        #    vertical theta advection (shared column kernel).
        #    Pass beta=0.0 here so the kernel does NOT off-center the
        #    vertical continuity in isolation — we apply Skamarock-Klemp
        #    off-centering to the FULL (vertical + horizontal) divergence
        #    below, so both legs receive consistent acoustic damping
        #    (cavecrew review: asymmetric off-centering between the
        #    vertical and horizontal continuity breaks the f-b stencil).
        w_new, theta_p_new, rho_p_vert = semi_implicit_acoustic_column_kernel(
            w_c, theta_p_c, rho_p_c,
            height_coord, J, dt_s, 0.0, g,
            implicit_buoyancy=implicit_buoyancy,
            precomputed_tridiag=tri_bands,
            si_w_vertical_filter_nu=si_w_filter_nu,
            theta_vert_van_leer=(
                getattr(euler_config, "acoustic_theta_advection", "centered")
                == "van_leer"),
        )
        # 3b. Frozen SAM moist buoyancy (vapour-virtual + condensate loading) on
        #     w, applied EACH substep so the condensate-loading drag balances the
        #     dry θ' buoyancy at the same frequency (fixes the updraft runaway).
        if b_moist is not None:
            w_new = (w_new + dt_s * b_moist).at[..., 0].set(0.0).at[..., -1].set(0.0)

        # 4. Horizontal mass-flux divergence (backward: uses new u, v).
        #    rho_total reflects the pre-update rho_p_c (forward-backward
        #    convention: the mass flux uses the state at the start of the
        #    substep, the velocity from the just-completed forward step).
        rho_xface = interp_cell_to_xface_vlast(rho_total, grid)
        rho_yface = interp_cell_to_yface_vlast(rho_total, grid)
        horiz_div = divergence_vlast(
            rho_xface * u_new, rho_yface * v_new, grid,
        )
        rho_p_new = rho_p_vert - dt_s * horiz_div
        # Off-center the FULL divergence update (Skamarock-Klemp 2008).
        # rho_p_vert already carries -dt_s*vert_div (kernel, beta=0), so
        # rho_p_new now carries the total -dt_s*(vert+horiz)_div; apply
        # the off-centering blend once on the combined result.
        if beta != 0.0:
            rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

        u_c, v_c, w_c, theta_p_c, rho_p_c = (
            u_new, v_new, w_new, theta_p_new, rho_p_new,
        )

    return PlaneNonHydrostaticState(
        u=state.u.replace(data=u_c),
        v=state.v.replace(data=v_c),
        w=state.w.replace(data=w_c),
        theta_prime=state.theta_prime.replace(data=theta_p_c),
        rho_prime=state.rho_prime.replace(data=rho_p_c),
        phis=state.phis,
        tracers=state.tracers,
    )


def plane_acoustic_substeps_si_horizontal_halo(
    state: PlaneNonHydrostaticState,
    slow_tend: PlaneNonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
    grid: PlaneGrid,
    layout,
) -> PlaneNonHydrostaticState:
    """Halo-aware full Skamarock-Klemp split-explicit acoustic substep.

    Multi-rank counterpart of
    :func:`plane_acoustic_substeps_si_horizontal`: same forward-backward
    scheme, but every horizontal stencil reads a halo-padded slab
    instead of a ``jnp.roll`` over the (absent) global array. Per
    substep it issues exactly TWO packed halo rounds:

    1. ``(theta', rho')`` before the update — the Exner perturbation
       and the sanitized ``theta_total``/``rho_total`` are pointwise,
       so they are computed ON the padded slabs; the C-grid pressure
       gradient and face interpolations then produce interior-shape
       ``u``/``v`` updates.
    2. the horizontal mass fluxes ``(rho_xface*u_new, rho_yface*v_new)``
       — re-padded so the backward C-grid divergence closes on the
       interior.

    The vertical implicit solve, vertical continuity and vertical theta
    advection reuse the shared column kernel (column-local, no MPI).
    At ``layout.n_ranks == 1`` the packed exchange degenerates to
    ``jnp.pad(mode='wrap')`` and this function reproduces the serial
    substep (same stencil arithmetic on wrapped halos).

    ``layout`` and ``n_substeps`` are static, so the Python substep
    loop unrolls at trace time with one sendrecv token chain per
    exchange.
    """
    del slow_tend
    from legoesm.atmosphere.dynamics.les import plane_operators_halo as oh
    from legoesm.parallel.plane_mpi import packed_exchange_halo_plane_yxz

    g = euler_config.g
    J = terrain_metric.jacobian
    beta = euler_config.acoustic_off_centering
    implicit_buoyancy = euler_config.implicit_buoyancy
    c_p = _c_pd_constant()
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    si_w_filter_nu = float(getattr(
        euler_config, "si_w_vertical_filter_nu", 0.0,
    ))
    h = layout.halo

    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    tri_bands = precompute_si_tridiag_bands(
        height_coord, J, dt_s, g, implicit_buoyancy,
        nlev=theta_p.shape[-1],
    )

    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        compute_exner_perturbation,
    )

    b_moist = _acoustic_moist_buoyancy_w(state, height_coord, euler_config, layout)
    u_c, v_c, w_c, theta_p_c, rho_p_c = u, v, w, theta_p, rho_p
    for _ in range(int(n_substeps)):
        # Exchange 1: θ', ρ'. sanitize + Exner are pointwise, so the
        # padded totals/π' are exact on the halo ring; the reach-1
        # C-grid stencils below then close on the interior.
        theta_p_pad, rho_p_pad = packed_exchange_halo_plane_yxz(
            theta_p_c, rho_p_c, layout=layout,
        )
        theta_total_pad, rho_total_pad = sanitize_theta_rho(
            theta_0 + theta_p_pad, rho_0 + rho_p_pad,
        )
        # 1+2. Horizontal pressure gradient -> forward u, v update
        #      (interior-shape results from the padded slabs).
        pi_p_pad = compute_exner_perturbation(
            rho_p_pad, theta_p_pad, height_coord,
        )
        grad_pi_x = oh.grad_x_vlast_halo(pi_p_pad, grid, h)
        grad_pi_y = oh.grad_y_vlast_halo(pi_p_pad, grid, h)
        theta_xface = oh.interp_cell_to_xface_vlast_halo(
            theta_total_pad, grid, h,
        )
        theta_yface = oh.interp_cell_to_yface_vlast_halo(
            theta_total_pad, grid, h,
        )
        u_new = u_c + dt_s * (-c_p * theta_xface * grad_pi_x)
        v_new = v_c + dt_s * (-c_p * theta_yface * grad_pi_y)

        # 3. Vertical acoustic implicit (w) + vertical continuity +
        #    vertical theta advection (shared column kernel; column-
        #    local, interior arrays). beta=0.0 here — the Skamarock-
        #    Klemp off-centering is applied ONCE on the combined
        #    vertical+horizontal divergence below, exactly like the
        #    serial si_horizontal substep.
        w_new, theta_p_new, rho_p_vert = semi_implicit_acoustic_column_kernel(
            w_c, theta_p_c, rho_p_c,
            height_coord, J, dt_s, 0.0, g,
            implicit_buoyancy=implicit_buoyancy,
            precomputed_tridiag=tri_bands,
            si_w_vertical_filter_nu=si_w_filter_nu,
            theta_vert_van_leer=(
                getattr(euler_config, "acoustic_theta_advection", "centered")
                == "van_leer"),
        )
        # 3b. Frozen SAM moist buoyancy on w, each substep (layout-aware
        #     global mean inside the helper when configured).
        if b_moist is not None:
            w_new = (w_new + dt_s * b_moist).at[..., 0].set(0.0).at[..., -1].set(0.0)

        # 4. Horizontal mass-flux divergence (backward: uses new u, v;
        #    mass flux uses the start-of-substep rho_total).
        rho_xface = oh.interp_cell_to_xface_vlast_halo(
            rho_total_pad, grid, h,
        )
        rho_yface = oh.interp_cell_to_yface_vlast_halo(
            rho_total_pad, grid, h,
        )
        # Exchange 2: re-pad the interior fluxes so the C-grid
        # divergence closes on the interior.
        flux_u_pad, flux_v_pad = packed_exchange_halo_plane_yxz(
            rho_xface * u_new, rho_yface * v_new, layout=layout,
        )
        horiz_div = oh.divergence_vlast_halo(flux_u_pad, flux_v_pad, grid, h)
        rho_p_new = rho_p_vert - dt_s * horiz_div
        # Off-center the FULL divergence update (Skamarock-Klemp 2008),
        # once, on the combined vertical+horizontal result.
        if beta != 0.0:
            rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

        u_c, v_c, w_c, theta_p_c, rho_p_c = (
            u_new, v_new, w_new, theta_p_new, rho_p_new,
        )

    return PlaneNonHydrostaticState(
        u=state.u.replace(data=u_c),
        v=state.v.replace(data=v_c),
        w=state.w.replace(data=w_c),
        theta_prime=state.theta_prime.replace(data=theta_p_c),
        rho_prime=state.rho_prime.replace(data=rho_p_c),
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
    # using the shared semi_implicit_acoustic_column_kernel).
    # ``sponge_coeff > 0`` is now supported (PR2d) — the Rayleigh
    # sponge is applied inside ``plane_compressible_euler_slow_tendencies``
    # to ``u``, ``v``, ``theta'`` at full levels and ``w`` at half
    # levels via the shared ``sponge_profile`` taper.
    # ``hyperdiff_coeff`` and friends are now supported (PR3a) — the
    # biharmonic ``-coeff * Lap(Lap(field))`` term is added to the
    # slow tendency in
    # :func:`plane_compressible_euler_slow_tendencies`. Setting all
    # three to zero (the ``CompressibleEulerConfig`` default for
    # plane-friendly setups) reproduces the PR2b behaviour exactly.
    # Negative coefficients would invert the damping sign and produce
    # exponential growth — almost certainly a user error — so reject
    # them up front rather than silently treating them as off.
    # substep_horizontal_acoustic moves the horizontal PG + continuity
    # OUT of the slow tendency; only the SI-horizontal acoustic substep
    # integrates them. With semi_implicit_acoustic=False the explicit
    # vertical substep would run instead and the horizontal PG would be
    # integrated NOWHERE — silent physics loss. Refuse the combination.
    if (getattr(config, "substep_horizontal_acoustic", False)
            and not config.semi_implicit_acoustic):
        raise ValueError(
            "substep_horizontal_acoustic=True requires "
            "semi_implicit_acoustic=True: the horizontal pressure "
            "gradient and mass-continuity divergence are removed from "
            "the slow tendency and integrated only by the SI-horizontal "
            "acoustic substep. Enable semi_implicit_acoustic, or turn "
            "substep_horizontal_acoustic off."
        )
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
    closure = getattr(config, "turbulence_closure", "smagorinsky")
    if closure not in _VALID_TURBULENCE_CLOSURES:
        raise ValueError(
            f"turbulence_closure={closure!r} invalid; use 'smagorinsky' "
            "(CRM/LES eddy viscosity, SAM-faithful default), 'vreman' (optional "
            "Vreman-2004 eddy viscosity), 'amd' (optional Anisotropic "
            "Minimum-Dissipation eddy viscosity), 'molecular' (DNS molecular "
            "viscosity), or 'none' (inviscid)."
        )
    if closure == "amd":
        if getattr(config, "amd_c", 0.0) <= 0.0:
            raise ValueError(
                f"amd_c={getattr(config, 'amd_c', 0.0)!r} must be > 0 for "
                "turbulence_closure='amd' (modified Poincaré const, e.g. 0.3)."
            )
        if getattr(config, "smagorinsky_dynamic", False):
            raise ValueError(
                "turbulence_closure='amd' is incompatible with "
                "smagorinsky_dynamic=True (AMD is an inherently static, "
                "self-contained closure — there is no dynamic-coefficient path)."
            )
        if config.smagorinsky_prandtl <= 0.0:
            raise ValueError(
                f"smagorinsky_prandtl={config.smagorinsky_prandtl!r} must be > 0 "
                "for turbulence_closure='amd' (K_h = K_m / Pr)."
            )
    if closure == "vreman":
        if config.vreman_c <= 0.0:
            raise ValueError(
                f"vreman_c={config.vreman_c!r} must be > 0 for "
                "turbulence_closure='vreman' (e.g. 0.07 ≈ 2.5·C_s²)."
            )
        if getattr(config, "smagorinsky_dynamic", False):
            raise ValueError(
                "turbulence_closure='vreman' is incompatible with "
                "smagorinsky_dynamic=True (Vreman is an inherently static, "
                "self-contained closure — there is no dynamic-coefficient path)."
            )
        if config.smagorinsky_prandtl <= 0.0:
            raise ValueError(
                f"smagorinsky_prandtl={config.smagorinsky_prandtl!r} must be > 0 "
                "for turbulence_closure='vreman' (K_h = K_m / Pr)."
            )
    if closure == "molecular":
        if getattr(config, "molecular_viscosity", 0.0) <= 0.0:
            raise ValueError(
                f"molecular_viscosity={getattr(config, 'molecular_viscosity', 0.0)!r}"
                " must be > 0 for turbulence_closure='molecular' (DNS); set e.g."
                " constants.nu_air = 1.5e-5 m^2/s."
            )
        if getattr(config, "molecular_prandtl", 0.0) <= 0.0:
            raise ValueError(
                f"molecular_prandtl={getattr(config, 'molecular_prandtl', 0.0)!r} "
                "must be > 0; the DNS heat/scalar diffusivity K_h = nu / Pr "
                "inverts or NaNs for Pr <= 0."
            )
        if not getattr(config, "sgs_vertical_diffusion", False):
            import warnings
            warnings.warn(
                "turbulence_closure='molecular' (DNS) with "
                "sgs_vertical_diffusion=False applies the molecular viscosity in "
                "the HORIZONTAL only; the vertical nu d^2/dz^2 leg is OFF, so this "
                "is NOT a full 3-D DNS. Set sgs_vertical_diffusion=True for the "
                "complete molecular operator (1/rho) d_z(rho nu d_z phi).",
                stacklevel=2,
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
        # Eagerly precomputed state-independent mass integral
        # sum(rho_ref * J * area_T * dz). Computed HERE (outside any
        # trace) so its bits match the eagerly computed target mass at
        # rest; inside _step_jit it is embedded as a literal constant
        # (an in-trace recompute would be const-folded by XLA with a
        # different accumulation order — see
        # fix_mass_nonhydrostatic_plane). Constant for the model's
        # lifetime, so no reset needed (unlike _target_mass). Only the
        # fix_mass branch of _step_jit reads it, so skip the device
        # reduction when the fixer is off (static config value —
        # feature-gate Python `if`, not jnp.where). The fixer itself
        # tolerates None (computes inline), so this never traps.
        self._background_mass: jax.Array | None = (
            _plane_background_mass(grid, height_coord, terrain_metric)
            if config.fix_mass else None
        )

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
        # NH plane does not thread a PhysicsState carry — refuse a stateful
        # make_physics fn rather than silently reseed (#405/#413).
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="NH plane step()")
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
        Acoustic substeps are vertical-only (column-local, no MPI)
        unless ``substep_horizontal_acoustic=True``, which routes to
        the halo-aware full Skamarock-Klemp substep
        (:func:`plane_acoustic_substeps_si_horizontal_halo` — two
        packed halo rounds per acoustic substep).

        Multi-rank runs a cached jit'd split-explicit core (mpi4jax
        sendrecv traces inside ``jax.jit`` — one token chain per
        exchange; see the jit-cache block). Single-rank gets jit speedup
        via the existing :meth:`step` path.

        Mass fixer is SKIPPED on multi-rank (compute_dry_mass_plane
        is a global reduction — separate MPI variant needed). Single
        rank with ``fix_mass=True`` falls back to :meth:`step`.

        Smagorinsky LES + vertical-θ diffusion now supported on the
        halo path (R4/R5 — see ``compressible_euler_plane_halo``).
        Mass fixer R7: pass ``owned_mask`` (rank-local 0/1 mask of
        cells the rank owns) to use the MPI-aware dry-mass fixer
        :func:`legoesm.atmosphere.dynamics.crm.rce_mpi.fix_mass_nonhydrostatic_plane_mpi`.
        Without ``owned_mask`` the multi-rank fix_mass branch is
        skipped (mass not anchored to target) — single-rank still
        uses the serial fixer via ``step``.
        """
        from legoesm.atmosphere.dynamics.les.compressible_euler_plane_halo import (
            plane_compressible_euler_slow_tendencies_halo,
        )
        from legoesm.timestepping.split_explicit import (
            split_explicit_step, SplitExplicitConfig,
        )

        if layout.n_ranks == 1:
            # Single-rank: halo path is bit-identical to the standard
            # jit'd step (proven by test_halo_equiv_*) — route to it
            # for the JIT speedup. Skips packed exchange overhead +
            # gives ~14x faster per-step on small grids. The standard
            # step() honours substep_horizontal_acoustic.
            return self.step(state_local, dt)

        # Full Skamarock-Klemp split on multi-rank: the halo-aware
        # SI-horizontal substep (plane_acoustic_substeps_si_horizontal_halo)
        # exchanges (θ', ρ') + the mass fluxes on EVERY acoustic substep
        # (two packed rounds/substep) and the halo slow tendency drops
        # the horizontal PG + continuity so nothing is double-counted.
        # validate_plane_config guarantees semi_implicit_acoustic=True
        # whenever this flag is set.
        substep_horiz = getattr(
            self.config, "substep_horizontal_acoustic", False,
        )

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
            if substep_horiz:
                def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                    return plane_acoustic_substeps_si_horizontal_halo(
                        s, slow_tend, dt_s, n_sub, cfg,
                        self.height_coord, self.terrain_metric, self.config,
                        self.grid, layout,
                    )
            else:
                def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                    return plane_acoustic_substeps_semi_implicit(
                        s, slow_tend, dt_s, n_sub, cfg,
                        self.height_coord, self.terrain_metric, self.config,
                        layout=layout,  # enables exact global moist-mean when configured
                    )
        else:
            def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                return plane_acoustic_substeps(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, self.config,
                    layout=layout,  # enables exact global moist-mean when configured
                )

        # Build + cache a JIT'd split-explicit core.  The eager multi-rank
        # path host-syncs on every op and every mpi4jax halo ``sendrecv``,
        # making it ~100x slower than np=1 (measured: np2 7135 ms vs np1
        # 68.6 ms/step).  The docstring's "mpi4jax not jit-safe on macOS
        # shared-mem" caveat does NOT hold on Linux/MPICH — ``voronoi_mpi``
        # runs mpi4jax collectives inside ``@jax.jit`` at scale.  JIT-ing the
        # slow-tendency (incl. its halo exchange) + acoustic substeps as one
        # fused executable collapses the eager dispatch/host-sync overhead.
        # The mass fixer stays eager below (cheap — one allreduce/step — and
        # it mutates the ``self._target_mass`` anchor, which a jit can't trace).
        # Cache keyed on the trace-invariant statics so we build the jit once;
        # ``dt`` is closed in (static), so a changed dt rebuilds.
        # The FULL layout NamedTuple (all-int, hashable) is part of the
        # key: the compiled closure bakes in rank, neighbour ranks, halo
        # width and start offsets via the captured halo exchanges — a
        # same-shaped but different layout (repartition, halo 1→3, rank
        # remap in tests) must NOT reuse the stale compiled graph.
        _jit_key = (
            layout,
            float(dt), self.config.n_acoustic_substeps,
            bool(self.config.semi_implicit_acoustic),
            bool(substep_horiz), id(f_pad_cached),
        )
        if (getattr(self, "_jit_halo_core", None) is None
                or getattr(self, "_jit_halo_key", None) != _jit_key):
            def _halo_core(s):
                return split_explicit_step(
                    s, slow_tendency_fn, acoustic_update_fn, dt, se_config,
                )
            self._jit_halo_core = jax.jit(_halo_core)
            self._jit_halo_key = _jit_key
        stepped = self._jit_halo_core(state_local)

        # MPI-aware dry-mass fixer (R7). Only fires when the user
        # passes an owned_mask + config.fix_mass is True. Without the
        # mask we can't compute a non-double-counted global mass.
        if self.config.fix_mass and owned_mask is not None:
            from legoesm.atmosphere.dynamics.crm.rce_mpi import (
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
            if getattr(self.config, "substep_horizontal_acoustic", False):
                def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
                    return plane_acoustic_substeps_si_horizontal(
                        s, slow_tend, dt_s, n_sub, cfg,
                        self.height_coord, self.terrain_metric, self.config,
                        self.grid,
                    )
            else:
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
                # Concrete eager constant read off the static ``self``
                # at trace time — bit-preserved as a jaxpr literal.
                background_mass=self._background_mass,
            )
        return state_new
