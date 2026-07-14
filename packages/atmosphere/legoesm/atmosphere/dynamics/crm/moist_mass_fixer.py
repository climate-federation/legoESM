"""Moist-mass conservation fixer for the plane NH CRM.

The dry-air mass fixer
(:func:`legoesm.atmosphere.dynamics.les.compressible_euler_plane.fix_mass_nonhydrostatic_plane`)
restores the dry-air mass integral after each step but leaves the
total water (q_v + q_c + q_r + q_i + ...) free to drift under:

* Microphysics phase changes that should be mass-conservative but
  accumulate floating-point round-off over long integrations.
* Tracer-positivity clipping that adds mass (when negatives are
  zeroed without compensation) or drops mass (when the compensated
  filter zeros a column with sum < 0).
* Surface evaporation that adds q_v but isn't tracked back to a
  conserved-quantity ledger.

This module provides a SIMPLE GLOBAL multiplicative fixer that
rescales the water-mass column so the total moist mass (Σ over
tracer slots) returns to a target. Inputs are kept POSITIVE (the
rescaling is a constant > 0 on every cell). Drift remains bounded
even under long-integration round-off accumulation.

Scope
-----
Plane only (vertical-last, uniform area). Operates on a configurable
subset of tracer slots (caller picks which slots count toward
"total water").
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


def _validate_water_slot_indices(
    water_slot_indices: Sequence[int], n_tracers: int,
) -> None:
    """Reject duplicates, negatives, and out-of-range slot indices
    (Codex iter-1). Silent indexing into the end of the tracer axis
    via negative indices, or double-counting via duplicates, would
    corrupt the moist-mass diagnostic without raising."""
    if len(water_slot_indices) == 0:
        raise ValueError(
            "water_slot_indices must contain at least one slot."
        )
    if len(set(water_slot_indices)) != len(water_slot_indices):
        raise ValueError(
            f"water_slot_indices contains duplicates: "
            f"{water_slot_indices} — double-counting would corrupt "
            f"the moist-mass diagnostic."
        )
    for idx in water_slot_indices:
        if not isinstance(idx, (int,)):
            raise ValueError(
                f"water_slot_indices entries must be int; "
                f"got {idx!r} of type {type(idx).__name__}."
            )
        if idx < 0 or idx >= n_tracers:
            raise ValueError(
                f"water slot {idx} out of range [0, {n_tracers}); "
                f"negative indices are not allowed (would silently "
                f"wrap to the end of the tracer axis)."
            )


def _validate_plane_shapes(state, height_coord, grid) -> None:
    """Plane-only shape contract (Codex iter-1). All horizontal
    quantities must agree on (ny, nx); vertical on nlev."""
    rho_p = state.rho_prime.data
    if rho_p.ndim != 3:
        raise ValueError(
            f"moist mass fixer is plane-only: expects rho_prime "
            f"shape (ny, nx, nlev); got ndim={rho_p.ndim} "
            f"shape={rho_p.shape}."
        )
    ny, nx, nlev = rho_p.shape
    if grid.area_T.shape != (ny, nx):
        raise ValueError(
            f"grid.area_T shape {grid.area_T.shape} != "
            f"(ny={ny}, nx={nx}); state/grid mismatch."
        )
    if height_coord.dz.shape != (nlev,):
        raise ValueError(
            f"hc.dz shape {height_coord.dz.shape} != (nlev={nlev},)."
        )
    if state.tracers.data.shape[:3] != (ny, nx, nlev):
        raise ValueError(
            f"tracers leading 3 axes {state.tracers.data.shape[:3]} "
            f"!= (ny, nx, nlev) = ({ny}, {nx}, {nlev})."
        )


def compute_total_water_mass_plane(
    state,
    height_coord,
    grid,
    water_slot_indices: Sequence[int] = (0, 1, 2),
) -> jax.Array:
    """Total water mass on the plane: ``Σ_slot Σ_(i,j,k) q · rho · J ·
    area_T · dz``.

    Uses dry density (no condensate loading) as the column weight —
    same convention as the dry-mass integral so the two diagnostics
    can be compared apples-to-apples.

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        Provides ``tracers.data`` (ny, nx, nlev, n_tracers) and
        ``rho_prime`` for the air density.
    height_coord : HeightCoordinate
        Provides ``rho_ref``, ``dz``.
    grid : PlaneGrid
        Provides ``area_T``.
    water_slot_indices : tuple of int
        Tracer-axis indices counted toward the moist mass. Default
        ``(0, 1, 2)`` = q_v + q_c + q_r (warm-rain microphysics).
        Use ``(0, 1, 2, 3)`` for ice schemes (q_v, q_c, q_r, q_i).
        **P3 warning**: do NOT include slots 4–5. Slot 4 is q_rim
        [kg/kg], which is a subset of q_i — including it double-counts
        rime mass. Slot 5 is B_rim [m³/kg_air], which is not a mixing
        ratio — summing it into a water-mass integral is a dimensional
        error. Use ``(0, 1, 2, 3)`` for P3.

    Returns
    -------
    jax.Array (scalar)
    """
    _validate_plane_shapes(state, height_coord, grid)
    _validate_water_slot_indices(
        water_slot_indices, state.tracers.data.shape[-1],
    )
    rho_total = height_coord.rho_ref + state.rho_prime.data    # (ny, nx, nlev)
    weight_2d = grid.area_T[:, :, None]                        # (ny, nx, 1)
    dz = height_coord.dz                                       # (nlev,)
    column_w = rho_total * weight_2d * dz                      # (ny, nx, nlev)
    q_total = jnp.zeros_like(column_w)
    for idx in water_slot_indices:
        q_total = q_total + state.tracers.data[..., idx]
    return jnp.sum(q_total * column_w)


def fix_moist_mass_plane(
    state,
    height_coord,
    grid,
    target_total_water: jax.Array,
    water_slot_indices: Sequence[int] = (0, 1, 2),
) -> "PlaneNonHydrostaticState":
    """Rescale all moist tracer slots so the total moist mass equals
    ``target_total_water``.

    Multiplicative correction ``q_new = q · factor`` with
    ``factor = target / current`` applied uniformly to every cell
    in every selected slot — preserves spatial pattern + relative
    slot partition; only absolute mass adjusts.

    Positivity caveat (Codex iter-1)
    --------------------------------
    The factor is positive when both target + current are positive,
    so positive inputs stay positive. BUT multiplying NEGATIVE
    inputs by a positive factor preserves the negatives — this
    fixer does NOT enforce positivity on its own. Run the tracer
    positivity filter (``legoesm.atmosphere.dynamics.shared.tracer_positivity``)
    BEFORE this fixer if the upstream advection or microphysics
    can produce negatives.

    Degenerate handling (Codex iter-1)
    ----------------------------------
    * ``current == 0`` AND ``target > 0`` → RAISES at the host
      assertion point (cannot scale zero up to a positive target;
      indicates the column has no water to redistribute).
    * ``current == 0`` AND ``target == 0`` → NO-OP (factor = 1).
    * ``current < 0`` → RAISES; this is a malformed state, the
      tracer-positivity filter should have caught it earlier.

    JIT-skip behaviour (Codex iter-2)
    ---------------------------------
    The degenerate-case checks call ``float(current)`` which raises
    ``TracerArrayConversionError`` under JIT; we trap that and skip
    the host-side raise so the function remains traceable. Inside a
    JIT-compiled hot loop the degenerate paths silently fall back
    to the ``jnp.where`` zero-current → factor = 1 no-op. Callers
    needing runtime enforcement inside JIT must wrap with
    ``jax.experimental.checkify.check`` upstream.

    Parameters
    ----------
    state, height_coord, grid : as in
        :func:`compute_total_water_mass_plane`.
    target_total_water : jax.Array
        Scalar target moist mass; typically snapshotted at IC time.
    water_slot_indices : tuple of int

    Returns
    -------
    PlaneNonHydrostaticState
    """
    _validate_plane_shapes(state, height_coord, grid)
    _validate_water_slot_indices(
        water_slot_indices, state.tracers.data.shape[-1],
    )
    current = compute_total_water_mass_plane(
        state, height_coord, grid, water_slot_indices=water_slot_indices,
    )
    # Host-side degenerate-case checks. Skipped under JIT (caller's
    # responsibility — see tracer_positivity for the same pattern).
    try:
        current_host = float(current)
        target_host = float(target_total_water)
        if current_host < 0.0:
            raise ValueError(
                f"moist mass fixer: current moist mass {current_host:.3e} "
                f"is negative; run tracer positivity filter first."
            )
        if current_host == 0.0 and target_host > 0.0:
            raise ValueError(
                f"moist mass fixer: current=0 but target={target_host:.3e}>0; "
                f"cannot scale zero column up to a positive target."
            )
    except (jax.errors.TracerArrayConversionError, TypeError):
        pass
    safe_current = jnp.where(current > 0.0, current, 1.0)
    factor = jnp.where(
        current > 0.0, target_total_water / safe_current, 1.0,
    )
    new_tracers = state.tracers.data
    for idx in water_slot_indices:
        new_tracers = new_tracers.at[..., idx].set(
            new_tracers[..., idx] * factor,
        )
    return state._replace(
        tracers=state.tracers.replace(data=new_tracers),
    )
