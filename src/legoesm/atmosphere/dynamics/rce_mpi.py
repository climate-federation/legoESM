"""MPI-aware RCE helpers — global reductions for the plane CRM.

The serial RCE stack (PRs #302, #303) uses local
``jnp.mean(state.u.data, axis=(0, 1))`` and
``jnp.sum(q_total * column_w)`` to compute the domain-mean wind +
total water mass. On a single rank these ARE the global reductions
the algorithms want; under MPI, per-rank reductions miss the
contributions from other ranks and silently scale by the
WRONG factor (mean wind becomes per-rank mean, moist mass becomes
per-rank mass).

This module provides MPI-aware variants:

* :func:`remove_horizontal_mean_wind_plane_mpi` — global mean wind
  via ``global_sum_mpi`` + ``layout.global_horizontal_cells``
  divisor. AD-safe (allreduce SUM).
* :func:`compute_total_water_mass_plane_mpi` — local sum over OWNED
  cells (interior excluding halos) + ``global_sum_mpi``. AD-safe.
* :func:`fix_moist_mass_plane_mpi` — uses the MPI total-water
  reduction in the multiplicative correction factor so every rank
  applies the same factor and the global moist mass returns to
  target.

The serial variants in :mod:`mean_wind_filter` and
:mod:`moist_mass_fixer` remain the right tool when no MPI layout
is in play (smoke tests, single-node integration).

Single-rank short-circuit
-------------------------
``layout.n_ranks == 1`` calls the serial variant under the hood
(skips MPI overhead). This lets the MPI helpers be unit-tested
without an ``mpirun`` launcher.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.mean_wind_filter import (
    remove_horizontal_mean_wind,
)
from legoesm.atmosphere.dynamics.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)


def _validate_layout(layout) -> None:
    """Layout must carry ``n_ranks`` + ``rank`` + the global
    ``ny_global``, ``nx_global`` cell counts."""
    for name in ("n_ranks", "rank", "ny_global", "nx_global"):
        if not hasattr(layout, name):
            raise TypeError(
                f"layout missing attribute {name!r}; "
                f"got {type(layout).__name__}."
            )


def _validate_owned_mask(owned_mask, expected_shape) -> None:
    if owned_mask.shape != expected_shape:
        raise ValueError(
            f"owned_mask shape {owned_mask.shape} != "
            f"expected {expected_shape}."
        )


def _validate_plane_state_for_mpi_water(
    state, height_coord, grid, water_slot_indices,
) -> None:
    """Codex iter-2: full shape contract on the multi-rank path
    (previously only enforced on the serial-fallback branch via the
    serial helper's own validators)."""
    rho_p = state.rho_prime.data
    if rho_p.ndim != 3:
        raise ValueError(
            f"rho_prime must be 3D (ny, nx, nlev); got ndim={rho_p.ndim} "
            f"shape={rho_p.shape}."
        )
    ny, nx, nlev = rho_p.shape
    if grid.area_T.shape != (ny, nx):
        raise ValueError(
            f"grid.area_T shape {grid.area_T.shape} != (ny={ny}, nx={nx})."
        )
    if height_coord.dz.shape != (nlev,):
        raise ValueError(
            f"hc.dz shape {height_coord.dz.shape} != (nlev={nlev},)."
        )
    tracers = state.tracers.data
    if tracers.ndim != 4:
        raise ValueError(
            f"tracers must be 4D (ny, nx, nlev, n_tracers); got "
            f"ndim={tracers.ndim} shape={tracers.shape} — a 3D array "
            f"would let shape[-1] alias the vertical axis as a tracer."
        )
    if tracers.shape[:3] != (ny, nx, nlev):
        raise ValueError(
            f"tracers leading 3 axes {tracers.shape[:3]} != "
            f"(ny, nx, nlev) = ({ny}, {nx}, {nlev})."
        )
    if len(water_slot_indices) == 0:
        raise ValueError("water_slot_indices must contain at least one slot.")
    if len(set(water_slot_indices)) != len(water_slot_indices):
        raise ValueError(
            f"water_slot_indices contains duplicates: "
            f"{water_slot_indices}."
        )
    n_tracers = tracers.shape[-1]
    for idx in water_slot_indices:
        if not isinstance(idx, int):
            raise ValueError(
                f"water_slot_indices entries must be int; got {idx!r}."
            )
        if idx < 0 or idx >= n_tracers:
            raise ValueError(
                f"water slot {idx} out of range [0, {n_tracers})."
            )


def remove_horizontal_mean_wind_plane_mpi(state, layout, owned_mask):
    """MPI-aware mean-wind removal (Codex iter-2: ownership-aware).

    Computes the GLOBAL horizontal mean of u, v across all ranks +
    subtracts it. Single-rank case short-circuits to the serial
    variant.

    ``owned_mask`` shape ``(ny_local, nx_local)`` — 1.0 on cells
    this rank is responsible for, 0.0 on halo / overlap rows that
    other ranks own. Used to weight both the numerator
    (``sum(u_local * mask)``) and the denominator
    (``global_sum_mpi(sum(mask))``), so halo cells are not
    double-counted across ranks.

    Caller is responsible for constructing ``owned_mask`` consistent
    with the pencil-layout decomposition (typically all-ones on
    interior + halo zeroed if the pencil includes ghost rows).
    """
    _validate_layout(layout)
    if layout.n_ranks == 1:
        return remove_horizontal_mean_wind(state)
    # Plane-only contract: u, v must be 3D.
    if state.u.data.ndim != 3 or state.v.data.ndim != 3:
        raise ValueError(
            f"remove_horizontal_mean_wind_plane_mpi requires plane "
            f"(3D) u, v arrays; got u.ndim={state.u.data.ndim}, "
            f"v.ndim={state.v.data.ndim}."
        )
    ny, nx, _ = state.u.data.shape
    _validate_owned_mask(owned_mask, (ny, nx))
    # Deferred import — keep this module importable without mpi4jax.
    from legoesm.parallel.reductions import global_sum_mpi

    mask_3d = owned_mask[:, :, None]
    u_local_sum = jnp.sum(state.u.data * mask_3d, axis=(0, 1))
    v_local_sum = jnp.sum(state.v.data * mask_3d, axis=(0, 1))
    # Global owned-cell count: same denominator for every level.
    local_count = jnp.sum(owned_mask)
    global_count = global_sum_mpi(local_count)
    u_global_mean = global_sum_mpi(u_local_sum) / global_count
    v_global_mean = global_sum_mpi(v_local_sum) / global_count
    new_u_data = state.u.data - u_global_mean[None, None, :]
    new_v_data = state.v.data - v_global_mean[None, None, :]
    return state._replace(
        u=state.u.replace(data=new_u_data),
        v=state.v.replace(data=new_v_data),
    )


def compute_total_water_mass_plane_mpi(
    state, height_coord, grid, layout, owned_mask,
    water_slot_indices: Sequence[int] = (0, 1, 2),
) -> jax.Array:
    """MPI-aware total-water mass.

    Per-rank local sum × owned_mask (zeroes the halo / overlap rows
    so they aren't double-counted), then ``global_sum_mpi`` across
    ranks. Returns the GLOBAL moist mass.

    ``owned_mask`` shape ``(ny_local, nx_local)`` — 1.0 on cells
    this rank is responsible for, 0.0 on duplicate halo rows.
    Pre-computed by the caller from the layout (mirrors the
    voronoi_mpi convention).

    Single-rank short-circuit uses
    :func:`compute_total_water_mass_plane` directly.
    """
    _validate_layout(layout)
    if layout.n_ranks == 1:
        return compute_total_water_mass_plane(
            state, height_coord, grid,
            water_slot_indices=water_slot_indices,
        )
    _validate_plane_state_for_mpi_water(
        state, height_coord, grid, water_slot_indices,
    )
    from legoesm.parallel.reductions import global_sum_mpi

    rho_p = state.rho_prime.data
    ny, nx, nlev = rho_p.shape
    _validate_owned_mask(owned_mask, (ny, nx))
    rho_total = height_coord.rho_ref + rho_p
    weight_2d = grid.area_T[:, :, None] * owned_mask[:, :, None]
    dz = height_coord.dz
    column_w = rho_total * weight_2d * dz
    q_total = jnp.zeros_like(column_w)
    for idx in water_slot_indices:
        q_total = q_total + state.tracers.data[..., idx]
    local_mass = jnp.sum(q_total * column_w)
    return global_sum_mpi(local_mass)


def fix_moist_mass_plane_mpi(
    state, height_coord, grid, layout, owned_mask,
    target_total_water,
    water_slot_indices: Sequence[int] = (0, 1, 2),
):
    """MPI-aware moist-mass fixer.

    Multiplicative rescale ``q_new = q · factor`` with
    ``factor = target / current_global``. Every rank gets the same
    factor because ``current_global`` flows through ``global_sum_mpi``.
    Local positivity rules from the serial fixer still apply.

    Single-rank short-circuit uses :func:`fix_moist_mass_plane`.
    """
    _validate_layout(layout)
    if layout.n_ranks == 1:
        return fix_moist_mass_plane(
            state, height_coord, grid, target_total_water,
            water_slot_indices=water_slot_indices,
        )
    current_global = compute_total_water_mass_plane_mpi(
        state, height_coord, grid, layout, owned_mask,
        water_slot_indices=water_slot_indices,
    )
    safe_current = jnp.where(current_global > 0.0, current_global, 1.0)
    factor = jnp.where(
        current_global > 0.0, target_total_water / safe_current, 1.0,
    )
    new_tracers = state.tracers.data
    for idx in water_slot_indices:
        new_tracers = new_tracers.at[..., idx].set(
            new_tracers[..., idx] * factor,
        )
    return state._replace(
        tracers=state.tracers.replace(data=new_tracers),
    )
