"""Mass-conserving eta-floor clamping for barotropic solvers (#176).

Redistributes mass injected by the floor clamp over cells with headroom.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.parallel.reductions import (
    batch_allreduce_mpi,
    batch_psum_spmd,
    is_multi_process,
)


def _global_sum_pair(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reduce two scalars in a single MPI message.

    The barotropic substep loop calls ``clamp_and_redistribute`` twice
    per substep, each iteration of which previously issued two separate
    ``allreduce`` calls.  At ``n_substeps≈30`` and ``n_iter=3`` that is
    ``30·2·3·2 = 360`` reductions per ocean timestep.  Stacking the
    two scalars and reducing once halves that to 180 — and the dominant
    barotropic-loop cost on multi-GPU runs is precisely this MPI
    latency, not bandwidth.

    SPMD (single-controller lat-band shard_map, route-B multi-GPU — no mpi4jax):
    ``a`` / ``b`` are PARTIAL sums over this device's latitude band and MUST be
    reduced across the ``"lat"`` mesh axis with ``jax.lax.psum`` — checked FIRST
    because ``is_multi_process()`` is FALSE under one process, so the MPI gate
    below would return each band's PARTIAL ``(mass_added, above_area)`` and the
    per-substep eta mass-redistribution correction (``mass_added/above_area``)
    would be per-band-WRONG (the dominant SPMD-equivalence error: this fires
    ``n_iter`` × per substep × per step even at the default
    ``barotropic_local_subcycle_clamp=False``).  Keyed on the ``"lat"`` axis BY
    NAME so a coupled cube-atm SPMD mesh falls through (ocean fields are never
    cube-sharded).  ``psum`` is self-transposing ⇒ AD-safe.  Inert for serial /
    MPI / cube.
    """
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() == "spmd":
        mesh = get_spmd_mesh()
        if mesh is None:
            raise RuntimeError(
                "_global_sum_pair: halo backend is 'spmd' but no SPMD mesh is "
                "set; arm it via activate_latlon_spmd_halo(mesh).")
        if "lat" in tuple(mesh.axis_names):
            # ONE packed psum for the pair instead of two separate psums
            # (M4 quick win: route through the canonical batched helper,
            # the same message-aggregation lever as the MPI leg below and
            # barotropic_implicit_latlon_cgrid's mass-fix reduction).
            # Packing is BIT-identical: the per-element reduction order
            # across the "lat" axis is unchanged by concatenation — gated
            # by tests/parallel/test_latlon_spmd_fused_halo.py
            # (test_global_sum_pair_spmd_batched_*).  psum stays
            # self-transposing ⇒ AD-safe.
            a_g, b_g = batch_psum_spmd([a, b], "lat")
            return a_g, b_g
    if is_multi_process():
        a_g, b_g = batch_allreduce_mpi([a, b], op="sum")
        return a_g, b_g
    return a, b


def clamp_and_redistribute(
    eta_unfloored: jnp.ndarray,
    eta_floor: jnp.ndarray,
    mask: jnp.ndarray,
    area: jnp.ndarray,
    n_iter: int = 3,
    *,
    owned_weight: jnp.ndarray | None = None,
    force_global: bool = False,
) -> jnp.ndarray:
    """Clamp eta to floor and redistribute injected mass (#176).

    ``owned_weight`` / ``force_global`` (distributed Voronoi/MPAS): the
    partition's local arrays carry HALO cells — unweighted local sums
    double-count them in the allreduce.  ``is_multi_process()`` IS
    layout-aware (2026-06-11), so the reduction fires on that path —
    which makes the owned weighting MANDATORY there: pass
    ``owned_weight=owned_mask_cells`` (and ``force_global=True`` for an
    explicit schedule).  Defaults preserve the historical lat-lon/band
    behavior bit-exactly (no weight, runtime-gated reduction; band rows
    partition without overlap so no weight is needed there).
    """
    eta_new = jnp.maximum(eta_unfloored, eta_floor) * mask
    _ow = (jnp.ones_like(mask) if owned_weight is None
           else owned_weight.astype(eta_new.dtype))

    for _ in range(n_iter):
        local_mass_added = jnp.sum(
            (eta_new - eta_unfloored) * area * mask * _ow)
        above_floor = (eta_new > eta_floor + 1e-14) & (mask > 0.5)
        above_mask = above_floor.astype(eta_new.dtype)
        local_above_area = jnp.sum(area * above_mask * _ow)
        # One allreduce instead of two (hot in barotropic substeps).
        if force_global:
            mass_added, above_area = batch_allreduce_mpi(
                [local_mass_added, local_above_area], op="sum",
            )
        else:
            mass_added, above_area = _global_sum_pair(
                local_mass_added, local_above_area,
            )
        has_headroom = above_area > 0.0
        correction = jnp.where(has_headroom, mass_added / jnp.maximum(above_area, 1.0), 0.0)
        eta_new = eta_new - correction * above_mask
        eta_new = jnp.maximum(eta_new, eta_floor) * mask

    return eta_new
