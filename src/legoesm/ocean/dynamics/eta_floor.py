"""Mass-conserving eta-floor clamping for barotropic solvers (#176).

Redistributes mass injected by the floor clamp over cells with headroom.
"""

from __future__ import annotations

import jax.numpy as jnp


def _is_multi_process() -> bool:
    import jax
    if jax.process_count() > 1:
        return True
    from legoesm.core.operators import _is_distributed
    return _is_distributed()


def _global_sum(x):
    if _is_multi_process():
        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(x)
    return x


def _global_sum_pair(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reduce two scalars in a single MPI message.

    The barotropic substep loop calls ``clamp_and_redistribute`` twice
    per substep, each iteration of which previously issued two separate
    ``allreduce`` calls.  At ``n_substeps≈30`` and ``n_iter=3`` that is
    ``30·2·3·2 = 360`` reductions per ocean timestep.  Stacking the
    two scalars and reducing once halves that to 180 — and the dominant
    barotropic-loop cost on multi-GPU runs is precisely this MPI
    latency, not bandwidth.
    """
    if _is_multi_process():
        from legoesm.parallel.reductions import batch_allreduce_mpi
        a_g, b_g = batch_allreduce_mpi([a, b], op="sum")
        return a_g, b_g
    return a, b


def clamp_and_redistribute(
    eta_unfloored: jnp.ndarray,
    eta_floor: jnp.ndarray,
    mask: jnp.ndarray,
    area: jnp.ndarray,
    n_iter: int = 3,
) -> jnp.ndarray:
    """Clamp eta to floor and redistribute injected mass (#176)."""
    eta_new = jnp.maximum(eta_unfloored, eta_floor) * mask

    for _ in range(n_iter):
        local_mass_added = jnp.sum((eta_new - eta_unfloored) * area * mask)
        above_floor = (eta_new > eta_floor + 1e-14) & (mask > 0.5)
        above_mask = above_floor.astype(eta_new.dtype)
        local_above_area = jnp.sum(area * above_mask)
        # One allreduce instead of two (hot in barotropic substeps).
        mass_added, above_area = _global_sum_pair(
            local_mass_added, local_above_area,
        )
        has_headroom = above_area > 0.0
        correction = jnp.where(has_headroom, mass_added / jnp.maximum(above_area, 1.0), 0.0)
        eta_new = eta_new - correction * above_mask
        eta_new = jnp.maximum(eta_new, eta_floor) * mask

    return eta_new
