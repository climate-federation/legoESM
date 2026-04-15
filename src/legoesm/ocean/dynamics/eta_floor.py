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
        mass_added = _global_sum(jnp.sum((eta_new - eta_unfloored) * area * mask))
        above_floor = (eta_new > eta_floor + 1e-14) & (mask > 0.5)
        above_mask = above_floor.astype(eta_new.dtype)
        above_area = _global_sum(jnp.sum(area * above_mask))
        has_headroom = above_area > 0.0
        correction = jnp.where(has_headroom, mass_added / jnp.maximum(above_area, 1.0), 0.0)
        eta_new = eta_new - correction * above_mask
        eta_new = jnp.maximum(eta_new, eta_floor) * mask

    return eta_new
