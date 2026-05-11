"""FV3_3D iter 446: shared FV3 sponge boost helper.

Factored from NH ``_apply_top_sponge_damp_boost`` (iter-440)
+ PE inline implementation (iter-438/439).  Both NH and PE
now call this single helper.

FV3 reference: ``dyn_core.F90:780, 792, 802``::

    if (k==1) d2_divg = max(0.01, d2_bg, d2_bg_k1)
    if (k==2 .and. d2_bg_k2 > 0.01)  d2_divg = max(d2_bg, d2_bg_k2)
    if (k==3 .and. d2_bg_k2 > 0.05)  d2_divg = max(d2_bg, 0.2*d2_bg_k2)
"""
from __future__ import annotations

import jax.numpy as jnp


def apply_top_sponge_damp_boost(
    damp_corner: jnp.ndarray,
    da_min_c: jnp.ndarray,
    d2_bg: float,
    d2_bg_k1: float,
    d2_bg_k2: float,
) -> jnp.ndarray:
    """Apply per-level FV3 sponge boost to the corner-div
    damping coefficient.

    Parameters
    ----------
    damp_corner : jax.Array, shape (..., nlev)
        Corner-divergence adaptive damping coefficient.  Last
        axis is level.
    da_min_c : jax.Array, scalar
        Min B-grid corner area (``min(area_corner)``).
    d2_bg : float
        Background damping coefficient
        (``corner_div_damp_d2_bg``).
    d2_bg_k1 : float
        Sponge-boost coefficient at k=0 (FV3 ``d2_bg_k1``).
        0.0 → no boost.
    d2_bg_k2 : float
        Sponge-boost coefficient at k=1, k=2 (FV3
        ``d2_bg_k2``).  Threshold gates: >0.01 for k=1, >0.05
        for k=2 (each overrides progressively).

    Returns
    -------
    jax.Array
        Per-level boosted ``damp_corner`` (same shape as input).
    """
    if d2_bg_k1 <= 0.0 and d2_bg_k2 <= 0.01:
        return damp_corner
    nlev = damp_corner.shape[-1]
    k_idx = jnp.arange(nlev)
    if d2_bg_k1 > 0.0:
        damp_k1 = da_min_c * jnp.maximum(d2_bg, d2_bg_k1)
        damp_corner = jnp.where(k_idx == 0, damp_k1, damp_corner)
    if d2_bg_k2 > 0.01:
        damp_k2 = da_min_c * jnp.maximum(d2_bg, d2_bg_k2)
        damp_corner = jnp.where(k_idx == 1, damp_k2, damp_corner)
        if d2_bg_k2 > 0.05:
            damp_k3 = da_min_c * jnp.maximum(d2_bg, 0.2 * d2_bg_k2)
            damp_corner = jnp.where(k_idx == 2, damp_k3, damp_corner)
    return damp_corner
