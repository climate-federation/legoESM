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


def apply_top_sponge_field_scale(
    field: jnp.ndarray,
    damp_x: float,
    nord_x: int,
    factor: float,
    d2_bg: float,
    d2_bg_k1: float,
    d2_bg_k2: float,
    apply_at_k2: bool = True,
) -> jnp.ndarray:
    """FV3_3D iter 447: shared FV3 sponge linear-scaling trick
    for ``damp_w`` and ``damp_v`` post-step del-n fluxes.

    Replaces three duplicate iter-441/442/443 inline blocks
    with one helper.  Exploits the linear ``damp^(nord+1)``
    dependence of del-n flux to swap the effective damping
    coefficient at sponge levels.

    Parameters
    ----------
    field : jax.Array, shape (..., nlev_or_more)
        del-n flux output (e.g., ``dw``, ``du_normal``,
        ``dv_normal``).  Last axis is level.
    damp_x : float
        Baseline damp coefficient (``damp_w`` or ``damp_v``).
    nord_x : int
        Order of del-n damping (``nord_w`` or ``nord_v``).
    factor : float
        FV3 ``damp_X`` scaling factor relative to ``d2_divg``:
        1.0 for damp_w (FV3 ``damp_w = d2_divg``),
        0.5 for damp_v (FV3 ``damp_vt = 0.5 * d2_divg``).
    d2_bg : float
        Background corner-div d2 coefficient.
    d2_bg_k1 : float
        Sponge boost at k=0 (FV3 ``d2_bg_k1``).
    d2_bg_k2 : float
        Sponge boost at k=1, k=2 (FV3 ``d2_bg_k2``).
    apply_at_k2 : bool
        True for damp_w (FV3 ``d2_divg`` at k=3 → damp_w);
        False for damp_v (FV3 k=3 has no do_vort_damp block).

    Returns
    -------
    jax.Array
        ``field`` with top-sponge levels scaled.  Same shape.
    """
    if d2_bg_k1 <= 0.0 and d2_bg_k2 <= 0.01:
        return field
    nlev = field.shape[-1]
    k_idx = jnp.arange(nlev)
    if d2_bg_k1 > 0.0:
        boosted_k1 = factor * jnp.maximum(d2_bg, d2_bg_k1)
        scale_k1 = (boosted_k1 / damp_x) ** (nord_x + 1)
        field = jnp.where(k_idx == 0, field * scale_k1, field)
    if d2_bg_k2 > 0.01:
        boosted_k2 = factor * jnp.maximum(d2_bg, d2_bg_k2)
        scale_k2 = (boosted_k2 / damp_x) ** (nord_x + 1)
        field = jnp.where(k_idx == 1, field * scale_k2, field)
        if apply_at_k2 and d2_bg_k2 > 0.05:
            boosted_k3 = factor * jnp.maximum(d2_bg, 0.2 * d2_bg_k2)
            scale_k3 = (boosted_k3 / damp_x) ** (nord_x + 1)
            field = jnp.where(k_idx == 2, field * scale_k3, field)
    return field


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
