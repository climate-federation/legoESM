"""Sea ice tracer advection.

Advects ice tracers (thickness, concentration, temperature) by the ice
velocity field using **centered** flux-form transport on the cubed sphere.

For conserved quantities (volume = h*a, area = a), uses the flux-form
divergence operator.  Temperature is advected as enthalpy (T*h*a) to
preserve conservation.

**Limitations**: The centered divergence operator is not monotone — it
can generate new extrema (negative thickness, concentration > 1, or
out-of-range temperatures).  Post-transport clamps enforce physical
bounds but do not guarantee strict conservation.  A proper upwind or
FCT scheme would be needed for both monotonicity and conservation.

All functions are JAX-compatible (differentiable, JIT-friendly).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.operators_3d import divergence_3d
from legoesm.grids.cubed_sphere import CubedSphereGrid


def advect_ice_tracers(
    h_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    T_ice: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid: CubedSphereGrid,
    dt: float,
    T_ice_min: float = 180.0,
    T_freeze_ocean: float = 271.35,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advect ice tracers by the ice velocity field.

    Uses the cubed-sphere divergence operator with centered flux-form
    transport.  This is **not** an upwind/monotone scheme — physical
    bounds are enforced by post-transport clamping.

    Accepts both single-category 2D inputs ``(6, n, n)`` and multi-category
    3D inputs ``(6, n, n, n_cat)``.  For multi-category, the divergence is
    computed in a single batched call (one ``pad_halo_vector_4d`` MPI
    exchange across all categories and all three quantities), avoiding the
    O(n_cat × 3) halo cost of the previous vmap-over-categories pattern.

    Parameters
    ----------
    h_ice, concentration, T_ice : arrays (6, n, n) or (6, n, n, n_cat)
        Ice tracer fields.
    u_ice, v_ice : arrays (6, n, n)
        Ice velocity [m/s] — same wind across all categories.
    grid : CubedSphereGrid
    dt : float
        Timestep [s].
    T_ice_min, T_freeze_ocean : float
        Physical temperature bounds [K].

    Returns
    -------
    h_new, conc_new, T_new : arrays
        Updated tracer fields with the same shape as the inputs.
    """
    eps = 1e-20

    # Volume = h * a (conserved extensive quantity)
    vol = h_ice * concentration
    # Enthalpy = T * h * a (conserved for temperature advection)
    enthalpy = T_ice * vol

    # Stack [vol, conc, enth] along a trailing axis so a single
    # ``divergence_3d`` (one ``pad_halo_vector_4d`` MPI exchange) handles
    # all three quantities at once.  For multi-category 3D inputs the
    # category axis is folded in too, giving one halo exchange instead of
    # ``n_cat × 3`` under the prior vmap-over-categories + per-quantity
    # divergence pattern.
    is_multicat = h_ice.ndim == 4
    if is_multicat:
        # (6, n, n, n_cat) → stack to (6, n, n, n_cat, 3) → flatten the
        # last two axes so the operator sees a single trailing axis.
        n_cat = h_ice.shape[-1]
        stacked = jnp.stack([vol, concentration, enthalpy], axis=-1)
        # (6, n, n, n_cat, 3)
        stacked_flat = stacked.reshape(*stacked.shape[:3], n_cat * 3)
    else:
        # 2D inputs: stack to (6, n, n, 3).
        stacked_flat = jnp.stack([vol, concentration, enthalpy], axis=-1)

    # Flux components: u_ice and v_ice are (6, n, n) and broadcast across
    # the trailing axis via ``[..., None]``.
    u_flux = u_ice[..., None] * stacked_flat
    v_flux = v_ice[..., None] * stacked_flat
    div_flat = divergence_3d(u_flux, v_flux, grid)

    if is_multicat:
        div_stack = div_flat.reshape(*stacked.shape)  # (6, n, n, n_cat, 3)
        div_vol = div_stack[..., 0]
        div_conc = div_stack[..., 1]
        div_enth = div_stack[..., 2]
    else:
        div_vol = div_flat[..., 0]
        div_conc = div_flat[..., 1]
        div_enth = div_flat[..., 2]

    # Forward Euler update
    vol_new = jnp.maximum(vol - dt * div_vol, 0.0)
    conc_new = jnp.clip(concentration - dt * div_conc, 0.0, 1.0)
    enth_new = enthalpy - dt * div_enth

    # Recover thickness and temperature
    conc_safe = jnp.maximum(conc_new, eps)
    h_new = jnp.where(conc_new > 0.0, vol_new / conc_safe, 0.0)
    vol_safe = jnp.maximum(vol_new, eps)
    T_new = jnp.where(
        vol_new > 0.0,
        enth_new / vol_safe,
        T_freeze_ocean,
    )

    # Clamp temperature to physical bounds (centered scheme is not monotone)
    T_new = jnp.clip(T_new, T_ice_min, T_freeze_ocean)

    return h_new, conc_new, T_new
