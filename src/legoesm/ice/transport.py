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

from legoesm.core.field import Field
from legoesm.core.operators import divergence
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

    Parameters
    ----------
    h_ice : array (6, n, n)
        Ice thickness [m].
    concentration : array (6, n, n)
        Areal fraction [0-1].
    T_ice : array (6, n, n)
        Ice surface temperature [K].
    u_ice, v_ice : arrays (6, n, n)
        Ice velocity [m/s].
    grid : CubedSphereGrid
    dt : float
        Timestep [s].
    T_ice_min : float
        Lower temperature bound [K] (default 180).
    T_freeze_ocean : float
        Upper temperature bound [K] (default 271.35).

    Returns
    -------
    h_new : array (6, n, n)
        Updated thickness (clamped >= 0).
    conc_new : array (6, n, n)
        Updated concentration (clamped to [0, 1]).
    T_new : array (6, n, n)
        Updated temperature (clamped to [T_ice_min, T_freeze_ocean]).
    """
    eps = 1e-20

    # Volume = h * a (conserved extensive quantity)
    vol = h_ice * concentration
    # Enthalpy = T * h * a (conserved for temperature advection)
    enthalpy = T_ice * vol

    # Flux-form transport: dq/dt = -div(q * u)
    # Using the existing cubed-sphere divergence operator
    u_f = Field(data=u_ice * vol, name="flux_u", dims=("face", "x", "y"), units="m^2/s")
    v_f = Field(data=v_ice * vol, name="flux_v", dims=("face", "x", "y"), units="m^2/s")
    div_vol = divergence(u_f, v_f, grid).data

    u_a = Field(data=u_ice * concentration, name="flux_u", dims=("face", "x", "y"), units="1/s")
    v_a = Field(data=v_ice * concentration, name="flux_v", dims=("face", "x", "y"), units="1/s")
    div_conc = divergence(u_a, v_a, grid).data

    u_e = Field(data=u_ice * enthalpy, name="flux_u", dims=("face", "x", "y"), units="K*m^2/s")
    v_e = Field(data=v_ice * enthalpy, name="flux_v", dims=("face", "x", "y"), units="K*m^2/s")
    div_enth = divergence(u_e, v_e, grid).data

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
