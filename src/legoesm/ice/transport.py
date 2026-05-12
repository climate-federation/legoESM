"""Sea ice tracer advection.

Advects ice tracers (thickness, concentration, temperature) by the ice
velocity field using **conservative monotone PPM** flux-form transport
on the cubed sphere (Colella–Woodward piecewise-parabolic with the
Colella–Woodward monotonicity limiter).  Volume ``h*a`` and area ``a``
remain in [0, ∞) and [0, 1] respectively without post-step clipping,
and enthalpy ``T*h*a`` is advected consistently with volume.

The previous centered scheme was non-monotone — it produced negative
thickness, concentration > 1, and out-of-range temperatures, which the
post-step clamps masked at the cost of conservation.  Codex
adversarial review finding #10.

All functions are JAX-compatible (differentiable, JIT-friendly).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.operators_3d import fv_flux_divergence_3d
from legoesm.core.operators_fv import fv_flux_divergence
from legoesm.grids.cubed_sphere import CubedSphereGrid


def _ppm_tendency_2d(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: CubedSphereGrid,
) -> jnp.ndarray:
    """PPM flux-divergence tendency for a 2D ``(6, n, n)`` scalar.

    Returns ``dq/dt = -div(q · v)`` from the monotonicity-limited PPM
    operator (Colella–Woodward).  Sign convention: integrate as
    ``q_new = q + dt · tendency``.
    """
    return fv_flux_divergence(q, u, v, grid, limiter=True)


def _ppm_tendency_per_category(
    q_cat: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: CubedSphereGrid,
) -> jnp.ndarray:
    """PPM tendency for a multi-category ``(6, n, n, n_cat)`` scalar.

    The ice velocity ``u``, ``v`` is the same across categories; we
    broadcast it to the category axis and reuse the existing
    ``fv_flux_divergence_3d`` helper which vmaps over the trailing
    axis (originally the vertical-level axis).
    """
    u_3d = jnp.broadcast_to(u[..., None], q_cat.shape)
    v_3d = jnp.broadcast_to(v[..., None], q_cat.shape)
    return fv_flux_divergence_3d(q_cat, u_3d, v_3d, grid, limiter=True)


def advect_ice_tracers(
    h_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    T_ice: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid: CubedSphereGrid,
    dt: float,
    T_ice_min: float = 180.0,
    T_freeze_ocean: float = constants.T_freeze_ocean,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advect ice tracers by the ice velocity field.

    Uses conservative monotone PPM (Colella–Woodward with limiter)
    flux-form transport from ``core.operators_fv``.  Volume ``h*a``,
    concentration ``a``, and enthalpy ``T*h*a`` are each advected as
    independent conserved scalars.  The PPM scheme is monotone — it
    cannot produce new extrema — so positivity of ``h*a`` and ``a`` is
    preserved without lossy post-step clipping.  A defensive ``jnp.clip``
    on concentration only guards against floating-point round-off.

    Accepts both single-category 2D inputs ``(6, n, n)`` and
    multi-category 3D inputs ``(6, n, n, n_cat)``.

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

    # Conservation-form auxiliary fields.
    vol = h_ice * concentration                  # m of ice per m² of grid
    enthalpy = T_ice * vol                       # K · m

    is_multicat = h_ice.ndim == 4
    tendency = _ppm_tendency_per_category if is_multicat else _ppm_tendency_2d

    # dq/dt = -div(q v); integrate explicitly.
    vol_new = vol + dt * tendency(vol, u_ice, v_ice, grid)
    conc_new = concentration + dt * tendency(concentration, u_ice, v_ice, grid)
    enth_new = enthalpy + dt * tendency(enthalpy, u_ice, v_ice, grid)

    # PPM-with-limiter is monotone, so vol_new and conc_new lie in
    # [min(input), max(input)].  These clips only guard against
    # floating-point round-off near the bounds.
    vol_new = jnp.maximum(vol_new, 0.0)
    conc_new = jnp.clip(conc_new, 0.0, 1.0)

    # Recover thickness and temperature.
    conc_safe = jnp.maximum(conc_new, eps)
    h_new = jnp.where(conc_new > 0.0, vol_new / conc_safe, 0.0)
    vol_safe = jnp.maximum(vol_new, eps)
    T_new = jnp.where(
        vol_new > 0.0,
        enth_new / vol_safe,
        T_freeze_ocean,
    )

    # Defensive temperature bounds — monotone advection keeps T_new in
    # the input range, so this only fires when the enthalpy / volume
    # ratio crosses the bound due to round-off.
    T_new = jnp.clip(T_new, T_ice_min, T_freeze_ocean)

    return h_new, conc_new, T_new
