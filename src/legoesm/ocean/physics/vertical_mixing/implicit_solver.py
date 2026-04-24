"""Backward-Euler implicit vertical diffusion for the ocean.

Solves the 1-D diffusion equation per column

    ∂φ/∂t = ∂/∂z [K · ∂φ/∂z]

for each grid column using a backward-Euler discretisation that is
unconditionally stable.  This is the key ingredient for issue #204:
without it, the ocean dycore must keep first-order-upwind vertical
momentum advection (whose numerical viscosity ~|w|·dz/2 silently damps
baroclinic shear); with it, the physical ``A_v`` / ``K_v`` and any
Richardson-number or KPP-based enhancement can do that job directly
and the resolved advection can be upgraded to higher-order schemes.

Discrete form (no-flux top + bottom):

    ρ · ∂φ/∂t = ∂/∂z ( K ∂φ/∂z )       →     backward-Euler
    (1 + α_k + β_k) φ^{n+1}_k
        − α_k φ^{n+1}_{k-1}
        − β_k φ^{n+1}_{k+1}
      = φ^{n}_k
with
    α_k = dt · K_{k-1/2} / ( dz_k · dz_half_{k-1/2} )
    β_k = dt · K_{k+1/2} / ( dz_k · dz_half_{k+1/2} )

``K`` lives on interfaces (``nlev-1`` values per column), ``dz`` on
layer centres (``nlev``), ``dz_half`` on interfaces (``nlev-1``).
No-flux boundaries are enforced by setting ``K_{-1/2} = K_{N-1/2} = 0``
(implicit via α_0 = 0 and β_{N-1} = 0).

References
----------
- Thomas, L.H. (1949) — the Thomas algorithm is already implemented in
  :mod:`legoesm.timestepping.tridiagonal`; this module just builds the
  per-column system.
- MOM6 Technical Manual §7 (implicit vertical viscosity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.timestepping.tridiagonal import thomas_solve

_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7


def implicit_vertical_diffusion_ocean(
    field: jax.Array,
    K: jax.Array | float,
    dz: jax.Array,
    dz_half: jax.Array,
    dt: float,
) -> jax.Array:
    """Backward-Euler implicit vertical diffusion for a column field.

    Applies one implicit step of

        (1 - dt · ∂_z K ∂_z) φ^{n+1} = φ^{n}

    per column with *zero-flux* boundary conditions at the top and the
    bottom (the ocean's natural choice for momentum and tracers in the
    absence of a prescribed surface flux).  Non-zero surface / bottom
    fluxes can be applied *externally* before or after this call — the
    helper is deliberately boundary-condition-simple so callers don't
    have to thread flux arrays through when they don't need them.

    Parameters
    ----------
    field : jax.Array, shape ``(..., nlev)``
        Field to diffuse.  The vertical axis is the last axis; any
        number of leading horizontal axes is allowed.  Examples:
        ``(n_lat, n_lon, nlev)`` for lat-lon, ``(nCells, nlev)`` for
        MPAS, ``(n_lat, n_lon+1, nlev)`` for u on a C-grid.
    K : jax.Array or float, shape ``(..., nlev-1)``
        Vertical viscosity / diffusivity at interior interfaces.
        Must be ≥ 0.  A scalar is broadcast to every interface and
        every column.
    dz : jax.Array, shape ``(..., nlev)`` or ``(nlev,)``
        Layer thickness at full levels.  Must match ``field`` along
        the last axis (a 1-D ``dz`` broadcasts across all columns).
    dz_half : jax.Array, shape ``(..., nlev-1)`` or ``(nlev-1,)``
        Distance between adjacent full-level centres
        (``dz_half_k = 0.5 (dz_k + dz_{k+1})`` is the standard choice).
    dt : float
        Time step [s].  Must be positive.

    Returns
    -------
    jax.Array
        Updated field with the same shape as ``field``.
    """
    # Only enforce the positivity check when ``dt`` is a concrete Python
    # scalar — under ``jax.jit`` it may be a traced argument, and a
    # Python-level ``if`` would raise ``TracerBoolConversionError``.
    if not isinstance(dt, jax.core.Tracer):
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")

    nlev = field.shape[-1]
    if nlev < 2:
        # One-level columns have no vertical gradient ⇒ no-op.
        return field

    # --- Promote K, dz, dz_half to match the field's leading shape ---
    K_arr = jnp.asarray(K)
    if K_arr.ndim == 0:
        K_arr = jnp.broadcast_to(K_arr, field.shape[:-1] + (nlev - 1,))
    elif K_arr.shape[-1] != nlev - 1:
        raise ValueError(
            f"K last dim {K_arr.shape[-1]} must equal nlev-1 = {nlev - 1}")

    dz_arr = jnp.asarray(dz)
    if dz_arr.ndim == 1:
        dz_arr = jnp.broadcast_to(dz_arr, field.shape)
    elif dz_arr.shape[-1] != nlev:
        raise ValueError(
            f"dz last dim {dz_arr.shape[-1]} must equal nlev = {nlev}")

    dzh_arr = jnp.asarray(dz_half)
    if dzh_arr.ndim == 1:
        dzh_arr = jnp.broadcast_to(dzh_arr, field.shape[:-1] + (nlev - 1,))
    elif dzh_arr.shape[-1] != nlev - 1:
        raise ValueError(
            f"dz_half last dim {dzh_arr.shape[-1]} must equal nlev-1 = "
            f"{nlev - 1}")

    # --- Build α and β at every cell (last axis = level) ---
    # α_k uses the (k-1/2) interface, β_k the (k+1/2) interface.
    # We pad K with an extra zero on each side so indexing is uniform;
    # the zeros naturally encode the no-flux BCs.
    K_safe = jnp.maximum(K_arr, 0.0)
    dzh_safe = jnp.maximum(dzh_arr, _EPS)

    # K / dz_half at interfaces (nlev-1)
    flux_coeff = K_safe / dzh_safe                   # (..., nlev-1)

    # Pad top and bottom with zero (no-flux):
    zero_face = jnp.zeros(flux_coeff.shape[:-1] + (1,),
                          dtype=flux_coeff.dtype)
    flux_top = jnp.concatenate([zero_face, flux_coeff], axis=-1)  # (..., nlev)
    flux_bot = jnp.concatenate([flux_coeff, zero_face], axis=-1)  # (..., nlev)

    inv_dz = 1.0 / jnp.maximum(dz_arr, _EPS)          # (..., nlev)
    alpha = dt * flux_top * inv_dz                    # (..., nlev)
    beta = dt * flux_bot * inv_dz                     # (..., nlev)

    # Tridiagonal coefficients:
    #   a_k = -α_k   (sub-diagonal, a_0 = 0)
    #   b_k = 1 + α_k + β_k
    #   c_k = -β_k   (super-diagonal, c_{N-1} = 0)
    #   d_k = φ^n_k
    a = -alpha
    b = 1.0 + alpha + beta
    c = -beta
    d = field

    return thomas_solve(a, b, c, d)


# ---------------------------------------------------------------------------
# Convenience: build dz_half from dz with the standard midpoint rule.
# ---------------------------------------------------------------------------


def build_dz_half(dz: jax.Array) -> jax.Array:
    """Midpoint distance between adjacent full-level centres.

    ``dz_half_k = 0.5 · (dz_k + dz_{k+1})`` with shape ``(..., nlev-1)``.
    Provided as a helper because most callers don't carry ``dz_half``
    separately from ``dz_ref`` but do need it to build the tridiagonal
    system.
    """
    return 0.5 * (dz[..., :-1] + dz[..., 1:])
