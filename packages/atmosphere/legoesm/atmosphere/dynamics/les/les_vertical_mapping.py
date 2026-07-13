"""Map a GCM column onto the LES fine grid + Newtonian top relaxation (gap #3).

Stage 5 of ``docs/COMPARE_REANALYSIS.md`` (§2 "Vertical mapping: interpolate +
relax above LES top"): interpolate the flagged GCM column's profiles onto the
finer LES height grid (the LES initial/reference state), and build the
Newtonian-relaxation rate that nudges the LES back toward the GCM profile near
the LES top — keeping the diagnosis region below the relaxation layer (so the
sponge cannot contaminate the diagnosed closure coefficient, §7).

Reuse (CLAUDE.md — no re-derived numerics):

* Vertical interpolation via ``jnp.interp`` (the codebase's z-interpolation
  primitive, e.g. ``sam_case_setup``); the relaxation *application* and the
  large-scale subsidence/advection forcing reuse
  :func:`legoesm.atmosphere.dynamics.plane_large_scale_forcing.make_plane_ls_forcing_physics`.
* The relaxation **rate** profile reuses the canonical Rayleigh-damping shape
  :func:`legoesm.atmosphere.dynamics.gcm.compressible_euler.sponge_profile`
  (``"sin2"`` / ``"sam_rational"``; raises on unknown shape).

Convention (**hard precondition**): all height arrays are **ascending in z**
(metres).  The plane dycore stores fields top-down, so the LES driver reverses
them before calling here, though the interpolation also sorts internally so it
is order-agnostic.  The per-step kernels
(:func:`interpolate_column_to_les`, :func:`top_relaxation_rate`,
:func:`relaxation_tendency`) are pure-JAX, AD-safe and jit/vmap-friendly;
:func:`build_top_relaxation` is a host-side *setup* helper (concrete column
heights) that assembles the ``(target, rate)`` those kernels consume.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.compressible_euler import sponge_profile


def interpolate_column_to_les(
    src_z: jax.Array, src_field: jax.Array, les_z: jax.Array
) -> jax.Array:
    """Linearly interpolate a GCM column profile onto the LES height grid.

    ``src_z`` / ``src_field`` are the GCM column heights + values; ``les_z`` the
    LES full-level heights.  The source is sorted ascending in ``z`` internally
    (so a top-down dycore profile interpolates correctly without a separate
    reverse step — robust to either order).  ``jnp.interp`` **clamps** to the
    column's endpoint values outside its range, so LES levels above the GCM
    column top take the GCM top value (the relaxation layer built by
    :func:`build_top_relaxation` then nudges toward exactly that).
    """
    src_z = jnp.asarray(src_z)
    src_field = jnp.asarray(src_field, dtype=src_z.dtype)
    les_z = jnp.asarray(les_z, dtype=src_z.dtype)
    # Sort ascending in z so jnp.interp (which needs increasing xp) is correct
    # regardless of the caller's storage order (the dycore stores top-down).
    order = jnp.argsort(src_z)
    return jnp.interp(les_z, src_z[order], src_field[order])


def top_relaxation_rate(
    les_z: jax.Array,
    domain_top_m: float,
    relax_width_m: float,
    inv_tau: float,
    shape: str = "sin2",
) -> jax.Array:
    """Newtonian-relaxation rate ``[1/s]`` ramping from 0 to ~``inv_tau`` at the top.

    Zero below ``domain_top − relax_width``, ramping toward ``inv_tau`` (``= 1/τ``)
    at the model top via the canonical :func:`sponge_profile` shape.  ``"sin2"``
    reaches exactly ``inv_tau`` at the top; ``"sam_rational"`` (SAM-fast) reaches
    its canonical ``100/101·inv_tau`` near-top value — ``inv_tau`` is the peak
    coefficient handed to the shared shape, not rescaled (keeping it byte-faithful
    to the dycore sponge).  Unknown shape raises.  Keep the LES flux-diagnosis
    region BELOW ``domain_top − relax_width`` so the relaxation does not bias the
    diagnosed coefficient.
    """
    return sponge_profile(
        jnp.asarray(les_z), domain_top_m, relax_width_m, inv_tau, shape=shape
    )


def relaxation_tendency(
    field: jax.Array, target: jax.Array, rate: jax.Array
) -> jax.Array:
    """Newtonian relaxation tendency ``−rate·(field − target)`` [field-units/s].

    Drives ``field`` toward ``target`` at the height-dependent ``rate`` (from
    :func:`top_relaxation_rate`); zero where ``rate = 0`` (below the relaxation
    layer) or ``field = target``.
    """
    dtype = jnp.result_type(field, target, rate, jnp.float32)
    field = jnp.asarray(field, dtype=dtype)
    target = jnp.asarray(target, dtype=dtype)
    rate = jnp.asarray(rate, dtype=dtype)
    return -rate * (field - target)


def build_top_relaxation(
    les_z: jax.Array,
    domain_top_m: float,
    gcm_z: jax.Array,
    gcm_field: jax.Array,
    *,
    relax_width_m: float,
    inv_tau: float,
    shape: str = "sin2",
) -> tuple[jax.Array, jax.Array]:
    """Assemble the ``(target, rate)`` for relaxing an LES field to the GCM column.

    ``target`` is the GCM ``gcm_field`` interpolated onto ``les_z``; ``rate`` is
    the top-relaxation profile.  Apply as
    ``relaxation_tendency(les_field, target, rate)`` each step.  Returns both as
    ``(nlev,)`` on the LES grid.

    This is a **host-side setup helper** — called once per column with concrete
    grid heights (``gcm_z`` is a concrete array, not a JIT tracer); the returned
    ``(target, rate)`` are then consumed by the jit'd time loop via the per-step
    :func:`relaxation_tendency`.  It raises if the LES domain top exceeds the GCM
    column top (the relaxation target above the column would be a flat
    extrapolation — unphysical), so the LES must sit within the column's extent.
    """
    import numpy as _np

    gcm_top = float(_np.asarray(gcm_z).max())
    if float(domain_top_m) > gcm_top:
        raise ValueError(
            f"LES domain top {domain_top_m} m exceeds the GCM column top "
            f"{gcm_top} m; the relaxation target above the column would be a "
            f"flat extrapolation. Lower the LES top or extend the GCM column."
        )
    target = interpolate_column_to_les(gcm_z, gcm_field, les_z)
    rate = top_relaxation_rate(les_z, domain_top_m, relax_width_m, inv_tau, shape)
    return target, rate
