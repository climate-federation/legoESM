"""Slope / flux limiters for TVD advection schemes.

This module is the single source of truth for the differentiable flux
limiters used across the atmosphere and ocean components. Until iter-179
the Van Leer limiter was duplicated at:

* ``src/legoesm/ocean/vertical.py:_van_leer_limiter_vert``
* ``src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:_van_leer_limiter``

iter-179 added a third atmospheric caller for the plane CRM's TVD
horizontal advection; that would have been a third copy. Promote here.

All limiters are pure JAX functions of a single ratio scalar (or
elementwise-broadcastable array). All are symmetric on ``r=0`` (reduce
to first-order upwind at extrema) and bounded by 2 (TVD).
"""
from __future__ import annotations

import jax.numpy as jnp


def van_leer_limiter(r: jnp.ndarray) -> jnp.ndarray:
    """Van Leer flux limiter ``phi(r) = (r + |r|) / (1 + |r|)``.

    Properties:
    * ``r <= 0`` (extremum) → ``phi = 0`` (first-order upwind).
    * ``r = 1`` (smooth) → ``phi = 1`` (Lax-Wendroff / 2nd-order).
    * ``r → ∞`` (sharp) → ``phi → 2`` (TVD bound).
    * Differentiable everywhere (analytic, no kinks for r != 0).
    """
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))
