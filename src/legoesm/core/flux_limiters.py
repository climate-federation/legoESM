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


def van_leer_face_values(f_im1, f_i, f_ip1, f_ip2, eps: float = 1e-30):
    """Van-Leer slope-limited reconstruction of the ``i+1/2`` face value from
    the 4-cell stencil ``[f_im1, f_i, f_ip1, f_ip2]``.

    Returns ``(phi_pos, phi_neg)`` — the upwind face values for POSITIVE face
    velocity (extrapolated from the left cell ``i``) and NEGATIVE face velocity
    (from the right cell ``i+1``). The caller selects by ``sign(u_face)`` and
    forms the flux ``u_face · phi``.

    Single source of truth for the plane CRM van-Leer reconstruction — serial +
    halo horizontal (x/y) AND the D5 vertical all route through here, so the
    smoothness-ratio sign lives in ONE place. Both ratios are
    ``(upwind slope)/(local slope)``: POSITIVE on smooth monotone data ⇒
    ``phi→1`` (2nd-order); they flip sign at an extremum ⇒ ``phi=0`` (1st-order
    upwind — the TVD fallback).

    HD-1 (codex iter-44/45): the ``r_neg`` numerator is ``f_ip1 - f_ip2`` (NOT
    ``f_ip2 - f_ip1``). The flipped sign silently forced ``phi_neg → 0``
    (1st-order, over-diffusive) for u<0; the flux divergence masks it under
    UNIFORM velocity, so the iter-179 uniform-velocity linear-field tests never
    caught it. Verified: this sign is 2nd-order on a smooth field with u<0 (≈6×
    smaller error than the flipped sign) and still TVD.

    Differentiable: the ``where``-guarded denominators + the van-Leer kink at
    ``r=0`` flow cleanly through ``jax.grad``.
    """
    delta_pos = f_ip1 - f_i
    r_pos = (f_i - f_im1) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    phi_pos = f_i + 0.5 * van_leer_limiter(r_pos) * delta_pos
    delta_neg = f_i - f_ip1
    r_neg = (f_ip1 - f_ip2) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    phi_neg = f_ip1 + 0.5 * van_leer_limiter(r_neg) * delta_neg
    return phi_pos, phi_neg
