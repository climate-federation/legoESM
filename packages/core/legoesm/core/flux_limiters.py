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

import jax
import jax.numpy as jnp
import numpy as np


def ratio_grad_floor(dtype) -> float:
    """Denominator floor below which a guarded ratio's GRADIENT is cut.

    ``cbrt(finfo(dtype).tiny)``: for any denominator above this, the
    ``1/den**2`` factor in the division derivative stays finite in
    ``dtype`` with O(1e13) headroom for the numerator (float32: floor
    2.3e-13, 1/den**2 <= 1.9e25 vs max 3.4e38; float64: floor 2.8e-103).

    Host-side numpy on purpose: callers evaluate this inside traced
    functions and need a static Python float (omnistaging would turn a
    ``jnp`` expression into a tracer).
    """
    return float(np.cbrt(np.finfo(np.dtype(dtype)).tiny))


def grad_safe_ratio(num: jnp.ndarray, den: jnp.ndarray,
                    ok: jnp.ndarray) -> jnp.ndarray:
    """``num / den`` with the derivative cut to zero where ``ok`` is False.

    PRIMAL-BIT-IDENTICAL to ``num / den`` (each ``where`` selects the same
    value either way); only the tangent/cotangent paths through BOTH
    operands are blocked at gated cells, and the blocking is a ``select``
    (never a multiply), so an inf/NaN produced by the division's
    derivative at a gated cell cannot leak.

    Why this exists: limiter slope ratios and thickness divisions guard
    their denominators with absolute floors (``where(|delta| > eps,
    delta, eps)``, ``maximum(h, eps)``, eps ~ 1e-30..1e-20). That keeps
    the PRIMAL finite, but the derivative of the division forms
    ``1/den**2`` — in float32 compute (the finite-volume default)
    ``den**2`` underflows to zero once ``den`` < ~1.1e-19 and the
    backward/forward pass produces ``inf * 0 = NaN``. Trigger in
    practice: near-uniform tracer fields (uniform initial salinity,
    mixed layers) and zero-thickness land/sub-seafloor cells — i.e.
    ordinary ocean states. See the 2026-06-11 ocean differentiability
    audit (FCT/Zalesak all-NaN reverse gradients; tvd/superbee/dst3 NaN
    forward-mode tangents) and
    ``tests/ocean/unit/test_advection_grad_underflow.py``.

    Cutting the gradient at gated cells is EXACT, not an approximation:
    with ``ok = den_raw > ratio_grad_floor(dtype)`` the gated cells sit
    in the floored/saturated regime of the guard, where the clamped
    primal's true one-sided derivative w.r.t. the raw inputs is zero.

    Implementation note: the DIFFERENTIATED division must never see a
    sub-floor denominator at all — not even with zero tangents. The
    div JVP/VJP is mathematically ``(t_num - r*t_den)/den`` (finite, and
    exactly what eager mode computes), but XLA fusion may legally
    re-associate it into forms containing ``den**2``, which underflows
    and turns the zero-tangent cells into ``0/0 = NaN`` under jit (f32;
    observed on the 4-step tvd rollout). Hence: gated cells divide by 1
    in the differentiated branch, and the exact primal ``num/den`` is
    restored through a stop_gradient ``where`` branch (a select, so the
    swap is primal-bit-identical and NaN cannot leak between branches).
    """
    sg = jax.lax.stop_gradient
    one = jnp.ones((), dtype=jnp.result_type(den))
    diff_branch = jnp.where(ok, num, sg(num)) / jnp.where(ok, den, one)
    return jnp.where(ok, diff_branch, sg(num / den))


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
    ``r=0`` flow cleanly through ``jax.grad`` — with the ratio derivative
    gated via :func:`grad_safe_ratio` (in float32 the ungated division
    derivative NaNs on near-uniform fields; see that docstring).
    """
    t_grad = ratio_grad_floor(jnp.result_type(f_i))
    delta_pos = f_ip1 - f_i
    r_pos = grad_safe_ratio(
        f_i - f_im1,
        jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps),
        jnp.abs(delta_pos) > t_grad,
    )
    phi_pos = f_i + 0.5 * van_leer_limiter(r_pos) * delta_pos
    delta_neg = f_i - f_ip1
    r_neg = grad_safe_ratio(
        f_ip1 - f_ip2,
        jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps),
        jnp.abs(delta_neg) > t_grad,
    )
    phi_neg = f_ip1 + 0.5 * van_leer_limiter(r_neg) * delta_neg
    return phi_pos, phi_neg
