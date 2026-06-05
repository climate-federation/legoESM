"""Shared TVD flux limiters for ocean tracer advection.

Centralizes flux-limiter functions that were previously duplicated across
``ocean_pe_latlon_cgrid.py``, ``vertical.py``, ``advection_mpas.py``, and
``advection.py``.

All limiters take a smoothness ratio ``r = (donor - upwind-of-upwind) /
(downstream - donor)`` and return ``psi(r) in [0, 2]`` such that the
limited face value is ``f_face = f_donor + 0.5 * psi(r) * (f_downstream
- f_donor)``.
"""

from __future__ import annotations

import jax.numpy as jnp


def van_leer_limiter(r: jnp.ndarray) -> jnp.ndarray:
    """Van Leer flux limiter: ``psi(r) = (r + |r|) / (1 + |r|)``.

    Smooth, second-order, TVD. Bounded by [0, 2). Differentiable
    everywhere. Less aggressive than Sweby/superbee.
    """
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


def sweby_limiter(r: jnp.ndarray) -> jnp.ndarray:
    """Sweby (superbee) flux limiter.

    ``psi(r) = max(0, min(1, 2r), min(2, r))``.

    Traces the upper boundary of the Sweby TVD region: maximum
    anti-diffusion subject to monotonicity. Standard choice in Veros
    (``enable_superbee_advection=True``) and MOM6 (``superbee``).
    """
    return jnp.maximum(
        0.0,
        jnp.maximum(jnp.minimum(1.0, 2.0 * r), jnp.minimum(2.0, r)),
    )


def resolve_tvd_limiter(name: str):
    """Map a ``tracer_advection`` literal to the limiter function.

    Raises ``ValueError`` on unknown names so dispatch sites that route
    through TVD with a configurable limiter cannot silently fall back.
    """
    if name == "tvd":
        return van_leer_limiter
    if name == "superbee":
        return sweby_limiter
    raise ValueError(
        f"resolve_tvd_limiter: unknown TVD literal {name!r}; "
        f"expected 'tvd' (Van Leer) or 'superbee' (Sweby)."
    )
