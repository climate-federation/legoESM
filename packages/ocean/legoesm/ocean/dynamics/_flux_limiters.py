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

# Van Leer / Sweby are the canonical core kernels (redundancy audit #519) —
# import instead of re-deriving phi(r); re-export so existing ocean callers
# (`from ..._flux_limiters import sweby_limiter`) keep working.
from legoesm.core.flux_limiters import sweby_limiter, van_leer_limiter

__all__ = ["van_leer_limiter", "sweby_limiter", "resolve_tvd_limiter"]


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
