"""SCM↔LES coupling primitives: vertical regrid + θ↔T (the tuner's grid bridge).

``tune_scm_to_les.py`` runs the dycore-free SCM against cached LES truth. The SCM
lives on a sigma-pressure grid in **temperature**, top-to-bottom; the LES artifact
lives on a **height** grid in **potential temperature**, surface-first. This module
is the coupling layer the bridge deferred (``bridge.artifact_to_scm_forcing`` only
reorients, it does not regrid): it (a) linearly regrids a profile between height
grids and (b) exposes the θ↔T conversions (reused from
``atmosphere.physics.thermodynamics`` — NOT re-derived).

The controlled-comparison rule (CLAUDE.md) is served by regridding the LES truth
onto ONE fixed evaluation grid held byte-identical across every closure and regime;
:func:`regrid_truth` produces that fixed-grid truth once, and every SCM arm is
scored against it.

Conventions: heights strictly increasing (surface-first) — the artifact/LES-native
ordering. Out-of-range targets clamp to the nearest source endpoint (no
extrapolation of turbulence beyond the resolved column).
"""
from __future__ import annotations

import jax.numpy as jnp
from legoesm.atmosphere.physics.thermodynamics import (
    potential_temperature_from_temperature,
    temperature_from_theta,
)

from .bridge import LESTruth

Array = jnp.ndarray

# Re-export the θ↔T pair so the tuner has a single coupling import surface. These
# are the canonical thermodynamics implementations (θ = T·(p0/p)^κ and its inverse);
# do not re-derive the Exner factor here.
theta_from_temperature = potential_temperature_from_temperature
T_from_theta = temperature_from_theta


def interp_profile(values: Array, z_src: Array, z_dst: Array) -> Array:
    """Linearly regrid ``values(z_src)`` onto ``z_dst`` (both surface-first).

    ``values`` is ``(nz_src,)`` or ``(nt, nz_src)`` (regrids the last axis);
    ``z_src`` is ``(nz_src,)`` strictly increasing. Targets outside ``z_src`` clamp
    to the nearest endpoint value (``jnp.interp`` semantics) — no extrapolation.
    """
    values = jnp.asarray(values)
    z_src = jnp.asarray(z_src, dtype=values.dtype)
    z_dst = jnp.asarray(z_dst, dtype=values.dtype)
    if z_src.ndim != 1 or z_dst.ndim != 1:
        raise ValueError("z_src and z_dst must be 1-D")
    if values.shape[-1] != z_src.shape[0]:
        raise ValueError(
            f"values last axis {values.shape[-1]} != len(z_src) {z_src.shape[0]}"
        )
    if z_src.shape[0] >= 2 and not bool(jnp.all(jnp.diff(z_src) > 0)):
        raise ValueError("z_src must be strictly increasing (surface-first)")
    if values.ndim == 1:
        return jnp.interp(z_dst, z_src, values)
    # (nt, nz): interp each row onto z_dst
    return jnp.stack([jnp.interp(z_dst, z_src, row) for row in values])


def regrid_truth(truth: LESTruth, z_dst: Array) -> LESTruth:
    """Regrid every profile of an :class:`LESTruth` onto ``z_dst`` (fixed eval grid).

    Interpolates θ, u, v, and the turbulent flux(es) — preserving the single-time
    ``(nz,)`` vs multi-time ``(nt, nz)`` shape — so a diagnostic or prognostic truth
    can be compared to an SCM on ``z_dst``. ``times_s`` and ``case_name`` pass
    through unchanged.
    """
    z_dst = jnp.asarray(z_dst)
    z_src = jnp.asarray(truth.heights_m)

    def _rg(arr):
        return None if arr is None else interp_profile(arr, z_src, z_dst)

    return LESTruth(
        case_name=truth.case_name,
        heights_m=z_dst,
        times_s=jnp.asarray(truth.times_s),
        theta=interp_profile(truth.theta, z_src, z_dst),
        u=interp_profile(truth.u, z_src, z_dst),
        v=interp_profile(truth.v, z_src, z_dst),
        wtheta=interp_profile(truth.wtheta, z_src, z_dst),
        qt=_rg(truth.qt),
        wqt=_rg(truth.wqt),
    )
