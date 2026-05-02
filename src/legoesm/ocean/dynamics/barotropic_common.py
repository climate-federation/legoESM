"""Grid-agnostic helpers shared by barotropic_*.py substep solvers.

Factors small repeated patterns out of the four grid-specific
barotropic solvers (cubed-sphere A-grid, cubed-sphere C-grid, lat-lon
C-grid, MPAS).  Closes #214 (Phase 2).

The helpers are intentionally thin: each replaces a few lines that were
character-for-character identical across solvers, so future updates to
the BEBT scheme, the cosine time filter, or MAXVEL clipping touch one
file instead of three.

This module performs no halo exchange and does not depend on any
grid-specific operator package.  Callers are responsible for staging
field shapes correctly before invoking these helpers.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def compute_filter_weights(
    n_substeps: int,
    dtype: jnp.dtype,
    *,
    use_cosine: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-substep accumulator weights for time-averaging.

    The cosine bell (Hanning window) suppresses the side lobes of the
    plain box filter that alias barotropic modes into the baroclinic
    coupling.  Both the lat-lon C-grid and MPAS solvers compute the
    weights with the same formula:

        w_i = 1 + cos(2π · (i - n/2) / n)         (cosine)
        w_i = 1                                   (box)

    Parameters
    ----------
    n_substeps : int
        Number of barotropic substeps.
    dtype : jnp.dtype
        Working precision for the weight array.
    use_cosine : bool
        ``config.barotropic_time_filter == "cosine"``.

    Returns
    -------
    w_filter : jax.Array, shape (n_substeps,)
        Per-substep weight passed as ``xs`` to ``lax.scan`` (or
        indexed inside ``fori_loop``).
    w_total : jax.Array, scalar
        ``sum(w_filter)`` — used to normalise the eta / velocity
        accumulators.  Transport accumulators (``Hu``) keep using
        ``n_substeps`` for exact volume conservation.
    """
    i = jnp.arange(n_substeps, dtype=dtype)
    if use_cosine:
        # Hanning-window weights.  At ``n_substeps == 1`` (rare, only
        # used by tests / 1-substep spin-ups) the formula
        # ``1 + cos(2 pi (0 - 0.5)/1) = 1 + cos(-pi) = 0`` collapses to
        # zero, which then divides by zero in
        # ``eta_sum / w_total`` downstream.  Fall back to the box
        # filter when the cosine bell would degenerate (codex
        # adversarial review iter-1, bug #4).
        if n_substeps < 2:
            w_filter = jnp.ones(n_substeps, dtype=dtype)
        else:
            w_filter = 1.0 + jnp.cos(
                2.0 * jnp.pi * (i - 0.5 * n_substeps) / n_substeps,
            )
    else:
        w_filter = jnp.ones(n_substeps, dtype=dtype)
    return w_filter, jnp.sum(w_filter)


def bebt_blend(
    eta_new: jnp.ndarray,
    eta_old: jnp.ndarray,
    bebt: float | jnp.ndarray,
) -> jnp.ndarray:
    """Backward-Euler/Backward-time blend of new and old eta for the PGF.

    ``bebt = 0`` recovers the standard forward-backward scheme; the
    MOM6 default ``bebt = 0.2`` introduces semi-implicit damping of
    the fastest barotropic gravity waves (#205).
    """
    return (1.0 - bebt) * eta_new + bebt * eta_old


def maxvel_clip(field: jnp.ndarray, maxvel: float | jnp.ndarray) -> jnp.ndarray:
    """Symmetric clip of barotropic velocity components.

    Used to suppress runaway velocities at single grid points that
    would otherwise crash the solver before the substep finishes.
    """
    return jnp.clip(field, -maxvel, maxvel)
