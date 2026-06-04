"""Output container for bottom drag."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class BottomDragOutput(NamedTuple):
    """Bottom drag tendencies, shape (6, n, n, nlev).

    Nonzero only at the bottom level.
    """
    du_dt: jnp.ndarray
    dv_dt: jnp.ndarray


def bottom_level_drag_output(
    drag_u_bottom: jnp.ndarray,
    drag_v_bottom: jnp.ndarray,
    nlev: int,
) -> "BottomDragOutput":
    """Place per-column bottom-level drag tendencies at the deepest level with
    zeros above, returning a full ``(..., nlev)`` BottomDragOutput.

    Shared by the linear and quadratic drag schemes (which compute the bottom
    row differently but pad identically). Single Pad HLO op rather than
    alloc + dynamic_update_slice.

    Parameters
    ----------
    drag_u_bottom, drag_v_bottom : array (...)
        Bottom-level drag tendency (one value per column).
    nlev : int
        Number of vertical levels.
    """
    pad_axes = ((0, 0),) * drag_u_bottom.ndim
    du_dt = jnp.pad(drag_u_bottom[..., None], (*pad_axes, (nlev - 1, 0)))
    dv_dt = jnp.pad(drag_v_bottom[..., None], (*pad_axes, (nlev - 1, 0)))
    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)
