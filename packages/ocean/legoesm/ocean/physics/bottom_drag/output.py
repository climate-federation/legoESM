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
