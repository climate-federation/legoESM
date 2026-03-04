"""Output container for lateral mixing."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class LateralMixingOutput(NamedTuple):
    """Lateral mixing tendencies, shape (6, n, n, nlev)."""
    du_dt: jnp.ndarray
    dv_dt: jnp.ndarray
    dT_dt: jnp.ndarray
    dS_dt: jnp.ndarray
