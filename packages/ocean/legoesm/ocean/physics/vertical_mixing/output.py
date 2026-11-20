"""Output container for vertical mixing."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class VerticalMixingOutput(NamedTuple):
    """Vertical mixing tendencies and diagnostics.

    Tendencies have shape (6, n, n, nlev).
    Diagnostics K_v, A_v have shape (6, n, n, nlev-1) at interfaces.
    """
    du_dt: jnp.ndarray
    dv_dt: jnp.ndarray
    dT_dt: jnp.ndarray
    dS_dt: jnp.ndarray
    K_v: jnp.ndarray   # Diffusivity diagnostic at interfaces
    A_v: jnp.ndarray   # Viscosity diagnostic at interfaces
