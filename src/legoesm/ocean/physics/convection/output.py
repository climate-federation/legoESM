"""Output container for ocean convection."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class OceanConvectionOutput(NamedTuple):
    """Ocean convection tendencies and diagnostics.

    Tendencies have shape (6, n, n, nlev).
    convection_flag has shape (6, n, n, nlev-1) at interfaces.
    """
    dT_dt: jnp.ndarray
    dS_dt: jnp.ndarray
    convection_flag: jnp.ndarray   # 1 where convection is active
