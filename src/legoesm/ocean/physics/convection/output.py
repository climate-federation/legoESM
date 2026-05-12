"""Output container for ocean convection."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class OceanConvectionOutput(NamedTuple):
    """Ocean convection tendencies and diagnostics.

    Tendencies have shape (..., nlev).
    convection_flag has shape (..., nlev-1) at interfaces.
    K_v has shape (..., nlev-1) at interfaces (None if not produced by
    the scheme — e.g., plume).
    """
    dT_dt: jnp.ndarray
    dS_dt: jnp.ndarray
    convection_flag: jnp.ndarray   # 1 where convection is active
    K_v: jnp.ndarray | None = None  # Effective diffusivity at interfaces
