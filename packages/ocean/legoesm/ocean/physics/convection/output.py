"""Output container for ocean convection."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class OceanConvectionOutput(NamedTuple):
    """Ocean convection tendencies and diagnostics.

    Tendencies have shape (..., nlev).
    convection_flag has shape (..., nlev-1) at interfaces.
    K_v / A_v have shape (..., nlev-1) at interfaces (None if not produced
    by the scheme — e.g., plume).

    ``du_dt`` / ``dv_dt`` are the explicit momentum tendencies from the
    convective viscosity ``A_v`` (None for schemes that do not mix
    momentum, e.g., plume).  ``K_v`` is the tracer diffusivity and
    ``A_v`` the momentum viscosity at interfaces — kept separate so the
    Oceananigans-style convective adjustment can carry independent
    ``convective_κz`` / ``convective_νz``.
    """
    dT_dt: jnp.ndarray
    dS_dt: jnp.ndarray
    convection_flag: jnp.ndarray   # 1 where convection is active
    K_v: jnp.ndarray | None = None  # Tracer diffusivity at interfaces
    du_dt: jnp.ndarray | None = None  # Momentum tendency (u) from A_v
    dv_dt: jnp.ndarray | None = None  # Momentum tendency (v) from A_v
    A_v: jnp.ndarray | None = None  # Momentum viscosity at interfaces
