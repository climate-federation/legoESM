"""Output container for surface forcing."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class SurfaceForcingOutput(NamedTuple):
    """Surface forcing tendencies and diagnostics.

    Tendencies have shape (6, n, n, nlev).
    Diagnostics have shape (6, n, n).
    """
    du_dt: jnp.ndarray
    dv_dt: jnp.ndarray
    dT_dt: jnp.ndarray
    dS_dt: jnp.ndarray
    Q_net: jnp.ndarray    # Net heat flux diagnostic [W/m^2]
    tau_x: jnp.ndarray    # Zonal wind stress diagnostic [N/m^2]
    tau_y: jnp.ndarray    # Meridional wind stress diagnostic [N/m^2]
