"""Gravity wave drag output container.

GWDOutput is the common interface between all GWD backends.
All schemes produce the same NamedTuple so that downstream integration
code can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.precision import get_policy


class GWDOutput(NamedTuple):
    """Output from a gravity wave drag scheme (backend-agnostic).

    Tendency fields have shape (ncol, nlev).
    Column-integrated fields have shape (ncol,).

    Fields
    ------
    du_dt : jax.Array
        Zonal wind tendency [m/s^2], shape (ncol, nlev).
    dv_dt : jax.Array
        Meridional wind tendency [m/s^2], shape (ncol, nlev).
    dT_dt : jax.Array
        Temperature tendency from wave breaking [K/s], shape (ncol, nlev).
    eps_gwd : jax.Array
        Column-integrated wave energy dissipation [W/m^2], shape (ncol,).
        eps_gwd = -sum(rho * (u*du_dt + v*dv_dt) * dz).  Positive for the
        wind-opposing diagnostic schemes (rayleigh/lindzen/mcfarlane/hines),
        where drag removes KE from the mean flow; ``prognostic_spectral``
        is positive by construction since the launch-level
        ``sign(c - U_launch)`` deposition factor landed (#856 closed
        F-GWD-1 -- the earlier deposition could accelerate the mean flow).
    dq_dt : dict[str, jax.Array] or None
        Tracer tendencies [tracer units/s], shape (ncol, nlev) each, keyed
        like the ``tracers`` the scheme was given (``mcfarlane``: CAM6
        gravity-wave eddy diffusion).  None for schemes that move no tracers.
    """
    du_dt: jax.Array
    dv_dt: jax.Array
    dT_dt: jax.Array
    eps_gwd: jax.Array
    dq_dt: dict | None = None


def make_zero_output(ncol: int, nlev: int, dtype=None) -> GWDOutput:
    """Create a zero-valued GWDOutput for the given dimensions."""
    if dtype is None:
        dtype = get_policy().storage
    z = jnp.zeros((ncol, nlev), dtype=dtype)
    return GWDOutput(
        du_dt=z,
        dv_dt=z,
        dT_dt=z,
        eps_gwd=jnp.zeros((ncol,), dtype=dtype),
    )
