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
        For the wind-opposing diagnostic schemes
        (rayleigh/lindzen/mcfarlane/hines) this is the mean-flow KE loss
        eps_gwd = -sum(rho * (u*du_dt + v*dv_dt) * dz) >= 0 (drag removes
        KE from the mean flow).  The experimental ``prognostic_spectral``
        scheme instead reports the column wave-dissipation (frictional
        heating) integral, >= 0 by construction, because with its two-sided
        prognostic spectrum (deposition sign ``sign(c - U_proj)`` fixed at
        launch, F-GWD-1) waves faster than the wind can accelerate the mean
        flow — the KE-loss form is NOT guaranteed positive there and is
        deliberately not what it reports.
    """
    du_dt: jax.Array
    dv_dt: jax.Array
    dT_dt: jax.Array
    eps_gwd: jax.Array


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
