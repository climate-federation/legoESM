"""Radiation output container.

RadiationOutput is the common interface between gray and RRTMGP backends.
Both schemes produce the same NamedTuple so that downstream integration
code (and future neural emulators) can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class RadiationOutput(NamedTuple):
    """Output from a radiation scheme (backend-agnostic).

    All fluxes are at interface (half) levels with shape (ncol, nlev+1).
    Heating rates are at full levels with shape (ncol, nlev).

    Fields
    ------
    lw_flux_up : jax.Array
        Upward LW flux [W/m^2], shape (ncol, nlev+1).
    lw_flux_down : jax.Array
        Downward LW flux [W/m^2], shape (ncol, nlev+1).
    sw_flux_up : jax.Array
        Upward SW flux [W/m^2], shape (ncol, nlev+1).
    sw_flux_down : jax.Array
        Downward SW flux [W/m^2], shape (ncol, nlev+1).
    heating_rate : jax.Array
        Total radiative heating rate [K/s], shape (ncol, nlev).
    lw_heating_rate : jax.Array
        LW radiative heating rate [K/s], shape (ncol, nlev).
    sw_heating_rate : jax.Array
        SW radiative heating rate [K/s], shape (ncol, nlev).
    toa_insolation : jax.Array or None
        Prescribed TOA incident SW the solver was given (ncol,) [W/m^2]
        (#620).  This is ``S_0·cos(SZA)`` (diurnal) or the daily-mean /
        RCE insolation, bounded [0, S_0] and 0 at night.  Used for the
        CMOR ``rsdt`` diagnostic so it reflects the true TOA insolation
        rather than the quadratically clamped top-halo SW flux.  ``None``
        (default) for paths that do not set it (e.g. zero-radiation stub).
    """
    lw_flux_up: jax.Array
    lw_flux_down: jax.Array
    sw_flux_up: jax.Array
    sw_flux_down: jax.Array
    heating_rate: jax.Array
    lw_heating_rate: jax.Array
    sw_heating_rate: jax.Array
    toa_insolation: jax.Array | None = None   # (ncol,) prescribed TOA incident SW [W/m^2] (#620)
    # (ncol,) clear-sky (clouds removed) TOA upward fluxes [W/m^2] from the
    # same solve, when ``solve_columns(clear_sky=True)``; else None.
    lw_flux_up_toa_clr: jax.Array | None = None
    sw_flux_up_toa_clr: jax.Array | None = None
