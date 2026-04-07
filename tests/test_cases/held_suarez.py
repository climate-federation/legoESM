"""Held-Suarez forcing and initialization — backward-compatibility re-exports.

The canonical implementation now lives in ``legoesm.atmosphere.held_suarez``
so that it is available at runtime without the ``tests`` package on
``sys.path``.  This module re-exports every public name for existing tests.
"""

from legoesm.atmosphere.held_suarez import (  # noqa: F401
    # Constants
    K_A,
    K_S,
    K_F,
    SIGMA_B,
    DELTA_T_Y,
    DELTA_THETA_Z,
    T_MIN,
    P_0,
    # Shared physics
    held_suarez_equilibrium_temperature,
    # Cubed-sphere
    held_suarez_forcing,
    held_suarez_init,
    # Lat-lon
    held_suarez_forcing_latlon,
    held_suarez_init_latlon,
    # MPAS
    held_suarez_forcing_mpas,
    held_suarez_init_mpas,
    # Spectral
    held_suarez_forcing_spectral,
)
