"""Backwards-compat shim.

The canonical Held-Suarez forcing and initialization live in
``legoesm.atmosphere.held_suarez`` so that production entry points
(``run_amip.py``, ``ModelDriver``, benchmark scripts) can import them
without requiring the ``tests/`` directory to be on ``sys.path``.

Keep this shim so existing ``from tests.test_cases.held_suarez import ...``
call sites continue to work.
"""

from legoesm.atmosphere.held_suarez import *  # noqa: F401,F403
from legoesm.atmosphere.held_suarez import (  # noqa: F401
    K_A,
    K_S,
    K_F,
    SIGMA_B,
    DELTA_T_Y,
    DELTA_THETA_Z,
    T_MIN,
    P_0,
    held_suarez_equilibrium_temperature,
    held_suarez_forcing,
    held_suarez_forcing_latlon,
    held_suarez_forcing_mpas,
    held_suarez_forcing_spectral,
    held_suarez_init,
    held_suarez_init_latlon,
    held_suarez_init_mpas,
)
