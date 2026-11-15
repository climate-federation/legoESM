"""CLM-ml lapse rate must be the Fortran oracle value (MLclm_varcon.F90:26).

The port previously used constants.g / constants.c_pd (0.00976 K/m); the
oracle uses 0.0098 K/m.  The tier-5 land audit bisection showed this constant
is the entire 1-day difference between the port and the author's JAX code.
"""

from legoesm import constants
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLclm_varcon


def test_lapse_rate_matches_fortran_oracle():
    assert constants.lapse_rate_clmml == 0.0098
    assert MLclm_varcon.lapse_rate == constants.lapse_rate_clmml
