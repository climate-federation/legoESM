"""DCMIP-2025 test cases for non-hydrostatic dynamical cores.

Test cases from the Dynamical Core Model Intercomparison Project 2025:

1. Mountain-triggered breaking gravity waves (dry)
2. Mountain-triggered mesoscale flow phenomena (dry, small-Earth)
   a. Gap flow
   b. Vortex shedding
3. Squall line (moist, small-Earth, Kessler microphysics)

References
----------
- DCMIP-2025: https://sites.google.com/umich.edu/dcmip-2025/home
- Klemp et al. (2015): Idealized Global Nonhydrostatic Atmospheric
  Test Cases on a Reduced-Radius Sphere. JAMES.
"""

from legoesm.atmosphere.dynamics.dcmip2025.test_case_1 import (
    dcmip25_tc1_init,
    dcmip25_tc1_topography,
)
from legoesm.atmosphere.dynamics.dcmip2025.test_case_2 import (
    dcmip25_tc2_init,
    dcmip25_tc2a_topography,
    dcmip25_tc2b_topography,
)
from legoesm.atmosphere.dynamics.dcmip2025.test_case_3 import (
    dcmip25_tc3_init,
)
from legoesm.atmosphere.dynamics.dcmip2025.common import (
    apply_small_earth_scaling,
)
