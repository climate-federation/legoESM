"""Re-export of idealized terrain helpers for test-case modules.

The actual implementations live in
:mod:`legoesm.atmosphere.idealized.topography` so that production
modules (e.g. :mod:`legoesm.atmosphere.idealized.held_suarez_topo`)
do not depend on the ``tests/`` tree.  This shim exists for
backwards compatibility with any third-party code that imported the
helpers from the original test-tree location.
"""

from legoesm.atmosphere.idealized.topography import (  # noqa: F401
    dcmip_2_0_0_mountain,
    gaussian_mountain,
    great_circle_distance,
    surface_geopotential,
    williamson5_cone,
)
