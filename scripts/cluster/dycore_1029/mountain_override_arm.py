"""#1029 scratch arm: run the matrix runner with the DCMIP ridge spacing overridden.

LEGOESM_PROBE_ZETA_FRAC (float) replaces ``zeta_frac`` in every
``dcmip_2_0_0_mountain`` call made by the held_suarez_topo inits.  Unset ->
the runner runs unchanged.  Prints the resulting mountain's grid-scale content
so each log proves the override reached the model.  Measurement only.
"""
import functools
import os
import runpy
import sys
from pathlib import Path

import numpy as np

import legoesm.atmosphere.idealized.held_suarez_topo as hst

_kw = {}
for _env, _key in (("LEGOESM_PROBE_ZETA_FRAC", "zeta_frac"),
                   ("LEGOESM_PROBE_RM_FRAC", "R_m_frac"),
                   ("LEGOESM_PROBE_LAT_C", "lat_c")):
    if os.environ.get(_env) is not None:
        _kw[_key] = float(os.environ[_env])
if _kw:
    _orig = hst.dcmip_2_0_0_mountain
    hst.dcmip_2_0_0_mountain = functools.partial(_orig, **_kw)
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(72, 144)
    from legoesm import constants
    z = np.asarray(hst._phis_latlon(g, 2000.0)) / constants.g
    c = z - 0.5 * (np.roll(z, 1, 1) + np.roll(z, -1, 1))
    print(f"#1029 PROBE {_kw}: latlon zmax={z.max():.1f} m "
          f"max|2dx lon|={np.abs(c).max():.1f} m", flush=True)
sys.argv = [str(Path(__file__).resolve().parents[3] / "scripts/matrix/run_atmosphere_test_matrix.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
