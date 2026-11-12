"""Lat-lon OMIP solar geometry: at the March equinox, local solar noon, cos(zenith) = cos(lat).

Before the fix the lat-lon setup converted the (already radian) grid coordinates to radians a
second time, so every cell sat inside 1.6 deg of the equator for the zenith calculation and
cos(zenith) was ~1 at noon at every latitude.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_run_omip_jra55_dispatch as D  # noqa: E402

R = D.run_omip


def test_equinox_noon_zenith_matches_latitude(tmp_path):
    from legoesm.forcing.jra55_do import _jra55_cos_zenith
    cache = D._make_synthetic_cache(tmp_path, n_lat=8, n_lon=16)
    grid, *_ = D._make_tiny_latlon_setup(n_lat=8, n_lon=16)
    js = R._setup_jra55_forcing_state(D._argparse_namespace(jra55_cache=str(cache)),
                                      grid, "latlon")
    lat_deg = np.rad2deg(np.asarray(grid.lat, np.float64))
    # noleap day-of-year 79 (20 March, 0-based) at 12 UTC: solar noon at lon 0 (column 0)
    cz = np.asarray(_jra55_cos_zenith(js["lat_2d"], js["lon_2d"], 79.5, ref_year=1958))
    # the zenith code takes day-of-year 80 here, where its declination is zero, and applies
    # no equation of time, so UTC noon at lon 0 is exact: compare at rounding tolerance
    np.testing.assert_allclose(cz[:, 0], np.cos(np.deg2rad(lat_deg)), rtol=0, atol=1e-6)
    # and midnight on the opposite meridian is dark
    assert (cz[:, grid.lon.shape[0] // 2] <= 1e-6).all()
