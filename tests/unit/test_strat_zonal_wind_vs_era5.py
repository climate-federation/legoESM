import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "scripts", "validate"))
from strat_zonal_wind_vs_era5 import (  # noqa: E402
    compare, interp_logp, model_zonal_mean, top_level_pressures_pa)


def test_top_level_pressures_and_refusal():
    a = np.array([0.002, 0.005, 0.01, 0.02])
    b = np.array([0.0, 0.0, 0.0, 0.1])
    np.testing.assert_allclose(top_level_pressures_pa([a, b], [0, 1], 1e5), [350.0, 750.0])
    with pytest.raises(ValueError, match="pure-pressure"):
        top_level_pressures_pa([a, b], [2], 1e5)


def test_zonal_mean_rows_are_latitude():
    n_lat, n_lon = 3, 4
    u = np.zeros((n_lat * n_lon, 2))
    u[:, 0] = np.repeat([1.0, 2.0, 3.0], n_lon) + np.tile([0.5, -0.5, 1.0, -1.0], n_lat)
    np.testing.assert_allclose(model_zonal_mean(u, n_lat, n_lon, [0])[:, 0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        model_zonal_mean(u, 4, 4, [0])


def test_interp_is_linear_in_log_p_and_refuses_extrapolation():
    p = np.array([100.0, 300.0, 500.0, 1000.0])
    f = 7.0 + 2.0 * np.log(p)
    out = interp_logp(f[None, :], p, [200.0, 700.0])
    np.testing.assert_allclose(out[0], 7.0 + 2.0 * np.log([200.0, 700.0]), rtol=1e-12)
    with pytest.raises(ValueError):
        interp_logp(f[None, :], p, [50.0])


def test_compare_zero_on_equal_and_picks_hemisphere_extrema():
    lat = np.array([-60.0, -20.0, 20.0, 60.0])
    zm = np.array([[-5.0], [10.0], [3.0], [40.0]])
    r = compare(zm, zm, lat)[0]
    assert r["rms"] == 0.0
    assert (r["model_nh_max"], r["model_nh_lat"]) == (40.0, 60.0)
    assert (r["model_sh_max"], r["model_sh_lat"]) == (10.0, -20.0)
    r2 = compare(zm + 2.0, zm, lat)[0]
    assert r2["rms"] == pytest.approx(2.0)
