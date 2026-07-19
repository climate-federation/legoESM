import numpy as np, numpy.testing as npt
from legoesm.land.carbon.climate_features import reduce_climatology_to_features

def _const_month(v, n=3): return np.full((n, 12), float(v))

def test_mat_is_annual_mean_T():
    t = np.tile(np.linspace(280, 300, 12), (2, 1))
    f = reduce_climatology_to_features(t, _const_month(2e-5, 2), _const_month(250, 2), _const_month(100, 2))
    npt.assert_allclose(f.mat_k, t.mean(axis=1), rtol=1e-9)

def test_map_is_annual_precip():
    f = reduce_climatology_to_features(_const_month(290), _const_month(2e-5), _const_month(250), _const_month(100))
    npt.assert_allclose(f.map_yr, 2e-5 * 365.0 * 86400.0, rtol=1e-6)

def test_seasonality_half_range():
    t = np.zeros((1, 12)); t[0, 0] = 270.0; t[0, 6] = 300.0; t[0, 1:6] = 285; t[0, 7:] = 285
    f = reduce_climatology_to_features(t, _const_month(2e-5, 1), _const_month(250, 1), _const_month(100, 1))
    npt.assert_allclose(f.t_seasonal_amp_k, 15.0, rtol=1e-9)

def test_aridity_wet_gt_dry():
    wet = reduce_climatology_to_features(_const_month(290), _const_month(1e-4), _const_month(250), _const_month(80))
    dry = reduce_climatology_to_features(_const_month(290), _const_month(1e-6), _const_month(250), _const_month(80))
    assert float(wet.aridity[0]) > float(dry.aridity[0])
