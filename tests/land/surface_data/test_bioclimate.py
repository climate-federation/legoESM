"""Unit tests for bioclimatic climate-zone classification."""

import numpy as np

from legoesm import constants
from legoesm.land.surface_data.bioclimate import (
    BioclimateConfig,
    coldest_month_temp_C,
    growing_degree_days,
    classify_climate_zones,
    climate_zones_from_t2m,
)


def _monthly(const_C, n_cells=1):
    """A flat (12, n_cells) climatology at a constant temperature [C]."""
    return np.full((12, n_cells), const_C + constants.T_freeze)


def test_coldest_month_temp():
    t = _monthly(10.0)
    t[0] = -5.0 + constants.T_freeze        # January coldest
    tc = coldest_month_temp_C(t)
    np.testing.assert_allclose(tc, -5.0)


def test_growing_degree_days_constant():
    # constant 10 C, base 5 C -> 5 deg excess every day of the year (365 days)
    gdd = growing_degree_days(_monthly(10.0))
    np.testing.assert_allclose(gdd, 5.0 * 365.0)
    # below base -> zero
    np.testing.assert_allclose(growing_degree_days(_monthly(2.0)), 0.0)


def test_classify_zones_partition_and_thresholds():
    tc = np.array([20.0, 0.0, -25.0, 0.0])
    gdd = np.array([5000.0, 2000.0, 100.0, 500.0])   # cell3: warm-ish Tc but low GDD
    z = classify_climate_zones(tc, gdd)
    # tropical: Tc>=15.5
    assert z.tropical.tolist() == [True, False, False, False]
    # temperate: not tropical, Tc>=-19 AND GDD>=1200
    assert z.temperate.tolist() == [False, True, False, False]
    # boreal: cold (cell2) OR insufficient GDD (cell3)
    assert z.boreal.tolist() == [False, False, True, True]
    # exact partition (each cell in exactly one zone)
    cover = z.tropical.astype(int) + z.temperate.astype(int) + z.boreal.astype(int)
    np.testing.assert_array_equal(cover, np.ones(4, dtype=int))


def test_climate_zones_from_t2m_end_to_end():
    # a warm uniform climatology -> tropical everywhere
    z = climate_zones_from_t2m(_monthly(25.0, n_cells=3))
    assert z.tropical.all()


def test_config_thresholds_respected():
    cfg = BioclimateConfig(tc_tropical_min_C=10.0)
    z = classify_climate_zones(np.array([12.0]), np.array([3000.0]), cfg)
    assert z.tropical[0]            # 12 C now counts as tropical under custom cfg
