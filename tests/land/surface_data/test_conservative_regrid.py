"""Tests for conservative, NaN-aware lat-lon regridding."""

import numpy as np

from legoesm.grids.regridding import conservative_regrid_latlon as cr


def _grid(n_lat, n_lon, lon0=0.0):
    lat = -90.0 + (np.arange(n_lat) + 0.5) * (180.0 / n_lat)
    lon = lon0 + (np.arange(n_lon) + 0.5) * (360.0 / n_lon)
    return lat, lon


def test_constant_preserved():
    slat, slon = _grid(8, 16)
    tlat, tlon = _grid(20, 40)
    out = cr(np.full((8, 16), 0.314), slat, slon, tlat, tlon)
    assert np.allclose(out, 0.314, atol=1e-6)


def test_global_mean_conserved():
    rng = np.random.default_rng(0)
    slat, slon = _grid(36, 72)
    tlat, tlon = _grid(12, 24)                     # coarsen
    f = rng.uniform(0, 1, (36, 72))
    out = cr(f, slat, slon, tlat, tlon)
    area = lambda la: np.cos(np.deg2rad(la))[:, None]
    src_mean = np.sum(f * area(slat)) / np.sum(np.ones_like(f) * area(slat))
    tgt_mean = np.sum(out * area(tlat)) / np.sum(np.ones_like(out) * area(tlat))
    assert np.isclose(src_mean, tgt_mean, rtol=2e-3)   # conservative


def test_nan_not_propagated_to_coastal_cells():
    # NaN block in the interior (away from the 0/360 seam) to isolate masking.
    slat, slon = _grid(36, 72)
    tlat, tlon = _grid(18, 36)
    f = np.full((36, 72), 5.0)
    f[:, 18:54] = np.nan                            # NaN over a central lon band
    out = cr(f, slat, slon, tlat, tlon)
    # cells entirely over the valid region stay exactly 5 (no NaN bleed)
    assert np.isclose(out[9, 0], 5.0)
    assert np.isclose(out[9, -1], 5.0)
    # a cell entirely inside the NaN band is NaN
    assert np.isnan(out[9, 18])


def test_layer_axis_and_coarsen_mean():
    slat, slon = _grid(4, 4)
    tlat, tlon = _grid(2, 2)                        # each tgt = 2x2 src block
    f = np.zeros((4, 4, 2))
    f[..., 0] = np.arange(16).reshape(4, 4)
    f[..., 1] = 1.0
    out = cr(f, slat, slon, tlat, tlon)
    assert out.shape == (2, 2, 2)
    assert np.allclose(out[..., 1], 1.0)           # constant layer preserved
    assert np.all(np.isfinite(out))
