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


def test_pole_centred_source_keeps_polar_cap():
    """ERA5-style source (centres include +/-90): the polar rows' half-cells
    must carry their cap area; unclamped edges at +/-90.5 gave them zero."""
    slat = np.linspace(-90.0, 90.0, 181)
    slon = np.arange(360) + 0.5
    tlat, tlon = _grid(90, 180)
    f = np.where(np.abs(slat) > 89.9, 100.0, 1.0)[:, None] * np.ones((1, 360))
    out = cr(f, slat, slon, tlat, tlon)
    e = np.clip(np.concatenate([[-90.0], 0.5 * (slat[:-1] + slat[1:]), [90.0]]),
                -90.0, 90.0)
    w_src = np.diff(np.sin(np.deg2rad(e)))[:, None]
    src_mean = np.sum(f * w_src) / np.sum(w_src * np.ones_like(f))
    w_tgt = np.cos(np.deg2rad(tlat))[:, None]
    tgt_mean = np.sum(out * w_tgt) / np.sum(w_tgt * np.ones_like(out))
    assert np.isclose(src_mean, tgt_mean, rtol=1e-6)


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


def test_regrid_scalar_nan_aware_vs_plain():
    import jax.numpy as jnp
    from legoesm.grids.regridding import (
        compute_latlon_to_voronoi_weights, regrid_scalar, regrid_scalar_nan_aware)
    slat, slon = _grid(8, 16)
    field = np.full((8, 16), 5.0); field[:, :8] = np.nan        # left half "ocean"
    # target points: deep-valid, boundary, deep-ocean (lon in deg)
    tlat = np.array([0.0, 0.0, 0.0]); tlon = np.array([280.0, 175.0, 70.0])
    w = compute_latlon_to_voronoi_weights(
        np.deg2rad(slat), np.deg2rad(slon), np.deg2rad(tlat), np.deg2rad(tlon), k_neighbors=4)
    plain = np.asarray(regrid_scalar(jnp.asarray(field), w))
    na = np.asarray(regrid_scalar_nan_aware(jnp.asarray(field), w))
    # NaN-aware: any target with >=1 valid neighbour is finite and == 5 (all valid==5)
    assert np.isfinite(na[0]) and np.isclose(na[0], 5.0)        # deep valid
    assert np.isfinite(na[1]) and np.isclose(na[1], 5.0)        # boundary -> no bleed
    assert np.isnan(na[2])                                      # deep ocean -> NaN
    # plain IDW bleeds NaN at the boundary cell
    assert np.isnan(plain[1])
