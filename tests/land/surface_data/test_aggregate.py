"""Unit tests for area-weighted block aggregation."""

import numpy as np
import pytest

from legoesm.land.surface_data.raster import EnviBilHeader
from legoesm.land.surface_data.aggregate import (
    area_weighted_block_mean,
    block_factors,
    coarse_grid_centers,
    row_cos_weights,
    aggregate_raster_streaming,
)


def test_constant_field_preserved():
    vals = np.full((4, 6), 7.0)
    valid = np.ones((4, 6), bool)
    w = np.ones(4)
    mean, frac = area_weighted_block_mean(vals, valid, w, by=2, bx=3)
    assert mean.shape == (2, 2)
    assert np.allclose(mean, 7.0)
    assert np.allclose(frac, 1.0)


def test_nodata_ignored_and_fraction():
    vals = np.array([[10.0, 99.0], [10.0, 99.0]])
    valid = np.array([[True, False], [True, False]])
    w = np.ones(2)
    mean, frac = area_weighted_block_mean(vals, valid, w, by=2, bx=2)  # one 2x2 block
    assert np.allclose(mean, 10.0)            # the masked 99s never enter the mean
    assert np.allclose(frac, 0.5)             # half the cell valid


def test_fully_invalid_block_is_nan():
    vals = np.zeros((2, 2))
    valid = np.zeros((2, 2), bool)
    mean, frac = area_weighted_block_mean(vals, valid, np.ones(2), by=2, bx=2)
    assert np.isnan(mean).all()
    assert np.allclose(frac, 0.0)


def test_area_weighting_uses_row_weights():
    # two rows, one block; row0 weight 3, row1 weight 1 -> weighted mean = (3*4+1*0)/4
    vals = np.array([[4.0, 4.0], [0.0, 0.0]])
    valid = np.ones((2, 2), bool)
    mean, _ = area_weighted_block_mean(vals, valid, np.array([3.0, 1.0]), by=2, bx=2)
    assert np.allclose(mean, 3.0)


def test_layer_axis_preserved():
    vals = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 5.0)], axis=-1)  # (2,2,2)
    valid = np.ones((2, 2), bool)
    mean, _ = area_weighted_block_mean(vals, valid, np.ones(2), by=2, bx=2)
    assert mean.shape == (1, 1, 2)
    assert np.allclose(mean[0, 0], [1.0, 5.0])


def test_per_layer_validity_masks_independently():
    # 2 layers; layer 0 valid in both rows, layer 1 only in row 0 (a data gap).
    vals = np.stack([np.array([[2.0, 2.0], [4.0, 4.0]]),
                     np.array([[10.0, 10.0], [-9.0, -9.0]])], axis=-1)   # (2,2,2)
    valid = np.stack([np.ones((2, 2), bool),
                      np.array([[True, True], [False, False]])], axis=-1)  # (2,2,2)
    mean, frac = area_weighted_block_mean(vals, valid, np.ones(2), by=2, bx=2)
    assert mean.shape == (1, 1, 2)
    assert np.allclose(mean[0, 0, 0], 3.0)        # layer 0 averages 2 and 4
    assert np.allclose(mean[0, 0, 1], 10.0)       # layer 1 ignores the -9 gap row
    assert np.allclose(frac, 1.0)                 # cell is land (some layer valid)


def test_row_cos_weights_clamped():
    w = row_cos_weights(np.array([0.0, 60.0, 89.9, 95.0]))
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 0.5, atol=1e-6)
    assert w[3] >= 0.0                         # beyond pole clamped, never negative


def _hdr(res=0.5):
    n_lat = int(round(180.0 / res))
    n_lon = int(round(360.0 / res))
    return EnviBilHeader(
        nrows=n_lat, ncols=n_lon, nbands=1, dtype=np.dtype("<u2"), nodata=65535.0,
        ulxmap=-180.0 + 0.5 * res, ulymap=90.0 - 0.5 * res, xdim=res, ydim=res,
        layout="BIL",
    )


def test_block_factors_and_validation():
    hdr = _hdr(res=1.0 / 120.0)               # 30 arc-second
    by, bx = block_factors(hdr, 0.25)
    assert (by, bx) == (30, 30)
    with pytest.raises(ValueError):
        block_factors(hdr, 0.01)               # 0.01/(1/120)=1.2, not an integer multiple


def test_coarse_grid_centers():
    lat, lon = coarse_grid_centers(0.25)
    assert lat.shape == (720,) and lon.shape == (1440,)
    assert lat[0] > lat[-1]                    # north -> south
    assert np.isclose(lat[0], 90.0 - 0.125)
    assert np.isclose(lon[0], -180.0 + 0.125)


def test_streaming_matches_direct_landfraction():
    # 0.5 deg fine "raster": a land patch in the NH; aggregate land fraction to 1 deg.
    hdr = _hdr(res=0.5)
    rng = np.random.default_rng(0)
    smu = np.full((hdr.nrows, hdr.ncols), hdr.nodata, dtype=np.uint16)
    smu[10:40, 20:50] = rng.integers(2, 1000, size=(30, 30)).astype(np.uint16)

    def land_transform(block, lat_block):
        valid = block != hdr.nodata
        return valid.astype(np.float64), valid           # value==1 over land

    mean, frac, clat, clon = aggregate_raster_streaming(
        smu, hdr, 1.0, land_transform, coarse_rows_per_chunk=17
    )
    assert mean.shape == (180, 360)
    # land fraction must lie in [0,1] and integrate to the source land area share
    valid_full = smu != hdr.nodata
    w = row_cos_weights(np.linspace(89.75, -89.75, hdr.nrows))[:, None]
    src_share = (valid_full * w).sum() / (np.ones_like(valid_full) * w).sum()
    dst_share = (frac * row_cos_weights(clat)[:, None]).sum() / (
        np.ones_like(frac) * row_cos_weights(clat)[:, None]
    ).sum()
    assert np.all((frac >= 0) & (frac <= 1.0 + 1e-9))
    assert np.isclose(src_share, dst_share, atol=1e-6)   # area conserved
