"""Synthetic tests for the ORAS5 -> ORCA2 regrid weights (no data files needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))

import regrid_oras5_to_orca2 as r  # noqa: E402


def _grids():
    # source: 2 x 4 columns, 3 layers of 10 m; target: 1 x 2 columns, 2 layers (15 m, 15 m)
    slon, slat = np.meshgrid([0.0, 10.0, 20.0, 30.0], [0.0, 5.0])
    sarea = np.ones((2, 4))
    swet = np.ones((3, 2, 4), bool)
    swet[2, 0, 0] = False                                   # a shallower source column
    stop = np.broadcast_to(np.array([0.0, 10.0, 20.0])[:, None, None], (3, 2, 4)).copy()
    sbot = stop + 10.0
    dlon, dlat = np.meshgrid([5.0, 25.0], [2.5])
    dwet = np.ones((2, 1, 2), bool)
    dwet[:, 0, 1] = [True, False]                           # second target column one layer deep
    dtop = np.broadcast_to(np.array([0.0, 15.0])[:, None, None], (2, 1, 2)).copy()
    dbot = dtop + 15.0
    return (slon, slat, sarea, swet, stop, sbot), (dlon, dlat, dwet, dtop, dbot)


def test_weights_conserve_content_and_count_drops_and_gaps():
    src, dst = _grids()
    W3, W2, counts = r.build_weights(src, dst)
    T = np.random.default_rng(0).normal(size=(3, 2, 4))
    out = r.apply(W3, T, src[3], dst[2].shape)
    assert np.isnan(out[1, 0, 1])                            # target land below layer 1
    # content of kept water is identical on both grids
    kept = np.asarray(W3.sum(1)).reshape(dst[2].shape)
    assert np.isclose(np.nansum(out * kept), (W3 @ np.where(src[3], T, 0).ravel()).sum())
    # water in the right-hand columns below 15 m lands on target land: 2 rows x 2 cols x 15 m
    total = 2 * 4 * 30 - 10
    assert np.isclose(counts["dropped_volume_fraction"], (2 * 2 * 15) / total)
    assert counts["orca2_wet_cells_without_data"] == 0
    ssh = r.apply(W2, np.arange(8.0).reshape(2, 4), src[3][0], dst[2].shape[1:])
    np.testing.assert_allclose(ssh, [[np.mean([0, 1, 4, 5]), np.mean([2, 3, 6, 7])]])


def test_a_target_cell_with_no_source_is_nan_and_counted():
    src, dst = _grids()
    swet = src[3].copy(); swet[:, :, :2] = False             # nothing feeds the left target column
    W3, W2, counts = r.build_weights((*src[:3], swet, *src[4:]), dst)
    assert counts["orca2_wet_cells_without_data"] == 2
    out = r.apply(W3, np.ones((3, 2, 4)), swet, dst[2].shape)
    assert np.all(np.isnan(out[:, 0, 0])) and np.all(np.isfinite(out[0, 0, 1]))
