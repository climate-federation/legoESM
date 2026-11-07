"""Multi-GPU lat-band SPMD south-padding helpers in run_omip_core2.

``--n-gpus N`` partitions the ocean state by latitude band across N devices, which
requires ``n_lat % N == 0``.  eORCA025 (``n_lat=1207``) is odd, so the driver
appends LAND rows at the SOUTH (the tripole north fold is north-relative and stays
put).  These tests pin the two pure-array helpers:

* ``south_pad_rows(n_lat, n_gpus)`` — the number of rows to append.
* ``pad_mask_bathy_south(land_mask, H_bathy, n_pad)`` — the mask/bathy pad
  (wet rows preserved bit-exact, added rows LAND, divisibility achieved).

The GRID-side pad (``pad_tripole_grid_south``, fold-preserving) is tested in
``tests/grids/test_tripole_internals.py``; the end-to-end build_tripole +
shard-across-2-devices integration is the eORCA025 smoke (sbatch, GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids import tripole as R


@pytest.mark.parametrize("n_lat,n_gpus,expect", [
    (1207, 2, 1),     # eORCA025 odd -> +1 = 1208 (the production case)
    (1207, 1, 0),     # single device -> no pad
    (1208, 2, 0),     # already even
    (1207, 4, 1),     # 1207 % 4 == 3 -> +1 = 1208
    (1207, 5, 3),     # 1207 % 5 == 2 -> +3 = 1210
    (180, 6, 0),      # regular latlon already divisible
    (100, 3, 2),      # 100 % 3 == 1 -> +2 = 102
])
def test_south_pad_rows(n_lat, n_gpus, expect):
    n_pad = R.south_pad_rows(n_lat, n_gpus)
    assert n_pad == expect
    assert (n_lat + n_pad) % max(1, n_gpus) == 0


def test_pad_mask_bathy_south_preserves_wet_and_lands_added():
    rng = np.random.default_rng(0)
    n_lat, n_lon, n_pad = 17, 12, 3
    land_mask = (rng.random((n_lat, n_lon)) > 0.4).astype(np.float64)
    H_bathy = (rng.random((n_lat, n_lon)) * 4000.0) * land_mask

    lm_p, hb_p = R.pad_mask_bathy_south(land_mask, H_bathy, n_pad)

    # shape grew by n_pad on the south (axis 0)
    assert lm_p.shape == (n_lat + n_pad, n_lon)
    assert hb_p.shape == (n_lat + n_pad, n_lon)
    # original (wet) rows preserved BIT-EXACT, shifted +n_pad
    np.testing.assert_array_equal(lm_p[n_pad:], land_mask)
    np.testing.assert_array_equal(hb_p[n_pad:], H_bathy)
    # the added SOUTH rows are LAND (mask 0, bathy 0)
    assert float(np.abs(lm_p[:n_pad]).max()) == 0.0
    assert float(np.abs(hb_p[:n_pad]).max()) == 0.0
    # wet-cell COUNT unchanged (no cell created/destroyed)
    assert int(lm_p.sum()) == int(land_mask.sum())


def test_pad_mask_bathy_south_zero_is_identity():
    lm = np.ones((5, 4))
    hb = np.full((5, 4), 100.0)
    lm_p, hb_p = R.pad_mask_bathy_south(lm, hb, 0)
    np.testing.assert_array_equal(lm_p, lm)
    np.testing.assert_array_equal(hb_p, hb)
