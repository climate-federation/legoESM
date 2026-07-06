"""Unit tests for the eORCA05 (1/2 deg) tripole mesh builder's decimation logic.

The end-to-end fold consistency is validated at build time by the model's own
create_tripole_grid._detect_fold (fail-loud). These tests pin the decimation
PHASES that make that pass: keep the fold row (j=[1::2]) and preserve the
de-haloed fold pairing (i=[::2] -> subsampled col i' pairs with (n_lon'-i')).
"""

from __future__ import annotations

import numpy as np

from scripts.data.build_eorca05_from_eorca025 import _dec2d, _J, _I


def test_decimation_halves_and_keeps_fold_row():
    # source (z, y, x) with y=1206 (fold at last row 1205), x=1440
    z, y, x = 3, 1206, 1440
    a = np.arange(z * y * x).reshape(z, y, x).astype(float)
    d = _dec2d(a)
    assert d.shape == (z, 603, 720)
    # j=[1::2] keeps the LAST row (1205, the tripole T-fold); drops south wall 0
    assert _J == slice(1, None, 2) and _I == slice(0, None, 2)
    assert d[0, -1, 0] == a[0, 1205, 0]          # last decimated row = fold row
    assert d[0, 0, 0] == a[0, 1, 0]              # first decimated row = j=1


def test_i_phase_preserves_fold_pairing():
    # eORCA025 de-haloed fold: col i pairs with (n_lon-i)%n_lon (n_lon=1440).
    # After i=[::2], subsampled col i' = 2i must pair with (n_lon'-i')%n_lon',
    # n_lon'=720. Verify the index identity that makes the fold survive.
    n_lon, n_lonp = 1440, 720
    for ip in range(n_lonp):
        i_full = 2 * ip                                   # source col of subsample
        partner_full = (n_lon - i_full) % n_lon           # source fold partner
        partner_sub = (n_lonp - ip) % n_lonp              # subsampled fold partner
        # the source partner must itself be an even col (in the subsample) whose
        # subsampled index is the subsampled partner
        assert partner_full % 2 == 0
        assert partner_full // 2 == partner_sub


def test_dec2d_2d_input():
    a = np.arange(1206 * 1440).reshape(1206, 1440).astype(float)
    d = _dec2d(a)
    assert d.shape == (603, 720)
