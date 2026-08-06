"""``fort(strict_rank=True)`` must catch the silent 3-D column broadcast.

``fort.__getitem__`` splits its subscript as ``i, j, *k``. On a 3-D array a
two-subscript read leaves ``k == []``, so it returns the whole vertical
column, which then broadcasts through the 2-D stage kernels and yields
NUMBERS instead of an error. That is the exact failure mode the km-general
six-face driver has to be immune to, so the guard is opt-in there and the
default is left alone for the bit-exact-certified 2-D call sites.

Every assertion below is paired with its synthetic violation: the guard is
shown to FIRE on the bad access and to STAY SILENT on the good one.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_native_gridstruct import fort


def _a3(nk=4):
    return np.arange(3 * 3 * nk, dtype=np.float64).reshape(3, 3, nk)


# ------------------------------------------------------ default is unchanged

def test_default_still_broadcasts_so_certified_callers_are_untouched():
    """The whole point of the default: existing bit-exact call sites must
    behave EXACTLY as before. If this ever starts raising, the certified
    2-D lane has been changed underneath."""
    f = fort(_a3(), 1, 1)
    got = f[1, 1]
    assert got.shape == (4,)
    assert np.array_equal(got, np.arange(4, dtype=np.float64))


def test_default_2d_access_on_2d_array_is_a_scalar():
    f = fort(np.arange(9.0).reshape(3, 3), 1, 1)
    assert f[2, 3] == pytest.approx(5.0)


# ---------------------------------------------------------- the guard fires

def test_strict_rank_rejects_a_2_subscript_read_of_a_3d_array():
    f = fort(_a3(), 1, 1, strict_rank=True)
    with pytest.raises(IndexError, match="broadcasts silently"):
        _ = f[1, 1]


def test_strict_rank_rejects_the_same_write():
    f = fort(_a3(), 1, 1, strict_rank=True)
    with pytest.raises(IndexError, match="broadcasts silently"):
        f[1, 1] = 0.0


def test_strict_rank_message_names_the_shape_and_the_subscripts():
    f = fort(_a3(nk=7), 1, 1, strict_rank=True)
    with pytest.raises(IndexError) as e:
        _ = f[2, 3]
    msg = str(e.value)
    assert "(3, 3, 7)" in msg and "3-D" in msg and "2 subscripts" in msg


# ------------------------------------------------- the guard does NOT overfire

def test_strict_rank_allows_the_correct_3_subscript_read():
    f = fort(_a3(), 1, 1, strict_rank=True)
    assert f[1, 1, 0] == pytest.approx(0.0)
    assert f[2, 2, 3] == pytest.approx(_a3()[1, 1, 3])


def test_strict_rank_allows_a_2_subscript_read_of_a_2d_array():
    f = fort(np.arange(9.0).reshape(3, 3), 1, 1, strict_rank=True)
    assert f[2, 3] == pytest.approx(5.0)
    f[2, 3] = 42.0
    assert f[2, 3] == pytest.approx(42.0)


def test_strict_rank_write_roundtrips_on_3d():
    a = _a3()
    f = fort(a, 1, 1, strict_rank=True)
    f[1, 1, 2] = -5.0
    assert a[0, 0, 2] == pytest.approx(-5.0)


def test_strict_rank_rejects_too_many_subscripts_on_2d():
    """Over-subscripting is the mirror error and must also be fatal."""
    f = fort(np.arange(9.0).reshape(3, 3), 1, 1, strict_rank=True)
    with pytest.raises(IndexError, match="3 subscripts"):
        _ = f[1, 1, 0]


def test_lo_offsets_are_applied_under_strict_rank():
    """The guard must not disturb the Fortran-origin arithmetic."""
    a = _a3()
    f = fort(a, 3, 5, strict_rank=True)
    assert f[3, 5, 1] == pytest.approx(a[0, 0, 1])
    assert f[5, 7, 2] == pytest.approx(a[2, 2, 2])
