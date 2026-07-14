"""FV3_3D iter 890: explicit nord range validation tests.

Validates ``validate_corner_div_damp_nord`` raises ValueError for
nord outside FV3 namelist range {0, 1, 2, 3} and accepts canonical
values silently.  Prevents silent misuse outside the regression-
guard range covered by iter-886/887/888/889.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    validate_corner_div_damp_nord,
)


def test_nord_canonical_range_accepted():
    """nord ∈ {0, 1, 2, 3} (FV3 namelist range) accepted silently."""
    for nord in (0, 1, 2, 3):
        validate_corner_div_damp_nord(nord)  # no raise


def test_nord_negative_rejected():
    """nord < 0 raises ValueError (no negative del-N order)."""
    with pytest.raises(ValueError, match="outside FV3 namelist range"):
        validate_corner_div_damp_nord(-1)


def test_nord_too_high_rejected():
    """nord > 3 rejected (not regression-guarded)."""
    with pytest.raises(ValueError, match="outside FV3 namelist range"):
        validate_corner_div_damp_nord(4)
    with pytest.raises(ValueError, match="outside FV3 namelist range"):
        validate_corner_div_damp_nord(10)


def test_nord_non_integer_rejected():
    """nord must be int (not float, not None) — FV3 namelist type."""
    with pytest.raises(ValueError):
        validate_corner_div_damp_nord(1.5)
    with pytest.raises(ValueError):
        validate_corner_div_damp_nord(None)


def test_error_message_quotes_value():
    """Error message includes the offending value for debuggability."""
    with pytest.raises(ValueError, match="nord=99"):
        validate_corner_div_damp_nord(99)
