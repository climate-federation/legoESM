"""A level count that makes columns unaligned must say so.

Measured on one A100 with the icosahedral core in float32: at 163,842 cells
the step takes 20.2 ms at 26 levels and 8.5 ms at 32 -- more work, less
time -- because a column of 26 float32 values is 104 bytes, which is not a
multiple of 16, so only every other column starts on a vector boundary.
Every level count divisible by four measured about three times cheaper per
level than the two that were not.

It stays a warning rather than an error: the answer is right either way, and
matching another model's grid has to remain possible.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from legoesm.grids.vertical import (
    create_sigma_coordinate,
    warn_if_unaligned_levels,
)


@pytest.mark.parametrize("n_levels", [13, 22, 25, 26, 27, 30])
def test_warns_on_unaligned_float32_level_counts(n_levels):
    with pytest.warns(UserWarning, match="column stride"):
        warn_if_unaligned_levels(n_levels, np.float32, where="test")


@pytest.mark.parametrize("n_levels", [4, 20, 24, 28, 32, 40, 52])
def test_silent_on_aligned_float32_level_counts(n_levels):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unaligned_levels(n_levels, np.float32, where="test")


def test_rule_follows_the_dtype_not_a_hardcoded_level_count():
    """float64 columns are 8 bytes per level, so 26 levels IS aligned there;
    a rule written as 'levels must divide by four' would be wrong."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unaligned_levels(26, np.float64, where="test")
    with pytest.warns(UserWarning, match="column stride"):
        warn_if_unaligned_levels(25, np.float64, where="test")


def test_message_tells_the_user_to_pad_storage_not_to_change_the_model():
    """The first version told users to run 28 levels instead of 26. That is
    a change to the model's vertical grid to satisfy a performance lint --
    the review's sharpest objection. The advice must be to keep the level
    count and pad the storage."""
    with pytest.warns(UserWarning) as rec:
        warn_if_unaligned_levels(26, np.float32, where="test")
    text = str(rec[0].message)
    assert "104-byte" in text, text
    assert "KEEP your 26 levels" in text, text
    assert "padded to 28" in text, text
    assert "changes no physics" in text, text
    # The mechanism must be named as a vectorized load, not a cache line:
    # a reader who cargo-cults 128-byte alignment learns the wrong rule.
    assert "vectorized load" in text, text


def test_rule_uses_the_state_dtype_not_the_coordinate_dtype():
    """A mixed-precision run can build a float64 coordinate over float32
    state. The stride that matters is the state's, so leaving the dtype
    unset must resolve it from the precision policy rather than silently
    passing because the coordinate happens to be wider."""
    from legoesm.core.precision import get_policy

    storage = np.dtype(get_policy().storage)
    n_levels = 26
    unaligned = (storage.itemsize * n_levels) % 16 != 0
    if unaligned:
        with pytest.warns(UserWarning, match="column stride"):
            warn_if_unaligned_levels(n_levels, where="test")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_unaligned_levels(n_levels, where="test")


def test_the_real_factory_warns_at_26_levels():
    """The advisory has to fire from the call users actually make."""
    with pytest.warns(UserWarning, match="create_sigma_coordinate"):
        create_sigma_coordinate(26, dtype=np.float32)


def test_the_real_factory_is_silent_at_28_levels():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        create_sigma_coordinate(28, dtype=np.float32)
