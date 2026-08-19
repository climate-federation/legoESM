"""A level count measured slow must say so — and must not claim a cause.

MEASURED on one A100, float32, icosahedral core, 163,842 cells, 60 timed
steps per arm, every level count inside ONE allocation: counts from 22 to 30
cost about 2.5x as much per level as 32 and above. The step is 20.3 ms at 26
levels and 8.5 ms at 32 — more work, less than half the time.

An earlier version of this guard asserted the cause was byte alignment of a
cell's column. The finer scan refuted it: 24 and 28 levels are aligned and
slow, 20 and 52 are unaligned-or-aligned and fast. These tests therefore pin
that the guard reports a RANGE and explicitly disclaims a mechanism, because
the tempting failure is to re-introduce a tidy rule the data does not carry.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from legoesm.grids.vertical import (
    _LEVELS_MEASURED_FAST,
    _LEVEL_COST_PS,
    create_sigma_coordinate,
    warn_if_unaligned_levels,
)


@pytest.mark.parametrize("n_levels", [13, 18, 21, 22, 24, 25, 26, 27, 28,
                                     30, 31, 34])
def test_warns_on_every_level_count_measured_expensive(n_levels):
    with pytest.warns(UserWarning, match="MEASURED expensive"):
        warn_if_unaligned_levels(n_levels, np.float32, where="test")


@pytest.mark.parametrize("n_levels", [16, 20, 32, 36, 40, 52])
def test_silent_on_every_level_count_measured_cheap(n_levels):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unaligned_levels(n_levels, np.float32, where="test")


@pytest.mark.parametrize("n_levels", [5, 8, 23, 29, 33, 64, 128])
def test_unmeasured_counts_do_not_warn(n_levels):
    """A count absent from the table is unmeasured. Warning about it would
    be inventing data; the message says so instead."""
    assert n_levels not in _LEVEL_COST_PS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unaligned_levels(n_levels, np.float32, where="test")


def test_the_table_contradicts_both_refuted_rules():
    """24 and 28 are byte-aligned yet expensive, which is what killed the
    alignment rule; 52 holds the largest live set yet is the cheapest, which
    is what killed the cache-capacity rule. If either fact is ever edited
    out of the table, the disclaimers above it stop being true."""
    assert _LEVEL_COST_PS[24] > 3000 and _LEVEL_COST_PS[28] > 3000
    # 52 holds the largest live set of any count measured and is among the
    # cheapest -- within 7% of the cheapest, and less than half the cost of
    # counts holding much less. A capacity explanation predicts the reverse.
    assert 52 in _LEVELS_MEASURED_FAST
    assert _LEVEL_COST_PS[52] < _LEVEL_COST_PS[30] / 2
    assert 24 not in _LEVELS_MEASURED_FAST
    assert 32 in _LEVELS_MEASURED_FAST


def test_message_disclaims_a_mechanism_and_scopes_the_result():
    with pytest.warns(UserWarning) as rec:
        warn_if_unaligned_levels(26, np.float32, where="test")
    text = str(rec[0].message)
    assert "NOT established" in text, text
    assert "refuted" in text, text
    assert "Re-measure" in text, text
    assert "unmeasured" in text, text
    # It must not tell the user to change the model as the first resort.
    assert "physics requirement, keep it" in text, text


def test_not_extrapolated_to_other_precisions():
    """The measurement is float32. A range whose cause is unknown cannot be
    rescaled to another width without inventing data."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_unaligned_levels(26, np.float64, where="test")


def test_the_real_factory_warns_at_26_levels():
    with pytest.warns(UserWarning, match="create_sigma_coordinate"):
        create_sigma_coordinate(26, dtype=np.float32)


def test_the_real_factory_is_silent_at_32_levels():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        create_sigma_coordinate(32, dtype=np.float32)
