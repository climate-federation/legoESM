"""``build_history_pairs`` must never pair across a window boundary.

The training windows are scattered months and years apart (the maxdata tier is
12 monthly 5-day windows per year, 1979-2014). Two entries adjacent in the
``ic_states`` LIST are adjacent in TIME only inside one window; at a boundary the
"previous" state can be a year away. Feeding that as a t-6 h input trains
without error and forecasts badly, so contiguity is decided on timestamps.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.training.neural_gcm_spectral import build_history_pairs


def _times(*groups, cadence_h=6):
    """Build timestamps for windows given as (start_iso, n_snapshots)."""
    out = []
    for start, n in groups:
        t0 = np.datetime64(start)
        out += [t0 + np.timedelta64(cadence_h * k, "h") for k in range(n)]
    return out


def test_zero_history_keeps_everything_and_is_the_default_shape():
    states = list(range(5))
    keep, prevs = build_history_pairs(states, None, 0, 6.0)
    assert keep == [0, 1, 2, 3, 4]
    assert prevs == [()] * 5


def test_first_sample_of_each_window_is_dropped():
    """Two 4-snapshot windows a year apart -> 2 drops, not 1."""
    states = list(range(8))
    times = _times(("2015-01-01T00", 4), ("2016-06-01T00", 4))
    keep, prevs = build_history_pairs(states, times, 1, 6.0)
    assert keep == [1, 2, 3, 5, 6, 7], keep
    # index 4 (the second window's first snapshot) must NOT appear: its
    # list-predecessor is index 3, a year earlier.
    assert 4 not in keep
    assert prevs == [(0,), (1,), (2,), (4,), (5,), (6,)]


def test_prev_states_are_oldest_first_for_multi_step_history():
    states = list(range(6))
    times = _times(("2015-01-01T00", 6))
    keep, prevs = build_history_pairs(states, times, 2, 6.0)
    assert keep == [2, 3, 4, 5]
    # sample 2's history is (0, 1): OLDEST FIRST, so index 1 is the most recent.
    assert prevs[0] == (0, 1)
    assert prevs[-1] == (3, 4)


def test_a_cadence_mismatch_drops_the_sample():
    """A gap that is not exactly one cadence is a boundary, not a predecessor."""
    states = list(range(4))
    times = [
        np.datetime64("2015-01-01T00"),
        np.datetime64("2015-01-01T06"),
        np.datetime64("2015-01-01T18"),   # 12 h gap, not 6
        np.datetime64("2015-01-02T00"),
    ]
    keep, _ = build_history_pairs(states, times, 1, 6.0)
    assert keep == [1, 3], keep


def test_missing_times_is_refused_not_assumed():
    states = list(range(4))
    with pytest.raises(ValueError, match="requires per-sample ic_times"):
        build_history_pairs(states, None, 1, 6.0)
    with pytest.raises(ValueError, match="requires per-sample ic_times"):
        build_history_pairs(states, [np.datetime64("2015-01-01T00"), None,
                                     None, None], 1, 6.0)


def test_non_six_hour_cadence_is_honoured():
    """The cadence is a parameter, not a hardcoded 6 h."""
    states = list(range(4))
    times = _times(("2015-01-01T00", 4), cadence_h=12)
    assert build_history_pairs(states, times, 1, 12.0)[0] == [1, 2, 3]
    # ...and the same data at the wrong declared cadence keeps nothing.
    assert build_history_pairs(states, times, 1, 6.0)[0] == []
