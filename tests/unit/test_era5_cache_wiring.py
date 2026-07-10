"""#895: the local ERA5 cache is scoped to the span of windows THIS call loads.
The scope must (a) prefer the explicit ``windows`` arg over ``config.windows``
(they can differ — else a caller's window is read against a cache built for the
wrong years), and (b) add +1 year of headroom so target snapshots that spill
into the following year are covered.
"""
from __future__ import annotations

from types import SimpleNamespace

from legoesm.training.neural_gcm_spectral import _training_year_range


def test_year_range_spans_windows_arg_with_target_headroom():
    wins = [(1979, 0, 5), (2000, 10, 5), (2014, 3, 5)]
    # min start year .. max start year + 1 (targets spill into the next year)
    assert _training_year_range(wins, SimpleNamespace(windows=None)) == (1979, 2015)


def test_windows_arg_overrides_config_windows():
    # The data actually read is the ARG, not config.windows -> cache the arg's span.
    cfg = SimpleNamespace(windows=[(1979, 0, 5)])
    assert _training_year_range([(2000, 0, 5)], cfg) == (2000, 2001)


def test_year_range_fallback_without_windows():
    cfg = SimpleNamespace(windows=None, start_year=2015, n_train_days=400)
    # 400 days spans 2 calendar years, +1 headroom for the target lead
    assert _training_year_range(None, cfg) == (2015, 2017)
