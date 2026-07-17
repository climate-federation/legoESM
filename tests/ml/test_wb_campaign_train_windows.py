"""The WB campaign yamls must carry scattered, all-season train_windows (#1160).

The default (no train_windows) is the consecutive back-compat mode: the first
n_days of each train_year = one autocorrelated ~60-sample January block, which
#1047 proved is a NULL result for the neural_gcm/sfno arms (they never beat the
pure dycore). The campaign configs pin 60 month-start 1-day windows so the NN
arms see independent synoptic scenes in every season. This guards that block
against a typo / accidental removal AND proves scale_build actually consumes it.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from legoesm.training.scale_build import _resolve_windows

_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN = _ROOT / "config" / "wb" / "campaign"


class _NoCliWindows:
    windows = None


def _windows_for(name):
    yml = yaml.safe_load(open(_CAMPAIGN / name))
    return _resolve_windows(_NoCliWindows(), yml), yml


def _month_starts(year):
    """Leap-aware day-of-year (0-indexed) of the 1st of each month."""
    import calendar
    offs, d = [], 0
    for m in range(1, 13):
        offs.append(d)
        d += calendar.monthrange(year, m)[1]
    return offs


def _check(name):
    windows, yml = _windows_for(name)
    # 12 month-start windows per train-year, all one day.
    assert windows is not None, f"{name}: train_windows resolved to None (consecutive mode)"
    years = list(yml["train_years"])
    assert len(windows) == 12 * len(years), (name, len(windows))
    assert all(nd == 1 for _, _, nd in windows)
    # Every train year is represented.
    assert sorted(set(y for y, _, _ in windows)) == sorted(years)
    # PER YEAR: exactly the 12 leap-aware month-start offsets (catches the leap
    # drift where a fixed non-leap offset table shifts Feb 29 and drops December).
    for y in years:
        offs = sorted(off for yy, off, _ in windows if yy == y)
        assert offs == _month_starts(y), (name, y, offs)


def test_t63_campaign_has_all_season_windows():
    _check("spectral_t63.yaml")


def test_t106_campaign_has_all_season_windows():
    _check("spectral_t106.yaml")
