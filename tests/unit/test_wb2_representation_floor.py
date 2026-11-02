"""Direct tests for the WeatherBench-deck path of wb2_representation_floor."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/validate/wb2_representation_floor.py"
_spec = importlib.util.spec_from_file_location("wb2_floor_under_test", _P)
floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(floor)


def test_deck_windows_reproduce_scorecard_dates_across_new_year():
    times = np.arange(np.datetime64("2019-12-30T00"), np.datetime64("2020-03-01T00"),
                      np.timedelta64(6, "h")).astype("datetime64[ns]")
    win = floor.deck_init_windows(times, 2020, 3, 360, 6)
    assert len(win) == 3
    for k, ((a, b), hours) in enumerate(win):
        assert hours == [0, 6, 12, 18]
        assert b - a == 6                       # 1 lead-in + 4 inits + 1 truth
        date = np.datetime64("2020-01-01T00") + np.timedelta64(360 * k, "h")
        assert times[a + 1] == date             # first paired init = scorecard date
        assert times[a] == date - np.timedelta64(6, "h")



def test_deck_windows_refuse_a_missing_record():
    times = np.arange(np.datetime64("2019-12-30T00"), np.datetime64("2020-03-01T00"),
                      np.timedelta64(6, "h")).astype("datetime64[ns]")
    gap = np.delete(times, np.searchsorted(times, np.datetime64("2020-01-16T06")))
    floor.deck_init_windows(times, 2020, 3, 360, 6)
    with pytest.raises(SystemExit, match="not the 6-hourly run"):
        floor.deck_init_windows(gap, 2020, 3, 360, 6)

def test_split_scores_separates_land_and_ocean():
    lat = np.linspace(-60, 60, 5)
    verif = {"t2m": np.zeros((5, 8))}
    land = np.zeros((5, 8), bool)
    land[:, :4] = True
    pred = {"t2m": np.where(land, 1.0, -3.0)}
    valid = {"t2m": np.ones((5, 8), bool)}
    s = floor.split_scores(pred, verif, valid, lat, land)
    assert s[("land", "t2m")] == pytest.approx((1.0, 1.0))
    assert s[("ocean", "t2m")] == pytest.approx((3.0, -3.0))
    assert s[("all", "t2m")] == pytest.approx((np.sqrt(5.0), -1.0))


def test_aggregate_is_mean_over_cases_and_nan_is_fatal():
    r = floor._aggregate({("overall", "land", "t2m"): [(1.0, 0.5), (3.0, -0.5)]})
    assert r["overall"]["land"]["t2m"] == {"rmse": 2.0, "bias": 0.0, "n": 2}
    with pytest.raises(SystemExit):
        floor._aggregate({("overall", "all", "t2m"): [(np.nan, 0.0)]})


@pytest.mark.parametrize("argv", [
    ["--config", "x.yaml", "--mode", "sfno"],                          # no sample
    ["--config", "x.yaml", "--mode", "sfno", "--n-inits", "24"],       # no stride
    ["--config", "x.yaml", "--n-inits", "24", "--init-stride-hours", "360"],
    ["--config", "x.yaml", "--mode", "sfno", "--n-inits", "0",
     "--init-stride-hours", "360"],
])
def test_deck_mode_refuses_implicit_sample_or_mode(argv):
    with pytest.raises(SystemExit):
        floor.main(argv)
