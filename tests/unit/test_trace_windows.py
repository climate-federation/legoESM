"""Profiler windows from LEGOESM_TRACE_STEPS / _WINDOWS / _RANKS."""
import pytest

from legoesm.driver.model_driver import trace_windows


def test_unset_traces_nothing():
    assert trace_windows(None, "0") == []
    assert trace_windows("", "0") == []


def test_one_window_keeps_the_original_layout():
    assert trace_windows("48:2:/t", "3") == [(48, 50, "/t/rank3")]


def test_back_to_back_windows_each_get_a_directory():
    assert trace_windows("48:2:/t", "0", "3") == [
        (48, 50, "/t/rank0/w00"), (50, 52, "/t/rank0/w01"), (52, 54, "/t/rank0/w02")]


def test_directory_may_contain_colons():
    assert trace_windows("1:2:/a:b", "0") == [(1, 3, "/a:b/rank0")]


def test_rank_filter():
    assert trace_windows("48:2:/t", "3", None, "0,5") == []
    assert trace_windows("48:2:/t", "5", None, "0,5") == [(48, 50, "/t/rank5")]


@pytest.mark.parametrize("spec,nw,ranks", [
    ("x:2:/t", None, None), ("48:2", None, None), ("-1:2:/t", None, None),
    ("48:0:/t", None, None), ("48:2:/t", "0", None), ("48:2:/t", "two", None),
    ("48:2:/t", None, "a,b")])
def test_malformed_settings_are_refused(spec, nw, ranks):
    with pytest.raises(ValueError):
        trace_windows(spec, "0", nw, ranks)
