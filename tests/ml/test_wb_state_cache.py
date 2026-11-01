"""WB model-grid state cache: store, gate, sample assembly, index selection."""
from __future__ import annotations

import collections
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from legoesm.training import scale_build as sb
from legoesm.training import wb_state_cache as wsc

Carry = collections.namedtuple("Carry", "T p_s")


def _rec(i):
    rng = np.random.default_rng(i)
    carry = Carry(rng.normal(size=(3, 4, 2)), rng.normal(size=(3, 4)))
    sfc = {"t2m": rng.normal(size=(3, 4)), "u10": rng.normal(size=(3, 4))}
    forcing = {"T_sfc": np.where(rng.random(12) > 0.5, np.nan, 280.0),
               "day_of_year": np.asarray(float(i)),
               "seconds_of_day": np.asarray(21600.0)}
    return carry, sfc, forcing


def _store(tmp_path, idx):
    p = tmp_path / "s.zarr"
    arrays, structure = wsc.record_arrays(*_rec(idx[0]))
    wsc.create_store(p, idx, arrays, structure, {})
    return p


def test_round_trip_float32_and_done_marks(tmp_path):
    p = _store(tmp_path, [10, 11, 12])
    for i in (10, 12):
        wsc.write_time(p, i, wsc.record_arrays(*_rec(i))[0])
    c = wsc.StateCache(p, _rec(10))
    assert c.missing([10, 11, 12]) == [11]
    carry, sfc, forcing = c.read(12)
    ref = _rec(12)
    np.testing.assert_array_equal(carry.T, ref[0].T.astype(np.float32).astype(np.float64))
    assert carry.T.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(forcing["T_sfc"]), np.isnan(ref[2]["T_sfc"]))
    with pytest.raises(RuntimeError, match="never written"):
        c.read(11)
    with pytest.raises(KeyError):
        c.read(13)


def test_gate_catches_one_float32_ulp_and_structure(tmp_path):
    p = _store(tmp_path, [10])
    arr = wsc.record_arrays(*_rec(10))[0]
    arr["c000"] = arr["c000"].copy()
    arr["c000"].flat[0] = np.nextafter(arr["c000"].flat[0], np.float32(np.inf))
    wsc.write_time(p, 10, arr)
    c = wsc.StateCache(p, _rec(10))
    with pytest.raises(RuntimeError, match="stale"):
        wsc.assert_cache_matches(_rec(10), c.read(10))
    other = (_rec(10)[0], {"t2m": _rec(10)[1]["t2m"]}, _rec(10)[2])
    with pytest.raises(RuntimeError, match="different sample structure"):
        wsc.StateCache(p, other)
    with pytest.raises(RuntimeError, match="different index list"):
        wsc.create_store(p, [10, 11], *wsc.record_arrays(*_rec(10)), {})


def _times(start="2019-12-30", n=40):
    return np.datetime64(start) + np.arange(n) * np.timedelta64(6, "h")


def test_targets_in_eval_year_are_dropped():
    cfg = SimpleNamespace(smoke=False, windows=None, n_days=None)
    yml = {"train_windows": [[2019, 0, 2]], "eval_years": [2020]}
    times = _times("2019-12-30")
    got = list(sb._training_sample_indices(cfg, yml, times, 4, 4))
    assert got and all(str(times[b])[:4] != "2020" for _, _, b in got)
    assert len(got) == 4                      # 30 Dec ICs only: 31 Dec -> 2020
    yml_no = {"train_windows": [[2019, 0, 2]]}
    assert len(list(sb._training_sample_indices(cfg, yml_no, times, 4, 4))) == 8


def test_validation_yaml_uses_test_protocol():
    yml = {"train_years": [2015], "eval_years": [2020], "train_windows": [[2015, 0, 1]],
           "validation": {"years": [2014], "n_inits": 24, "init_stride_hours": 360}}
    v = sb.wb_validation_yaml(yml, 24)
    assert "train_windows" not in v and v["train_years"] == [2014]
    assert v[sb._VAL_INITS_KEY] == [[2014, 360 * k] for k in range(24)]
    times = np.datetime64("2014-01-01") + np.arange(4 * 400) * np.timedelta64(6, "h")
    idx = list(sb._training_sample_indices(SimpleNamespace(smoke=False), v, times, 4, 4))
    assert [a for _, a, _ in idx] == [60 * k for k in range(24)]
    for bad, msg in ((dict(years=[2020]), "eval_years"), (dict(years=[2015]), "train_years"),
                     (dict(years=[2013, 2014]), "exactly one")):
        y = dict(yml, validation=dict(yml["validation"], **bad))
        with pytest.raises(ValueError, match=msg):
            sb.wb_validation_yaml(y, 24)
    with pytest.raises(ValueError, match="n_inits"):
        sb.wb_validation_yaml(yml, 12)


def test_cached_samples_gate_assembly_and_missing(tmp_path, monkeypatch):
    """Lazy samples from the cache: leads become (carry, sfc) targets, the
    first sample's times are gated against a fresh build, a gap is refused."""
    monkeypatch.setattr(sb, "wb_state_record", lambda i, *a: _rec(i))
    monkeypatch.setattr(sb, "wb_state_cache_configs", lambda cfg, yml: (None, None, "none"))
    monkeypatch.setattr(sb, "validate_carry_holds_scheme", lambda *a, **k: 3)
    p = _store(tmp_path, [0, 1, 2, 3, 4])
    monkeypatch.setattr(sb, "wb_state_cache_path", lambda *a: p)
    for i in range(5):
        wsc.write_time(p, i, wsc.record_arrays(*_rec(i))[0])
    cfg = SimpleNamespace(mode="sfno")
    mine = [(2015, 0, 2), (2015, 2, 4)]
    s = sb._state_cached_samples(cfg, {}, None, None, mine, (6, 12), 1,
                                 _times(), None, None, None, "none", None)
    assert len(s) == 2
    ic, target, forcing = s[1]
    assert len(target) == 2 and set(target[0][1]) == {"t2m", "u10"}
    np.testing.assert_array_equal(target[1][0].p_s,
                                  _rec(4)[0].p_s.astype(np.float32).astype(np.float64))
    assert [x[0].T.shape for x in s] == [(3, 4, 2)] * 2
    with pytest.raises(RuntimeError, match="lacks 1"):
        sb._state_cached_samples(cfg, {}, None, None, [(2015, 3, 5)], (6, 12), 1,
                                 _times(), None, None, None, "none", None)


def _builder():
    path = Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_wb_state_cache.py"
    spec = importlib.util.spec_from_file_location("build_wb_state_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builder_needs_train_leads_and_validation_times():
    mod = _builder()
    assert mod.build_parser().parse_args(
        ["--config", "x", "--mode", "sfno", "--task", "1", "--n-tasks", "4"]).task == 1
    times = np.datetime64("2014-01-01") + np.arange(4 * 800) * np.timedelta64(6, "h")
    yml = {"train_windows": [[2015, 0, 1]], "eval_years": [2020],
           "loss": {"train_leads_hours": [6, 12]},
           "validation": {"years": [2014], "n_inits": 2, "init_stride_hours": 360}}
    cfg = SimpleNamespace(smoke=False, windows=None, n_days=None, multi_step_hours=None)
    idx = mod.needed_indices(cfg, yml, times)
    base15 = 365 * 4
    assert set(range(base15, base15 + 6)) <= set(idx.tolist())     # 4 ICs + 2 leads
    assert {0, 1, 2, 60, 61, 62} <= set(idx.tolist())               # 2014 inits + leads
    assert len(idx) == 6 + 6


def test_validation_init_that_cannot_be_scored_is_an_error():
    yml = {"eval_years": [2020], sb._VAL_INITS_KEY: [[2019, 8754]]}
    times = np.datetime64("2019-01-01") + np.arange(4 * 366) * np.timedelta64(6, "h")
    with pytest.raises(ValueError, match="validation init 2019"):
        list(sb._training_sample_indices(SimpleNamespace(smoke=False), yml, times, 4, 4))


def test_validation_init_must_exist_in_the_store():
    yml = {sb._VAL_INITS_KEY: [[2014, 0]]}
    times = np.datetime64("2015-01-01") + np.arange(40) * np.timedelta64(6, "h")
    with pytest.raises(ValueError, match="not in the ERA5 store"):
        list(sb._training_sample_indices(SimpleNamespace(smoke=False), yml, times, 4, 4))
