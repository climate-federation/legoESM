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
               "sfc_shf": np.full(12, float(i)),
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



def test_extending_a_store_keeps_written_times(tmp_path):
    p = _store(tmp_path, [10, 12])
    wsc.write_time(p, 12, wsc.record_arrays(*_rec(12))[0])
    arrays, structure = wsc.record_arrays(*_rec(10))
    wsc.create_store(p, [10, 11, 13], arrays, structure, {})   # adds 11, 13; keeps 12
    c = wsc.StateCache(p, _rec(10))
    assert c.missing([10, 11, 12, 13]) == [10, 11, 13]
    wsc.write_time(p, 13, wsc.record_arrays(*_rec(13))[0])
    c = wsc.StateCache(p, _rec(10))
    for i in (12, 13):
        np.testing.assert_array_equal(
            c.read(i)[0].p_s, _rec(i)[0].p_s.astype(np.float32).astype(np.float64))
    with pytest.raises(RuntimeError, match="never written"):
        c.read(11)
    with pytest.raises(RuntimeError, match="different structure"):
        wsc.create_store(p, [10], arrays, structure, {"other": 1})

def _gate(tmp_path, edit):
    p = _store(tmp_path, [10])
    arr = {k: v.copy() for k, v in wsc.record_arrays(*_rec(10))[0].items()}
    edit(arr)
    wsc.write_time(p, 10, arr)
    wsc.assert_cache_matches(_rec(10), wsc.StateCache(p, _rec(10)).read(10))


def test_gate_tolerates_gpu_rebuild_noise_only(tmp_path):
    eps = float(np.finfo(np.float32).eps)

    def shift(k):          # move element 0 by k float32 eps x the array's max |value|
        def edit(arr):
            a = arr["c000"]
            a.flat[0] += np.float32(k * eps * np.max(np.abs(a)))
        return edit
    _gate(tmp_path / "a", shift(2))       # rebuild-sized noise: passes
    with pytest.raises(RuntimeError, match="stale"):
        _gate(tmp_path / "b", shift(32))  # a real change fails

    def nan_flip(arr):
        arr["f000"].flat[np.flatnonzero(~np.isnan(arr["f000"]))[0]] = np.nan
    with pytest.raises(RuntimeError, match="stale"):
        _gate(tmp_path / "c", nan_flip)

    def inf_at_0(arr):     # cached inf where the fresh build is finite
        arr["c000"].flat[0] = np.inf
    with pytest.raises(RuntimeError, match="stale"):
        _gate(tmp_path / "d", inf_at_0)


def test_gate_catches_structure(tmp_path):
    p = _store(tmp_path, [10])
    wsc.write_time(p, 10, wsc.record_arrays(*_rec(10))[0])
    other = (_rec(10)[0], {"t2m": _rec(10)[1]["t2m"]}, _rec(10)[2])
    with pytest.raises(RuntimeError, match="different sample structure"):
        wsc.StateCache(p, other)


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
    s = sb._state_cached_samples(cfg, {}, None, None, mine, (6, 12), (1, 2),
                                 _times(), None, None, None, "none", None)
    assert len(s) == 2
    ic, target, forcing = s[1]
    assert len(target) == 2 and set(target[0][1]) == {"t2m", "u10"}
    np.testing.assert_array_equal(target[1][0].p_s,
                                  _rec(4)[0].p_s.astype(np.float32).astype(np.float64))
    assert [x[0].T.shape for x in s] == [(3, 4, 2)] * 2
    sp = sb._state_cached_samples(cfg, {}, None, None, [(2015, 0, 4)], (6, 12, 24),
                                  (1, 2, 4), _times(), None, None, None, "none", None)
    np.testing.assert_array_equal(  # third lead is time 4, not 3
        sp[0][1][2][0].p_s, _rec(4)[0].p_s.astype(np.float32).astype(np.float64))
    with pytest.raises(RuntimeError, match="lacks 1"):
        sb._state_cached_samples(cfg, {}, None, None, [(2015, 3, 5)], (6, 12), (1, 2),
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

    sparse = {**yml, "loss": {"train_leads_hours": [6, 12, 24]}}
    idx = set(mod.needed_indices(cfg, sparse, times).tolist())
    # 4 ICs at base15+0..3, leads +1, +2, +4: offset 3 (18 h) only as another IC.
    assert idx == ({base15 + k for k in range(4)} | {base15 + k + o for k in range(4)
                                                     for o in (1, 2, 4)}
                   | {0, 1, 2, 4, 60, 61, 62, 64})


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


def test_deck_without_validation_block_validates_on_2014():
    v = sb.wb_validation_yaml({"train_years": [2015], "eval_years": [2020]}, 24)
    assert v["train_years"] == [2014]
    assert v[sb._VAL_INITS_KEY] == [[2014, 360 * k] for k in range(24)]
    with pytest.raises(ValueError, match="n_inits"):
        sb.wb_validation_yaml({"train_years": [2015], "eval_years": [2020]}, 6)


def test_soil_moisture_plane_and_land_inputs_flag():
    import jax.numpy as jnp
    from types import SimpleNamespace as NS
    assert sb.wb_needs_land_frac("sfno", {"sfno": {"land_inputs": True}})
    assert not sb.wb_needs_land_frac("sfno", {"sfno": {}})
    c = sb.wb_era5_config(NS(mode="sfno"), {"era5_zarr": "x", "era5_soil_moisture": True})
    assert c.load_soil_moisture
    e, f, _ = sb.wb_state_cache_configs(NS(mode="sfno"), {"era5_zarr": "x",
                                                          "era5_soil_moisture": True})
    assert f.load_soil_moisture and not e.load_soil_moisture


def test_empty_validation_block_is_an_error_and_builder_always_validates():
    with pytest.raises(ValueError, match="needs years"):
        sb.wb_validation_yaml({"validation": {}}, 24)
    mod = _builder()
    times = np.datetime64("2014-01-01") + np.arange(4 * 800) * np.timedelta64(6, "h")
    cfg = SimpleNamespace(smoke=False, windows=None, n_days=None, multi_step_hours=None)
    idx = mod.needed_indices(cfg, {"train_windows": [[2015, 0, 1]], "eval_years": [2020]},
                             times)
    assert {0, 60 * 23} <= set(idx.tolist())     # implicit 2014 validation inits


def test_every_lead_uses_the_initial_time_forcing(tmp_path, monkeypatch):
    """A multi-lead sample carries ONE forcing: the initial time's record."""
    monkeypatch.setattr(sb, "wb_state_record", lambda i, *a: _rec(i))
    monkeypatch.setattr(sb, "wb_state_cache_configs", lambda cfg, yml: (None, None, "none"))
    monkeypatch.setattr(sb, "validate_carry_holds_scheme", lambda *a, **k: 3)
    p = _store(tmp_path, [0, 1, 2, 3, 4])
    monkeypatch.setattr(sb, "wb_state_cache_path", lambda *a: p)
    for i in range(5):
        wsc.write_time(p, i, wsc.record_arrays(*_rec(i))[0])
    s = sb._state_cached_samples(SimpleNamespace(mode="physics"), {}, None, None,
                                 [(2015, 2, 4)], (6, 12), (1, 2), _times(), None, None,
                                 None, "none", None)
    _, target, forcing = s[0]
    assert float(forcing["day_of_year"]) == 2.0          # _rec(i) codes i here
    np.testing.assert_array_equal(forcing["sfc_shf"], 2.0)   # IC fluxes, not a lead's
    assert len(target) == 2


def test_task_shares_cover_every_needed_time_once_whatever_is_done():
    mod = _builder()
    idx = np.arange(100, 120)
    for done_set in (set(), {101, 105, 106, 119}):
        done = {int(i): int(i) in done_set for i in idx}
        shares = [mod.task_share(idx, done, t, 3) for t in range(3)]
        got = sorted(int(idx[k]) for sh in shares for k in sh)
        assert got == sorted(set(idx.tolist()) - done_set)
    # Tasks that start one after another, each seeing the previous ones'
    # writes, still cover every time (a split of only the unwritten times
    # would shift and skip some).
    written = set()
    for t in range(3):
        done = {int(i): int(i) in written for i in idx}
        written |= {int(idx[k]) for k in mod.task_share(idx, done, t, 3)}
    assert written == set(idx.tolist())
