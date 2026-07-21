"""Direct tests for the cross-node cube shard_map scaling aggregator (pure Python,
no JAX). Covers every gate branch: baseline present/missing, ppermute
True/False/None, config consistency, duplicate counts, decisive efficiency
pass/fail, malformed input, and CSV/report/verdict emission."""

from __future__ import annotations

import json

import pytest

import scripts.bench.aggregate_cube_shardmap_scaling as agg


def _pt(tmp, n, ms, coll, *, n_grid=12, n_lev=4, prec="float64", backend="cpu", procs=None):
    d = tmp / f"n{n}"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "mode": "point", "n_devices": n, "ms_per_step": ms,
        "spmd_hlo_collective": coll, "process_count": procs if procs is not None else n,
        "process_index": 0,
        "config": {"n_grid": n_grid, "n_lev": n_lev, "dt": 450.0,
                   "precision": prec, "backend": backend,
                   "n_warmup": 1, "n_timing": 3},
    }))
    return d / "results.json"


def _agg(tmp, pass_eff=0.6, decisive=False, required=None):
    paths = sorted((tmp).glob("*/results.json"))
    return agg.aggregate(agg._load_points(paths), pass_eff, decisive, required)


# --- Curve math + happy path -----------------------------------------------
def test_curve_speedup_and_efficiency(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, True)
    _pt(tmp_path, 3, 5.0, True)
    r = _agg(tmp_path)
    rows = {row["n_devices"]: row for row in r["curve"]}
    assert rows[1]["speedup"] == pytest.approx(1.0)
    assert rows[2]["speedup"] == pytest.approx(10.0 / 6.0)
    assert rows[3]["efficiency"] == pytest.approx((10.0 / 5.0) / 3)
    assert r["gates"]["baseline_present"] is True
    assert r["gates"]["ppermute"] is True
    assert r["overall_pass"] is True


# --- Baseline gate ----------------------------------------------------------
def test_missing_baseline_fails(tmp_path):
    _pt(tmp_path, 2, 6.0, True)
    _pt(tmp_path, 3, 5.0, True)
    r = _agg(tmp_path)
    assert r["gates"]["baseline_present"] is False
    assert r["overall_pass"] is False
    assert any("baseline" in x for x in r["reasons"])
    # speed-up undefined without a baseline
    assert all(row["speedup"] is None for row in r["curve"])


# --- ppermute proof gate ----------------------------------------------------
def test_ppermute_false_fails(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, True)
    _pt(tmp_path, 3, 5.0, False)   # a >1 point that did NOT lower ppermute
    r = _agg(tmp_path)
    assert r["gates"]["ppermute"] is False
    assert r["overall_pass"] is False


def test_ppermute_none_fails(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, None)    # probe unsupported -> must not pass
    r = _agg(tmp_path)
    assert r["gates"]["ppermute"] is False
    assert r["overall_pass"] is False


# --- Config consistency -----------------------------------------------------
def test_config_mismatch_fails(tmp_path):
    _pt(tmp_path, 1, 10.0, None, n_grid=12)
    _pt(tmp_path, 2, 6.0, True, n_grid=24)   # different problem size
    r = _agg(tmp_path)
    assert r["gates"]["config_consistent"] is False
    assert r["overall_pass"] is False
    assert any("n_grid" in x for x in r["reasons"])


def test_config_mismatch_precision_fails(tmp_path):
    _pt(tmp_path, 1, 10.0, None, prec="float32")
    _pt(tmp_path, 2, 6.0, True, prec="float64")
    assert _agg(tmp_path)["gates"]["config_consistent"] is False


def test_config_mismatch_backend_fails(tmp_path):
    # A cpu baseline mixed with a gpu point would fake superlinear efficiency.
    _pt(tmp_path, 1, 10.0, None, backend="cpu")
    _pt(tmp_path, 2, 6.0, True, backend="gpu")
    assert _agg(tmp_path)["gates"]["config_consistent"] is False


# --- Required cross-node counts (the decisive claim names them) -------------
def test_require_counts_present_passes(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, True)
    _pt(tmp_path, 3, 5.0, True)
    r = _agg(tmp_path, required=[1, 2, 3])
    assert r["gates"]["required_counts_present"] is True
    assert r["overall_pass"] is True


def test_require_counts_missing_fails(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, True)   # missing the required n=6 (the actual cliff point)
    r = _agg(tmp_path, decisive=True, pass_eff=0.0, required=[1, 2, 6])
    assert r["gates"]["required_counts_present"] is False
    assert r["overall_pass"] is False       # even though efficiency@2 clears 0.0
    assert any("required device counts absent" in x for x in r["reasons"])


def test_lone_baseline_nondecisive_fails(tmp_path):
    _pt(tmp_path, 1, 10.0, None)             # nothing to scale
    r = _agg(tmp_path)
    assert r["gates"]["has_multi_point"] is False
    assert r["overall_pass"] is False


# --- Duplicate device counts are ambiguous ----------------------------------
def test_duplicate_counts_raise(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    (tmp_path / "n1b").mkdir()
    (tmp_path / "n1b" / "results.json").write_text(json.dumps({
        "mode": "point", "n_devices": 1, "ms_per_step": 9.0,
        "spmd_hlo_collective": None, "process_count": 1, "process_index": 0,
        "config": {"n_grid": 12, "n_lev": 4, "dt": 450.0, "precision": "float64",
                   "backend": "cpu", "n_warmup": 1, "n_timing": 3},
    }))
    with pytest.raises(ValueError, match="duplicate n_devices"):
        _agg(tmp_path)


# --- Decisive efficiency gate ----------------------------------------------
def test_decisive_efficiency_pass(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, True)
    _pt(tmp_path, 3, 5.0, True)   # eff at 3 = 0.667 >= 0.6
    r = _agg(tmp_path, pass_eff=0.6, decisive=True)
    assert r["gates"]["efficiency"] is True
    assert r["overall_pass"] is True


def test_decisive_efficiency_fail(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 9.0, True)
    _pt(tmp_path, 3, 10.0, True)  # eff at 3 = 0.33 < 0.6
    r = _agg(tmp_path, pass_eff=0.6, decisive=True)
    assert r["gates"]["efficiency"] is False
    assert r["overall_pass"] is False


def test_decisive_requires_multi_device(tmp_path):
    _pt(tmp_path, 1, 10.0, None)   # baseline only
    r = _agg(tmp_path, decisive=True)
    assert r["overall_pass"] is False
    assert any("n_devices > 1" in x for x in r["reasons"])


# --- Malformed input --------------------------------------------------------
def test_missing_keys_raises(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"mode": "sweep", "scaling": []}))
    with pytest.raises(ValueError, match="not a point"):
        agg._load_points([d / "results.json"])


def test_wrong_mode_raises(tmp_path):
    # All required keys present, but mode != point -> the mode branch must fire.
    d = tmp_path / "bad"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "mode": "sweep", "n_devices": 1, "ms_per_step": 1.0,
        "spmd_hlo_collective": None,
        "config": {"n_grid": 12, "n_lev": 4, "precision": "float64", "backend": "cpu"},
    }))
    with pytest.raises(ValueError, match="expected 'point'"):
        agg._load_points([d / "results.json"])


def test_config_not_dict_raises(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "mode": "point", "n_devices": 1, "ms_per_step": 1.0,
        "spmd_hlo_collective": None, "config": None,
    }))
    with pytest.raises(ValueError, match="config must be an object"):
        agg._load_points([d / "results.json"])


def test_collect_paths_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no point results"):
        agg._collect_paths(tmp_path, None)


# --- Semantic schema validation (reject garbage before gate math) ----------
def _raw_point(tmp, **overrides):
    base = {"mode": "point", "n_devices": 2, "ms_per_step": 5.0,
            "spmd_hlo_collective": True, "process_count": 2, "process_index": 0,
            "config": {"n_grid": 12, "n_lev": 4, "dt": 450.0, "precision": "float64",
                       "backend": "cpu", "n_warmup": 1, "n_timing": 3}}
    base.update(overrides)
    d = tmp / "pt"
    d.mkdir()
    (d / "results.json").write_text(json.dumps(base))
    return d / "results.json"


@pytest.mark.parametrize("bad,match", [
    ({"ms_per_step": -5.0}, "ms_per_step"),
    ({"ms_per_step": 0.0}, "ms_per_step"),
    ({"ms_per_step": "fast"}, "ms_per_step"),
    ({"n_devices": 2.5}, "n_devices"),
    ({"n_devices": 0}, "n_devices"),
    ({"spmd_hlo_collective": "yes"}, "spmd_hlo_collective"),
    ({"spmd_hlo_collective": 1}, "spmd_hlo_collective"),   # 1==True must NOT pass
    ({"process_count": -1}, "process_count"),
])
def test_semantic_validation_rejects_garbage(tmp_path, bad, match):
    path = _raw_point(tmp_path, **bad)
    with pytest.raises(ValueError, match=match):
        agg._load_points([path])


# --- Output emission + CLI exit code ---------------------------------------
def test_cli_writes_outputs_and_exit_code(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 6.0, True)
    _pt(tmp_path, 3, 5.0, True)
    out = tmp_path / "agg"
    rc = agg.main(["--root", str(tmp_path), "--gate-efficiency",
                   "--pass-efficiency", "0.6", "--output-dir", str(out)])
    assert rc == 0
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["overall_pass"] is True
    assert (out / "scaling.csv").exists()
    assert (out / "report.md").exists()
    # CSV has header + 3 rows
    rows = (out / "scaling.csv").read_text().strip().splitlines()
    assert len(rows) == 4


def test_cli_nonzero_on_gate_fail(tmp_path):
    _pt(tmp_path, 1, 10.0, None)
    _pt(tmp_path, 2, 9.0, True)
    _pt(tmp_path, 3, 10.0, True)  # eff below threshold
    out = tmp_path / "agg"
    rc = agg.main(["--root", str(tmp_path), "--gate-efficiency",
                   "--pass-efficiency", "0.6", "--output-dir", str(out)])
    assert rc == 1
