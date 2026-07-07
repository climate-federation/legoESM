"""Direct tests for scripts/plot/plot_convection_scorecard.py (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "plot" / "plot_convection_scorecard.py")
_spec = importlib.util.spec_from_file_location("plot_convection_scorecard", _SCRIPT)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def _write_ts(run_dir: Path, *, precip, rsut, rlut, hfls=88.0, hfss=20.0,
              cwv=24.5, tlow=288.0, rsdt=340.0, n=12):
    run_dir.mkdir(parents=True, exist_ok=True)
    days = np.linspace(5.0, 90.0, n)
    one = np.ones(n)
    np.savez(
        run_dir / "timeseries.npz",
        days=days, precip=one * precip, sw_up_toa=one * rsut,
        lw_up_toa=one * rlut, hfls=one * hfls, hfss=one * hfss,
        CWV=one * cwv, T_low=one * tlow, rsdt=one * rsdt,
    )


def test_realistic_run_passes_all_bands(tmp_path):
    # values inside every acceptance band (albedo = 100/340 = 0.294)
    _write_ts(tmp_path / "conv_sbm", precip=2.9, rsut=100.0, rlut=239.0)
    s = sc.score_run(tmp_path / "conv_sbm")
    assert s is not None
    assert s["metrics"]["pr"]["pass"]
    assert s["metrics"]["rsut"]["pass"]
    assert s["metrics"]["albedo"]["pass"]
    assert s["metrics"]["R_TOA"]["pass"]           # 340-100-239 = +1, |1|<=5
    assert s["n_pass"] == s["n_graded"]


def test_overbright_dry_run_fails_expected_metrics(tmp_path):
    # the tiedtke failure mode: no rain, too bright, low LH flux
    _write_ts(tmp_path / "conv_tiedtke", precip=0.24, rsut=240.0, rlut=175.0,
              hfls=18.0)
    s = sc.score_run(tmp_path / "conv_tiedtke")
    assert not s["metrics"]["pr"]["pass"]          # 0.24 vs 2.9
    assert not s["metrics"]["rsut"]["pass"]        # 240 vs 100
    assert not s["metrics"]["albedo"]["pass"]      # 240/340 = 0.71
    assert not s["metrics"]["hfls"]["pass"]        # 18 vs 88
    assert s["n_pass"] < s["n_graded"]


def test_ranking_orders_realistic_above_broken(tmp_path):
    _write_ts(tmp_path / "conv_sbm", precip=2.9, rsut=100.0, rlut=239.0)
    _write_ts(tmp_path / "conv_tiedtke", precip=0.24, rsut=240.0, rlut=175.0,
              hfls=18.0)
    scores = {d.name.replace("conv_", ""): sc.score_run(d)
              for d in tmp_path.glob("conv_*")}
    ranked = sc.rank_scheme_scores(scores)
    assert ranked[0][0] == "sbm"                   # more passes wins
    assert ranked[-1][0] == "tiedtke"


def test_missing_timeseries_returns_none(tmp_path):
    (tmp_path / "conv_empty").mkdir()
    assert sc.score_run(tmp_path / "conv_empty") is None


def test_main_writes_png_and_names_best(tmp_path, capsys):
    _write_ts(tmp_path / "conv_sbm", precip=2.9, rsut=100.0, rlut=239.0)
    _write_ts(tmp_path / "conv_bechtold", precip=0.21, rsut=226.0, rlut=229.0,
              hfls=31.0)
    out = tmp_path / "scorecard.png"
    rc = sc.main([str(tmp_path), "--out", str(out)])
    assert rc == 0 and out.exists()
    printed = capsys.readouterr().out
    assert "Most realistic: 'sbm'" in printed
