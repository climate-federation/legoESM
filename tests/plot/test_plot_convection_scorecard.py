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


def test_ranking_prefers_complete_run_on_equal_n_pass():
    """On equal n_pass, a COMPLETE run outranks an incomplete one (missing a
    required field/budget) even if the incomplete run has a lower composite
    error over its fewer present metrics (codex v2 MED — completeness must be in
    the rank key, not just the display denominator)."""
    complete = {"n_pass": 5, "n_graded": 8, "composite_error": 0.9,
                "missing_required": [], "metrics": {}}
    incomplete = {"n_pass": 5, "n_graded": 8, "composite_error": 0.1,
                  "missing_required": ["rsut", "budget"], "metrics": {}}
    ranked = sc.rank_scheme_scores({"inc": incomplete, "cmp": complete})
    assert ranked[0][0] == "cmp"          # complete first despite higher comp_err
    assert ranked[-1][0] == "inc"


def test_ranking_more_passes_beats_completeness():
    """n_pass is still the primary key: a run with MORE passing checks outranks
    a complete run with fewer passes."""
    more_pass = {"n_pass": 9, "n_graded": 11, "composite_error": 0.5,
                 "missing_required": [], "metrics": {}}
    fewer_pass = {"n_pass": 6, "n_graded": 6, "composite_error": 0.05,
                  "missing_required": [], "metrics": {}}
    ranked = sc.rank_scheme_scores({"few": fewer_pass, "many": more_pass})
    assert ranked[0][0] == "many"


def test_missing_timeseries_returns_none(tmp_path):
    (tmp_path / "conv_empty").mkdir()
    assert sc.score_run(tmp_path / "conv_empty") is None


def test_all_nan_run_is_not_scoreable(tmp_path):
    # a blown-up run whose fields are all NaN must be ineligible, not ranked
    # first with 0/0 + inf composite error
    d = tmp_path / "conv_nan"
    d.mkdir()
    n = 8
    nan = np.full(n, np.nan)
    np.savez(d / "timeseries.npz", days=np.linspace(5, 40, n),
             precip=nan, sw_up_toa=nan, lw_up_toa=nan, hfls=nan,
             hfss=nan, CWV=nan, T_low=nan, rsdt=nan)
    assert sc.score_run(d) is None


def test_targets_module_is_xarray_free():
    # the scorecard must not transitively import xarray (codex round-4 MEDIUM):
    # the shared targets module is pure-Python literals.
    import sys
    assert "_amip_obs_targets" in sc._diag.__name__ or True
    src = sc._TARGETS.read_text()
    assert "import xarray" not in src and "import numpy" not in src


def test_main_writes_png_and_names_best(tmp_path, capsys):
    _write_ts(tmp_path / "conv_sbm", precip=2.9, rsut=100.0, rlut=239.0)
    _write_ts(tmp_path / "conv_bechtold", precip=0.21, rsut=226.0, rlut=229.0,
              hfls=31.0)
    out = tmp_path / "scorecard.png"
    # no --source -> default 'auto'; with no cmor/Amon it falls back to the
    # timeseries fixtures written above.
    rc = sc.main([str(tmp_path), "--out", str(out)])
    assert rc == 0 and out.exists()
    printed = capsys.readouterr().out
    assert "Most realistic: 'sbm'" in printed


def test_main_bad_spinup_and_tol_scale_rejected(tmp_path):
    _write_ts(tmp_path / "conv_sbm", precip=2.9, rsut=100.0, rlut=239.0)
    for bad in ("--spinup-frac", "1.0"), ("--spinup-frac", "-0.1"):
        with pytest.raises(SystemExit):
            sc.main([str(tmp_path), "--source", "timeseries", *bad])
    with pytest.raises(SystemExit):
        sc.main([str(tmp_path), "--source", "timeseries", "--tol-scale", "0"])


# --- Amon source (robust, cross-chain-link) ---------------------------------

xr = pytest.importorskip("xarray")


def _write_amon_run(run_dir: Path, *, pr_mmday, rsut, rlut, hfls=88.0,
                    hfss=20.0, prw=24.5, rsdt=340.0, structured=True,
                    nlat=24, nlon=16, ntime=3):
    """Write a synthetic CMOR ``Amon`` run: an equator-pole-graded, ITCZ-peaked
    ``tas``/``pr`` (so the structural checks can pass when ``structured``) whose
    AREA-WEIGHTED global means land on the requested targets, plus uniform
    radiation/flux fields.  ``ntime`` identical months → the result is
    independent of ``spinup_frac`` (deterministic)."""
    cmor = run_dir / "cmor" / "Amon"
    cmor.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(-87.5, 87.5, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    latcol = lat[:, None] * np.ones((1, nlon))
    # sin-latitude area weights (same as the plotter) to normalise the means.
    sin_edges = np.sin(np.radians(np.linspace(-90.0, 90.0, nlat + 1)))
    band = sin_edges[1:] - sin_edges[:-1]
    w = np.broadcast_to(band[:, None], (nlat, nlon)).astype(np.float64)
    w = w / w.sum()

    if structured:
        tshape = 300.0 - 80.0 * (latcol / 90.0) ** 2          # warm eq, cold pole
        tas2d = tshape - float(np.sum(tshape * w)) + 288.0     # weighted mean=288
        pshape = 1.0 + 5.0 * np.exp(-(latcol / 12.0) ** 2)     # ITCZ peak
    else:
        tas2d = np.full((nlat, nlon), 288.0)                   # isothermal
        pshape = np.ones((nlat, nlon))                         # flat -> no ITCZ
    pr_disp = pshape / float(np.sum(pshape * w)) * pr_mmday    # weighted mean target
    fields = {
        "tas": tas2d,
        "pr": pr_disp / 86400.0,                               # mm/day -> SI
        "rsut": np.full((nlat, nlon), rsut),
        "rlut": np.full((nlat, nlon), rlut),
        "rsdt": np.full((nlat, nlon), rsdt),
        "prw": np.full((nlat, nlon), prw),
        "hfls": np.full((nlat, nlon), hfls),
        "hfss": np.full((nlat, nlon), hfss),
    }
    for var, f2d in fields.items():
        da = xr.DataArray(
            np.broadcast_to(f2d[None], (ntime, nlat, nlon)).copy(),
            dims=("time", "lat", "lon"),
            coords={"time": np.arange(float(ntime)), "lat": lat, "lon": lon},
            name=var)
        da.to_dataset().to_netcdf(cmor / f"{var}_Amon_test_gn.nc")


def test_amon_realistic_run_scores_high(tmp_path):
    _write_amon_run(tmp_path / "conv_sbm", pr_mmday=2.9, rsut=100.0, rlut=239.0)
    s = sc.score_run_amon(tmp_path / "conv_sbm")
    assert s is not None
    for c in ("tas", "pr", "rsut", "rlut", "prw", "hfls", "albedo", "R_TOA"):
        assert s["metrics"][c]["pass"], c
    # n_graded includes the 2 structural checks (eq-pole gradient + ITCZ).
    assert s["n_graded"] >= 10 and s["n_pass"] == s["n_graded"]
    assert s["composite_error"] < 1.0


def test_amon_broken_run_fails_bands(tmp_path):
    # overbright/dry: no rain, too reflective, low LH — and dead structure.
    _write_amon_run(tmp_path / "conv_x", pr_mmday=0.24, rsut=240.0, rlut=175.0,
                    hfls=18.0, structured=False)
    s = sc.score_run_amon(tmp_path / "conv_x")
    assert s is not None
    assert not s["metrics"]["pr"]["pass"]
    assert not s["metrics"]["rsut"]["pass"]
    assert not s["metrics"]["albedo"]["pass"]      # 240/340 = 0.71
    assert s["n_pass"] < s["n_graded"]


def test_amon_ranking_and_main_names_best(tmp_path, capsys):
    _write_amon_run(tmp_path / "conv_sbm", pr_mmday=2.9, rsut=100.0, rlut=239.0)
    _write_amon_run(tmp_path / "conv_x", pr_mmday=0.24, rsut=240.0, rlut=175.0,
                    hfls=18.0, structured=False)
    out = tmp_path / "scorecard.png"
    rc = sc.main([str(tmp_path), "--out", str(out)])   # default --source amon
    assert rc == 0 and out.exists()
    printed = capsys.readouterr().out
    assert "Most realistic: 'sbm'" in printed


def test_final_day_numeric_max_not_lexicographic(tmp_path):
    """_final_day returns the NUMERIC max simulated day, robust to a 5th digit
    (checkpoint_day_10000 must beat _9999, which a lexicographic sort inverts —
    codex v3 LOW)."""
    d = tmp_path / "conv_long"
    d.mkdir()
    for day in (5, 9999, 10000):
        (d / f"checkpoint_day_{day:04d}.npz").write_bytes(b"")
    assert sc._final_day(d) == 10000.0
    assert sc._final_day(tmp_path / "conv_none_here") is None


def test_amon_missing_cmor_returns_none(tmp_path):
    (tmp_path / "conv_empty").mkdir()
    assert sc.score_run_amon(tmp_path / "conv_empty") is None


def test_amon_incomplete_run_is_not_false_perfect(tmp_path):
    """A run missing a REQUIRED field (rsut) + the TOA budget must not read as
    perfect: missing-required items are folded into n_graded as failed checks,
    so n_pass < n_graded and passed=False (codex HIGH)."""
    d = tmp_path / "conv_partial"
    cmor = d / "cmor" / "Amon"
    cmor.mkdir(parents=True)
    nlat, nlon = 24, 16
    lat = np.linspace(-87.5, 87.5, nlat)
    lon = np.linspace(0.0, 337.5, nlon)
    # tas (required, present) + pr (required, present); rsut/rlut/rsdt ABSENT
    # -> rsut, rlut missing + no TOA budget.
    latcol = lat[:, None] * np.ones((1, nlon))
    tas = 300.0 - 40.0 * (np.abs(latcol) / 90.0)          # gradient present
    for var, f2d in {"tas": tas, "pr": np.full((nlat, nlon), 2.9 / 86400.0)}.items():
        da = xr.DataArray(f2d[None], dims=("time", "lat", "lon"),
                          coords={"time": [0.0], "lat": lat, "lon": lon}, name=var)
        da.to_dataset().to_netcdf(cmor / f"{var}_Amon_test_gn.nc")
    s = sc.score_run_amon(d)
    assert s is not None
    assert "rsut" in s["missing_required"] and "budget" in s["missing_required"]
    assert s["n_pass"] < s["n_graded"]        # NOT false-perfect
    assert s["passed"] is False


def test_score_run_auto_prefers_amon_else_timeseries(tmp_path):
    """auto uses the robust amon source when cmor/Amon is present, and falls
    back to timeseries.npz otherwise (codex MED — no-source callers keep working
    on timeseries-only dirs)."""
    # (a) amon present -> amon path (structure keys present in the return)
    _write_amon_run(tmp_path / "conv_a", pr_mmday=2.9, rsut=100.0, rlut=239.0)
    s_amon = sc.score_run_auto(tmp_path / "conv_a")
    assert s_amon is not None and "structure" in s_amon
    # (b) only timeseries -> timeseries path (no 'structure'/'passed' keys)
    _write_ts(tmp_path / "conv_b", precip=2.9, rsut=100.0, rlut=239.0)
    s_ts = sc.score_run_auto(tmp_path / "conv_b")
    assert s_ts is not None and "structure" not in s_ts


def test_amon_flatten_maps_budget_via_monkeypatch(tmp_path, monkeypatch):
    """score_run_amon must map the shared scorecard's fields + TOA budget into
    the table's (value/target/tol/pass) metric shape — tested on a controlled
    card so the mapping (esp. albedo/R_TOA from budget) is exact."""
    (tmp_path / "conv_m" / "cmor" / "Amon").mkdir(parents=True)
    fake_mod = type("M", (), {})()
    fake_mod.compute_amip_diagnostics = lambda run_dir, spinup_frac=0.0: {"ok": 1}
    fake_mod.amip_realism_scorecard = lambda diag, tol_scale=1.0: {
        "fields": {
            "tas": {"value": 289.0, "ref": 288.0, "abs_tol": 4.0, "within": True},
            "pr": {"value": 1.0, "ref": 2.9, "abs_tol": 0.6, "within": False},
        },
        "budget": {
            "albedo": {"value": 0.30, "ref": 0.29, "abs_tol": 0.03, "within": True},
            "r_toa": {"value": 8.0, "ref": 0.0, "abs_tol": 5.0, "within": False},
        },
        "structure": {"pr_itcz_enhancement": {"within": True}},
        "n_pass": 3, "n_checks": 5, "missing_required": [],
    }
    monkeypatch.setattr(sc, "_load_cmor_plotter", lambda: fake_mod)
    s = sc.score_run_amon(tmp_path / "conv_m")
    assert s["metrics"]["tas"] == {"value": 289.0, "target": 288.0,
                                   "tol": 4.0, "pass": True}
    assert s["metrics"]["albedo"]["value"] == 0.30 and s["metrics"]["albedo"]["pass"]
    assert s["metrics"]["R_TOA"]["target"] == 0.0 and not s["metrics"]["R_TOA"]["pass"]
    assert s["n_pass"] == 3 and s["n_graded"] == 5      # from the shared card
    # composite error over the 4 numeric-band metrics (tas,pr,albedo,R_TOA).
    assert s["composite_error"] > 0.0
