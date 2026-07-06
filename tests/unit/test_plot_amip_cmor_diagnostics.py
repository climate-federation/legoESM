"""Tests for scripts/plot/plot_amip_cmor_diagnostics.py — the systematic AMIP
CMOR-Amon diagnostics plotter.  Verifies the pure compute core (area-weighted
global means + TOA budget from synthetic Amon files) and that the figure
renders, so the plot path can't silently rot."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_MOD_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "plot" / "plot_amip_cmor_diagnostics.py")
_spec = importlib.util.spec_from_file_location("plot_amip_cmor_diagnostics", _MOD_PATH)
plotmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plotmod)


def _write_amon(cmor_amon: pathlib.Path, var: str, field2d: np.ndarray, lat, lon):
    """Write a minimal CMOR-Amon-style NetCDF: (time=1, lat, lon)."""
    cmor_amon.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(field2d[None, :, :], dims=("time", "lat", "lon"),
                      coords={"time": [0.0], "lat": lat, "lon": lon}, name=var)
    da.to_dataset().to_netcdf(cmor_amon / f"{var}_Amon_test_gn.nc")


def test_weights_sum_to_one_and_are_nonuniform():
    w = plotmod._sinlat_area_weights(18, 36)
    assert w.shape == (18, 36)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    # polar rows carry LESS area than equatorial rows (sin-latitude bands).
    assert w[0, 0] < w[9, 0]


def test_constant_field_global_mean_is_the_constant(tmp_path):
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(5, 355, nlon)
    cmor = tmp_path / "cmor" / "Amon"
    _write_amon(cmor, "tas", np.full((nlat, nlon), 288.0), lat, lon)
    diag = plotmod.compute_amip_diagnostics(tmp_path)
    gm, ref, unit = diag["global_means"]["tas"]
    assert gm == pytest.approx(288.0, abs=1e-9)   # weights sum to 1
    assert unit == "K"


def test_hemispheric_field_area_weighted_mean_is_half(tmp_path):
    """A field that is 1 in the SH and 0 in the NH integrates to 0.5 under the
    (hemispherically symmetric) sin-latitude weights — the exactness check."""
    nlat, nlon = 40, 20
    lat = np.linspace(-88.0, 88.0, nlat)
    lon = np.linspace(0, 342, nlon)
    field = np.where(lat[:, None] < 0.0, 1.0, 0.0) * np.ones((1, nlon))
    cmor = tmp_path / "cmor" / "Amon"
    _write_amon(cmor, "tas", field, lat, lon)
    diag = plotmod.compute_amip_diagnostics(tmp_path)
    assert diag["global_means"]["tas"][0] == pytest.approx(0.5, abs=1e-2)


def test_toa_budget_and_albedo(tmp_path):
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(5, 355, nlon)
    cmor = tmp_path / "cmor" / "Amon"
    _write_amon(cmor, "tas", np.full((nlat, nlon), 288.0), lat, lon)
    _write_amon(cmor, "rsdt", np.full((nlat, nlon), 340.0), lat, lon)
    _write_amon(cmor, "rsut", np.full((nlat, nlon), 100.0), lat, lon)
    _write_amon(cmor, "rlut", np.full((nlat, nlon), 239.0), lat, lon)
    diag = plotmod.compute_amip_diagnostics(tmp_path)
    b = diag["budget"]
    assert b["R_TOA"] == pytest.approx(340.0 - 100.0 - 239.0, abs=1e-6)
    assert b["albedo"] == pytest.approx(100.0 / 340.0, abs=1e-6)


def test_missing_tas_raises(tmp_path):
    (tmp_path / "cmor" / "Amon").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        plotmod.compute_amip_diagnostics(tmp_path)


def test_figure_renders(tmp_path):
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(5, 355, nlon)
    cmor = tmp_path / "cmor" / "Amon"
    for v, val in [("tas", 288.0), ("pr", 3.0e-5), ("rsut", 100.0),
                   ("rsdt", 340.0), ("rlut", 239.0), ("clt", 60.0)]:
        _write_amon(cmor, v, np.full((nlat, nlon), val), lat, lon)
    out = plotmod.plot_amip_cmor_diagnostics(tmp_path, "unit", tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 0


def _diag(global_means, budget=None):
    """Minimal diagnostics dict — only the keys the scorecard reads."""
    return {"global_means": dict(global_means), "budget": budget}


def _at_ref(*vars):
    """global_means entries sitting exactly on their Earth reference."""
    refs = {v: (r, u) for v, _s, u, r, _c in plotmod.FIELD_TABLE}
    return {v: (refs[v][0], refs[v][0], refs[v][1]) for v in vars}


def test_scorecard_passes_when_all_within_band():
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    sc = plotmod.amip_realism_scorecard(
        _diag(gm, budget={"albedo": 0.29, "R_TOA": 0.0}))
    assert sc["passed"] is True
    assert sc["missing_required"] == []
    assert sc["n_pass"] == sc["n_checks"] == 6   # 4 fields + albedo + R_TOA
    assert all(d["within"] for d in sc["fields"].values())


def test_scorecard_fails_field_out_of_band():
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    gm["tas"] = (288.0 + 10.0, 288.0, "K")       # 10 K bias >> 4 K band
    sc = plotmod.amip_realism_scorecard(
        _diag(gm, budget={"albedo": 0.29, "R_TOA": 0.0}))  # budget OK -> isolate tas
    assert sc["passed"] is False
    assert sc["fields"]["tas"]["within"] is False
    assert sc["fields"]["pr"]["within"] is True
    assert sc["missing_required"] == []
    assert "out_of_band=tas" in plotmod.format_scorecard_line(sc)
    assert plotmod.format_scorecard_line(sc).startswith("FAIL")


def test_scorecard_missing_required_field():
    gm = _at_ref("tas", "pr", "rlut")            # rsut absent
    sc = plotmod.amip_realism_scorecard(
        _diag(gm, budget={"albedo": 0.29, "R_TOA": 0.0}))  # budget OK -> only rsut missing
    assert sc["missing_required"] == ["rsut"]
    assert sc["passed"] is False
    assert "missing_required=rsut" in plotmod.format_scorecard_line(sc)


def test_scorecard_requires_toa_budget():
    """A run with all required fields but NO TOA budget must NOT silently pass —
    the budget triplet (rsdt/rsut/rlut) is a required check (codex HIGH #1)."""
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    sc = plotmod.amip_realism_scorecard(_diag(gm, budget=None))
    assert "budget" in sc["missing_required"]
    assert sc["passed"] is False
    assert sc["budget"] is None
    assert "missing_required=budget" in plotmod.format_scorecard_line(sc)


def test_scorecard_partial_abs_tol_override_keeps_required_bands():
    """A partial ``abs_tol`` override must MERGE onto defaults, not replace them,
    so an un-overridden required field is still checked (codex HIGH #2)."""
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    gm["rsut"] = (100.0 + 50.0, 100.0, "W/m2")   # 50 >> default 12 band
    sc = plotmod.amip_realism_scorecard(
        _diag(gm, budget={"albedo": 0.29, "R_TOA": 0.0}),
        abs_tol={"tas": 1.0})                    # override only tas
    assert "rsut" in sc["fields"]                # still checked, not dropped
    assert sc["fields"]["rsut"]["within"] is False
    assert sc["passed"] is False


def test_scorecard_tol_scale_nonpositive_raises():
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            plotmod.amip_realism_scorecard(_diag(gm), tol_scale=bad)


def test_scorecard_malformed_budget_is_not_keyerror():
    """A budget dict missing a key is treated as absent, not a crash (codex MED)."""
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    sc = plotmod.amip_realism_scorecard(_diag(gm, budget={"albedo": 0.29}))  # no R_TOA
    assert "budget" in sc["missing_required"]
    assert sc["budget"] is None
    assert sc["passed"] is False


def test_scorecard_nan_field_fails_and_serializes(tmp_path):
    """A blown-up run (NaN global mean) scores within=False and still writes
    VALID JSON (NaN -> null), not the non-standard ``NaN`` token (codex MED)."""
    import json
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    gm["tas"] = (float("nan"), 288.0, "K")
    sc = plotmod.amip_realism_scorecard(
        _diag(gm, budget={"albedo": 0.29, "R_TOA": 0.0}))
    assert sc["fields"]["tas"]["within"] is False
    assert sc["passed"] is False
    p = plotmod.write_scorecard(sc, tmp_path / "nan.json")
    loaded = json.loads(p.read_text())           # would raise if NaN token written
    assert loaded["fields"]["tas"]["value"] is None


def test_scorecard_budget_albedo_and_r_toa_bands():
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    sc = plotmod.amip_realism_scorecard(
        _diag(gm, budget={"albedo": 0.29 + 0.05, "R_TOA": 2.0}))
    assert sc["budget"]["albedo"]["within"] is False   # 0.05 > 0.03 band
    assert sc["budget"]["r_toa"]["within"] is True      # 2.0 < 5.0 band
    assert sc["passed"] is False


def test_scorecard_tol_scale_widens_band():
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    gm["tas"] = (288.0 + 5.0, 288.0, "K")        # 5 K: outside 4 K, inside 8 K
    assert plotmod.amip_realism_scorecard(_diag(gm))["fields"]["tas"]["within"] is False
    wide = plotmod.amip_realism_scorecard(_diag(gm), tol_scale=2.0)
    assert wide["fields"]["tas"]["within"] is True


def test_write_scorecard_json_roundtrips(tmp_path):
    import json
    gm = _at_ref("tas", "pr", "rsut", "rlut")
    sc = plotmod.amip_realism_scorecard(_diag(gm, budget={"albedo": 0.29, "R_TOA": 0.0}))
    p = plotmod.write_scorecard(sc, tmp_path / "score.json")
    assert p.exists()
    assert json.loads(p.read_text()) == sc   # pure JSON scalars/lists — exact


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
