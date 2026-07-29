"""Smoke tests for the big-leaf vs multi-layer canopy comparison plotters.

``scripts/plot/plot_ec_site_canopy_comparison.py`` (per-site) and
``..._summary.py`` (pooled) read two ``run_ec_site.py`` outputs for one site/
window plus the driver (for closure-corrected ET_CORR/H_CORR) and render a PNG.
These build tiny synthetic datasets — no model run — and assert a non-empty PNG
is produced, so a refactor that breaks the plotting contract fails fast.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import xarray as xr
import pytest

matplotlib = pytest.importorskip("matplotlib")

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PLOT = _REPO / "scripts" / "plot"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _PLOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_model_nc(path, n, seed):
    """Minimal run_ec_site-style output: mod/obs fluxes + score_valid."""
    rng = np.random.default_rng(seed)
    hour = (np.arange(n) * 0.5) % 24.0
    day = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    def series(scale):
        return ("time", scale * day + rng.normal(0, 0.05 * scale + 1e-3, n))
    ds = xr.Dataset(
        {
            "le_mod": series(200.0), "le_obs": series(180.0),
            "h_mod": series(120.0), "h_obs": series(100.0),
            "gpp_mod": series(15.0), "gpp_obs": series(14.0),
            "score_valid": ("time", np.ones(n, dtype="i1")),
        },
        coords={"time": np.arange(n)},
    )
    ds.to_netcdf(path)


def _make_driver_nc(path, n, with_corr=True):
    hour = (np.arange(n) * 0.5) % 24.0
    day = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    data = {"ET_CORR": ("time", (6.0 * day if with_corr else np.full(n, np.nan))),
            "H_CORR": ("time", (110.0 * day if with_corr else np.full(n, np.nan)))}
    ds = xr.Dataset(data, coords={"time": (np.datetime64("2015-07-01T00:00")
                                           + np.arange(n) * np.timedelta64(30, "m"))})
    ds.to_netcdf(path)


@pytest.mark.parametrize("with_corr", [True, False])
def test_per_site_comparison_writes_png(tmp_path, with_corr):
    n = 96
    big = str(tmp_path / "big.nc"); ml = str(tmp_path / "ml.nc")
    drv = str(tmp_path / "drv.nc"); out = str(tmp_path / "cmp.png")
    _make_model_nc(big, n, 1); _make_model_nc(ml, n, 2)
    _make_driver_nc(drv, n, with_corr)
    mod = _load("plot_ec_site_canopy_comparison")
    mod.main(big, ml, drv, 0, n, out, "SYN")
    assert pathlib.Path(out).stat().st_size > 0


def test_pooled_summary_writes_png(tmp_path):
    n = 96
    runs = []
    for i, site in enumerate(("A", "B")):
        big = str(tmp_path / f"{site}_big.nc"); ml = str(tmp_path / f"{site}_ml.nc")
        drv = str(tmp_path / f"{site}_drv.nc")
        _make_model_nc(big, n, i); _make_model_nc(ml, n, i + 10)
        _make_driver_nc(drv, n, True)
        runs.append([site, big, ml, drv, 0, n])
    out = str(tmp_path / "summary.png")
    mod = _load("plot_ec_site_canopy_summary")
    mod.main(runs, out)
    assert pathlib.Path(out).stat().st_size > 0


def test_comparison_rejects_mismatched_masks(tmp_path):
    """A controlled comparison requires identical scoring masks on both arms."""
    n = 48
    big = str(tmp_path / "big.nc"); ml = str(tmp_path / "ml.nc")
    drv = str(tmp_path / "drv.nc")
    _make_model_nc(big, n, 1); _make_model_nc(ml, n, 2)
    # corrupt the multi-layer arm's mask so it differs from big-leaf
    d = xr.open_dataset(ml); v = d["score_valid"].values.copy(); v[0] = 0
    d["score_valid"] = ("time", v); d.to_netcdf(ml + ".tmp")
    import os; os.replace(ml + ".tmp", ml)
    _make_driver_nc(drv, n, True)
    mod = _load("plot_ec_site_canopy_comparison")
    with pytest.raises(SystemExit, match="controlled"):
        mod.main(big, ml, drv, 0, n, str(tmp_path / "x.png"), "SYN")
