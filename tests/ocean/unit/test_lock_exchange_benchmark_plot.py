"""Direct test for scripts/plot/plot_lock_exchange_benchmark.py."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]


def _load():
    p = _REPO / "scripts" / "plot" / "plot_lock_exchange_benchmark.py"
    spec = importlib.util.spec_from_file_location("_lockex_plot", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lockex_plot"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_arm(root: Path, grid: str, res: str) -> None:
    d = root / grid / "lock_exchange" / grid / res
    d.mkdir(parents=True)
    nt, nlat, nlon, nlev = 3, 8, 12, 4
    rng = np.random.default_rng(0)
    T3 = 5.0 + 25.0 * rng.random((nt, nlat, nlon, nlev))
    np.savez(d / "snapshots_latlon.npz",
             times_days=np.array([0.0, 1.0, 5.0]),
             lat=np.linspace(-80, 80, nlat), lon=np.linspace(0, 355, nlon),
             T_3d=T3, SST=T3[..., 0],
             land_mask=np.ones((nt, nlat, nlon)))
    with open(d / "mean_timeseries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "time_days", "RPE_rel", "T_min", "T_max"])
        w.writeheader()
        for t, r in [(0.0, 0.0), (1.0, 1e-8), (5.0, 2e-8)]:
            w.writerow({"time_days": t, "RPE_rel": r,
                        "T_min": 5.0, "T_max": 30.0})
    (d / "results.txt").write_text("status: PASS\n")


def test_build_figure_writes_png(tmp_path):
    mod = _load()
    for grid, res, _c, _tag in mod.ARMS:
        _fake_arm(tmp_path, grid, res)
    out = mod.build_figure(tmp_path, tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_masked_section_nans_land_columns():
    """Land columns must be NaN in the section (structured arms pin land
    tracers at 0 C; painting them as 0-degree water is the land-0
    artifact the matrix gates had)."""
    mod = _load()
    nt, nlat, nlon, nlev = 2, 4, 6, 3
    T3 = np.full((nt, nlat, nlon, nlev), 15.0)
    mask = np.ones((nlat, nlon))
    mask[:, 2] = 0.0                      # land strip
    sec = mod._masked_section(T3, mask, it=1, j=1)
    assert sec.shape == (nlev, nlon)
    assert np.all(np.isnan(sec[:, 2]))
    assert np.all(sec[:, 3] == 15.0)


def test_benchmark_depths_are_the_stretched_coordinate():
    """Depths must come from the ACTUAL matrix z-star coordinate, not
    uniform k+0.5 indices (dz_ref is 0.095..1.905 m at nlev=20/H_max=20;
    the uniform assumption mislabels the section by up to ~0.9 m)."""
    mod = _load()
    d = mod._benchmark_depths(20)
    assert d.shape == (20,)
    assert np.all(np.diff(d) > 0) and 0.0 < d[0] < 20.0 and d[-1] < 20.0
    uniform = np.arange(20) + 0.5
    assert np.max(np.abs(d - uniform)) > 0.3


def test_missing_arm_is_hard_error(tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(sys, "argv",
                        ["x", "--runs-root", str(tmp_path),
                         "--out", str(tmp_path / "f.png")])
    import pytest
    with pytest.raises(SystemExit, match="missing arms"):
        mod.main()
