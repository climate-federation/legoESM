"""Direct tests for ``scripts/plot/plot_held_suarez_climatology.py``.

The scientific hazard this script exists to avoid is comparing an
INSTANTANEOUS zonal mean to HS94's TIME-MEAN published figures. The most
important test here is therefore that a snapshot-only run is REFUSED rather
than silently plotted.
"""
from __future__ import annotations

import importlib.util
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(_REPO_ROOT, "scripts", "plot",
                            "plot_held_suarez_climatology.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "plot_held_suarez_climatology", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _load()

_NLAT, _NLEV = 36, 26


def _npz(tmp_path, with_timemean=True, peak_u=28.0, peak_lat_idx=30,
         peak_lev_idx=6, name="run.npz"):
    lat = np.linspace(-87.5, 87.5, _NLAT)
    sigma = np.linspace(0.02, 0.98, _NLEV)
    uz = np.zeros((_NLAT, _NLEV))
    uz[peak_lat_idx, peak_lev_idx] = peak_u
    tz = np.full((_NLAT, _NLEV), 250.0)
    eke = np.column_stack([np.arange(1, 51.0), np.linspace(1e-4, 5.0, 50)])
    payload = dict(
        uz=uz * 0.5, lat_bins=lat, sigma_full=sigma, eke_t=eke,
        u=np.zeros((10, _NLEV)), v=np.zeros((10, _NLEV)),
        T=np.zeros((10, _NLEV)), p_s=np.zeros(10),
        lat=lat[:10], cos_angle=np.ones(10), sin_angle=np.zeros(10),
        sypd=32.0, wall_loop_s=3600.0, jit_s=5.0, dt_seconds=300.0,
        days=1200.0, n_cells=6144.0, nlev=float(_NLEV),
    )
    if with_timemean:
        payload.update(uz_timemean=uz, Tz_timemean=tz,
                       n_mean_samples=1000, mean_from_day=200.0)
    path = tmp_path / name
    np.savez(path, **payload)
    return str(path)


def test_snapshot_only_run_is_refused(tmp_path):
    """THE key gate: HS94 figures are time means.

    A run without --mean-from-day has only an instantaneous zonal mean, which
    is a different physical quantity. Plotting it against HS94 would be an
    invalid comparison, so it must fail loudly rather than fall back.
    """
    path = _npz(tmp_path, with_timemean=False)
    with pytest.raises(SystemExit, match="uz_timemean"):
        plot.load(path)


def test_timemean_run_loads(tmp_path):
    d = plot.load(_npz(tmp_path))
    assert "uz_timemean" in d.files


def test_jet_metrics_locate_the_peak(tmp_path):
    d = plot.load(_npz(tmp_path, peak_u=28.0, peak_lat_idx=30, peak_lev_idx=6))
    m = plot.jet_metrics(d)
    assert m["jet_u_ms"] == pytest.approx(28.0)
    assert m["jet_lat_deg"] == pytest.approx(np.linspace(-87.5, 87.5,
                                                        _NLAT)[30])
    assert m["jet_sigma"] == pytest.approx(np.linspace(0.02, 0.98, _NLEV)[6])


def test_jet_ratio_against_hs94(tmp_path):
    d = plot.load(_npz(tmp_path, peak_u=14.0))
    m = plot.jet_metrics(d)
    assert m["u_ratio_vs_hs94"] == pytest.approx(14.0 / plot.HS94_JET_MS)


def test_repo_floor_pass_and_fail(tmp_path):
    strong = plot.jet_metrics(plot.load(_npz(tmp_path, peak_u=25.0,
                                             name="a.npz")))
    weak = plot.jet_metrics(plot.load(_npz(tmp_path, peak_u=7.3,
                                           name="b.npz")))
    assert strong["passes_repo_floor"] is True
    assert weak["passes_repo_floor"] is False, (
        "a collapsed jet must not be reported as passing")


def test_perf_metrics_derive_wall_hours_and_throughput(tmp_path):
    d = plot.load(_npz(tmp_path))
    p = plot.perf_metrics(d)
    assert p["wall_hours"] == pytest.approx(1.0)
    # 1200 d at dt=300 s -> 345600 steps; 6144 columns x 26 levels.
    steps = 1200 * 86400 / 300
    expected = 6144 * 26 * steps / 3600.0 / 1e6
    assert p["mcells_per_s"] == pytest.approx(expected)


def test_make_figure_writes_png_and_pdf(tmp_path):
    d = plot.load(_npz(tmp_path))
    written = plot.make_figure(d, str(tmp_path / "out"), "fig_test")
    assert len(written) == 2
    for path in written:
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_hs94_reference_constants_are_the_published_ones():
    """Guards against someone 'tuning' the benchmark to match a weak run."""
    assert plot.HS94_JET_MS == 28.0
    assert plot.HS94_JET_LAT_DEG == 45.0
    assert plot.JET_FLOOR_MS == 20.0
