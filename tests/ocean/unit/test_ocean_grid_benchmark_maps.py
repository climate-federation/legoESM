"""Direct test for scripts/plot/plot_ocean_grid_benchmark_maps.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load():
    p = _REPO / "scripts" / "plot" / "plot_ocean_grid_benchmark_maps.py"
    spec = importlib.util.spec_from_file_location("_ocn_maps", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ocn_maps"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake(root: Path, case: str, grid: str, res: str, value=None):
    d = root / case / case / grid / res
    d.mkdir(parents=True)
    nt, nlat, nlon = 2, 6, 8
    rng = np.random.default_rng(0)
    sst = (np.full((nt, nlat, nlon), value) if value is not None
           else 10.0 + rng.random((nt, nlat, nlon)))
    np.savez(d / "snapshots_latlon.npz",
             times_days=np.array([0.0, 1.0]),
             lat=np.linspace(-80, 80, nlat), lon=np.linspace(0, 350, nlon),
             SST=sst, eta=sst * 0.01,
             land_mask=np.ones((nt, nlat, nlon)))
    (d / "results.txt").write_text("status: PASS\n")


def test_degenerate_field_is_not_drawn_as_structure():
    """A field that is uniform to machine precision must NOT be
    autoscaled -- that renders round-off as vivid structure and the panel
    reads as a broken model."""
    mod = _load()
    flat = [np.full((4, 4), 10.0), np.full((4, 4), 10.0 + 1e-14)]
    kw, degenerate, (lo, hi) = mod._shared_scale(flat, "viridis")
    assert degenerate
    assert kw["vmax"] - kw["vmin"] > 1.0       # widened to a real band
    varied = [np.linspace(0, 1, 16).reshape(4, 4)]
    kw2, degenerate2, _ = mod._shared_scale(varied, "viridis")
    assert not degenerate2
    assert kw2["vmin"] == 0.0 and kw2["vmax"] == 1.0


def test_signed_scale_is_symmetric_about_zero():
    mod = _load()
    kw, _deg, _rng = mod._shared_scale([np.array([-0.2, 0.9])], "RdBu_r")
    assert kw["vmin"] == -kw["vmax"]


def test_grid_lookup_is_exact_not_prefix(tmp_path):
    """`latlon_regional` must never be picked up as `latlon` -- a prefix
    match would silently plot the wrong arm."""
    mod = _load()
    _fake(tmp_path, "lock_exchange", "latlon_regional", "4x64")
    assert mod._find(tmp_path, "lock_exchange", "latlon") is None
    _fake(tmp_path, "lock_exchange", "latlon", "36x72")
    assert mod._find(tmp_path, "lock_exchange", "latlon") is not None


def test_build_figures(tmp_path):
    mod = _load()
    for g, r in (("latlon", "36x72"), ("mpas", "ico3")):
        _fake(tmp_path, "lock_exchange", g, r)
    out = mod.build_case_figure(tmp_path, "lock_exchange", tmp_path / "o")
    assert out is not None and out.stat().st_size > 5_000
    sheet = mod.build_contact_sheet(tmp_path, tmp_path / "o" / "all.png")
    assert sheet.stat().st_size > 5_000


def test_missing_case_returns_none(tmp_path):
    mod = _load()
    assert mod.build_case_figure(tmp_path, "lock_exchange",
                                 tmp_path / "o") is None
