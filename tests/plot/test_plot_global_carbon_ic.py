"""Smoke test for the global carbon-IC map plotter.

Builds a tiny synthetic finidat npz on a regular (n_lat, n_lon) grid with the
exact keys ``plot_global_carbon_ic`` consumes, drives ``main`` end-to-end, and
asserts a non-empty PNG is written. Guards the reshape / longitude-roll / zonal
-mean paths against regressions (the plotter is otherwise only exercised on the
real 13824-cell finidat via sbatch).
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

# Load the script module by path (scripts/ is not an importable package).
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "plot_global_carbon_ic", _ROOT / "scripts" / "plot" / "plot_global_carbon_ic.py")
plotter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plotter)


def _synthetic_finidat(path, n_lat=4, n_lon=8):
    """Row-major (n_lat*n_lon,) per-cell arrays with every key the plotter reads."""
    lat1d = np.linspace(-60.0, 80.0, n_lat)
    lon1d = np.linspace(0.0, 315.0, n_lon)  # 0..360 so the >180 roll is exercised
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")
    lat = lat2d.ravel()
    lon = lon2d.ravel()
    ncell = lat.size
    ramp = np.linspace(0.0, 20000.0, ncell)  # gC/m2 spread across pools
    land_mask = (np.arange(ncell) % 5) != 0  # ~80% land, some ocean -> NaN path
    data = dict(
        C_lab=ramp * 0.05, C_fol=ramp * 0.05, C_root=ramp * 0.2, C_wood=ramp * 0.7,
        C_som_active=ramp * 0.2, C_som_slow=ramp * 0.3, C_som_passive=ramp * 0.5,
        lat=lat, lon=lon, land_mask=land_mask,
        soil_frozen_fraction=np.clip((lat - 40.0) / 40.0, 0.0, 1.0),
        resolution_deg=np.asarray(1.0),
    )
    np.savez(path, **data)


def test_main_writes_nonempty_png(tmp_path):
    finidat = tmp_path / "finidat.npz"
    out = tmp_path / "maps.png"
    _synthetic_finidat(finidat)
    plotter.main(["--finidat", str(finidat), "--out", str(out)])
    assert out.exists(), "plotter did not write the PNG"
    assert out.stat().st_size > 0, "plotter wrote an empty PNG"


def test_reshape_matches_unique_grid(tmp_path):
    """The (ncell,) -> (n_lat, n_lon) reshape must match the unique lat/lon count."""
    finidat = tmp_path / "f.npz"
    _synthetic_finidat(finidat, n_lat=5, n_lon=6)
    z = np.load(finidat, allow_pickle=True)
    lat = np.asarray(z["lat"], float)
    lon = np.asarray(z["lon"], float)
    lm = np.asarray(z["land_mask"]).astype(bool)
    g = plotter._grid(np.asarray(z["C_som_active"], float), lm,
                      np.unique(lat).size, np.unique(lon).size)
    assert g.shape == (5, 6)
    assert np.isnan(g).any()  # ocean cells masked to NaN
