"""Tests for the residual-variance SSO construction (#1712).

The point of the new mode is a property the classic construction lacks: a
mountain the MODEL RESOLVES must contribute ~nothing to the launch-stress
field, while sub-cutoff roughness contributes in full.  Synthetic waves with
known wavelengths make that a pass/fail rather than a plausibility argument.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GEN = _ROOT / "scripts" / "data" / "prep_subgrid_orography.py"


def _load():
    sys.path.insert(0, str(_GEN.parent))
    spec = importlib.util.spec_from_file_location("_sso_gen", _GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sso_gen"] = mod
    spec.loader.exec_module(mod)
    return mod


def _terrain_ds(z):
    n_lat, n_lon = z.shape
    lat = np.linspace(-90 + 90.0 / n_lat, 90 - 90.0 / n_lat, n_lat)
    lon = np.linspace(0.25 / 2, 360 - 0.25 / 2, n_lon)
    return xr.Dataset({"elevation": (("lat", "lon"), z)},
                      coords={"lat": lat, "lon": lon})


def _grid(fine=0.25):
    n_lat, n_lon = int(180 / fine), int(360 / fine)
    lat = np.linspace(-90 + 90.0 / n_lat, 90 - 90.0 / n_lat, n_lat)
    lon = np.linspace(fine / 2, 360 - fine / 2, n_lon)
    return np.meshgrid(lat, lon, indexing="ij"), (n_lat, n_lon)


def test_constant_terrain_gives_zero():
    mod = _load()
    (_, _), (n_lat, n_lon) = _grid()
    out = mod.subgrid_orography_residual_stddev(
        _terrain_ds(np.full((n_lat, n_lon), 500.0)),
        fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    assert float(np.abs(out["SSO_STDH"].values).max()) < 1e-6


def test_resolved_long_wave_is_removed_but_classic_reports_it():
    """THE discriminating property (#1712's double-count defect).

    A 30-degree zonal wave is resolved by any model this field feeds, so the
    residual construction must return ~nothing for it — while the classic
    per-block stddev happily reports it as launch-stress orography.  If both
    constructions agree on this terrain, the new mode adds nothing and the
    double-count is still being fed.
    """
    mod = _load()
    (LAT, LON), (n_lat, n_lon) = _grid()
    z = 800.0 + 500.0 * np.sin(2 * np.pi * LON / 30.0)     # >= 0 everywhere
    ds = _terrain_ds(z)
    resid = mod.subgrid_orography_residual_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    classic = mod.subgrid_orography_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0)
    band = slice(30, 690)   # away from the truncated polar smooth
    r = float(np.mean(resid["SSO_STDH"].values[band]))
    c = float(np.mean(classic["SSO_STDH"].values[band]))
    assert c > 30.0, f"classic construction does not even see the wave ({c})"
    assert r < 0.15 * c, (
        f"residual mode keeps {r:.1f} m of a resolved 30-deg wave the classic "
        f"mode reports as {c:.1f} m — the scale decomposition is not working")


def test_subgrid_short_wave_survives_in_full():
    """Roughness below the cutoff must come through ~undiminished."""
    mod = _load()
    (LAT, LON), (n_lat, n_lon) = _grid()
    z = 800.0 + 500.0 * np.sin(2 * np.pi * LON / 1.0)      # 1-deg wave
    ds = _terrain_ds(z)
    resid = mod.subgrid_orography_residual_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    classic = mod.subgrid_orography_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0)
    band = slice(30, 690)
    r = float(np.mean(resid["SSO_STDH"].values[band]))
    c = float(np.mean(classic["SSO_STDH"].values[band]))
    assert c > 200.0
    assert r > 0.9 * c, (r, c)


def test_cutoff_below_block_is_refused():
    mod = _load()
    (_, _), (n_lat, n_lon) = _grid(1.0)
    with pytest.raises(ValueError, match="cutoff"):
        mod.subgrid_orography_residual_stddev(
            _terrain_ds(np.zeros((n_lat, n_lon))),
            fine_res_deg=1.0, block_deg=4.0, resolved_cutoff_deg=2.0)
