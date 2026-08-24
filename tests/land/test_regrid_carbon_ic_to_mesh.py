"""The carbon-IC regridder must land the pools where the land is.

The interpolation is not conservative and the driver says so; what it must NOT
do is move carbon somewhere else. A synthetic source with a single known land
patch is enough to catch the failure that would otherwise be invisible in a
global mean: a transposed reshape, which preserves the total exactly and puts
the Amazon in the Indian Ocean.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "data" / "regrid_carbon_ic_to_mesh.py"

pytest.importorskip("legoesm.land.carbon.config")
from legoesm.land.carbon.config import CarbonState  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location("regrid_carbon_ic", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["regrid_carbon_ic"] = mod
    spec.loader.exec_module(mod)
    return mod


# Two land boxes with very different carbon, on opposite sides of the globe.
# One box alone cannot test placement any more: the interpolator fills every
# mesh cell from its nearest LAND cell, so a single-box source puts the same
# value everywhere and a transposed map would look identical. Two contrasting
# boxes make placement observable again -- cells near the rich box must be rich
# and cells near the poor box poor, which a transpose cannot satisfy.
_RICH = (10.0, 30.0, 20.0, 60.0)      # lat0, lat1, lon0, lon1
_POOR = (-30.0, -10.0, 200.0, 240.0)
_RICH_C, _POOR_C = 1000.0, 10.0


def _synthetic(path, *, n_lat=48, n_lon=96):
    lat_1d = np.linspace(-90.0, 90.0, n_lat)
    lon_1d = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lat = np.repeat(lat_1d, n_lon)
    lon = np.tile(lon_1d, n_lat)

    def _box(b):
        la0, la1, lo0, lo1 = b
        return ((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))

    rich, poor = _box(_RICH), _box(_POOR)
    land = rich | poor
    val = np.where(rich, _RICH_C, np.where(poor, _POOR_C, 0.0))
    pools = {f: val.copy() for f in CarbonState._fields}
    npft = 3
    pw = np.zeros((lat.size, npft))
    pw[rich, 0] = 1.0
    pw[poor, 2] = 1.0
    np.savez(path, lat=lat, lon=lon, land_mask=land.astype(float),
             soil_frozen_fraction=np.where(rich, 0.8, 0.0),
             pft_weights=pw, dominant_pft=np.argmax(pw, axis=1),
             pft_present=(pw > 0),
             n_layers=10, soil_depth=3.0, resolution_deg=180.0 / n_lat,
             **pools)


def _run(tmp_path):
    src = tmp_path / "src.npz"
    _synthetic(src)
    out = tmp_path / "out.npz"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--source", str(src),
         "--resolution", "3", "--output", str(out)],
        capture_output=True, text=True, env={**_env()},
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return np.load(out)


def _in_box(lat, lon, b, halo=8.0):
    la0, la1, lo0, lo1 = b
    return ((lat >= la0 - halo) & (lat <= la1 + halo)
            & (lon >= lo0 - halo) & (lon <= lo1 + halo))


def test_carbon_lands_where_its_source_is(tmp_path):
    """Rich box stays rich, poor box stays poor. A transpose cannot do both."""
    z = _run(tmp_path)
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    soc = np.asarray(z["C_som_active"])
    rich = _in_box(lat, lon, _RICH)
    poor = _in_box(lat, lon, _POOR)
    assert rich.any() and poor.any(), "synthetic boxes missed the mesh entirely"
    assert soc[rich].mean() > 0.5 * _RICH_C, (
        f"the rich box came back at {soc[rich].mean():.1f} gC/m2")
    assert soc[poor].mean() < 10.0 * _POOR_C, (
        f"the poor box came back at {soc[poor].mean():.1f} gC/m2 -- carbon "
        f"from the other side of the globe landed on it")


def test_every_per_cell_field_is_on_the_target_grid(tmp_path):
    """No field may keep the SOURCE cell count -- a stale one indexes a
    different planet, and the loader would not catch it."""
    z = _run(tmp_path)
    ncol = np.asarray(z["lat"]).size
    for name in z.files:
        a = np.asarray(z[name])
        if a.ndim >= 1 and a.shape[0] not in (ncol, 1) and a.size > 8:
            raise AssertionError(
                f"{name} has leading dimension {a.shape[0]}, not {ncol}")


def test_categories_stay_categories(tmp_path):
    """A dominant plant type of 4.37 is not a plant type."""
    z = _run(tmp_path)
    dom = np.asarray(z["dominant_pft"])
    assert np.allclose(dom, np.round(dom)), "dominant_pft was interpolated"
    assert np.asarray(z["pft_present"]).dtype == bool


def test_frozen_fraction_is_never_gap_filled_to_zero(tmp_path):
    """Zero frozen fraction silently disables the frozen-ground protection, so
    a cell must inherit its nearest land value rather than a fill."""
    z = _run(tmp_path)
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    phi = np.asarray(z["soil_frozen_fraction"])
    assert np.all(np.isfinite(phi))
    rich = _in_box(lat, lon, _RICH)
    assert phi[rich].mean() > 0.4, (
        f"the frozen box came back at {phi[rich].mean():.2f}")


def test_non_regular_source_is_rejected(tmp_path):
    """A source that is not a complete regular grid must raise, not reshape."""
    mod = _load_script()
    lat = np.array([0.0, 0.0, 1.0])       # 3 cells: not n_lat x n_lon
    lon = np.array([0.0, 10.0, 0.0])
    with pytest.raises(SystemExit, match="regular"):
        mod._axes_from_flat(lat, lon)


def test_transposed_source_is_rejected(tmp_path):
    """Latitude-fastest storage must raise rather than silently transpose."""
    mod = _load_script()
    lat_1d = np.array([-10.0, 0.0, 10.0])
    lon_1d = np.array([0.0, 90.0])
    lat = np.tile(lat_1d, lon_1d.size)     # latitude fastest = transposed
    lon = np.repeat(lon_1d, lat_1d.size)
    with pytest.raises(SystemExit, match="row-major"):
        mod._axes_from_flat(lat, lon)


def _env():
    import os
    pkg = _ROOT / "packages"
    return {**os.environ,
            "PYTHONPATH": ":".join(str(p) for p in [
                _ROOT / "src", pkg / "core", pkg / "atmosphere", pkg / "coupler",
                pkg / "tools", pkg / "ml", pkg / "ocean", pkg / "land",
                pkg / "ice"]),
            "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"}
