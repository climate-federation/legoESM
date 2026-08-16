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


def _synthetic(path, *, n_lat=48, n_lon=96, patch=(10.0, 30.0, 20.0, 60.0)):
    """A source whose only land is one lat-lon box, carbon 1000 gC/m2 flat."""
    lat_1d = np.linspace(-90.0, 90.0, n_lat)
    lon_1d = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lat = np.repeat(lat_1d, n_lon)
    lon = np.tile(lon_1d, n_lat)
    la0, la1, lo0, lo1 = patch
    land = ((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))
    pools = {f: np.where(land, 1000.0, 0.0) for f in CarbonState._fields}
    np.savez(path, lat=lat, lon=lon, land_mask=land.astype(float),
             soil_frozen_fraction=np.zeros_like(lat),
             n_layers=10, soil_depth=3.0, resolution_deg=180.0 / n_lat,
             **pools)
    return patch


def test_carbon_lands_on_the_source_patch_and_nowhere_else(tmp_path):
    src = tmp_path / "src.npz"
    la0, la1, lo0, lo1 = _synthetic(src)
    out = tmp_path / "out.npz"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--source", str(src),
         "--resolution", "3", "--output", str(out)],
        capture_output=True, text=True, env={**_env()},
    )
    assert r.returncode == 0, r.stderr[-2000:]

    z = np.load(out)
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    soc = np.asarray(z["C_som_active"])
    inside = (lat >= la0 - 15) & (lat <= la1 + 15) & \
             (lon >= lo0 - 15) & (lon <= lo1 + 15)

    assert soc[inside].max() > 500.0, "no carbon landed on the source patch"
    # Outside a generous halo of the patch there must be none: this is the
    # assertion a transposed reshape fails, and a global-mean check does not.
    assert soc[~inside].max() < 1e-6, (
        f"carbon appeared {soc[~inside].max():.1f} gC/m2 away from the only "
        f"land in the source")


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
