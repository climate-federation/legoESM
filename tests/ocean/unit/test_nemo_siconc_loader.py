"""Unit test for the NEMO sea-ice-concentration loader (--ice-albedo surrogate).

``load_nemo_siconc`` lives in the run driver (scripts/run/run_omip_core2.py); it
IDW-regrids NEMO ORCA1 annual `siconc` onto the model grid for the SW-albedo fix.
Validated against the real ORCA1 RUN_REF icemod file (skipped if absent), since
the loader's whole job is the curvilinear->target regrid + clip + land masking.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


def _runner_module():
    if not hasattr(_runner_module, "_mod"):
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "scripts" / "run" / "run_omip_core2.py"
        if str(repo_root / "scripts") not in sys.path:
            sys.path.insert(0, str(repo_root / "scripts"))
        spec = importlib.util.spec_from_file_location("_run_omip_core2_for_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _runner_module._mod = mod
    return _runner_module._mod


def _latlon_target(nlat=90, nlon=180):
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat2d = np.broadcast_to(lat[:, None], (nlat, nlon)).copy()
    lon2d = np.broadcast_to(lon[None, :], (nlat, nlon)).copy()
    return lat2d, lon2d


def test_load_nemo_siconc_real_file():
    rom = _runner_module()
    sic_path = rom._SICONC_NC
    if not os.path.exists(sic_path):
        pytest.skip(f"NEMO icemod file absent: {sic_path}")
    lat2d, lon2d = _latlon_target()
    # grid is unused in the loader body; grid_type only labels the log line.
    sic = rom.load_nemo_siconc(None, "latlon", lat2d, lon2d)

    assert sic.shape == lat2d.shape
    assert np.isfinite(sic).all()
    assert sic.min() >= 0.0 and sic.max() <= 1.0
    # Physical: Antarctic + Arctic carry ice; the deep tropics do not.
    antarctic = lat2d < -60.0
    arctic = lat2d > 70.0
    tropics = np.abs(lat2d) < 20.0
    assert sic[antarctic].max() > 0.5, "no Antarctic sea ice found"
    assert sic[arctic].max() > 0.5, "no Arctic sea ice found"
    assert sic[tropics].max() < 0.15, "spurious tropical sea ice"


def test_load_nemo_siconc_land_mask_zeroes():
    """A land_mask (0 over land) zeroes siconc on land cells."""
    rom = _runner_module()
    sic_path = rom._SICONC_NC
    if not os.path.exists(sic_path):
        pytest.skip(f"NEMO icemod file absent: {sic_path}")
    lat2d, lon2d = _latlon_target()
    land_mask = np.ones_like(lat2d)
    land_mask[lat2d > 0.0] = 0.0   # call the whole NH "land"
    sic = rom.load_nemo_siconc(None, "latlon", lat2d, lon2d, land_mask=land_mask)
    assert np.all(sic[land_mask <= 0.5] == 0.0)
