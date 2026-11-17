"""Unit tests for ``scripts/validate/ocean_fidelity/omip_forcing_sampling_box.py``.

``box_means`` decides every number the probe prints.  On synthetic forcing:

* spatially uniform channels -> every sampling method gives identical box
  means (sampling cannot matter for a constant field);
* a wind field with structure at the source scale -> nearest, bilinear and
  bicubic-wind box means all DIFFER (a probe that routed two methods to the
  same sampler fails here).  ``nemo_scrip`` needs ORCA1's weight files and is
  cross-checked in the real run (equal to ``nemo_weights`` to <1e-6).
"""

from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_MOD = (Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "ocean_fidelity"
        / "omip_forcing_sampling_box.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("omip_fs_box", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _forcing(u_fn):
    lat = np.linspace(-29.0, 29.0, 30)                 # 2-deg centres
    lon = np.arange(0.0, 360.0, 2.0)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    one = np.ones((2,) + LA.shape)
    from legoesm import constants
    return types.SimpleNamespace(
        lat=lat, lon=lon, time_s=np.array([0.0, 21600.0]),
        u10=one * u_fn(LA, LO), v10=one * 1.0,
        T_air=one * (constants.T_freeze + 25.0), q_air=one * 0.015,
        sw_down=one * 250.0, lw_down=one * 400.0, precip=one * 0.0,
        snow=None, slp=None)


def _points():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-5.0, 5.0, 200)
    lon = rng.uniform(210.0, 270.0, 200)
    return lat, lon


def test_constant_field_every_method_agrees(mod):
    from legoesm import constants
    f = _forcing(lambda la, lo: -6.0 + 0.0 * la)
    lat, lon = _points()
    sst = np.full(lat.size, constants.T_freeze + 26.0)
    r = mod.box_means(f, [0, 1], lat, lon, sst, np.ones(lat.size),
                      mod.METHODS)
    for m in mod.METHODS:
        for k in mod.QUANTITIES:
            assert r[m][k] == pytest.approx(r["bilinear"][k], rel=1e-12), (m, k)


def test_structured_wind_separates_nearest_from_bilinear(mod):
    from legoesm import constants
    f = _forcing(lambda la, lo: -6.0 + 3.0 * np.cos(np.radians(lo) * 90.0))
    lat, lon = _points()
    sst = np.full(lat.size, constants.T_freeze + 26.0)
    r = mod.box_means(f, [0, 1], lat, lon, sst, np.ones(lat.size),
                      mod.METHODS)
    ws = {m: r[m]["wind_speed"] for m in mod.METHODS}
    assert abs(ws["nearest"] - ws["bilinear"]) > 1e-2
    assert abs(ws["nemo_weights"] - ws["bilinear"]) > 1e-3   # bicubic winds
