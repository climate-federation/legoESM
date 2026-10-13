"""Unit test for ``scripts/validate/ocean_fidelity/omip_eq_wind_response.py``.

``transports`` decides every number the probe prints.  Synthetic fields on the
common grid with known answers: v = +0.1 m/s north of the equator and -0.1 m/s
south over the top 50 m gives V_N = +5, V_S = -5, div_ek = 10 m^2/s; a jet of
1.2 m/s at one level/longitude is found as the undercurrent core; a uniform
-0.4 m/s surface current is the SEC.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"


@pytest.fixture(scope="module")
def mod():
    import sys
    sys.path.insert(0, str(_DIR)); sys.path.insert(0, str(_DIR.parent))
    spec = importlib.util.spec_from_file_location("eq_wind", _DIR / "omip_eq_wind_response.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_known_transports_and_jets(mod):
    zi = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0])
    zN = 0.5 * (zi[1:] + zi[:-1])
    shp = (zN.size, mod.GLAT.size, mod.GLON.size)
    v = np.zeros(shp); u = np.zeros(shp)
    v[:3, mod.GLAT > 0, :] = 0.1
    v[:3, mod.GLAT < 0, :] = -0.1
    v[3:, :, :] = 7.0                                    # below 50 m: must not count
    u[:2] = -0.4
    i = int(np.argmin(np.abs(mod.GLON - 230.5)))
    u[4, np.abs(mod.GLAT) <= 1.0, i] = 3.6               # core at k=4, one longitude
    r = mod.transports(u, v, zN, zi)
    assert r["V_N"] == pytest.approx(5.0) and r["V_S"] == pytest.approx(-5.0)
    assert r["div_ek"] == pytest.approx(10.0)
    assert r["u_sec"] == pytest.approx(-0.4)
    assert r["z_euc"] == pytest.approx(zN[4]) and r["lon_euc"] == pytest.approx(230.5)
    assert r["u_euc"] == pytest.approx(3.6)
