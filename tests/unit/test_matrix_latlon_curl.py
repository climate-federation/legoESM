"""Direct test for ``_latlon_curl`` (matrix snapshot vorticity, #521).

Solid-body rotation ``u = u0 cos(lat)``, ``v = 0`` has the analytic
relative vorticity ``zeta = 2 u0 sin(lat) / R`` — pin the numerical
spherical curl on the (181, 360) snapshot canvas against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))

from run_atmosphere_test_matrix import _latlon_curl  # noqa: E402

from legoesm import constants  # noqa: E402

_R = constants.R_earth


def _canvas():
    lat1d = np.linspace(-90.0, 90.0, 181)
    lon1d = np.linspace(-180.0, 180.0, 360, endpoint=False)
    return np.meshgrid(lon1d, lat1d)


def test_solid_body_rotation_matches_analytic():
    lon2d, lat2d = _canvas()
    u0 = 40.0
    u = u0 * np.cos(np.radians(lat2d))
    v = np.zeros_like(u)
    zeta = _latlon_curl(u, v, _R)
    zeta_exact = 2.0 * u0 * np.sin(np.radians(lat2d)) / _R
    # Interior rows only: the two rows nearest each pole are zeroed by
    # design (metric-singular).
    err = np.abs(zeta[3:-3] - zeta_exact[3:-3])
    assert float(err.max()) < 1e-8   # ~1e-3 of the 1.25e-5 peak


def test_pole_rows_zeroed_and_shape():
    lon2d, lat2d = _canvas()
    u = np.ones_like(lat2d)
    v = np.ones_like(lat2d)
    zeta = _latlon_curl(u, v, _R)
    assert zeta.shape == (181, 360)
    assert np.all(zeta[:2] == 0.0)
    assert np.all(zeta[-2:] == 0.0)
    assert np.all(np.isfinite(zeta))


def test_pure_rotational_gaussian_sign():
    """zeta = grad^2(psi): a NEGATIVE streamfunction well (cyclone,
    counter-clockwise with u = -dpsi/dy, v = +dpsi/dx) has zeta > 0 at
    its centre."""
    lon2d, lat2d = _canvas()
    # Azimuthal flow around (lon=0, lat=30): local tangent-plane approx.
    x = np.radians(lon2d) * np.cos(np.radians(30.0))
    y = np.radians(lat2d - 30.0)
    r2 = x ** 2 + y ** 2
    psi = -np.exp(-r2 / 0.02)         # streamfunction well = cyclone
    u = -np.gradient(psi, np.radians(1.0), axis=0)
    v = np.gradient(psi, np.radians(1.0) * np.cos(np.radians(30.0)), axis=1)
    zeta = _latlon_curl(u * _R, v * _R, _R)
    i_c = 120, 180                    # lat=30, lon=0
    assert zeta[i_c] > 0.0
