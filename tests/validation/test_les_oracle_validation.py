"""MOST diagnostics used to validate the plane-LES against the jax-alfa oracle.

A synthetic neutral log-law profile U(z)=(u_*/κ)ln(z/z0) must round-trip through
``log_law_fit`` (recover u_*) and ``phi_m`` (→ 1 in the surface layer).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from legoesm import constants

_KAPPA = constants.kappa_von_karman
_ROOT = Path(__file__).resolve().parents[2]


def _load_validator():
    path = _ROOT / "scripts" / "validate" / "validate_les_vs_oracle.py"
    spec = importlib.util.spec_from_file_location("validate_les_vs_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_log_law_fit_recovers_ustar():
    m = _load_validator()
    u_star, z0 = 0.35, 0.1
    z = np.linspace(2.0, 200.0, 60)
    U = (u_star / _KAPPA) * np.log(z / z0)
    u_fit, npts = m.log_law_fit(z, U, z0, z_lo=2.0, z_hi=100.0)
    assert npts > 5
    assert abs(u_fit - u_star) < 1e-3


def test_phi_m_unity_in_neutral_surface_layer():
    m = _load_validator()
    u_star, z0 = 0.4, 0.1
    z = np.linspace(1.0, 500.0, 200)
    U = (u_star / _KAPPA) * np.log(z / z0)
    zc, phim = m.phi_m(z, U, u_star)
    sl = (zc > 1.0) & (zc < 50.0)              # surface layer
    assert abs(float(np.mean(phim[sl])) - 1.0) < 0.05
