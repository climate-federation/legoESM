"""Manufactured-solution checks for the layer heat budget used to decide
whether the equatorial cold bias at 20 m is advective (2026-09-08).

The instrument decides what we believe, so each term is checked against a case
whose answer is known analytically, and each check is written so that it FAILS
if the corresponding term is dropped or given the wrong sign.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = (Path(__file__).resolve().parents[3]
        / "scripts" / "validate" / "ocean_fidelity" / "equatorial_thermocline.py")


def _mod():
    spec = importlib.util.spec_from_file_location("eqthermo", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NY = NX = 3
NZ = 4


def _grid():
    z_if = np.arange(NZ + 1, dtype=np.float64)          # 0,1,2,3,4 m depth
    zc = 0.5 * (z_if[:-1] + z_if[1:])                   # 0.5 .. 3.5
    area = np.ones((NY, NX))
    kl = (zc >= 1.0) & (zc < 3.0)                       # the 1-3 m layer
    box = np.ones((NY, NX), dtype=bool)
    return zc, z_if, area, kl, box


def _linear_T(zc):
    """20 C at the surface, 1 K colder per metre of depth."""
    return np.broadcast_to(20.0 - zc, (NY, NX, NZ)).copy()


def test_vertical_advection_matches_minus_w_dTdz():
    m = _mod()
    zc, z_if, area, kl, box = _grid()
    T = _linear_T(zc)
    W = 1.0e-5                                          # m/s upward, area 1 m2
    Fup = np.full((NY, NX, NZ + 1), W)
    zero_x = np.zeros((NY, NX + 1, NZ))
    zero_y = np.zeros((NY + 1, NX, NZ))
    K = np.zeros((NY, NX, NZ - 1))
    adv_h, adv_v, dif = m._heat_terms(zero_x, zero_y, Fup, T, K, area,
                                      zc, z_if, kl, 1, 3, box)
    # dT/dz = +1 K/m (warmer upward), so -w dT/dz = -W: upwelling COOLS.
    assert adv_v == pytest.approx(-W * m._SEC_PER_MONTH, rel=1e-12)
    assert adv_v < 0.0
    assert adv_h == 0.0 and dif == 0.0


def test_uniform_gradient_and_uniform_K_gives_no_net_diffusion():
    m = _mod()
    zc, z_if, area, kl, box = _grid()
    T = _linear_T(zc)
    K = np.full((NY, NX, NZ - 1), 1.0e-3)
    z3 = np.zeros((NY, NX, NZ + 1))
    _, _, dif = m._heat_terms(np.zeros((NY, NX + 1, NZ)), np.zeros((NY + 1, NX, NZ)),
                              z3, T, K, area, zc, z_if, kl, 1, 3, box)
    assert dif == pytest.approx(0.0, abs=1e-12)


def test_stronger_diffusion_below_drains_the_layer():
    m = _mod()
    zc, z_if, area, kl, box = _grid()
    T = _linear_T(zc)
    K = np.full((NY, NX, NZ - 1), 1.0e-3)
    K[..., 2] = 2.0e-3                       # interface 3 = the layer's base
    z3 = np.zeros((NY, NX, NZ + 1))
    _, _, dif = m._heat_terms(np.zeros((NY, NX + 1, NZ)), np.zeros((NY + 1, NX, NZ)),
                              z3, T, K, area, zc, z_if, kl, 1, 3, box)
    # layer thickness 2 m; the extra downward flux at the base is K*dT/dz = 1e-3
    assert dif == pytest.approx(-1.0e-3 / 2.0 * m._SEC_PER_MONTH, rel=1e-12)
    assert dif < 0.0


def test_uniform_temperature_makes_horizontal_advection_vanish():
    m = _mod()
    zc, z_if, area, kl, box = _grid()
    T = np.full((NY, NX, NZ), 12.0)
    Fw = np.full((NY, NX + 1, NZ), 3.0)      # uniform eastward, no divergence
    Fs = np.zeros((NY + 1, NX, NZ))
    K = np.zeros((NY, NX, NZ - 1))
    adv_h, _, _ = m._heat_terms(Fw, Fs, np.zeros((NY, NX, NZ + 1)), T, K, area,
                                zc, z_if, kl, 1, 3, box)
    assert adv_h == pytest.approx(0.0, abs=1e-12)


def test_zonal_temperature_gradient_is_advected_with_the_right_sign():
    m = _mod()
    zc, z_if, area, kl, box = _grid()
    # warm in the west, cold in the east; uniform eastward flow warms each cell
    T = np.zeros((NY, NX, NZ))
    T[:, :, :] = np.array([22.0, 21.0, 20.0])[None, :, None]
    U = 2.0
    Fw = np.full((NY, NX + 1, NZ), U)
    adv_h, _, _ = m._heat_terms(Fw, np.zeros((NY + 1, NX, NZ)),
                                np.zeros((NY, NX, NZ + 1)), T,
                                np.zeros((NY, NX, NZ - 1)), area,
                                zc, z_if, kl, 1, 3, box)
    assert adv_h > 0.0
