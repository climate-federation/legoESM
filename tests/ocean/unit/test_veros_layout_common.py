"""Unit tests for the shared Veros↔legoESM data-prep glue (#433).

Covers the layout bridges, the MIT-grid tau shift (both x boundaries) and the
T-cell area weights — the functions factored out of the global_flexible /
global_1deg recipes' copy-pasted twins.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from legoesm.ocean.fidelity.veros_layout_common import (
    veros_area_t,
    veros_mit_tau_shift,
    veros_xy_to_legoesm,
    veros_xyz_to_legoesm,
)


def test_xyz_bridge_transposes_and_flips_z():
    nx, ny, nz = 7, 5, 4
    arr = np.zeros((nx, ny, nz))
    arr[5, 3, nz - 1] = 42.0          # SURFACE in Veros z-order (k=nz-1 here)
    arr[5, 3, 0] = -7.0               # DEEPEST
    out = veros_xyz_to_legoesm(arr)
    assert out.shape == (ny + 2, nx, nz)
    assert out[4, 5, 0] == 42.0       # surface lands at legoESM k=0
    assert out[4, 5, nz - 1] == -7.0  # deepest at k=nz-1
    assert (out[0] == 0).all() and (out[-1] == 0).all()   # wall rows


def test_xy_bridge_transposes_with_wall_rows():
    nx, ny = 6, 4
    arr = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    out = veros_xy_to_legoesm(arr)
    assert out.shape == (ny + 2, nx)
    np.testing.assert_array_equal(out[1:-1, :], arr.T)
    assert (out[0] == 0).all() and (out[-1] == 0).all()


def test_tau_shift_cyclic_x():
    rng = np.random.default_rng(0)
    taux = rng.normal(size=(8, 5, 12))
    tauy = rng.normal(size=(8, 5, 12))
    tx, ty = veros_mit_tau_shift(taux, tauy, x_cyclic=True)
    np.testing.assert_array_equal(tx[:-1], taux[1:])
    np.testing.assert_array_equal(tx[-1], taux[0])        # cyclic wrap
    np.testing.assert_array_equal(ty[:, :-1], tauy[:, 1:])
    np.testing.assert_array_equal(ty[:, -1], 0.0)         # zero y ghost


def test_tau_shift_zero_ghost_x():
    rng = np.random.default_rng(1)
    taux = rng.normal(size=(8, 5, 12))
    tauy = rng.normal(size=(8, 5, 12))
    tx, ty = veros_mit_tau_shift(taux, tauy, x_cyclic=False)
    np.testing.assert_array_equal(tx[:-1], taux[1:])
    np.testing.assert_array_equal(tx[-1], 0.0)            # zero x ghost (no wrap)
    np.testing.assert_array_equal(ty[:, :-1], tauy[:, 1:])
    np.testing.assert_array_equal(ty[:, -1], 0.0)


def test_area_scalar_vs_array_dyt():
    yt = np.array([-30.0, 0.0, 30.0])
    a_scalar = veros_area_t(yt, dx_deg=1.0, dyt_deg=1.0)
    a_array = veros_area_t(yt, dx_deg=1.0, dyt_deg=np.full(3, 1.0))
    np.testing.assert_allclose(a_scalar, a_array)
    assert np.all(a_scalar > 0.0)
    # equator (cos=1) is the largest at uniform dy
    assert a_scalar[1] == a_scalar.max()


def test_area_formula_matches_dxt_dyt_cost():
    from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
    yt = np.array([45.0])
    degtom = VEROS_CONSTANTS_CONFIG.R_earth * np.pi / 180.0
    expect = (4.0 * degtom) * (2.0 * degtom) * np.cos(np.deg2rad(45.0))
    got = veros_area_t(yt, dx_deg=4.0, dyt_deg=2.0)[0]
    np.testing.assert_allclose(got, expect)
